import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.boolean_expression_parser import And, Not, Or, Var, Xor, parse_boolean_expression
from rysy_dl.boolean_expression_to_netlist import NetlistBuildError, build_netlist


def _sink(netlist):
    sinks = [node for node in netlist.nodes() if netlist.out_degree(node) == 0]
    assert len(sinks) == 1, f"expected exactly one sink, found {sinks}"
    return sinks[0]


def _gates(netlist, gate_type):
    return [node for node, data in netlist.nodes(data=True) if data['gate'] == gate_type]


def _src_id(netlist, name):
    matches = [node for node, data in netlist.nodes(data=True)
               if data.get('gate') == 'src' and data.get('label') == name]
    assert len(matches) == 1, f"expected exactly one src node labeled {name!r}, found {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# Single terminal
# ---------------------------------------------------------------------------

def test_single_variable_is_one_src_node():
    netlist = build_netlist(Var('A'))
    assert dict(netlist.nodes(data=True)) == {0: {'gate': 'src', 'label': 'A'}}
    assert list(netlist.edges()) == []


def test_root_is_always_the_sink():
    netlist = build_netlist(And([Var('A'), Var('B')]))
    assert _sink(netlist) == 0


# ---------------------------------------------------------------------------
# One gate, no fan-out
# ---------------------------------------------------------------------------

def test_not_gate():
    netlist = build_netlist(Not(Var('A')))
    gate = _sink(netlist)
    assert netlist.nodes[gate]['gate'] == 'NOT'
    a = _src_id(netlist, 'A')
    assert list(netlist.edges()) == [(a, gate)]


def test_and_gate_two_operands():
    netlist = build_netlist(And([Var('A'), Var('B')]))
    gate = _sink(netlist)
    assert netlist.nodes[gate]['gate'] == 'AND'
    a, b = _src_id(netlist, 'A'), _src_id(netlist, 'B')
    assert set(netlist.predecessors(gate)) == {a, b}


def test_or_gate_two_operands():
    netlist = build_netlist(Or([Var('A'), Var('B')]))
    gate = _sink(netlist)
    assert netlist.nodes[gate]['gate'] == 'OR'
    a, b = _src_id(netlist, 'A'), _src_id(netlist, 'B')
    assert set(netlist.predecessors(gate)) == {a, b}


def test_xor_gate():
    netlist = build_netlist(Xor(Var('A'), Var('B')))
    gate = _sink(netlist)
    assert netlist.nodes[gate]['gate'] == 'XOR'
    a, b = _src_id(netlist, 'A'), _src_id(netlist, 'B')
    assert set(netlist.predecessors(gate)) == {a, b}


def test_and_gate_three_operands_is_one_gate_with_fan_in_three():
    netlist = build_netlist(And([Var('A'), Var('B'), Var('C')]))
    gate = _sink(netlist)
    assert netlist.in_degree(gate) == 3
    assert len(_gates(netlist, 'AND')) == 1


# ---------------------------------------------------------------------------
# Operand order is preserved (predecessors() must yield left-to-right, since
# circuit_analyzer.circuit_to_expression_helper renders operands in that
# order via list(netlist.predecessors(node)))
# ---------------------------------------------------------------------------

def test_and_operand_order_is_preserved():
    netlist = build_netlist(And([Var('A'), Var('B'), Var('C')]))
    gate = _sink(netlist)
    ordered_labels = [netlist.nodes[p]['label'] for p in netlist.predecessors(gate)]
    assert ordered_labels == ['A', 'B', 'C']


def test_xor_operand_order_is_preserved():
    netlist = build_netlist(Xor(Var('B'), Var('A')))
    gate = _sink(netlist)
    ordered_labels = [netlist.nodes[p]['label'] for p in netlist.predecessors(gate)]
    assert ordered_labels == ['B', 'A']


# ---------------------------------------------------------------------------
# Explicit nesting produces separate gate nodes (not flattened), mirroring
# the parser's own design: `(A*B)*C` is two distinct AND gates.
# ---------------------------------------------------------------------------

def test_nested_and_produces_two_distinct_and_gates():
    # (A*B)*C
    netlist = build_netlist(And([And([Var('A'), Var('B')]), Var('C')]))
    outer = _sink(netlist)
    assert netlist.nodes[outer]['gate'] == 'AND'
    assert netlist.in_degree(outer) == 2

    inner = [p for p in netlist.predecessors(outer) if netlist.nodes[p]['gate'] == 'AND']
    assert len(inner) == 1
    inner = inner[0]
    assert set(netlist.nodes[p]['label'] for p in netlist.predecessors(inner)) == {'A', 'B'}

    c = [p for p in netlist.predecessors(outer) if netlist.nodes[p].get('label') == 'C']
    assert len(c) == 1


def test_structurally_identical_subtrees_stay_distinct_gates():
    # (A*B) + (A*B)' -- two separately-constructed And nodes, as the parser
    # would produce for two independent occurrences of "A*B".
    ast = Or([
        And([Var('A'), Var('B')]),
        Not(And([Var('A'), Var('B')])),
    ])
    netlist = build_netlist(ast)

    and_gates = _gates(netlist, 'AND')
    assert len(and_gates) == 2, "each occurrence of (A*B) must be its own gate"

    # ... but they can (and should) share the same underlying A/B sources.
    assert len(_gates(netlist, 'src')) == 2


# ---------------------------------------------------------------------------
# Fan-out: the same variable name reuses one 'src' node across gates
# ---------------------------------------------------------------------------

def test_repeated_variable_across_different_gates_shares_one_source():
    # A*B + A*C
    ast = Or([And([Var('A'), Var('B')]), And([Var('A'), Var('C')])])
    netlist = build_netlist(ast)

    assert len(_gates(netlist, 'src')) == 3  # A, B, C -- not 4
    a = _src_id(netlist, 'A')
    assert netlist.out_degree(a) == 2  # A feeds both AND gates
    assert len(_gates(netlist, 'AND')) == 2


def test_repeated_variable_feeding_different_positions_does_not_error():
    # A' * A -- both operands trace back to source A, but through different
    # edges (one via a NOT gate, one directly), so no edge is duplicated.
    ast = And([Not(Var('A')), Var('A')])
    netlist = build_netlist(ast)

    gate = _sink(netlist)
    assert netlist.in_degree(gate) == 2
    a = _src_id(netlist, 'A')
    assert netlist.out_degree(a) == 2  # feeds the NOT gate and the AND gate directly


# ---------------------------------------------------------------------------
# Errors: a source can't feed the same gate through two identical direct
# operands -- a plain DiGraph has no way to represent the resulting fan-in.
# ---------------------------------------------------------------------------

def test_and_with_duplicate_operand_is_an_error():
    with pytest.raises(NetlistBuildError):
        build_netlist(And([Var('A'), Var('A')]))


def test_and_with_duplicate_nonadjacent_operand_is_an_error():
    with pytest.raises(NetlistBuildError):
        build_netlist(And([Var('A'), Var('B'), Var('A')]))


def test_or_with_duplicate_operand_is_an_error():
    with pytest.raises(NetlistBuildError):
        build_netlist(Or([Var('A'), Var('A')]))


def test_xor_with_duplicate_operand_is_an_error():
    with pytest.raises(NetlistBuildError):
        build_netlist(Xor(Var('A'), Var('A')))


def test_duplicate_operand_error_message_names_the_source():
    with pytest.raises(NetlistBuildError) as exc_info:
        build_netlist(And([Var('A'), Var('A')]))
    assert 'A' in str(exc_info.value)


# ---------------------------------------------------------------------------
# End to end with the real parser
# ---------------------------------------------------------------------------

def test_build_netlist_from_parsed_expression():
    ast = parse_boolean_expression("A*B + C'")
    netlist = build_netlist(ast)

    gate = _sink(netlist)
    assert netlist.nodes[gate]['gate'] == 'OR'
    assert len(_gates(netlist, 'src')) == 3
    assert len(_gates(netlist, 'AND')) == 1
    assert len(_gates(netlist, 'NOT')) == 1


def test_build_netlist_from_parsed_expression_with_fan_out():
    ast = parse_boolean_expression('A*B + A*C')
    netlist = build_netlist(ast)

    assert len(_gates(netlist, 'src')) == 3
    a = _src_id(netlist, 'A')
    assert netlist.out_degree(a) == 2
