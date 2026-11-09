"""
Unit tests for circuit_filters.py."""

import os
import random
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit_generator
from rysy_dl.circuit import Circuit
from rysy_dl.circuit_filters import (
    GATE_ONLY_FILTERS,
    every_input_affects_output,
    no_absorption,
    no_collapsable_children,
    no_visible_absorption,
    no_constant_subexpressions,
    valid_xor_xnor_fan_in_filter,
)

# ── test no_collapsable_children ────────────────────────────────────────────────────────────────────


def test_no_collapsable_children_empty_graph_is_true():
    graph = nx.DiGraph()

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_no_edges_is_vacuously_true():
    # No edges means no parent/child pair to check, regardless of gate assignment.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    graph.nodes[0]['gate'] = 'AND'
    graph.nodes[1]['gate'] = 'AND'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_unassigned_gate_is_ignored():
    # Neither node has a 'gate' attribute, so both lookups return None;
    # (None, None) is not in the collapsable set.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_and_and_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'AND'
    graph.nodes[1]['gate'] = 'AND'

    assert no_collapsable_children(graph) is False


def test_no_collapsable_children_nand_and_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'AND'
    graph.nodes[1]['gate'] = 'NAND'

    assert no_collapsable_children(graph) is False


def test_no_collapsable_children_or_or_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'OR'
    graph.nodes[1]['gate'] = 'OR'

    assert no_collapsable_children(graph) is False


def test_no_collapsable_children_nor_or_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'OR'
    graph.nodes[1]['gate'] = 'NOR'

    assert no_collapsable_children(graph) is False


def test_no_collapsable_children_and_or_is_accepted():
    # AND parent with an OR child is not associative-collapsable.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'OR'
    graph.nodes[1]['gate'] = 'AND'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_nand_nand_is_accepted():
    # NAND is not associative, so a NAND child never collapses into a NAND parent.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'NAND'
    graph.nodes[1]['gate'] = 'NAND'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_nor_nor_is_accepted():
    # NOR is not associative, so a NOR child never collapses into a NOR parent.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'NOR'
    graph.nodes[1]['gate'] = 'NOR'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_direction_matters():
    # (AND child, NAND parent) is not a collapsable pair -- only
    # (AND parent, AND/NAND child) is. Swapping which node is the parent
    # should flip the result.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[0]['gate'] = 'NAND'
    graph.nodes[1]['gate'] = 'AND'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_rejects_collapsable_child_even_when_a_sibling_differs():
    """A parent with one matching (collapsable) child and one differing child must
    still be rejected. This is the case the old some_child_differs_from_parent_filter
    got wrong: it only inspected predecessors until the first one differed from the
    parent, then moved on without checking the rest.

    Edges are added so the differing child (src) is discovered before the
    collapsable child (AND), reproducing the exact order that fooled the old code.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2, 3, 4])
    graph.add_edge(0, 2)  # inputs a, b -> inner AND
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)  # differing (src) child of parent, added first
    graph.add_edge(2, 4)  # collapsable (AND) child of parent, added second

    graph.nodes[0]['gate'] = 'src'
    graph.nodes[1]['gate'] = 'src'
    graph.nodes[2]['gate'] = 'AND'
    graph.nodes[3]['gate'] = 'src'
    graph.nodes[4]['gate'] = 'AND'

    assert no_collapsable_children(graph) is False


def test_no_collapsable_children_accepts_mixed_children_with_no_collapsable_pair():
    """A parent whose children differ from it in gate type (and aren't a
    collapsable NAND/AND or NOR/OR pair) should not be rejected."""
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2, 3, 4])
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    graph.add_edge(2, 4)

    graph.nodes[0]['gate'] = 'src'
    graph.nodes[1]['gate'] = 'src'
    graph.nodes[2]['gate'] = 'OR'
    graph.nodes[3]['gate'] = 'src'
    graph.nodes[4]['gate'] = 'AND'

    assert no_collapsable_children(graph) is True


def test_no_collapsable_children_scans_whole_graph_not_just_first_parent():
    # 0 -> 1 is fine (src -> AND), but 2 -> 3 is a collapsable AND -> AND
    # pair elsewhere in the graph. The whole graph must be scanned, not just
    # the first parent encountered.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)

    graph.nodes[0]['gate'] = 'src'
    graph.nodes[1]['gate'] = 'AND'
    graph.nodes[2]['gate'] = 'AND'
    graph.nodes[3]['gate'] = 'AND'

    assert no_collapsable_children(graph) is False


# ── test valid_xor_xnor_fan_in_filter ───────────────────────────────────────────────────────────────


def test_valid_xor_xnor_fan_in_filter_empty_graph_is_true():
    graph = nx.DiGraph()

    assert valid_xor_xnor_fan_in_filter(graph) is True


def test_valid_xor_xnor_fan_in_filter_no_xor_or_xnor_gates_is_true():
    # Non-XOR/XNOR gates are never checked, no matter their fan-in.
    graph = nx.DiGraph()
    graph.add_edge(0, 3)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.nodes[3]['gate'] = 'AND'

    assert valid_xor_xnor_fan_in_filter(graph) is True


def test_valid_xor_xnor_fan_in_filter_xor_with_two_inputs_is_true():
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.nodes[2]['gate'] = 'XOR'

    assert valid_xor_xnor_fan_in_filter(graph) is True


def test_valid_xor_xnor_fan_in_filter_xnor_with_two_inputs_is_true():
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.nodes[2]['gate'] = 'XNOR'

    assert valid_xor_xnor_fan_in_filter(graph) is True


def test_valid_xor_xnor_fan_in_filter_xor_with_one_input_is_false():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.nodes[1]['gate'] = 'XOR'

    assert valid_xor_xnor_fan_in_filter(graph) is False


def test_valid_xor_xnor_fan_in_filter_xor_with_three_inputs_is_false():
    graph = nx.DiGraph()
    graph.add_edge(0, 3)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.nodes[3]['gate'] = 'XOR'

    assert valid_xor_xnor_fan_in_filter(graph) is False


def test_valid_xor_xnor_fan_in_filter_xnor_with_zero_inputs_is_false():
    # A source node can't legitimately be an XNOR gate, but the filter only
    # looks at the 'gate' attribute and in_degree, so this is still exercised.
    graph = nx.DiGraph()
    graph.add_node(0)
    graph.nodes[0]['gate'] = 'XNOR'

    assert valid_xor_xnor_fan_in_filter(graph) is False


def test_valid_xor_xnor_fan_in_filter_unassigned_gate_is_ignored():
    graph = nx.DiGraph()
    graph.add_node(0)

    assert valid_xor_xnor_fan_in_filter(graph) is True


def test_valid_xor_xnor_fan_in_filter_scans_whole_graph_not_just_first_gate():
    # Node 2 is a valid XOR (fan-in 2), but node 5 is an invalid XNOR
    # (fan-in 3) elsewhere in the graph. The whole graph must be scanned.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    graph.add_edge(2, 5)

    graph.nodes[2]['gate'] = 'XOR'
    graph.nodes[5]['gate'] = 'XNOR'

    assert valid_xor_xnor_fan_in_filter(graph) is False


def test_valid_xor_xnor_fan_in_filter_mixed_gates_only_xor_xnor_checked():
    # An AND gate with fan-in 3 is fine; only the XOR/XNOR gates are subject
    # to the fan-in-2 restriction.
    graph = nx.DiGraph()
    graph.add_edge(0, 3)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(4, 6)
    graph.add_edge(5, 6)

    graph.nodes[3]['gate'] = 'AND'
    graph.nodes[6]['gate'] = 'XOR'

    assert valid_xor_xnor_fan_in_filter(graph) is True


# ── test no_constant_subexpressions ──────────────────────────────────────────────────────────────────


def _netlist(expression):
    return Circuit.from_expression(expression).netlist


def test_no_constant_subexpressions_empty_graph_is_true():
    graph = nx.DiGraph()

    assert no_constant_subexpressions(graph) is True


@pytest.mark.parametrize("expression", [
    "A*B + C'",
    "(A + B)*C",
    "(A ^ B)'",
    "A + A'*B",
    "A*B + A'*C",
])
def test_no_constant_subexpressions_accepts_expressions_with_no_constant_subexpression(expression):
    assert no_constant_subexpressions(_netlist(expression)) is True


@pytest.mark.parametrize("expression", [
    "A*A'",
    "A + A'",
    "A ^ A'",
    "A*B*A'",
    "(A*B)' + A",
])
def test_no_constant_subexpressions_rejects_constant_output(expression):
    assert no_constant_subexpressions(_netlist(expression)) is False


def test_no_constant_subexpressions_rejects_constant_inner_subexpression():
    # The output (A*A') + B == B is not constant, but the A*A' subexpression is always 0.
    assert no_constant_subexpressions(_netlist("(A*A') + B")) is False


def test_no_constant_subexpressions_rejects_constant_subexpression_feeding_a_nor():
    # (A*B)' + (A' + B')' -- the NAND and the NOR are complements, so the OR is always 1.
    assert no_constant_subexpressions(_netlist("(A*B)' + (A' + B')'")) is False


def test_no_constant_subexpressions_xnor_of_complements_is_rejected():
    # XNOR(A, A') is always 0. (The parser has no XNOR, so build the graph by hand.)
    graph = nx.DiGraph()
    graph.add_node(0, gate='src')
    graph.add_node(1, gate='NOT')
    graph.add_node(2, gate='XNOR')
    graph.add_edges_from([(0, 1), (0, 2), (1, 2)])

    assert no_constant_subexpressions(graph) is False


def test_no_constant_subexpressions_xnor_of_independent_inputs_is_accepted():
    graph = nx.DiGraph()
    graph.add_node(0, gate='src')
    graph.add_node(1, gate='src')
    graph.add_node(2, gate='XNOR')
    graph.add_edges_from([(0, 2), (1, 2)])

    assert no_constant_subexpressions(graph) is True


def test_no_constant_subexpressions_sources_identified_by_in_degree_not_gate():
    # Sources from a bare graph may have no 'gate' attribute; they still count as inputs.
    graph = nx.DiGraph()
    graph.add_node(2, gate='AND')
    graph.add_edges_from([(0, 2), (1, 2)])

    assert no_constant_subexpressions(graph) is True


def test_no_constant_subexpressions_unsupported_gate_raises():
    graph = nx.DiGraph()
    graph.add_node(1, gate='MUX')
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="Unsupported gate type"):
        no_constant_subexpressions(graph)


# ── test every_input_affects_output ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('expression', [
    'A*B + C',
    "A + A'*B",            # = A + B: both inputs matter
    'A ^ B',
    "(A + B)*(A' + C)",
    "A*B + A'*C",          # a multiplexer: all three matter
    'A ^ (B*(A + C))',
])
def test_every_input_affects_output_accepts(expression):
    assert every_input_affects_output(_netlist(expression)) is True


@pytest.mark.parametrize('expression', [
    'A + A*B',             # = A: B never matters
    'A*(A + B)',           # = A
    'A ^ (A ^ B)',         # = B: A never matters
    "A*B + A*B'",          # = A
    "(A + B)*(A + B')*C",  # = A*C
])
def test_every_input_affects_output_rejects(expression):
    assert every_input_affects_output(_netlist(expression)) is False


def test_every_input_affects_output_rejects_an_input_with_no_edges():
    graph = _netlist('A*B')
    graph.add_node(99, gate='src', label='C')

    assert every_input_affects_output(graph) is False


def test_every_input_affects_output_checks_each_input_against_every_output():
    # two outputs: A affects only the first, B only the second -- both inputs matter
    graph = nx.DiGraph()
    graph.add_node(0, gate='src')
    graph.add_node(1, gate='src')
    graph.add_node(2, gate='NOT')
    graph.add_node(3, gate='NOT')
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)

    assert every_input_affects_output(graph) is True


def test_every_input_affects_output_is_not_a_gate_only_filter():
    # NOTs on source edges change its answer (A + A*B vs. A + A'*B), so it runs after them
    assert every_input_affects_output not in GATE_ONLY_FILTERS


# ── test no_absorption ───────────────────────────────────────────────────────────────────────────────


def _gate_graph(gates, edges):
    # gates: {node: gate}; sources are given gate 'src'
    graph = nx.DiGraph()
    for node, gate in gates.items():
        graph.add_node(node, gate=gate)
    graph.add_edges_from(edges)
    return graph


def test_no_absorption_empty_graph_is_true():
    assert no_absorption(nx.DiGraph()) is True


@pytest.mark.parametrize("expression", [
    "A*B + C'",
    "(A + B)*C",
    "A + A'*B",            # A + A'B = A + B is a different law, not absorption
    "A*(A' + B)",          # likewise A*(A' + B) = A*B
    "A*B + A'*C",
    "A*B + A*C",           # shared factor, but neither term implies the other
    "(A + B)*(A + C)",     # distributive, not absorption
    "(A*B)' + C",
    "A ^ (A*B)",           # XOR(A, A*B) = A*B'; XOR does not absorb
])
def test_no_absorption_accepts_expressions_without_absorption(expression):
    assert no_absorption(_netlist(expression)) is True


@pytest.mark.parametrize("expression", [
    "A + A*B",
    "A*(A + B)",
    "A*B + A",             # order of the terms does not matter
    "(A + B)*A",
    "A' + A'*B",           # the absorbing term can be a complemented literal
    "A*B + A*B*C",         # the absorbing term can be a product
    "(A + B)*(A + B + C)",  # ... or a sum
    "A + B + A*C",         # multi-input OR
    "A*B*(A + C)",         # multi-input AND
    "A + A*(B + C)",       # the absorbed term can be arbitrarily complicated
    "A + A*(B + C*D)'",
    "A*B + A*B*(C + D)",   # absorbing term is not a single literal: A*B*(C + D) implies A*B
])
def test_no_absorption_rejects_absorption(expression):
    assert no_absorption(_netlist(expression)) is False


@pytest.mark.parametrize("expression", [
    "A + (A' + B)'",       # (A' + B)' = A*B', so this is A + A*B'
    "(A + B)' + A'",       # (A + B)' = A'*B', so this is A' + A'*B'
    "A*(A'*B)'",           # (A'*B)' = A + B', so this is A*(A + B')
    "(A*B)'*A'",           # (A*B)' = A' + B', so this is A'*(A' + B')
])
def test_no_absorption_rejects_absorption_hidden_by_de_morgan(expression):
    assert no_absorption(_netlist(expression)) is False


@pytest.mark.parametrize("expression", [
    "(A*B)' + (A*B)'*C",   # terms A', B', (A*B)'*C -- the absorber (A*B)' was split in two
    "A + B + (A + B)*C",   # the absorber A + B is two terms of the outer OR
    "A*B*(A*B + C)",       # dual: the absorber A*B is two factors of the outer AND
    "(A + B)'*((A + B)' + C)",
])
def test_no_absorption_rejects_absorber_split_across_several_terms(expression):
    assert no_absorption(_netlist(expression)) is False


def test_no_absorption_rejects_absorption_below_the_output():
    # The output gate is fine; the absorption is in the subexpression A + A*(B + D).
    assert no_absorption(_netlist("C*(A + A*(B + D))")) is False


def test_no_absorption_rejects_identical_terms_from_different_gates():
    # A*B and B*A are different gates computing the same function: X + X.
    assert no_absorption(_netlist("A*B + B*A")) is False


def test_no_absorption_rejects_constant_term():
    # A*A' is always 0, and 0 + B is a (degenerate) absorption.
    assert no_absorption(_netlist("(A*A') + B")) is False


def test_no_absorption_rejects_absorption_across_nested_or_gates():
    # OR(OR(A, C), A*B) = A + C + A*B: the absorbing term A is one level down.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'OR', 4: 'AND', 5: 'OR'},
        [(0, 3), (2, 3), (0, 4), (1, 4), (3, 5), (4, 5)],
    )
    assert no_absorption(graph) is False


def test_no_absorption_rejects_absorbed_term_inside_nested_or_gate():
    # OR(OR(A*B, C), A) = A*B + C + A: the absorbed term A*B is one level down,
    # so neither direct input of the outer OR implies the other.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'AND', 4: 'OR', 5: 'OR'},
        [(0, 3), (1, 3), (3, 4), (2, 4), (4, 5), (0, 5)],
    )
    assert no_absorption(graph) is False


def test_no_absorption_rejects_absorption_inside_nor():
    # NOR(A, A*B) = (A + A*B)'
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'AND', 3: 'NOR'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_absorption(graph) is False


def test_no_absorption_rejects_absorption_inside_nand():
    # NAND(A, A + B) = (A*(A + B))'
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'OR', 3: 'NAND'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_absorption(graph) is False


def test_no_absorption_rejects_absorption_through_nand_child():
    # OR(NAND(A', B'), A) -- NAND(A', B') = A + B, so this is A + B + A.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'NOT', 3: 'NOT', 4: 'NAND', 5: 'OR'},
        [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5), (0, 5)],
    )
    assert no_absorption(graph) is False


def test_no_absorption_accepts_nand_and_nor_without_absorption():
    # NOR(NAND(A, B), C)
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'NAND', 4: 'NOR'},
        [(0, 3), (1, 3), (3, 4), (2, 4)],
    )
    assert no_absorption(graph) is True


def test_no_absorption_does_not_flatten_xnor():
    # XNOR(A, A*B) is not an absorption pattern.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'AND', 3: 'XNOR'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_absorption(graph) is True


def test_no_absorption_sources_identified_by_in_degree_not_gate():
    # Sources from a bare graph may have no 'gate' attribute; they still count as inputs.
    graph = nx.DiGraph()
    graph.add_node(2, gate='AND')
    graph.add_node(3, gate='OR')
    graph.add_edges_from([(0, 2), (1, 2), (0, 3), (2, 3)])  # A + A*B

    assert no_absorption(graph) is False


@pytest.mark.parametrize("expression", [
    "A + B*(C + A*D)",     # = A + B*C, but only after distributing B
    "A*B + A'*C + B*C",    # consensus: B*C is covered by the other two terms together
])
def test_no_absorption_does_not_detect_redundancy_beyond_absorption(expression):
    # Documents the filter's scope: these simplify, but not by absorption among the
    # terms of a single (flattened) AND/OR.
    assert no_absorption(_netlist(expression)) is True


def test_no_absorption_unsupported_gate_raises():
    graph = nx.DiGraph()
    graph.add_node(1, gate='MUX')
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="Unsupported gate type"):
        no_absorption(graph)


# ── test no_visible_absorption ───────────────────────────────────────────────────────────────────────


def test_no_visible_absorption_empty_graph_is_true():
    assert no_visible_absorption(nx.DiGraph()) is True


@pytest.mark.parametrize("expression", [
    "A*B + C'",
    "(A + B)*C",
    "A + A'*B",
    "A*(A' + B)",
    "A*B + A'*C",
    "A*B + A*C",
    "(A + B)*(A + C)",
    "(A*B)' + C",
    "A ^ (A*B)",
    "A + B*(C + A*D)",
    "A*B + A'*C + B*C",
])
def test_no_visible_absorption_accepts_expressions_without_absorption(expression):
    assert no_visible_absorption(_netlist(expression)) is True


@pytest.mark.parametrize("expression", [
    "A + A*B",
    "A*(A + B)",
    "A*B + A",
    "(A + B)*A",
    "A' + A'*B",
    "A*B + A*B*C",
    "(A + B)*(A + B + C)",
    "A + B + A*C",
    "A*B*(A + C)",
    "A + A*(B + C)",
    "A + A*(B + C*D)'",
    "A*B + A*B*(C + D)",
    "C*(A + A*(B + D))",       # absorption below the output
    "(A*B)' + (A*B)'*C",       # the absorber may be a negated subexpression
    "(A ^ B) + (A ^ B)*C",     # ... or an XOR
    "A + B + (A + B)*C",       # the absorber spans several terms of the OR
    "A*B*(A*B + C)",           # ... or several factors of the AND
])
def test_no_visible_absorption_rejects_visible_absorption(expression):
    assert no_visible_absorption(_netlist(expression)) is False


@pytest.mark.parametrize("expression", [
    "A*B + B*A",               # inputs are compared in sorted order
    "(A + B)*(B + A)",
    "(A ^ B) + (B ^ A)",
    "A*B*C + C*(B*A)",         # nested ANDs are merged before comparing
    "(A*B)' + (B*A)'",
])
def test_no_visible_absorption_rejects_terms_written_the_same_way(expression):
    assert no_visible_absorption(_netlist(expression)) is False


@pytest.mark.parametrize("expression", [
    "A + (A' + B)'",           # = A + A*B', but only after De Morgan
    "(A + B)' + A'",           # = A'*B' + A'
    "A*(A'*B)'",               # = A*(A + B')
    "(A*B)'*A'",               # = A'*(A' + B')
    "(A ^ B)*(A*B)'",          # A ^ B implies (A*B)', but nothing on the page shows it
    "A*B*C*(B*(A ^ D)*D)'",    # when A = 1, (B*(A ^ D)*D)' is always 1
    "A ^ (A ^ C) + C",         # A ^ (A ^ C) is C, but is not written as C
])
def test_no_visible_absorption_accepts_absorption_that_is_not_visible(expression):
    graph = _netlist(expression)
    assert no_absorption(graph) is False
    assert no_visible_absorption(graph) is True


def test_no_visible_absorption_shared_gate_counts_as_written_twice():
    # X = A*B fans out to both inputs of an OR: X + X*C.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'AND', 4: 'AND', 5: 'OR'},
        [(0, 3), (1, 3), (3, 4), (2, 4), (3, 5), (4, 5)],
    )
    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_distinct_sources_are_distinct_variables():
    # A*B + C*D: sources are identified by node, so no two terms match.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'src', 4: 'AND', 5: 'AND', 6: 'OR'},
        [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)],
    )
    assert no_visible_absorption(graph) is True


def test_no_visible_absorption_rejects_absorption_across_nested_or_gates():
    # OR(OR(A, C), A*B) = A + C + A*B
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'OR', 4: 'AND', 5: 'OR'},
        [(0, 3), (2, 3), (0, 4), (1, 4), (3, 5), (4, 5)],
    )
    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_rejects_absorption_inside_nor():
    # NOR(A, A*B)
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'AND', 3: 'NOR'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_rejects_absorption_inside_nand():
    # NAND(A, A + B)
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'OR', 3: 'NAND'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_nand_gate_matches_negated_and():
    # NAND(A, B) + (A*B)'*C -- a NAND gate and NOT(AND) are written the same way, (A*B)'.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'src', 3: 'NAND', 4: 'AND', 5: 'NOT', 6: 'AND', 7: 'OR'},
        [(0, 3), (1, 3), (0, 4), (1, 4), (4, 5), (5, 6), (2, 6), (3, 7), (6, 7)],
    )
    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_does_not_look_through_de_morgan_into_nand_inputs():
    # OR(NAND(A', B'), A) = (A' * B')' + A = A + B + A -- true absorption, but written
    # as (A'*B')' + A it does not look like X + X*Y.
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'NOT', 3: 'NOT', 4: 'NAND', 5: 'OR'},
        [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5), (0, 5)],
    )
    assert no_absorption(graph) is False
    assert no_visible_absorption(graph) is True


def test_no_visible_absorption_does_not_flatten_xnor():
    graph = _gate_graph(
        {0: 'src', 1: 'src', 2: 'AND', 3: 'XNOR'},
        [(0, 2), (1, 2), (0, 3), (2, 3)],
    )
    assert no_visible_absorption(graph) is True


def test_no_visible_absorption_sources_identified_by_in_degree_not_gate():
    graph = nx.DiGraph()
    graph.add_node(2, gate='AND')
    graph.add_node(3, gate='OR')
    graph.add_edges_from([(0, 2), (1, 2), (0, 3), (2, 3)])  # A + A*B

    assert no_visible_absorption(graph) is False


def test_no_visible_absorption_unsupported_gate_raises():
    graph = nx.DiGraph()
    graph.add_node(1, gate='MUX')
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="Unsupported gate type"):
        no_visible_absorption(graph)


def test_no_visible_absorption_never_rejects_what_no_absorption_accepts():
    # Visible absorption is a special case of absorption: on random circuits, whenever
    # no_absorption accepts, no_visible_absorption must accept too.
    random.seed(351)
    shapes = [
        [(0, 3), (1, 3), (3, 4), (2, 4), (1, 5), (4, 5)],
        [(0, 3), (1, 3), (0, 4), (3, 4), (2, 5), (4, 5), (3, 6), (5, 6)],
        [(0, 4), (1, 4), (2, 5), (4, 5), (0, 6), (3, 6), (5, 7), (6, 7), (4, 7)],
    ]
    gates = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']
    both_reject = 0
    for edges in shapes:
        for _ in range(500):
            graph = nx.DiGraph(edges)
            circuit_generator.assign_gates_randomly(graph, ensure_diversity=False, gate_list=gates)
            # randomly replace some 1-input positions with NOT by inserting inverters
            for u, v in list(graph.edges()):
                if random.random() < 0.2:
                    n = graph.number_of_nodes()
                    graph.remove_edge(u, v)
                    graph.add_node(n, gate='NOT')
                    graph.add_edges_from([(u, n), (n, v)])
            if no_absorption(graph):
                assert no_visible_absorption(graph), sorted(graph.nodes(data='gate'))
            elif not no_visible_absorption(graph):
                both_reject += 1
    assert both_reject > 0  # the comparison actually exercised rejecting circuits
