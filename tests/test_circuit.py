"""
Unit tests for circuit.py."""

import os
import random
import sys
from unittest import mock

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit as circuit_module
from rysy_dl import circuit_filters
from rysy_dl.circuit import Circuit

# ── test Circuit.assign_gates_randomly ──────────────────────────────────────────────────────────────


def _raw_graph():
    # Three sources (0, 1, 2) feeding three gates; 5 is the single sink.
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 3), (1, 3), (3, 4), (2, 4), (1, 5), (4, 5)])
    return graph


def _fake_generator(monkeypatch, assignments):
    """Replace circuit_generator.assign_gates_randomly with a mock that, on its i-th
    call, sets every node's gate to assignments[i] (a {node: gate} dict). Returns the mock."""
    calls = iter(assignments)

    def fake(netlist, *args, **kwargs):
        nx.set_node_attributes(netlist, next(calls), 'gate')

    generator = mock.Mock(side_effect=fake)
    monkeypatch.setattr(circuit_module.circuit_generator, 'assign_gates_randomly', generator)
    return generator


def _gates(gate3, gate4, gate5):
    return {0: 'src', 1: 'src', 2: 'src', 3: gate3, 4: gate4, 5: gate5}


def _count_filter(result):
    # a filter that always returns `result` and records how often it was called
    return mock.Mock(return_value=result)


def test_assign_gates_randomly_default_rejects_collapsable_children(monkeypatch):
    # Default filters are no_collapsable_children (+ the required XOR/XNOR filter):
    # the first assignment has an AND feeding an AND, so it must be retried.
    generator = _fake_generator(monkeypatch, [
        _gates('AND', 'AND', 'OR'),
        _gates('AND', 'OR', 'AND'),
    ])
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly()

    assert generator.call_count == 2
    assert circuit.netlist.nodes[4]['gate'] == 'OR'


def test_assign_gates_randomly_applies_every_user_filter(monkeypatch):
    # The first assignment passes filter_a but not filter_b, so it must be retried.
    generator = _fake_generator(monkeypatch, [
        _gates('AND', 'OR', 'AND'),
        _gates('OR', 'AND', 'OR'),
    ])
    filter_a = mock.Mock(return_value=True)
    filter_b = mock.Mock(side_effect=lambda g: g.nodes[3]['gate'] == 'OR')
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly(filters=[filter_a, filter_b])

    assert generator.call_count == 2
    assert circuit.netlist.nodes[3]['gate'] == 'OR'
    assert filter_b.call_count == 2


def test_assign_gates_randomly_custom_filters_replace_the_default(monkeypatch):
    # AND feeding AND would be rejected by the default, but the caller's filter list
    # does not include no_collapsable_children, so it is accepted.
    generator = _fake_generator(monkeypatch, [_gates('AND', 'AND', 'OR')])
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly(filters=[_count_filter(True)])

    assert generator.call_count == 1


def test_assign_gates_randomly_empty_filters_still_requires_xor_fan_in(monkeypatch):
    # Give node 5 a fan-in of 3, so an XOR there is invalid; filters=() must still reject it.
    graph = _raw_graph()
    graph.add_edge(0, 5)  # node 5 now has fan-in 3
    generator = _fake_generator(monkeypatch, [
        _gates('AND', 'OR', 'XOR'),
        _gates('AND', 'OR', 'AND'),
    ])
    circuit = Circuit(graph)

    circuit.assign_gates_randomly(filters=())

    assert generator.call_count == 2
    assert circuit.netlist.nodes[5]['gate'] == 'AND'


def test_assign_gates_randomly_xor_filter_is_always_called(monkeypatch):
    _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')])
    xor_filter = mock.Mock(return_value=True)
    monkeypatch.setattr(circuit_module.circuit_filters, 'valid_xor_xnor_fan_in_filter', xor_filter)
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly(filters=[_count_filter(True)])

    xor_filter.assert_called_once_with(circuit.netlist)


def test_assign_gates_randomly_xor_filter_listed_explicitly_is_not_run_twice(monkeypatch):
    _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')])
    xor_filter = mock.Mock(return_value=True)
    monkeypatch.setattr(circuit_module.circuit_filters, 'valid_xor_xnor_fan_in_filter', xor_filter)
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly(filters=[xor_filter])

    xor_filter.assert_called_once_with(circuit.netlist)


def test_assign_gates_randomly_fallback_drops_xor_and_keeps_user_filters(monkeypatch):
    # The user's filter rejects the first 100 (diversity) assignments, so the method
    # falls back to AND/OR/NAND/NOR-only assignments -- which must still pass the filter.
    attempts = []

    def fake(netlist, *args, **kwargs):
        attempts.append((args, kwargs))
        gate = 'OR' if len(attempts) >= 102 else 'AND'
        nx.set_node_attributes(netlist, _gates(gate, 'NAND', 'NOR'), 'gate')

    monkeypatch.setattr(circuit_module.circuit_generator, 'assign_gates_randomly', fake)
    user_filter = mock.Mock(side_effect=lambda g: g.nodes[3]['gate'] == 'OR')
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly(filters=[user_filter])

    assert len(attempts) == 102
    assert all(a == ((), {}) for a in attempts[:100])
    assert all(
        a == ((), {'ensure_diversity': False, 'gate_list': ['AND', 'OR', 'NAND', 'NOR']})
        for a in attempts[100:]
    )
    assert circuit.netlist.nodes[3]['gate'] == 'OR'


def test_assign_gates_randomly_raises_when_no_assignment_passes(monkeypatch):
    generator = _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')] * 600)
    circuit = Circuit(_raw_graph())

    with pytest.raises(AssertionError, match='Unable to randomly assign gates'):
        circuit.assign_gates_randomly(filters=[_count_filter(False)])

    assert generator.call_count == 600


def test_assign_gates_randomly_real_generator_satisfies_all_filters():
    # No mocks: the result must pass every requested filter plus the XOR/XNOR filter.
    filters = [
        circuit_filters.no_collapsable_children,
        circuit_filters.no_absorption,
        circuit_filters.no_constant_subexpressions,
    ]
    for _ in range(20):
        circuit = Circuit(_raw_graph())
        circuit.assign_gates_randomly(filters=filters)

        assert circuit_filters.valid_xor_xnor_fan_in_filter(circuit.netlist)
        for f in filters:
            assert f(circuit.netlist)


def _not_nodes(graph):
    return [v for v, gate in graph.nodes(data='gate') if gate == 'NOT']


def _add_not(netlist, source=0, target=3):
    # stand-in for insert_random_source_edge_nots: route source -> target through a NOT
    node = len(netlist)
    netlist.add_node(node, gate='NOT')
    netlist.remove_edge(source, target)
    netlist.add_edges_from([(source, node), (node, target)])


def test_assign_gates_randomly_never_adds_nots():
    circuit = Circuit(_raw_graph())

    circuit.assign_gates_randomly()

    assert _not_nodes(circuit.netlist) == []
    assert len(circuit.netlist) == 6


def test_assign_gates_randomly_keeps_existing_nots():
    graph = _raw_graph()
    _add_not(graph)
    circuit = Circuit(graph)

    circuit.assign_gates_randomly()

    assert _not_nodes(circuit.netlist) == [6]


def test_assign_gates_randomly_keeps_the_same_netlist_object():
    circuit = Circuit(_raw_graph())
    netlist = circuit.netlist

    circuit.assign_gates_randomly()

    assert circuit.netlist is netlist


# ── test Circuit.insert_random_source_nots ──────────────────────────────────────────────────────────


def test_insert_random_source_nots_puts_nots_only_after_sources():
    random.seed(4)
    saw_not = False
    for _ in range(20):
        circuit = Circuit(_raw_graph())
        circuit.assign_gates_randomly()
        gates = dict(circuit.netlist.nodes(data='gate'))

        circuit.insert_random_source_nots()

        netlist = circuit.netlist
        assert len(_not_nodes(netlist)) <= 3  # at most one NOT per source
        for node in _not_nodes(netlist):
            (parent,) = netlist.predecessors(node)
            assert netlist.in_degree(parent) == 0
            saw_not = True
        assert all(netlist.nodes[v]['gate'] == gate for v, gate in gates.items())  # gates unchanged
    assert saw_not


def test_insert_random_source_nots_does_not_filter(monkeypatch):
    calls = []
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', calls.append)
    circuit = Circuit(_raw_graph())

    circuit.insert_random_source_nots()

    assert calls == [circuit.netlist]


def test_insert_random_source_nots_rejects_netlist_with_nots():
    graph = _raw_graph()
    _add_not(graph)
    circuit = Circuit(graph)

    with pytest.raises(ValueError, match='no NOT gates'):
        circuit.insert_random_source_nots()


# ── test Circuit.randomize ──────────────────────────────────────────────────────────────────────────


def test_randomize_filters_after_inserting_nots(monkeypatch):
    # The filter only accepts a circuit that already has a NOT gate.
    _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')])
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', _add_not)
    circuit = Circuit(_raw_graph())

    circuit.randomize(filters=[lambda g: len(_not_nodes(g)) == 1])

    assert _not_nodes(circuit.netlist) == [6]


def test_randomize_without_source_nots_adds_no_nots(monkeypatch):
    nots = mock.Mock()
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', nots)
    circuit = Circuit(_raw_graph())

    circuit.randomize(source_nots=False)

    nots.assert_not_called()
    assert _not_nodes(circuit.netlist) == []


def test_randomize_tries_several_not_patterns_per_gate_assignment(monkeypatch):
    # The NOT-dependent filter rejects the first 3 finished circuits, so the 4th NOT pattern
    # on the *same* gate assignment is accepted -- no new gate assignment is drawn.
    generator = _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')])
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', _add_not)
    after_nots = mock.Mock(side_effect=[False, False, False, True])
    circuit = Circuit(_raw_graph())

    circuit.randomize(filters=[after_nots])

    assert generator.call_count == 1
    assert after_nots.call_count == Circuit._NOT_PATTERNS_PER_GATE_ASSIGNMENT == 4
    assert _not_nodes(circuit.netlist) == [6]  # NOTs from rejected patterns don't accumulate


def test_randomize_draws_new_gates_after_all_not_patterns_fail(monkeypatch):
    generator = _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND'), _gates('OR', 'AND', 'OR')])
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', _add_not)
    after_nots = mock.Mock(side_effect=lambda g: g.nodes[3]['gate'] == 'OR')
    circuit = Circuit(_raw_graph())

    circuit.randomize(filters=[after_nots])

    assert generator.call_count == 2
    assert after_nots.call_count == 4 + 1
    assert _not_nodes(circuit.netlist) == [6]


def test_randomize_checks_gate_only_filters_before_adding_nots(monkeypatch):
    # AND feeding AND fails no_collapsable_children; NOTs can't fix that, so none are tried.
    generator = _fake_generator(monkeypatch, [_gates('AND', 'AND', 'OR'), _gates('AND', 'OR', 'AND')])
    nots = mock.Mock(side_effect=_add_not)
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', nots)
    circuit = Circuit(_raw_graph())

    circuit.randomize(filters=[circuit_filters.no_collapsable_children])

    assert generator.call_count == 2
    assert nots.call_count == 1  # only for the second gate assignment


def test_randomize_gives_up_after_all_attempts(monkeypatch):
    generator = _fake_generator(monkeypatch, [_gates('AND', 'OR', 'AND')] * 600)
    nots = mock.Mock()
    monkeypatch.setattr(circuit_module.circuit_generator, 'insert_random_source_edge_nots', nots)
    circuit = Circuit(_raw_graph())

    with pytest.raises(AssertionError, match='Unable to randomly assign gates'):
        circuit.randomize(filters=[mock.Mock(return_value=False)])

    assert generator.call_count == 600
    assert nots.call_count == 600 * 4


def test_randomize_keeps_the_same_netlist_object_and_labels():
    circuit = Circuit(_raw_graph())
    circuit.assign_source_labels()
    netlist = circuit.netlist

    circuit.randomize()

    assert circuit.netlist is netlist
    assert [netlist.nodes[v]['label'] for v in (0, 1, 2)] == ['a', 'b', 'c']
    assert sorted(netlist.nodes) == list(range(len(netlist)))


def test_randomize_rejects_netlist_with_nots():
    graph = _raw_graph()
    _add_not(graph)
    circuit = Circuit(graph)

    with pytest.raises(ValueError, match='no NOT gates'):
        circuit.randomize()


def test_randomize_real_generator_satisfies_all_filters():
    # No mocks: the finished circuit -- NOTs included -- passes every filter. In particular
    # no_constant_subexpressions must see the NOTs (A*A' only appears after they are added).
    random.seed(3)
    filters = [
        circuit_filters.no_collapsable_children,
        circuit_filters.no_visible_absorption,
        circuit_filters.no_constant_subexpressions,
    ]
    saw_not = False
    for _ in range(30):
        circuit = Circuit(_raw_graph())
        circuit.randomize(filters=filters)

        saw_not |= bool(_not_nodes(circuit.netlist))
        assert circuit_filters.valid_xor_xnor_fan_in_filter(circuit.netlist)
        for f in filters:
            assert f(circuit.netlist)
    assert saw_not


# ── test circuit_filters.GATE_ONLY_FILTERS ──────────────────────────────────────────────────────────


def test_gate_only_filters_really_ignore_source_nots():
    # Adding source NOTs never changes the answer of a filter listed as gate-only.
    from rysy_dl import circuit_generator
    random.seed(5)
    for _ in range(300):
        circuit = Circuit(_raw_graph())
        circuit_generator.assign_gates_randomly(
            circuit.netlist, ensure_diversity=False, gate_list=['AND', 'OR', 'XOR', 'NAND', 'NOR'])
        before = {f: f(circuit.netlist) for f in circuit_filters.GATE_ONLY_FILTERS}
        circuit.insert_random_source_nots()
        assert {f: f(circuit.netlist) for f in circuit_filters.GATE_ONLY_FILTERS} == before
