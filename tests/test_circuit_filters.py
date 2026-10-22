"""
Unit tests for circuit_filters.py."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.circuit import Circuit
from rysy_dl.circuit_filters import (
    no_collapsable_children,
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
