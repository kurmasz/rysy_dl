"""
Tests for circuit_layout: regression tests for specific bugs.

Whether a drawing is correct in general is checked by layout_validity (see
test_layout_validity.py); each test here pins down one bug that got past it,
usually because the corpus happened never to contain the triggering input.
"""

import networkx as nx
import pytest
import schemdraw

from rysy_dl.circuit_layout import build_layout, compute_layout
from rysy_dl.layout import ordering
from rysy_dl.layout_validity import check_layout


@pytest.fixture(autouse=True)
def svg_backend():
    schemdraw.use('svg')


def netlist_from(gates, edges):
    """`gates` maps node id -> gate type ('src' for an input)."""
    netlist = nx.DiGraph()
    for node, gate in gates.items():
        netlist.add_node(node, gate=gate)
    netlist.add_edges_from(edges)
    return netlist


def assert_valid(netlist):
    state = build_layout(netlist)
    assert check_layout(netlist, state) == []
    return state


def test_sink_may_be_node_zero():
    # 0 is falsy, and the sink's id used to be tested for truth.
    assert_valid(netlist_from({0: 'OR', 1: 'src', 2: 'src'}, [(1, 0), (2, 0)]))


def test_node_ids_need_not_be_contiguous():
    # Source 1 reaches gate 4 across a layer, so a dummy node is needed. Its id
    # used to be len(netlist) == 4: gate 4 itself. This used to hang.
    netlist = netlist_from({0: 'src', 1: 'src', 3: 'AND', 4: 'OR'},
                           [(0, 3), (1, 3), (1, 4), (3, 4)])
    assert_valid(netlist)


def test_node_ids_need_not_be_integers():
    netlist = netlist_from({'a': 'src', 'b': 'src', 'g1': 'AND', 'g2': 'OR'},
                           [('a', 'g1'), ('b', 'g1'), ('b', 'g2'), ('g1', 'g2')])
    assert_valid(netlist)


def test_layout_does_not_modify_the_callers_netlist():
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'AND', 3: 'OR'},
                           [(0, 2), (1, 2), (1, 3), (2, 3)])
    nodes, edges = sorted(netlist.nodes()), sorted(netlist.edges())
    build_layout(netlist)
    assert (sorted(netlist.nodes()), sorted(netlist.edges())) == (nodes, edges)


def test_more_than_one_sink_is_rejected_clearly():
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'AND', 3: 'OR'},
                           [(0, 2), (1, 2), (0, 3), (1, 3)])
    with pytest.raises(ValueError, match='exactly one sink'):
        build_layout(netlist)


def test_drawing_leaves_each_layer_in_bottom_to_top_order():
    # Drawing used to re-sort every layer top to bottom, in place, reversing the
    # order the layout phases (and anything run after them) rely on.
    netlist = netlist_from(
        {0: 'src', 1: 'src', 2: 'src', 3: 'src', 4: 'AND', 5: 'OR', 6: 'XOR'},
        [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)])
    state = assert_valid(netlist)
    for layer in state.layers:
        heights = [state.y[node] for node in layer]
        assert heights == sorted(heights)


def test_a_wire_running_straight_into_a_pin_past_a_branch_gets_a_dot():
    # Source 1 feeds both gates. Wherever its wire carries straight on into one
    # pin while its bus branches away to the other, that is a T and needs a dot;
    # check_layout's missing_dot / spurious_dot checks cover both directions.
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'src', 3: 'AND', 4: 'AND', 5: 'OR'},
                           [(0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)])
    assert_valid(netlist)


# ── the phases, individually ─────────────────────────────────────────────────

def test_layout_can_be_computed_without_drawing():
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'AND'}, [(0, 2), (1, 2)])
    state = compute_layout(netlist)
    assert state.wires and not state.diagram.elements
    assert state.y[2] == 0.0 and state.x(2) == 0.0      # the sink's output is the origin
    assert state.x(0) < 0                               # everything else is to its left


def test_highest_predecessor_takes_the_highest_pin():
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'src', 3: 'AND'}, [(0, 3), (1, 3), (2, 3)])
    state = compute_layout(netlist)
    by_height = sorted(state.sources, key=lambda node: -state.y[node])
    assert [state.pin_index[(node, 3)] for node in by_height] == [0, 1, 2]
    pin_heights = [state.pin_y(3, node) for node in by_height]
    assert pin_heights == sorted(pin_heights, reverse=True)


def test_count_crossings():
    netlist = nx.DiGraph([('a', 'x'), ('b', 'y')])
    assert ordering.count_crossings(netlist, ['a', 'b'], ['x', 'y']) == 0
    assert ordering.count_crossings(netlist, ['a', 'b'], ['y', 'x']) == 1
    # Connections that share an end meet at a point; they don't cross.
    fan = nx.DiGraph([('a', 'x'), ('a', 'y')])
    assert ordering.count_crossings(fan, ['a'], ['y', 'x']) == 0


def test_long_connections_are_broken_into_single_layer_hops():
    # Source 1 feeds the sink directly, two layers away.
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'NOT', 3: 'AND'}, [(0, 2), (2, 3), (1, 3)])
    state = compute_layout(netlist)
    for pred, consumer in state.netlist.edges():
        assert state.layer_of[pred] - state.layer_of[consumer] == 1
    assert sum(1 for node in state.netlist if state.is_dummy(node)) == 1
