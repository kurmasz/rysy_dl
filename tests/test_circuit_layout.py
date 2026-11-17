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
from rysy_dl.layout import ordering, placement, refinement, routing
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


# ── wires sharing a track ────────────────────────────────────────────────────
#
# One net's run (from its output to its channel) lies along another net's stub
# (from its channel to a pin) when the output is level with the pin and the
# first net's channel is the further right. See routing.must_be_left_of.

def nets(**heights):
    """{net: (output height, [pin heights])} from net=(out, pin, pin, ...)."""
    return {net: (values[0], list(values[1:])) for net, values in heights.items()}


def test_must_be_left_of_pairs_an_output_with_a_level_pin_of_another_net():
    gap = nets(a=(1.0, 3.0), b=(2.0, 1.02), c=(5.0, 5.0))
    # a's output (1.0) is level with b's pin (1.02). c is level only with its own pin.
    assert routing.must_be_left_of(gap, clearance=0.1) == {('a', 'b')}


def test_channel_order_keeps_a_run_off_another_nets_stub():
    gap = nets(a=(1.0, 3.0), b=(2.0, 1.02))
    for start in (['a', 'b'], ['b', 'a']):
        (shared_tracks, _), order = routing.best_channel_order(gap, start, clearance=0.1)
        assert shared_tracks == 0
        assert order.index('a') > order.index('b')      # a's channel is the further left


def test_channel_order_prefers_a_crossing_to_a_shared_track():
    # Here the only order with no shared track has one more crossing than the
    # best of the rest, and is still the right choice.
    gap = nets(a=(0.0, 6.0, 0.0), b=(3.0, 0.0), c=(2.0, 6.0, 3.0))
    left_of = routing.must_be_left_of(gap, 0.1)
    assert routing._gap_cost(gap, ['a', 'c', 'b'], left_of) == (1, 2)
    assert routing._gap_cost(gap, ['c', 'b', 'a'], left_of) == (0, 3)
    for start in (['a', 'c', 'b'], ['a', 'b', 'c'], ['b', 'a', 'c']):
        assert routing.best_channel_order(gap, start, clearance=0.1) == ((0, 3), ['c', 'b', 'a'])


def test_contradictory_requirements_leave_one_shared_track_for_routing():
    # Each net's output is level with the other's pin: no order satisfies both.
    gap = nets(a=(1.0, 2.0), b=(2.0, 1.0))
    (shared_tracks, _), _ = routing.best_channel_order(gap, ['a', 'b'], clearance=0.1)
    assert shared_tracks == 1


def test_a_long_chain_of_requirements_is_honoured_from_a_bad_start():
    # a left of b left of c left of d (each output is level with the next
    # net's pin), starting from exactly the wrong order. Channel 0, the first
    # in the list, is the furthest right.
    gap = nets(a=(0.0, 9.0), b=(1.0, 0.0), c=(2.0, 1.0), d=(3.0, 2.0))
    assert routing.must_be_left_of(gap, 0.1) == {('a', 'b'), ('b', 'c'), ('c', 'd')}
    (shared_tracks, _), order = routing.best_channel_order(gap, ['a', 'b', 'c', 'd'], clearance=0.1)
    assert shared_tracks == 0 and order == ['d', 'c', 'b', 'a']


def test_gate_output_level_with_another_nets_pin_is_not_drawn_as_one_wire():
    # From the corpus: a NOT gate's output came out 0.002 below the pin that an
    # unrelated wire was heading for, and its channel was the further right, so
    # the two were drawn as a single line -- a drawing of a different circuit.
    netlist = netlist_from(
        {0: 'src', 1: 'src', 2: 'src', 3: 'NOR', 4: 'OR', 5: 'AND', 6: 'NAND', 7: 'XOR',
         8: 'NOT', 9: 'NOT'},
        [(0, 3), (0, 5), (0, 8), (1, 4), (1, 5), (1, 6), (2, 9), (3, 5), (4, 7), (5, 7),
         (6, 3), (6, 4), (8, 6), (9, 6)])
    assert_valid(netlist)


def test_nets_level_with_each_others_pins_are_moved_apart():
    # From the corpus: two tightly packed nets, each level with a pin of the
    # other, so that no channel order parts them and one has to be moved.
    netlist = netlist_from(
        {0: 'XOR', 1: 'NOR', 2: 'OR', 3: 'NAND', 4: 'src', 5: 'src', 6: 'AND', 7: 'AND',
         8: 'NOT', 9: 'NOT'},
        [(0, 3), (0, 6), (1, 7), (2, 7), (3, 7), (4, 2), (4, 8), (5, 0), (5, 1), (5, 9),
         (6, 1), (6, 2), (8, 0), (8, 1), (8, 3), (8, 6), (9, 2), (9, 3), (9, 6)])
    state = assert_valid(netlist)
    for layer in state.layers[1:]:
        assert routing.gap_cost(state, layer)[0] == 0
        heights = [state.y[node] for node in layer]
        assert heights == sorted(heights)                # still in order
        for below, above in zip(layer, layer[1:]):       # and no closer than allowed
            assert state.y[above] - state.y[below] >= placement.min_gap(state, below, above) - 1e-9


# ── layer spacing ────────────────────────────────────────────────────────────

def test_each_gap_is_only_as_wide_as_its_own_wires_need():
    # Four sources feed two gates, which feed the sink: four nets cross the
    # gap on the left, two the gap on the right.
    netlist = netlist_from(
        {0: 'src', 1: 'src', 2: 'src', 3: 'src', 4: 'AND', 5: 'AND', 6: 'AND'},
        [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)])
    state = assert_valid(netlist)
    spacing = state.spacing
    gate, margins = 1.85, 2 * spacing.pin_to_vertical_spacing
    assert state.layer_x[0] == 0.0
    assert state.layer_x[0] - state.layer_x[1] == pytest.approx(gate + margins + 1 * spacing.wire_spacing)
    assert state.layer_x[1] - state.layer_x[2] == pytest.approx(gate + margins + 3 * spacing.wire_spacing)


def test_every_channel_clears_the_pins_and_the_outputs_either_side_of_it():
    netlist = netlist_from(
        {0: 'src', 1: 'src', 2: 'src', 3: 'NOR', 4: 'OR', 5: 'AND', 6: 'NAND', 7: 'XOR',
         8: 'NOT', 9: 'NOT'},
        [(0, 3), (0, 5), (0, 8), (1, 4), (1, 5), (1, 6), (2, 9), (3, 5), (4, 7), (5, 7),
         (6, 3), (6, 4), (8, 6), (9, 6)])
    state = assert_valid(netlist)
    margin = state.spacing.pin_to_vertical_spacing
    buses = [wire for wire in state.wires if wire.start[0] == wire.end[0]]
    assert buses
    for wire in buses:
        x = wire.start[0]
        k = min(k for k in range(1, len(state.layers)) if state.layer_x[k] < x)     # the gap it is in
        pins_x = state.layer_x[k - 1] - 1.85
        outputs_x = max(state.output_xy(node)[0] for node in state.layers[k])
        assert x <= pins_x - margin + 1e-9
        assert x >= outputs_x + margin - 0.05 - 1e-9    # an OR's output overhangs by 0.05


# ── placement methods ────────────────────────────────────────────────────────

EXAMPLE = netlist_from(
    {0: 'src', 1: 'src', 2: 'src', 3: 'NOR', 4: 'OR', 5: 'AND', 6: 'NAND', 7: 'XOR',
     8: 'NOT', 9: 'NOT'},
    [(0, 3), (0, 5), (0, 8), (1, 4), (1, 5), (1, 6), (2, 9), (3, 5), (4, 7), (5, 7),
     (6, 3), (6, 4), (8, 6), (9, 6)])


@pytest.mark.parametrize('method', ['layered', 'global', 'best'])
def test_every_placement_method_gives_a_valid_ordered_layout(method):
    state = build_layout(EXAMPLE, placement_method=method)
    assert check_layout(EXAMPLE, state) == []
    for layer in state.layers:
        for below, above in zip(layer, layer[1:]):
            assert state.y[above] - state.y[below] >= placement.min_gap(state, below, above) - 1e-9
    assert state.y[state.sink] == 0.0


def test_unknown_placement_method_is_rejected():
    with pytest.raises(ValueError, match='Unknown placement method'):
        compute_layout(EXAMPLE, placement_method='sideways')


def test_best_placement_is_no_worse_than_either_method():
    best = refinement.quality(compute_layout(EXAMPLE, placement_method='best'))
    for method in ('layered', 'global'):
        assert best <= refinement.quality(compute_layout(EXAMPLE, placement_method=method))


def test_global_placement_lines_sources_up_with_the_pins_they_feed():
    # Three sources into one gate: nothing stops every wire being straight.
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'src', 3: 'AND'}, [(0, 3), (1, 3), (2, 3)])
    state = compute_layout(netlist, placement_method='global')
    for source in state.sources:
        assert state.y[source] == pytest.approx(state.pin_y(3, source), abs=1e-9)


def test_global_placement_keeps_a_long_wire_straight():
    # Source 1 reaches the sink across two layers of gates, through dummies.
    netlist = netlist_from({0: 'src', 1: 'src', 2: 'NOT', 3: 'NOT', 4: 'AND'},
                           [(0, 2), (2, 3), (3, 4), (1, 4)])
    state = compute_layout(netlist, placement_method='global')
    (path,) = [p for p in refinement.wire_paths(state.netlist) if p[0] == 1]
    assert len(path) == 4                                # source, two dummies, the sink
    heights = refinement.path_heights(state, path)
    assert max(heights) - min(heights) < 1e-9
