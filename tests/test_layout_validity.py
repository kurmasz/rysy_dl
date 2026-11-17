"""
Tests for layout_validity (the checks themselves) and layout_harness (the
snapshot/compare machinery built on them).

Most of these feed the checks a drawing with one deliberate fault and assert
that the fault is reported. A checker that never complains is indistinguishable
from one that works on a corpus where everything happens to pass, so each check
has to be shown failing before its silence means anything.
"""

import dataclasses

import networkx as nx
import pytest
import schemdraw

from rysy_dl import layout_harness
from rysy_dl.circuit_layout import GATE_CLASSES, build_layout
from rysy_dl.layout_validity import (HARD_KINDS, Gate, Geometry, _reference_shape,
                                     _shape_of, canonical_geometry, check_geometry,
                                     check_layout, extract_geometry)


@pytest.fixture(autouse=True)
def svg_backend():
    schemdraw.use('svg')


def kinds(violations):
    return sorted({violation.kind for violation in violations})


# ── a small hand-built drawing ───────────────────────────────────────────────
#
#   a (0,2) ──────────────┐
#                         ├─ AND(2) ──┐
#   b (0,1) ──────┬───────┘           ├─ OR(3)
#                 └───────────────────┘
#
# b fans out to both gates, so the drawing has one junction dot, at (1, 1).
# Node ids are contiguous from 0, as circuit_layout currently requires.

def small_netlist():
    netlist = nx.DiGraph()
    netlist.add_node(0, gate='src', label='a')
    netlist.add_node(1, gate='src', label='b')
    netlist.add_node(2, gate='AND')
    netlist.add_node(3, gate='OR')
    netlist.add_edges_from([(0, 2), (1, 2), (1, 3), (2, 3)])
    return netlist


def small_geometry():
    return Geometry(
        gates={
            2: Gate(node=2, kind='AND', box=(2, 0.8, 4, 2.2), out=(4, 1.5),
                    inputs=((2, 2), (2, 1)), shape=_reference_shape('AND', 2)),
            3: Gate(node=3, kind='OR', box=(6, -0.2, 8, 2.0), out=(8, 1.0),
                    inputs=((6, 1.5), (6, 0)), shape=_reference_shape('OR', 2)),
        },
        sources={0: (0, 2, 'a'), 1: (0, 1, 'b')},
        wires=[
            ((0, 2), (2, 2)),      # a -> AND in1
            ((0, 1), (2, 1)),      # b -> AND in2
            ((1, 1), (1, 0)),      # b's branch, down from the dot
            ((1, 0), (6, 0)),      # ... and across to OR in2
            ((4, 1.5), (6, 1.5)),  # AND -> OR in1
        ],
        dots=[(1, 1)],
    )


def test_hand_built_drawing_is_valid():
    assert check_geometry(small_netlist(), small_geometry()) == []


def test_missing_wire_is_a_connectivity_error():
    geometry = small_geometry()
    del geometry.wires[0]
    assert 'connectivity' in kinds(check_geometry(small_netlist(), geometry))


def test_wire_to_wrong_pin_is_a_connectivity_error():
    # The drawing is fine; it is a drawing of a different circuit.
    netlist = small_netlist()
    netlist.remove_edge(0, 2)
    netlist.add_edge(0, 3)
    netlist.remove_edge(1, 3)
    assert 'connectivity' in kinds(check_geometry(netlist, small_geometry()))


def test_crossing_wires_are_not_connected():
    # Reroute a's wire so that it crosses b's branch without touching an end.
    geometry = small_geometry()
    geometry.wires[0] = ((0, 2), (0.5, 2))
    geometry.wires += [((0.5, 2), (0.5, 0.5)), ((0.5, 0.5), (1.5, 0.5)),
                       ((1.5, 0.5), (1.5, 2)), ((1.5, 2), (2, 2))]
    assert check_geometry(small_netlist(), geometry) == []


def test_dot_on_a_crossing_shorts_the_two_wires():
    geometry = small_geometry()
    geometry.wires[0] = ((0, 2), (0.5, 2))
    geometry.wires += [((0.5, 2), (0.5, 0.5)), ((0.5, 0.5), (1.5, 0.5)),
                       ((1.5, 0.5), (1.5, 2)), ((1.5, 2), (2, 2))]
    geometry.dots.append((1, 0.5))
    assert 'connectivity' in kinds(check_geometry(small_netlist(), geometry))


def test_wires_sharing_a_track_are_shorted():
    geometry = small_geometry()
    geometry.wires.append(((0.5, 2), (0.5, 1)))  # ties a's wire to b's
    assert 'connectivity' in kinds(check_geometry(small_netlist(), geometry))


def test_branch_without_a_dot_is_reported():
    geometry = small_geometry()
    geometry.dots.clear()
    assert kinds(check_geometry(small_netlist(), geometry)) == ['missing_dot']


def test_dot_on_a_corner_is_reported():
    geometry = small_geometry()
    geometry.dots.append((1, 0))
    assert kinds(check_geometry(small_netlist(), geometry)) == ['spurious_dot']


def test_overlapping_gates_are_reported():
    geometry = small_geometry()
    geometry.gates[3] = dataclasses.replace(geometry.gates[3], box=(3, -0.2, 8, 2.0))
    assert 'gate_overlap' in kinds(check_geometry(small_netlist(), geometry))


def test_wire_through_a_gate_is_reported():
    geometry = small_geometry()
    geometry.wires.append(((3, 3), (3, 2.5)))   # stops short of the gate: fine
    assert 'wire_through_gate' not in kinds(check_geometry(small_netlist(), geometry))
    geometry.wires.append(((3, 3), (3, 1.2)))   # runs into it
    assert 'wire_through_gate' in kinds(check_geometry(small_netlist(), geometry))


def test_wire_ending_on_a_pin_does_not_enter_the_gate():
    assert 'wire_through_gate' not in kinds(check_geometry(small_netlist(), small_geometry()))


def test_wires_running_side_by_side_are_reported():
    geometry = small_geometry()
    geometry.wires[0] = ((0, 2), (0.3, 2))
    geometry.wires += [((0.3, 2), (0.3, 1.05)), ((0.3, 1.05), (0.7, 1.05)),
                       ((0.7, 1.05), (0.7, 2)), ((0.7, 2), (2, 2))]
    assert kinds(check_geometry(small_netlist(), geometry)) == ['wire_overlap']


def test_diagonal_wire_is_reported():
    geometry = small_geometry()
    geometry.wires.append(((4, 3), (5, 3.5)))
    assert 'diagonal_wire' in kinds(check_geometry(small_netlist(), geometry))


def test_close_source_labels_are_reported():
    geometry = small_geometry()
    geometry.sources[0] = (0, 1.1, 'a')
    assert 'label_overlap' in kinds(check_geometry(small_netlist(), geometry))


def test_gate_drawn_as_the_wrong_type_is_reported():
    geometry = small_geometry()
    geometry.gates[2] = dataclasses.replace(geometry.gates[2], shape=_reference_shape('NAND', 2))
    assert kinds(check_geometry(small_netlist(), geometry)) == ['gate_shape']


def test_invisible_gap_is_a_flaw_not_a_broken_circuit():
    geometry = small_geometry()
    geometry.wires[0] = ((0, 2.0003), (2, 2.0003))
    assert kinds(check_geometry(small_netlist(), geometry)) == ['misaligned_join']
    geometry.wires[0] = ((0, 2.05), (2, 2.05))
    assert 'connectivity' in kinds(check_geometry(small_netlist(), geometry))


@pytest.mark.parametrize('fan_in', [2, 3, 4])
def test_every_gate_type_has_a_distinct_shape(fan_in):
    # The gate_shape check is only as good as this.
    shapes = [_shape_of(GATE_CLASSES[kind](inputs=fan_in))
              for kind in ('AND', 'OR', 'XOR', 'NAND', 'NOR')]
    assert len(set(shapes)) == len(shapes)


# ── real layouts ─────────────────────────────────────────────────────────────

def test_real_layout_matches_its_netlist():
    netlist = small_netlist()
    state = build_layout(netlist)
    assert check_layout(netlist, state) == []
    geometry = extract_geometry(state)
    assert sorted(geometry.gates) == [2, 3]
    assert sorted(geometry.sources) == [0, 1]
    assert geometry.gates[2].kind == 'AND' and len(geometry.gates[2].inputs) == 2


def test_real_layout_fails_against_a_different_netlist():
    netlist = small_netlist()
    state = build_layout(netlist)
    other = small_netlist()
    other.nodes[2]['gate'] = 'NOR'
    assert 'gate_shape' in kinds(check_layout(other, state))


def test_packaged_circuits_lay_out_validly():
    """A small slice of the harness corpus, run on every test run: no drawing
    may crash or be invalid. (Flaws such as wire_overlap are tracked by the
    harness, not asserted here, because the current layout has some.)"""
    hard = []
    count = 0
    for key, netlist in layout_harness.corpus(limit=25, variants=1, examples_dir=None):
        record, _ = layout_harness.examine(netlist)
        count += 1
        hard += [(key, kind, detail) for kind, detail in record['violations'] if kind in HARD_KINDS]
    assert count == 50
    assert hard == []


# ── snapshot / compare ───────────────────────────────────────────────────────

def test_canonical_geometry_ignores_how_a_wire_was_pieced_together():
    netlist = small_netlist()
    whole = small_geometry()
    pieces = small_geometry()
    pieces.wires[3:4] = [((6, 0), (3, 0)), ((3, 0), (1, 0))]
    pieces.wires.reverse()
    assert canonical_geometry(netlist, whole) == canonical_geometry(netlist, pieces)


def test_canonical_geometry_notices_a_moved_wire():
    netlist = small_netlist()
    moved = small_geometry()
    moved.wires[2] = ((1, 1), (1, -0.1))
    moved.wires[3] = ((1, -0.1), (6, -0.1))
    assert canonical_geometry(netlist, small_geometry()) != canonical_geometry(netlist, moved)


def _snapshot(geometry, violations=()):
    netlist = small_netlist()
    return {'circuits': {'c': {
        'expression': '',
        'geometry': canonical_geometry(netlist, geometry),
        'violations': [list(v) for v in violations],
        'metrics': None,
    }}}


def test_compare_treats_float_noise_as_identical():
    noisy = small_geometry()
    noisy.wires[0] = ((0, 2 + 1e-9), (2, 2 + 1e-9))
    result = layout_harness.compare(_snapshot(small_geometry()), _snapshot(noisy))
    assert result['changed'] == []


def test_compare_reports_a_changed_drawing_and_a_new_violation():
    moved = small_geometry()
    moved.dots.clear()
    result = layout_harness.compare(
        _snapshot(small_geometry()),
        _snapshot(moved, violations=[('missing_dot', 'wires branch at (1, 1) with no dot')]))
    assert result['changed'] == ['c']
    assert [v[:2] for v in result['new_hard_violations']] == [('c', 'missing_dot')]


def test_compare_does_not_call_an_existing_violation_new():
    flaw = [('wire_overlap', 'x')]
    result = layout_harness.compare(_snapshot(small_geometry(), flaw),
                                    _snapshot(small_geometry(), flaw))
    assert result['new_violations'] == []
