"""
layout_validity.py

Check that a finished circuit drawing is *valid*: that it shows the circuit it
was asked to show, and shows it unambiguously.

`layout_metrics` scores how good a layout looks (fewer crossings, fewer bends).
This module asks the prior question -- is the drawing right at all? -- and
answers it from the drawn geometry alone: where the gates, pins, wires and
junction dots ended up. It deliberately knows nothing about how the layout was
computed (layers, channels, dummy nodes), so the checks keep meaning the same
thing while `circuit_layout` is restructured underneath them.

The checks (see `check_layout`):

    connectivity     Reading the drawing the way a person would -- wires that
                     meet end to end or at a T are connected, wires that merely
                     cross are not, unless there is a dot -- every gate input
                     is driven by exactly the gates/sources the netlist says.
    gate_shape       Each gate is drawn as the gate type the netlist names.
    gate_overlap     No two gate bodies overlap.
    label_overlap    No two source labels sit on top of each other.
    wire_through_gate  No wire passes through a gate body.
    wire_overlap     No two parallel wires run close enough to read as one.
    diagonal_wire    Every wire segment is horizontal or vertical.
    missing_dot      Every point where three or more wire arms meet has a dot.
    spurious_dot     Every dot sits where three or more wire arms meet.
    misaligned_join  Wires that are meant to meet do meet exactly, rather than
                     missing by a gap too small to see.

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

from dataclasses import dataclass, field

import schemdraw.elements as elm

from .circuit_layout import GATE_CLASSES

# Positions are in drawing units of order 1, so anything below this is
# floating-point noise rather than a real gap or offset.
EPS = 1e-6

# Wire ends nearer than this look joined: the gap is a fraction of the line
# width. Connectivity is judged at this tolerance, because it is about what a
# reader sees; ends that only meet this loosely are reported as a flaw of
# their own (misaligned_join) rather than as a broken circuit.
JOIN_TOLERANCE = 0.005

# Two parallel wires closer than this read as a single line at normal size.
WIRE_CLEARANCE = 0.1

# Two source labels in the same column closer than this overlap as text.
LABEL_CLEARANCE = 0.3

# Violations that mean the drawing is wrong or misleading, as opposed to ugly.
# A drawing with any of these does not correctly depict its circuit.
HARD_KINDS = ('error', 'connectivity', 'gate_shape', 'gate_overlap',
              'diagonal_wire', 'missing_dot')


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


@dataclass(frozen=True)
class Gate:
    node: int
    kind: str      # the netlist's gate type, e.g. 'NAND'
    box: tuple     # (xmin, ymin, xmax, ymax) of the drawn body, leads included
    out: tuple     # (x, y) of the output pin
    inputs: tuple  # ((x, y), ...) of the input pins, in1 first
    shape: tuple   # fingerprint of what was actually drawn (see _shape_of)


@dataclass
class Geometry:
    """What was drawn, in absolute drawing coordinates."""
    gates: dict = field(default_factory=dict)    # node -> Gate (real gates only)
    sources: dict = field(default_factory=dict)  # node -> (x, y, label)
    wires: list = field(default_factory=list)    # ((x1, y1), (x2, y2)), as drawn
    dots: list = field(default_factory=list)     # (x, y)


def _xy(point):
    return (float(point[0]), float(point[1]))


def _shape_of(element):
    """A fingerprint of a gate's outline: how many strokes, of which kinds, in
    how big a box. Enough to tell every gate type apart (an XOR has one more
    arc than an OR, a NAND one more circle than an AND, and so on) without
    depending on schemdraw's class names, several of which are shared."""
    box = element.get_bbox(includetext=False)
    # Text is a label hung on the gate (e.g. the sink's 'out'), not part of its outline.
    strokes = tuple(sorted(type(segment).__name__ for segment in element.segments
                           if type(segment).__name__ != 'SegmentText'))
    return (strokes, round(box.xmax - box.xmin, 4), round(box.ymax - box.ymin, 4))


def extract_geometry(state):
    """Read the drawn geometry out of a LayoutState returned by `build_layout`.

    Gates, wires and dots are taken from the schemdraw elements that were
    actually drawn, not from the layout's own record of where it meant to put
    them, so that the drawing step is checked along with everything before it.

    This is the only function here that looks inside LayoutState, so it is the
    only one that needs to change if LayoutState does.
    """
    geometry = Geometry()

    for node in state.sources:
        x, y = state.output_xy(node)
        geometry.sources[node] = (float(x), float(y),
                                  str(state.netlist.nodes[node].get('label', node)))

    for node, element in state.gate_elements.items():
        anchors = element.absanchors
        box = element.get_bbox(transform=True, includetext=False)
        inputs = []
        while f'in{len(inputs) + 1}' in anchors:
            inputs.append(_xy(anchors[f'in{len(inputs) + 1}']))
        geometry.gates[node] = Gate(
            node=node, kind=state.kind(node),
            box=(float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax)),
            out=_xy(anchors['out']), inputs=tuple(inputs), shape=_shape_of(element))

    for element in state.diagram.elements:
        if isinstance(element, elm.Dot):
            anchors = element.absanchors
            geometry.dots.append(_xy(anchors.get('center', anchors.get('start'))))
        elif isinstance(element, elm.Line):
            geometry.wires.append((_xy(element.absanchors['start']),
                                   _xy(element.absanchors['end'])))

    return geometry


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _is_horizontal(segment):
    return abs(segment[0][1] - segment[1][1]) <= EPS


def _is_vertical(segment):
    return abs(segment[0][0] - segment[1][0]) <= EPS


def _length(segment):
    return abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1])


def _on_segment(point, segment, tolerance=EPS):
    """True if `point` lies on `segment`, ends included (orthogonal segments only)."""
    (x1, y1), (x2, y2) = segment
    x, y = point
    return (min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
            and (_is_horizontal(segment) or _is_vertical(segment)))


def _same_point(a, b):
    return abs(a[0] - b[0]) <= EPS and abs(a[1] - b[1]) <= EPS


def _arms(point, wires):
    """The directions in which wire leaves `point`: a subset of 'LRUD'.

    A straight run through a point has two arms, a corner two, a T three and a
    four-way meeting four. A junction dot is needed exactly when there are
    three or more.
    """
    x, y = point
    arms = set()
    for segment in wires:
        if not _on_segment(point, segment):
            continue
        (x1, y1), (x2, y2) = segment
        if _is_horizontal(segment):
            if min(x1, x2) < x - EPS:
                arms.add('L')
            if max(x1, x2) > x + EPS:
                arms.add('R')
        else:
            if min(y1, y2) < y - EPS:
                arms.add('D')
            if max(y1, y2) > y + EPS:
                arms.add('U')
    return arms


class _Groups:
    """Union-find over integer ids."""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def wire_nets(geometry, tolerance=EPS):
    """Group the drawing's wires into nets, the way a reader would.

    Returns (net_of_wire, net_at): `net_of_wire[i]` is the net id of
    `geometry.wires[i]`, and `net_at(point)` is the net id of whichever wire
    touches `point`, or None if none does.

    Two wires are joined when an *end* of one lies on the other -- end to end,
    at a corner, or at a T -- or when they overlap along the same track. Wires
    that merely cross are not joined unless a dot sits on the crossing.
    `tolerance` is how near counts as touching.
    """
    wires = geometry.wires
    straight = [i for i, w in enumerate(wires) if _is_horizontal(w) or _is_vertical(w)]
    groups = _Groups(len(wires))

    for position, i in enumerate(straight):
        for j in straight[position + 1:]:
            if any(_on_segment(end, wires[j], tolerance) for end in wires[i]) or \
                    any(_on_segment(end, wires[i], tolerance) for end in wires[j]):
                groups.union(i, j)
    for dot in geometry.dots:
        touching = [i for i in straight if _on_segment(dot, wires[i], tolerance)]
        for other in touching[1:]:
            groups.union(touching[0], other)

    def net_at(point):
        for i in straight:
            if _on_segment(point, wires[i], tolerance):
                return groups.find(i)
        return None

    return [groups.find(i) for i in range(len(geometry.wires))], net_at


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def _check_connectivity(netlist, geometry):
    """The drawing connects exactly what the netlist connects, as a reader
    sees it. If it does, but only because gaps too small to see were let
    through, say that instead."""
    found = _connectivity_errors(netlist, geometry, JOIN_TOLERANCE)
    if found:
        return found
    return [Violation('misaligned_join', violation.detail + ' (gap too small to see)')
            for violation in _connectivity_errors(netlist, geometry, EPS)]


def _connectivity_errors(netlist, geometry, tolerance):
    found = []
    _, net_at = wire_nets(geometry, tolerance)

    drivers_of_net = {}
    outputs = {node: gate.out for node, gate in geometry.gates.items()}
    outputs.update({node: (x, y) for node, (x, y, _) in geometry.sources.items()})
    for node, point in outputs.items():
        net = net_at(point)
        if net is None:
            if netlist.out_degree(node) > 0:
                found.append(Violation('connectivity', f'output of node {node} has no wire'))
            continue
        drivers_of_net.setdefault(net, []).append(node)

    for net, drivers in drivers_of_net.items():
        if len(drivers) > 1:
            found.append(Violation('connectivity',
                                   f'outputs of nodes {sorted(drivers)} are wired together'))

    for node in netlist.nodes():
        if netlist.nodes[node]['gate'] == 'src':
            continue
        if node not in geometry.gates:
            found.append(Violation('connectivity', f'gate {node} was not drawn'))
            continue
        expected = sorted(netlist.predecessors(node))
        gate = geometry.gates[node]
        if len(gate.inputs) != len(expected):
            found.append(Violation(
                'connectivity',
                f'gate {node} drawn with {len(gate.inputs)} inputs, netlist has {len(expected)}'))
            continue
        drawn = []
        for number, pin in enumerate(gate.inputs, start=1):
            drivers = drivers_of_net.get(net_at(pin), [])
            if not drivers:
                found.append(Violation('connectivity',
                                       f'input {number} of gate {node} is not driven'))
            drawn.extend(drivers)
        if len(drawn) >= len(expected) and sorted(drawn) != expected:
            found.append(Violation(
                'connectivity',
                f'gate {node} is driven by {sorted(drawn)}, netlist says {expected}'))
    return found


_REFERENCE_SHAPES = {}


def _reference_shape(kind, fan_in):
    if (kind, fan_in) not in _REFERENCE_SHAPES:
        _REFERENCE_SHAPES[(kind, fan_in)] = _shape_of(GATE_CLASSES[kind](inputs=fan_in))
    return _REFERENCE_SHAPES[(kind, fan_in)]


def _check_gate_shapes(netlist, geometry):
    """Each gate is drawn as what the caller's netlist says it is -- not what
    the layout's own copy of the netlist says, which is the thing under test."""
    found = []
    for gate in geometry.gates.values():
        if gate.node not in netlist:
            continue
        kind = netlist.nodes[gate.node]['gate']
        if gate.shape != _reference_shape(kind, len(gate.inputs)):
            found.append(Violation('gate_shape', f'gate {gate.node} should be drawn as {kind}'))
    return found


def _check_gate_overlap(geometry):
    found = []
    gates = sorted(geometry.gates.values(), key=lambda g: g.node)
    for i, a in enumerate(gates):
        for b in gates[i + 1:]:
            if (min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]) > EPS
                    and min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]) > EPS):
                found.append(Violation('gate_overlap', f'gates {a.node} and {b.node} overlap'))
    return found


def _check_label_overlap(geometry, clearance):
    found = []
    sources = sorted(geometry.sources.items())
    for i, (a, (ax, ay, _)) in enumerate(sources):
        for b, (bx, by, _) in sources[i + 1:]:
            if abs(ax - bx) <= EPS and abs(ay - by) < clearance - EPS:
                found.append(Violation('label_overlap',
                                       f'sources {a} and {b} are {abs(ay - by):.3f} apart'))
    return found


def _check_wires_through_gates(geometry):
    """A wire may end on a gate's edge (that is what a pin is) but not enter it."""
    found = []
    for (x1, y1), (x2, y2) in geometry.wires:
        for gate in geometry.gates.values():
            xmin, ymin, xmax, ymax = gate.box
            if (min(max(x1, x2), xmax) - max(min(x1, x2), xmin) > -EPS
                    and min(max(y1, y2), ymax) - max(min(y1, y2), ymin) > -EPS
                    and max(x1, x2) > xmin + EPS and min(x1, x2) < xmax - EPS
                    and max(y1, y2) > ymin + EPS and min(y1, y2) < ymax - EPS):
                found.append(Violation(
                    'wire_through_gate',
                    f'wire ({x1:.3f}, {y1:.3f})-({x2:.3f}, {y2:.3f}) enters gate {gate.node}'))
    return found


def _check_wire_overlap(geometry, clearance):
    """Parallel wires on nearly, but not exactly, the same track.

    Exactly the same track isn't reported here: such wires read as one wire,
    which is either harmless (they are one net) or a connectivity error.
    """
    found = []
    for horizontal in (True, False):
        tracks = []
        for segment in geometry.wires:
            if _length(segment) <= EPS:
                continue
            if horizontal and _is_horizontal(segment):
                tracks.append((segment[0][1], *sorted((segment[0][0], segment[1][0]))))
            elif not horizontal and _is_vertical(segment) and not _is_horizontal(segment):
                tracks.append((segment[0][0], *sorted((segment[0][1], segment[1][1]))))
        for i, (track_a, low_a, high_a) in enumerate(tracks):
            for track_b, low_b, high_b in tracks[i + 1:]:
                if (EPS < abs(track_a - track_b) < clearance - EPS
                        and min(high_a, high_b) - max(low_a, low_b) > EPS):
                    axis = 'y' if horizontal else 'x'
                    found.append(Violation(
                        'wire_overlap',
                        f'wires at {axis}={track_a:.3f} and {axis}={track_b:.3f} '
                        f'run side by side from {max(low_a, low_b):.3f} to {min(high_a, high_b):.3f}'))
    return found


def _check_diagonals(geometry):
    return [Violation('diagonal_wire', f'wire {a}-{b} is neither horizontal nor vertical')
            for a, b in geometry.wires
            if not _is_horizontal((a, b)) and not _is_vertical((a, b))]


def _check_dots(geometry):
    found = []
    for dot in geometry.dots:
        arms = _arms(dot, geometry.wires)
        if len(arms) < 3:
            found.append(Violation(
                'spurious_dot',
                f'dot at ({dot[0]:.3f}, {dot[1]:.3f}) where only {len(arms)} wire arm(s) meet'))

    seen = []
    for segment in geometry.wires:
        for end in segment:
            if any(_same_point(end, other) for other in seen):
                continue
            seen.append(end)
            if len(_arms(end, geometry.wires)) >= 3 and \
                    not any(_same_point(end, dot) for dot in geometry.dots):
                found.append(Violation(
                    'missing_dot', f'wires branch at ({end[0]:.3f}, {end[1]:.3f}) with no dot'))
    return found


def check_geometry(netlist, geometry, wire_clearance=WIRE_CLEARANCE,
                   label_clearance=LABEL_CLEARANCE):
    """Return every Violation found in `geometry`, a drawing of `netlist`.

    `netlist` is the circuit as the caller supplied it, not the copy inside
    LayoutState, which has dummy nodes spliced into it.
    """
    return (_check_connectivity(netlist, geometry)
            + _check_gate_shapes(netlist, geometry)
            + _check_gate_overlap(geometry)
            + _check_label_overlap(geometry, label_clearance)
            + _check_wires_through_gates(geometry)
            + _check_wire_overlap(geometry, wire_clearance)
            + _check_diagonals(geometry)
            + _check_dots(geometry))


def check_layout(netlist, state):
    """Check the drawing in `state` (from `build_layout(netlist)`) against `netlist`."""
    return check_geometry(netlist, extract_geometry(state))


# --------------------------------------------------------------------------
# A canonical description of a drawing, for telling whether two are the same
# --------------------------------------------------------------------------

def _merged_ink(wires):
    """Wires as the ink they leave: maximal straight strokes, however many
    pieces each was drawn in and in whatever order. Two drawings with the same
    ink look identical, which is what "a refactor changed nothing" means."""
    strokes = []
    for horizontal in (True, False):
        by_track = {}
        for (x1, y1), (x2, y2) in wires:
            if abs(x1 - x2) + abs(y1 - y2) <= EPS:
                continue
            if horizontal and abs(y1 - y2) <= EPS:
                by_track.setdefault(round(y1, 6), []).append(sorted((x1, x2)))
            elif not horizontal and abs(x1 - x2) <= EPS and abs(y1 - y2) > EPS:
                by_track.setdefault(round(x1, 6), []).append(sorted((y1, y2)))
        for track, spans in by_track.items():
            spans.sort()
            low, high = spans[0]
            for next_low, next_high in spans[1:] + [[float('inf'), float('inf')]]:
                if next_low <= high + EPS:
                    high = max(high, next_high)
                    continue
                if horizontal:
                    strokes.append([round(low, 6), track, round(high, 6), track])
                else:
                    strokes.append([track, round(low, 6), track, round(high, 6)])
                low, high = next_low, next_high
    for a, b in wires:
        if not _is_horizontal((a, b)) and not _is_vertical((a, b)):
            strokes.append([round(v, 6) for v in (*a, *b)])
    return sorted(strokes)


def canonical_geometry(netlist, geometry):
    """A JSON-safe description of the drawing that is the same for any two
    drawings that look the same. Only the caller's nodes are named; dummy
    nodes are internal to the layout and appear only as the ink they leave."""
    return {
        'gates': {str(node): [gate.kind, len(gate.inputs),
                              round(gate.out[0], 6), round(gate.out[1], 6)]
                  for node, gate in sorted(geometry.gates.items()) if node in netlist},
        'sources': {str(node): [round(x, 6), round(y, 6), label]
                    for node, (x, y, label) in sorted(geometry.sources.items())},
        'ink': _merged_ink(geometry.wires),
        'dots': sorted([round(x, 6), round(y, 6)] for x, y in geometry.dots),
    }
