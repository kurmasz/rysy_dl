"""
circuit_layout.py

Lay out a combinatorial netlist into a Schemdraw Drawing object.

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

import networkx as nx
import io
import logging
from dataclasses import dataclass, field

import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic
from .custom_gates import FixedNot, Dummy

logger = logging.getLogger(__name__)


def assign_source_labels(graph, name_list=None):
    if name_list is None:
        name_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    source_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    source_nodes.sort()
    for i, node in enumerate(source_nodes):
        graph.nodes[node]['label'] = name_list[i]

GATE_CLASSES = {
    'OR': logic.Or,
    'XOR': logic.Xor,
    'AND': logic.And,
    'NAND': logic.Nand,
    'NOR': logic.Nor,
    'NOT': FixedNot,
    'DUMMY': Dummy
}

@dataclass(frozen=True)
class DebugColors:
    pin: str = 'red'
    shared: str = 'blue'

DEBUG : bool = False  # Flip to True to draw pin/shared-wire markers in DebugColors instead of black.
@dataclass(frozen=True)
class Colors:
    pin: str = DebugColors.pin if DEBUG else 'black'
    shared: str = DebugColors.shared if DEBUG else 'black'
    direct: str = 'green' if DEBUG else 'black'
    dummy: str = 'purple' if DEBUG else 'black'

@dataclass(frozen=True)
class SpacingConfig:
    """Fixed drawing-style constants that control wire and gate spacing.
    Frozen, unlike LayoutState: these aren't computed or mutated as layout
    proceeds, they're just parameters of the drawing style."""

    wire_spacing: float = 0.25  # The horizontal distance between wires in a bundle, used to stagger the bend points and vertical channels for each input pin.
    
    # QQQ pin_to_vertical spacing is not a constant. Not all gates have the same width.
    pin_to_vertical_spacing: float = 0.4 # The horizontal distance between a pin and the closest vertical channel.
    y_margin: float = 0.3       # The fixed vertical gap between adjacent nodes in the same layer, regardless of their input-pin count.

    # Sources and dummy waypoints aren't gate bodies -- a source is just a
    # label, and a dummy is a bend in a wire -- so they don't need a gate's
    # clearance. Spacing them by y_margin puts them 0.45 apart, but a gate's
    # input pins sit only 0.33 (3-input) to 0.40 apart, so a row of sources
    # feeding one gate could never line up with its pins and every wire took a
    # small step on the way in. 0.18 gives a 0.33 pitch, matching the closest
    # pin spacing any gate has; going lower buys nothing, since no gate's pins
    # are nearer than that.
    label_y_margin: float = 0.18

    # Two wires running along the same horizontal track closer than this read
    # as a single line, which is worse than a crossing: a crossing is at least
    # unambiguous, whereas two wires sharing a track hide one of them.
    wire_clearance: float = 0.1
    

@dataclass
class LayoutState:
    """Bundles the netlist graph together with everything computed about its
    layout, so the layout/drawing functions below can take a single argument
    instead of threading the same handful of values through every call.
    Fields are filled in progressively as layout proceeds; nothing here is
    frozen, since later phases build on values computed by earlier ones.

    The netlist is cloned on construction (see __post_init__) since layout
    (e.g. _add_dummy_nodes) mutates it, and callers shouldn't see their graph
    changed as a side effect of computing a layout."""

    netlist: nx.DiGraph
    spacing: SpacingConfig = field(default_factory=SpacingConfig)

    def __post_init__(self):
        self.netlist = self.netlist.copy()

    # Set by _compute_node_layers
    nodes_at_layer: list = field(default_factory=list)
    layer_for_node: dict = field(default_factory=dict)
    source_nodes: list = field(default_factory=list)

    # Set by _compute_spacing
    node_num_of_sink: object = None
    x_spacing: float = 0.0
    gate_width: float = 0.0
    gate_height: float = 0.0
    gate_width_by_type: dict = field(default_factory=dict)
    gate_height_by_node: dict = field(default_factory=dict)

    # Set by compute_layout
    position_y: dict = field(default_factory=dict)
    position: dict = field(default_factory=dict)
    # objects: dict = field(default_factory=dict)  # placeholder gates used only to read pin anchor geometry

    # Set while drawing
    diagram: schemdraw.Drawing = field(default_factory=schemdraw.Drawing)
    gate_elements: dict = field(default_factory=dict)
    consumers: dict = field(default_factory=dict)
    channel_of: dict = field(default_factory=dict)


def _compute_node_layers(state):
    """Assign each node a layer via longest path to a sink, then force all
    source nodes (in-degree 0) to the deepest layer so inputs line up flush left.
    Populates state.nodes_at_layer, state.layer_for_node, and node_num_of_sink."""
    assert state
    
    netlist = state.netlist
    reverse_order = reversed(list(nx.topological_sort(netlist)))

    layer_for_node = {}
    for node in reverse_order:
        successors = list(netlist.successors(node))
        layer = 0 if not successors else max(layer_for_node[s] for s in successors) + 1
        layer_for_node[node] = layer

    max_layer = max(layer_for_node.values())

    # Force every source (input) to sit at the same (deepest) layer, all the way to the left
    state.source_nodes = [n for n in netlist.nodes() if netlist.in_degree(n) == 0]
    if not state.source_nodes:
        raise ValueError("Circuit has no source nodes (nodes with in-degree 0)")
    max_source_layer = max(layer_for_node[n] for n in state.source_nodes)
    for n in state.source_nodes:
        layer_for_node[n] = max_source_layer

    nodes_at_layer = [[] for _ in range(max_layer + 1)]
    for node, layer in layer_for_node.items():
        nodes_at_layer[layer].append(node)

    state.node_num_of_sink = [n for n in netlist.nodes() if netlist.out_degree(n) == 0][0]
    state.layer_for_node = layer_for_node
    state.nodes_at_layer = nodes_at_layer


def _add_dummy_nodes(state):
    """Insert a dummy routing node into each intermediate layer for any edge that
    spans more than one layer, so long edges get a dedicated lane through each
    intermediate column instead of cutting straight across real gates that live there.
    Mutates state.netlist, state.nodes_at_layer, and state.layer_for_node in place."""
    assert state
    
    netlist = state.netlist
    nodes_at_layer = state.nodes_at_layer
    layer_for_node = state.layer_for_node
    
    # Make sure _compute_node_layers has been called.
    assert nodes_at_layer and layer_for_node, "_compute_node_layers must run before _add_dummy_nodes"

    for layer, nodes_at_current_layer in enumerate(nodes_at_layer):
        dummies_needed = {}  # Map from predecessor to all nodes in this level that consume this pred.
        for node in nodes_at_current_layer:
            for pred in netlist.predecessors(node):
                gap = layer_for_node[pred] - layer
                if gap > 1:
                    dummies_needed.setdefault(pred, []).append(node)
        for pred, successors in dummies_needed.items():
            new_node = len(netlist)
            netlist.add_node(new_node)
            netlist.nodes[new_node]['gate'] = 'DUMMY'
            netlist.add_edge(pred, new_node)
            for successor in successors:
                netlist.add_edge(new_node, successor)
                netlist.remove_edge(pred, successor)
            nodes_at_layer[layer + 1].append(new_node)
            layer_for_node[new_node] = layer + 1
    
    state.dummies_added = True


def _compute_spacing(state):
    """Derive gate dimensions and inter-layer spacing from a reference 2-input AND
    gate. 
    
    Populates state.x_spacing, state.gate_width, and state.gate_height.
    
    Mutates the netlist
    """
    assert state
    assert state.nodes_at_layer, "Must call _compute_node_layers first"
    assert state.dummies_added, "Must call _add_dummy_nodes first"

    bbox = logic.And(inputs=6).get_bbox()

    state.gate_width = bbox.xmax - bbox.xmin   # 1.85; used as the reference width for column spacing
    state.gate_height = bbox.ymax - bbox.ymin  # 2.4; the tallest gate, kept only as a reference

    # Height is not one number: a 2- or 3-input gate is 1.0 tall, a 6-input one
    # 2.4. Spacing every gate by the largest of those drives small gates far
    # apart for no reason and leaves wires floating well above them, so record
    # what each node will actually be drawn as (see _node_height).
    height_of = {}
    for node in state.netlist.nodes():
        gate_type = state.netlist.nodes[node]['gate']
        if gate_type in ('src', 'DUMMY'):
            continue
        shape = (gate_type, state.netlist.in_degree(node))
        if shape not in height_of:
            node_bbox = GATE_CLASSES[gate_type](inputs=shape[1]).get_bbox()
            height_of[shape] = node_bbox.ymax - node_bbox.ymin
        state.gate_height_by_node[node] = height_of[shape]

    # Not every gate type is actually state.gate_width wide (e.g. Or/Xor/Nor's body
    # is 0.05 wider than And/Nand's). Since gates are drawn anchored on their output
    # pin, a wider body pushes its input pins further left than a narrower one at the
    # same column x. Record each type's true width so _draw_gates can compensate and
    # keep input pins aligned across gate types within a layer.
    state.gate_width_by_type = {
        name: cls(inputs=2).get_bbox().xmax - cls(inputs=2).get_bbox().xmin
        for name, cls in GATE_CLASSES.items()
    }

    # Because we have already added the dummy nodes, we know exactly how many vertical channels we'll need.
    max_vertical_channels = max(map(lambda l: len(l), state.nodes_at_layer))

    # The horizontal distance between layers (output pin to output pin).
    state.x_spacing = state.spacing.wire_spacing * (max_vertical_channels - 1) + (state.spacing.pin_to_vertical_spacing * 2) + state.gate_width


def _crossings_between_layers(netlist, upper_order, lower_order):
    """Count crossings between two adjacent layers with fixed orders.

    `upper_order` is the layer further from the sink (the predecessors);
    `lower_order` is the layer nearer the sink (their successors). Both run
    bottom-to-top. Two wire segments cross exactly when their endpoints appear
    in opposite order in the two layers. Segments sharing an endpoint meet at
    a point rather than crossing, so they aren't counted.
    """
    lower_index = {n: i for i, n in enumerate(lower_order)}
    segments = sorted(
        (i, lower_index[v])
        for i, u in enumerate(upper_order)
        for v in netlist.successors(u)
        if v in lower_index)

    crossings = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i][0] < segments[j][0] and segments[i][1] > segments[j][1]:
                crossings += 1
    return crossings


def _total_crossings(netlist, order_by_layer):
    """Crossings summed over every adjacent pair of layers."""
    return sum(_crossings_between_layers(netlist, order_by_layer[k + 1], order_by_layer[k])
               for k in range(len(order_by_layer) - 1))


def _median_order(layer_nodes, neighbours_of, neighbour_index):
    """Reorder one layer by the median index of each node's neighbours in the
    adjacent, already-ordered layer.

    The median rather than the mean is the standard choice: it isn't dragged
    off position by a single distant neighbour, which matters here because a
    source can fan out to gates at wildly different heights.
    """
    def key(position_and_node):
        position, node = position_and_node
        indices = sorted(neighbour_index[m] for m in neighbours_of(node) if m in neighbour_index)
        # Nothing to align to: keep the current position rather than
        # collapsing to one end of the layer.
        return indices[len(indices) // 2] if indices else position

    return [node for _, node in sorted(enumerate(layer_nodes), key=key)]


def _transpose_adjacent(netlist, order_by_layer):
    """Greedily swap adjacent pairs within each layer, keeping any swap that
    reduces crossings. Median ordering gets close but leaves locally fixable
    crossings behind; this cleans those up. Mutates order_by_layer in place."""
    def local_crossings(k):
        total = 0
        if k > 0:
            total += _crossings_between_layers(netlist, order_by_layer[k], order_by_layer[k - 1])
        if k + 1 < len(order_by_layer):
            total += _crossings_between_layers(netlist, order_by_layer[k + 1], order_by_layer[k])
        return total

    improved = True
    while improved:
        improved = False
        for k, layer in enumerate(order_by_layer):
            for i in range(len(layer) - 1):
                before = local_crossings(k)
                layer[i], layer[i + 1] = layer[i + 1], layer[i]
                if local_crossings(k) < before:
                    improved = True
                else:
                    layer[i], layer[i + 1] = layer[i + 1], layer[i]


def _order_nodes_in_layers(state, iterations=8):
    """Choose the order of nodes within each layer so wires cross as little as
    possible, storing it back into state.nodes_at_layer (index 0 = bottom).

    This is the ordering phase of a layered ("Sugiyama") layout, and it is
    deliberately separate from -- and prior to -- position assignment.
    Ordering is a combinatorial problem (minimise crossings) while positioning
    is a geometric one (minimise bends); solving them together optimises
    neither. _assign_initial_gate_positions treats the order chosen here as
    fixed and only picks coordinates within it.

    Alternating median sweeps propagate a consistent order outward from the
    sink and back again, and the transposition pass cleans up what the sweeps
    miss. Every candidate is scored by actual crossing count and only the best
    one encountered is kept, so raising `iterations` can never return a worse
    ordering than a lower value would have.
    """
    netlist = state.netlist
    order = [list(layer) for layer in state.nodes_at_layer]

    best = [list(layer) for layer in order]
    best_crossings = _total_crossings(netlist, order)

    for iteration in range(iterations):
        if iteration % 2 == 0:
            # Sweep away from the sink: order each layer by its successors.
            for k in range(1, len(order)):
                index_below = {n: i for i, n in enumerate(order[k - 1])}
                order[k] = _median_order(order[k], netlist.successors, index_below)
        else:
            # Sweep back toward the sink: order each layer by its predecessors.
            for k in range(len(order) - 2, -1, -1):
                index_above = {n: i for i, n in enumerate(order[k + 1])}
                order[k] = _median_order(order[k], netlist.predecessors, index_above)

        _transpose_adjacent(netlist, order)

        crossings = _total_crossings(netlist, order)
        if crossings < best_crossings:
            best_crossings = crossings
            best = [list(layer) for layer in order]
        if best_crossings == 0:
            break

    state.nodes_at_layer = best


def _node_height(state, node):
    """How much vertical room a node takes up.

    A source is a label and a dummy is a bend in a wire, so both take almost
    none. A gate's height depends on its fan-in -- 1.0 for two or three inputs,
    rising to 2.4 at six -- so spacing every gate as though it were the tallest
    gate in the library leaves a conspicuous empty band above the small ones,
    which most gates are.
    """
    node_type = state.netlist.nodes[node]['gate']
    if node_type in ('DUMMY', 'src'):
        return 0.15
    return state.gate_height_by_node[node]


def _min_gap(state, u, v):
    """Smallest vertical distance allowed between two nodes in the same layer.

    Two label/waypoint nodes may sit closer than two gate bodies, so that wires
    running into a gate can line up with its input pins (see label_y_margin).
    """
    if all(state.netlist.nodes[n]['gate'] in ('src', 'DUMMY') for n in (u, v)):
        margin = state.spacing.label_y_margin
    else:
        margin = state.spacing.y_margin
    return _node_height(state, u) / 2 + _node_height(state, v) / 2 + margin


def _successor_priority(netlist, nodes):
    """Priority for _resolve_layer_positions: -1 * a node's out-degree.

    A node with a single successor (a dummy waypoint, or any node feeding just
    one gate) has an exact, unambiguous target -- moving it creates a bend
    that wasn't already implied. A node with several successors already has a
    target that's just an average of conflicting pulls (see _pin_barycenter),
    so it can absorb a nudge more cheaply. Fewer successors therefore means
    higher (less negative) priority: those nodes get to keep their desired
    position, and multi-successor nodes yield around them.
    """
    return {n: -len(list(netlist.successors(n))) for n in nodes}


def _resolve_layer_positions(state, nodes_sorted, desired, priority):
    """Given nodes sorted ascending by desired y, return final y positions that
    honor the minimum gate spacing while staying as close as possible to each
    node's desired position -- preferring to disturb lower-priority nodes over
    higher-priority ones when the two conflict (see _successor_priority).

    Nodes are placed one at a time, highest priority first, each starting at
    its own desired position. Placing a node pushes already-placed neighbors
    of *lower* priority out of the way (cascading further if that push causes
    a further collision); an already-placed neighbor of equal-or-higher
    priority is treated as a wall instead -- the new node (and anything it
    would have pushed) recoils just enough to clear it.

    A final feasibility sweep guarantees no two nodes end up closer than the
    minimum gap even in the rare case where a node is pinned between two
    immovable walls with too little room between them -- a situation that
    hasn't come up in the circuits used to develop this, but the sweep keeps
    it from ever producing overlapping gate bodies if it does.
    """
    n = len(nodes_sorted)

    def gap(i, j):
        return _min_gap(state, nodes_sorted[i], nodes_sorted[j])

    pos = [None] * n
    order = sorted(range(n), key=lambda i: (-priority[nodes_sorted[i]], i))

    for i in order:
        pos[i] = desired[nodes_sorted[i]]

        # Push already-placed, lower-priority neighbors above out of the way;
        # stop at (and recoil from) the first equal-or-higher-priority wall.
        wall = None
        j = i + 1
        while j < n and pos[j] is not None and pos[j] < pos[j - 1] + gap(j - 1, j):
            if priority[nodes_sorted[j]] >= priority[nodes_sorted[i]]:
                wall = j
                break
            pos[j] = pos[j - 1] + gap(j - 1, j)
            j += 1
        if wall is not None:
            limit = pos[wall] - gap(wall - 1, wall)
            for k in range(wall - 1, i - 1, -1):
                pos[k] = min(pos[k], limit)
                if k > i:
                    limit = pos[k] - gap(k - 1, k)

        # Symmetric handling below.
        wall = None
        j = i - 1
        while j >= 0 and pos[j] is not None and pos[j] > pos[j + 1] - gap(j, j + 1):
            if priority[nodes_sorted[j]] >= priority[nodes_sorted[i]]:
                wall = j
                break
            pos[j] = pos[j + 1] - gap(j, j + 1)
            j -= 1
        if wall is not None:
            limit = pos[wall] + gap(wall, wall + 1)
            for k in range(wall + 1, i + 1):
                pos[k] = max(pos[k], limit)
                if k < i:
                    limit = pos[k] + gap(k, k + 1)

    for i in range(1, n):
        min_y = pos[i - 1] + gap(i - 1, i)
        if pos[i] < min_y:
            pos[i] = min_y

    return pos


def _anchor_barycenter(netlist, position_y, n):
    """A node's successors' anchor (output-pin) y, averaged. Cheap but coarse:
    it doesn't know which specific input pin of a multi-input successor this
    edge lands on, only the successor's reference point. Good enough as a
    provisional order to feed _pin_barycenter (see below)."""
    successors = list(netlist.successors(n))
    return sum(position_y[s] for s in successors) / len(successors) if successors else 0.0


def _pin_barycenter(state, layer, provisional_position_y):
    """For every node in `layer`, return {node: desired_y}, where desired_y is
    the average -- over all of the node's successors -- of the y-coordinate of
    the SPECIFIC input pin that edge connects to, not just the successor's
    anchor point. A multi-input gate's pins sit at different y-offsets from
    its anchor (closer together than a full gate height), so aiming for the
    anchor point pulls every predecessor toward nearly the same target and
    leaves a short unavoidable jog at the pins themselves.

    Because _add_dummy_nodes guarantees every edge spans exactly one layer,
    a successor's full predecessor set is exactly the set of `layer` nodes
    feeding it, so it can be reconstructed here from the graph alone.
    Pin assignment mirrors _route_stubs (highest y first, into in1, in2, ...),
    using `provisional_position_y` -- e.g. from _anchor_barycenter, already
    resolved for this layer -- to stand in for the not-yet-final y order;
    only the relative order matters for which predecessor lands on which pin,
    so an approximate provisional order is enough.
    """
    netlist = state.netlist
    predecessors_of = {}
    for n in state.nodes_at_layer[layer]:
        for s in netlist.successors(n):
            predecessors_of.setdefault(s, []).append(n)

    pin_targets = {n: [] for n in state.nodes_at_layer[layer]}
    for s, preds in predecessors_of.items():
        preds_sorted = sorted(preds, key=lambda p: provisional_position_y[p], reverse=True)
        gate = state.gate_elements[s]
        for i, p in enumerate(preds_sorted):
            pin_targets[p].append(gate.anchors[f'in{i + 1}'][1] + provisional_position_y[s])

    return {n: (sum(ys) / len(ys) if ys else 0.0) for n, ys in pin_targets.items()}


def _assign_initial_gate_positions(state):
    """Assigns every node an initial (x, y) position.

    Origin (0,0) is the location of the output pin of the sink gate.
    The entire netlist is drawn to the left of this point.

    Each layer is placed in two rounds:
      1. A coarse pass using _anchor_barycenter, resolved with
         _resolve_layer_positions, just to get a reasonable provisional
         y-order for the layer.
      2. That provisional order feeds _pin_barycenter to compute exact
         per-pin targets, which are resolved again (final _resolve_layer_positions
         call) into this layer's real positions.
    _resolve_layer_positions also reconciles targets against the minimum gate
    spacing (see its docstring), which replaces the old post-hoc _nudge_y step.
    """
    assert state
    assert state.node_num_of_sink, "Must call _compute_node_layers first"
    assert state.dummies_added

    netlist = state.netlist
    position_y = {state.node_num_of_sink: 0.0}
    state.position[state.node_num_of_sink] = (0.0, 0.0)
    state.gate_elements[state.node_num_of_sink] = GATE_CLASSES['AND'](
        anchor='out', xy=(0.0, 0.0),
        inputs=len(list(netlist.predecessors(state.node_num_of_sink))), theta=0)

    for layer, nodes_at_current_layer in enumerate(state.nodes_at_layer):
        if layer == 0:
            continue  # sink already placed above

        priority = _successor_priority(netlist, nodes_at_current_layer)

        # The order within this layer was fixed by _order_nodes_in_layers to
        # minimise crossings; positioning only chooses coordinates within it.
        nodes_sorted = list(nodes_at_current_layer)

        provisional_desired = {n: _anchor_barycenter(netlist, position_y, n) for n in nodes_at_current_layer}
        provisional_y = _resolve_layer_positions(state, nodes_sorted, provisional_desired, priority)
        provisional_position_y = dict(position_y)
        provisional_position_y.update(zip(nodes_sorted, provisional_y))

        desired = _pin_barycenter(state, layer, provisional_position_y)
        final_y = _resolve_layer_positions(state, nodes_sorted, desired, priority)

        for n, y_pos in zip(nodes_sorted, final_y):
            position_y[n] = y_pos
            state.position[n] = (-state.layer_for_node[n] * state.x_spacing, y_pos)

            predecessors = list(netlist.predecessors(n))
            num_inputs = len(predecessors)
            state.gate_elements[n] = GATE_CLASSES['AND'](anchor='out', xy=state.position[n], inputs=num_inputs, theta=0)


def _wire_paths(netlist):
    """Every real-node-to-real-node wire, as the list of nodes it runs through.

    A wire leaves a gate or a source, passes through zero or more dummy
    waypoints, and lands on a consumer. `_add_dummy_nodes` lets a waypoint
    branch to several consumers, so one predecessor can begin several paths.
    """
    def is_dummy(node):
        return netlist.nodes[node]['gate'] == 'DUMMY'

    paths = []
    for node in netlist.nodes():
        if is_dummy(node):
            continue
        pending = [[node, successor] for successor in netlist.successors(node)]
        while pending:
            path = pending.pop()
            if is_dummy(path[-1]):
                pending.extend(path + [s] for s in netlist.successors(path[-1]))
            else:
                paths.append(path)
    return paths


def _pin_height(state, consumer, pred):
    """Height of the input pin of `consumer` that `pred`'s wire lands on.

    Mirrors the pin assignment `_route_stubs` performs -- predecessors ordered
    by height, the highest into in1. Every gate type puts its input pins at the
    same heights for a given fan-in, so the placeholder gate held in
    gate_elements reports the same pin geometry the drawn gate will.
    """
    predecessors = sorted(state.netlist.predecessors(consumer),
                          key=lambda p: state.position[p], reverse=True)
    gate = state.gate_elements[consumer]
    return gate.anchors[f'in{predecessors.index(pred) + 1}'][1] + state.position[consumer][1]


def _pin_heights_by_source(state):
    """Which input-pin heights each node's wire runs to."""
    netlist = state.netlist
    pins_of = {}
    for consumer in netlist.nodes():
        if netlist.in_degree(consumer) == 0:
            continue
        for pred in netlist.predecessors(consumer):
            pins_of.setdefault(pred, []).append(_pin_height(state, consumer, pred))
    return pins_of


def _foreign_tracks(state, node, pins_of):
    """Heights other nets' wires occupy in the gaps on either side of `node`.

    A node's height shows up as a wire track twice over: to its right, where
    its own output runs across to the channel, and to its left, where whatever
    feeds it arrives. So both gaps have to be checked, and each holds two kinds
    of track -- the output runs of the nodes in the layer above, and the stubs
    reaching the pins of the layer below. Looking only at one layer misses
    half of them, and the wire that ends up sharing a track usually belongs to
    a node in the neighbouring column.

    A node's own predecessors are left out: their wire and this node's are the
    same net, and same-net segments meet end to end rather than running side
    by side.
    """
    layer_index = state.layer_for_node[node]
    tracks = []

    # Stubs into the layer below, carrying other nodes' wires.
    for other in state.nodes_at_layer[layer_index]:
        if other != node:
            tracks.extend(pins_of.get(other, []))

    # Output runs of the layer above, crossing the gap on this node's left.
    if layer_index + 1 < len(state.nodes_at_layer):
        predecessors = set(state.netlist.predecessors(node))
        for other in state.nodes_at_layer[layer_index + 1]:
            if other not in predecessors:
                tracks.append(state.position[other][1])

    return tracks


def _wire_jogs(state, path):
    """Direction changes along one wire, measured pin to pin."""
    heights = [state.position[node][1] for node in path[:-1]]
    heights.append(_pin_height(state, path[-1], path[-2]))
    return sum(1 for i in range(len(heights) - 1)
               if abs(heights[i] - heights[i + 1]) > 1e-9)


def _travel_limits(state, node):
    """How far `node` may slide without passing its neighbours in its layer.

    Bounding moves this way keeps the crossing-minimal order chosen by
    `_order_nodes_in_layers` intact, so a later adjustment can't undo it.
    """
    layer = state.nodes_at_layer[state.layer_for_node[node]]
    index = layer.index(node)
    low, high = float('-inf'), float('inf')
    if index > 0:
        below = layer[index - 1]
        low = state.position[below][1] + _min_gap(state, below, node)
    if index < len(layer) - 1:
        above = layer[index + 1]
        high = state.position[above][1] - _min_gap(state, above, node)
    return low, high


def _separate_wire_tracks(state):
    """Nudge a node aside when its wire would run along the same horizontal
    track as a different net's.

    Placement keeps *nodes* apart, but says nothing about the tracks their
    wires occupy. A node can therefore land a hair away from an input pin that
    some other node's wire runs to, and the two wires travel the gap between
    layers side by side close enough to read as one line -- worse than a
    crossing, which at least stays legible.

    Only sources and waypoints are moved. Both are effectively weightless -- a
    label and a bend in a wire -- whereas shifting a gate would drag its own
    input pins along with it and just relocate the problem. Moves stay within
    the layer's order (see `_travel_limits`), and a node that cannot be cleared
    is left where it is rather than forced somewhere worse.
    """
    clearance = state.spacing.wire_clearance
    netlist = state.netlist
    pins_of = _pin_heights_by_source(state)

    for layer_nodes in state.nodes_at_layer:
        for node in layer_nodes:
            if netlist.nodes[node]['gate'] not in ('src', 'DUMMY'):
                continue
            foreign = _foreign_tracks(state, node, pins_of)
            current = state.position[node][1]
            if all(abs(current - y) >= clearance for y in foreign):
                continue

            low, high = _travel_limits(state, node)
            candidates = [current]
            for y in foreign:
                candidates.extend((y + clearance, y - clearance))

            best = None
            for candidate in candidates:
                if not low - 1e-9 <= candidate <= high + 1e-9:
                    continue
                if any(abs(candidate - y) < clearance - 1e-9 for y in foreign):
                    continue
                if best is None or abs(candidate - current) < abs(best - current):
                    best = candidate

            if best is not None:
                state.position[node] = (state.position[node][0], best)


def _align_fanout_nodes(state):
    """Sit a node that feeds several gates level with one of the pins it
    reaches, instead of at the average of them.

    `_pin_barycenter` aims a node at the mean of the pins its wire lands on.
    For a node feeding a single gate that is exactly right, but for one feeding
    several it lands between them, level with none, and two things follow.
    Every one of its wires then has to step, where sitting level with a pin
    would have left that wire straight. And its connection to the shared bus
    falls in the middle of the bus rather than at an end, so a junction dot is
    drawn there -- frequently right where some other wire crosses, leaving a
    reader no way to tell a connection from a crossing.

    Levelling with the highest or lowest pin addresses both at once: one wire
    runs straight, and the bus connection becomes an end of the bus, where
    `_connect_consumers` draws no dot at all.
    """
    netlist = state.netlist
    paths = _wire_paths(netlist)
    pins_of = _pin_heights_by_source(state)
    clearance = state.spacing.wire_clearance

    def total_jogs():
        return sum(_wire_jogs(state, path) for path in paths)

    for node in netlist.nodes():
        if netlist.out_degree(node) < 2:
            continue
        # Levelling with a pin is no good if it drops the node onto a track
        # another net's wire already runs along (see _separate_wire_tracks).
        foreign = _foreign_tracks(state, node, pins_of)
        pins = [_pin_height(state, consumer, node) for consumer in netlist.successors(node)]
        if len(set(round(pin, 9) for pin in pins)) < 2:
            continue

        current = state.position[node][1]
        low, high = _travel_limits(state, node)
        best = total_jogs()
        chosen = None
        # The extremes are what put the bus connection at an end, so no dot.
        for candidate in sorted({max(pins), min(pins)}, key=lambda y: abs(y - current)):
            if not low - 1e-9 <= candidate <= high + 1e-9:
                continue
            if any(abs(candidate - y) < clearance - 1e-9 for y in foreign):
                continue
            state.position[node] = (state.position[node][0], candidate)
            score = total_jogs()
            if score < best:
                best, chosen = score, candidate
            state.position[node] = (state.position[node][0], current)

        if chosen is not None:
            state.position[node] = (state.position[node][0], chosen)


def _straighten_wires(state, rounds=3):
    """Remove jogs a wire doesn't need, by sliding its dummy waypoints.

    Positions are assigned one layer at a time, so a layer knows nothing about
    what the layers either side of it needed. A wire threading several layers
    picks up an independent little step at each one and arrives as a staircase,
    even where a single step would have done. Two shapes turn up constantly: a
    pair of waypoints differing by a hair, and a waypoint parked outside the
    range its own wire spans, so the wire climbs and comes straight back down.

    Both are easy to remove once the layout exists, which is why this is a
    cleanup rather than something placement tries to foresee: here the whole
    drawing is known, so "is this move legal?" is a lookup instead of a
    prediction. Waypoints are also the ideal thing to move, since a dummy is
    invisible -- a bend in a wire and nothing else -- so sliding one relocates
    no gate, disturbs no label, and can only shrink the drawing.

    For each wire this tries putting the single step at each point along its
    chain, keeping the first arrangement that lowers the total. Moves are
    bounded by the neighbours in the layer's fixed order, so the
    crossing-minimal ordering from `_order_nodes_in_layers` still holds, and
    anything that fails to improve the total is undone -- so the pass can lower
    the jog count or leave it alone, never raise it.
    """
    paths = [path for path in _wire_paths(state.netlist) if len(path) > 2]
    if not paths:
        return

    def total_jogs():
        return sum(_wire_jogs(state, path) for path in paths)

    best = total_jogs()
    for _ in range(rounds):
        improved = False
        for path in paths:
            waypoints = path[1:-1]
            start = state.position[path[0]][1]
            end = _pin_height(state, path[-1], path[-2])
            saved = [state.position[w] for w in waypoints]

            # Put the step at each point along the chain in turn: the first
            # `split` waypoints stay level with the start, the rest with the end.
            for split in range(len(waypoints) + 1):
                plan = [start] * split + [end] * (len(waypoints) - split)
                if not all(low - 1e-9 <= y <= high + 1e-9
                           for (low, high), y in
                           ((_travel_limits(state, w), y) for w, y in zip(waypoints, plan))):
                    continue

                for waypoint, y in zip(waypoints, plan):
                    state.position[waypoint] = (state.position[waypoint][0], y)

                if total_jogs() < best:
                    best = total_jogs()
                    improved = True
                    break
                for waypoint, xy in zip(waypoints, saved):
                    state.position[waypoint] = xy

        if not improved:
            break


def _draw_gates(state):
    """Phase 1: draw every non-source gate at its precomputed position. Source
    nodes are skipped here and drawn as labels in _label_sources instead.
    QQQ May want to defer the drawing of the gates, in case nudging the gates up or down makes for a cleaner drawing."""

    netlist = state.netlist
    for node in netlist.nodes():
        if netlist.nodes[node]['gate'] == 'src':
            continue

        gate_type = netlist.nodes[node]['gate']
        predecessors = list(netlist.predecessors(node))
        num_inputs = len(predecessors)
        x, y = state.position[node]
        gate_class = GATE_CLASSES[gate_type]

        # Shift by this type's width delta from the reference width so that,
        # regardless of body shape, the drawn input pins land on the same x as
        # every other gate type in this layer (see _compute_spacing).
        x += state.gate_width_by_type[gate_type] - state.gate_width

        new_item = gate_class(anchor='out', xy=(x, y), inputs=num_inputs, theta=0)
        if DEBUG:
            new_item.label(f'{node}', loc='top')
            
        if node == state.node_num_of_sink:
            new_item.label('$out$', loc='right')
        gate = state.diagram.add(new_item)

        state.gate_elements[node] = gate


def _label_sources(state):
    """Phase 2: label every source at its precomputed position."""
    graph = state.netlist
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            x, y = state.position[node]
            label = graph.nodes[node].get('label', str(node))
            new_item = elm.Label().at((x, y)).label(f'${label}$', loc='left', ofst=0.15)
            if DEBUG:
                new_item.label(f'{node}', loc='top')
            state.diagram.add(new_item)


def _output_height(state, node):
    """The height at which a wire leaves `node`.

    Drawn gates report their own output-pin anchor rather than the nominal
    column position: _draw_gates shifts some gate types horizontally to keep
    their input pins aligned, which moves the output pin too. Sources are
    drawn as a label rather than a gate, so their position is used directly.
    """
    if state.netlist.in_degree(node) > 0:
        return float(state.gate_elements[node].absanchors['out'][1])
    return float(state.position[node][1])


def _consumer_pin_heights(state):
    """Map each predecessor to the heights of the input pins it feeds.

    Mirrors the pin assignment _route_stubs performs -- predecessors sorted by
    height, the highest into in1 -- so the two agree on where each wire lands.
    """
    graph = state.netlist
    heights = {}
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            continue
        gate = state.gate_elements[node]
        predecessors = sorted(graph.predecessors(node),
                              key=lambda p: state.position[p], reverse=True)
        for i, pred in enumerate(predecessors):
            heights.setdefault(pred, []).append(float(getattr(gate, f'in{i + 1}')[1]))
    return heights


def _gap_crossings(nets, order):
    """Count crossings within one inter-layer gap for a given channel order.

    `order[k]` is the predecessor routed in channel k. Channel 0 sits nearest
    the consumer gates and higher indices step left toward the predecessors,
    so each wire's horizontal runs cross exactly the channels lying between
    where it starts and where it turns: the run from a predecessor's output
    crosses channels to its left (higher index), while each stub out to a
    consumer pin crosses channels to its right (lower index). A horizontal
    that merely meets the end of a vertical is a T-junction -- a connection,
    not a crossing -- so the span test is strict at both ends.
    """
    channel_of = {pred: k for k, pred in enumerate(order)}
    spans = {}
    for pred in order:
        out_y, pin_ys = nets[pred]
        spans[pred] = (min([out_y] + pin_ys), max([out_y] + pin_ys))

    crossings = 0
    for pred in order:
        channel = channel_of[pred]
        out_y, pin_ys = nets[pred]
        for other in order:
            if other == pred:
                continue
            low, high = spans[other]
            if channel_of[other] > channel:
                if low < out_y < high:
                    crossings += 1
            else:
                crossings += sum(1 for y in pin_ys if low < y < high)
    return crossings


def _minimise_gap_crossings(nets, order):
    """Reorder one gap's channels by repeatedly swapping adjacent pairs and
    keeping any swap that reduces crossings.

    Swapping two channels moves no gate and changes no height -- it only
    changes how far each wire runs before turning -- so unlike reordering the
    nodes themselves, this can reduce crossings without enlarging the drawing.
    """
    order = list(order)
    best = _gap_crossings(nets, order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            score = _gap_crossings(nets, order)
            if score < best:
                best, improved = score, True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]
    return order


def _assign_vertical_channels(state):
    """Choose which vertical channel carries each predecessor's wire.

    Every predecessor needs one channel -- a single x at which its wire makes
    its vertical run -- shared by all of that predecessor's consumers, since
    _connect_consumers draws them as a single bus. Ranking those channels by
    height, as the drawing did originally, is arbitrary with respect to
    crossings; this picks an order that reduces them instead.

    Must run after _draw_gates, since it reads the drawn gates' pin anchors.
    """
    graph = state.netlist
    pin_heights = _consumer_pin_heights(state)

    channel_of = {}
    for layer in range(1, len(state.nodes_at_layer)):
        predecessors = [n for n in state.nodes_at_layer[layer] if graph.out_degree(n) > 0]
        if not predecessors:
            continue
        nets = {p: (_output_height(state, p), pin_heights.get(p, [])) for p in predecessors}
        # Start from the height order the drawing used before, then improve it.
        order = sorted(predecessors, key=lambda p: state.position[p][1], reverse=True)
        for k, pred in enumerate(_minimise_gap_crossings(nets, order)):
            channel_of[pred] = k

    state.channel_of = channel_of


def _route_stubs(state):
    """Phase 3a: draw the horizontal wire from an input pin to the vertical wire.
    Phase 3b: draw the wires through any dummy waypoints used for edges that skip layers.

    This process also collects final arrival points grouped by predecessor for the
    shared junction, stored in state.consumers for _connect_consumers to draw."""
    graph = state.netlist
    nodes_at_layer = state.nodes_at_layer
    layer_for_node = state.layer_for_node
    gate_position = state.position

    # Sort the nodes at each layer by their y-position, so that we can route wires in a consistent order.
    for nodes in nodes_at_layer:
        nodes.sort(key=lambda n: gate_position[n][1], reverse=True)

    consumers = {}  # The list of stubs that will need to be connected.
    for node in graph.nodes():

        # Skip sources, they have no input pins.
        if graph.in_degree(node) == 0:
            continue

        gate = state.gate_elements[node]

        # Sort the predecessors by the y-position of the node (or dummy waypoint) that they are connected to.  This should minimize wire crossings.
        predecessors = list(graph.predecessors(node))
        predecessors.sort(key=lambda p: gate_position[p], reverse=True)

        layer = layer_for_node[node]
        num_inputs = len(nodes_at_layer[layer + 1])  # The number of vertical channels in the next layer, which is the number of nodes at that layer.
        for i, pred in enumerate(predecessors):
            # the dummy node, if there is one for this connection
            immediate_pred = pred

            # The vertical channels are where wires move vertically to connect pins. Each output from the preceding layer needs a dedicated channel.
            # Note that we need one channel per source, not one channel per input pin in the current layer (because a single source can feed multiple inputs).
            vertical_channel_index = state.channel_of.get(immediate_pred, 0)

            # We could give each vertical channel its own fixed x-position, however, when there are only a few (2 or 3) sources,
            # the vertical channels may not overlap. In this case the diagram looks cleaner if we "reuse" these channels.
            # The heuristic here is to let the first and last channels share an x coordinate, then the second and second-to-last, etc.
            # This is done by computing the distance from the edge of the channel list, and using that to compute the x-position.
            distance_from_edge = vertical_channel_index  # min(vertical_channel_index, num_inputs - 1 - vertical_channel_index)
            if distance_from_edge < 0:
                raise ValueError(f"distance_from_edge is negative for pred {pred} and node {node}.  vertical_channel_index={vertical_channel_index}, num_inputs={num_inputs}")

            # QQQ pin_to_vertical spacing is not a constant. Not all gates have the same width.
            stub_length = state.spacing.pin_to_vertical_spacing + distance_from_edge * state.spacing.wire_spacing

            pin = getattr(gate, f'in{i+1}')
            stub = state.diagram.add(elm.Line().left().at(pin).length(stub_length)).color(Colors.pin)

            # Hop through each dummy waypoint, in order from the consumer toward the predecessor.
            # Draw each end but the last (the last hop to the predecessor)
            current = stub.end

            # At this point, current is either the stub that was just drawn, or the last hop to the predecessor.
            # Rather than drawing this line, we store it for later, because a single predecessor may feed multiple consumers,
            # and we want to draw a single shared bus line to all of them.
            consumers.setdefault(pred, []).append(current)

    state.consumers = consumers


def _connect_consumers(state):
    """Phase 4: connect each predecessor's single fixed position to all of its consumer
    endpoints. For a predecessor with multiple consumers, draw one shared bus line out
    to the farthest branch point, with a dot placed only where a wire actually merges
    into the bus (not at the bus's own starting/farthest end, since that's not a real fork)."""
    diagram = state.diagram
    gate_position = state.position

    for pred, pins in state.consumers.items():
        # The starting point. For drawn gates, this must be the gate's actual
        # drawn output-pin location, not the nominal column position: _draw_gates
        # shifts some gate types off that nominal x to keep their input pins
        # aligned with other types in the same layer, which moves the output pin
        # by the same amount. Sources are drawn as a label at their exact
        # position rather than a gate element, so use state.position for those
        # (state.gate_elements holds an un-drawn placeholder for every node,
        # including sources, so its presence alone isn't a reliable check).
        if state.netlist.in_degree(pred) > 0:
            px, py = state.gate_elements[pred].absanchors['out']
        else:
            px, py = gate_position[pred]

        # If this predecessor feeds only one gate at the current level, just draw a single line to it.
        if len(pins) == 1:
            pin = pins[0]
            if abs(pin[1] - py) > 0.001:
                if pin[1] > py:
                    v = diagram.add(elm.Line().down().at(pin).toy(py)).color(Colors.direct)
                else:
                    v = diagram.add(elm.Line().up().at(pin).toy(py)).color(Colors.direct)
                start = v.end
            else:
                start = pin
            diagram.add(elm.Line().left().at(start).tox(px)).color(Colors.direct)
            continue

        #
        # Handle the case where the predecessor feeds multiple consumers at the current level.
        #
        vertical_x = pins[0][0]  # All pins should have the same x coordinate.

        # The wire to the gate must also be connected to the vertical wire.
        pins.append((vertical_x, py))

        # Need vertical wire from highest to lowest connecting point
        top = max(pins, key=lambda p: p[1])
        bottom = min(pins, key=lambda p: p[1])
        diagram.add(elm.Line().down().at(top).toy(bottom[1])).color(Colors.shared)

        # Now add junction dots for the other predecessors attaching to the same vertical line.
        for pin in pins:
            if pin is top or pin is bottom:
                continue
            diagram.add(elm.Dot().at(pin)).color(Colors.shared)

        # Finally, add the line from the pin to vertical line (and a connecting dot)
        diagram.add(elm.Line().at((px, py)).tox(vertical_x)).color(Colors.shared)


def compute_layout(netlist, spacing=None):
    """Run the geometry phases only -- layering, dummy insertion, spacing, and
    position assignment -- and return the populated LayoutState without drawing
    anything.

    Split out of `layout_circuit` so the geometry can be inspected on its own,
    e.g. by `layout_metrics`, which scores a layout's crossings and bends
    without paying for (or depending on) the schemdraw rendering.

    `spacing` overrides the default SpacingConfig, which is mainly useful for
    comparing one set of drawing constants against another.
    """
    state = LayoutState(netlist=netlist) if spacing is None else LayoutState(netlist=netlist, spacing=spacing)

    _compute_node_layers(state)
    logger.debug('layer_for_node: %s', state.layer_for_node)
    logger.debug('nodes_at_layer: %s', state.nodes_at_layer)

    _add_dummy_nodes(state)
    _order_nodes_in_layers(state)
    _compute_spacing(state)
    _assign_initial_gate_positions(state)
    _align_fanout_nodes(state)
    _straighten_wires(state)
    _separate_wire_tracks(state)

    return state


def build_layout(netlist, spacing=None):
    """Compute the geometry and draw it, returning the whole LayoutState.

    `layout_circuit` returns only the finished Drawing, which is all a caller
    normally wants. `layout_metrics` needs the state behind it as well -- the
    node positions, and the drawn wire segments hanging off state.diagram --
    in order to score the result, so the two are separated here.
    """
    state = compute_layout(netlist, spacing)

    _draw_gates(state)
    _label_sources(state)
    _assign_vertical_channels(state)
    _route_stubs(state)
    _connect_consumers(state)

    return state


def layout_circuit(netlist):
    """Generates a Schemdraw Drawing object for the given netlist.

        Notes:
        * Origin (0,0) is the location of the output pin of the sink gate. The entire netlist is drawn to the left of this point.
          Thus, most x-coordinates will be negative.
    """
    return build_layout(netlist).diagram


def layout_circuit_png(netlist):
    schemdraw.use('matplotlib')
    circuit_diagram = layout_circuit(netlist)
    return io.BytesIO(circuit_diagram.get_imagedata(fmt='png'))
