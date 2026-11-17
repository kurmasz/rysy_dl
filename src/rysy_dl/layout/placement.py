"""
Placement: give every node a height, keeping each layer in the order that
`ordering` chose.

Layers are placed one at a time, starting from the sink and working upstream.
Each node wants to sit level with the input pins its wire runs to; where those
wishes would put two nodes too close together, `_spread` settles it.
"""

from .gates import reference_width


def gap_width(state, layer):
    """The distance from `layer` to the layer downstream of it: that layer's
    gates, then room for one vertical channel per net leaving `layer`, with a
    margin either side of the channels.

    Each gap is as wide as its own nets need and no wider. A gap carrying two
    wires is not made to match one carrying eight.
    """
    spacing = state.spacing
    channels = len(state.layers[layer])
    return (reference_width()
            + 2 * spacing.pin_to_vertical_spacing
            + spacing.wire_spacing * (channels - 1))


def set_layer_positions(state):
    """Set state.layer_x: the sink's layer at x = 0, and each layer upstream
    one gap further left."""
    state.layer_x = [0.0]
    for layer in range(1, len(state.layers)):
        state.layer_x.append(state.layer_x[-1] - gap_width(state, layer))


def node_height(state, node):
    """How much vertical room a node takes up. A gate's height depends on its
    fan-in (1.0 for two or three inputs, rising to 2.4 for six)."""
    if state.is_weightless(node):
        return state.spacing.weightless_height
    return state.shape(node).height


def min_gap(state, a, b):
    """The least vertical distance allowed between two neighbours in a layer."""
    if state.is_weightless(a) and state.is_weightless(b):
        margin = state.spacing.label_y_margin
    else:
        margin = state.spacing.y_margin
    return node_height(state, a) / 2 + node_height(state, b) / 2 + margin


def _desired_height(state, node):
    """Level with the input pins `node`'s wire runs to -- their average, if
    there are several. Aiming at the pins themselves, not at the consuming
    gates' centres, is what lets a wire arrive without a last small step."""
    pins = [state.pin_y(consumer, node) for consumer in state.netlist.successors(node)]
    return sum(pins) / len(pins) if pins else 0.0


def _priority(state, node):
    """Who gets their desired height when two neighbours can't both: the node
    with fewer consumers.

    A node with one consumer has an exact target, and moving it off that
    creates a bend that wasn't otherwise needed. A node with several is aiming
    at an average of conflicting pulls already, so a nudge costs it less.
    """
    return -state.netlist.out_degree(node)


def _spread(state, layer, desired, priority):
    """Heights for `layer` (bottom to top) that keep every pair of neighbours
    at least `min_gap` apart while staying near each node's desired height,
    disturbing lower-priority nodes in preference to higher-priority ones.

    Nodes are placed one at a time, highest priority first, each at its own
    desired height. Placing a node pushes already-placed neighbours of *lower*
    priority out of its way, and they push theirs in turn. An already-placed
    neighbour of equal or higher priority doesn't move: it is a wall, and the
    new node, along with anything it pushed, backs off just far enough to
    clear it.

    A node caught between two walls with too little room can still be left
    overlapping, so a final sweep moves nodes up wherever a gap is too small.
    """
    count = len(layer)
    heights = [None] * count
    rank = [priority[node] for node in layer]

    def gap(i, j):
        return min_gap(state, layer[i], layer[j])

    for i in sorted(range(count), key=lambda i: (-rank[i], i)):
        heights[i] = desired[layer[i]]

        # Upward: push lower-priority neighbours up; back off from a wall.
        wall = None
        j = i + 1
        while j < count and heights[j] is not None and heights[j] < heights[j - 1] + gap(j - 1, j):
            if rank[j] >= rank[i]:
                wall = j
                break
            heights[j] = heights[j - 1] + gap(j - 1, j)
            j += 1
        if wall is not None:
            limit = heights[wall] - gap(wall - 1, wall)
            for k in range(wall - 1, i - 1, -1):
                heights[k] = min(heights[k], limit)
                if k > i:
                    limit = heights[k] - gap(k - 1, k)

        # Downward: the same, mirrored.
        wall = None
        j = i - 1
        while j >= 0 and heights[j] is not None and heights[j] > heights[j + 1] - gap(j, j + 1):
            if rank[j] >= rank[i]:
                wall = j
                break
            heights[j] = heights[j + 1] - gap(j, j + 1)
            j -= 1
        if wall is not None:
            limit = heights[wall] + gap(wall, wall + 1)
            for k in range(wall + 1, i + 1):
                heights[k] = max(heights[k], limit)
                if k < i:
                    limit = heights[k] + gap(k, k + 1)

    for i in range(1, count):
        heights[i] = max(heights[i], heights[i - 1] + gap(i - 1, i))

    return heights


def place_nodes(state):
    """Set state.y for every node. The sink's output pin is the origin."""
    state.y[state.sink] = 0.0
    for layer in state.layers[1:]:
        desired = {node: _desired_height(state, node) for node in layer}
        priority = {node: _priority(state, node) for node in layer}
        for node, height in zip(layer, _spread(state, layer, desired, priority)):
            state.y[node] = height
