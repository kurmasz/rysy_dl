"""
Ordering: choose the bottom-to-top order of the nodes within each layer so that
wires cross as little as possible.

This is the ordering phase of a layered ("Sugiyama") layout. It is deliberately
separate from, and before, placement: ordering is a combinatorial problem
(minimise crossings) and placement a geometric one (minimise bends), and
solving them together optimises neither. Every later phase treats the order
chosen here as fixed.
"""


def count_crossings(netlist, upstream_order, downstream_order):
    """Crossings between two adjacent layers in the given orders.

    Two connections cross exactly when their ends appear in opposite orders in
    the two layers. Connections that share an end meet at a point rather than
    crossing, and aren't counted.
    """
    downstream_index = {node: i for i, node in enumerate(downstream_order)}
    connections = sorted(
        (i, downstream_index[consumer])
        for i, node in enumerate(upstream_order)
        for consumer in netlist.successors(node)
        if consumer in downstream_index)

    crossings = 0
    for i in range(len(connections)):
        for j in range(i + 1, len(connections)):
            if connections[i][0] < connections[j][0] and connections[i][1] > connections[j][1]:
                crossings += 1
    return crossings


def total_crossings(netlist, layers):
    return sum(count_crossings(netlist, layers[k + 1], layers[k])
               for k in range(len(layers) - 1))


def _median_order(layer, neighbours_of, neighbour_index):
    """Reorder one layer by the median index of each node's neighbours in an
    adjacent layer whose order is already settled.

    The median rather than the mean is the standard choice: it isn't dragged
    off by a single distant neighbour, which matters here because a source can
    fan out to gates at wildly different heights.
    """
    def key(index_and_node):
        index, node = index_and_node
        indices = sorted(neighbour_index[m] for m in neighbours_of(node) if m in neighbour_index)
        # With nothing to align to, stay put rather than collapse to one end.
        return indices[len(indices) // 2] if indices else index

    return [node for _, node in sorted(enumerate(layer), key=key)]


def _swap_neighbours(netlist, layers):
    """Swap adjacent pairs within each layer wherever doing so reduces
    crossings, until no swap helps. Median ordering gets close but leaves
    locally fixable crossings behind. Modifies `layers` in place."""
    def crossings_around(k):
        total = 0
        if k > 0:
            total += count_crossings(netlist, layers[k], layers[k - 1])
        if k + 1 < len(layers):
            total += count_crossings(netlist, layers[k + 1], layers[k])
        return total

    improved = True
    while improved:
        improved = False
        for k, layer in enumerate(layers):
            for i in range(len(layer) - 1):
                before = crossings_around(k)
                layer[i], layer[i + 1] = layer[i + 1], layer[i]
                if crossings_around(k) < before:
                    improved = True
                else:
                    layer[i], layer[i + 1] = layer[i + 1], layer[i]


def order_layers(state, sweeps=8):
    """Reorder each of state.layers (bottom to top) to minimise crossings.

    Alternating median sweeps carry a consistent order outward from the sink
    and back again, with a pass of neighbour swaps after each. Every result is
    scored by its actual crossing count and the best one seen is kept, so more
    sweeps can never give a worse order than fewer.
    """
    netlist = state.netlist
    layers = [list(layer) for layer in state.layers]

    best = [list(layer) for layer in layers]
    fewest = total_crossings(netlist, layers)

    for sweep in range(sweeps):
        if sweep % 2 == 0:
            # Away from the sink: order each layer by its consumers.
            for k in range(1, len(layers)):
                downstream_index = {node: i for i, node in enumerate(layers[k - 1])}
                layers[k] = _median_order(layers[k], netlist.successors, downstream_index)
        else:
            # Back toward the sink: order each layer by its predecessors.
            for k in range(len(layers) - 2, -1, -1):
                upstream_index = {node: i for i, node in enumerate(layers[k + 1])}
                layers[k] = _median_order(layers[k], netlist.predecessors, upstream_index)

        _swap_neighbours(netlist, layers)

        crossings = total_crossings(netlist, layers)
        if crossings < fewest:
            fewest = crossings
            best = [list(layer) for layer in layers]
        if fewest == 0:
            break

    state.layers = best


def assign_pins(state):
    """Decide which input pin of each gate each incoming wire lands on: the
    highest predecessor takes the highest pin, and so on down, so that wires
    arriving at a gate don't cross each other on the way in.

    A gate's predecessors are all in the layer just upstream, and that layer's
    order is now fixed, so this is settled once here; everything afterwards
    looks it up. Sets state.pin_index.
    """
    netlist = state.netlist
    rank = {node: i for layer in state.layers for i, node in enumerate(layer)}
    for consumer in netlist.nodes():
        highest_first = sorted(netlist.predecessors(consumer), key=rank.get, reverse=True)
        for pin, pred in enumerate(highest_first):
            state.pin_index[(pred, consumer)] = pin
