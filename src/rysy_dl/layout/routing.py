"""
Routing: turn the placed nodes into wire segments and junction dots.

Every net is drawn the same way. In the gap between its layer and the one
downstream it is given a vertical *channel*, an x of its own:

    output ──────────┐  <- run: from the output across to the channel
                     │
                     ├───── pin    <- stubs: from the channel out to each
                     │                input pin the net drives
                     └───── pin
                     ^
                     bus: up and down the channel, spanning everything that
                     attaches to it

A net with one consumer is the same picture with one stub; if the output is
level with the pin, the bus has no length and the wire is a straight line.
"""

from .state import FLOAT_NOISE, Wire


def must_be_left_of(nets, clearance):
    """The pairs (a, b) for which net `a`'s channel has to be further left
    than net `b`'s, or two wires will run along the same track.

    Horizontal wires in a gap are of two kinds: a net's run, from its output
    rightward to its channel, and its stubs, from its channel rightward to its
    pins. Two runs can't share a track (their outputs are in one layer, a
    gate's height apart), nor can two stubs (no two pins are that close). But
    `a`'s run and one of `b`'s stubs can, if `a`'s output is level with a pin
    of `b` -- and then they overlap exactly when `a`'s run carries on past
    `b`'s channel, that is, when `a`'s channel is the further right. Put `a`'s
    channel to the left and its run stops short of where `b`'s stub begins.
    """
    return {(a, b)
            for a, (out_y, _) in nets.items()
            for b, (_, pin_ys) in nets.items()
            if a != b and any(abs(out_y - pin_y) < clearance for pin_y in pin_ys)}


def _gap_cost(nets, order, left_of):
    """(wires sharing a track, crossings) within one gap between layers, for a
    given channel order -- compared in that order, because two wires on one
    track are misleading where a crossing is merely untidy.

    `nets[net]` is (height of the output, heights of the pins it drives),
    `order[k]` is the net in channel k, and `left_of` is `must_be_left_of`
    these nets. Channel 0 is nearest the consumers and higher channels step
    left toward the outputs. So a net's run, coming from the left, crosses the
    buses of the channels to the *left* of its own, and each of its stubs,
    heading right, crosses the buses of the channels to the *right* of its own
    -- in each case only those buses that span the height in question.
    Touching the very end of a bus is a corner, not a crossing, so the
    comparisons are strict.
    """
    channel_of = {net: k for k, net in enumerate(order)}
    shared_tracks = sum(1 for a, b in left_of if channel_of[a] < channel_of[b])

    span = {}
    for net in order:
        out_y, pin_ys = nets[net]
        span[net] = (min([out_y] + pin_ys), max([out_y] + pin_ys))

    crossings = 0
    for net in order:
        out_y, pin_ys = nets[net]
        for other in order:
            if other == net:
                continue
            low, high = span[other]
            if channel_of[other] > channel_of[net]:
                if low < out_y < high:
                    crossings += 1
            else:
                crossings += sum(1 for y in pin_ys if low < y < high)
    return (shared_tracks, crossings)


def _improve_by_swaps(nets, order, left_of):
    """Swap neighbouring channels for as long as a swap lowers the cost.
    Returns (cost, order).

    Swapping two channels moves no node and changes no height -- only how far
    each wire runs before turning -- so whatever it gains is free.
    """
    order = list(order)
    lowest = _gap_cost(nets, order, left_of)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            cost = _gap_cost(nets, order, left_of)
            if cost < lowest:
                lowest, improved = cost, True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]
    return lowest, order


def _order_respecting(order, left_of):
    """`order` rearranged as little as possible so that every pair in
    `left_of` is honoured: each net as early as it can be without coming
    before something that must be to its right. Where the pairs contradict
    each other (a left of b, b left of a) one of them is given up."""
    waiting_on = {net: {b for a, b in left_of if a == net} for net in order}
    result = []
    remaining = list(order)
    while remaining:
        placed = set(result)
        ready = [net for net in remaining if waiting_on[net] <= placed]
        chosen = ready[0] if ready else remaining[0]
        result.append(chosen)
        remaining.remove(chosen)
    return result


def best_channel_order(nets, order, clearance):
    """The channel order for one gap: no two wires on one track if that is
    possible at all, and as few crossings as swapping neighbours can reach.
    Returns (cost, order), with cost as in `_gap_cost`.

    Swapping can only ever improve on where it starts, and from the natural
    starting order it can stall with wires still sharing a track. So it is
    also started from an order built to have none, and the better result wins.
    """
    left_of = must_be_left_of(nets, clearance)
    candidates = [_improve_by_swaps(nets, order, left_of)]
    if left_of:
        candidates.append(_improve_by_swaps(nets, _order_respecting(order, left_of), left_of))
    return min(candidates, key=lambda candidate: candidate[0])


def gap_nets(state, layer):
    """{net: (height of its output, heights of the pins it drives)} for the
    nets leaving `layer`."""
    return {node: (state.y[node],
                   [state.pin_y(consumer, node) for consumer in state.netlist.successors(node)])
            for node in layer}


def gap_cost(state, layer):
    """(wires sharing a track, crossings) in the gap downstream of `layer`,
    once its channels have been given the best order found. Wires share a
    track here only if no order could part them: one net must be left of
    another, and that one left of it."""
    clearance = state.spacing.wire_clearance - FLOAT_NOISE
    cost, _ = best_channel_order(gap_nets(state, layer), list(reversed(layer)), clearance)
    return cost


def assign_channels(state):
    """{net: channel} -- which vertical channel, counting from the consumers'
    side of its gap, each net's bus runs in."""
    clearance = state.spacing.wire_clearance - FLOAT_NOISE
    channel_of = {}
    for layer in state.layers[1:]:
        # Start with the highest net nearest the consumers, then improve on it.
        _, order = best_channel_order(gap_nets(state, layer), list(reversed(layer)), clearance)
        for channel, net in enumerate(order):
            channel_of[net] = channel
    return channel_of


def _route_net(state, net, channel):
    """The wires and dots joining `net`'s output to every pin it drives."""
    spacing = state.spacing
    consumers = list(state.netlist.successors(net))
    role = 'shared' if len(consumers) > 1 else 'direct'
    out_x, out_y = state.output_xy(net)
    stub_length = spacing.pin_to_vertical_spacing + channel * spacing.wire_spacing

    wires, dots = [], []
    stub_heights = []
    channel_x = None
    for consumer in consumers:
        pin_x, pin_y = state.pin_xy(consumer, net)
        if channel_x is None:
            channel_x = pin_x - stub_length
        wires.append(Wire((pin_x, pin_y), (pin_x - stub_length, pin_y), 'stub'))
        stub_heights.append(out_y if abs(pin_y - out_y) <= FLOAT_NOISE else pin_y)

    top = max(stub_heights + [out_y])
    bottom = min(stub_heights + [out_y])
    if top > bottom:
        wires.append(Wire((channel_x, top), (channel_x, bottom), role))
    wires.append(Wire((out_x, out_y), (channel_x, out_y), role))

    # A dot belongs wherever three or more arms of wire meet, and nowhere
    # else. At each height where something attaches to the bus, the arms are:
    # the bus continuing up, the bus continuing down, a stub leaving to the
    # right, and the run arriving from the left.
    for height in sorted(set(stub_heights + [out_y])):
        arms = (height < top) + (height > bottom) + (height in stub_heights) + (height == out_y)
        if arms >= 3:
            dots.append((channel_x, height))

    return wires, dots


def route(state):
    """Set state.wires and state.dots."""
    channel_of = assign_channels(state)
    state.wires, state.dots = [], []

    for layer in state.layers[1:]:
        for node in layer:
            wires, dots = _route_net(state, node, channel_of[node])
            state.wires.extend(wires)
            state.dots.extend(dots)

    # A dummy is the stretch of wire that carries its net across a layer: from
    # where an input pin would be, to where an output pin would be.
    for node in state.netlist.nodes():
        if state.is_dummy(node):
            (pred,) = state.netlist.predecessors(node)
            state.wires.append(Wire(state.pin_xy(node, pred), state.output_xy(node), 'dummy'))
