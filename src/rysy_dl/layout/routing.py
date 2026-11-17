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


def _gap_crossings(nets, order):
    """Crossings within one gap between layers, for a given channel order.

    `nets[net]` is (height of the output, heights of the pins it drives), and
    `order[k]` is the net in channel k. Channel 0 is nearest the consumers and
    higher channels step left toward the outputs. So a net's run, coming from
    the left, crosses the buses of the channels to the *left* of its own, and
    each of its stubs, heading right, crosses the buses of the channels to the
    *right* of its own -- in each case only those buses that span the height
    in question. Touching the very end of a bus is a corner, not a crossing,
    so the comparisons are strict.
    """
    channel_of = {net: k for k, net in enumerate(order)}
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
    return crossings


def _best_channel_order(nets, order):
    """Improve a gap's channel order by swapping neighbouring channels for as
    long as a swap reduces crossings.

    Swapping two channels moves no node and changes no height -- only how far
    each wire runs before turning -- so it reduces crossings for free.
    """
    order = list(order)
    fewest = _gap_crossings(nets, order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            crossings = _gap_crossings(nets, order)
            if crossings < fewest:
                fewest, improved = crossings, True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]
    return order


def assign_channels(state):
    """{net: channel} -- which vertical channel, counting from the consumers'
    side of its gap, each net's bus runs in."""
    netlist = state.netlist
    channel_of = {}
    for layer in state.layers[1:]:
        nets = {node: (state.y[node],
                       [state.pin_y(consumer, node) for consumer in netlist.successors(node)])
                for node in layer}
        # Start with the highest net nearest the consumers, then improve on it.
        for channel, net in enumerate(_best_channel_order(nets, list(reversed(layer)))):
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
