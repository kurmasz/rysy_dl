import types

import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic
import networkx as nx


def draw_not_gate(diagram, x, y, width):
    """Draw a NOT gate anchored at its output pin (x, y), pointing left.

    The gate is composed of exactly one dot and three lines:
      - Line 1: horizontal input wire
      - Line 2: top diagonal edge of triangle
      - Line 3: bottom diagonal edge of triangle
      - Dot:    output negation bubble

    Total width = pin_length + triangle_width + dot_diameter, so the gate
    occupies exactly [x - width, x] horizontally, giving fully predictable sizing.

    Args:
        diagram: schemdraw Drawing to add elements to.
        x, y:   position of the output pin (rightmost point of gate).
        width:  total gate width (pin + triangle + dot).

    Returns:
        SimpleNamespace with .in1 = (x, y) of the input pin and
                              .out = (x, y) of the output pin.
    """
    # Proportions derived from schemdraw's built-in Not constants:
    #   leadin=0.35, gatel=0.65, notbubble*2=0.24  =>  total=1.24
    #   gateh/gatel = 1.0/0.65 for triangle height
    dot_d   = width * (0.24 / 1.24)   # dot diameter
    pin_len = width * (0.35 / 1.24)   # input wire length
    tri_w   = width - dot_d - pin_len  # triangle horizontal span
    tri_h   = tri_w * (1.0 / 0.65)    # triangle height (same aspect ratio as built-in)
    dot_r   = dot_d / 2

    # x-coordinates, right-to-left from output
    x_out      = x
    x_dot_ctr  = x - dot_r
    x_apex     = x - dot_d
    x_tri_left = x_apex - tri_w
    x_in       = x_tri_left - pin_len

    # Line 1: input wire
    diagram.add(elm.Line().endpoints((x_in, y), (x_tri_left, y)))
    # Line 2: top diagonal (triangle top edge)
    diagram.add(elm.Line().endpoints((x_tri_left, y + tri_h / 2), (x_apex, y)))
    # Line 3: bottom diagonal (triangle bottom edge)
    diagram.add(elm.Line().endpoints((x_tri_left, y - tri_h / 2), (x_apex, y)))
    # Dot: output negation bubble
    diagram.add(elm.Dot(open=True, radius=dot_r).at((x_dot_ctr, y)))

    pins = types.SimpleNamespace()
    pins.in1 = (x_in, y)
    pins.out = (x_out, y)
    return pins

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
    'NOT': logic.Not,
}


def compute_layout(graph, sink, x_spacing=4.0, y_unit=1.2, y_margin=0.6):
    """Assigns every node a fixed (x, y) position before anything is drawn.
    Also inserts dummy/virtual routing nodes for any edge that spans more than one
    layer, so long edges get a dedicated lane through each intermediate column
    instead of cutting straight across real gates that live there."""
    topo_order = list(nx.topological_sort(graph))
    reverse_order = list(reversed(topo_order))

    layer = {}
    for node in reverse_order:
        successors = list(graph.successors(node))
        if not successors:
            layer[node] = 0
        else:
            layer[node] = max(layer[s] for s in successors) + 1

    # force every source to sit at the same (deepest) layer, all the way to the left
    source_layers = [layer[n] for n in graph.nodes() if graph.in_degree(n) == 0]
    if source_layers:
        max_source_layer = max(source_layers)
        for n in graph.nodes():
            if graph.in_degree(n) == 0:
                layer[n] = max_source_layer

    dummy_successor = {}
    dummy_layer = {}
    edge_chain = {}
    dummy_counter = [0]

    for node in graph.nodes():
        for pred in graph.predecessors(node):
            gap = layer[pred] - layer[node]
            if gap > 1:
                chain = []
                prev = node
                for l in range(layer[node] + 1, layer[pred]):
                    dummy_id = f"__dummy_{dummy_counter[0]}__"
                    dummy_counter[0] += 1
                    dummy_layer[dummy_id] = l
                    dummy_successor[dummy_id] = prev
                    prev = dummy_id
                    chain.append(dummy_id)
                edge_chain[(pred, node)] = chain
            else:
                edge_chain[(pred, node)] = []

    layers = {}
    for node, l in layer.items():
        layers.setdefault(l, []).append(node)
    for dummy_id, l in dummy_layer.items():
        layers.setdefault(l, []).append(dummy_id)

    position_y = {sink: 0.0}
    max_layer = max(layers.keys())

    def successors_for_barycenter(n):
        if n in dummy_successor:
            return [dummy_successor[n]]
        return list(graph.successors(n))

    for l in range(1, max_layer + 1):
        nodes_at_layer = layers.get(l, [])

        def barycenter(n):
            succs = successors_for_barycenter(n)
            if not succs:
                return 0.0
            return sum(position_y[s] for s in succs) / len(succs)

        nodes_sorted = sorted(nodes_at_layer, key=barycenter)

        y = 0.0
        slots = []
        for n in nodes_sorted:
            if n in dummy_layer:
                height = 0.0
            else:
                height = max(1, graph.in_degree(n)) * y_unit
            slots.append(y + height / 2)
            y += height + y_margin

        offset = sum(slots) / len(slots) if slots else 0.0
        for n, y_pos in zip(nodes_sorted, slots):
            position_y[n] = y_pos - offset

    position = {}
    for node in graph.nodes():
        position[node] = (-layer[node] * x_spacing, position_y[node])
    for dummy_id, l in dummy_layer.items():
        position[dummy_id] = (-l * x_spacing, position_y[dummy_id])

    return position, layer, edge_chain


def gen_circuit(graph, show=False, max_wire_layer=5):
    sink = [n for n in graph.nodes() if graph.out_degree(n) == 0][0]
    position, layer, edge_chain = compute_layout(graph, sink)

    diagram = schemdraw.Drawing()
    gate_objs = {}

    def adjacent_entity(p, s):
        chain = edge_chain.get((p, s), [])
        return chain[-1] if chain else p

    # assign each node an index within its own layer, used to slightly stagger bend lengths
    # so gates sharing a layer never produce identical vertical-jump x-coordinates
    layer_node_index = {}
    layer_counts = {}
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            continue
        l = layer[node]
        layer_node_index[node] = layer_counts.get(l, 0)
        layer_counts[l] = layer_counts.get(l, 0) + 1

    # Phase 1: draw every gate at its precomputed position
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            continue

        gate_type = graph.nodes[node]['gate']
        predecessors = list(graph.predecessors(node))
        num_inputs = len(predecessors)
        x, y = position[node]
        cls = GATE_CLASSES[gate_type]

        if node == sink:
            if gate_type == 'NOT':
                gate = draw_not_gate(diagram, x, y, width=1.24)
                diagram.add(elm.Label().at(gate.out).label('$out$', loc='right'))
            else:
                gate = diagram.add(cls(anchor='out', xy=(x, y), inputs=num_inputs, theta=0).label('$out$', loc='right'))
        else:
            if gate_type == 'NOT':
                gate = draw_not_gate(diagram, x, y, width=1.24)
            else:
                gate = diagram.add(cls(anchor='out', xy=(x, y), inputs=num_inputs, theta=0))

        gate_objs[node] = gate

    # Phase 2: label every source at its precomputed position
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            x, y = position[node]
            label = graph.nodes[node].get('label', str(node))
            diagram.add(elm.Label().at((x, y)).label(f'${label}$', loc='left'))

    # Phase 3: build the bend stub at each consumer pin, then route through any
    # dummy waypoints required for edges that skip layers, collecting final
    # arrival points grouped by predecessor for the shared junction
    consumers = {}
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            continue
        if layer[node] > max_wire_layer - 1:
            continue  # this consumer is past the layer we're currently testing

        gate = gate_objs[node]
        gate_type = graph.nodes[node]['gate']
        predecessors = list(graph.predecessors(node))
        # sort by the position of whichever entity is actually adjacent to this gate
        # (the last dummy hop, if any, rather than the original far-away predecessor)
        # so pin order reflects the wire's real final approach direction
        predecessors.sort(key=lambda p: -position[adjacent_entity(p, node)][1])
        num_inputs = len(predecessors)

        for i, pred in enumerate(predecessors):
            if gate_type == 'NOT':
                pin = gate.in1
            else:
                pin = getattr(gate, f'in{i+1}')

            distance_from_edge = min(i, num_inputs - 1 - i) + 1
            desired_bend = diagram.unit * 0.1 * 2 ** (2 - layer[node]) * distance_from_edge
            desired_bend += diagram.unit * 0.05 * layer_node_index[node]
            desired_bend += diagram.unit * 0.15 * i

            chain = edge_chain.get((pred, node), [])
            first_target_x = position[chain[0]][0] if chain else position[pred][0]
            available_distance = pin[0] - first_target_x
            cap_fraction = 0.3 + 0.05 * i + 0.02 * layer_node_index[node]
            bend_length = min(desired_bend, available_distance * cap_fraction)

            stub = diagram.add(elm.Line().left().at(pin).length(bend_length))
            current = stub.end

            # hop through each dummy waypoint, in order from the consumer side outward
            for waypoint in chain:
                wx, wy = position[waypoint]
                if abs(current[1] - wy) > 0.001:
                    if current[1] > wy:
                        v = diagram.add(elm.Line().down().at(current).toy(wy))
                    else:
                        v = diagram.add(elm.Line().up().at(current).toy(wy))
                    current = v.end
                h = diagram.add(elm.Line().left().at(current).tox(wx))
                current = h.end

            consumers.setdefault(pred, []).append(current)

    # Phase 4: connect each predecessor's single fixed position to all of its consumer endpoints.
    # For a predecessor with multiple consumers, draw one shared bus line out to the farthest
    # branch point, with a dot placed only where a wire actually merges into the bus (not at
    # the bus's own starting/farthest end, since that's not a real fork).
    for pred, pins in consumers.items():
        px, py = position[pred]

        if len(pins) == 1:
            pin = pins[0]
            if abs(pin[1] - py) > 0.001:
                if pin[1] > py:
                    v = diagram.add(elm.Line().down().at(pin).toy(py))
                else:
                    v = diagram.add(elm.Line().up().at(pin).toy(py))
                start = v.end
            else:
                start = pin
            diagram.add(elm.Line().left().at(start).tox(px))
            continue

        tap_xs = []
        for pin in pins:
            if abs(pin[1] - py) > 0.001:
                if pin[1] > py:
                    v = diagram.add(elm.Line().down().at(pin).toy(py))
                else:
                    v = diagram.add(elm.Line().up().at(pin).toy(py))
                tap_x = v.end[0]
            else:
                tap_x = pin[0]
            tap_xs.append(tap_x)

        bus_end_x = max(tap_xs)

        for tap_x in tap_xs:
            if tap_x < bus_end_x:
                diagram.add(elm.Dot().at((tap_x, py)))

        diagram.add(elm.Line().left().at((bus_end_x, py)).tox(px))

    return diagram