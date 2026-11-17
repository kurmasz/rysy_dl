"""
Refinement: adjust the heights that placement chose, now that the whole layout
exists to judge them against.

    align_fanout_nodes       a node feeding several gates sits level with none
    straighten_wires         a long wire takes a step at every layer it crosses
    resolve_track_conflicts  two unrelated wires would run along the same track

Each pass moves nodes only in ways that keep every layer in order and every
pair of neighbours their minimum distance apart, and keeps a move only if it
helps. `refine` runs them all; `quality` scores the result.
"""

import contextlib

from . import routing
from .gates import DUMMY
from .placement import min_gap
from .state import FLOAT_NOISE


def wire_paths(netlist):
    """Every connection from a gate or source to a gate, as the list of nodes
    it passes through: [start, dummy, ..., dummy, consumer].

    Dummies can branch (see `layering.insert_dummies`), so one start may begin
    several paths that share their first few dummies.
    """
    def is_dummy(node):
        return netlist.nodes[node]['gate'] == DUMMY

    paths = []
    for node in netlist.nodes():
        if is_dummy(node):
            continue
        unfinished = [[node, successor] for successor in netlist.successors(node)]
        while unfinished:
            path = unfinished.pop()
            if is_dummy(path[-1]):
                unfinished.extend(path + [s] for s in netlist.successors(path[-1]))
            else:
                paths.append(path)
    return paths


def path_heights(state, path):
    """The height of a wire at each point along `path`: where it leaves the
    start, at each dummy, and at the input pin it finally lands on."""
    return [state.y[node] for node in path[:-1]] + [state.pin_y(path[-1], path[-2])]


def _jogs(state, paths):
    """How many times the wires along `paths` change height, in total."""
    total = 0
    for path in paths:
        heights = path_heights(state, path)
        total += sum(1 for a, b in zip(heights, heights[1:]) if abs(a - b) > FLOAT_NOISE)
    return total


@contextlib.contextmanager
def _moved(state, heights):
    """Temporarily put nodes at the given {node: height}."""
    saved = {node: state.y[node] for node in heights}
    state.y.update(heights)
    try:
        yield
    finally:
        state.y.update(saved)


def _travel_limits(state, node):
    """(lowest, highest) height `node` may take without coming too close to,
    or passing, its neighbours in its layer."""
    layer = state.layers[state.layer_of[node]]
    index = layer.index(node)
    low, high = float('-inf'), float('inf')
    if index > 0:
        below = layer[index - 1]
        low = state.y[below] + min_gap(state, below, node)
    if index < len(layer) - 1:
        above = layer[index + 1]
        high = state.y[above] - min_gap(state, above, node)
    return low, high


def _within(height, limits):
    low, high = limits
    return low - FLOAT_NOISE <= height <= high + FLOAT_NOISE


def _pin_heights_by_net(state):
    """{node: the heights of the input pins its wire runs to}."""
    netlist = state.netlist
    pins = {}
    for consumer in netlist.nodes():
        for pred in netlist.predecessors(consumer):
            pins.setdefault(pred, []).append(state.pin_y(consumer, pred))
    return pins


def _foreign_tracks(state, node, pin_heights):
    """The heights at which other nets' wires run horizontally beside `node`.

    A node's height is a horizontal track on both sides of it: to its right
    its own wire runs out to a channel, and to its left the wires feeding it
    arrive. The gap on the right also carries the other nets of this layer,
    on their way to their pins; the gap on the left carries the wires leaving
    the layer upstream. `node`'s own predecessors are left out of the latter:
    their wires join this node, rather than run alongside it.
    """
    layer = state.layer_of[node]
    tracks = []

    for other in state.layers[layer]:
        if other != node:
            tracks.extend(pin_heights.get(other, []))

    if layer + 1 < len(state.layers):
        predecessors = set(state.netlist.predecessors(node))
        for other in state.layers[layer + 1]:
            if other not in predecessors:
                tracks.append(state.y[other])

    return tracks


def _clear_of(height, tracks, clearance):
    return all(abs(height - track) >= clearance for track in tracks)


def align_fanout_nodes(state):
    """Move a node that feeds several gates level with the highest or lowest
    pin it reaches, if that reduces bends.

    Placement puts such a node at the average of its pins: between them, level
    with none, so every one of its wires has to step. Level with a pin, one
    wire runs straight. And level with the highest or lowest pin, the node's
    connection to its bus moves to an end of the bus -- still a junction dot,
    but at the bus's corner, clear of the wires that cross its length.
    """
    netlist = state.netlist
    paths = wire_paths(netlist)
    pin_heights = _pin_heights_by_net(state)
    clearance = state.spacing.wire_clearance - FLOAT_NOISE

    for node in netlist.nodes():
        if netlist.out_degree(node) < 2:
            continue
        tracks = _foreign_tracks(state, node, pin_heights)
        pins = [state.pin_y(consumer, node) for consumer in netlist.successors(node)]
        if len(set(round(pin, 9) for pin in pins)) < 2:
            continue

        current = state.y[node]
        limits = _travel_limits(state, node)
        fewest = _jogs(state, paths)
        chosen = None
        for candidate in sorted({max(pins), min(pins)}, key=lambda height: abs(height - current)):
            if not _within(candidate, limits) or not _clear_of(candidate, tracks, clearance):
                continue
            with _moved(state, {node: candidate}):
                jogs = _jogs(state, paths)
            if jogs < fewest:
                fewest, chosen = jogs, candidate

        if chosen is not None:
            state.y[node] = chosen


def straighten_wires(state, rounds=3):
    """Slide the dummies along each long wire so that it changes height once
    rather than at every layer it crosses.

    A wire through several dummies starts at one height and must end at
    another, which needs only a single step somewhere along the way. This
    tries the step at each position in turn -- the dummies before it level
    with the start, those after it level with the end -- and keeps the first
    arrangement that lowers the total number of bends in the drawing.

    Dummies are the ideal thing to move: a dummy is only a bend in a wire, so
    sliding one relocates no gate and disturbs no label.
    """
    paths = [path for path in wire_paths(state.netlist) if len(path) > 2]
    if not paths:
        return

    fewest = _jogs(state, paths)
    for _ in range(rounds):
        improved = False
        for path in paths:
            dummies = path[1:-1]
            heights = path_heights(state, path)
            start, end = heights[0], heights[-1]

            for step_at in range(len(dummies) + 1):
                plan = dict(zip(dummies, [start] * step_at + [end] * (len(dummies) - step_at)))
                if not all(_within(height, _travel_limits(state, dummy))
                           for dummy, height in plan.items()):
                    continue
                with _moved(state, plan):
                    jogs = _jogs(state, paths)
                if jogs < fewest:
                    state.y.update(plan)
                    fewest = jogs
                    improved = True
                    break

        if not improved:
            break


def _push_plan(state, node, height):
    """{node: new height} for moving `node` to `height` and pushing along any
    neighbours in its layer that would otherwise be left too close to it --
    and theirs in turn. The layer's order, and every minimum gap, survive."""
    layer = state.layers[state.layer_of[node]]
    index = layer.index(node)
    plan = {node: height}
    if height > state.y[node]:
        for above, below in zip(layer[index + 1:], layer[index:]):
            needed = plan[below] + min_gap(state, below, above)
            if state.y[above] >= needed - FLOAT_NOISE:
                break
            plan[above] = needed
    else:
        for below, above in zip(reversed(layer[:index]), reversed(layer[:index + 1])):
            needed = plan[above] - min_gap(state, below, above)
            if state.y[below] <= needed + FLOAT_NOISE:
                break
            plan[below] = needed
    return plan


def resolve_track_conflicts(state):
    """Move a node where two nets would share a track whatever channels
    routing gave them.

    Routing keeps one net's run off another's stub by putting the first net's
    channel to the left of the second's (see `routing.must_be_left_of`), which
    costs nothing. That fails only when the requirements contradict each
    other -- each of two nets has its output level with a pin of the other, so
    each would have to be left of the other. Then one of them has to move.

    For each gap that has such a contradiction, this tries moving each net
    involved to just clear of each pin it is near, pushing its neighbours
    along if they are in the way (these nodes are usually packed tight, which
    is how they came to be level with each other's pins). It takes the move
    that leaves the fewest shared tracks, then the fewest crossings, then the
    fewest bends, then moves the fewest gates, the shortest distance. A gate's
    pins move with it, so the gap upstream is counted too.
    """
    netlist = state.netlist
    clearance = state.spacing.wire_clearance
    paths = wire_paths(netlist)

    def cost_around(k):
        """(shared tracks, crossings) over the gaps either side of layer k."""
        costs = [routing.gap_cost(state, state.layers[gap])
                 for gap in (k, k + 1) if gap < len(state.layers)]
        return tuple(sum(values) for values in zip(*costs))

    for k in range(1, len(state.layers)):
        layer = state.layers[k]
        for _ in layer:     # each move settles at least one pair, so this is ample
            if routing.gap_cost(state, layer)[0] == 0:
                break
            nets = routing.gap_nets(state, layer)
            involved = {net for pair in routing.must_be_left_of(nets, clearance - FLOAT_NOISE)
                        for net in pair}

            best = cost_around(k) + (_jogs(state, paths), 0, 0.0)
            best_plan = None
            for node in sorted(involved, key=layer.index):
                for other, (_, pin_ys) in nets.items():
                    if other == node:
                        continue
                    for pin_y in pin_ys:
                        for candidate in (pin_y + clearance, pin_y - clearance):
                            plan = _push_plan(state, node, candidate)
                            gates_moved = sum(1 for moved in plan if not state.is_weightless(moved))
                            distance = sum(abs(height - state.y[moved]) for moved, height in plan.items())
                            with _moved(state, plan):
                                result = cost_around(k) + (_jogs(state, paths), gates_moved, distance)
                            if result < best:
                                best, best_plan = result, plan

            if best_plan is None:
                break
            state.y.update(best_plan)


def refine(state):
    align_fanout_nodes(state)
    straighten_wires(state)
    resolve_track_conflicts(state)


def quality(state):
    """How good the placement in state.y is, as a tuple in which lower is
    better and earlier entries matter more: wires that will share a track,
    wire crossings, bends, and total vertical wire."""
    paths = wire_paths(state.netlist)
    gap_costs = [routing.gap_cost(state, layer) for layer in state.layers[1:]]
    vertical_wire = sum(abs(a - b)
                        for path in paths
                        for heights in [path_heights(state, path)]
                        for a, b in zip(heights, heights[1:]))
    return (sum(shared_tracks for shared_tracks, _ in gap_costs),
            sum(crossings for _, crossings in gap_costs),
            _jogs(state, paths),
            vertical_wire)
