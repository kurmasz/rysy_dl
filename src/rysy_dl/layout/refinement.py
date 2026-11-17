"""
Refinement: adjust the heights that `placement` chose, now that the whole
layout exists to judge them against.

Placement works one layer at a time, so a layer knows nothing of what the
layers either side of it needed. That leaves faults which are easy to see, and
easy to fix, once every node has a height:

    align_fanout_nodes     a node feeding several gates sits level with none
    straighten_wires       a long wire takes a step at every layer it crosses
    separate_wire_tracks   two unrelated wires run along the same track

Each pass moves one node at a time, never past its neighbours in its layer
(so the crossing-minimal order survives), and keeps a move only if it helps.
"""

import contextlib

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


def separate_wire_tracks(state):
    """Nudge a weightless node aside when its wire would run along the same
    horizontal track as a different net's.

    Placement keeps *nodes* apart but says nothing about the tracks their
    wires run along, so a node can land a hair away from a pin that some other
    net's wire runs to, and the two wires cross the gap between layers side by
    side, close enough to read as one.

    Only weightless nodes are moved: shifting a gate would take its input pins
    with it and just move the problem. A node that can't be cleared without
    passing a neighbour is left where it is.
    """
    clearance = state.spacing.wire_clearance
    pin_heights = _pin_heights_by_net(state)

    for layer in state.layers:
        for node in layer:
            if not state.is_weightless(node):
                continue
            tracks = _foreign_tracks(state, node, pin_heights)
            current = state.y[node]
            if _clear_of(current, tracks, clearance):
                continue

            limits = _travel_limits(state, node)
            candidates = [current]
            for track in tracks:
                candidates.extend((track + clearance, track - clearance))

            nearest = None
            for candidate in candidates:
                if not _within(candidate, limits):
                    continue
                if not _clear_of(candidate, tracks, clearance - FLOAT_NOISE):
                    continue
                if nearest is None or abs(candidate - current) < abs(nearest - current):
                    nearest = candidate

            if nearest is not None:
                state.y[node] = nearest
