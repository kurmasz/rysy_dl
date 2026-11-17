"""
Global placement: give every node a height by solving for all of them at once.

`placement.place_nodes` works a layer at a time from the sink, so each node is
placed knowing where its consumers are but not where its inputs will be. Here
the whole drawing is one optimisation problem:

    minimise    the total, over every connection, of
                    weight * |height the wire leaves at - height of the pin it lands on|
    subject to  neighbours in a layer staying in order, at least `min_gap` apart
                the sink's output at height 0

A connection whose term is zero is a straight wire, so this is "as little
vertical wire as possible". Minimising a sum of absolute values has a useful
habit: it prefers to make a few terms large and the rest exactly zero, over
making them all small. So it yields straight wires and a few clean steps, not
many wires each slightly off.

The absolute values are made linear in the standard way -- one extra variable
per connection, constrained to be at least the difference and at least its
negative -- which makes this a linear program.
"""

import numpy as np
from scipy.optimize import linprog

from .placement import min_gap

# How much a bend costs, by how many ends of the connection are dummies. A long
# wire is a chain of dummy-to-dummy hops, and a bend in the middle of one is
# worse than a bend where a wire meets a gate, so those hops are held straight
# hardest. (These are the weights Graphviz uses for the same purpose.)
BEND_WEIGHT = {0: 1.0, 1: 2.0, 2: 8.0}


def place_nodes(state, bend_weight=BEND_WEIGHT):
    """Set state.y for every node."""
    netlist = state.netlist
    nodes = [node for layer in state.layers for node in layer]
    column = {node: i for i, node in enumerate(nodes)}
    connections = list(netlist.edges())
    gaps = [(below, above) for layer in state.layers for below, above in zip(layer, layer[1:])]

    # Variables: a height per node, then a bend size per connection.
    variables = len(nodes) + len(connections)
    cost = np.zeros(variables)
    upper = np.zeros((2 * len(connections) + len(gaps), variables))
    bound = np.zeros(len(upper))

    for i, (pred, consumer) in enumerate(connections):
        bend = len(nodes) + i
        dummies = state.is_dummy(pred) + state.is_dummy(consumer)
        cost[bend] = bend_weight[dummies]
        pin_offset = state.shape(consumer).pins[state.pin_index[(pred, consumer)]][1]
        # bend >= y[pred] - (y[consumer] + pin_offset), and >= its negative
        upper[2 * i, [column[pred], column[consumer], bend]] = (1, -1, -1)
        bound[2 * i] = pin_offset
        upper[2 * i + 1, [column[pred], column[consumer], bend]] = (-1, 1, -1)
        bound[2 * i + 1] = -pin_offset

    for i, (below, above) in enumerate(gaps):
        row = 2 * len(connections) + i
        # y[above] - y[below] >= min_gap
        upper[row, [column[below], column[above]]] = (1, -1)
        bound[row] = -min_gap(state, below, above)

    sink_at_origin = np.zeros((1, variables))
    sink_at_origin[0, column[state.sink]] = 1

    # Dual simplex returns a corner of the feasible region. At a corner, as many
    # constraints as possible hold exactly, which is what makes wires exactly
    # straight and not merely nearly so.
    result = linprog(cost, A_ub=upper, b_ub=bound, A_eq=sink_at_origin, b_eq=[0.0],
                     bounds=[(None, None)] * len(nodes) + [(0, None)] * len(connections),
                     method='highs-ds')
    if not result.success:
        raise RuntimeError(f"Global placement failed: {result.message}")

    for node in nodes:
        state.y[node] = float(result.x[column[node]])
