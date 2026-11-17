"""
Layering: decide which layer each node goes in, then insert dummy nodes so that
every connection joins two adjacent layers.
"""

import itertools

import networkx as nx

from .gates import DUMMY


def assign_layers(state):
    """Put each node in a layer: its longest distance from the sink, except
    that every source goes in one layer, the furthest left, so the inputs line
    up down the left edge.

    Sets state.sink, state.sources, state.layers and state.layer_of.
    """
    netlist = state.netlist

    # The layout is built outward from a single output: every layer is measured
    # from it and every position is relative to it.
    sinks = [n for n in netlist.nodes() if netlist.out_degree(n) == 0]
    if len(sinks) != 1:
        raise ValueError(f"Circuit must have exactly one sink (a node with out-degree 0); "
                         f"found {len(sinks)}: {sinks}")

    layer_of = {}
    for node in reversed(list(nx.topological_sort(netlist))):
        successors = list(netlist.successors(node))
        layer_of[node] = max(layer_of[s] for s in successors) + 1 if successors else 0
    deepest_layer = max(layer_of.values())

    sources = [n for n in netlist.nodes() if netlist.in_degree(n) == 0]
    source_layer = max(layer_of[n] for n in sources)
    for node in sources:
        layer_of[node] = source_layer

    layers = [[] for _ in range(deepest_layer + 1)]
    for node, layer in layer_of.items():
        layers[layer].append(node)

    state.sink = sinks[0]
    state.sources = sources
    state.layer_of = layer_of
    state.layers = layers


def insert_dummies(state):
    """Break every connection that spans more than one layer into a chain of
    single-layer hops through dummy nodes, one per layer crossed.

    This gives a long wire a place of its own in each layer it crosses, rather
    than leaving it to cut across whatever gates are there, and it means every
    later phase can assume a node's predecessors are all in the layer
    immediately upstream. Consumers in the same layer share one dummy per
    predecessor, so a net fans out through a tree of dummies, not one chain
    per consumer.

    Modifies state.netlist, state.layers and state.layer_of.
    """
    netlist = state.netlist
    layers = state.layers
    layer_of = state.layer_of

    # Counting up from the largest integer id can't collide with the caller's
    # ids, whatever they are. (len(netlist) would, for any numbering with gaps.)
    next_dummy_id = itertools.count(
        max((n for n in netlist.nodes() if isinstance(n, int)), default=-1) + 1)

    # Dummies are appended to the next layer upstream as this goes, so by the
    # time that layer is reached they are waiting there to be extended in turn.
    for layer, nodes in enumerate(layers):
        far_predecessors = {}   # predecessor -> the nodes in this layer it reaches from afar
        for node in nodes:
            for pred in netlist.predecessors(node):
                if layer_of[pred] - layer > 1:
                    far_predecessors.setdefault(pred, []).append(node)

        for pred, consumers in far_predecessors.items():
            dummy = next(next_dummy_id)
            netlist.add_node(dummy, gate=DUMMY)
            netlist.add_edge(pred, dummy)
            for consumer in consumers:
                netlist.add_edge(dummy, consumer)
                netlist.remove_edge(pred, consumer)
            layers[layer + 1].append(dummy)
            layer_of[dummy] = layer + 1
