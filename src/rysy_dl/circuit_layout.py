"""
circuit_layout.py

Lay out a combinational netlist as a schemdraw Drawing.

The netlist is a networkx DiGraph in which every node has a 'gate' attribute:
'src' for an input, otherwise a key of GATE_CLASSES. Edges run from each node
to the gates it feeds, and exactly one node -- the output -- feeds nothing.

This module is the entry point; the work is done by the phases in
`rysy_dl.layout`, which see for how the layout is arrived at.

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

import io

import schemdraw

from .layout import layering, ordering, placement, refinement, render, routing
from .layout.gates import GATE_CLASSES  # noqa: F401  (part of this module's interface)
from .layout.state import LayoutState, SpacingConfig


def assign_source_labels(graph, name_list=None, source_order=None):
    """Set each source node's 'label' attribute, in order, from `name_list`.

    :param graph: the netlist whose source nodes (in-degree 0) are labeled in place.
    :param name_list: the labels to hand out; defaults to 'a' through 'h'.
    :param source_order: the source nodes in the order they should receive labels
                         (e.g. from `sources_top_to_bottom`). Defaults to sorted node ids.
    """
    if name_list is None:
        name_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if source_order is None:
        source_order = sorted(node for node in graph.nodes() if graph.in_degree(node) == 0)
    for i, node in enumerate(source_order):
        graph.nodes[node]['label'] = name_list[i]


def sources_top_to_bottom(netlist):
    """Return the netlist's source nodes in the order they appear in its layout,
    top to bottom.

    Labels don't affect the layout's geometry, so this can be called before the
    sources are labeled, and its result passed to `assign_source_labels`.
    """
    state = compute_layout(netlist)
    return sorted(state.sources, key=lambda node: -state.y[node])


def compute_layout(netlist, spacing=None):
    """Work out where everything goes -- nodes, wires and junction dots -- and
    return it as a LayoutState, without drawing anything.

    `spacing` overrides the default SpacingConfig.
    """
    state = LayoutState(netlist=netlist, spacing=spacing or SpacingConfig())

    layering.assign_layers(state)
    layering.insert_dummies(state)

    ordering.order_layers(state)
    ordering.assign_pins(state)

    placement.set_layer_positions(state)
    placement.place_nodes(state)

    refinement.align_fanout_nodes(state)
    refinement.straighten_wires(state)
    refinement.separate_wire_tracks(state)
    refinement.resolve_track_conflicts(state)

    routing.route(state)
    return state


def build_layout(netlist, spacing=None):
    """Compute the layout and draw it, returning the whole LayoutState. The
    Drawing is `state.diagram`; callers that want only that should use
    `layout_circuit`."""
    state = compute_layout(netlist, spacing)
    render.draw(state)
    return state


def layout_circuit(netlist):
    """Return a schemdraw Drawing of `netlist`.

    The output pin of the circuit's output gate is at (0, 0) and everything
    else is drawn to its left, so most x-coordinates are negative.
    """
    return build_layout(netlist).diagram


def layout_circuit_png(netlist):
    schemdraw.use('matplotlib')
    circuit_diagram = layout_circuit(netlist)
    return io.BytesIO(circuit_diagram.get_imagedata(fmt='png'))
