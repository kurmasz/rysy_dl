"""
Render: draw a finished layout with schemdraw. No decisions are made here;
every coordinate comes from the LayoutState.
"""

import schemdraw.elements as elm

from .gates import GATE_CLASSES

# Set to True to colour wires by role and label every node with its id.
DEBUG = False

_DEBUG_COLORS = {'stub': 'red', 'shared': 'blue', 'direct': 'green', 'dummy': 'purple'}


def _color(role):
    return _DEBUG_COLORS[role] if DEBUG else 'black'


def draw(state):
    """Draw the layout into state.diagram, recording each drawn gate in
    state.gate_elements."""
    netlist = state.netlist
    diagram = state.diagram

    for node in netlist.nodes():
        if state.is_source(node):
            label = netlist.nodes[node].get('label', str(node))
            element = elm.Label().at(state.output_xy(node)).label(f'${label}$', loc='left', ofst=0.15)
        elif state.is_dummy(node):
            if not DEBUG:
                continue    # nothing to draw but its wire
            element = elm.Label().at(state.output_xy(node))
        else:
            element = GATE_CLASSES[state.kind(node)](
                anchor='out', xy=state.output_xy(node), inputs=netlist.in_degree(node), theta=0)
            if node == state.sink:
                element.label('$out$', loc='right')
            state.gate_elements[node] = element

        if DEBUG:
            element.label(f'{node}', loc='top')
        diagram.add(element)

    for wire in state.wires:
        diagram.add(elm.Line().at(wire.start).to(wire.end).color(_color(wire.role)))
    for dot in state.dots:
        diagram.add(elm.Dot().at(dot).color(_color('shared')))
