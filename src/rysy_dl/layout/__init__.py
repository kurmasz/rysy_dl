"""
The phases of circuit layout, one module each, in the order they run:

    layering    which column (layer) each node goes in; dummy waypoints for
                connections that span more than one layer
    ordering    the bottom-to-top order of the nodes within each layer, chosen
                to minimise wire crossings; fixes which input pin each wire
                lands on
    placement   a height for every node, within that order
    refinement  small adjustments to heights that remove needless bends and
                keep unrelated wires off the same track
    routing     the wire segments and junction dots joining outputs to pins
    render      draws the result with schemdraw

Everything up to and including routing is plain geometry: it never touches
schemdraw's drawing machinery, and its whole result is held in a LayoutState
(see `state`). `gates` supplies the only facts about the drawn gates those
phases need -- how big each gate is and where its pins are.

`rysy_dl.circuit_layout` is the public entry point; it runs these in order.

Vocabulary used throughout:

    layer       a column of nodes. Layer 0 holds the sink (the circuit's
                output) and is drawn furthest right; higher layers are further
                left, ending with the sources.
    upstream    toward the sources (a higher layer, further left).
    downstream  toward the sink (a lower layer, further right).
    above/below higher/lower on the page, i.e. larger/smaller y. Never used
                for layers.
    net         one output and every input pin it drives.
    dummy       a waypoint standing in for a stretch of wire that crosses a
                layer, so that every connection joins adjacent layers.
    weightless  a source or a dummy: a label or a bend in a wire, with no gate
                body, so it can be moved without dragging pins along.
"""
