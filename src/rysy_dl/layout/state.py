"""
LayoutState: the netlist being laid out, plus everything the layout phases
have worked out about it so far.
"""

from dataclasses import dataclass, field

import networkx as nx
import schemdraw

from .gates import DUMMY, gate_shape

# Heights that differ by no more than this are the same height reached by a
# different order of arithmetic, not a real offset.
FLOAT_NOISE = 1e-9


@dataclass(frozen=True)
class SpacingConfig:
    """The drawing-style constants that control wire and gate spacing."""

    # Horizontal distance between neighbouring vertical channels.
    wire_spacing: float = 0.25

    # Horizontal distance from a layer's input pins to the nearest vertical
    # channel, and from the furthest channel to the layer upstream.
    pin_to_vertical_spacing: float = 0.4

    # Vertical gap between adjacent nodes in a layer when either is a gate.
    y_margin: float = 0.3

    # Vertical gap between two adjacent weightless nodes. With
    # weightless_height this gives a pitch of 0.33, the closest any gate's
    # input pins sit, so a row of sources feeding one gate can line up with its
    # pins. Going lower buys nothing: no gate's pins are nearer than that.
    label_y_margin: float = 0.18

    # Vertical room taken by a weightless node.
    weightless_height: float = 0.15

    # Two different nets' wires on horizontal tracks closer than this read as
    # one line -- worse than a crossing, which at least is unambiguous.
    wire_clearance: float = 0.1


@dataclass(frozen=True)
class Wire:
    """One straight piece of wire. `role` says what part of a net it is, which
    matters only for colouring a debug drawing: 'stub' (out from an input pin
    to its channel), 'direct' (the rest of a net with one consumer), 'shared'
    (the rest of a net with several), or 'dummy' (the run across a layer)."""
    start: tuple
    end: tuple
    role: str


@dataclass
class LayoutState:
    """Fields are filled in phase by phase; each is marked with the phase that
    sets it. The netlist is copied on construction, because layering splices
    dummy nodes into it and callers shouldn't see their graph change."""

    netlist: nx.DiGraph
    spacing: SpacingConfig = field(default_factory=SpacingConfig)

    # layering
    sink: object = None                             # the output node (may be 0, so test with `is None`)
    sources: list = field(default_factory=list)     # nodes with no inputs
    layers: list = field(default_factory=list)      # layers[k] = the nodes in layer k
    layer_of: dict = field(default_factory=dict)    # node -> its layer

    # ordering. From here on each layers[k] runs bottom to top, and later
    # phases move nodes only in ways that keep it so.
    pin_index: dict = field(default_factory=dict)   # (pred, consumer) -> which input pin (0 = highest)

    # placement
    x_spacing: float = 0.0                          # distance between adjacent layers
    y: dict = field(default_factory=dict)           # node -> height

    # routing
    wires: list = field(default_factory=list)       # [Wire]
    dots: list = field(default_factory=list)        # [(x, y)] junction dots

    # render
    diagram: schemdraw.Drawing = field(default_factory=schemdraw.Drawing)
    gate_elements: dict = field(default_factory=dict)   # node -> the drawn gate (real gates only)

    def __post_init__(self):
        self.netlist = self.netlist.copy()

    # ── what kind of node ────────────────────────────────────────────────────

    def kind(self, node):
        return self.netlist.nodes[node]['gate']

    def is_source(self, node):
        return self.netlist.in_degree(node) == 0

    def is_dummy(self, node):
        return self.kind(node) == DUMMY

    def is_weightless(self, node):
        return self.is_source(node) or self.is_dummy(node)

    def shape(self, node):
        """The GateShape of a gate or dummy (sources have none)."""
        return gate_shape(self.kind(node), self.netlist.in_degree(node))

    # ── where things are ─────────────────────────────────────────────────────
    #
    # A node's position (x, y) is where its wire leaves it, give or take
    # out_dx: the label point of a source, the output pin of a gate.

    def x(self, node):
        """The sink is at x = 0; everything else is to its left."""
        return -self.layer_of[node] * self.x_spacing

    @property
    def position(self):
        """{node: (x, y)} for every placed node."""
        return {node: (self.x(node), y) for node, y in self.y.items()}

    def output_xy(self, node):
        """Where a wire leaves `node`."""
        if self.is_source(node):
            return (self.x(node), self.y[node])
        return (self.x(node) + self.shape(node).out_dx, self.y[node])

    def pin_xy(self, consumer, pred):
        """The input pin of `consumer` that `pred`'s wire lands on."""
        out_x, out_y = self.output_xy(consumer)
        dx, dy = self.shape(consumer).pins[self.pin_index[(pred, consumer)]]
        return (out_x + dx, out_y + dy)

    def pin_y(self, consumer, pred):
        return self.shape(consumer).pins[self.pin_index[(pred, consumer)]][1] + self.y[consumer]
