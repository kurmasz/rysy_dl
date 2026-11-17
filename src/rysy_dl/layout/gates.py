"""
The geometry of the gates being laid out: how tall each is and where its pins
sit. This is all that layout needs to know about a gate's appearance, and it is
measured from the schemdraw elements themselves, once per (type, fan-in).
"""

import functools
from dataclasses import dataclass

import schemdraw.logic as logic

from ..custom_gates import Dummy, FixedNot

SOURCE = 'src'
DUMMY = 'DUMMY'

GATE_CLASSES = {
    'OR': logic.Or,
    'XOR': logic.Xor,
    'AND': logic.And,
    'NAND': logic.Nand,
    'NOR': logic.Nor,
    'NOT': FixedNot,
    DUMMY: Dummy,
}


@dataclass(frozen=True)
class GateShape:
    height: float   # top to bottom of the gate body
    out_dx: float   # how far right of its layer's x the output pin sits
    pins: tuple     # (dx, dy) of each input pin from the output pin; first pin (highest) first


@functools.lru_cache(maxsize=None)
def reference_width():
    """The width every layer's spacing is based on: input pins to output pin
    of an AND gate (1.85)."""
    bbox = logic.And(inputs=6).get_bbox()
    return bbox.xmax - bbox.xmin


@functools.lru_cache(maxsize=None)
def _width(gate_type):
    bbox = GATE_CLASSES[gate_type](inputs=2).get_bbox()
    return bbox.xmax - bbox.xmin


@functools.lru_cache(maxsize=None)
def gate_shape(gate_type, fan_in):
    """The GateShape of a `gate_type` gate with `fan_in` inputs.

    Gate types differ in width (an OR's body is 0.05 wider than an AND's). All
    the gates in a layer are placed so that their *input* pins share one x --
    which is what lets the wires arriving at a layer share vertical channels --
    so a wider gate's output pin lands that much further right. `out_dx` is
    that overhang.
    """
    element = GATE_CLASSES[gate_type](inputs=fan_in)
    bbox = element.get_bbox()
    out_x, out_y = element.anchors['out']
    pins = tuple((float(element.anchors[f'in{i}'][0] - out_x),
                  float(element.anchors[f'in{i}'][1] - out_y))
                 for i in range(1, fan_in + 1))
    return GateShape(height=bbox.ymax - bbox.ymin,
                     out_dx=_width(gate_type) - reference_width(),
                     pins=pins)
