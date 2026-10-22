"""
custom_gates.py

Custom subclasses of `schemedraw.elements.Element`.  Standard gates (`logic.And`, `logic.Or`, etc.) are 
subclasses of Element.  `logic.Not` is a subclass of a different class, and, therefore, has a different interface
and slightly different behavior. 

The classes here allow all elements of a circuit layout to exhibit consistent interfaces and behavior.
"""

from schemdraw.elements.elements import Element
from schemdraw.segments import Segment, SegmentCircle

# Proportions derived from schemdraw's built-in Not + standard lead lengths:
#   leadin=0.35, gatel=0.65, 2*notbubble=0.24, leadout=0.35  ->  total=1.59
#   triangle height ratio: gateh/gatel = 1.0/0.65
_TOTAL = 0.35 + 0.65 + 0.24 + 0.35  # 1.59
_DEFAULT_WIDTH = 1.85

class FixedNot(Element):
    """NOT gate drawn as five line segments plus an open output bubble.

    The gate occupies exactly ``width`` drawing units horizontally, from the
    input pin tip to the output pin tip.  Internal proportions match schemdraw's
    built-in Not constants (leadin=0.35, gatel=0.65, 2*notbubble=0.24, leadout=0.35).

    Unlike schemdraw's built-in Not (an Element2Term subclass), this gate does
    not auto-stretch its leads to fill the drawing's default element length --
    its size is fixed, matching the other (Element-based) multi-input gates.

    Segments drawn:
        1. Input wire  (horizontal, left lead)
        2. Triangle left edge  (vertical)
        3. Triangle top edge   (diagonal)
        4. Triangle bottom edge (diagonal)
        5. Output wire (horizontal, right lead)
        + SegmentCircle: open negation bubble

    Anchors:
        in1  - input pin  (left edge, x=0)
        out  - output pin (right edge, x=width)

    Args:
        width: Total gate width from input pin to output pin.
               Defaults to 1.85  (matches the default 2-input And/Or gate width).
    """

    def __init__(self, width: float = _DEFAULT_WIDTH, **kwargs):
        super().__init__(**kwargs)

        pin_len = width * (0.35 / _TOTAL)  # input wire length (= output wire length)
        tri_w = width * (0.65 / _TOTAL)    # triangle horizontal span
        dot_d = width * (0.24 / _TOTAL)    # dot diameter
        tri_h = tri_w * (1.0 / 0.65)       # triangle height (same aspect ratio as built-in)
        dot_r = dot_d / 2

        # x-coordinates in element-local space (in1 at origin, out at width)
        x_tri_left = pin_len
        x_apex = pin_len + tri_w
        x_dot_ctr = x_apex + dot_r
        x_out = x_apex + dot_d + pin_len  # == width

        # Segment 1: input wire
        self.segments.append(Segment([(0, 0), (x_tri_left, 0)]))
        # Segment 2: triangle left edge
        self.segments.append(
            Segment([(x_tri_left, -tri_h / 2), (x_tri_left, tri_h / 2)])
        )
        # Segment 3: triangle top edge
        self.segments.append(Segment([(x_tri_left, tri_h / 2), (x_apex, 0)]))
        # Segment 4: triangle bottom edge
        self.segments.append(Segment([(x_tri_left, -tri_h / 2), (x_apex, 0)]))
        # Open negation bubble
        self.segments.append(SegmentCircle((x_dot_ctr, 0), dot_r, fill=False))
        # Segment 5: output wire
        self.segments.append(Segment([(x_apex + dot_d, 0), (x_out, 0)]))

        self.anchors["in1"] = (0, 0)
        self.anchors["out"] = (x_out, 0)

class Dummy(Element):
    """Used as a placeholder to route wires that skip a level."""
     
    def __init__(self, width: float = _DEFAULT_WIDTH, color: str = 'black', **kwargs):
        super().__init__(**kwargs)

        self.segments.append(Segment([(-width, 0), (0, 0)], color=color))
        self.anchors["in1"] = (-width, 0)
        self.anchors["out"] = (0, 0)
