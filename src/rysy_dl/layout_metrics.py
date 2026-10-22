"""
layout_metrics.py

Score the quality of the layouts produced by `circuit_layout` over a corpus of
circuits, so changes to the layout algorithm can be evaluated by measurement
instead of by eye.

Every metric here is "lower is better".

    crossings     Places where two wires visibly cross. Measured on the wire
                  segments actually drawn -- stub, vertical channel, and
                  horizontal run -- not on an idealised straight line between
                  layers, because which wires cross depends on the vertical
                  channel each one is routed through, and channel assignment
                  is part of what a layout change might alter. Wires that meet
                  at a T-junction are connections, not crossings, and aren't
                  counted.

    jogs          Total direction changes over all wires, measured between the
                  real endpoints: the output pin a wire leaves from and the
                  specific input pin it lands on. Measuring node-to-node
                  instead misses the most common jog of all -- several sources
                  running into one gate, where the sources are stacked at a
                  different pitch than the gate's input pins, so every wire
                  steps a little on the way in.

                  No allowance is subtracted for jogs that look unavoidable. A
                  source's height is free (it is only a label), so a wire into
                  one can always in principle be straightened; deciding which
                  of the rest are truly forced needs the whole layout, not one
                  wire, so this counts them all and leaves the judgement to
                  whoever reads the comparison.

    straight_wire_pct
                  Percentage of wires drawn as a single straight line, pin to
                  pin. Higher is better. The most legible thing a wire can be.

    collisions    Pairs of parallel wires running along the same track closely
                  enough to read as one line. A perpendicular crossing is at
                  least unambiguous -- you can still see which wire is which --
                  but two wires sharing a track are genuinely misleading, so
                  these matter more than crossings, not less. The segments of a
                  single wire meet end to end rather than overlapping, so a
                  wire is never counted against itself.

    crowded_junctions
                  Junction dots sitting close to a wire crossing. A dot means
                  "these wires connect" and a crossing means "these wires do
                  not"; put them within a few pixels of each other and the
                  reader cannot tell which is which, so the diagram stops
                  saying what it means.

    wire_length   Total length of the drawn wire segments.
    area          Bounding-box width * height of the placed nodes.

Typical use -- record a baseline, change the algorithm, then compare against it
(the same --seed and --limit must be used for both, or the two runs aren't
scoring the same circuits)::

    python -m rysy_dl.layout_metrics --db temp_out/circuits3.db --out baseline.json
    ...edit circuit_layout.py...
    python -m rysy_dl.layout_metrics --db temp_out/circuits3.db --compare baseline.json

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

import argparse
import json
import pickle
import random
import sqlite3
import statistics
import sys

import schemdraw
import schemdraw.elements as elm

from .circuit_layout import (_output_height, _wire_paths, assign_source_labels,
                             build_layout)
from .circuit_serialization import serializable_to_circuit
from .db import resolve_db_path

# Positions are in drawing units of order 1, so anything below this is
# floating-point noise rather than a real gap, bend, or overlap.
EPS = 1e-6

# Per-circuit fields that get averaged across the corpus, in report order.
# Two parallel wires closer than this read as a single line at normal size.
COLLISION_CLEARANCE = 0.1

# A junction dot nearer than this to a crossing is hard to tell apart from it.
JUNCTION_CLEARANCE = 0.3

REPORTED = (
    'collisions',
    'crowded_junctions',
    'crossings',
    'jogs',
    'worst_wire_jogs',
    'wire_length',
    'area',
    'height',
    'width',
)

# Reported alongside REPORTED, but these are percentages where higher is better.
HIGHER_IS_BETTER = ('straight_wire_pct', 'collision_free_pct', 'crossing_free_pct')


def drawn_segments(state):
    """The wire segments actually drawn, as ((x1, y1), (x2, y2)) pairs.

    Gate bodies and labels are their own element types, so every schemdraw
    Line on the drawing is a piece of wire.
    """
    segments = []
    for element in state.diagram.elements:
        if isinstance(element, elm.Line):
            start = element.absanchors['start']
            end = element.absanchors['end']
            segments.append(((float(start[0]), float(start[1])),
                             (float(end[0]), float(end[1]))))
    return segments


def count_crossings(segments):
    """Count places where a horizontal wire crosses through a vertical one.

    The routing is orthogonal, so every crossing is one horizontal segment
    meeting one vertical segment. The comparison is strict at both ends: a
    wire that *terminates* on another (a T-junction, i.e. a real electrical
    connection, drawn with a dot) touches at an endpoint rather than passing
    through, and is correctly not counted.
    """
    horizontals, verticals = [], []
    for (x1, y1), (x2, y2) in segments:
        if abs(y1 - y2) <= EPS and abs(x1 - x2) > EPS:
            horizontals.append((y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) <= EPS and abs(y1 - y2) > EPS:
            verticals.append((x1, min(y1, y2), max(y1, y2)))

    crossings = 0
    for y, x_low, x_high in horizontals:
        for x, y_low, y_high in verticals:
            if x_low + EPS < x < x_high - EPS and y_low + EPS < y < y_high - EPS:
                crossings += 1
    return crossings


def count_collisions(segments, clearance=COLLISION_CLEARANCE):
    """Count pairs of parallel wires sharing a track (see module docstring).

    Two segments collide when they run in the same direction, their extents
    genuinely overlap rather than merely touching end to end, and their tracks
    are nearer than `clearance`.
    """
    horizontals, verticals = [], []
    for (x1, y1), (x2, y2) in segments:
        if abs(y1 - y2) <= EPS and abs(x1 - x2) > EPS:
            horizontals.append((y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) <= EPS and abs(y1 - y2) > EPS:
            verticals.append((x1, min(y1, y2), max(y1, y2)))

    total = 0
    for group in (horizontals, verticals):
        for i in range(len(group)):
            track_i, low_i, high_i = group[i]
            for j in range(i + 1, len(group)):
                track_j, low_j, high_j = group[j]
                # `clearance - EPS`, not `clearance`: a wire deliberately placed
                # exactly one clearance away lands a float-ulp short of it.
                if (abs(track_i - track_j) < clearance - EPS
                        and min(high_i, high_j) - max(low_i, low_j) > EPS):
                    total += 1
    return total


def junction_points(state):
    """Where the drawing puts a junction dot -- a real electrical connection."""
    points = []
    for element in state.diagram.elements:
        if isinstance(element, elm.Dot):
            anchors = element.absanchors
            point = anchors.get('center', anchors.get('start'))
            points.append((float(point[0]), float(point[1])))
    return points


def crossing_points(segments):
    """Where a horizontal wire passes through a vertical one (see count_crossings)."""
    horizontals, verticals = [], []
    for (x1, y1), (x2, y2) in segments:
        if abs(y1 - y2) <= EPS and abs(x1 - x2) > EPS:
            horizontals.append((y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) <= EPS and abs(y1 - y2) > EPS:
            verticals.append((x1, min(y1, y2), max(y1, y2)))

    points = []
    for y, x_low, x_high in horizontals:
        for x, y_low, y_high in verticals:
            if x_low + EPS < x < x_high - EPS and y_low + EPS < y < y_high - EPS:
                points.append((x, y))
    return points


def count_crowded_junctions(state, segments, clearance=JUNCTION_CLEARANCE):
    """Count junction dots sitting within `clearance` of a crossing."""
    crossings = crossing_points(segments)
    total = 0
    for dot_x, dot_y in junction_points(state):
        for cross_x, cross_y in crossings:
            if (dot_x - cross_x) ** 2 + (dot_y - cross_y) ** 2 < clearance ** 2:
                total += 1
    return total


def _input_pin_height(state, consumer, pred):
    """Height of the specific input pin of `consumer` that `pred`'s wire lands on.

    Mirrors the pin assignment _route_stubs performs -- predecessors sorted by
    height, the highest into in1 -- so this agrees with what was drawn.
    """
    predecessors = sorted(state.netlist.predecessors(consumer),
                          key=lambda p: state.position[p], reverse=True)
    gate = state.gate_elements[consumer]
    return float(getattr(gate, f'in{predecessors.index(pred) + 1}')[1])


def count_jogs(state):
    """Return (total jogs, worst single wire, percent of wires drawn straight).

    Heights are taken at the real pins at each end. Dummy waypoints in the
    middle pass a wire straight through (their in and out anchors share a
    height), so their node position is the wire's height there.
    """
    total = worst = straight = wires = 0
    for path in _wire_paths(state.netlist):
        heights = [_output_height(state, path[0])]
        heights.extend(state.position[node][1] for node in path[1:-1])
        heights.append(_input_pin_height(state, path[-1], path[-2]))

        jogs = sum(1 for i in range(len(heights) - 1)
                   if abs(heights[i] - heights[i + 1]) > EPS)
        total += jogs
        worst = max(worst, jogs)
        wires += 1
        if jogs == 0:
            straight += 1

    return total, worst, (100.0 * straight / wires if wires else 0.0)


def layout_metrics(netlist):
    """Lay out `netlist` and return a dict of quality metrics (see module docstring)."""
    state = build_layout(netlist)
    segments = drawn_segments(state)
    jogs, worst, straight_pct = count_jogs(state)

    wire_length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in segments)
    xs = [p[0] for p in state.position.values()]
    ys = [p[1] for p in state.position.values()]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return {
        'collisions': count_collisions(segments),
        'crowded_junctions': count_crowded_junctions(state, segments),
        'crossings': count_crossings(segments),
        'jogs': jogs,
        'worst_wire_jogs': worst,
        'straight_wire_pct': straight_pct,
        'wire_length': wire_length,
        'area': width * height,
        'height': height,
        'width': width,
    }


def sample_circuits(db_path, limit, seed):
    """Yield (circuit_id, graph) for a deterministic sample of the database.

    Ids are sampled rather than rows so that a huge database isn't pulled into
    memory just to throw most of it away; the same (limit, seed) always selects
    the same circuits, which is what makes two runs comparable.
    """
    connection = sqlite3.connect(resolve_db_path(db_path))
    ids = [row[0] for row in connection.execute("SELECT id FROM circuits ORDER BY id")]
    if limit and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit))

    for circuit_id in ids:
        (blob,) = connection.execute(
            "SELECT graph FROM circuits WHERE id = ?", (circuit_id,)).fetchone()
        data = pickle.loads(blob)
        graph = serializable_to_circuit(data)
        if not data.get('labels'):
            assign_source_labels(graph)
        yield circuit_id, graph


def score_corpus(db_path, limit, seed):
    """Lay out a sample of the corpus and return (aggregate dict, failures list)."""
    per_circuit = []
    failures = []
    for circuit_id, graph in sample_circuits(db_path, limit, seed):
        try:
            per_circuit.append(layout_metrics(graph))
        except Exception as exc:  # a layout that raises is itself a result worth reporting
            failures.append({'id': circuit_id, 'error': repr(exc)})

    if not per_circuit:
        raise SystemExit("No circuits could be laid out; nothing to report.")

    aggregate = {field: statistics.mean(m[field] for m in per_circuit) for field in REPORTED}
    aggregate['straight_wire_pct'] = statistics.mean(m['straight_wire_pct'] for m in per_circuit)
    aggregate['collision_free_pct'] = (
        100.0 * sum(1 for m in per_circuit if m['collisions'] == 0) / len(per_circuit))
    aggregate['crossing_free_pct'] = (
        100.0 * sum(1 for m in per_circuit if m['crossings'] == 0) / len(per_circuit))
    aggregate['circuits'] = len(per_circuit)
    aggregate['failures'] = len(failures)
    return aggregate, failures


def _format_row(label, value, baseline=None, higher_is_better=False):
    if baseline is None:
        return f"  {label:<20} {value:>12.4f}"
    delta = value - baseline
    if abs(baseline) > EPS:
        change = f"{delta:+.4f} ({100.0 * delta / abs(baseline):+.1f}%)"
    else:
        change = f"{delta:+.4f}"
    if abs(delta) <= EPS:
        marker = ''
    else:
        better = (delta > 0) if higher_is_better else (delta < 0)
        marker = '  better' if better else '  WORSE'
    return f"  {label:<20} {value:>12.4f}   was {baseline:>10.4f}   {change}{marker}"


def report(aggregate, failures, baseline=None):
    print(f"circuits scored: {aggregate['circuits']}   failures: {aggregate['failures']}")
    print()
    for field in REPORTED:
        print(_format_row(field, aggregate[field], baseline.get(field) if baseline else None))
    for field in HIGHER_IS_BETTER:
        print(_format_row(field, aggregate[field],
                          baseline.get(field) if baseline else None,
                          higher_is_better=True))

    if failures:
        print(f"\n{len(failures)} failure(s); first few:")
        for failure in failures[:5]:
            print(f"  id={failure['id']}: {failure['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score circuit_layout's output over a corpus of circuits.")
    parser.add_argument('--db', default=None,
                        help="Circuit database to sample (default: the packaged database). "
                             "A '@name' path is resolved against the rysy_dl.db package.")
    parser.add_argument('--limit', type=int, default=300,
                        help="Number of circuits to score; 0 scores the whole database (default: 300).")
    parser.add_argument('--seed', type=int, default=20260820,
                        help="Seed for choosing which circuits to sample (default: 20260820). "
                             "Two runs must share --seed and --limit to be comparable.")
    parser.add_argument('--out', default=None, help="Write the aggregate scores to this JSON file.")
    parser.add_argument('--compare', default=None,
                        help="Read a JSON file written by --out and show the change against it.")
    args = parser.parse_args(argv)

    schemdraw.use('matplotlib')
    aggregate, failures = score_corpus(args.db, args.limit, args.seed)

    baseline = None
    if args.compare:
        with open(args.compare) as handle:
            baseline = json.load(handle)
        if baseline.get('circuits') != aggregate['circuits']:
            print(f"WARNING: baseline scored {baseline.get('circuits')} circuits, "
                  f"this run scored {aggregate['circuits']}; "
                  f"re-run with matching --limit/--seed for a meaningful comparison.\n",
                  file=sys.stderr)

    report(aggregate, failures, baseline)

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
