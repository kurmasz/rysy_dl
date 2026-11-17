"""
layout_harness.py

Run `circuit_layout` over a fixed corpus of circuits and record, per circuit,
exactly what was drawn, whether it is valid, and how good it looks -- so a
change to the layout code can be judged against the run before it.

`layout_metrics` reports corpus-wide averages. That says whether a change
helped overall, but not *which* drawings changed, and an average can hide a
handful of circuits that got much worse or stopped being correct. This records
every circuit separately, which answers three questions the averages can't:

  * Did a refactor change anything at all? (It shouldn't have.)
  * Did any drawing become invalid? (See `layout_validity`.)
  * Which circuits got better or worse, and what do they look like now?

Typical use::

    python -m rysy_dl.layout_harness snapshot temp_out_layout/before
    ...edit circuit_layout.py...
    python -m rysy_dl.layout_harness snapshot temp_out_layout/after
    python -m rysy_dl.layout_harness compare temp_out_layout/before temp_out_layout/after

`compare` prints a summary and writes `compare.html` into the second snapshot:
the changed circuits side by side, most-regressed first, for judging by eye.
For a change that is meant to leave the drawings alone, add
`--expect-identical`, which makes any difference a failure.

The corpus is a seeded sample of a circuit database. Each sampled circuit is
laid out as stored and again with gates assigned at random and NOT gates
inserted (as `Circuit.randomize` does for a real exercise), since the stored
circuits alone are AND-only and would never exercise a NOT, an XOR, or a mix
of gate widths. The `.pkl` examples in `test_graphs/` are included when that
directory is present.

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

import argparse
import base64
import collections
import glob
import html
import json
import os
import pickle
import random
import statistics
import sys

import schemdraw

from .circuit import Circuit
from .circuit_analyzer import circuit_to_expression
from .circuit_database import CircuitDatabase
from .circuit_layout import assign_source_labels, build_layout, sources_top_to_bottom
from .layout_metrics import HIGHER_IS_BETTER, REPORTED, metrics_from_state
from .layout_validity import (HARD_KINDS, canonical_geometry, check_geometry,
                              extract_geometry)

DEFAULT_LIMIT = 300
DEFAULT_SEED = 20260820
DEFAULT_VARIANTS = 2
EXAMPLES_DIR = 'test_graphs'

# Coordinates in a snapshot are rounded to 6 places; allow for a last-place
# difference when the same value was reached by a different order of sums.
GEOMETRY_TOLERANCE = 2e-6

METRIC_FIELDS = REPORTED + ('straight_wire_pct',)


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

def corpus(db_path=None, limit=DEFAULT_LIMIT, seed=DEFAULT_SEED,
           variants=DEFAULT_VARIANTS, examples_dir=EXAMPLES_DIR):
    """Yield (key, netlist) for every circuit in the corpus.

    The same arguments always yield the same circuits under the same keys,
    which is what lets two snapshots be compared circuit by circuit. Gate
    randomisation draws from the global `random` module, so it is reseeded per
    circuit: each circuit's variants then don't depend on which others were
    sampled before it.
    """
    if examples_dir and os.path.isdir(examples_dir):
        for path in sorted(glob.glob(os.path.join(examples_dir, '*.p*'))):
            with open(path, 'rb') as handle:
                yield f'file:{os.path.basename(path)}', pickle.load(handle)

    with CircuitDatabase(db_path, read_only=True) as db:
        ids = [row['id'] for row in db._open_conn().execute("SELECT id FROM circuits ORDER BY id")]
        if limit and limit < len(ids):
            ids = sorted(random.Random(seed).sample(ids, limit))

        for circuit_id in ids:
            stored = db.get_entry(circuit_id)['graph']
            for variant in range(variants + 1):
                netlist = stored.copy()
                if variant:
                    random.seed(f'{seed}:{circuit_id}:{variant}')
                    has_nots = any(gate == 'NOT' for _, gate in netlist.nodes(data='gate'))
                    Circuit(netlist).randomize(source_nots=not has_nots)
                # Label the way the exercise generator does: top to bottom.
                assign_source_labels(netlist, source_order=sources_top_to_bottom(netlist))
                yield f'db:{circuit_id}:{"stored" if not variant else f"random{variant}"}', netlist


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------

def examine(netlist):
    """Lay out one circuit and return (record, svg bytes or None).

    A layout that raises is recorded as an 'error' violation rather than
    propagated: a crash on one circuit is a result, not a reason to learn
    nothing about the rest.
    """
    record = {'expression': circuit_to_expression(netlist)}
    try:
        state = build_layout(netlist)
        geometry = extract_geometry(state)
        record['geometry'] = canonical_geometry(netlist, geometry)
        record['violations'] = [[v.kind, v.detail] for v in check_geometry(netlist, geometry)]
        record['metrics'] = metrics_from_state(state)
        return record, state.diagram.get_imagedata('svg')
    except Exception as exc:
        record.update(geometry=None, metrics=None, violations=[['error', repr(exc)]])
        return record, None


def take_snapshot(directory, images=True, **corpus_options):
    """Examine the whole corpus and write the results under `directory`:
    `snapshot.json`, plus one SVG per circuit in `svg/` if `images`."""
    schemdraw.use('svg')
    os.makedirs(directory, exist_ok=True)
    if images:
        os.makedirs(os.path.join(directory, 'svg'), exist_ok=True)

    circuits = {}
    for key, netlist in corpus(**corpus_options):
        record, svg = examine(netlist)
        circuits[key] = record
        if images and svg is not None:
            with open(os.path.join(directory, 'svg', _image_name(key)), 'wb') as handle:
                handle.write(svg)

    snapshot = {'corpus': {k: v for k, v in corpus_options.items()}, 'circuits': circuits}
    with open(os.path.join(directory, 'snapshot.json'), 'w') as handle:
        json.dump(snapshot, handle, sort_keys=True)
    return snapshot


def load_snapshot(directory):
    with open(os.path.join(directory, 'snapshot.json')) as handle:
        return json.load(handle)


def _image_name(key):
    return key.replace(':', '_') + '.svg'


def violation_counts(snapshot):
    """{kind: (circuits affected, total violations)} over a snapshot."""
    affected, total = collections.Counter(), collections.Counter()
    for record in snapshot['circuits'].values():
        kinds = [kind for kind, _ in record['violations']]
        total.update(kinds)
        affected.update(set(kinds))
    return {kind: (affected[kind], total[kind]) for kind in total}


def _mean_metrics(snapshot):
    scored = [r['metrics'] for r in snapshot['circuits'].values() if r['metrics']]
    return {name: statistics.mean(m[name] for m in scored) for name in METRIC_FIELDS} if scored else {}


def print_summary(snapshot):
    circuits = snapshot['circuits']
    print(f"circuits: {len(circuits)}")
    counts = violation_counts(snapshot)
    if not counts:
        print("violations: none")
    for kind in sorted(counts, key=lambda k: (k not in HARD_KINDS, k)):
        affected, total = counts[kind]
        severity = 'INVALID' if kind in HARD_KINDS else 'flaw'
        example = next((f"{key}: {detail}" for key, record in circuits.items()
                        for k, detail in record['violations'] if k == kind), '')
        print(f"  {kind:<18} {severity:<8} {affected:>4} circuits, {total:>4} total   e.g. {example}")
    print()
    for name, value in _mean_metrics(snapshot).items():
        print(f"  {name:<20} {value:>12.4f}")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def _numbers(value):
    """Flatten nested JSON into (structure, numbers) so two geometries can be
    compared exactly in shape and to a tolerance in coordinates."""
    structure, numbers = [], []

    def walk(item):
        if isinstance(item, dict):
            for key in sorted(item):
                structure.append(key)
                walk(item[key])
        elif isinstance(item, list):
            structure.append(len(item))
            for child in item:
                walk(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            numbers.append(item)
        else:
            structure.append(item)

    walk(value)
    return structure, numbers


def same_geometry(a, b):
    if a is None or b is None:
        return a is b
    structure_a, numbers_a = _numbers(a)
    structure_b, numbers_b = _numbers(b)
    return (structure_a == structure_b and len(numbers_a) == len(numbers_b)
            and all(abs(x - y) <= GEOMETRY_TOLERANCE for x, y in zip(numbers_a, numbers_b)))


def _badness(before, after):
    """How much worse `after` is than `before`, as a tuple that sorts the most
    regressed circuit first: validity first, then the flaws a reader trips on,
    then the cosmetic ones."""
    def hard(record):
        return sum(1 for kind, _ in record['violations'] if kind in HARD_KINDS)

    def soft(record):
        return sum(1 for kind, _ in record['violations'] if kind not in HARD_KINDS)

    def metric(record, name):
        return record['metrics'][name] if record['metrics'] else 0

    return (hard(after) - hard(before),
            soft(after) - soft(before),
            metric(after, 'collisions') - metric(before, 'collisions'),
            metric(after, 'crowded_junctions') - metric(before, 'crowded_junctions'),
            metric(after, 'crossings') - metric(before, 'crossings'),
            metric(after, 'jogs') - metric(before, 'jogs'),
            metric(after, 'wire_length') - metric(before, 'wire_length'))


def compare(before, after):
    """Compare two snapshots. Returns a dict describing every difference."""
    keys_before, keys_after = set(before['circuits']), set(after['circuits'])
    shared = sorted(keys_before & keys_after)

    changed = [key for key in shared
               if not same_geometry(before['circuits'][key]['geometry'],
                                    after['circuits'][key]['geometry'])]
    changed.sort(key=lambda key: _badness(before['circuits'][key], after['circuits'][key]),
                 reverse=True)

    new_violations = []
    for key in shared:
        old = collections.Counter(kind for kind, _ in before['circuits'][key]['violations'])
        for kind, detail in after['circuits'][key]['violations']:
            if old[kind] > 0:
                old[kind] -= 1
            else:
                new_violations.append((key, kind, detail))

    per_metric = {}
    for name in METRIC_FIELDS:
        better = worse = 0
        for key in shared:
            a, b = before['circuits'][key]['metrics'], after['circuits'][key]['metrics']
            if not a or not b or abs(a[name] - b[name]) <= 1e-6:
                continue
            improved = (b[name] > a[name]) if name in HIGHER_IS_BETTER else (b[name] < a[name])
            better += improved
            worse += not improved
        per_metric[name] = (better, worse)

    return {
        'only_before': sorted(keys_before - keys_after),
        'only_after': sorted(keys_after - keys_before),
        'shared': shared,
        'changed': changed,
        'new_violations': new_violations,
        'new_hard_violations': [v for v in new_violations if v[1] in HARD_KINDS],
        'per_metric': per_metric,
    }


def print_comparison(before, after, result):
    shared, changed = result['shared'], result['changed']
    print(f"circuits compared: {len(shared)}")
    if result['only_before'] or result['only_after']:
        print(f"  WARNING: the snapshots cover different circuits "
              f"({len(result['only_before'])} only in the first, "
              f"{len(result['only_after'])} only in the second); "
              f"take both with the same corpus options.")
    print(f"drawings identical: {len(shared) - len(changed)}    changed: {len(changed)}")

    print("\nviolations (circuits affected):")
    counts_before, counts_after = violation_counts(before), violation_counts(after)
    kinds = sorted(set(counts_before) | set(counts_after), key=lambda k: (k not in HARD_KINDS, k))
    if not kinds:
        print("  none in either snapshot")
    for kind in kinds:
        was, now = counts_before.get(kind, (0, 0))[0], counts_after.get(kind, (0, 0))[0]
        severity = 'INVALID' if kind in HARD_KINDS else 'flaw'
        marker = '' if was == now else ('  better' if now < was else '  WORSE')
        print(f"  {kind:<18} {severity:<8} {now:>5}   was {was:>5}{marker}")
    if result['new_violations']:
        print(f"\n{len(result['new_violations'])} new violation(s); first few:")
        for key, kind, detail in sorted(result['new_violations'],
                                        key=lambda v: v[1] not in HARD_KINDS)[:8]:
            print(f"  {key}: {kind}: {detail}")

    means_before, means_after = _mean_metrics(before), _mean_metrics(after)
    if means_before and means_after:
        print(f"\n  {'metric (mean)':<20} {'now':>10} {'was':>10} {'change':>9}"
              f"   circuits better / worse")
        for name in METRIC_FIELDS:
            was, now = means_before[name], means_after[name]
            percent = f"{100.0 * (now - was) / abs(was):+.1f}%" if abs(was) > 1e-6 else ''
            better, worse = result['per_metric'][name]
            if abs(now - was) <= 1e-6:
                marker = ''
            else:
                improved = (now > was) if name in HIGHER_IS_BETTER else (now < was)
                marker = '  better' if improved else '  WORSE'
            print(f"  {name:<20} {now:>10.4f} {was:>10.4f} {percent:>9}   "
                  f"{better:>6} / {worse:<6}{marker}")


def write_gallery(before_dir, after_dir, before, after, result, limit=40):
    """Write `compare.html` into `after_dir`: the changed circuits, before and
    after side by side, most-regressed first. Returns the path, or None if
    either snapshot was taken without images."""
    if not all(os.path.isdir(os.path.join(d, 'svg')) for d in (before_dir, after_dir)):
        return None

    def cell(directory, key, record):
        # Embedded rather than linked, so the page is one file that still
        # shows its drawings when moved, mailed, or opened in a sandboxed viewer.
        path = os.path.join(directory, 'svg', _image_name(key))
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                encoded = base64.b64encode(handle.read()).decode('ascii')
            image = f'<img src="data:image/svg+xml;base64,{encoded}">'
        else:
            image = '<p class="bad">no drawing</p>'
        notes = ''.join(
            f'<li class="{"bad" if kind in HARD_KINDS else "flaw"}">{html.escape(kind)}: '
            f'{html.escape(detail)}</li>' for kind, detail in record['violations'])
        metrics = record['metrics'] or {}
        numbers = ' &middot; '.join(
            f'{name} {metrics[name]:g}' for name in ('crossings', 'jogs', 'collisions',
                                                    'crowded_junctions') if name in metrics)
        return f'<td>{image}<p class="num">{numbers}</p><ul>{notes}</ul></td>'

    rows = []
    for key in result['changed'][:limit]:
        old, new = before['circuits'][key], after['circuits'][key]
        rows.append(
            f'<tr><th colspan="2">{html.escape(key)} &nbsp; '
            f'<code>{html.escape(new["expression"])}</code></th></tr>'
            f'<tr>{cell(before_dir, key, old)}{cell(after_dir, key, new)}</tr>')

    shown = min(limit, len(result['changed']))
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Layout comparison</title>
<style>
  body {{ font: 14px system-ui, sans-serif; margin: 24px; color: #222; background: #fff; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; padding: 18px 8px 4px; border-top: 1px solid #ccc; font-weight: 600; }}
  td {{ width: 50%; vertical-align: top; padding: 8px; }}
  img {{ max-width: 100%; background: #fff; }}
  .num {{ color: #555; margin: 4px 0; }}
  ul {{ margin: 0; padding-left: 18px; }}
  .bad {{ color: #b00020; font-weight: 600; }}
  .flaw {{ color: #8a5a00; }}
  code {{ font-weight: 400; color: #555; }}
</style>
<h1>Layout comparison</h1>
<p>{len(result['changed'])} of {len(result['shared'])} drawings changed; showing {shown},
most-regressed first. Left: <code>{html.escape(before_dir)}</code>.
Right: <code>{html.escape(after_dir)}</code>.</p>
<table>
<tr><th>Before</th><th>After</th></tr>
{''.join(rows)}
</table>
"""
    path = os.path.join(after_dir, 'compare.html')
    with open(path, 'w') as handle:
        handle.write(page)
    return path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record what circuit_layout draws for a corpus of circuits, "
                    "and compare one recording against another.")
    commands = parser.add_subparsers(dest='command', required=True)

    snapshot_parser = commands.add_parser(
        'snapshot', help="Lay out the corpus and record the results in a directory.")
    snapshot_parser.add_argument('directory')
    snapshot_parser.add_argument('--db', default=None,
                                 help="Circuit database to sample (default: the packaged one).")
    snapshot_parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                                 help=f"Circuits to sample; 0 takes them all (default: {DEFAULT_LIMIT}).")
    snapshot_parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    snapshot_parser.add_argument('--variants', type=int, default=DEFAULT_VARIANTS,
                                 help="Randomly gated versions of each sampled circuit, in addition "
                                      f"to the stored one (default: {DEFAULT_VARIANTS}).")
    snapshot_parser.add_argument('--no-images', action='store_true',
                                 help="Skip writing an SVG per circuit.")

    compare_parser = commands.add_parser(
        'compare', help="Compare two snapshot directories, the earlier one first.")
    compare_parser.add_argument('before')
    compare_parser.add_argument('after')
    compare_parser.add_argument('--show', type=int, default=40,
                                help="Changed circuits to put in compare.html (default: 40).")
    compare_parser.add_argument('--expect-identical', action='store_true',
                                help="Fail if any drawing differs (for refactors).")

    args = parser.parse_args(argv)

    if args.command == 'snapshot':
        snapshot = take_snapshot(args.directory, images=not args.no_images, db_path=args.db,
                                 limit=args.limit, seed=args.seed, variants=args.variants)
        print_summary(snapshot)
        print(f"\nwrote {os.path.join(args.directory, 'snapshot.json')}")
        return 0

    before, after = load_snapshot(args.before), load_snapshot(args.after)
    result = compare(before, after)
    print_comparison(before, after, result)
    gallery = write_gallery(args.before, args.after, before, after, result, limit=args.show)
    if gallery and result['changed']:
        print(f"\nside-by-side drawings: {gallery}")

    if result['new_hard_violations']:
        print(f"\nFAIL: {len(result['new_hard_violations'])} drawing(s) became invalid.",
              file=sys.stderr)
        return 1
    if args.expect_identical and (result['changed'] or result['only_before'] or result['only_after']):
        print(f"\nFAIL: expected identical drawings, but {len(result['changed'])} changed.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
