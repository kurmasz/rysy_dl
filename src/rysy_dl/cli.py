"""
cli.py

Command-line interface for circuit_generator.

:copyright: 2026, Zachary Kurmas
"""

import argparse
import logging
import os
import pickle
import sys
from . import circuit_database
from . import circuit_generator_display
from .circuit_generator import generate_circuits_in_ranges, plan_dag_sizes
from .circuit_analyzer import circuit_to_expression
from .circuit_layout import layout_circuit
from .logging_config import PROGRESS
from rysy_dl import circuit_generator

#
# Logging setup
#

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'progress': PROGRESS,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


def _int_range(text):
    """argparse/config type for an inclusive range: "N" (just N) or "MIN-MAX". Returns (MIN, MAX)."""
    lo, sep, hi = str(text).partition('-')
    try:
        low, high = (int(lo), int(hi)) if sep else (int(lo), int(lo))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or a MIN-MAX range, got {text!r}")
    if low > high:
        raise argparse.ArgumentTypeError(f"range {text!r} is empty (MIN is larger than MAX)")
    return low, high


GENERATE_CIRCUITS_DEFAULTS = {
    'db_name': None,
    'num_inputs': (2, 4),
    'num_gates': (2, 4),
    'max_nodes': None,
    'min_fan_in': 2,
    'max_fan_in': 6,
    'min_fan_out': 1,
    'max_fan_out': 5,
    'gate_assignment': 'random',
    'not_gates': 'none',
    'show_graphs': False,
    'output_dir': 'temp_out',
    'nauty_dir': None,
    'if_exists': None,
    'dry_run': False,
    'status_interval': 10.0,
}

GENERATE_CIRCUITS_CHOICES = {
    'gate_assignment': ['random', 'all', 'and-only'],
    'not_gates': ['none', 'random', 'all'],
}

GENERATE_CIRCUITS_TYPES = {
    'num_inputs': _int_range,
    'num_gates': _int_range,
    'max_nodes': int,
    'min_fan_in': int,
    'max_fan_in': int,
    'min_fan_out': int,
    'max_fan_out': int,
    'gate_assignment': str,
    'not_gates': str,
    'show_graphs': bool,
    'output_dir': str,
    'nauty_dir': str,
    'db_name': str,
    'if_exists': str,
    'dry_run': bool,
    'status_interval': float,
}

TRUE_STRINGS = ('true', 'yes', 'on', '1')
FALSE_STRINGS = ('false', 'no', 'off', '0')


def parse_config_file(path):
    """Parse a config file of 'key: value' (or 'key = value') lines, one
    per line. Blank lines and lines starting with '#' are ignored."""
    config = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for sep in (':', '='):
                if sep in line:
                    key, _, value = line.partition(sep)
                    break
            else:
                print(f"Error in config file '{path}', line {lineno}: expected 'key: value' or 'key = value'", file=sys.stderr)
                sys.exit(1)
            config[key.strip().replace('-', '_')] = value.strip()
    return config


def _coerce(name, raw, expected_type, path):
    if expected_type is bool:
        low = raw.lower()
        if low in TRUE_STRINGS:
            return True
        if low in FALSE_STRINGS:
            return False
        print(f"Error in config file '{path}': '{name}' must be one of "
              f"{TRUE_STRINGS + FALSE_STRINGS}, got {raw!r}", file=sys.stderr)
        sys.exit(1)
    if expected_type is int:
        try:
            return int(raw)
        except ValueError:
            print(f"Error in config file '{path}': '{name}' must be an integer, got {raw!r}", file=sys.stderr)
            sys.exit(1)
    if expected_type is float:
        try:
            return float(raw)
        except ValueError:
            print(f"Error in config file '{path}': '{name}' must be a number, got {raw!r}", file=sys.stderr)
            sys.exit(1)
    if expected_type is _int_range:
        try:
            return _int_range(raw)
        except argparse.ArgumentTypeError as e:
            print(f"Error in config file '{path}': '{name}': {e}", file=sys.stderr)
            sys.exit(1)
    return raw


def apply_config_file(args, defaults, types, positional=()):
    """Fill in any argument left unset on the command line (i.e. still None)
    from a config file, falling back to `defaults` if the config file
    doesn't set it either. Command-line values always win."""
    config = {}
    if args.config is not None:
        try:
            config = parse_config_file(args.config)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error reading config file '{args.config}': {e}", file=sys.stderr)
            sys.exit(1)

        known = set(defaults) | set(positional)
        for name in config:
            if name not in known:
                print(f"Warning: unrecognized key '{name}' in config file '{args.config}'", file=sys.stderr)

    for name, default in defaults.items():
        if getattr(args, name) is None:
            if name in config:
                setattr(args, name, _coerce(name, config[name], types[name], args.config))
            else:
                setattr(args, name, default)

    for name in positional:
        if getattr(args, name) is None:
            if name in config:
                setattr(args, name, _coerce(name, config[name], types[name], args.config))
            else:
                print(f"Error: '{name}' must be given on the command line or in the config file", file=sys.stderr)
                sys.exit(1)


def cmd_generate_circuits(args):
    apply_config_file(args, GENERATE_CIRCUITS_DEFAULTS, GENERATE_CIRCUITS_TYPES)
    if not args.dry_run and args.db_name is None:
        print("Error: 'db_name' must be given on the command line or in the config file", file=sys.stderr)
        sys.exit(1)

    for name, choices in GENERATE_CIRCUITS_CHOICES.items():
        if getattr(args, name) not in choices:
            print(f"Error: '{name}' must be one of {choices}, got {getattr(args, name)!r}", file=sys.stderr)
            sys.exit(1)

    if args.if_exists is not None and args.if_exists not in ('insert', 'overwrite'):
        print(f"Error: 'if_exists' must be one of ['insert', 'overwrite'], got {args.if_exists!r}", file=sys.stderr)
        sys.exit(1)

    if args.status_interval <= 0:
        print(f"Error: 'status_interval' must be greater than 0, got {args.status_interval!r}", file=sys.stderr)
        sys.exit(1)

    for name in ('num_inputs', 'num_gates'):
        if getattr(args, name)[0] < 1:
            print(f"Error: '{name}' must be at least 1, got {getattr(args, name)!r}", file=sys.stderr)
            sys.exit(1)
    if args.max_nodes is not None and args.max_nodes < 2:
        print(f"Error: 'max_nodes' must be at least 2, got {args.max_nodes!r}", file=sys.stderr)
        sys.exit(1)

    # Check before creating or prompting about the database.
    if not plan_dag_sizes(args.num_inputs, args.num_gates, args.max_nodes):
        print(f"Error: no circuits have {args.num_inputs[0]}-{args.num_inputs[1]} inputs and "
              f"{args.num_gates[0]}-{args.num_gates[1]} gates"
              + (f" with at most {args.max_nodes} nodes" if args.max_nodes is not None else ""), file=sys.stderr)
        sys.exit(1)

    run_options = dict(
        num_sources=args.num_inputs,
        num_gates=args.num_gates,
        max_nodes=args.max_nodes,
        min_num_inputs=args.min_fan_in,
        max_num_inputs=args.max_fan_in,
        min_num_outputs=args.min_fan_out,
        max_num_outputs=args.max_fan_out,
        gate_assignment=args.gate_assignment,
        not_gates=args.not_gates,
        display_circuits=args.show_graphs,
        output_dir=args.output_dir,
        nauty_dir=args.nauty_dir,
        status_interval=args.status_interval,
    )

    if args.dry_run:
        _run_generate(db=None, dry_run=True, **run_options)
        return

    if os.path.exists(args.db_name):
        if args.if_exists == 'insert':
            answer = 'i'
        elif args.if_exists == 'overwrite':
            answer = 'o'
        else:
            answer = None
            while answer not in ('i', 'o', 'q'):
                answer = input(
                    f"Database '{args.db_name}' already exists. "
                    f"(I)nsert into it, (O)verwrite it, or (Q)uit? "
                ).strip().lower()[:1]
            if answer == 'q':
                print("Quitting: database already exists and user chose not to proceed.")
                # Exit non-zero so scripts that run several generate-circuits commands can stop.
                sys.exit(1)
        if answer == 'o':
            os.remove(args.db_name)
            circuit_database.create(args.db_name)
    else:
        circuit_database.create(args.db_name)

    with circuit_database.CircuitDatabase(args.db_name) as db:
        _run_generate(db=db, **run_options)


def _run_generate(**options):
    try:
        generate_circuits_in_ranges(**options)
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        print(f"Error: nauty setup problem: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_layout(args):
    try:
        with open(args.pkl_file, 'rb') as f:
            graph = pickle.load(f)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    import schemdraw
    schemdraw.use('matplotlib')
    print(circuit_to_expression(graph))
    diagram = layout_circuit(graph)
    diagram.draw()


# view-circuits options that restrict which circuits are shown: one per keyword of
# CircuitDatabase.get_random_entry / iterate_entries, mapped to its --help description.
VIEW_CIRCUITS_FILTERS = {
    'num_gates': 'number of gates (non-input nodes, including NOT gates)',
    'num_inputs': 'number of input variables',
    'depth': 'length of the longest path from an input to the output',
    'max_indegree': 'largest fan-in of any gate',
    'max_gate_outdegree': 'largest fan-out of any gate',
    'max_input_outdegree': 'largest fan-out of any input variable',
    'gate_type_mask': 'gate_type_mask bitmask (see circuit_database.GATE_BITS)',
    'max_shared_subcircuit_depth': 'depth of the deepest gate whose output feeds more than one gate (0 if none)',
}


def _int_or_range(text):
    """argparse type for a view-circuits filter: an int (exact match) or MIN-MAX (inclusive range)."""
    lo, sep, hi = text.partition('-')
    try:
        return (int(lo), int(hi)) if sep else int(lo)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or a MIN-MAX range, got {text!r}")


def cmd_view_circuits(args):
    filters = {name: getattr(args, name) for name in VIEW_CIRCUITS_FILTERS}
    if args.id is not None and any(value is not None for value in filters.values()):
        print("Error: --id can't be combined with filter options", file=sys.stderr)
        sys.exit(1)

    try:
        with circuit_database.CircuitDatabase(args.db_name) as db:
            if args.id is not None:
                record = db.get_entry(args.id)
                if record is None:
                    print(f"Error: no circuit with id {args.id} in '{args.db_name}'", file=sys.stderr)
                    sys.exit(1)
                records = [record]
            else:
                records = db.iterate_entries(**filters)

            num_shown = 0
            for record in records:
                num_shown += 1
                netlist = record['graph']

                # circuit_generator.assign_gates_randomly(netlist)
                # The id is shown in the title so it can be passed to --id later to redisplay this same circuit.
                if circuit_generator_display.show_netlist(netlist, title_prefix=f"id={record['id']}"):
                    print(" .")
                    break

            if num_shown == 0:
                print(f"No circuits in '{db.db_path}' match the given filters")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='warning',
        choices=['debug', 'info', 'progress', 'warning', 'error', 'critical'],
        help='Logging level (default: warning)',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    gc_parser = subparsers.add_parser('generate-circuits', help='Exhaustively generate combinatorial circuits')
    gc_parser.add_argument('db_name', type=str, nargs='?', default=None,
                            help='Path to the sqlite database where generated circuits are recorded. '
                                 'If it already exists, you will be prompted to insert, overwrite, or quit. '
                                 'May instead be given as "db_name" in --config. Not required with --dry-run.')
    gc_parser.add_argument('--config', type=str, default=None,
                            help='Path to a file of "key: value" parameters, one per line, using the same names '
                                 'as these options (e.g. "num_gates: 3-5", "min_fan_in: 3"). Lines starting with "#" '
                                 'are ignored. Command-line arguments override values from this file.')
    gc_parser.add_argument('--num-inputs', type=_int_range, default=None, metavar='N|MIN-MAX',
                           help='Number of input variables (source nodes), e.g. 3 or 2-5 (default: 2-4)')
    gc_parser.add_argument('--num-gates', type=_int_range, default=None, metavar='N|MIN-MAX',
                           help='Number of gates, including the output gate but not inserted NOT gates, '
                                'e.g. 4 or 3-5 (default: 3-4)')
    gc_parser.add_argument('--max-nodes', type=int, default=None,
                           help='Leave out circuits with more than this many inputs + gates. The circuit size, '
                                'not the number of inputs, is what makes enumeration expensive (default: no limit)')
    gc_parser.add_argument('--min-fan-in', type=int, default=None, help='Minimum number of inputs for each gate (default: 2)')
    gc_parser.add_argument('--max-fan-in', type=int, default=None, help='Maximum number of inputs for each gate (default: 6)')
    gc_parser.add_argument('--min-fan-out', type=int, default=None, help='Minimum number of outputs for each gate (default: 1)')
    gc_parser.add_argument('--max-fan-out', type=int, default=None, help='Maximum number of outputs for each gate (default: 5)')
    gc_parser.add_argument('--gate-assignment', type=str, default=None, choices=['random', 'all', 'and-only'], help='Gate assignment method: random or all (default: random)')
    gc_parser.add_argument('--not-gates', type=str, default=None, choices=['none', 'random', 'all'],
                            help='NOT gates on edges leaving the inputs: none, random (randomly insert them), '
                                 'or all (every reduced NOT pattern for each gate assignment; patterns that only '
                                 'swap an input for its complement are skipped) (default: none)')
    gc_parser.add_argument('--show-graphs', action=argparse.BooleanOptionalAction, default=None, help='Show generated graphs (default: False)')
    gc_parser.add_argument('--generate-image', action=argparse.BooleanOptionalAction, default=None, help='Write a PNG image for each generated circuit (default: True)')
    gc_parser.add_argument('--output-dir', type=str, default=None, help='Directory to save generated graphs and expressions (default: temp_out)')
    gc_parser.add_argument('--nauty-dir', type=str, default=None, help='Path to directory containing nauty binaries. Defaults to $NAUTY env var, then PATH (geng/directg or nauty-geng/nauty-directg)')
    gc_parser.add_argument('--if-exists', type=str, default=None, choices=['insert', 'overwrite'],
                            help='What to do if db_name already exists: (i)nsert into it or (o)verwrite it. '
                                 'If not given, you will be prompted interactively.')
    gc_parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=None,
                            help='Run generation without writing to the database. The database file is '
                                 'never created, opened, or modified (default: False)')
    gc_parser.add_argument('--status-interval', type=float, default=None,
                            help='Seconds between progress updates (default: 10)')
    gc_parser.set_defaults(func=cmd_generate_circuits)

    layout_parser = subparsers.add_parser('layout', help='Render a circuit from a .pkl file')
    layout_parser.add_argument('pkl_file', help='Path to the .pkl file containing the circuit graph')
    layout_parser.set_defaults(func=cmd_layout)

    view_circuits_parser = subparsers.add_parser('view-circuits', help='Render every circuit in a database, one at a time')
    view_circuits_parser.add_argument('db_name', nargs='?', default=None,
                                       help='Path to the sqlite database of circuits. '
                                            'Defaults to the database packaged with rysy_dl.')
    view_circuits_parser.add_argument('--id', type=int, default=None,
                                       help='Only display the circuit with this id, instead of stepping through '
                                            'the whole database. The id is shown in the title of each circuit '
                                            'as it is displayed.')
    for name, description in VIEW_CIRCUITS_FILTERS.items():
        view_circuits_parser.add_argument(f"--{name.replace('_', '-')}", type=_int_or_range, default=None,
                                           metavar='N|MIN-MAX',
                                           help=f'Only display circuits whose {description} is N, '
                                                f'or between MIN and MAX inclusive.')
    view_circuits_parser.set_defaults(func=cmd_view_circuits)

    args = parser.parse_args()
    logging.basicConfig(level=LOG_LEVELS[args.log_level])
    args.func(args)


if __name__ == '__main__':
    main()
