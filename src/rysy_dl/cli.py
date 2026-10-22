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
from .circuit_generator import generate_circuits
from .circuit_analyzer import circuit_to_expression
from .circuit_layout import layout_circuit, assign_source_labels
from .logging_config import PROGRESS

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


GENERATE_CIRCUITS_DEFAULTS = {
    'db_name': None,
    'dag_size': 7,
    'min_input_variables': 2,
    'max_input_variables': 7,
    'min_fan_in': 2,
    'max_fan_in': 6,
    'min_fan_out': 1,
    'max_fan_out': 5,
    'gate_assignment': 'random',
    'show_graphs': False,
    'output_dir': 'temp_out',
    'nauty_dir': None,
    'if_exists': None,
    'dry_run': False,
}

GENERATE_CIRCUITS_CHOICES = {
    'gate_assignment': ['random', 'all', 'and-only'],
}

GENERATE_CIRCUITS_TYPES = {
    'dag_size': int,
    'min_input_variables': int,
    'max_input_variables': int,
    'min_fan_in': int,
    'max_fan_in': int,
    'min_fan_out': int,
    'max_fan_out': int,
    'gate_assignment': str,
    'show_graphs': bool,
    'output_dir': str,
    'nauty_dir': str,
    'db_name': str,
    'if_exists': str,
    'dry_run': bool,
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

    if args.dry_run:
        try:
            generate_circuits(
                dag_size=args.dag_size,
                db=None,
                min_num_source=args.min_input_variables,
                max_num_source=args.max_input_variables,
                min_num_inputs=args.min_fan_in,
                max_num_inputs=args.max_fan_in,
                min_num_outputs=args.min_fan_out,
                max_num_outputs=args.max_fan_out,
                gate_assignment=args.gate_assignment,
                display_circuits=args.show_graphs,
                output_dir=args.output_dir,
                nauty_dir=args.nauty_dir,
                dry_run=True,
            )
        except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
            print(f"Error: nauty setup problem: {e}", file=sys.stderr)
            sys.exit(1)
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
                return
        if answer == 'o':
            os.remove(args.db_name)
            circuit_database.create(args.db_name)
    else:
        circuit_database.create(args.db_name)

    try:
        with circuit_database.CircuitDatabase(args.db_name) as db:
            generate_circuits(
                dag_size=args.dag_size,
                db=db,
                min_num_source=args.min_input_variables,
                max_num_source=args.max_input_variables,
                min_num_inputs=args.min_fan_in,
                max_num_inputs=args.max_fan_in,
                min_num_outputs=args.min_fan_out,
                max_num_outputs=args.max_fan_out,
                gate_assignment=args.gate_assignment,
                display_circuits=args.show_graphs,
                output_dir=args.output_dir,
                nauty_dir=args.nauty_dir
            )
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        print(f"Error: nauty setup problem: {e}", file=sys.stderr)
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


def cmd_view_circuits(args):
    import io
    import cairosvg  # imported lazily: only this many-graphs display loop needs the matplotlib-leak workaround.
    import schemdraw
    schemdraw.use('svg')  # Use pure-Python SVG backend to avoid matplotlib's memory leak

    try:
        with circuit_database.CircuitDatabase(args.db_name) as db:
            if args.id is not None:
                record = db.get_circuit(args.id)
                if record is None:
                    print(f"Error: no circuit with id {args.id} in '{args.db_name}'", file=sys.stderr)
                    sys.exit(1)
                records = [record]
            else:
                records = db.iterate()

            for record in records:
                netlist = record['graph']
                assign_source_labels(netlist)
                # circuit_generator_display.display_raw_graph(netlist)
                # The id is shown in the title so it can be passed to --id later to redisplay this same circuit.
                circuit_generator_display.set_figure_title(f"id={record['id']}: {circuit_to_expression(netlist)}")

                assign_source_labels(netlist)
                circuit_diagram = layout_circuit(netlist)
                png = io.BytesIO()
                cairosvg.svg2png(bytestring=circuit_diagram.get_imagedata(fmt='svg'), write_to=png)
                circuit_generator_display.display_circuit_from_bytestring(png.getvalue(), graph_width_in=0)
                if circuit_generator_display.wait_for_keypress():
                    print(" .")
                    break
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
                                 'as these options (e.g. "dag_size: 9", "min_fan_in: 3"). Lines starting with "#" '
                                 'are ignored. Command-line arguments override values from this file.')
    gc_parser.add_argument('--dag-size', type=int, default=None, help='Number of nodes in the generated DAGs (gates + inputs, excluding inserted NOT gates) (default: 7)')
    gc_parser.add_argument('--min-input-variables', type=int, default=None, help='Minimum number of inputs variables (source nodes) (default: 2)')
    gc_parser.add_argument('--max-input-variables', type=int, default=None, help='Maximum number of input variables (source nodes) (default: 7)')
    gc_parser.add_argument('--min-fan-in', type=int, default=None, help='Minimum number of inputs for each gate (default: 2)')
    gc_parser.add_argument('--max-fan-in', type=int, default=None, help='Maximum number of inputs for each gate (default: 6)')
    gc_parser.add_argument('--min-fan-out', type=int, default=None, help='Minimum number of outputs for each gate (default: 1)')
    gc_parser.add_argument('--max-fan-out', type=int, default=None, help='Maximum number of outputs for each gate (default: 5)')
    gc_parser.add_argument('--gate-assignment', type=str, default=None, choices=['random', 'all', 'and-only'], help='Gate assignment method: random or all (default: random)')
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
    view_circuits_parser.set_defaults(func=cmd_view_circuits)

    args = parser.parse_args()
    logging.basicConfig(level=LOG_LEVELS[args.log_level])
    args.func(args)


if __name__ == '__main__':
    main()
