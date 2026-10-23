"""
Unit tests for cli.py.

Covers the config-file merging logic (parse_config_file, _coerce,
apply_config_file) and the generate-circuits / layout subcommands.
Heavy third-party dependencies (schemdraw, matplotlib, ...) are mocked out
before import, same as test_circuit_generator.py, since cli.py imports
modules that depend on them. networkx is deliberately NOT mocked here (see
test_circuit_generator.py's note on this) -- mocking it into sys.modules
would leak into any other test file that needs a real nx.DiGraph, since
sys.modules is shared process-wide for the whole test session.
"""

import argparse
import os
import sys
from unittest.mock import MagicMock

import pytest

# ── mock heavy dependencies before importing the module ──────────────────────
for _mod in ['cairosvg', 'graph_image_gen', 'schemdraw',
             'schemdraw.elements', 'schemdraw.logic', 'schemdraw.elements.elements', 'schemdraw.segments',
             'matplotlib', 'matplotlib.pyplot', 'matplotlib.widgets', 'matplotlib.image']:
    if _mod not in sys.modules:
        _mock = MagicMock()
        _mock.__all__ = []
        sys.modules[_mod] = _mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rysy_dl import cli  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

class FakeDB:
    """Stand-in for circuit_database.CircuitDatabase, used as a context manager."""
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def make_args(**overrides):
    """Build a Namespace with every generate-circuits attribute defaulted to
    None (as argparse produces when nothing is given on the command line),
    then apply `overrides`."""
    defaults = {name: None for name in cli.GENERATE_CIRCUITS_DEFAULTS}
    defaults['db_name'] = None
    defaults['config'] = None
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ── parse_config_file ─────────────────────────────────────────────────────────

def test_parse_config_file_colon_and_equals(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: 9\nmin_fan_in = 3\n")
    assert cli.parse_config_file(str(cfg)) == {'dag_size': '9', 'min_fan_in': '3'}


def test_parse_config_file_ignores_comments_and_blank_lines(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("# a comment\n\ndag_size: 9\n   \n# another\n")
    assert cli.parse_config_file(str(cfg)) == {'dag_size': '9'}


def test_parse_config_file_normalizes_dashes_to_underscores(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("min-fan-in: 3\n")
    assert cli.parse_config_file(str(cfg)) == {'min_fan_in': '3'}


def test_parse_config_file_malformed_line_exits(tmp_path, capsys):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("this line has no separator\n")
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_config_file(str(cfg))
    assert exc_info.value.code == 1
    assert "expected 'key: value'" in capsys.readouterr().err


def test_parse_config_file_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        cli.parse_config_file("/no/such/file.cfg")


# ── _coerce ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('raw', ['true', 'Yes', 'ON', '1'])
def test_coerce_bool_true_variants(raw):
    assert cli._coerce('show_graphs', raw, bool, 'params.cfg') is True


@pytest.mark.parametrize('raw', ['false', 'No', 'OFF', '0'])
def test_coerce_bool_false_variants(raw):
    assert cli._coerce('show_graphs', raw, bool, 'params.cfg') is False


def test_coerce_bool_invalid_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli._coerce('show_graphs', 'maybe', bool, 'params.cfg')
    assert exc_info.value.code == 1
    assert "must be one of" in capsys.readouterr().err


def test_coerce_int_valid():
    assert cli._coerce('dag_size', '9', int, 'params.cfg') == 9


def test_coerce_int_invalid_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli._coerce('dag_size', 'nine', int, 'params.cfg')
    assert exc_info.value.code == 1
    assert "must be an integer" in capsys.readouterr().err


def test_coerce_str_passthrough():
    assert cli._coerce('gate_assignment', 'all', str, 'params.cfg') == 'all'


# ── apply_config_file ─────────────────────────────────────────────────────

def test_apply_config_file_cli_overrides_file(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: 9\n")
    args = make_args(config=str(cfg), dag_size=4, db_name='foo.db')

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert args.dag_size == 4


def test_apply_config_file_fills_from_file_when_cli_unset(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: 9\nmin_fan_in: 3\n")
    args = make_args(config=str(cfg), db_name='foo.db')

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert args.dag_size == 9
    assert args.min_fan_in == 3


def test_apply_config_file_falls_back_to_hardcoded_defaults(tmp_path):
    # Called the way cmd_generate_circuits calls it: db_name is an ordinary (optional) default,
    # not a required positional, so it too falls back to its default when left unset.
    args = make_args()  # no --config, nothing set on the command line

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES)

    for name, default in cli.GENERATE_CIRCUITS_DEFAULTS.items():
        assert getattr(args, name) == default


def test_apply_config_file_missing_positional_exits(capsys):
    args = make_args()  # no db_name, no config

    with pytest.raises(SystemExit) as exc_info:
        cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert exc_info.value.code == 1
    assert "db_name" in capsys.readouterr().err


def test_apply_config_file_positional_from_config(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("db_name: circuits.db\n")
    args = make_args(config=str(cfg))

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert args.db_name == 'circuits.db'


def test_apply_config_file_warns_on_unrecognized_key(tmp_path, capsys):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("gate_assingment: all\n")  # typo
    args = make_args(config=str(cfg), db_name='foo.db')

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert "Warning: unrecognized key 'gate_assingment'" in capsys.readouterr().err


def test_apply_config_file_type_coercion(tmp_path):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: 9\nshow_graphs: true\n")
    args = make_args(config=str(cfg), db_name='foo.db')

    cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert args.dag_size == 9 and isinstance(args.dag_size, int)
    assert args.show_graphs is True


def test_apply_config_file_bad_value_in_file_exits(tmp_path, capsys):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: nine\n")
    args = make_args(config=str(cfg), db_name='foo.db')

    with pytest.raises(SystemExit) as exc_info:
        cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert exc_info.value.code == 1
    assert "must be an integer" in capsys.readouterr().err


def test_apply_config_file_missing_config_file_exits(tmp_path, capsys):
    args = make_args(config=str(tmp_path / "missing.cfg"), db_name='foo.db')

    with pytest.raises(SystemExit) as exc_info:
        cli.apply_config_file(args, cli.GENERATE_CIRCUITS_DEFAULTS, cli.GENERATE_CIRCUITS_TYPES, positional=('db_name',))

    assert exc_info.value.code == 1
    assert "Error reading config file" in capsys.readouterr().err


def test_defaults_and_types_are_consistent():
    """Every default parameter (plus db_name) must have a known type, or
    apply_config_file/_coerce will KeyError when a config file sets it."""
    assert set(cli.GENERATE_CIRCUITS_DEFAULTS) <= set(cli.GENERATE_CIRCUITS_TYPES)
    assert 'db_name' in cli.GENERATE_CIRCUITS_TYPES


# ── cmd_generate_circuits ──────────────────────────────────────────────────

def test_cmd_generate_circuits_rejects_bad_gate_assignment(tmp_path, capsys):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("gate_assignment: bogus\n")
    args = make_args(config=str(cfg), db_name=str(tmp_path / 'circuits.db'))

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_generate_circuits(args)

    assert exc_info.value.code == 1
    assert "gate_assignment" in capsys.readouterr().err


def test_cmd_generate_circuits_creates_new_db_and_calls_generate_circuits(tmp_path, monkeypatch):
    db_path = tmp_path / "new_circuits.db"
    assert not db_path.exists()

    created = {}
    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: created.setdefault('path', p))
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())

    calls = {}
    monkeypatch.setattr(cli, 'generate_circuits', lambda **kw: calls.update(kw))

    args = make_args(db_name=str(db_path), dag_size=5, min_fan_in=3)
    cli.cmd_generate_circuits(args)

    assert created['path'] == str(db_path)
    assert calls['dag_size'] == 5
    assert calls['min_num_inputs'] == 3
    assert calls['gate_assignment'] == 'random'   # default
    assert calls['display_circuits'] is False     # default
    assert isinstance(calls['db'], FakeDB)


def test_cmd_generate_circuits_overwrite_existing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "circuits.db"
    db_path.write_bytes(b"old contents")

    monkeypatch.setattr('builtins.input', lambda prompt: 'o')

    removed = []
    monkeypatch.setattr(cli.os, 'remove', lambda p: removed.append(p))

    created = []
    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: created.append(p))
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())
    monkeypatch.setattr(cli, 'generate_circuits', lambda **kw: None)

    args = make_args(db_name=str(db_path))
    cli.cmd_generate_circuits(args)

    assert removed == [str(db_path)]
    assert created == [str(db_path)]


def test_cmd_generate_circuits_quit_when_db_exists(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "circuits.db"
    db_path.write_bytes(b"old contents")

    monkeypatch.setattr('builtins.input', lambda prompt: 'q')

    def fail_if_called(**kw):
        pytest.fail("generate_circuits should not be called when the user quits")
    monkeypatch.setattr(cli, 'generate_circuits', fail_if_called)

    args = make_args(db_name=str(db_path))
    cli.cmd_generate_circuits(args)

    assert "Quitting" in capsys.readouterr().out
    assert db_path.read_bytes() == b"old contents"  # left untouched


def test_cmd_generate_circuits_if_exists_insert_skips_prompt(tmp_path, monkeypatch):
    db_path = tmp_path / "circuits.db"
    db_path.write_bytes(b"old contents")

    def fail_if_called(prompt):
        pytest.fail("input() should not be called when --if-exists is given")
    monkeypatch.setattr('builtins.input', fail_if_called)

    removed = []
    monkeypatch.setattr(cli.os, 'remove', lambda p: removed.append(p))
    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: None)
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())
    monkeypatch.setattr(cli, 'generate_circuits', lambda **kw: None)

    args = make_args(db_name=str(db_path), if_exists='insert')
    cli.cmd_generate_circuits(args)

    assert removed == []
    assert db_path.read_bytes() == b"old contents"  # left untouched, not overwritten


def test_cmd_generate_circuits_if_exists_overwrite_skips_prompt(tmp_path, monkeypatch):
    db_path = tmp_path / "circuits.db"
    db_path.write_bytes(b"old contents")

    def fail_if_called(prompt):
        pytest.fail("input() should not be called when --if-exists is given")
    monkeypatch.setattr('builtins.input', fail_if_called)

    removed = []
    monkeypatch.setattr(cli.os, 'remove', lambda p: removed.append(p))
    created = []
    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: created.append(p))
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())
    monkeypatch.setattr(cli, 'generate_circuits', lambda **kw: None)

    args = make_args(db_name=str(db_path), if_exists='overwrite')
    cli.cmd_generate_circuits(args)

    assert removed == [str(db_path)]
    assert created == [str(db_path)]


def test_cmd_generate_circuits_if_exists_rejects_bad_value(tmp_path, capsys):
    args = make_args(db_name=str(tmp_path / 'circuits.db'), if_exists='bogus')

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_generate_circuits(args)

    assert exc_info.value.code == 1
    assert "if_exists" in capsys.readouterr().err


def test_cmd_generate_circuits_reports_nauty_setup_error(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "circuits.db"

    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: None)
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())

    def raise_bad_nauty_dir(**kw):
        raise NotADirectoryError("bad nauty dir")
    monkeypatch.setattr(cli, 'generate_circuits', raise_bad_nauty_dir)

    args = make_args(db_name=str(db_path), nauty_dir='/bad/dir')

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_generate_circuits(args)

    assert exc_info.value.code == 1
    assert "nauty setup problem" in capsys.readouterr().err


# ── cmd_layout ──────────────────────────────────────────────────────────────

def test_cmd_layout_missing_file_exits(tmp_path, capsys):
    missing = tmp_path / "no_such.pkl"
    args = argparse.Namespace(pkl_file=str(missing))

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_layout(args)

    assert exc_info.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_cmd_layout_success(tmp_path, monkeypatch, capsys):
    import pickle
    pkl_path = tmp_path / "circuit.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({"fake": "graph"}, f)

    monkeypatch.setattr(cli, 'circuit_to_expression', lambda graph: "a*b")
    fake_diagram = MagicMock()
    monkeypatch.setattr(cli, 'layout_circuit', lambda graph: fake_diagram)

    args = argparse.Namespace(pkl_file=str(pkl_path))
    cli.cmd_layout(args)

    assert "a*b" in capsys.readouterr().out
    fake_diagram.draw.assert_called_once()


# ── main() end-to-end ────────────────────────────────────────────────────────

def test_main_generate_circuits_merges_config_and_cli(tmp_path, monkeypatch):
    cfg = tmp_path / "params.cfg"
    cfg.write_text("dag_size: 9\nmin_fan_in: 3\ngate_assignment: all\n")
    db_path = tmp_path / "circuits.db"

    monkeypatch.setattr(cli.circuit_database, 'create', lambda p: None)
    monkeypatch.setattr(cli.circuit_database, 'CircuitDatabase', lambda p: FakeDB())

    calls = {}
    monkeypatch.setattr(cli, 'generate_circuits', lambda **kw: calls.update(kw))

    monkeypatch.setattr(sys, 'argv', [
        'rysy_dl', 'generate-circuits', str(db_path),
        '--config', str(cfg), '--dag-size', '4',
    ])

    cli.main()

    assert calls['dag_size'] == 4              # CLI overrides file
    assert calls['min_num_inputs'] == 3         # from file
    assert calls['gate_assignment'] == 'all'    # from file
    assert calls['display_circuits'] is False        # hardcoded default
