"""
Unit tests for opening and closing CircuitDatabase connections, including the packaged default."""

import os
import shutil
import sqlite3
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit_database
from rysy_dl.circuit_database import CircuitDatabase
from rysy_dl.db import DEFAULT_DB_PATH


def _and_tree(num_inputs):
    """A circuit with `num_inputs` sources all feeding a single AND gate."""
    graph = nx.DiGraph()
    for source in range(num_inputs):
        graph.add_node(source, gate='src')
    graph.add_node(num_inputs, gate='AND')
    for source in range(num_inputs):
        graph.add_edge(source, num_inputs)
    return graph


@pytest.fixture
def db_path(tmp_path):
    """A database holding one AND tree, in a directory whose name needs URI escaping."""
    directory = tmp_path / "dir with space"
    directory.mkdir()
    path = str(directory / "circuits.sqlite")
    circuit_database.create(path)
    with CircuitDatabase(path) as database:
        database.add(_and_tree(2))
    return path


# ── the packaged default database ─────────────────────────────────────────────────────────────

def test_default_works_after_a_previous_with_block():
    with CircuitDatabase() as db:
        db.get_random_entry()
    with CircuitDatabase() as db:
        assert db.get_random_entry() is not None


def test_default_held_connection_survives_another_callers_close():
    held = CircuitDatabase()
    with CircuitDatabase() as other:
        other.get_random_entry()
    assert held.get_random_entry() is not None
    held.close()


@pytest.fixture
def default_copy(tmp_path, monkeypatch):
    """Point the default database at a copy, so a failing test can't write into the packaged one."""
    copy = tmp_path / "default.sqlite"
    shutil.copy(DEFAULT_DB_PATH, copy)
    monkeypatch.setattr(circuit_database, 'DEFAULT_DB_PATH', copy)
    return copy


def test_default_is_read_only(default_copy):
    with CircuitDatabase() as db:
        with pytest.raises(sqlite3.OperationalError):
            db.add(_and_tree(2))


def test_default_can_be_opened_writable_explicitly(default_copy):
    with CircuitDatabase(read_only=False) as db:
        db.add(_and_tree(2))


# ── read_only ─────────────────────────────────────────────────────────────────────────────────

def test_explicit_path_is_writable_by_default(db_path):
    with CircuitDatabase(db_path) as db:
        db.add(_and_tree(3))
    with CircuitDatabase(db_path) as db:
        assert len(db.query()) == 2


def test_read_only_can_query(db_path):
    with CircuitDatabase(db_path, read_only=True) as db:
        assert len(db.query()) == 1


def test_read_only_rejects_add(db_path):
    with CircuitDatabase(db_path, read_only=True) as db:
        with pytest.raises(sqlite3.OperationalError):
            db.add(_and_tree(3))
    with CircuitDatabase(db_path) as db:
        assert len(db.query()) == 1


# ── closing ───────────────────────────────────────────────────────────────────────────────────

def test_closed_is_false_while_open_and_true_after_close(db_path):
    db = CircuitDatabase(db_path)
    assert db.closed is False
    db.close()
    assert db.closed is True


def test_leaving_with_block_closes(db_path):
    with CircuitDatabase(db_path) as db:
        pass
    assert db.closed is True


def test_close_is_safe_to_call_twice(db_path):
    db = CircuitDatabase(db_path)
    db.close()
    db.close()
    assert db.closed is True


@pytest.mark.parametrize('use', [
    lambda db: db.query(),
    lambda db: list(db.iterate()),
    lambda db: db.get_random_entry(),
    lambda db: db.add(_and_tree(2)),
    lambda db: db.commit(),
])
def test_using_a_closed_database_raises_a_clear_error(db_path, use):
    db = CircuitDatabase(db_path)
    db.close()
    with pytest.raises(ValueError, match="CircuitDatabase is closed"):
        use(db)
