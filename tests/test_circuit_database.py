"""
Unit tests for the filtering parts of circuit_database.py, and for upgrading old databases."""

import os
import pickle
import sqlite3
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit_database
from rysy_dl.circuit_database import CircuitDatabase, stat_filter_where
from rysy_dl.circuit_serialization import circuit_to_serializable


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
def db(tmp_path):
    """A database holding AND trees with 2, 3, 4, and 5 inputs (ids 1-4, in that order)."""
    db_path = str(tmp_path / "circuits.sqlite")
    circuit_database.create(db_path)
    with CircuitDatabase(db_path) as database:
        for num_inputs in (2, 3, 4, 5):
            database.add(_and_tree(num_inputs))
        database.commit()
        yield database


# ── test stat_filter_where ────────────────────────────────────────────────────────────────────

def test_stat_filter_where_no_filters():
    assert stat_filter_where() == (None, ())


def test_stat_filter_where_all_none_is_no_filter():
    assert stat_filter_where(num_gates_core=None, depth=None) == (None, ())


def test_stat_filter_where_exact_value():
    assert stat_filter_where(num_inputs=3) == ("num_inputs = ?", (3,))


def test_stat_filter_where_range():
    assert stat_filter_where(depth=(2, 4)) == ("depth BETWEEN ? AND ?", (2, 4))


def test_stat_filter_where_combines_with_and():
    where, params = stat_filter_where(num_gates_core=1, num_inputs=(2, 3))

    assert where == "num_gates_core = ? AND num_inputs BETWEEN ? AND ?"
    assert params == (1, 2, 3)


def test_stat_filter_where_rejects_unknown_column():
    with pytest.raises(ValueError):
        stat_filter_where(graph=1)


# ── test iterate_entries ────────────────────────────────────────────────────────────────────

def test_iterate_entries_no_filters_returns_everything_in_id_order(db):
    assert [entry['num_inputs'] for entry in db.iterate_entries()] == [2, 3, 4, 5]


def test_iterate_entries_exact_value(db):
    entries = list(db.iterate_entries(num_inputs=4))

    assert [entry['id'] for entry in entries] == [3]
    assert isinstance(entries[0]['graph'], nx.DiGraph)


def test_iterate_entries_range_is_inclusive(db):
    assert [entry['num_inputs'] for entry in db.iterate_entries(num_inputs=(3, 4))] == [3, 4]


def test_iterate_entries_shared_subcircuit_depth_filter(db):
    # AND trees never share an internal node.
    assert list(db.iterate_entries(max_shared_subcircuit_depth=(1, 99))) == []
    assert len(list(db.iterate_entries(max_shared_subcircuit_depth=0))) == 4


def test_iterate_entries_no_match(db):
    assert list(db.iterate_entries(num_inputs=9)) == []


# ── test get_random_entry ────────────────────────────────────────────────────────────────────

def test_get_random_entry_applies_filters(db):
    for _ in range(10):
        assert db.get_random_entry(num_inputs=5)['id'] == 4


def test_get_random_entry_no_match_raises(db):
    with pytest.raises(ValueError):
        db.get_random_entry(num_inputs=9)


def test_get_random_entry_no_match_allowed(db):
    assert db.get_random_entry(num_inputs=9, allow_empty_result=True) is None


def test_get_random_entry_has_shared_subexpressions(db):
    # Inputs 0, 1 -> AND 2 -> AND 3, and 3 feeds both 4 and 5, which feed the output 6.
    shared = nx.DiGraph()
    shared.add_nodes_from([0, 1], gate='src')
    shared.add_nodes_from([2, 3, 4, 5, 6], gate='AND')
    shared.add_edges_from([(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (3, 5), (1, 4), (0, 5), (4, 6), (5, 6)])
    db.add(shared)
    db.commit()

    for _ in range(10):
        assert db.get_random_entry(has_shared_subexpressions=True)['id'] == 5
        assert db.get_random_entry(has_shared_subexpressions=False)['id'] != 5


def test_get_random_entry_rejects_both_shared_filters(db):
    with pytest.raises(ValueError):
        db.get_random_entry(has_shared_subexpressions=False, max_shared_subcircuit_depth=0)


# ── test circuit_stats ────────────────────────────────────────────────────────────────────

def test_circuit_stats_records_max_shared_subcircuit_depth():
    # Inputs 0, 1 -> AND 2 -> AND 3, and 3 feeds both 4 and 5, which feed the output 6.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1], gate='src')
    graph.add_nodes_from([2, 3, 4, 5, 6], gate='AND')
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (3, 5), (1, 4), (0, 5), (4, 6), (5, 6)])

    assert circuit_database.circuit_stats(graph)['max_shared_subcircuit_depth'] == 2


def _with_source_nots():
    """(A' * B) + (A * B'): inputs 0 and 1, NOTs 2 and 3, ANDs 4 and 5, output OR 6."""
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1], gate='src')
    graph.add_nodes_from([2, 3], gate='NOT')
    graph.add_nodes_from([4, 5], gate='AND')
    graph.add_node(6, gate='OR')
    graph.add_edges_from([(0, 2), (2, 4), (1, 4), (0, 5), (1, 3), (3, 5), (4, 6), (5, 6)])
    return graph


def test_circuit_stats_core_gates_exclude_nots():
    stats = circuit_database.circuit_stats(_with_source_nots())
    assert stats['num_gates_core'] == 3
    assert stats['num_gates_total'] == 5


def test_circuit_stats_counts_nots_by_gate_not_shape():
    # Gate 2 has one input and one output, like a NOT, but it is an AND.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1], gate='src')
    graph.add_nodes_from([2, 3], gate='AND')
    graph.add_edges_from([(0, 2), (2, 3), (1, 3)])
    stats = circuit_database.circuit_stats(graph)
    assert stats['num_gates_core'] == stats['num_gates_total'] == 2


def test_get_random_entry_filters_on_core_and_total_gates(db):
    db.add(_with_source_nots())
    db.commit()
    # The AND trees (ids 1-4) each have one gate and no NOTs.
    assert db.get_random_entry(num_inputs=None, num_gates_core=3)['id'] == 5
    assert db.get_random_entry(num_inputs=None, num_gates_total=5)['id'] == 5
    assert db.get_random_entry(num_inputs=None, num_gates_total=3, allow_empty_result=True) is None


# ── test upgrade ────────────────────────────────────────────────────────────────────────────

# The schema before num_gates was split into num_gates_core and num_gates_total.
_OLD_SCHEMA = """
CREATE TABLE circuits (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    num_gates           INTEGER NOT NULL,
    num_inputs          INTEGER NOT NULL,
    depth               INTEGER NOT NULL,
    max_indegree        INTEGER NOT NULL,
    max_gate_outdegree  INTEGER NOT NULL,
    max_input_outdegree INTEGER NOT NULL,
    gate_type_mask      INTEGER NOT NULL,
    max_shared_subcircuit_depth INTEGER NOT NULL,
    graph               BLOB NOT NULL
)
"""


@pytest.fixture
def old_db_path(tmp_path):
    """An old-schema database holding an AND tree (id 1) and a circuit with two NOTs (id 2)."""
    path = str(tmp_path / "old.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(_OLD_SCHEMA)
    for graph in (_and_tree(3), _with_source_nots()):
        stats = circuit_database.circuit_stats(graph)
        conn.execute(
            "INSERT INTO circuits (num_gates, num_inputs, depth, max_indegree, max_gate_outdegree, "
            "max_input_outdegree, gate_type_mask, max_shared_subcircuit_depth, graph) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (stats['num_gates_total'], stats['num_inputs'], stats['depth'], stats['max_indegree'],
             stats['max_gate_outdegree'], stats['max_input_outdegree'], stats['gate_type_mask'],
             stats['max_shared_subcircuit_depth'], pickle.dumps(circuit_to_serializable(graph))))
    conn.commit()
    conn.close()
    return path


def test_upgrade_splits_num_gates(old_db_path):
    assert circuit_database.upgrade(old_db_path) == 2

    with CircuitDatabase(old_db_path, read_only=True) as db:
        tree, with_nots = db.query()
    assert (tree['num_gates_core'], tree['num_gates_total']) == (1, 1)
    assert (with_nots['num_gates_core'], with_nots['num_gates_total']) == (3, 5)


def test_upgraded_database_matches_a_new_one(old_db_path, tmp_path):
    circuit_database.upgrade(old_db_path)
    new_path = str(tmp_path / "new.sqlite")
    circuit_database.create(new_path)
    with CircuitDatabase(new_path) as db:
        db.add(_and_tree(3))
        db.add(_with_source_nots())

    with CircuitDatabase(old_db_path, read_only=True) as old, CircuitDatabase(new_path, read_only=True) as new:
        strip = lambda records: [{k: v for k, v in r.items() if k != 'graph'} for r in records]
        assert strip(old.query()) == strip(new.query())


def test_upgraded_database_accepts_new_circuits(old_db_path):
    circuit_database.upgrade(old_db_path)
    with CircuitDatabase(old_db_path) as db:
        db.add(_and_tree(2))
        assert db.get_random_entry(num_inputs=None, num_gates_core=1, num_gates_total=1)['id'] in (1, 3)


def test_upgrade_twice_does_nothing_the_second_time(old_db_path):
    circuit_database.upgrade(old_db_path)
    assert circuit_database.upgrade(old_db_path) == 0


def test_upgrade_current_database_does_nothing(tmp_path):
    path = str(tmp_path / "new.sqlite")
    circuit_database.create(path)
    assert circuit_database.upgrade(path) == 0


def test_upgrade_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        circuit_database.upgrade(str(tmp_path / "missing.sqlite"))


def test_upgrade_rejects_file_without_circuits_table(tmp_path):
    path = str(tmp_path / "other.sqlite")
    sqlite3.connect(path).close()
    with pytest.raises(ValueError):
        circuit_database.upgrade(path)


def test_upgrade_failure_leaves_database_unchanged(old_db_path, monkeypatch):
    def broken(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(circuit_database, 'serializable_to_circuit', broken)

    with pytest.raises(RuntimeError):
        circuit_database.upgrade(old_db_path)

    conn = sqlite3.connect(old_db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(circuits)")}
    conn.close()
    assert 'num_gates' in columns and 'num_gates_core' not in columns


def test_upgrade_while_another_program_reads_fails_fast_and_changes_nothing(old_db_path):
    reader = sqlite3.connect(old_db_path)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM circuits").fetchone()    # holds a read lock, like an open view-circuits
    try:
        with pytest.raises(RuntimeError, match="in use by another program"):
            circuit_database.upgrade(old_db_path)
    finally:
        reader.close()

    assert circuit_database.upgrade(old_db_path) == 2       # works once the reader is gone


# ── format version ──────────────────────────────────────────────────────────────────────────

def _user_version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_new_database_is_marked_with_current_format(tmp_path):
    path = str(tmp_path / "new.sqlite")
    circuit_database.create(path)
    assert _user_version(path) == circuit_database.FORMAT_VERSION


def test_opening_old_format_says_to_upgrade(old_db_path):
    for read_only in (True, False):
        with pytest.raises(ValueError, match="rysy_dl upgrade-db"):
            CircuitDatabase(old_db_path, read_only=read_only)


def test_opening_newer_format_says_to_upgrade_rysy_dl(tmp_path):
    path = str(tmp_path / "new.sqlite")
    circuit_database.create(path)
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {circuit_database.FORMAT_VERSION + 1}")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="newer"):
        CircuitDatabase(path, read_only=True)
    with pytest.raises(ValueError, match="newer"):
        circuit_database.upgrade(path)


def test_upgrade_marks_database_with_current_format(old_db_path):
    circuit_database.upgrade(old_db_path)
    assert _user_version(old_db_path) == circuit_database.FORMAT_VERSION


def test_upgrade_keeps_graphs_identical(old_db_path):
    before = [_and_tree(3), _with_source_nots()]
    circuit_database.upgrade(old_db_path)
    with CircuitDatabase(old_db_path, read_only=True) as db:
        after = [record['graph'] for record in db.query()]
    for a, b in zip(after, before):
        assert list(a.nodes(data=True)) == list(b.nodes(data=True))
        assert list(a.edges()) == list(b.edges())


def test_upgrade_converts_graphs_when_gate_counts_are_already_split(old_db_path):
    # The state of databases upgraded before the compact encoding: new columns, pickled graphs.
    conn = sqlite3.connect(old_db_path)
    conn.execute("ALTER TABLE circuits RENAME COLUMN num_gates TO num_gates_total")
    conn.execute("ALTER TABLE circuits ADD COLUMN num_gates_core INTEGER NOT NULL DEFAULT 0")
    conn.execute("UPDATE circuits SET num_gates_core = num_gates_total - (id = 2) * 2")
    conn.commit()
    conn.close()

    assert circuit_database.upgrade(old_db_path) == 2
    with CircuitDatabase(old_db_path, read_only=True) as db:
        tree, with_nots = db.query()
    assert (tree['num_gates_core'], tree['num_gates_total']) == (1, 1)
    assert (with_nots['num_gates_core'], with_nots['num_gates_total']) == (3, 5)


def test_graphs_are_stored_compactly(tmp_path):
    path = str(tmp_path / "new.sqlite")
    circuit_database.create(path)
    with CircuitDatabase(path) as db:
        db.add(_with_source_nots())
    conn = sqlite3.connect(path)
    (blob,) = conn.execute("SELECT graph FROM circuits").fetchone()
    conn.close()
    assert blob == circuit_database.circuit_to_bytes(_with_source_nots())
