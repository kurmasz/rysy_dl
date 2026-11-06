"""
Unit tests for the filtering parts of circuit_database.py."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit_database
from rysy_dl.circuit_database import CircuitDatabase, stat_filter_where


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
    assert stat_filter_where(num_gates=None, depth=None) == (None, ())


def test_stat_filter_where_exact_value():
    assert stat_filter_where(num_inputs=3) == ("num_inputs = ?", (3,))


def test_stat_filter_where_range():
    assert stat_filter_where(depth=(2, 4)) == ("depth BETWEEN ? AND ?", (2, 4))


def test_stat_filter_where_combines_with_and():
    where, params = stat_filter_where(num_gates=1, num_inputs=(2, 3))

    assert where == "num_gates = ? AND num_inputs BETWEEN ? AND ?"
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


# ── test circuit_stats ────────────────────────────────────────────────────────────────────

def test_circuit_stats_records_max_shared_subcircuit_depth():
    # Inputs 0, 1 -> AND 2 -> AND 3, and 3 feeds both 4 and 5, which feed the output 6.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1], gate='src')
    graph.add_nodes_from([2, 3, 4, 5, 6], gate='AND')
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (3, 5), (1, 4), (0, 5), (4, 6), (5, 6)])

    assert circuit_database.circuit_stats(graph)['max_shared_subcircuit_depth'] == 2
