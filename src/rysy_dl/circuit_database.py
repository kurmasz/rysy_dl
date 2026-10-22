"""
circuit_database.py

Manage a database of combinatorial circuits that can be used for computer organization homework problems.

Schema:
  id: int, primary key
  num_gates: int
  num_inputs: int
  depth: int
  max_indegree: int
  max_gate_outdegree: int
  max_input_outdegree: int
  gate_type_mask: integer bitmask
  has_shared_subexpressions: boolean (0/1)
  graph: blob
"""

import os
import pickle
import sqlite3
from typing import Callable, ClassVar, Optional, TYPE_CHECKING

import networkx as nx

from .graph_filters import has_shared_internal_node

from .graph_analyzers import num_sources_and_gates
from .circuit_serialization import circuit_to_serializable, serializable_to_circuit
from .db import DEFAULT_DB_PATH, resolve_db_path
from .logging_config import get_logger

if TYPE_CHECKING:
    from .circuit import Circuit

logger = get_logger(__name__)

TABLE_NAME = 'circuits'

# One bit per gate type that can appear in a circuit's 'gate' node attribute.
# 'src' (source/input nodes) is deliberately excluded -- it isn't a logic gate,
# so it shouldn't count toward gate_type_mask.
GATE_BITS = {
    'AND': 1 << 0,
    'OR': 1 << 1,
    'XOR': 1 << 2,
    'NAND': 1 << 3,
    'NOR': 1 << 4,
    'NOT': 1 << 5,
    'XNOR': 1 << 6,
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    num_gates           INTEGER NOT NULL,
    num_inputs          INTEGER NOT NULL,
    depth               INTEGER NOT NULL,
    max_indegree        INTEGER NOT NULL,
    max_gate_outdegree  INTEGER NOT NULL,
    max_input_outdegree INTEGER NOT NULL,
    gate_type_mask      INTEGER NOT NULL,
    has_shared_subexpressions INTEGER NOT NULL,
    graph               BLOB NOT NULL
)
"""

# Every column except id/graph, in schema order -- reused by CircuitDatabase.add (INSERT)
# and row_to_record (turning a row back into a dict).
_STAT_COLUMNS = ('num_gates', 'num_inputs', 'depth', 'max_indegree',
                  'max_gate_outdegree', 'max_input_outdegree', 'gate_type_mask',
                  'has_shared_subexpressions')


def gate_type_mask_of(graph):
    """
    Compute the `gate_type_mask` bitmask for `graph`: the bitwise OR of `GATE_BITS[gate]` for every
    distinct gate type (e.g. 'AND', 'NOT') found among the graph's node 'gate' attributes.

    :param graph: A networkx DiGraph representing a circuit, where every node has a 'gate' attribute.
    :returns: An int bitmask. Use ``gate_type_mask_of(graph) & GATE_BITS['NAND']`` to test for a specific gate.
    """
    mask = 0
    for node in graph.nodes():
        mask |= GATE_BITS.get(graph.nodes[node].get('gate'), 0)
    return mask


def circuit_stats(graph):
    """
    Compute every schema column (other than `id` and `graph`) for a circuit graph.

    :param graph: A networkx DiGraph representing a circuit, as produced by `circuit_generator`.
    :returns: A dict keyed by column name: num_gates, num_inputs, depth, max_indegree,
              max_gate_outdegree, max_input_outdegree, gate_type_mask, has_shared_subexpressions.
    """
    num_sources, num_gates, _not_gates = num_sources_and_gates(graph)

    gate_outdegrees = [graph.out_degree(n) for n in graph.nodes() if graph.in_degree(n) > 0]
    input_outdegrees = [graph.out_degree(n) for n in graph.nodes() if graph.in_degree(n) == 0]

    return {
        'num_gates': num_gates,
        'num_inputs': num_sources,
        'depth': nx.dag_longest_path_length(graph),
        'max_indegree': max((graph.in_degree(n) for n in graph.nodes()), default=0),
        'max_gate_outdegree': max(gate_outdegrees, default=0),
        'max_input_outdegree': max(input_outdegrees, default=0),
        'gate_type_mask': gate_type_mask_of(graph),
        'has_shared_subexpressions': has_shared_internal_node(graph),
    }


def row_to_record(row):
    """
    Convert a `sqlite3.Row` (or any mapping with the schema's column names) into a plain dict
    whose `graph` value is a reconstructed networkx DiGraph rather than a pickled blob.

    :param row: A row from the `circuits` table, with column access by name (e.g. `row['graph']`).
    :returns: A dict with keys id, num_gates, num_inputs, depth, max_indegree, max_gate_outdegree,
              max_input_outdegree, gate_type_mask, has_shared_subexpressions, and graph (a networkx DiGraph).
    """
    record = {col: row[col] for col in ('id',) + _STAT_COLUMNS}
    record['graph'] = serializable_to_circuit(pickle.loads(row['graph']))
    return record


class CircuitDatabase:
    """
    A connection to a circuit database that stays open across many `add`/`query`
    calls, instead of reconnecting for every single one.

    Reconnecting per call turns out to be cheap on its own (tens of microseconds) next to the cost
    of committing on every single insert (hundreds of microseconds, dominated by the fsync-like
    durability sync sqlite performs on every commit). So beyond just keeping the connection open,
    this class also batches commits: writes are committed every `commit_every` rows, and always on
    `close()` / context-manager exit. That trades a bit of crash-durability (you could lose up to
    `commit_every - 1` uncommitted rows if the process dies hard) for roughly two orders of
    magnitude less time spent on fsync during a big insertion run.

    Use as a context manager so the final commit and close happen even if an exception is raised
    partway through a batch of inserts::

        with CircuitDatabase(db_path) as db:
            for graph in circuits:
                db.add(graph)
    """

    # Factory used by `get_circuit` to turn a record into a `Circuit`. Left unset here so this
    # module has no dependency on `circuit.py`; the package `__init__.py` wires in `Circuit.from_entry`.
    default_circuit_factory: ClassVar[Optional[Callable[[dict], "Circuit"]]] = None

    def __init__(self, db_path=None, commit_every=500, circuit_factory=None):
        """
        Open the circuit database at `db_path`.

        :param db_path: Filesystem path where the sqlite database lives. Must already exist -- use
                         `create_database` to create a new database file. Defaults to
                         `rysy_dl.db.DEFAULT_DB_PATH`, the database packaged with rysy_dl. A string
                         prefixed with '@' (e.g. ``'@rysy_dag_only_4-8.sqlite'``) is resolved against
                         the `rysy_dl.db` package directory instead.
        :param commit_every: Commit after this many `add` calls since the last commit.
                              Larger values amortize the commit cost over more rows, at the cost of
                              losing more uncommitted work if the process crashes before the next commit.
        :param circuit_factory: Callable used by `get_circuit` to turn a record into a `Circuit`.
                                 Defaults to `default_circuit_factory` (set by the package `__init__.py`).
        :raises FileNotFoundError: If `db_path` doesn't exist.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        db_path = resolve_db_path(db_path)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Circuit database not found: {db_path}")
        self.db_path = db_path
        self.commit_every = commit_every
        self._circuit_factory = circuit_factory or type(self).default_circuit_factory
        self._pending = 0
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)   # Create the table if it isn't in this file yet.
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Commit any pending inserts and close the connection. Safe to call more than once."""
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None

    def add(self, netlist):
        """
        Add a circuit, computing all schema columns from `graph`.

        The graph itself is not pickled directly; instead `circuit_serialization.circuit_to_serializable`
        is used to extract only what's needed to recreate it, and that (smaller) structure is what
        gets pickled into the `graph` blob column. `row_to_record` reverses this when reading rows back.

        The insert is not necessarily committed immediately -- see the `commit_every` constructor
        argument. Call `commit()` directly if you need this row durable right away.

        :param graph: A networkx DiGraph representing a circuit, as produced by `circuit_generator`.
        :returns: The `id` of the newly inserted row.
        """
        stats = circuit_stats(netlist)
        blob = pickle.dumps(circuit_to_serializable(netlist), protocol=pickle.HIGHEST_PROTOCOL)

        columns = _STAT_COLUMNS + ('graph',)
        placeholders = ', '.join('?' for _ in columns)
        values = tuple(stats[col] for col in _STAT_COLUMNS) + (blob,)

        cursor = self._conn.execute(
            f"INSERT INTO {TABLE_NAME} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self._pending += 1
        if self._pending >= self.commit_every:
            self.commit()
        return cursor.lastrowid

    def commit(self):
        """Commit any inserts made since the last commit."""
        self._conn.commit()
        self._pending = 0

    def query(self, where=None, params=(), limit=None):
        """
        Query circuits from the database.

        :param where: Optional SQL boolean expression (without the leading ``WHERE``) using ``?``
                      placeholders for values, e.g. ``"num_gates <= ? AND gate_type_mask & ? != 0"``.
                      If omitted, every row is returned.
        :param params: Tuple of values to substitute for the ``?`` placeholders in `where`.
        :param limit: Optional maximum number of rows to return.
        :returns: A list of dicts (see `row_to_record`), one per matching circuit, ordered by `id`.
        """
        sql = f"SELECT * FROM {TABLE_NAME}"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params = tuple(params) + (limit,)

        rows = self._conn.execute(sql, params).fetchall()
        return [row_to_record(row) for row in rows]

    def iterate(self, where=None, params=()):
        """
        Iterate over circuits one row at a time, without loading the whole result set into memory.

        :param where: Optional SQL boolean expression (without the leading ``WHERE``) using ``?``
                      placeholders for values, e.g. ``"num_gates <= ?"``. If omitted, every row is
                      returned.
        :param params: Tuple of values to substitute for the ``?`` placeholders in `where`.
        :yields: A dict (see `row_to_record`) for each matching circuit, ordered by `id`.
        """
        sql = f"SELECT * FROM {TABLE_NAME}"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY id"

        cursor = self._conn.execute(sql, params)
        for row in cursor:
            yield row_to_record(row)

    def get_entry(self, circuit_id):
        """
        Look up a single circuit by its primary key.

        :param circuit_id: The `id` of the circuit to fetch.
        :returns: A dict (see `row_to_record`), or ``None`` if no circuit with that id exists.
        """
        results = self.query(where="id = ?", params=(circuit_id,), limit=1)
        return results[0] if results else None

    def _to_circuit(self, entry):
        """Apply `_circuit_factory` to `entry` (see `row_to_record`), or return ``None`` if there's no entry."""
        if self._circuit_factory is None:
            raise RuntimeError("no circuit_factory configured")
        return self._circuit_factory(entry) if entry is not None else None

    def get_circuit(self, circuit_id):
        """
        Look up a single circuit by its primary key, as a `Circuit` object.

        :param circuit_id: The `id` of the circuit to fetch.
        :returns: A `Circuit`, or ``None`` if no circuit with that id exists.
        :raises RuntimeError: If no `circuit_factory` was configured (see `__init__`).
        """
        return self._to_circuit(self.get_entry(circuit_id))

    def get_random_entry(self, *, num_gates=None, num_inputs=(1, 4), depth=None,
                          max_indegree=None, max_gate_outdegree=None, max_input_outdegree=None,
                          gate_type_mask=None, has_shared_subexpressions=None, allow_empty_result=False):
        """
        Fetch a single circuit chosen uniformly at random, optionally filtered.

        Each keyword corresponds to a column in `_STAT_COLUMNS`. Pass a single value to require
        an exact match, a ``(min, max)`` tuple for an inclusive range (e.g. ``gate_type_mask=(2, 10)``),
        or leave it at its default (``None``) to skip filtering on that column.

        :returns: A dict (see `row_to_record`), or ``None`` if no circuit matches.
        """
        filters = {
            'num_gates': num_gates,
            'num_inputs': num_inputs,
            'depth': depth,
            'max_indegree': max_indegree,
            'max_gate_outdegree': max_gate_outdegree,
            'max_input_outdegree': max_input_outdegree,
            'gate_type_mask': gate_type_mask,
            'has_shared_subexpressions': has_shared_subexpressions,
        }

        where_parts = []
        params = []
        for column, value in filters.items():
            if value is None:
                continue
            if isinstance(value, tuple):
                lo, hi = value
                where_parts.append(f"{column} BETWEEN ? AND ?")
                params.extend([lo, hi])
            else:
                where_parts.append(f"{column} = ?")
                params.append(value)

        sql = f"SELECT * FROM {TABLE_NAME}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY RANDOM() LIMIT 1"

        row = self._conn.execute(sql, params).fetchone()
        if row is None and not allow_empty_result:
            raise ValueError('No entry satisfied all requirements: ', sql, params)
        
        return row_to_record(row) if row is not None else None

    def get_random_circuit(self, *args, **kwargs):
        """
        Fetch a single circuit chosen uniformly at random from the whole table, as a `Circuit` object.

        :returns: A `Circuit`, or ``None`` if the table is empty.
        :raises RuntimeError: If no `circuit_factory` was configured (see `__init__`).
        """
        return self._to_circuit(self.get_random_entry(*args, **kwargs))


def create(db_path):
    """
    Create a new circuit database at `db_path`, or verify that an existing file at that path
    already has the required table. Safe to call every time (`CREATE TABLE IF NOT EXISTS`).

    This just ensures the schema exists and doesn't keep a connection open; if you're about to add
    or query circuits, use `CircuitDatabase` directly (ideally as a context manager) instead, so you
    don't pay a separate reconnect for every operation.

    :param db_path: Filesystem path where the sqlite database should live. A string prefixed with
                    '@' (e.g. ``'@rysy_dag_only_4-8.sqlite'``) is resolved against the `rysy_dl.db`
                    package directory instead.
    """
    conn = sqlite3.connect(resolve_db_path(db_path))
    try:
        conn.execute(SCHEMA)
        conn.commit()
    finally:
        conn.close()
