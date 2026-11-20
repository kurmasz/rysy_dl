"""
circuit_database.py

Manage a database of combinatorial circuits that can be used for computer organization homework problems.

Schema:
  id: int, primary key
  num_gates_core: int (every gate except NOT gates; "core" gates)
  num_gates_total: int (every gate, including NOT gates)
  num_inputs: int
  depth: int
  max_indegree: int
  max_gate_outdegree: int
  max_input_outdegree: int
  gate_type_mask: integer bitmask
  max_shared_subcircuit_depth: int (0 if no gate output is shared; see graph_filters.max_shared_subcircuit_depth)
  graph: blob (see circuit_serialization.circuit_to_bytes)

The file's format version is stored in `PRAGMA user_version` (see `FORMAT_VERSION`); `upgrade`
converts files written in older formats.
"""

import os
import pickle   # only for reading graphs written in format version 0, in `upgrade`
import sqlite3
from pathlib import Path
import sys
from typing import Callable, ClassVar, Optional, TYPE_CHECKING

import networkx as nx

from . import graph_filters

from .graph_analyzers import num_sources_and_gates
from .circuit_serialization import bytes_to_circuit, circuit_to_bytes, serializable_to_circuit
from .db import DEFAULT_DB_PATH, resolve_db_path
from .logging_config import get_logger

if TYPE_CHECKING:
    from .circuit import Circuit

logger = get_logger(__name__)

TABLE_NAME = 'circuits'

# The database format this code reads and writes, stored in the file's PRAGMA user_version:
#   0: graphs pickled as circuit_serialization.circuit_to_serializable dicts
#   1: graphs encoded with circuit_serialization.circuit_to_bytes
# `upgrade` converts older files.
FORMAT_VERSION = 1

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
    num_gates_core      INTEGER NOT NULL,
    num_gates_total     INTEGER NOT NULL,
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

# Lets the usual question queries read only the matching rows instead of the whole table: for
# concrete_circuits_n4-8, a 3-input query reads 0.8 MB instead of 28 MB, for 5.6 MB more on disk.
# It speeds up queries but doesn't change what's stored, so it isn't part of FORMAT_VERSION;
# `upgrade` adds it to databases created without it.
INDEX_NAME = f"{TABLE_NAME}_by_size"
INDEX_SCHEMA = f"""
CREATE INDEX IF NOT EXISTS {INDEX_NAME}
    ON {TABLE_NAME} (num_inputs, num_gates_core, max_shared_subcircuit_depth)
"""

# Every column except id/graph, in schema order -- reused by CircuitDatabase.add (INSERT)
# and row_to_record (turning a row back into a dict).
_STAT_COLUMNS = ('num_gates_core', 'num_gates_total', 'num_inputs', 'depth', 'max_indegree',
                  'max_gate_outdegree', 'max_input_outdegree', 'gate_type_mask',
                  'max_shared_subcircuit_depth')


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
    :returns: A dict keyed by column name: num_gates_core, num_gates_total, num_inputs, depth,
              max_indegree, max_gate_outdegree, max_input_outdegree, gate_type_mask,
              max_shared_subcircuit_depth.
    """
    num_sources, num_gates_total, _ = num_sources_and_gates(graph)
    # Count NOTs by their assigned gate, not by shape: in a graph whose gates aren't assigned yet,
    # a node with one input and one output is not necessarily a NOT.
    num_not_gates = sum(1 for _, gate in graph.nodes(data='gate') if gate == 'NOT')

    gate_outdegrees = [graph.out_degree(n) for n in graph.nodes() if graph.in_degree(n) > 0]
    input_outdegrees = [graph.out_degree(n) for n in graph.nodes() if graph.in_degree(n) == 0]

    return {
        'num_gates_core': num_gates_total - num_not_gates,
        'num_gates_total': num_gates_total,
        'num_inputs': num_sources,
        'depth': nx.dag_longest_path_length(graph),
        'max_indegree': max((graph.in_degree(n) for n in graph.nodes()), default=0),
        'max_gate_outdegree': max(gate_outdegrees, default=0),
        'max_input_outdegree': max(input_outdegrees, default=0),
        'gate_type_mask': gate_type_mask_of(graph),
        'max_shared_subcircuit_depth': graph_filters.max_shared_subcircuit_depth(graph),
    }


def stat_filter_where(**filters):
    """
    Build a SQL WHERE clause that filters circuits on the schema's stat columns.

    Each keyword names a column in `_STAT_COLUMNS`. A single value requires an exact match, a
    ``(min, max)`` tuple requires a value in that inclusive range, and ``None`` skips filtering
    on that column.

    :returns: A ``(where, params)`` pair suitable for `CircuitDatabase.iterate`: `where` is the
              clause without the leading ``WHERE`` (or ``None`` if nothing is filtered), and
              `params` is the tuple of values for its ``?`` placeholders.
    :raises ValueError: If a keyword isn't a stat column.
    """
    where_parts = []
    params = []
    for column, value in filters.items():
        if column not in _STAT_COLUMNS:
            raise ValueError(f"Not a circuit stat column: {column!r}")
        if value is None:
            continue
        if isinstance(value, tuple):
            lo, hi = value
            where_parts.append(f"{column} BETWEEN ? AND ?")
            params.extend([lo, hi])
        else:
            where_parts.append(f"{column} = ?")
            params.append(value)
    return (" AND ".join(where_parts) or None), tuple(params)


def row_to_record(row):
    """
    Convert a `sqlite3.Row` (or any mapping with the schema's column names) into a plain dict
    whose `graph` value is a reconstructed networkx DiGraph rather than an encoded blob.

    :param row: A row from the `circuits` table, with column access by name (e.g. `row['graph']`).
    :returns: A dict with keys id, num_gates_core, num_gates_total, num_inputs, depth, max_indegree, max_gate_outdegree,
              max_input_outdegree, gate_type_mask, max_shared_subcircuit_depth, and graph (a networkx DiGraph).
    """
    record = {col: row[col] for col in ('id',) + _STAT_COLUMNS}
    record['graph'] = bytes_to_circuit(row['graph'])
    return record


def _create_schema(conn):
    """Create the circuits table and its index if the file doesn't have them yet, marking a new file with FORMAT_VERSION."""
    if not _schema_object_exists(conn, 'table', TABLE_NAME):
        conn.execute(SCHEMA)
        conn.execute(INDEX_SCHEMA)
        conn.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
    conn.commit()


def _schema_object_exists(conn, kind, name):
    """True if the database has a table or index (`kind`) called `name`."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", (kind, name)).fetchone() is not None


def _format_error(db_path, version):
    """The error for a database whose format version isn't FORMAT_VERSION."""
    if version < FORMAT_VERSION:
        return ValueError(f"{db_path} uses an older database format (version {version}). "
                          f"Convert it with: rysy_dl upgrade-db {db_path}")
    return ValueError(f"{db_path} uses database format version {version}, which is newer than this "
                      f"version of rysy_dl supports ({FORMAT_VERSION}). Upgrade rysy_dl to read it.")


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

    With no `db_path`, it opens the database packaged with rysy_dl, read-only::

        with CircuitDatabase() as db:
            circuit = db.get_random_circuit(num_inputs=3, num_gates_core=3)
    """

    # Factory used by `get_circuit` to turn a record into a `Circuit`. Left unset here so this
    # module has no dependency on `circuit.py`; the package `__init__.py` wires in `Circuit.from_entry`.
    default_circuit_factory: ClassVar[Optional[Callable[[dict], "Circuit"]]] = None

    def __init__(self, db_path=None, commit_every=500, circuit_factory=None, read_only=None):
        """
        Open the circuit database at `db_path`.

        :param db_path: Filesystem path where the sqlite database lives. Must already exist -- use
                         `create_database` to create a new database file. Defaults to
                         `rysy_dl.db.DEFAULT_DB_PATH`, the database packaged with rysy_dl. A string
                         prefixed with '@' (e.g. ``'@circuit_templates_n3-9.sqlite'``) is resolved against
                         the `rysy_dl.db` package directory instead.
        :param commit_every: Commit after this many `add` calls since the last commit.
                              Larger values amortize the commit cost over more rows, at the cost of
                              losing more uncommitted work if the process crashes before the next commit.
        :param circuit_factory: Callable used by `get_circuit` to turn a record into a `Circuit`.
                                 Defaults to `default_circuit_factory` (set by the package `__init__.py`).
        :param read_only: Open the database read-only, so `add` raises instead of writing. Defaults to
                           True when `db_path` is omitted (the packaged database is only meant to be
                           queried) and False otherwise.
        :raises FileNotFoundError: If `db_path` doesn't exist.
        :raises ValueError: If the database is in an older or newer format (see `upgrade`).
        """
        if read_only is None:
            read_only = db_path is None
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        db_path = resolve_db_path(db_path)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Circuit database not found: {db_path}")
        self.db_path = db_path
        self.commit_every = commit_every
        self._circuit_factory = circuit_factory or type(self).default_circuit_factory
        self._pending = 0
        if read_only:
            self._conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
        else:
            self._conn = sqlite3.connect(db_path)
            _create_schema(self._conn)
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != FORMAT_VERSION:
            self._conn.close()
            self._conn = None
            raise _format_error(db_path, version)
        self._conn.row_factory = sqlite3.Row

    @property
    def closed(self):
        """True once `close` has been called."""
        return self._conn is None

    def _open_conn(self):
        """Return the sqlite connection, or raise a clear error if `close` has been called."""
        if self._conn is None:
            raise ValueError("CircuitDatabase is closed")
        return self._conn

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

        The graph is stored in the `graph` blob column in the compact format of
        `circuit_serialization.circuit_to_bytes`; `row_to_record` reverses this when reading rows back.

        The insert is not necessarily committed immediately -- see the `commit_every` constructor
        argument. Call `commit()` directly if you need this row durable right away.

        :param graph: A networkx DiGraph representing a circuit, as produced by `circuit_generator`.
        :returns: The `id` of the newly inserted row.
        """
        stats = circuit_stats(netlist)
        blob = circuit_to_bytes(netlist)

        columns = _STAT_COLUMNS + ('graph',)
        placeholders = ', '.join('?' for _ in columns)
        values = tuple(stats[col] for col in _STAT_COLUMNS) + (blob,)

        cursor = self._open_conn().execute(
            f"INSERT INTO {TABLE_NAME} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self._pending += 1
        if self._pending >= self.commit_every:
            self.commit()
        return cursor.lastrowid

    def commit(self):
        """Commit any inserts made since the last commit."""
        self._open_conn().commit()
        self._pending = 0

    def query(self, where=None, params=(), limit=None):
        """
        Query circuits from the database.

        :param where: Optional SQL boolean expression (without the leading ``WHERE``) using ``?``
                      placeholders for values, e.g. ``"num_gates_core <= ? AND gate_type_mask & ? != 0"``.
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

        rows = self._open_conn().execute(sql, params).fetchall()
        return [row_to_record(row) for row in rows]

    def iterate(self, where=None, params=()):
        """
        Iterate over circuits one row at a time, without loading the whole result set into memory.

        :param where: Optional SQL boolean expression (without the leading ``WHERE``) using ``?``
                      placeholders for values, e.g. ``"num_gates_core <= ?"``. If omitted, every row is
                      returned.
        :param params: Tuple of values to substitute for the ``?`` placeholders in `where`.
        :yields: A dict (see `row_to_record`) for each matching circuit, ordered by `id`.
        """
        sql = f"SELECT * FROM {TABLE_NAME}"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY id"

        cursor = self._open_conn().execute(sql, params)
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

    def iterate_entries(self, *, num_gates_core=None, num_gates_total=None, num_inputs=None, depth=None,
                        max_indegree=None, max_gate_outdegree=None, max_input_outdegree=None,
                        gate_type_mask=None, max_shared_subcircuit_depth=None):
        """
        Iterate over the circuits that match the given filters, one row at a time, ordered by `id`.

        Takes the same filter keywords as `get_random_entry` (see `stat_filter_where`), except that
        every filter defaults to ``None``, so with no arguments every circuit is returned.

        :yields: A dict (see `row_to_record`) for each matching circuit.
        """
        where, params = stat_filter_where(
            num_gates_core=num_gates_core, num_gates_total=num_gates_total, num_inputs=num_inputs,
            depth=depth, max_indegree=max_indegree,
            max_gate_outdegree=max_gate_outdegree, max_input_outdegree=max_input_outdegree,
            gate_type_mask=gate_type_mask, max_shared_subcircuit_depth=max_shared_subcircuit_depth,
        )
        return self.iterate(where, params)

    def get_random_entry(self, *, num_gates_core=None, num_gates_total=None, num_inputs=(1, 4), depth=None,
                          max_indegree=None, max_gate_outdegree=None, max_input_outdegree=None,
                          gate_type_mask=None, max_shared_subcircuit_depth=None, allow_empty_result=False,
                          has_shared_subexpressions=None):
        """
        Fetch a single circuit chosen uniformly at random, optionally filtered.

        Each keyword corresponds to a column in `_STAT_COLUMNS`. Pass a single value to require
        an exact match, a ``(min, max)`` tuple for an inclusive range (e.g. ``gate_type_mask=(2, 10)``),
        or leave it at its default (``None``) to skip filtering on that column.

        :returns: A dict (see `row_to_record`), or ``None`` if no circuit matches.
        """
        
        ### QQQ TODO Deprecate has_shared_subexpressions in favor of max_shared_subcircuit_depth, which is more general.
        if max_shared_subcircuit_depth is not None and has_shared_subexpressions is not None:
            raise ValueError("max_shared_subcircuit_depth and has_shared_subexpressions are mutually exclusive")
        
        if max_shared_subcircuit_depth is None and has_shared_subexpressions is not None:
            # True means at least one shared gate output, i.e. any depth of 1 or more.
            max_shared_subcircuit_depth = 0 if not has_shared_subexpressions else (1, sys.maxsize)
        
        
        where, params = stat_filter_where(
            num_gates_core=num_gates_core, num_gates_total=num_gates_total, num_inputs=num_inputs,
            depth=depth, max_indegree=max_indegree,
            max_gate_outdegree=max_gate_outdegree, max_input_outdegree=max_input_outdegree,
            gate_type_mask=gate_type_mask, max_shared_subcircuit_depth=max_shared_subcircuit_depth,
        )

        sql = f"SELECT * FROM {TABLE_NAME}"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY RANDOM() LIMIT 1"

        row = self._open_conn().execute(sql, params).fetchone()
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
                    '@' (e.g. ``'@circuit_templates_n3-9.sqlite'``) is resolved against the `rysy_dl.db`
                    package directory instead.
    """
    conn = sqlite3.connect(resolve_db_path(db_path))
    try:
        _create_schema(conn)
    finally:
        conn.close()


def upgrade(db_path):
    """
    Bring an existing circuit database up to the current format (`FORMAT_VERSION`), in place.

    The changes so far:
      * The old `num_gates` column (which counted NOT gates) becomes `num_gates_total`, and a new
        `num_gates_core` column counts every gate except NOT gates, computed from each stored graph.
      * Graphs pickled as `circuit_to_serializable` dicts (format version 0) are re-encoded with
        `circuit_serialization.circuit_to_bytes` (version 1), about a fifth of the size.
      * The query index (`INDEX_NAME`) is added if it's missing.

    The upgrade runs in a single transaction, so a failure leaves the file unchanged, and it
    compacts the file afterwards. Upgrading a database that is already current does nothing.

    :param db_path: Path to the sqlite database. A string prefixed with '@' is resolved against
                    the `rysy_dl.db` package directory.
    :returns: A list of short descriptions of the changes made; empty if the database was already
              current.
    :raises FileNotFoundError: If `db_path` doesn't exist.
    :raises ValueError: If the file doesn't have a recognizable circuits table, or is in a format
                        newer than this code knows.
    :raises RuntimeError: If another program has the database open.
    """
    db_path = resolve_db_path(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Circuit database not found: {db_path}")

    conn = sqlite3.connect(db_path, isolation_level=None)   # transactions managed explicitly below
    try:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_old_gate_count = 'num_gates' in columns
        if not has_old_gate_count and not {'num_gates_core', 'num_gates_total'} <= columns:
            raise ValueError(f"{db_path} has no '{TABLE_NAME}' table to upgrade")
        if version > FORMAT_VERSION:
            raise _format_error(db_path, version)
        rewrite_rows = has_old_gate_count or version < FORMAT_VERSION
        add_index = not _schema_object_exists(conn, 'index', INDEX_NAME)
        if not rewrite_rows and not add_index:
            return []

        # Take exclusive access up front. Otherwise another program reading the file (such as an open
        # view-circuits window) makes every write wait out the busy timeout, so the upgrade crawls
        # instead of failing.
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as e:
            if 'locked' not in str(e):
                raise
            raise RuntimeError(f"{db_path} is in use by another program (for example, an open "
                               f"view-circuits window). Close it and try again.") from e
        changes = []
        try:
            if has_old_gate_count:
                conn.execute(f"ALTER TABLE {TABLE_NAME} RENAME COLUMN num_gates TO num_gates_total")
                conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN num_gates_core INTEGER NOT NULL DEFAULT 0")
                changes.append("split num_gates into num_gates_core and num_gates_total")
            if rewrite_rows:
                updates = []
                for circuit_id, blob in conn.execute(f"SELECT id, graph FROM {TABLE_NAME}"):
                    if version == 0:
                        graph = serializable_to_circuit(pickle.loads(blob))
                    else:
                        graph = bytes_to_circuit(blob)
                    num_not_gates = sum(1 for _, gate in graph.nodes(data='gate') if gate == 'NOT')
                    updates.append((num_not_gates, circuit_to_bytes(graph), circuit_id))
                conn.executemany(
                    f"UPDATE {TABLE_NAME} SET num_gates_core = num_gates_total - ?, graph = ? WHERE id = ?",
                    updates)
                if version < FORMAT_VERSION:
                    changes.append(f"re-encoded {len(updates)} graphs compactly")
                conn.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
            if add_index:
                conn.execute(INDEX_SCHEMA)
                changes.append("added the query index")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if rewrite_rows:
            # Rewriting every row leaves pages partly empty (filling in num_gates_core alone made
            # concrete_circuits_n4-8 about 12% larger, and shorter graph blobs leave even more gaps).
            # VACUUM repacks the file.
            conn.execute("VACUUM")
        return changes
    finally:
        conn.close()
