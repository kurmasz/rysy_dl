"""
Unit tests for circuit_generator.py.
"""

import inspect
import io
import os
import sys
import stat
from collections import Counter
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

# ── mock heavy dependencies before importing the module ──────────────────────
# Note: networkx is NOT mocked — no_collapsable_children needs a real DiGraph.
#
# circuit_generator.py itself no longer imports any of these (schemdraw is only
# imported lazily inside generate_circuits's display_circuits branch). The
# generate_circuits tests that exercise that branch instead patch the
# `circuit_generator_display` attribute directly on the `rysy_dl` package
# object (see the display tests below for why: `from . import
# circuit_generator_display`, re-run on every generate_circuits call, resolves
# via getattr on the already-imported rysy_dl package rather than consulting
# sys.modules, so patching sys.modules alone doesn't reliably intercept it
# once some other test file's unmocked import has made it a real attribute).
# But `from rysy_dl.circuit_generator import ...` still
# initializes the rysy_dl package first, and rysy_dl/__init__.py eagerly imports
# circuit.py -> circuit_layout.py -> schemdraw (and custom_gates.py, which pulls
# in schemdraw.elements.elements and schemdraw.segments). Mocking schemdraw here
# short-circuits that whole cascade -- and since schemdraw's own real import is
# what would otherwise drag in matplotlib.pyplot as a side effect, mocking it
# also means matplotlib itself is never touched, so it doesn't need mocking too.
# cairosvg and graph_image_gen aren't imported by anything in that cascade at
# all (cairosvg is only ever imported lazily, inside circuit_generator_display
# functions; graph_image_gen isn't imported anywhere in the package), so they
# don't need mocking either.
for _mod in ['schemdraw', 'schemdraw.elements', 'schemdraw.logic',
             'schemdraw.elements.elements', 'schemdraw.segments']:
    if _mod not in sys.modules:
        _mock = MagicMock()
        _mock.__all__ = []
        sys.modules[_mod] = _mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import rysy_dl  # noqa: E402 --- needed to patch rysy_dl.circuit_generator_display directly (see below)
from rysy_dl.circuit_generator import (  # noqa: E402 --- tell linter to ignore import position
    verify_nauty, find_nauty, generate_nauty_digraphs, nauty_to_networkx,
    _add_single_sink, assign_all_gates_and, assign_gates_randomly_ensure_diversity,
    assign_gates_randomly, all_gate_assignment, generate_circuits,
    _not_node_inserter, insert_random_source_edge_nots,
)
from rysy_dl.circuit_filters import no_collapsable_children  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_executable(path):
    """Add executable bits to a file."""
    current = os.stat(path).st_mode
    os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_non_executable(path):
    """Remove all executable bits from a file."""
    current = os.stat(path).st_mode
    os.chmod(path, current & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _touch(path, executable=True):
    """Create an empty file, optionally marking it executable."""
    with open(path, 'w'):
        pass
    if executable:
        _make_executable(path)
    else:
        _make_non_executable(path)


def _real_nauty_available():
    """Use the same lookup generate_nauty_digraphs uses, so the skip condition below
    checks the same locations a real call would use."""
    try:
        find_nauty()
        return True
    except (NotADirectoryError, FileNotFoundError, PermissionError):
        return False


# Real ``geng N | directg -aTq`` output for N = 2, 3, 4, captured from ~/Code/nauty2_9_3
# (excluding the ">A"/">Z" progress lines, which are stderr, not stdout). Note: no ``-c``
# flag on geng -- the pipeline in `generate_nauty_digraphs` does not restrict to connected
# graphs, so disconnected DAGs (e.g. "3 0", the edgeless graph on 3 nodes) are expected too.
# Shared by the mocked unit test and the real-binary integration test below, so both stay
# pinned to the same known-correct output.
GENG_DIRECTG_SAMPLES = {
    2: [
        "2 0",
        "2 1 0 1",
    ],
    3: [
        "3 0",
        "3 1 0 2",
        "3 2 0 2 1 2",
        "3 2 0 2 2 1",
        "3 2 2 0 2 1",
        "3 3 0 1 0 2 1 2",
    ],
    4: [
        "4 0",
        "4 1 0 3",
        "4 2 0 3 1 3",
        "4 2 0 3 3 1",
        "4 2 3 0 3 1",
        "4 3 0 3 1 3 2 3",
        "4 3 0 3 1 3 3 2",
        "4 3 0 3 3 1 3 2",
        "4 3 3 0 3 1 3 2",
        "4 2 0 2 1 3",
        "4 3 0 2 0 3 1 3",
        "4 3 0 2 0 3 3 1",
        "4 3 0 2 3 0 1 3",
        "4 3 2 0 0 3 1 3",
        "4 3 0 2 0 3 2 3",
        "4 4 0 2 0 3 1 3 2 3",
        "4 4 0 2 0 3 1 3 3 2",
        "4 4 0 2 0 3 3 1 2 3",
        "4 4 0 2 0 3 3 1 3 2",
        "4 4 0 2 3 0 1 3 3 2",
        "4 4 0 2 3 0 3 1 3 2",
        "4 4 0 2 0 3 1 2 1 3",
        "4 4 0 2 0 3 1 2 3 1",
        "4 4 0 2 0 3 2 1 3 1",
        "4 5 0 2 0 3 1 2 1 3 2 3",
        "4 5 0 2 0 3 1 2 3 1 3 2",
        "4 5 0 2 0 3 2 1 3 1 2 3",
        "4 5 0 2 3 0 1 2 3 1 3 2",
        "4 5 0 2 3 0 2 1 3 1 3 2",
        "4 5 2 0 3 0 2 1 3 1 2 3",
        "4 6 0 1 0 2 0 3 1 2 1 3 2 3",
    ],
}


# ── test verify_nauty ────────────────────────────────────────────────────────────────────

"""
verify_nauty(nauty_dir) checks that:
  - nauty_dir exists and is a directory   → NotADirectoryError otherwise
  - geng and directg are present          → FileNotFoundError otherwise
  - geng and directg are executable       → PermissionError otherwise
"""

def test_nonexistent_directory(tmp_path):
    missing = tmp_path / 'no_such_dir'
    with pytest.raises(NotADirectoryError):
        verify_nauty(str(missing))


def test_path_is_file_not_directory(tmp_path):
    f = tmp_path / 'not_a_dir'
    f.write_text('')
    with pytest.raises(NotADirectoryError):
        verify_nauty(str(f))


def test_missing_geng(tmp_path):
    # directg present, geng absent
    _touch(str(tmp_path / 'directg'))
    with pytest.raises(FileNotFoundError):
        verify_nauty(str(tmp_path))


def test_missing_directg(tmp_path):
    # geng present, directg absent
    _touch(str(tmp_path / 'geng'))
    with pytest.raises(FileNotFoundError):
        verify_nauty(str(tmp_path))


def test_geng_not_executable(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=False)
    _touch(str(tmp_path / 'directg'), executable=True)
    with pytest.raises(PermissionError):
        verify_nauty(str(tmp_path))


def test_directg_not_executable(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=True)
    _touch(str(tmp_path / 'directg'), executable=False)
    with pytest.raises(PermissionError):
        verify_nauty(str(tmp_path))


def test_neither_binary_executable(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=False)
    _touch(str(tmp_path / 'directg'), executable=False)
    with pytest.raises(PermissionError):
        verify_nauty(str(tmp_path))


def test_valid_directory(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=True)
    _touch(str(tmp_path / 'directg'), executable=True)
    # should not raise
    verify_nauty(str(tmp_path))


def test_extra_files_do_not_matter(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=True)
    _touch(str(tmp_path / 'directg'), executable=True)
    _touch(str(tmp_path / 'unrelated'), executable=False)
    verify_nauty(str(tmp_path))


def test_error_message_includes_path(tmp_path):
    """Error messages should identify the problematic path."""
    missing = tmp_path / 'absent'
    with pytest.raises(NotADirectoryError, match=str(missing)):
        verify_nauty(str(missing))


def test_filenotfound_message_includes_binary_name(tmp_path):
    # only directg present — geng missing → error mentions 'geng'
    _touch(str(tmp_path / 'directg'))
    with pytest.raises(FileNotFoundError, match='geng'):
        verify_nauty(str(tmp_path))


def test_permissionerror_message_includes_binary_name(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=False)
    _touch(str(tmp_path / 'directg'), executable=True)
    with pytest.raises(PermissionError, match='geng'):
        verify_nauty(str(tmp_path))


def test_valid_directory_returns_paths(tmp_path):
    _touch(str(tmp_path / 'geng'))
    _touch(str(tmp_path / 'directg'))
    assert verify_nauty(str(tmp_path)) == {
        'geng': str(tmp_path / 'geng'),
        'directg': str(tmp_path / 'directg'),
    }


def test_prefixed_names_accepted(tmp_path):
    """Debian/Ubuntu's apt package installs nauty-geng and nauty-directg."""
    _touch(str(tmp_path / 'nauty-geng'))
    _touch(str(tmp_path / 'nauty-directg'))
    assert verify_nauty(str(tmp_path)) == {
        'geng': str(tmp_path / 'nauty-geng'),
        'directg': str(tmp_path / 'nauty-directg'),
    }


def test_plain_name_preferred_over_prefixed(tmp_path):
    for name in ('geng', 'nauty-geng', 'directg', 'nauty-directg'):
        _touch(str(tmp_path / name))
    assert verify_nauty(str(tmp_path)) == {
        'geng': str(tmp_path / 'geng'),
        'directg': str(tmp_path / 'directg'),
    }


def test_executable_prefixed_name_used_when_plain_name_not_executable(tmp_path):
    _touch(str(tmp_path / 'geng'), executable=False)
    _touch(str(tmp_path / 'nauty-geng'))
    _touch(str(tmp_path / 'directg'))
    assert verify_nauty(str(tmp_path))['geng'] == str(tmp_path / 'nauty-geng')

# ── test find_nauty ─────────────────────────────────────────────────────────────────

@pytest.fixture
def nauty_search_env(tmp_path, monkeypatch):
    """Isolate find_nauty from the real machine: no $NAUTY, and a PATH holding only an
    empty `bin` directory. Tests add binaries to `bin` as needed."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.delenv('NAUTY', raising=False)
    monkeypatch.setenv('PATH', str(bin_dir))
    return bin_dir


def _install_nauty(directory, prefix=''):
    os.makedirs(directory, exist_ok=True)
    _touch(os.path.join(directory, prefix + 'geng'))
    _touch(os.path.join(directory, prefix + 'directg'))


def test_find_nauty_uses_explicit_dir(tmp_path, nauty_search_env):
    _install_nauty(str(tmp_path / 'explicit'))
    assert find_nauty(str(tmp_path / 'explicit'))['geng'] == str(tmp_path / 'explicit' / 'geng')


def test_find_nauty_uses_nauty_env_var(tmp_path, nauty_search_env, monkeypatch):
    _install_nauty(str(tmp_path / 'env'))
    monkeypatch.setenv('NAUTY', str(tmp_path / 'env'))
    assert find_nauty()['geng'] == str(tmp_path / 'env' / 'geng')


def test_find_nauty_explicit_dir_does_not_fall_back_to_path(tmp_path, nauty_search_env):
    """A directory the user named is used as-is, so a typo is reported instead of
    silently running some other nauty from PATH."""
    _install_nauty(str(nauty_search_env))
    with pytest.raises(NotADirectoryError):
        find_nauty(str(tmp_path / 'typo'))


@pytest.mark.parametrize('prefix', ['', 'nauty-'])
def test_find_nauty_falls_back_to_path(nauty_search_env, prefix):
    bin_dir = nauty_search_env
    _install_nauty(str(bin_dir), prefix=prefix)
    assert find_nauty() == {
        'geng': str(bin_dir / f'{prefix}geng'),
        'directg': str(bin_dir / f'{prefix}directg'),
    }


def test_find_nauty_ignores_home_nauty2_9_3(tmp_path, nauty_search_env, monkeypatch):
    """~/nauty2_9_3 used to be searched by default; it no longer is."""
    monkeypatch.setenv('HOME', str(tmp_path))
    _install_nauty(str(tmp_path / 'nauty2_9_3'))
    with pytest.raises(FileNotFoundError):
        find_nauty()


def test_find_nauty_not_found_anywhere(nauty_search_env):
    with pytest.raises(FileNotFoundError, match='NAUTY'):
        find_nauty()

 # ── test generate_nauty_digraphs ────────────────────────────────────────────────

def _mock_popen_proc():
    """A fake Popen result with empty stdout/stderr and a clean exit -- enough for
    generate_nauty_digraphs's eager (pre-iteration) setup to run without touching a
    real process."""
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO("")
    mock_proc.stderr = io.StringIO("")
    mock_proc.wait.return_value = 0
    return mock_proc


def test_generate_nauty_digraphs_calls_find_nauty_with_nauty_dir(tmp_path):
    """generate_nauty_digraphs must locate the nauty binaries (via find_nauty, which
    tests above cover in detail) before running the geng | directg pipeline -- so a
    bad/missing install fails fast with a clear error instead of a cryptic subprocess
    failure. find_nauty runs eagerly (generate_nauty_digraphs isn't itself a generator
    function -- it calls find_nauty and subprocess.Popen directly, only wrapping the
    stdout-reading loop in a nested generator), so simply calling the function -- with
    no iteration -- is enough to trigger it.
    """
    found = {'geng': '/found/nauty-geng', 'directg': '/found/nauty-directg'}

    # 1. nauty_dir is passed through as-is, and the paths find_nauty returns are the
    # ones the pipeline runs (prefixed names included).
    for nauty_dir in (str(tmp_path), None):
        with patch('rysy_dl.circuit_generator.find_nauty', return_value=found) as mock_find, \
             patch('rysy_dl.circuit_generator.subprocess.Popen', return_value=_mock_popen_proc()) as mock_popen:
            generate_nauty_digraphs(2, nauty_dir=nauty_dir)

        mock_find.assert_called_once_with(nauty_dir)
        cmd = mock_popen.call_args.args[0]
        assert cmd[-1] == '/found/nauty-geng 2 | /found/nauty-directg -aTq'

    # 2. If find_nauty fails, the pipeline must never be started -- the exception
    # should propagate straight out of generate_nauty_digraphs itself, since
    # find_nauty runs before subprocess.Popen and before any generator is handed
    # back to the caller.
    with patch('rysy_dl.circuit_generator.find_nauty', side_effect=FileNotFoundError('missing geng')) as mock_find, \
         patch('rysy_dl.circuit_generator.subprocess.Popen') as mock_popen:
        with pytest.raises(FileNotFoundError, match='missing geng'):
            generate_nauty_digraphs(2, nauty_dir=str(tmp_path))

    mock_find.assert_called_once_with(str(tmp_path))
    mock_popen.assert_not_called()


def test_generate_nauty_digraphs_quotes_paths_with_spaces(tmp_path):
    found = {'geng': '/my nauty/geng', 'directg': '/my nauty/directg'}
    with patch('rysy_dl.circuit_generator.find_nauty', return_value=found), \
         patch('rysy_dl.circuit_generator.subprocess.Popen', return_value=_mock_popen_proc()) as mock_popen:
        generate_nauty_digraphs(2)

    assert mock_popen.call_args.args[0][-1] == "'/my nauty/geng' 2 | '/my nauty/directg' -aTq"


def test_generate_nauty_digraphs_passes_correct_parameters_and_yields_digraphs(tmp_path):
    """generate_nauty_digraphs() should yield exactly the lines geng | directg write to
    stdout, stripped of whitespace -- and nothing else. In particular, the ">A ..." /
    ">Z ... graphs generated ..." progress messages that geng writes (visible above only
    because the terminal interleaves stdout and stderr) go to stderr, not stdout, and
    must never show up in the yielded results.

    Sample data (`GENG_DIRECTG_SAMPLES`) is real ``geng N | directg -aTq`` output for
    N = 2, 3, 4, captured from ~/Code/nauty2_9_3.
    """
    _touch(str(tmp_path / 'geng'), executable=True)
    _touch(str(tmp_path / 'directg'), executable=True)

    for num_nodes, expected_lines in GENG_DIRECTG_SAMPLES.items():
        # geng's own ">A"/">Z" progress messages are stderr output; stdout carries only
        # the digraph data lines, so that's all the mocked pipeline's stdout contains.
        stdout_text = "\n".join(expected_lines) + "\n"

        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO(stdout_text)
        mock_proc.stderr = io.StringIO("")
        mock_proc.wait.return_value = 0

        with patch('rysy_dl.circuit_generator.subprocess.Popen', return_value=mock_proc) as mock_popen:
            result = list(generate_nauty_digraphs(num_nodes, nauty_dir=str(tmp_path)))

        assert result == expected_lines

        cmd = mock_popen.call_args.args[0]
        assert f'{tmp_path}/geng {num_nodes}' in cmd[-1]
        assert f'{tmp_path}/directg -aTq' in cmd[-1]


def test_generate_nauty_digraphs_reads_stdout_lazily(tmp_path):
    """generate_nauty_digraphs() must stream lines from the geng | directg pipeline
    lazily rather than reading all of stdout up front. This is load-bearing, not
    just an implementation detail: generate_circuits (circuit_generator.py) chains
    generate_nauty_digraphs into a pipeline of map()/filter() generators that feeds
    a for-loop writing straight to the database, one digraph at a time -- the whole
    point being to avoid buffering nauty's (combinatorially large) enumeration in
    memory. A refactor that collected stdout into a list internally would silently
    reintroduce that memory blowup without breaking any of the other tests, which
    only check the final collected results, not when each line is pulled.
    """
    _touch(str(tmp_path / 'geng'), executable=True)
    _touch(str(tmp_path / 'directg'), executable=True)

    lines = ['line1', 'line2', 'line3']
    read_count = {'n': 0}

    def tracking_stdout():
        for line in lines:
            read_count['n'] += 1
            yield line + '\n'

    mock_proc = MagicMock()
    mock_proc.stdout = tracking_stdout()
    mock_proc.stderr = io.StringIO("")
    mock_proc.wait.return_value = 0

    with patch('rysy_dl.circuit_generator.subprocess.Popen', return_value=mock_proc):
        result = generate_nauty_digraphs(2, nauty_dir=str(tmp_path))

        assert inspect.isgenerator(result)
        assert read_count['n'] == 0  # calling it alone must not pull any lines

        assert next(result) == 'line1'
        assert read_count['n'] == 1  # only the first line was consumed

        assert list(result) == ['line2', 'line3']
        assert read_count['n'] == 3


@pytest.mark.skipif(
    not _real_nauty_available(),
    reason="real geng/directg binaries not found (set $NAUTY, or put them on PATH)",
)
def test_generate_nauty_digraphs_integration_with_real_geng_directg():
    """
    Integration test: runs the actual geng | directg pipeline (no mocking) and checks
    its output against the exact lines in `GENG_DIRECTG_SAMPLES` -- the same known-good
    data the mocked unit test above pins its parsing logic to. This is a stronger check
    than merely counting results or verifying they parse into valid DAGs: it also
    catches a malformed, duplicated, or missing line that a structural check would miss.

    Both sides are sorted before comparing, since nauty's enumeration order isn't a
    contract worth pinning down -- only the *set* of DAGs it produces is. That keeps
    this test from failing over a harmless reordering (e.g. from a different nauty
    version or build) while still catching an actual content mismatch.
    """
    for num_nodes, expected_lines in GENG_DIRECTG_SAMPLES.items():
        result = list(generate_nauty_digraphs(num_nodes))
        assert sorted(result) == sorted(expected_lines)

# ── test nauty_to_networkx ─────────────────────────────────────────────

def test_no_edges():
    """A line with 0 edges should produce a graph with no edges, but with all of
    its (isolated) nodes still present -- e.g. "3 0" is the edgeless graph on
    nodes 0, 1, 2."""
    G = nauty_to_networkx("3 0")
    assert sorted(G.nodes()) == [0, 1, 2]
    assert list(G.edges()) == []


def test_single_edge():
    G = nauty_to_networkx("2 1 0 1")
    assert sorted(G.nodes()) == [0, 1]
    assert list(G.edges()) == [(0, 1)]


def test_multiple_edges_and_direction_is_preserved():
    """Each `u v` pair is a directed edge u -> v; edge direction (not just the
    unordered pair) must be preserved, since nauty_to_networkx builds a DiGraph."""
    G = nauty_to_networkx("3 2 0 2 1 2")
    assert sorted(G.nodes()) == [0, 1, 2]
    assert sorted(G.edges()) == [(0, 2), (1, 2)]
    # The reverse edges were not added.
    assert not G.has_edge(2, 0)
    assert not G.has_edge(2, 1)


def test_returns_a_networkx_digraph():
    G = nauty_to_networkx("2 1 0 1")
    assert isinstance(G, nx.DiGraph)


def test_node_count_matches_declared_vertex_count_even_with_no_edges():
    """Nodes 0..nv-1 must all be present even when none of them appear in an
    edge pair -- node presence comes from `nv`, not from being mentioned as an
    edge endpoint."""
    G = nauty_to_networkx("4 0")
    assert sorted(G.nodes()) == [0, 1, 2, 3]


@pytest.mark.parametrize("num_nodes,line", [
    (num_nodes, line)
    for num_nodes, lines in GENG_DIRECTG_SAMPLES.items()
    for line in lines
])
def test_parses_all_sample_lines_without_error(num_nodes, line):
    """Every line generate_nauty_digraphs can actually produce (the real
    geng/directg samples used elsewhere in this file) must parse into a DiGraph
    with the declared number of nodes and edges."""
    parts = line.split()
    declared_nv, declared_ne = int(parts[0]), int(parts[1])

    G = nauty_to_networkx(line)

    assert G.number_of_nodes() == declared_nv == num_nodes
    assert G.number_of_edges() == declared_ne


# no_collapsable_children itself is now tested in test_circuit_filters.py;
# the remaining reference to it below only exercises its use as a filter
# passed into all_gate_assignment.

# ── test _add_single_sink ─────────────────────────────────────────────

def test_adds_node_to_empty_graph():
    """An empty graph has no out_degree-0 nodes to connect from, so the new sink
    node (id 0, since len(graph) == 0) is added with no incoming edges."""
    G = nx.DiGraph()
    result = _add_single_sink(G)
    assert list(result.nodes()) == [0]
    assert list(result.edges()) == []


def test_connects_single_isolated_node():
    G = nx.DiGraph()
    G.add_node(0)
    result = _add_single_sink(G)
    assert sorted(result.nodes()) == [0, 1]
    assert list(result.edges()) == [(0, 1)]


def test_connects_all_out_degree_zero_nodes_to_new_sink():
    """A disconnected/multi-sink graph -- node 1 (end of the 0->1 chain) and the
    isolated node 2 both have out_degree 0 -- must have both routed into the new
    sink node so the result has exactly one sink."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_node(2)
    result = _add_single_sink(G)
    assert sorted(result.nodes()) == [0, 1, 2, 3]
    assert sorted(result.edges()) == [(0, 1), (1, 3), (2, 3)]


def test_does_not_add_extra_edge_for_node_with_existing_out_edge():
    """Node 0 already has an outgoing edge (out_degree > 0), so it must not gain a
    second edge to the new sink -- only true out_degree-0 nodes get connected."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    result = _add_single_sink(G)
    assert result.out_degree(0) == 1
    assert list(result.successors(0)) == [1]


def test_node_with_out_degree_zero_is_connected_even_if_it_has_in_edges():
    """A node that is already the sink of an inner sub-circuit (has in-edges, but
    out_degree 0) must still be wired to the new global sink -- out_degree, not
    in_degree, is what determines whether a node needs connecting."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    result = _add_single_sink(G)
    assert sorted(result.nodes()) == [0, 1, 2, 3]
    assert sorted(result.edges()) == [(0, 2), (1, 2), (2, 3)]


def test_new_sink_id_is_one_past_original_node_count():
    """The new node's id must be len(graph) as measured *before* it is added, so it
    doesn't collide with any existing (0-indexed, contiguous) node id."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    result = _add_single_sink(G)
    assert 4 in result.nodes()
    assert result.number_of_nodes() == 5


def test_result_has_exactly_one_sink_node():
    """The whole point of _add_single_sink: after the call, exactly one node in the
    graph has out_degree 0, regardless of how many sinks the input graph had."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_node(3)  # a second, unrelated isolated sink
    result = _add_single_sink(G)
    sinks = [n for n in result.nodes() if result.out_degree(n) == 0]
    assert sinks == [4]


def test_preserves_existing_edges_unchanged():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    original_edges = set(G.edges())
    result = _add_single_sink(G)
    assert original_edges <= set(result.edges())


def test_mutates_and_returns_the_same_graph_object():
    """_add_single_sink mutates its argument in place (unlike the filter/generator
    functions elsewhere in this module, which operate on copies) -- the returned
    object must be the exact same graph instance the caller passed in."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    result = _add_single_sink(G)
    assert result is G


# ── test assign_all_gates_and ─────────────────────────────────────────────

"""
assign_all_gates_and(graph) labels every node in place:
  - in_degree == 0 (a source/input node)     -> gate = 'src'
  - in_degree > 0  (every other node)        -> gate = 'AND'
It relies on node ids being the contiguous range 0..len(graph)-1, matching the
rest of this module's graph convention (nauty_to_networkx, _add_single_sink).
"""


def test_source_nodes_labeled_src():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    assign_all_gates_and(G)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'src'


def test_non_source_node_labeled_and():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    assign_all_gates_and(G)
    assert G.nodes[2]['gate'] == 'AND'


def test_assign_all_gates_and_every_node_gets_a_gate_label():
    """No node should be left without a 'gate' attribute after assignment."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    assign_all_gates_and(G)
    for node in G.nodes():
        assert 'gate' in G.nodes[node]


def test_multiple_non_source_layers_all_get_and():
    """A node whose in-edges come from another gate node (not just from raw
    sources) is still non-source, and must also be labeled 'AND'."""
    G = nx.DiGraph()
    G.add_edge(0, 2)  # sources 0, 1 -> inner gate 2
    G.add_edge(1, 2)
    G.add_edge(2, 3)  # inner gate 2, source 4 -> outer gate 3
    G.add_edge(4, 3)
    assign_all_gates_and(G)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'src'
    assert G.nodes[4]['gate'] == 'src'
    assert G.nodes[2]['gate'] == 'AND'
    assert G.nodes[3]['gate'] == 'AND'


def test_fan_in_does_not_affect_gate_choice():
    """A node's gate depends only on whether it has any in-edges at all, not on
    how many -- a 3-input AND still gets labeled the same as a 2-input one."""
    G = nx.DiGraph()
    G.add_edge(0, 3)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    assign_all_gates_and(G)
    assert G.nodes[3]['gate'] == 'AND'
    assert all(G.nodes[n]['gate'] == 'src' for n in (0, 1, 2))


def test_isolated_node_with_no_edges_is_treated_as_source():
    """A node with in_degree 0 counts as a source even if it also has no
    out-edges (i.e. it isn't wired to anything yet)."""
    G = nx.DiGraph()
    G.add_node(0)
    assign_all_gates_and(G)
    assert G.nodes[0]['gate'] == 'src'


def test_empty_graph_does_not_raise():
    G = nx.DiGraph()
    assign_all_gates_and(G)  # should simply do nothing
    assert list(G.nodes()) == []


def test_existing_gate_attribute_is_overwritten():
    """Any pre-existing 'gate' value must be replaced, not merged with or left
    alone -- assign_all_gates_and is a full reassignment, not a patch."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.nodes[0]['gate'] = 'OR'
    G.nodes[1]['gate'] = 'NOR'
    assign_all_gates_and(G)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'AND'


def test_mutates_in_place_and_returns_none():
    """assign_all_gates_and has no return statement -- callers rely on it mutating
    the graph they passed in, not on a return value."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    result = assign_all_gates_and(G)
    assert result is None
    assert G.nodes[1]['gate'] == 'AND'  # the mutation happened on the same object


def test_edges_are_unchanged():
    """assign_all_gates_and only sets node attributes -- it must not add, remove,
    or otherwise alter any edges."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    original_edges = sorted(G.edges())
    assign_all_gates_and(G)
    assert sorted(G.edges()) == original_edges


# ── test assign_gates_randomly_ensure_diversity ─────────────────────────────────────────────

"""
assign_gates_randomly_ensure_diversity(graph, include_xnor=False) labels every node in place:
  - in_degree == 0 (a source node)  -> gate = 'src'
  - in_degree > 0  (a gate node)    -> gate = one of AND/OR/XOR/NAND/NOR (+XNOR if include_xnor)

num_gates (from num_sources_and_gates) selects one of four composition strategies. Several of
the boundary values used below make the "extra" gates sampled from a pool exactly the same
size as the pool itself -- at those sizes, random.sample necessarily returns the whole
population, so the resulting *multiset* of gate labels is deterministic even though
random.sample/random.choice are real, unmocked calls.

XOR/XNOR are only valid on a fan-in-2 node: if the initial random placement puts one on a
node whose fan-in isn't 2, it gets swapped with a fan-in-2 node (or, if none is available,
replaced outright with AND/OR). `_graph_with_num_gates` gives every gate node fan-in 2 so
that these composition-strategy tests aren't entangled with that swap logic; the swap/fallback
behavior itself is covered separately, below.
"""

ALL_SIMPLE = {'AND', 'OR', 'XOR'}
ALL_TRICKY_NO_XNOR = {'NAND', 'NOR'}
ALL_TRICKY_XNOR = {'NAND', 'NOR', 'XNOR'}


def _graph_with_num_gates(num_gates):
    """Two sources (nodes 0, 1) wired into `num_gates` independent gate nodes
    (nodes 2..num_gates+1), so num_sources_and_gates reports exactly `num_gates`
    gates, each with fan-in 2 (keeping XOR/XNOR fan-in-valid everywhere)."""
    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)
    for gate_node in range(2, num_gates + 2):
        G.add_edge(0, gate_node)
        G.add_edge(1, gate_node)
    return G


def _gate_labels(graph):
    """The 'gate' attribute of every non-source node, as a list (one entry per gate node)."""
    return [graph.nodes[n]['gate'] for n in graph.nodes() if graph.in_degree(n) != 0]


def test_source_node_labeled_src():
    G = _graph_with_num_gates(3)
    assign_gates_randomly_ensure_diversity(G)
    assert G.nodes[0]['gate'] == 'src'


def test_every_node_gets_a_gate_label():
    G = _graph_with_num_gates(5)
    assign_gates_randomly_ensure_diversity(G)
    for node in G.nodes():
        assert 'gate' in G.nodes[node]


def test_returns_none_and_mutates_in_place():
    G = _graph_with_num_gates(4)
    result = assign_gates_randomly_ensure_diversity(G)
    assert result is None
    gate_node = next(n for n in G.nodes() if G.in_degree(n) != 0)
    assert G.nodes[gate_node]['gate'] in ALL_SIMPLE | ALL_TRICKY_NO_XNOR


def test_edges_are_unchanged_by_gate_assignment():
    G = _graph_with_num_gates(6)
    original_edges = sorted(G.edges())
    assign_gates_randomly_ensure_diversity(G)
    assert sorted(G.edges()) == original_edges


def test_empty_gate_graph_does_not_raise():
    """A graph with zero gate nodes (only sources) should produce no gate labels
    and must not raise despite num_gates == 0."""
    G = _graph_with_num_gates(0)
    assign_gates_randomly_ensure_diversity(G)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'src'
    assert _gate_labels(G) == []


@pytest.mark.parametrize("num_gates", range(0, 13))
def test_labels_come_from_allowed_pool_without_xnor(num_gates):
    """Across the whole supported range (0..12 without XNOR), every gate label must
    come from the five allowed gate types, and XNOR must never appear since it isn't
    in the pool unless include_xnor=True."""
    G = _graph_with_num_gates(num_gates)
    assign_gates_randomly_ensure_diversity(G, include_xnor=False)
    labels = _gate_labels(G)
    assert len(labels) == num_gates
    assert set(labels) <= (ALL_SIMPLE | ALL_TRICKY_NO_XNOR)


@pytest.mark.parametrize("num_gates", range(0, 14))
def test_labels_come_from_allowed_pool_with_xnor(num_gates):
    """Same as above, but across the wider 0..13 range that opens up when
    include_xnor=True."""
    G = _graph_with_num_gates(num_gates)
    assign_gates_randomly_ensure_diversity(G, include_xnor=True)
    labels = _gate_labels(G)
    assert len(labels) == num_gates
    assert set(labels) <= (ALL_SIMPLE | ALL_TRICKY_XNOR)


def test_small_circuits_use_each_simple_gate_at_most_once():
    """num_gates <= 3 samples without replacement from just the 3 simple gates, so
    no gate type should repeat."""
    for num_gates in (1, 2, 3):
        G = _graph_with_num_gates(num_gates)
        assign_gates_randomly_ensure_diversity(G)
        labels = _gate_labels(G)
        assert len(labels) == len(set(labels))
        assert set(labels) <= ALL_SIMPLE


def test_three_gates_uses_exactly_the_three_simple_gates():
    """At num_gates == len(simple_gates), random.sample draws the whole population,
    so the exact multiset is deterministic regardless of the random draw."""
    G = _graph_with_num_gates(3)
    assign_gates_randomly_ensure_diversity(G)
    assert Counter(_gate_labels(G)) == Counter(['AND', 'OR', 'XOR'])


def test_four_gates_uses_exactly_one_tricky_gate_with_no_repeats():
    """num_gates == 4 (below the num_gate_choices boundary): all 3 simple gates
    plus exactly 1 tricky gate, no repeats."""
    G = _graph_with_num_gates(4)
    assign_gates_randomly_ensure_diversity(G)
    labels = _gate_labels(G)
    assert len(labels) == len(set(labels)) == 4
    assert set(labels) & ALL_SIMPLE == ALL_SIMPLE
    assert len(set(labels) & ALL_TRICKY_NO_XNOR) == 1


def test_five_gates_without_xnor_uses_each_gate_type_exactly_once():
    """num_gates == num_gate_choices (5, no XNOR): simple gates are used wholesale,
    and the tricky-gate sample size (2) equals the tricky pool size (2), so the
    outcome is deterministic: one of each of the 5 gate types."""
    G = _graph_with_num_gates(5)
    assign_gates_randomly_ensure_diversity(G, include_xnor=False)
    assert Counter(_gate_labels(G)) == Counter(['AND', 'OR', 'XOR', 'NAND', 'NOR'])


def test_six_gates_with_xnor_uses_each_gate_type_exactly_once():
    """Same boundary as above, but with XNOR enabled (num_gate_choices == 6)."""
    G = _graph_with_num_gates(6)
    assign_gates_randomly_ensure_diversity(G, include_xnor=True)
    assert Counter(_gate_labels(G)) == Counter(['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR'])


def test_seven_gates_caps_simple_gate_repetition_at_two():
    """num_gates == 7 (between the two full-population boundaries, no XNOR): NAND
    and NOR are each used exactly once, and the 2 padding slots go to 2 of the 3
    simple gates (at random), so simple-gate counts are some permutation of
    [1, 2, 2] -- no simple gate can ever repeat more than twice in this branch."""
    G = _graph_with_num_gates(7)
    assign_gates_randomly_ensure_diversity(G, include_xnor=False)
    counts = Counter(_gate_labels(G))
    assert counts['NAND'] == 1
    assert counts['NOR'] == 1
    assert sorted(counts[g] for g in ALL_SIMPLE) == [1, 2, 2]


def test_eight_gates_without_xnor_doubles_each_simple_gate():
    """num_gates == num_gate_choices + len(simple_gates) (8, no XNOR): the padding
    sample size (3) equals the simple-gate pool size (3), so each simple gate is
    guaranteed to appear exactly twice (once as a base gate, once from padding),
    while NAND and NOR each appear exactly once."""
    G = _graph_with_num_gates(8)
    assign_gates_randomly_ensure_diversity(G, include_xnor=False)
    counts = Counter(_gate_labels(G))
    assert counts['AND'] == counts['OR'] == counts['XOR'] == 2
    assert counts['NAND'] == counts['NOR'] == 1
    assert sum(counts.values()) == 8


def test_twelve_gates_without_xnor_is_the_maximum_supported():
    """num_gates == max_gates_supported (12, no XNOR): the final padding sample
    size (6) equals the size of simple_gates+simple_gates (6), so each simple gate
    is deterministically tripled. NAND/NOR together get exactly 3 (one each from
    the base pool, plus one more split randomly between them), so each has at
    least 1 and together they sum to 3."""
    G = _graph_with_num_gates(12)
    assign_gates_randomly_ensure_diversity(G, include_xnor=False)
    counts = Counter(_gate_labels(G))
    assert counts['AND'] == counts['OR'] == counts['XOR'] == 3
    assert counts['NAND'] >= 1 and counts['NOR'] >= 1
    assert counts['NAND'] + counts['NOR'] == 3
    assert sum(counts.values()) == 12


def test_thirteen_gates_with_xnor_is_the_maximum_supported():
    """Same boundary as above, but with XNOR enabled (max_gates_supported == 13).
    XNOR is part of the base pool and never gets the "extra tricky gate" pick
    (which is drawn only from ['NAND', 'NOR']), so it's guaranteed to appear
    exactly once."""
    G = _graph_with_num_gates(13)
    assign_gates_randomly_ensure_diversity(G, include_xnor=True)
    counts = Counter(_gate_labels(G))
    assert counts['AND'] == counts['OR'] == counts['XOR'] == 3
    assert counts['XNOR'] == 1
    assert counts['NAND'] + counts['NOR'] == 3
    assert sum(counts.values()) == 13


def test_raises_when_too_many_gates_without_xnor():
    G = _graph_with_num_gates(13)
    with pytest.raises(ValueError, match='up to 12 gates without XNOR'):
        assign_gates_randomly_ensure_diversity(G, include_xnor=False)


def test_raises_when_too_many_gates_with_xnor():
    G = _graph_with_num_gates(14)
    with pytest.raises(ValueError, match='up to 13 gates with XNOR'):
        assign_gates_randomly_ensure_diversity(G, include_xnor=True)


def test_error_message_reports_the_actual_gate_count():
    G = _graph_with_num_gates(13)
    with pytest.raises(ValueError, match='got 13'):
        assign_gates_randomly_ensure_diversity(G, include_xnor=False)


def test_sources_interspersed_with_gates():
    """Node ids don't need to have all sources first -- in_degree alone determines
    src vs. gate, matching the convention the rest of the module relies on."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(4, 3)
    assign_gates_randomly_ensure_diversity(G)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'src'
    assert G.nodes[4]['gate'] == 'src'
    assert G.nodes[2]['gate'] in ALL_SIMPLE | ALL_TRICKY_NO_XNOR
    assert G.nodes[3]['gate'] in ALL_SIMPLE | ALL_TRICKY_NO_XNOR


# ── test assign_gates_randomly_ensure_diversity: XOR/XNOR fan-in swap ───────────────────────

"""
XOR/XNOR are only valid on a fan-in-2 node. If the initial (fan-in-blind) placement puts one
on a node whose fan-in isn't 2, assign_gates_randomly_ensure_diversity swaps it with a
fan-in-2 node holding a different gate; if no such node exists, it replaces the gate outright
with AND or OR. Both branches depend on *where* the random placement happens to land XOR/XNOR,
so these run many trials and check the invariant rather than a single fixed outcome.
"""


def test_xor_swapped_away_from_non_two_input_node():
    """num_gates == 3 deterministically chooses {AND, OR, XOR} (random.sample draws the
    whole population), so XOR is guaranteed every run -- only its landing node is random.
    Node 5 (fan-in 3) must never end up with XOR; wherever XOR does land must be fan-in 2."""
    for _ in range(200):
        G = nx.DiGraph()
        G.add_edge(0, 3)
        G.add_edge(1, 3)  # node 3: fan-in 2
        G.add_edge(1, 4)
        G.add_edge(2, 4)  # node 4: fan-in 2
        G.add_edge(0, 5)
        G.add_edge(1, 5)
        G.add_edge(2, 5)  # node 5: fan-in 3
        assign_gates_randomly_ensure_diversity(G)
        assert G.nodes[5]['gate'] not in ('XOR', 'XNOR')
        for node in (3, 4, 5):
            if G.nodes[node]['gate'] in ('XOR', 'XNOR'):
                assert G.in_degree(node) == 2
        assert Counter(_gate_labels(G)) == Counter(['AND', 'OR', 'XOR'])


def test_xor_falls_back_to_and_or_when_no_two_input_node_available():
    """Same deterministic {AND, OR, XOR} set as above, but every gate node has fan-in 1,
    so no swap partner ever exists -- the would-be-XOR node must fall back to AND/OR
    rather than being left with an invalid gate."""
    for _ in range(200):
        G = nx.DiGraph()
        G.add_node(0)
        for gate_node in (1, 2, 3):
            G.add_edge(0, gate_node)  # every gate node: fan-in 1
        assign_gates_randomly_ensure_diversity(G)
        labels = _gate_labels(G)
        assert 'XOR' not in labels
        assert 'XNOR' not in labels
        assert set(labels) <= {'AND', 'OR'}


def test_only_one_two_input_node_absorbs_at_most_one_xor_or_xnor():
    """num_gates == 6 with include_xnor=True deterministically chooses one of each of
    AND/OR/XOR/NAND/NOR/XNOR. With only one fan-in-2 node among the six gates, at most
    one of XOR/XNOR can be swapped into it -- the other (if also misplaced) must fall
    back to AND/OR, so at most one gate node may end up labeled XOR or XNOR."""
    for _ in range(200):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(2, 1)  # node 1: fan-in 2 (the only one)
        for gate_node in (2, 3, 4, 5, 6):
            G.add_edge(0, gate_node)  # fan-in 1 each
        assign_gates_randomly_ensure_diversity(G, include_xnor=True)
        labels = _gate_labels(G)
        xor_xnor_count = sum(1 for g in labels if g in ('XOR', 'XNOR'))
        assert xor_xnor_count <= 1
        if xor_xnor_count == 1:
            xor_xnor_node = next(
                n for n in (1, 2, 3, 4, 5, 6) if G.nodes[n]['gate'] in ('XOR', 'XNOR')
            )
            assert G.in_degree(xor_xnor_node) == 2


# ── test assign_gates_randomly ───────────────────────────────────────────────────────────────

"""
assign_gates_randomly(netlist, ensure_diversity=True, gate_list=None) labels every node in place:
  - in_degree == 0 (a source node) -> gate = 'src'
  - in_degree > 0  (a gate node)   -> gate = a random entry from `gate_list`
                                       (default ["AND", "OR", "XOR", "NAND", "NOR"])
  - ensure_diversity=True delegates entirely to assign_gates_randomly_ensure_diversity
    and ignores gate_list.

As with the diversity function, XOR/XNOR must never land on a node whose fan-in isn't 2.
"""


def test_ensure_diversity_true_delegates_to_diversity_function(monkeypatch):
    """ensure_diversity=True must hand off to assign_gates_randomly_ensure_diversity
    (and gate_list must be ignored) rather than running the plain-random branch."""
    import rysy_dl.circuit_generator as cg

    mock_diversity = MagicMock()
    monkeypatch.setattr(cg, 'assign_gates_randomly_ensure_diversity', mock_diversity)

    G = _graph_with_num_gates(3)
    result = cg.assign_gates_randomly(G, ensure_diversity=True, gate_list=['AND'])

    mock_diversity.assert_called_once_with(G)
    assert result is None


def test_random_source_nodes_labeled_src():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    assign_gates_randomly(G, ensure_diversity=False)
    assert G.nodes[0]['gate'] == 'src'
    assert G.nodes[1]['gate'] == 'src'


def test_random_every_node_gets_a_gate_label():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(4, 3)
    assign_gates_randomly(G, ensure_diversity=False)
    for node in G.nodes():
        assert 'gate' in G.nodes[node]


def test_random_returns_none_and_mutates_in_place():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(2, 1)
    result = assign_gates_randomly(G, ensure_diversity=False)
    assert result is None
    assert G.nodes[1]['gate'] in {'AND', 'OR', 'XOR', 'NAND', 'NOR'}


def test_random_edges_are_unchanged():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    original_edges = sorted(G.edges())
    assign_gates_randomly(G, ensure_diversity=False)
    assert sorted(G.edges()) == original_edges


def test_random_empty_graph_does_not_raise():
    G = nx.DiGraph()
    assign_gates_randomly(G, ensure_diversity=False)
    assert list(G.nodes()) == []


def test_default_gate_list_labels_come_from_allowed_pool():
    G = nx.DiGraph()
    for gate_node in range(1, 6):
        G.add_edge(0, gate_node)
    for _ in range(50):
        assign_gates_randomly(G, ensure_diversity=False)
        labels = _gate_labels(G)
        assert set(labels) <= {'AND', 'OR', 'XOR', 'NAND', 'NOR'}


def test_custom_gate_list_is_respected():
    G = nx.DiGraph()
    for gate_node in range(1, 6):
        G.add_edge(0, gate_node)
    for _ in range(50):
        assign_gates_randomly(G, ensure_diversity=False, gate_list=['AND', 'OR'])
        labels = _gate_labels(G)
        assert set(labels) <= {'AND', 'OR'}


def test_gate_list_without_xor_or_xnor_ignores_fan_in():
    """When gate_list has no XOR/XNOR, fan-in is irrelevant -- a fan-in-3 node can
    freely receive any gate from the list."""
    G = nx.DiGraph()
    G.add_edge(0, 3)
    G.add_edge(1, 3)
    G.add_edge(2, 3)  # node 3: fan-in 3
    for _ in range(50):
        assign_gates_randomly(G, ensure_diversity=False, gate_list=['AND', 'OR', 'NAND', 'NOR'])
        assert G.nodes[3]['gate'] in {'AND', 'OR', 'NAND', 'NOR'}


def test_xor_and_xnor_never_assigned_to_non_two_input_node():
    """A node with fan-in != 2 must never be labeled XOR or XNOR, across many trials,
    even though the draw is otherwise uniformly random over the whole gate_list."""
    G = nx.DiGraph()
    G.add_edge(0, 3)
    G.add_edge(1, 3)  # node 3: fan-in 2
    G.add_edge(0, 4)
    G.add_edge(1, 4)
    G.add_edge(2, 4)  # node 4: fan-in 3
    G.add_edge(0, 5)  # node 5: fan-in 1
    gate_list = ['AND', 'OR', 'XOR', 'XNOR']
    for _ in range(300):
        assign_gates_randomly(G, ensure_diversity=False, gate_list=gate_list)
        assert G.nodes[4]['gate'] not in ('XOR', 'XNOR')
        assert G.nodes[5]['gate'] not in ('XOR', 'XNOR')


def test_xor_and_xnor_can_be_assigned_to_a_two_input_node():
    """Sanity check for the above: fan-in-2 nodes must still be eligible to draw
    XOR/XNOR -- the restriction should apply only to the wrong-fan-in nodes."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: fan-in 2
    gate_list = ['XOR', 'XNOR']
    seen = set()
    for _ in range(50):
        assign_gates_randomly(G, ensure_diversity=False, gate_list=gate_list)
        seen.add(G.nodes[2]['gate'])
    assert seen == {'XOR', 'XNOR'}


# ── test all_gate_assignment ─────────────────────────────────────────────

def test_source_nodes_always_labeled_src():
    """Every yielded assignment must label the graph's source nodes 'src',
    regardless of which gate combination was drawn for the other nodes."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: gate, fan-in 2
    for circuit in all_gate_assignment(G, gate_list=('AND', 'OR')):
        assert circuit.nodes[0]['gate'] == 'src'
        assert circuit.nodes[1]['gate'] == 'src'


def test_no_duplicate_assignments_from_source_node_slots():
    """Source nodes contribute no real degrees of freedom, so the number of
    yielded assignments must equal len(gate_list) ** (number of gate nodes),
    not ** (total node count), and every yielded gate choice must be unique."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: the only gate node, fan-in 2
    gate_list = ('AND', 'OR', 'XOR')
    results = list(all_gate_assignment(G, gate_list=gate_list))
    assert len(results) == len(gate_list)
    seen = {circuit.nodes[2]['gate'] for circuit in results}
    assert seen == {'AND', 'OR', 'XOR'}


def test_all_gate_assignment_xor_and_xnor_never_assigned_to_non_two_input_node():
    """A node with fan-in != 2 must never be labeled XOR or XNOR in any
    yielded assignment, including fan-in 3 and up."""
    G = nx.DiGraph()
    G.add_edge(0, 3)  # node 3: fan-in 1
    G.add_edge(0, 4)
    G.add_edge(1, 4)
    G.add_edge(2, 4)  # node 4: fan-in 3
    gate_list = ('AND', 'OR', 'XOR', 'XNOR')
    for circuit in all_gate_assignment(G, gate_list=gate_list):
        assert circuit.nodes[3]['gate'] not in ('XOR', 'XNOR')
        assert circuit.nodes[4]['gate'] not in ('XOR', 'XNOR')


def test_all_gate_assignment_xor_and_xnor_can_be_assigned_to_a_two_input_node():
    """Sanity check for the above: a fan-in-2 node must still be eligible to
    receive XOR/XNOR -- the restriction applies only to the wrong-fan-in nodes."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: fan-in 2
    gate_list = ('AND', 'XOR', 'XNOR')
    seen = {circuit.nodes[2]['gate'] for circuit in all_gate_assignment(G, gate_list=gate_list)}
    assert seen == {'AND', 'XOR', 'XNOR'}


def test_root_pruning_excludes_collapsable_parent_child_pairs():
    """The output (root) node must never end up paired with a child gate that
    would trivially collapse into it (AND/NAND -> AND child, OR/NOR -> OR child),
    across every yielded assignment -- and the exact combinatorics (4 root choices
    x 3 surviving choices for each of its two fan-in-2 gate children, since exactly
    one gate is excluded per child) must match by hand."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: gate, fan-in 2, child of root
    G.add_edge(3, 5)
    G.add_edge(4, 5)  # node 5: gate, fan-in 2, child of root
    G.add_edge(2, 6)
    G.add_edge(5, 6)  # node 6: root, fan-in 2, children are nodes 2 and 5

    gate_list = ('AND', 'OR', 'NAND', 'NOR')
    forbidden_child = {'AND': 'AND', 'NAND': 'AND', 'OR': 'OR', 'NOR': 'OR'}
    results = list(all_gate_assignment(G, gate_list=gate_list))

    assert len(results) == 4 * 3 * 3
    for circuit in results:
        root_gate = circuit.nodes[6]['gate']
        for child in (2, 5):
            assert circuit.nodes[child]['gate'] != forbidden_child[root_gate]


def test_root_pruning_does_not_reach_non_root_pairs():
    """Only the root's own children are pruned -- a collapsable pair further down
    the graph (here, node 4's gate matching its own child node 2's gate) is left
    for the caller's filter_list to catch, so it must still appear in the output."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)  # node 2: gate, fan-in 2 (sources 0, 1)
    G.add_edge(2, 4)
    G.add_edge(3, 4)  # node 4: gate, fan-in 2 -- parent of node 2, but not the root
    G.add_edge(4, 5)  # node 5: root, fan-in 1, child is node 4

    gate_list = ('AND', 'OR')
    results = list(all_gate_assignment(G, gate_list=gate_list))

    # root (5) and its only child (4) are pruned: 4 must never match 5's forbidden gate
    for circuit in results:
        root_gate = circuit.nodes[5]['gate']
        assert circuit.nodes[4]['gate'] != root_gate

    # but node 4 (parent) matching node 2 (its own child) is a collapsable pair one
    # level below the root, and must still show up since it isn't pruned
    assert any(
        circuit.nodes[4]['gate'] == 'AND' and circuit.nodes[2]['gate'] == 'AND'
        for circuit in results
    )


def test_filter_list_is_still_applied_after_pruning():
    """filter_list must still run against the (already root-pruned) assignments --
    passing no_collapsable_children here should remove the non-root collapsable
    pairs that root-pruning intentionally leaves in place."""
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(2, 4)
    G.add_edge(3, 4)
    G.add_edge(4, 5)

    gate_list = ('AND', 'OR')
    unfiltered = list(all_gate_assignment(G, gate_list=gate_list))
    filtered = list(all_gate_assignment(G, filter_list=[no_collapsable_children], gate_list=gate_list))

    assert len(filtered) < len(unfiltered)
    for circuit in filtered:
        assert not (circuit.nodes[4]['gate'] == 'AND' and circuit.nodes[2]['gate'] == 'AND')


def test_default_gate_list_is_used_when_not_specified():
    G = nx.DiGraph()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    results = list(all_gate_assignment(G))
    seen = {circuit.nodes[2]['gate'] for circuit in results}
    assert seen <= {'AND', 'OR', 'XOR', 'NAND', 'NOR'}
    assert len(seen) == 5


def test_isolated_node_graph_labeled_src_and_does_not_raise():
    """A single node with no edges (no root, no gate nodes at all) should yield
    exactly one circuit, labeled 'src', without error."""
    G = nx.DiGraph()
    G.add_node(0)
    results = list(all_gate_assignment(G))
    assert len(results) == 1
    assert results[0].nodes[0]['gate'] == 'src'


def test_all_gate_assignment_empty_graph_does_not_raise():
    G = nx.DiGraph()
    results = list(all_gate_assignment(G))
    assert len(results) == 1
    assert len(results[0].nodes) == 0


# ── test _not_node_inserter ─────────────────────────────────────────────

def test_all_false_sequence_leaves_graph_unchanged():
    """A sequence of all False (nothing to negate) must be a no-op: no NOT
    node added, no edges touched."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    original_nodes = set(G.nodes())
    original_edges = set(G.edges())
    _not_node_inserter(G, 0, (False, False))
    assert set(G.nodes()) == original_nodes
    assert set(G.edges()) == original_edges


def test_empty_sequence_is_a_no_op():
    """A source with no successors (out_degree 0) is called with an empty
    sequence; any(()) is False, so nothing happens."""
    G = nx.DiGraph()
    G.add_node(0)
    _not_node_inserter(G, 0, ())
    assert list(G.nodes()) == [0]
    assert list(G.edges()) == []


def test_all_true_sequence_routes_every_successor_through_one_not_gate():
    """Every successor gets rerouted through a single, shared NOT node, and
    the direct source->successor edges are removed."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    _not_node_inserter(G, 0, (True, True))
    not_node = 3  # len(graph) == 3 at the time the NOT node was created
    assert G.nodes[not_node]['gate'] == 'NOT'
    assert sorted(G.successors(0)) == [not_node]
    assert sorted(G.successors(not_node)) == [1, 2]
    assert (0, 1) not in G.edges()
    assert (0, 2) not in G.edges()


def test_mixed_sequence_negates_only_the_matching_successor():
    """The core behavior the rest of this module relies on: a source with two
    successors can have exactly one of the two edges negated, not just both
    or neither."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    _not_node_inserter(G, 0, (True, False))
    not_node = 3
    # successor 1 (index 0, sequence value True) is routed through the NOT gate
    assert (0, not_node) in G.edges()
    assert (not_node, 1) in G.edges()
    assert (0, 1) not in G.edges()
    # successor 2 (index 1, sequence value False) keeps its direct edge
    assert (0, 2) in G.edges()


def test_mixed_sequence_other_direction_negates_the_other_successor():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    _not_node_inserter(G, 0, (False, True))
    not_node = 3
    assert (0, 1) in G.edges()
    assert (0, not_node) in G.edges()
    assert (not_node, 2) in G.edges()
    assert (0, 2) not in G.edges()


def test_multiple_true_values_share_a_single_not_node():
    """Only one NOT node should ever be created per call, even when several
    successors need negating -- they all share it."""
    G = nx.DiGraph()
    for succ in (1, 2, 3):
        G.add_edge(0, succ)
    before = G.number_of_nodes()
    _not_node_inserter(G, 0, (True, True, True))
    assert G.number_of_nodes() == before + 1


def test_new_not_node_id_is_one_past_original_node_count():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    _not_node_inserter(G, 0, (True,))
    assert 2 in G.nodes()
    assert G.number_of_nodes() == 3


def test_source_node_attributes_are_not_touched():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.nodes[0]['gate'] = 'src'
    _not_node_inserter(G, 0, (True,))
    assert G.nodes[0]['gate'] == 'src'


def test_mutates_in_place_and_returns_none():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    result = _not_node_inserter(G, 0, (True,))
    assert result is None
    assert G.number_of_nodes() == 3  # mutated the same object passed in


# ── test insert_random_source_edge_nots ─────────────────────────────────

def test_empty_graph_is_a_no_op():
    G = nx.DiGraph()
    result = insert_random_source_edge_nots(G)
    assert result is None
    assert G.number_of_nodes() == 0


def test_source_with_no_out_edges_is_never_modified():
    """out_degree(node) == 0 means not_list is empty; _not_node_inserter's
    any(()) check makes this a guaranteed no-op regardless of randomness."""
    G = nx.DiGraph()
    G.add_node(0)
    insert_random_source_edge_nots(G)
    assert list(G.nodes()) == [0]
    assert list(G.edges()) == []


def test_only_source_nodes_edges_can_be_negated():
    """insert_random_source_edge_nots only ever calls _not_node_inserter on
    in_degree-0 nodes -- a non-source node's own outgoing edge must never be
    rerouted through a NOT gate, no matter what randomness picks for the
    sources feeding into it."""
    for _ in range(200):
        G = nx.DiGraph()
        G.add_edge(0, 2)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        insert_random_source_edge_nots(G)
        assert (2, 3) in G.edges()


def test_source_with_two_successors_can_negate_only_one_of_them():
    """Each out-edge of a source gets an independent random choice, so it
    must be possible for exactly one of two edges from the same source to be
    negated while the other stays direct -- not just all-or-nothing per
    source. Run enough trials that both asymmetric outcomes are observed;
    the odds of missing one for 300 independent fair-coin trials are
    astronomically small."""
    saw_only_first_negated = False
    saw_only_second_negated = False
    for _ in range(300):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        insert_random_source_edge_nots(G)
        first_negated = (0, 1) not in G.edges()
        second_negated = (0, 2) not in G.edges()
        if first_negated and not second_negated:
            saw_only_first_negated = True
        elif second_negated and not first_negated:
            saw_only_second_negated = True
        if saw_only_first_negated and saw_only_second_negated:
            break
    assert saw_only_first_negated
    assert saw_only_second_negated


def test_multiple_sources_are_negated_independently():
    """Two unrelated sources must not be forced to the same negation
    decision -- across enough trials, one gets negated while the other
    doesn't (in either direction)."""
    saw_mixed = False
    for _ in range(300):
        G = nx.DiGraph()
        G.add_edge(0, 2)
        G.add_edge(1, 3)
        insert_random_source_edge_nots(G)
        zero_negated = (0, 2) not in G.edges()
        one_negated = (1, 3) not in G.edges()
        if zero_negated != one_negated:
            saw_mixed = True
            break
    assert saw_mixed


def test_negated_edge_passes_through_a_not_labeled_node_and_mutates_in_place():
    for _ in range(200):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        result = insert_random_source_edge_nots(G)
        if G.number_of_nodes() == 3:
            assert result is None
            assert G.nodes[2]['gate'] == 'NOT'
            assert (0, 2) in G.edges()
            assert (2, 1) in G.edges()
            assert (0, 1) not in G.edges()
            return
    pytest.fail("random.choice never chose to negate the single edge in 200 attempts")


# ── test generate_circuits ────────────────────────────────────────────────────

# generate_circuits' own responsibility is: pull raw DAGs from
# generate_nauty_digraphs, run them through nauty_to_networkx/_add_single_sink
# and the source/fan-in/fan-out filters, assign gates per `gate_assignment`,
# and hand each resulting netlist to `db.add` (skipped when dry_run). These
# tests stub out generate_nauty_digraphs with canned nauty-format lines (real
# text, real nauty_to_networkx/_add_single_sink/filters/gate-assignment code
# runs on them) so no nauty binary or real database is needed.
#
# Two nauty lines are used throughout:
#   "1 0"  -- one isolated node. After _add_single_sink: source 0 -> sink 1
#             (1 source, sink fan-in 1, source fan-out 1).
#   "2 0"  -- two isolated nodes. After _add_single_sink: sources 0, 1 both
#             feed a single combined gate/sink node 2 (2 sources, node 2
#             fan-in 2, sources' fan-out 1 each).


class _RecordingDB:
    """Stand-in for circuit_database.CircuitDatabase: just records every
    netlist passed to add(), in order."""
    def __init__(self):
        self.added = []

    def add(self, netlist):
        self.added.append(netlist)


def _run_generate_circuits(dag_lines, db, **overrides):
    """Run generate_circuits with generate_nauty_digraphs stubbed to yield
    `dag_lines`, and every filter defaulted wide open ([0, 10]) so a test
    only has to narrow the bounds it actually cares about."""
    kwargs = dict(
        dag_size=1,
        min_num_source=0, max_num_source=10,
        min_num_inputs=0, max_num_inputs=10,
        min_num_outputs=0, max_num_outputs=10,
        gate_assignment='and-only',
        display_circuits=False,
        output_dir=None,
        quiet=True,
    )
    kwargs.update(overrides)
    with patch('rysy_dl.circuit_generator.generate_nauty_digraphs',
               lambda dag_size, nauty_dir=None: iter(dag_lines)):
        return generate_circuits(db=db, **kwargs)


def test_generate_circuits_invalid_gate_assignment_raises():
    with pytest.raises(ValueError, match="Invalid gate_assignment"):
        _run_generate_circuits(["1 0"], db=None, gate_assignment='bogus')


def test_generate_circuits_and_only_assigns_and_to_every_gate():
    db = _RecordingDB()

    result = _run_generate_circuits(["1 0"], db, gate_assignment='and-only')

    assert result == 1
    assert len(db.added) == 1
    netlist = db.added[0]
    assert netlist.nodes[0]['gate'] == 'src'
    assert netlist.nodes[1]['gate'] == 'AND'


def test_generate_circuits_random_assigns_a_valid_gate():
    db = _RecordingDB()

    result = _run_generate_circuits(["1 0"], db, gate_assignment='random')

    assert result == 1
    netlist = db.added[0]
    assert netlist.nodes[0]['gate'] == 'src'
    # Fan-in is 1, so a misplaced XOR/XNOR gets swapped down to AND/OR (see
    # assign_gates_randomly_ensure_diversity); either way this is never 'src'.
    assert netlist.nodes[1]['gate'] in {'AND', 'OR', 'XOR'}


def test_generate_circuits_all_enumerates_every_gate_assignment():
    db = _RecordingDB()

    result = _run_generate_circuits(["2 0"], db, gate_assignment='all')

    # Node 2's two children (0 and 1) are both 'src', so no assignment can
    # collapse into its parent -- all 5 default gate choices survive the
    # no_collapsable_children filter used internally by generate_circuits.
    assert result == 5
    assert len(db.added) == 5
    gates_seen = {netlist.nodes[2]['gate'] for netlist in db.added}
    assert gates_seen == {'AND', 'OR', 'XOR', 'NAND', 'NOR'}


def test_generate_circuits_filters_out_graphs_with_wrong_source_count():
    db = _RecordingDB()

    result = _run_generate_circuits(
        ["1 0", "2 0"], db,
        min_num_source=1, max_num_source=1,
        gate_assignment='and-only',
    )

    # Only "1 0" (1 source) satisfies [1, 1]; "2 0" (2 sources) is dropped.
    assert result == 1
    assert len(db.added) == 1
    netlist = db.added[0]
    assert sum(1 for node in netlist.nodes if netlist.in_degree(node) == 0) == 1


def test_generate_circuits_filters_out_graphs_with_fan_in_below_min():
    db = _RecordingDB()

    # "1 0" -> sink node 1 has fan-in 1, which is below the required minimum of 2.
    result = _run_generate_circuits(
        ["1 0"], db,
        min_num_inputs=2, max_num_inputs=2,
        gate_assignment='and-only',
    )

    assert result == 0
    assert db.added == []


def test_generate_circuits_filters_out_graphs_with_fan_out_above_max():
    db = _RecordingDB()

    # "1 0" -> source node 0 has fan-out 1, which is above the allowed maximum of 0.
    result = _run_generate_circuits(
        ["1 0"], db,
        min_num_outputs=0, max_num_outputs=0,
        gate_assignment='and-only',
    )

    assert result == 0
    assert db.added == []


def test_generate_circuits_dry_run_skips_db_and_db_may_be_none():
    result = _run_generate_circuits(
        ["1 0"], db=None, dry_run=True, gate_assignment='and-only',
    )

    # n is still counted even though nothing is written anywhere.
    assert result == 1


def test_generate_circuits_sums_results_across_multiple_dags():
    db = _RecordingDB()

    result = _run_generate_circuits(["1 0", "1 0", "1 0"], db, gate_assignment='and-only')

    assert result == 3
    assert len(db.added) == 3


def test_generate_circuits_quiet_suppresses_parameter_dump(capsys):
    _run_generate_circuits(["1 0"], _RecordingDB(), quiet=True)

    assert "dag_size=" not in capsys.readouterr().out


def test_generate_circuits_not_quiet_prints_parameter_dump(capsys):
    _run_generate_circuits(["1 0"], _RecordingDB(), quiet=False, dag_size=1)

    assert "dag_size=1" in capsys.readouterr().out


def test_generate_circuits_always_prints_final_count_even_when_quiet(capsys):
    _run_generate_circuits(["1 0"], _RecordingDB(), quiet=True)

    assert "Generated 1 circuits" in capsys.readouterr().out


def test_generate_circuits_display_stops_early_when_show_netlist_quits(monkeypatch):
    mock_display = MagicMock()
    mock_display.show_netlist.return_value = True  # simulates the user pressing 'q'
    monkeypatch.setattr(rysy_dl, 'circuit_generator_display', mock_display, raising=False)
    db = _RecordingDB()

    result = _run_generate_circuits(
        ["1 0", "1 0", "1 0"], db, display_circuits=True, gate_assignment='and-only',
    )

    # The first netlist is added to the db, then show_netlist reports 'quit',
    # so the remaining two dag lines are never processed.
    assert result == 1
    assert len(db.added) == 1
    assert mock_display.show_netlist.call_count == 1


def test_generate_circuits_display_continues_when_show_netlist_does_not_quit(monkeypatch):
    mock_display = MagicMock()
    mock_display.show_netlist.return_value = False
    monkeypatch.setattr(rysy_dl, 'circuit_generator_display', mock_display, raising=False)
    db = _RecordingDB()

    result = _run_generate_circuits(
        ["1 0", "1 0", "1 0"], db, display_circuits=True, gate_assignment='and-only',
    )

    assert result == 3
    assert len(db.added) == 3
    assert mock_display.show_netlist.call_count == 3


def test_generate_circuits_prints_status_and_collects_garbage_every_100000(capsys):
    db = _RecordingDB()
    lines = ["1 0"] * 100000

    with patch('rysy_dl.circuit_generator.gc.collect') as mock_gc_collect:
        result = _run_generate_circuits(lines, db, gate_assignment='and-only', quiet=True)

    assert result == 100000
    assert mock_gc_collect.call_count == 1
    assert "Processed 100000 graphs" in capsys.readouterr().out
