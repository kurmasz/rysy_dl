"""
Unit tests for graph_filters.py."""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import graph_filters
from rysy_dl.circuit_generator import _add_single_sink, nauty_to_networkx
from rysy_dl.graph_filters import (
    has_shared_internal_node,
    nauty_in_degrees_in_range,
    non_sink_out_degrees_in_range,
    non_source_in_degrees_in_range,
    num_sources_in_range,
    only_leaves_shared,
)

# graph_filters.num_sources_in_range now delegates the actual counting to
# graph_analyzers.num_sources (tested separately in test_graph_analyzers.py).
# These tests stub that call out so they exercise only the range-checking
# logic, independent of graph structure. The graph argument itself is never
# inspected once stubbed, so a sentinel stands in for a real DiGraph.
_SENTINEL_GRAPH = object()

# ── test num_sources_in_range ────────────────────────────────────────────────────────────────────

def _stub_num_sources(monkeypatch, count):
    monkeypatch.setattr(graph_filters.graph_analyzers, 'num_sources', lambda graph: count)


def test_num_sources_in_range_passes_graph_through_to_analyzer(monkeypatch):
    received = []
    monkeypatch.setattr(graph_filters.graph_analyzers, 'num_sources', lambda graph: received.append(graph) or 1)

    num_sources_in_range(_SENTINEL_GRAPH, 0, 5)

    assert received == [_SENTINEL_GRAPH]


def test_num_sources_in_range_within_range(monkeypatch):
    _stub_num_sources(monkeypatch, 3)

    assert num_sources_in_range(_SENTINEL_GRAPH, 0, 5) is True


def test_num_sources_in_range_below_min(monkeypatch):
    _stub_num_sources(monkeypatch, 1)

    assert num_sources_in_range(_SENTINEL_GRAPH, 2, 5) is False


def test_num_sources_in_range_above_max(monkeypatch):
    _stub_num_sources(monkeypatch, 6)

    assert num_sources_in_range(_SENTINEL_GRAPH, 2, 5) is False


def test_num_sources_in_range_at_min_boundary(monkeypatch):
    _stub_num_sources(monkeypatch, 2)

    assert num_sources_in_range(_SENTINEL_GRAPH, 2, 5) is True


def test_num_sources_in_range_at_max_boundary(monkeypatch):
    _stub_num_sources(monkeypatch, 5)

    assert num_sources_in_range(_SENTINEL_GRAPH, 2, 5) is True


def test_num_sources_in_range_min_equals_max_and_matches(monkeypatch):
    _stub_num_sources(monkeypatch, 3)

    assert num_sources_in_range(_SENTINEL_GRAPH, 3, 3) is True


def test_num_sources_in_range_min_equals_max_and_does_not_match(monkeypatch):
    _stub_num_sources(monkeypatch, 4)

    assert num_sources_in_range(_SENTINEL_GRAPH, 3, 3) is False


def test_num_sources_in_range_zero_sources(monkeypatch):
    _stub_num_sources(monkeypatch, 0)

    assert num_sources_in_range(_SENTINEL_GRAPH, 0, 0) is True
    assert num_sources_in_range(_SENTINEL_GRAPH, 1, 5) is False

# ── test non_source_in_degrees_in_range ────────────────────────────────────────────────────────────────────

# non_source_in_degrees_in_range examines graph.in_degree() directly rather
# than delegating to graph_analyzers, so these tests build real graphs
# (unlike the num_sources_in_range tests above, which stub the delegate).


def test_non_source_in_degrees_in_range_empty_graph():
    graph = nx.DiGraph()

    assert non_source_in_degrees_in_range(graph, 1, 1) is True


def test_non_source_in_degrees_in_range_all_sources_is_vacuously_true():
    # No edges, so every node is a source (in_degree 0) and there are no
    # non-source nodes to check. The range should be irrelevant.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert non_source_in_degrees_in_range(graph, 5, 5) is True


def test_non_source_in_degrees_in_range_source_node_is_ignored():
    # 0 is a source (in_degree 0) and should be skipped even though 0 is
    # outside [min_num, max_num]; only node 1 (in_degree 1) is checked.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_source_in_degrees_in_range(graph, 1, 1) is True


def test_non_source_in_degrees_in_range_single_non_source_within_range():
    # 0 -> 1 -> 2: nodes 1 and 2 are non-source, each with in_degree 1.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    assert non_source_in_degrees_in_range(graph, 1, 1) is True
    assert non_source_in_degrees_in_range(graph, 0, 3) is True


def test_non_source_in_degrees_in_range_above_max():
    # Node 2 has in_degree 2 (edges from 0 and 1).
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)

    assert non_source_in_degrees_in_range(graph, 1, 1) is False


def test_non_source_in_degrees_in_range_below_min():
    # Node 1 has in_degree 1, which is below the required minimum of 2.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_source_in_degrees_in_range(graph, 2, 5) is False


def test_non_source_in_degrees_in_range_mixed_pass_and_fail():
    # Node 2 has in_degree 1 (within range); node 5 has in_degree 3 (out of
    # range). Any single out-of-range non-source node should cause failure.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 5)
    graph.add_edge(2, 5)
    graph.add_edge(3, 5)

    assert non_source_in_degrees_in_range(graph, 1, 2) is False


def test_non_source_in_degrees_in_range_at_min_boundary():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_source_in_degrees_in_range(graph, 1, 3) is True


def test_non_source_in_degrees_in_range_at_max_boundary():
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)

    assert non_source_in_degrees_in_range(graph, 1, 2) is True


def test_non_source_in_degrees_in_range_multiple_non_sources_all_within_range():
    # Diamond graph: 0 -> 2, 1 -> 2, 2 -> 4, 3 -> 4.
    # Non-source nodes 2 and 4 both have in_degree 2.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    graph.add_edge(3, 4)

    assert non_source_in_degrees_in_range(graph, 2, 2) is True

# ── test non_sink_out_degrees_in_range ────────────────────────────────────────────────────────────────────

# non_sink_out_degrees_in_range mirrors non_source_in_degrees_in_range but
# checks graph.out_degree() and skips sink nodes (out_degree 0) instead of
# source nodes, so it gets the same style of graph-building tests.


def test_non_sink_out_degrees_in_range_empty_graph():
    graph = nx.DiGraph()

    assert non_sink_out_degrees_in_range(graph, 1, 1) is True


def test_non_sink_out_degrees_in_range_all_sinks_is_vacuously_true():
    # No edges, so every node is a sink (out_degree 0) and there are no
    # non-sink nodes to check. The range should be irrelevant.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert non_sink_out_degrees_in_range(graph, 5, 5) is True


def test_non_sink_out_degrees_in_range_sink_node_is_ignored():
    # 1 is a sink (out_degree 0) and should be skipped even though 0 is
    # outside [min_num, max_num]; only node 0 (out_degree 1) is checked.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_sink_out_degrees_in_range(graph, 1, 1) is True


def test_non_sink_out_degrees_in_range_single_non_sink_within_range():
    # 0 -> 1 -> 2: nodes 0 and 1 are non-sink, each with out_degree 1.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    assert non_sink_out_degrees_in_range(graph, 1, 1) is True
    assert non_sink_out_degrees_in_range(graph, 0, 3) is True


def test_non_sink_out_degrees_in_range_above_max():
    # Node 0 has out_degree 2 (edges to 1 and 2).
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    assert non_sink_out_degrees_in_range(graph, 1, 1) is False


def test_non_sink_out_degrees_in_range_below_min():
    # Node 0 has out_degree 1, which is below the required minimum of 2.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_sink_out_degrees_in_range(graph, 2, 5) is False


def test_non_sink_out_degrees_in_range_mixed_pass_and_fail():
    # Node 0 has out_degree 1 (within range); node 5 has out_degree 3 (out of
    # range). Any single out-of-range non-sink node should cause failure.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(5, 1)
    graph.add_edge(5, 2)
    graph.add_edge(5, 3)

    assert non_sink_out_degrees_in_range(graph, 1, 2) is False


def test_non_sink_out_degrees_in_range_at_min_boundary():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    assert non_sink_out_degrees_in_range(graph, 1, 3) is True


def test_non_sink_out_degrees_in_range_at_max_boundary():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    assert non_sink_out_degrees_in_range(graph, 1, 2) is True


def test_non_sink_out_degrees_in_range_multiple_non_sinks_all_within_range():
    # Reverse diamond: 0 -> 2, 0 -> 3, 1 -> 2, 1 -> 3.
    # Non-sink nodes 0 and 1 both have out_degree 2.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)

    assert non_sink_out_degrees_in_range(graph, 2, 2) is True

# ── test has_shared_internal_node ────────────────────────────────────────────────────────────────────

# has_shared_internal_node examines both in_degree (to find non-source nodes)
# and out_degree (to find sharing) directly on the graph, so these tests build
# real graphs rather than stubbing a delegate.


def test_has_shared_internal_node_empty_graph():
    graph = nx.DiGraph()

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_all_sources_is_vacuously_false():
    # No edges, so every node is a source (in_degree 0) and there are no
    # non-source nodes to check for sharing.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_source_fan_out_is_ignored():
    # 0 is a source with out_degree 2, but sources are excluded from the
    # check. Nodes 1 and 2 are non-source with out_degree 0.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_linear_chain_no_sharing():
    # 0 -> 1 -> 2 -> 3: every non-source node has out_degree at most 1.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_tree_no_sharing():
    # A tree rooted at 0: fan-out only happens at the source (0 -> 1, 0 -> 2),
    # and every non-source node (1, 2) has out_degree at most 1, so no
    # non-source node's result is reused by more than one consumer.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_non_source_out_degree_two():
    # 0 -> 1, 1 -> 2, 1 -> 3: node 1 is non-source (in_degree 1) with
    # out_degree 2, so its result is shared by two downstream nodes.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)

    assert has_shared_internal_node(graph) is True


def test_has_shared_internal_node_reconverge_then_fan_out():
    # Diamond that reconverges at 3, then 3 fans out to 4 and 5. Node 3 has
    # in_degree 2 (non-source) and out_degree 2 (shared).
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)

    assert has_shared_internal_node(graph) is True


def test_has_shared_internal_node_at_boundary_out_degree_one_is_false():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    assert has_shared_internal_node(graph) is False


def test_has_shared_internal_node_only_one_of_several_non_source_nodes_shares():
    # Two disjoint chains (0 -> 1 -> 2 and 3 -> 4 -> 5) contribute non-source
    # nodes 1, 2, 4, 5, all with out_degree at most 1. Node 6 is the only
    # non-source node that shares, fanning out to 7 and 8.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(0, 6)
    graph.add_edge(6, 7)
    graph.add_edge(6, 8)

    assert has_shared_internal_node(graph) is True

def test_has_shared_internal_node_only_reused_sources_one_sink():
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 5)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)

    assert has_shared_internal_node(graph) is False


# ── test only_leaves_shared ────────────────────────────────────────────────────────────────────

# only_leaves_shared is defined as `not has_shared_internal_node(graph)`.
# These tests first confirm that delegation with a stub, then exercise a
# couple of real graphs to confirm the negation reads correctly end to end.


def test_only_leaves_shared_delegates_and_negates_true(monkeypatch):
    monkeypatch.setattr(graph_filters, 'has_shared_internal_node', lambda graph: True)

    assert only_leaves_shared(_SENTINEL_GRAPH) is False


def test_only_leaves_shared_delegates_and_negates_false(monkeypatch):
    monkeypatch.setattr(graph_filters, 'has_shared_internal_node', lambda graph: False)

    assert only_leaves_shared(_SENTINEL_GRAPH) is True


def test_only_leaves_shared_passes_graph_through_to_delegate(monkeypatch):
    received = []
    monkeypatch.setattr(graph_filters, 'has_shared_internal_node', lambda graph: received.append(graph) or False)

    only_leaves_shared(_SENTINEL_GRAPH)

    assert received == [_SENTINEL_GRAPH]


def test_only_leaves_shared_empty_graph_is_true():
    graph = nx.DiGraph()

    assert only_leaves_shared(graph) is True


def test_only_leaves_shared_source_fan_out_only_is_true():
    # Only the source (0) shares its output; no non-source node does, so
    # sharing is confined to the leaves.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    assert only_leaves_shared(graph) is True


def test_only_leaves_shared_non_source_sharing_is_false():
    # Node 1 is non-source and shares its output with nodes 2 and 3.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)

    assert only_leaves_shared(graph) is False


# ── test nauty_in_degrees_in_range ────────────────────────────────────────────────────────────────────

# nauty_in_degrees_in_range takes one line of nauty's directg -aTq output
# ("num_vertices num_edges u1 v1 u2 v2 ...") and must account for the sink that
# _add_single_sink adds afterward (fed by every node with out_degree 0).


def test_nauty_in_degrees_in_range_all_gates_within_range():
    # 0 -> 2, 1 -> 2: node 2 has fan-in 2 and is the only original sink, so
    # the added sink has fan-in 1.
    assert nauty_in_degrees_in_range("3 2 0 2 1 2", 1, 2) is True


def test_nauty_in_degrees_in_range_gate_below_min():
    # 0 -> 1 gives node 1 fan-in 1. Nodes 1 and 2 are sinks, so the added sink has fan-in 2.
    assert nauty_in_degrees_in_range("3 1 0 1", 2, 6) is False


def test_nauty_in_degrees_in_range_gate_above_max():
    # Node 3 has fan-in 3.
    assert nauty_in_degrees_in_range("4 3 0 3 1 3 2 3", 1, 2) is False


def test_nauty_in_degrees_in_range_sources_are_ignored():
    # Nodes 0 and 1 have fan-in 0, which is below min but they are sources.
    # Node 2 and the added sink are the only gates.
    assert nauty_in_degrees_in_range("4 2 0 2 1 2", 2, 2) is True


def test_nauty_in_degrees_in_range_single_original_sink_rejected():
    # Node 2 is the only original sink, so the added sink would have fan-in 1.
    assert nauty_in_degrees_in_range("3 2 0 2 1 2", 2, 6) is False


def test_nauty_in_degrees_in_range_isolated_nodes_feed_added_sink():
    # No edges: all 3 nodes are sources with out_degree 0, so the added sink has fan-in 3.
    assert nauty_in_degrees_in_range("3 0", 3, 3) is True
    assert nauty_in_degrees_in_range("3 0", 2, 2) is False


def test_nauty_in_degrees_in_range_matches_networkx_filter_exhaustively():
    # Every DAG on 5 labeled nodes whose edges all go from lower to higher label
    # (every subset of the 10 possible i -> j edges with i < j), checked against
    # the networkx filter applied after _add_single_sink.
    num_nodes = 5
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    ranges = [(1, 1), (1, 6), (2, 2), (2, 3), (2, 6), (3, 5)]
    for mask in range(1 << len(possible_edges)):
        edges = [edge for bit, edge in enumerate(possible_edges) if mask >> bit & 1]
        line = " ".join(map(str, [num_nodes, len(edges)] + [n for edge in edges for n in edge]))
        graph = _add_single_sink(nauty_to_networkx(line))
        for min_num, max_num in ranges:
            expected = non_source_in_degrees_in_range(graph, min_num, max_num)
            assert nauty_in_degrees_in_range(line, min_num, max_num) is expected, (line, min_num, max_num)
