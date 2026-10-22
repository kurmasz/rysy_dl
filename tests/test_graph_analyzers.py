"""
Unit tests for graph_analyzers.py."""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.graph_analyzers import num_sources, num_sources_and_gates


def test_num_sources_empty_graph():
    graph = nx.DiGraph()

    assert num_sources(graph) == 0


def test_num_sources_no_sources():
    # A cycle: every node has in_degree 1, so there are no source nodes.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    assert num_sources(graph) == 0


def test_num_sources_single_source():
    # 0 -> 1 -> 2 has exactly one source node (0).
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    assert num_sources(graph) == 1


def test_num_sources_multiple_sources():
    # Two disjoint chains: 0 -> 1 and 2 -> 3 have two source nodes (0 and 2).
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)

    assert num_sources(graph) == 2


def test_num_sources_all_isolated_nodes():
    # Nodes with no edges all count as sources.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2, 3])

    assert num_sources(graph) == 4


def test_num_sources_ignores_out_degree():
    # A node with high out-degree but no incoming edges is still a single source.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)

    assert num_sources(graph) == 1

def test_num_sources_medium_complexity():
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(2, 4)
    graph.add_edge(0, 3)
    graph.add_edge(1, 3)
    graph.add_edge(3, 4)

    assert num_sources(graph) == 2


# ── test num_sources_and_gates ────────────────────────────────────────────────────────────────────
#
# A source is a node with in_degree 0; a gate is any other node (in_degree > 0);
# a NOT gate is a gate with in_degree == 1 and out_degree == 1. num_sources delegates
# its counting to this function, so these tests focus on the gate/not-gate counts
# that num_sources itself doesn't exercise.


def test_num_sources_and_gates_empty_graph():
    graph = nx.DiGraph()

    assert num_sources_and_gates(graph) == (0, 0, 0)


def test_num_sources_and_gates_all_isolated_nodes():
    # Every node is a source (in_degree 0); none are gates.
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert num_sources_and_gates(graph) == (3, 0, 0)


def test_num_sources_and_gates_simple_chain_counts_not_gate():
    # 0 -> 1 -> 2: node 0 is a source; nodes 1 and 2 are gates.
    # Node 1 (in_degree 1, out_degree 1) qualifies as a NOT gate; node 2
    # (in_degree 1, out_degree 0) does not, since its out_degree isn't 1.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    assert num_sources_and_gates(graph) == (1, 2, 1)


def test_num_sources_and_gates_high_in_degree_gate_is_not_a_not_gate():
    # Node 2 has in_degree 2 (AND-like), so it's a gate but not a NOT gate
    # even though its out_degree is 1. Node 3 (in_degree 1, out_degree 0) isn't
    # a NOT gate either, since its out_degree isn't 1.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    assert num_sources_and_gates(graph) == (2, 2, 0)


def test_num_sources_and_gates_fan_out_gate_is_not_a_not_gate():
    # Node 1 has in_degree 1 but out_degree 2 (fans out to 2 and 3), so it's a
    # gate but does not qualify as a NOT gate.
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)

    assert num_sources_and_gates(graph) == (1, 3, 0)


def test_num_sources_and_gates_sink_gate_is_not_a_not_gate():
    # Node 2 has in_degree 1 but out_degree 0 (it's the sink), so it's a gate
    # but not a NOT gate.
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)

    assert num_sources_and_gates(graph) == (2, 1, 0)


def test_num_sources_and_gates_multiple_not_gates():
    # Two disjoint NOT-shaped chains: 0 -> 1 -> 2 and 3 -> 4 -> 5.
    # Sources: 0, 3. Gates: 1, 2, 4, 5. NOT gates: 1 and 4 (in_degree 1, out_degree 1).
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)

    assert num_sources_and_gates(graph) == (2, 4, 2)


def test_num_sources_and_gates_source_is_never_counted_as_not_gate():
    # An isolated node has in_degree 0 and out_degree 0. It's a source, and
    # must not be miscounted as a NOT gate even though out_degree == 0 != 1
    # already excludes it -- this pins that in_degree 0 nodes are excluded by
    # the source/gate split before the NOT-gate check ever applies.
    graph = nx.DiGraph()
    graph.add_node(0)

    assert num_sources_and_gates(graph) == (1, 0, 0)


def test_num_sources_and_gates_returns_a_three_tuple_of_ints():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    result = num_sources_and_gates(graph)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(n, int) for n in result)


def test_num_sources_delegates_to_num_sources_and_gates():
    # num_sources(graph) must equal the first element of num_sources_and_gates(graph)
    # for the same graph, since it's defined as num_sources_and_gates(graph)[0].
    graph = nx.DiGraph()
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    assert num_sources(graph) == num_sources_and_gates(graph)[0]