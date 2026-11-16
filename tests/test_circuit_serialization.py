"""
Unit tests for circuit_serialization.py's compact byte encoding."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.circuit_serialization import bytes_to_circuit, circuit_to_bytes


def _circuit():
    """(A' * B) + (A * C), with A's NOT on its first edge and edges added out of node order."""
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2], gate='src')
    graph.add_node(3, gate='AND')
    graph.add_node(4, gate='AND')
    graph.add_node(5, gate='OR')
    graph.add_node(6, gate='NOT')
    graph.add_edges_from([(0, 6), (6, 3), (1, 3), (0, 4), (2, 4), (4, 5), (3, 5)])
    return graph


def _same(a, b):
    """Same nodes, attributes and edges, in the same order, including each node's successors."""
    return (list(a.nodes(data=True)) == list(b.nodes(data=True))
            and list(a.edges()) == list(b.edges())
            and all(list(a.successors(n)) == list(b.successors(n)) for n in a))


def test_round_trip_is_identical():
    graph = _circuit()
    assert _same(bytes_to_circuit(circuit_to_bytes(graph)), graph)


def test_round_trip_keeps_successor_order():
    # Node 0's successors are 6 then 4; the order decides which edge is a source's "first" edge.
    graph = bytes_to_circuit(circuit_to_bytes(_circuit()))
    assert list(graph.successors(0)) == [6, 4]


def test_every_gate_type_round_trips():
    graph = nx.DiGraph()
    for node, gate in enumerate(['src', 'AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'NOT']):
        graph.add_node(node, gate=gate)
    assert _same(bytes_to_circuit(circuit_to_bytes(graph)), graph)


def test_labels_round_trip():
    graph = _circuit()
    graph.nodes[0]['label'] = 'a'
    graph.nodes[2]['label'] = 'x_1'
    assert _same(bytes_to_circuit(circuit_to_bytes(graph)), graph)


def test_size():
    # 1 node count + 7 gates + 1 edge count + 7 edges * 2 + 1 label count
    assert len(circuit_to_bytes(_circuit())) == 24


def test_rejects_nodes_not_numbered_in_order():
    graph = nx.DiGraph()
    graph.add_node(1, gate='src')
    graph.add_node(0, gate='src')
    with pytest.raises(ValueError, match="numbered 0 to n-1"):
        circuit_to_bytes(graph)


def test_rejects_gaps_in_node_numbers():
    graph = nx.DiGraph()
    graph.add_node(0, gate='src')
    graph.add_node(2, gate='AND')
    with pytest.raises(ValueError, match="numbered 0 to n-1"):
        circuit_to_bytes(graph)


def test_rejects_unknown_gate():
    graph = nx.DiGraph()
    graph.add_node(0, gate='MUX')
    with pytest.raises(ValueError, match="gate"):
        circuit_to_bytes(graph)


def test_rejects_missing_gate():
    graph = nx.DiGraph()
    graph.add_node(0)
    with pytest.raises(ValueError, match="gate"):
        circuit_to_bytes(graph)


def test_rejects_more_than_255_nodes():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(256), gate='src')
    with pytest.raises(ValueError, match="255"):
        circuit_to_bytes(graph)
