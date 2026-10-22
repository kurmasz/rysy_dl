import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.circuit_analyzer import evaluate_circuit, generate_truth_table


def test_evaluate_circuit_handles_and_not_gates():
    circuit = nx.DiGraph()
    circuit.add_nodes_from([0, 1, 2, 3])
    circuit.add_edge(0, 2)
    circuit.add_edge(1, 2)
    circuit.add_edge(2, 3)

    circuit.nodes[0]['gate'] = 'src'
    circuit.nodes[1]['gate'] = 'src'
    circuit.nodes[2]['gate'] = 'AND'
    circuit.nodes[3]['gate'] = 'NOT'

    assert evaluate_circuit(circuit, 0) is True
    assert evaluate_circuit(circuit, 1) is True
    assert evaluate_circuit(circuit, 3) is False


def _make_y_and_not_x_circuit():
    """y=node 0, x=node 1 -- AND(y, NOT x). Node-id order (0, 1) is (y, x),
    the reverse of alphabetical, so it's a good check that ordering follows
    node id / an explicit variable_order rather than label alphabetization."""
    circuit = nx.DiGraph()
    circuit.add_nodes_from([0, 1, 2, 3])
    circuit.add_edge(1, 2)  # NOT x
    circuit.add_edge(0, 3)  # y  --\
    circuit.add_edge(2, 3)  # NOT x --/ AND

    circuit.nodes[0]['gate'] = 'src'
    circuit.nodes[0]['label'] = 'y'
    circuit.nodes[1]['gate'] = 'src'
    circuit.nodes[1]['label'] = 'x'
    circuit.nodes[2]['gate'] = 'NOT'
    circuit.nodes[3]['gate'] = 'AND'
    return circuit


def test_evaluate_circuit_defaults_to_node_id_order():
    circuit = _make_y_and_not_x_circuit()
    # Default order is by node id: node 0 (y) is MSB, node 1 (x) is LSB.
    assert evaluate_circuit(circuit, 0b10) is True   # y=1, x=0
    assert evaluate_circuit(circuit, 0b01) is False  # y=0, x=1


def test_evaluate_circuit_accepts_explicit_variable_order():
    circuit = _make_y_and_not_x_circuit()
    # Explicit order reverses which bit belongs to which source.
    assert evaluate_circuit(circuit, 0b10, variable_order=['x', 'y']) is False  # x=1, y=0
    assert evaluate_circuit(circuit, 0b01, variable_order=['x', 'y']) is True   # x=0, y=1


def test_generate_truth_table_default_and_explicit_order():
    circuit = _make_y_and_not_x_circuit()
    assert generate_truth_table(circuit) == {'00': 0, '01': 0, '10': 1, '11': 0}
    assert generate_truth_table(circuit, variable_order=['x', 'y']) == {
        '00': 0, '01': 1, '10': 0, '11': 0,
    }


def test_evaluate_circuit_rejects_unknown_variable_name():
    circuit = _make_y_and_not_x_circuit()
    with pytest.raises(ValueError):
        evaluate_circuit(circuit, 0, variable_order=['x', 'z'])


def test_evaluate_circuit_rejects_missing_variable_name():
    circuit = _make_y_and_not_x_circuit()
    with pytest.raises(ValueError):
        evaluate_circuit(circuit, 0, variable_order=['x'])
