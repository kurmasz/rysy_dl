"""
graph_analyzers.py

Functions to analyze a networkx DiGraph. These functions only consider the graph structure,
they do not consider any gate assignments.
"""

def num_sources_and_gates(graph):
    """
    Returns a tuple of (number of source nodes, number of gate nodes, number of NOT gates) in the given networkx DiGraph.
    Technically, the input is a graph, not a circuit, so there haven't been gates assigned.
    However, we define sources and gates as follows:
   
    A source node is defined as a node with in_degree 0.
    A gate node is defined as a node with in_degree > 0.
    A NOT gate is defined as a node with in_degree 1 and out_degree 1
    """
   
    source_num = 0
    gate_num = 0
    not_gates = 0
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            source_num += 1
        else:
            gate_num += 1

        if graph.out_degree(node) == 1 and graph.in_degree(node) == 1:
            not_gates += 1
    return source_num, gate_num, not_gates

def num_sources(graph):
    """
    Returns the number of source nodes (nodes with in_degree 0) in the given networkx DiGraph.
    """
    return num_sources_and_gates(graph)[0]




