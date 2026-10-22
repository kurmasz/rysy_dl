"""
graph_filters.py

Boolean functions used to analyze a networkx DiGraph. These functions only consider the graph structure,
they do not consider any gate assignments.
"""

from . import graph_analyzers

def num_sources_in_range(graph, min_num: int, max_num: int):
    """
    Verifies that the number of source nodes (nodes with in_degree 0) in the given networkx DiGraph is within the specified range [min_num, max_num].
    """
    source_num = graph_analyzers.num_sources(graph)
    return source_num <= max_num and source_num >= min_num


def non_source_in_degrees_in_range(graph, min_num: int, max_num:int):
    """
    Verifies that all non-source nodes (nodes with in_degree > 0) in the given networkx DiGraph 
    have in_degrees within the specified range [min_num, max_num].
    
    This corresponds to checking the fan-in for all gates in a circuit
    """
    for node in graph.nodes():
        if graph.in_degree(node) != 0:
            if graph.in_degree(node) > max_num or graph.in_degree(node) < min_num:
                return False
    return True

def non_sink_out_degrees_in_range(graph, min_num: int, max_num:int):
    """
    Verifies that the out-degree of all non-sinknodes falls in within the desired range.
    """
    for node in graph.nodes():
        # ensuring final gate won't trigger an issue
        if graph.out_degree(node) != 0:
            if graph.out_degree(node) > max_num or graph.out_degree(node) < min_num:
                return False
    return True

def has_shared_internal_node(graph):
    """
    Verifies that a circuit generated from this graph will have shared intermediate terms.
    
    In other words, this graph is not a tree, with at least one shared node that is not a source.

    :param graph: A networkx DiGraph representing a circuit.
    :returns: True if at least one non-source node (a node with in_degree > 0) has an out_degree of at least 2.
    """
    return any(graph.out_degree(node) >= 2 for node in graph.nodes() if graph.in_degree(node) > 0)


def only_leaves_shared(graph):
    """
    Verifies that a circuit generated from this graph will not have any shared intermediate terms.
    
    In other words, this graph is almost a tree, with the only shared nodes being the sources.

    :param graph: A networkx DiGraph representing a circuit.
    :returns: True if every non-source node (a node with in_degree > 0) has an out_degree of at most 1.
    """
    return not has_shared_internal_node(graph)
