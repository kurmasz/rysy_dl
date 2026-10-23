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

def nauty_in_degrees_in_range(nauty_graph: str, min_num: int, max_num: int):
    """
    The same check as `non_source_in_degrees_in_range`, but applied to one line of nauty's
    ``directg -aTq`` output (``num_vertices num_edges u1 v1 u2 v2 ...``) before it is converted
    to a networkx DiGraph.

    Building the networkx DiGraph is the most expensive step in generating circuits, and most
    DAGs nauty produces are rejected because some gate's fan-in is out of range. Checking fan-in
    on the raw edge list rejects those DAGs without building a graph for them.

    The check accounts for the sink that `circuit_generator._add_single_sink` will add: that
    sink's fan-in is the number of nodes with out_degree 0 in the nauty DAG. So for any line,
    this returns the same result as
    ``non_source_in_degrees_in_range(_add_single_sink(nauty_to_networkx(line)), min_num, max_num)``.
    """
    parts = list(map(int, nauty_graph.split()))
    num_nodes = parts[0]
    in_degrees = [0] * num_nodes
    has_out_edge = [False] * num_nodes
    # parts[2:] holds the edges as u v pairs: u's are at even offsets, v's at odd offsets.
    for u in parts[2::2]:
        has_out_edge[u] = True
    for v in parts[3::2]:
        in_degrees[v] += 1

    for in_degree in in_degrees:
        if in_degree != 0 and (in_degree > max_num or in_degree < min_num):
            return False

    sink_in_degree = has_out_edge.count(False)
    return min_num <= sink_in_degree <= max_num

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
