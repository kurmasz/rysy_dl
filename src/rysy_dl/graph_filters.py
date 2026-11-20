"""
graph_filters.py

Boolean functions used to analyze a networkx DiGraph. These functions only consider the graph structure,
they do not consider any gate assignments.
"""

import networkx as nx

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

def nauty_in_degrees_in_range(nauty_graph: str, min_num: int, max_num: int, output_connections: str = "sinks"):
    """
    The same check as `non_source_in_degrees_in_range`, but applied to one line of nauty's
    ``directg -aTq`` output (``num_vertices num_edges u1 v1 u2 v2 ...``) before it is converted
    to a networkx DiGraph.

    Building the networkx DiGraph is the most expensive step in generating circuits, and most
    DAGs nauty produces are rejected because some gate's fan-in is out of range. Checking fan-in
    on the raw edge list rejects those DAGs without building a graph for them.

    The check accounts for the output gate that `circuit_generator` will add:

    * ``output_connections='sinks'``: the output reads exactly the nodes with out_degree 0 (see
      `circuit_generator._add_single_sink`), so its fan-in is the number of such nodes. For any
      line, this returns the same result as
      ``non_source_in_degrees_in_range(_add_single_sink(nauty_to_networkx(line)), min_num, max_num)``.
    * ``output_connections='any'``: the output reads every out_degree-0 node plus any subset of
      the others (see `circuit_generator._single_sink_extensions`), so its fan-in can be anything
      from the number of out_degree-0 nodes to the number of nodes. The line passes if some
      fan-in in that span is within [min_num, max_num].
    """
    if output_connections not in ("sinks", "any"):
        raise ValueError(f"Invalid output_connections: {output_connections!r} (expected 'sinks' or 'any')")
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
    if output_connections == "sinks":
        return min_num <= sink_in_degree <= max_num
    # 'any': the output's fan-in ranges over [sink_in_degree, num_nodes].
    return sink_in_degree <= max_num and num_nodes >= min_num

def num_extra_edges(num_edges: int, num_gates: int):
    """
    The number of edges beyond two per gate: ``num_edges - 2 * num_gates``.

    Every gate has at least two inputs, so a circuit with g gates has at least 2g edges, and
    this is 0 for a circuit whose gates all have exactly two inputs. A 3-input gate adds one
    extra edge, a 4-input gate two, and so on. Limiting extra edges (rather than total edges)
    applies the same limit to circuits of every size.
    """
    return num_edges - 2 * num_gates


def edges_within_limits(graph, max_edges=None, max_extra_edges=None):
    """
    Verifies that a circuit's graph has at most `max_edges` edges and at most `max_extra_edges`
    edges beyond two per gate (see `num_extra_edges`). A limit of ``None`` is not checked.

    Every node with in_degree > 0 counts as a gate, so apply this to a graph before NOT gates
    are inserted (a NOT gate would count as a gate and add an edge).

    Most of the time spent generating circuits goes to gate assignments of graphs with many
    edges, and few of those circuits pass the circuit filters, so limiting edges makes
    generation much faster while losing few circuits.
    """
    num_edges = graph.number_of_edges()
    if max_edges is not None and num_edges > max_edges:
        return False
    if max_extra_edges is not None:
        num_gates = sum(1 for node in graph.nodes() if graph.in_degree(node) > 0)
        if num_extra_edges(num_edges, num_gates) > max_extra_edges:
            return False
    return True


def nauty_edges_within_limits(nauty_graph: str, max_edges=None, max_extra_edges=None, min_output_fan_in: int = 1):
    """
    The `edges_within_limits` check applied to one line of nauty's ``directg -aTq`` output
    (``num_vertices num_edges u1 v1 u2 v2 ...``), before the output gate is added.

    Returns False only if no circuit built from this DAG can pass `edges_within_limits`. The
    output gate (see `circuit_generator._add_single_sink` and
    `circuit_generator._single_sink_extensions`) reads every node with out_degree 0 and at
    least `min_output_fan_in` nodes, so the finished circuit has at least
    ``num_edges + max(num_out_degree_0_nodes, min_output_fan_in)`` edges, and exactly one more
    gate than the DAG has non-source nodes.
    """
    if max_edges is None and max_extra_edges is None:
        return True
    parts = list(map(int, nauty_graph.split()))
    num_nodes, num_edges = parts[0], parts[1]
    has_in_edge = [False] * num_nodes
    has_out_edge = [False] * num_nodes
    for u in parts[2::2]:
        has_out_edge[u] = True
    for v in parts[3::2]:
        has_in_edge[v] = True
    fewest_edges = num_edges + max(has_out_edge.count(False), min_output_fan_in)
    if max_edges is not None and fewest_edges > max_edges:
        return False
    if max_extra_edges is not None:
        num_gates = has_in_edge.count(True) + 1
        if num_extra_edges(fewest_edges, num_gates) > max_extra_edges:
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

def max_shared_subcircuit_depth(graph):
    """
    Measures how deep the deepest shared intermediate term in a circuit generated from this graph is.

    A "shared" node is a non-source node (in_degree > 0) whose output is used more than once
    (out_degree >= 2). The depth of the subcircuit rooted at a node is the length (in edges) of the
    longest path from any source to that node -- the same convention used for overall circuit depth
    (`nx.dag_longest_path_length`). So a shared gate fed directly by inputs has depth 1.

    :param graph: A networkx DiGraph (must be acyclic) representing a circuit.
    :returns: The maximum subcircuit depth over all shared non-source nodes, or 0 if no
              non-source node is shared (i.e., only sources/inputs are reused).
    """
    level = {}
    deepest = 0
    for node in nx.topological_sort(graph):
        level[node] = max((level[p] + 1 for p in graph.predecessors(node)), default=0)
        if level[node] > 0 and graph.out_degree(node) >= 2:
            deepest = max(deepest, level[node])
    return deepest


def has_shared_internal_node(graph):
    """
    Verifies that a circuit generated from this graph will have shared intermediate terms.
    
    In other words, this graph is not a tree, with at least one shared node that is not a source.

    :param graph: A networkx DiGraph representing a circuit.
    :returns: True if at least one non-source node (a node with in_degree > 0) has an out_degree of at least 2.
    """
    return max_shared_subcircuit_depth(graph) > 0


def only_leaves_shared(graph):
    """
    Verifies that a circuit generated from this graph will not have any shared intermediate terms.
    
    In other words, this graph is almost a tree, with the only shared nodes being the sources.

    :param graph: A networkx DiGraph representing a circuit.
    :returns: True if every non-source node (a node with in_degree > 0) has an out_degree of at most 1.
    """
    return not has_shared_internal_node(graph)
