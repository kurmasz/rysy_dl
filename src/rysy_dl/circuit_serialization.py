"""
circuit_serialization.py

Convert a circuit's networkx DiGraph to and from a plain dict of built-in types
(dict/list/str/int), suitable for pickling or JSON serialization -- e.g. for
`circuit_database`'s pickled blob column, or PrairieLearn's JSON question data.

:copyright: 2026, Zachary and Andrew Kurmas
"""

import networkx as nx


def circuit_to_serializable(graph):
    """
    Extract the minimal data needed to recreate `graph` with `serializable_to_circuit`,
    in a form built only from plain types (dict/list/str/int) so it's safe to pickle or
    serialize as JSON.

    A circuit graph's only essential content is (1) which node has which 'gate'
    attribute, (2) how the nodes are wired together, and (3) any 'label' attribute
    assigned to source nodes (e.g. by `circuit_layout.assign_source_labels`, or any
    other labeling scheme a caller might use). Labels are captured as-is rather than
    recomputed by `serializable_to_circuit`, so the result reflects whatever labeling
    algorithm was actually used, not a hardcoded one.

    :param graph: A networkx DiGraph representing a circuit, where every node has
                  a 'gate' attribute (e.g. 'src', 'AND', 'NOT', ...).
    :returns: A dict with ``'gates'``, a ``{node: gate_type}`` mapping covering every
              node (including isolated ones, if any); ``'edges'``, a list of
              ``(u, v)`` tuples; and ``'labels'``, a ``{node: label}`` mapping
              covering only the nodes that have a 'label' attribute (source nodes,
              if labels have been assigned -- an empty dict otherwise).
    """
    gates = {node: graph.nodes[node]['gate'] for node in graph.nodes()}
    edges = list(graph.edges())
    labels = {node: graph.nodes[node]['label'] for node in graph.nodes() if 'label' in graph.nodes[node]}
    return {'gates': gates, 'edges': edges, 'labels': labels}


def serializable_to_circuit(data):
    """
    Rebuild the networkx DiGraph produced by `circuit_to_serializable`.

    :param data: A dict in the format returned by `circuit_to_serializable`. Node ids
                 (both the keys of `gates`/`labels` and the endpoints in `edges`) are
                 accepted as either int or str, since round-tripping through JSON
                 (e.g. PrairieLearn persisting question data between phases) turns
                 dict keys into strings while leaving ints inside list/tuple values
                 (like `edges`) untouched -- so all node ids are normalized back to
                 int to make sure they refer to the same nodes. `'labels'` is
                 optional and defaults to empty, for data serialized before any
                 labels were assigned.
    :returns: A networkx DiGraph equivalent to the one `data` was extracted from,
              including the original 'label' attributes on labeled nodes.
    """
    graph = nx.DiGraph()
    for node, gate in data['gates'].items():
        graph.add_node(int(node), gate=gate)
    graph.add_edges_from((int(u), int(v)) for u, v in data['edges'])
    for node, label in data.get('labels', {}).items():
        graph.nodes[int(node)]['label'] = label
    return graph
