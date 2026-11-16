"""
circuit_serialization.py

Convert a circuit's networkx DiGraph to and from a plain dict of built-in types
(dict/list/str/int), suitable for JSON serialization -- e.g. for PrairieLearn's JSON
question data -- and to and from a compact byte string, used for `circuit_database`'s
graph blob column.

:copyright: 2026, Zachary and Andrew Kurmas
"""

import networkx as nx

# One byte per gate in the compact encoding. Codes must never be reused for a different gate,
# since stored databases depend on them.
_GATE_CODES = {'src': 1, 'AND': 2, 'OR': 3, 'XOR': 4, 'NAND': 5, 'NOR': 6, 'XNOR': 7, 'NOT': 8}
_GATES_BY_CODE = {code: gate for gate, code in _GATE_CODES.items()}


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


def circuit_to_bytes(graph):
    """
    Encode `graph` as a compact byte string that `bytes_to_circuit` turns back into an identical
    graph: the same nodes, gates and labels, with nodes and edges in the same order (the order of
    a node's successors matters, e.g. to `circuit_generator.all_reduced_source_nots`).

    Format: the number of nodes n; one gate code per node, for nodes 0 to n-1; the number of edges,
    then each edge as two node numbers, in `graph.edges()` order; the number of labeled nodes, then
    for each one its node number, the length of its label in UTF-8 bytes, and the label. A typical
    circuit takes about 40 bytes, a fifth of its pickled `circuit_to_serializable` dict.

    :param graph: A networkx DiGraph whose nodes are numbered 0 to n-1, in that order, each with a
                  'gate' attribute.
    :returns: The encoded bytes.
    :raises ValueError: If the nodes aren't numbered 0 to n-1 in order, a gate type is unknown, or
                        the graph has more than 255 nodes or edges, or a label longer than 255 bytes.
    """
    nodes = list(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("circuit_to_bytes requires nodes numbered 0 to n-1, in that order")
    edges = list(graph.edges())
    labels = [(node, graph.nodes[node]['label'].encode('utf-8'))
              for node in nodes if 'label' in graph.nodes[node]]
    if len(nodes) > 255 or len(edges) > 255 or any(len(label) > 255 for _, label in labels):
        raise ValueError("circuit_to_bytes supports at most 255 nodes, edges and bytes per label")
    try:
        gates = [_GATE_CODES[graph.nodes[node]['gate']] for node in nodes]
    except KeyError as e:
        raise ValueError(f"circuit_to_bytes: unknown or missing gate {e}") from None

    out = bytearray([len(nodes), *gates, len(edges)])
    for u, v in edges:
        out += bytes((u, v))
    out.append(len(labels))
    for node, label in labels:
        out += bytes((node, len(label))) + label
    return bytes(out)


def bytes_to_circuit(data):
    """
    Rebuild the networkx DiGraph encoded by `circuit_to_bytes`.

    :param data: Bytes produced by `circuit_to_bytes`.
    :returns: A networkx DiGraph identical to the one that was encoded.
    """
    graph = nx.DiGraph()
    num_nodes = data[0]
    for node in range(num_nodes):
        graph.add_node(node, gate=_GATES_BY_CODE[data[1 + node]])
    pos = 1 + num_nodes
    num_edges = data[pos]
    pos += 1
    graph.add_edges_from((data[pos + 2 * i], data[pos + 2 * i + 1]) for i in range(num_edges))
    pos += 2 * num_edges
    num_labels = data[pos]
    pos += 1
    for _ in range(num_labels):
        node, length = data[pos], data[pos + 1]
        graph.nodes[node]['label'] = data[pos + 2:pos + 2 + length].decode('utf-8')
        pos += 2 + length
    return graph
