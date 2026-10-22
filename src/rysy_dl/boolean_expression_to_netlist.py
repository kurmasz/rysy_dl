"""
boolean_expression_to_netlist.py

Builds a netlist (the networkx DiGraph that Circuit wraps -- see circuit.py)
from the AST produced by boolean_expression_parser. This is a third producer
of netlists alongside circuit_generator's random DAG enumeration and
circuit_database's stored circuits, except that here the structure and every
gate type are already fully determined by the expression -- there is no
separate gate-assignment step.

Two AST design decisions (documented in boolean_expression_parser's module
docstring) drive how this module builds the graph:

- Repeated variable names are *not* repeated source nodes. The parser
  produces a fresh Var object per occurrence, but that's an artifact of
  parsing, not a description of the physical circuit: `A*B + A*C` is one
  input pin `A` fanning out to two different gates. This module groups Var
  nodes by `.name` and reuses a single 'src' node for every occurrence of
  the same name.

- Gate nodes (And/Or/Not/Xor) are never deduplicated, even when two
  subtrees are structurally identical (e.g. both operands of
  `(A*B) + (A*B)'`) -- each occurrence is a distinct physical gate, so a
  fresh netlist node is created per AST node instance.

One consequence of grouping sources by name: a plain (non-multi) DiGraph
cannot hold two parallel edges, so an expression that wires the very same
source directly into the very same gate more than once (e.g. `A*A`, `A^A`)
cannot be represented without silently dropping fan-in below what the
expression wrote. Rather than do that, `build_netlist` raises
NetlistBuildError for that specific case.

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

import networkx as nx

from .boolean_expression_parser import And, Node, Not, Or, Var, Xor


class NetlistBuildError(Exception):
    """Raised when an AST cannot be represented as a netlist."""


def build_netlist(ast: Node) -> nx.DiGraph:
    """
    Build a netlist (networkx DiGraph) from a boolean-expression AST.

    Each Var becomes (or reuses) a 'src' node labeled with its variable
    name. Each Not/And/Or/Xor becomes a fresh 'NOT'/'AND'/'OR'/'XOR' gate
    node, wired to its operands in left-to-right order, so that
    `netlist.predecessors(node)` yields operands in the same order they
    appeared in the expression. The root of the AST is always the
    netlist's single sink (the only node with out-degree 0).

    Args:
        ast (Node): The root of a boolean expression AST, as returned by
            `boolean_expression_parser.parse_boolean_expression`.

    Returns:
        networkx.DiGraph: A netlist ready to hand to `Circuit`.

    Raises:
        NetlistBuildError: If the same source would have to feed the same
            gate through more than one identical, unparenthesized operand
            (e.g. `A*A`), which cannot be represented as distinct wires in
            a netlist that has one edge per (source, gate) pair.
    """
    netlist = nx.DiGraph()
    sources = {}
    _add_node(netlist, ast, sources)
    return netlist


def _add_node(netlist: nx.DiGraph, node: Node, sources: dict) -> int:
    """Recursively add `node` (and any new descendants) to `netlist`; return its id."""
    if isinstance(node, Var):
        if node.name in sources:
            return sources[node.name]
        node_id = netlist.number_of_nodes()
        netlist.add_node(node_id, gate='src', label=node.name)
        sources[node.name] = node_id
        return node_id

    node_id = netlist.number_of_nodes()

    if isinstance(node, Not):
        netlist.add_node(node_id, gate='NOT')
        _connect(netlist, _add_node(netlist, node.child, sources), node_id)
    elif isinstance(node, And):
        netlist.add_node(node_id, gate='AND')
        for child in node.children:
            _connect(netlist, _add_node(netlist, child, sources), node_id)
    elif isinstance(node, Or):
        netlist.add_node(node_id, gate='OR')
        for child in node.children:
            _connect(netlist, _add_node(netlist, child, sources), node_id)
    elif isinstance(node, Xor):
        netlist.add_node(node_id, gate='XOR')
        _connect(netlist, _add_node(netlist, node.left, sources), node_id)
        _connect(netlist, _add_node(netlist, node.right, sources), node_id)
    else:
        raise NetlistBuildError(f"Unknown AST node type: {type(node).__name__}")

    return node_id


def _connect(netlist: nx.DiGraph, source_id: int, gate_id: int) -> None:
    if netlist.has_edge(source_id, gate_id):
        label = netlist.nodes[source_id].get('label', source_id)
        raise NetlistBuildError(
            f"Source {label!r} would have to feed the same gate through two "
            "identical operands (e.g. 'A*A' or 'A^A'); this can't be "
            "represented without reducing that gate's fan-in below the "
            "number of operands written in the expression."
        )
    netlist.add_edge(source_id, gate_id)
