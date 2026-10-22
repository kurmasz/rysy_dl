"""
circuit_filters.py

Boolean functions used to filter networkx DiGraphs that have gates assigned to their
nodes. Unlike graph_filters.py, these functions inspect each node's "gate" attribute,
not just the graph structure.
"""

import networkx as nx

# Gate Filters
# (parent gate, child gate) pairs where the child can be trivially folded into the
# parent via associativity, e.g. AND(AND(a,b), c) == AND(a,b,c) and
# NAND(AND(a,b), c) == NAND(a,b,c). NAND/NOR are not associative, so a NAND or NOR
# child never collapses this way.
_COLLAPSABLE_PARENT_CHILD_GATES = {
    ("AND", "AND"),
    ("NAND", "AND"),
    ("OR", "OR"),
    ("NOR", "OR"),
}
# parent gate -> the one child gate that collapses into it (derived from the pairs above)
_FORBIDDEN_CHILD_GATE = dict(_COLLAPSABLE_PARENT_CHILD_GATES)


def no_collapsable_children(graph):
    # filters out circuits where any single parent/child gate pair could be trivially merged
    for parent_node in graph.nodes():
        parent_gate = graph.nodes[parent_node].get("gate")
        for child_node in graph.predecessors(parent_node):
            child_gate = graph.nodes[child_node].get("gate")
            if (parent_gate, child_gate) in _COLLAPSABLE_PARENT_CHILD_GATES:
                return False
    return True


# xor/xnor>2 input
def valid_xor_xnor_fan_in_filter(graph):
    # filters out graphs that have an XOR or XNOR gate with more or less than 2 inputs
    for node in graph.nodes():
        # looks through all nodes
        if (
            graph.nodes[node].get("gate") == "XOR"
            or graph.nodes[node].get("gate") == "XNOR"
        ):
            # checks if node is a XOR or XNOR gate
            if graph.in_degree(node) != 2:
                # returns false if doesn't have two inputs
                # print("XOR fail")
                return False
    return True


# Constant subexpressions
def _subexpression_truth_tables(graph):
    # Returns {node: truth table} for every node. Each truth table is an int whose bit i
    # is the node's output on input combination i (bit-parallel evaluation), so all
    # 2^n input combinations are evaluated in a single pass over the graph.
    sources = sorted(node for node in graph.nodes() if graph.in_degree(node) == 0)
    num_rows = 2 ** len(sources)
    all_ones = (1 << num_rows) - 1

    tables = {}
    for index, source in enumerate(sources):
        # source `index` is 1 on exactly the rows whose bit `index` is set
        tables[source] = sum(1 << row for row in range(num_rows) if (row >> index) & 1)

    for node in nx.topological_sort(graph):
        if node in tables:
            continue
        gate = graph.nodes[node].get("gate")
        values = [tables[child] for child in graph.predecessors(node)]

        if gate in ("AND", "NAND"):
            result = all_ones
            for value in values:
                result &= value
        elif gate in ("OR", "NOR"):
            result = 0
            for value in values:
                result |= value
        elif gate in ("XOR", "XNOR"):
            result = 0
            for value in values:
                result ^= value
        elif gate == "NOT":
            result = values[0]
        else:
            raise ValueError(f"Unsupported gate type: {gate}")

        if gate in ("NAND", "NOR", "XNOR", "NOT"):
            result ^= all_ones
        tables[node] = result

    return tables, all_ones


def no_constant_subexpressions(graph):
    # filters out circuits where some gate outputs the same value (always 0 or always 1)
    # for every combination of the circuit's inputs, e.g. A*A' or A + (A' + B)
    tables, all_ones = _subexpression_truth_tables(graph)
    for node, table in tables.items():
        if graph.in_degree(node) > 0 and table in (0, all_ones):
            return False
    return True
