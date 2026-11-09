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


def no_absorption(graph):
    # Determine whether the graph has any absorption patterns, e.g. A + (A * B) = A or A * (A + B) = A
    # Be sure this filter detects all absorption patterns, not just the ones that are easy to see.
    # For example, A + (A * (B + C)) = A is also an absorption pattern.
    #
    # Absorption is X + Y = X whenever Y implies X (and dually X * Y = X whenever X implies Y),
    # where X and Y can be arbitrary subexpressions. So rather than pattern-match, this
    # filter compares truth tables: Y implies X exactly when Y's table is a subset of X's.
    #
    # Each gate is first flattened into the terms of one big OR (or AND), looking through
    # same-type children (associativity), NOTs, and NAND/NOR (De Morgan). That way
    # A + C + (A * B) is caught even when A + C is its own OR gate, and A + (A' + B)'
    # (= A + A*B') is caught even though no AND gate appears. Any term implying a different
    # term of the same flattened OR (or being implied by one in a flattened AND) is absorption.
    # Because flattening can split the absorbing X into several terms (A + B + (A + B)*C,
    # (A*B)' + (A*B)'*C), a term is also absorbed when one of its factors implies the other
    # terms together.
    #
    # See no_visible_absorption for a version that rejects only absorption a student would
    # see in the expression or diagram.
    #
    # Notes:
    #   * Two terms with identical truth tables (X + X) also count, as does a constant term
    #     (0 + X, 1 * X); no_constant_subexpressions rejects the latter as well.
    #   * Only absorption among the terms of a single flattened AND/OR is detected. A
    #     redundancy that appears only after distributing, e.g. A + B*(C + A*D) = A + B*C,
    #     or one that needs two terms together (consensus, A*B + A'*C + B*C) is not.
    #   * XOR/XNOR gates are not flattened; they are treated as opaque terms.
    tables, all_ones = _subexpression_truth_tables(graph)

    # memo of (node, positive) -> truth tables of the OR-terms of node (positive) or node' (not positive)
    disjuncts = {}

    def or_terms(node, positive):
        key = (node, positive)
        if key not in disjuncts:
            gate = graph.nodes[node].get("gate") if graph.in_degree(node) > 0 else None
            children = list(graph.predecessors(node))
            if gate == "NOT":
                terms = or_terms(children[0], not positive)
            elif (gate, positive) in (("OR", True), ("NOR", False)):
                # A + B + ...  /  ((A + B + ...)')' -- the children themselves are the terms
                terms = [t for child in children for t in or_terms(child, True)]
            elif (gate, positive) in (("NAND", True), ("AND", False)):
                # (A * B * ...)' = A' + B' + ... -- the complemented children are the terms
                terms = [t for child in children for t in or_terms(child, False)]
            else:
                # sources, XOR/XNOR, and AND/OR in the polarity that doesn't flatten into an OR
                terms = [(node, positive)]
            disjuncts[key] = terms
        return disjuncts[key]

    def table(term):
        node, positive = term
        return tables[node] if positive else tables[node] ^ all_ones

    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            continue
        # Positive polarity finds OR absorption (X + X*Y). Negative polarity finds AND
        # absorption, because X * (X + Y) is absorbed iff X' + (X + Y)' = X' + X'*Y' is.
        for positive in (True, False):
            terms = or_terms(node, positive)
            term_tables = [table(t) for t in terms]
            for j, absorbed in enumerate(term_tables):
                # absorbed implies some single other term
                for i, absorber in enumerate(term_tables):
                    if i != j and absorbed & ~absorber == 0:
                        return False
                # X + X*Y where X spans several terms (flattening split it up), e.g.
                # (A*B)' + (A*B)'*C, whose terms are A', B', (A*B)'*C: some factor of the
                # absorbed term implies the other terms together. Only for terms with 2+
                # factors -- otherwise this would flag consensus-style redundancy.
                term_node, term_positive = terms[j]
                factors = [table(t) ^ all_ones for t in or_terms(term_node, not term_positive)]
                if len(factors) >= 2:
                    others = 0
                    for i, other in enumerate(term_tables):
                        if i != j:
                            others |= other
                    if any(factor & ~others == 0 for factor in factors):
                        return False
    return True


# Gates whose inputs, together with the inputs of a child of the given type, are visibly
# one AND (or OR) list: AND(AND(a, b), c) reads as a*b*c, and NAND(AND(a, b), c) as (a*b*c)'.
_FLATTENS_CHILD = {"AND": "AND", "NAND": "AND", "OR": "OR", "NOR": "OR"}
# NOT(X) is written the same way as X's gate with a bubble: (A*B)' is NAND(A, B).
_INVERTED_GATE = {"AND": "NAND", "OR": "NOR", "XOR": "XNOR"}
_GATES = {"AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"}


def no_visible_absorption(graph):
    # Like no_absorption, but only rejects absorption a student can see in the expression or
    # circuit diagram -- i.e., circuits that look dumb -- not every logical implication.
    #
    # Rejected: X + X*Y and X*(X + Y), where X and Y are written the same way, e.g.
    #   A + A*B,  A' + A'*B,  A*B + A*B*C,  (A + B)*(A + B + C),  A + A*(B + C),
    #   NAND(A, A + B), NOR(A, A*B), and repeated terms such as A*B + B*A.
    # Accepted (no_absorption rejects these):
    #   A + (A' + B)'       -- absorption only after applying De Morgan
    #   (A ^ B)*(A*B)'      -- A ^ B implies (A*B)', but nothing on the page shows it
    #   A*B*C*(B*(A ^ D)*D)'
    #
    # Subexpressions are compared by a canonical written form instead of by truth table:
    # inputs are sorted (A*B is B*A), same-type children are merged (A*(B*C) is A*B*C), and
    # NOT(AND/OR/XOR) is the same as NAND/NOR/XNOR. Then, at every OR/NOR gate, term T is
    # absorbed when
    #   * some other term S has factors that are a subset of T's factors (S + S*Y), or
    #   * one of T's factors is an OR of other terms (A + B + (A + B)*C);
    # and dually at every AND/NAND gate. Every circuit this rejects, no_absorption also rejects.
    canonical_forms = {}

    def gate_of(node):
        return graph.nodes[node].get("gate") if graph.in_degree(node) > 0 else None

    def operands(node):
        # node's inputs as one flattened AND/OR list (just its inputs, for other gates)
        flattens = _FLATTENS_CHILD.get(gate_of(node))
        result = []
        for child in graph.predecessors(node):
            if flattens is not None and gate_of(child) == flattens:
                result.extend(operands(child))
            else:
                result.append(child)
        return result

    def written(gate, nodes):
        return gate + "(" + ",".join(sorted(canonical(n) for n in nodes)) + ")"

    def canonical(node):
        if node not in canonical_forms:
            gate = gate_of(node)
            if gate is None:
                form = f"src{node}"
            elif gate not in _GATES:
                raise ValueError(f"Unsupported gate type: {gate}")
            elif gate == "NOT":
                child = next(iter(graph.predecessors(node)))
                if gate_of(child) in _INVERTED_GATE:
                    form = written(_INVERTED_GATE[gate_of(child)], operands(child))
                else:
                    form = "NOT(" + canonical(child) + ")"
            else:
                form = written(gate, operands(node))
            canonical_forms[node] = form
        return canonical_forms[node]

    def parts(node, gate):
        # the factors of node if it is an AND (gate == "AND"), or its terms if it is an
        # OR (gate == "OR"); anything else is a single factor/term
        if gate_of(node) == gate:
            return {canonical(n) for n in operands(node)}
        return {canonical(node)}

    for node in graph.nodes():
        gate = gate_of(node)
        if gate is None:
            continue
        canonical(node)  # rejects unsupported gates even where no AND/OR is involved
        if gate in ("OR", "NOR"):
            # the terms of an OR, each viewed as its set of AND-factors
            outer, inner = "OR", "AND"
        elif gate in ("AND", "NAND"):
            # the factors of an AND, each viewed as its set of OR-terms
            outer, inner = "AND", "OR"
        else:
            continue
        items = operands(node)
        item_forms = [canonical(n) for n in items]
        item_parts = [parts(n, inner) for n in items]
        for j, absorbed in enumerate(items):
            others = set(item_forms[:j] + item_forms[j + 1:])
            # S + S*Y (or S*(S + Y)): another item's parts are a subset of this item's parts
            for i, absorber_parts in enumerate(item_parts):
                if i != j and absorber_parts <= item_parts[j]:
                    return False
            # A + B + (A + B)*C (or A*B*(A*B + C)): one of this item's parts is itself an
            # OR (AND) of other items
            if gate_of(absorbed) == inner:
                for part in operands(absorbed):
                    if gate_of(part) == outer and {canonical(n) for n in operands(part)} <= others:
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


def every_input_affects_output(graph):
    # filters out circuits with an input that never changes the output, e.g. A + A*B (= A, so B
    # doesn't matter) or A ^ (A ^ B) (= B, so A doesn't matter). In the truth table, flipping
    # that input's column never changes the output column, which students notice.
    #
    # Input i affects an output if, for some row with bit i = 0, the output differs from the
    # row with bit i = 1. Bit r of a truth table is row r, so shifting the table right by
    # 2**i lines each row with bit i = 1 up with its partner; `rows_with_bit_clear` keeps
    # only the pairs. If a circuit has several outputs, each input must affect at least one.
    tables, _ = _subexpression_truth_tables(graph)
    sources = sorted(node for node in graph.nodes() if graph.in_degree(node) == 0)
    outputs = [node for node in graph.nodes() if graph.out_degree(node) == 0 and graph.in_degree(node) > 0]
    num_rows = 2 ** len(sources)
    for index in range(len(sources)):
        step = 1 << index
        rows_with_bit_clear = sum(1 << row for row in range(num_rows) if not row & step)
        if not any(
            (tables[out] ^ (tables[out] >> step)) & rows_with_bit_clear for out in outputs
        ):
            return False
    return True


# Filters whose answer depends only on the gate assignment, not on NOT gates later inserted on
# the edges leaving the sources (circuit_generator.insert_random_source_edge_nots): such a NOT
# replaces a source -> gate edge one-for-one, so it can't change a gate's fan-in or create or
# remove a collapsable parent/child pair. Circuit.randomize checks these before adding NOTs.
GATE_ONLY_FILTERS = frozenset({no_collapsable_children, valid_xor_xnor_fan_in_filter})
