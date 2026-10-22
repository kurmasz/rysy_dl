"""
ast_analyzer.py

Functions to evaluate and compare Boolean expressions.
"""

try:
    from .boolean_expression_parser import And, Node, Not, Or, Var, Xor, parse_boolean_expression
except ImportError:
    # Allows running this file directly (e.g. `python ast_analyzer.py`) for
    # quick manual checks, where the leading-dot relative import above fails
    # because the file isn't being run as part of the rysy_dl package.
    from boolean_expression_parser import And, Node, Not, Or, Var, Xor, parse_boolean_expression


def _walk_variable_names(ast: Node):
    """Yield each variable name in `ast` once per occurrence, in left-to-right,
    depth-first order -- the same traversal `boolean_expression_to_netlist.
    build_netlist` uses to assign node ids."""
    if isinstance(ast, Var):
        yield ast.name
    elif isinstance(ast, Not):
        yield from _walk_variable_names(ast.child)
    elif isinstance(ast, (And, Or)):
        for child in ast.children:
            yield from _walk_variable_names(child)
    elif isinstance(ast, Xor):
        yield from _walk_variable_names(ast.left)
        yield from _walk_variable_names(ast.right)
    else:
        raise TypeError(f"Unknown AST node type: {type(ast).__name__}")


def _collect_variable_names(ast: Node) -> list:
    """Return the distinct variable names appearing in `ast`, in order of
    first appearance (left-to-right, depth-first)."""
    names = []
    seen = set()
    for name in _walk_variable_names(ast):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def evaluate_ast(ast: Node, assignment: dict) -> bool:
    """
        Evaluate the Boolean expression represented by `ast` under a given
        variable assignment.

        Args:
            ast (Node): A boolean expression AST.
            assignment (dict[str, bool]): Maps every variable name appearing
                in `ast` to a boolean value.

        Returns:
            bool: The value of the expression under `assignment`.
    """
    if isinstance(ast, Var):
        return bool(assignment[ast.name])
    if isinstance(ast, Not):
        return not evaluate_ast(ast.child, assignment)
    if isinstance(ast, And):
        return all(evaluate_ast(child, assignment) for child in ast.children)
    if isinstance(ast, Or):
        return any(evaluate_ast(child, assignment) for child in ast.children)
    if isinstance(ast, Xor):
        return evaluate_ast(ast.left, assignment) ^ evaluate_ast(ast.right, assignment)
    raise TypeError(f"Unknown AST node type: {type(ast).__name__}")


def truth_table_from_ast(ast: Node, variable_order: list = None) -> dict:
    """
        Return the truth table for the Boolean expression represented by the AST.

        Args:
            ast (Node): A boolean expression AST.
            variable_order (list[str] | None): The variable names to use, in
                the order they should map to bits (most significant bit
                first). Defaults to the order variables first appear in `ast`
                (left-to-right, depth-first) -- the same order a circuit
                built from this expression via `boolean_expression_to_netlist`
                would assign via `assign_source_labels`.

        Returns:
            dict[str, int]: Maps a zero-padded bit string -- one bit per
            variable, in `variable_order` -- to the expression's output (0 or
            1) under that assignment.
    """
    names = variable_order if variable_order is not None else _collect_variable_names(ast)
    truth_table = {}
    for row in range(2 ** len(names)):
        bit_string = format(row, 'b').zfill(len(names))
        assignment = {name: bit == '1' for name, bit in zip(names, bit_string)}
        truth_table[bit_string] = 1 if evaluate_ast(ast, assignment) else 0
    return truth_table


def _canonical_sort_key(node: Node):
    """
        A total, order-independent sort key for `node`.

        Every key is a tuple tagged with the node's type first, so keys for
        different node types are always comparable (Python only compares the
        rest of the tuple once the tags already match). An And/Or/Xor node's
        key is built from the *sorted* keys of its own (already-canonical)
        children, so two nodes with the same multiset of children -- written
        in any order -- produce an identical key.
    """
    if isinstance(node, Var):
        return ('Var', node.name)
    if isinstance(node, Not):
        return ('Not', _canonical_sort_key(node.child))
    if isinstance(node, And):
        return ('And', tuple(sorted(_canonical_sort_key(c) for c in node.children)))
    if isinstance(node, Or):
        return ('Or', tuple(sorted(_canonical_sort_key(c) for c in node.children)))
    if isinstance(node, Xor):
        return ('Xor', tuple(sorted([_canonical_sort_key(node.left), _canonical_sort_key(node.right)])))
    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


def canonicalize_ast(ast: Node) -> Node:
    """
        Return a canonical form of the AST, such that two equivalent Boolean expressions
        will have the same canonical form.

        Two ASTs canonicalize to equal results exactly when they describe the
        same gates wired the same way, up to reordering the operands of each
        And/Or/Xor node -- i.e. `A*B + C` and `C + B*A` canonicalize equal.
        This is NOT Boolean-algebra equivalence: canonicalization never
        applies associativity, absorption, De Morgan, etc., so `A*B*C` (one
        3-input And) and `(A*B)*C` (two nested 2-input Ands) still
        canonicalize to different, unequal results, even though they compute
        the same function.

        Returns a brand-new AST; `ast` itself is left untouched.
    """
    if isinstance(ast, Var):
        return Var(ast.name)
    if isinstance(ast, Not):
        return Not(canonicalize_ast(ast.child))
    if isinstance(ast, And):
        children = sorted((canonicalize_ast(c) for c in ast.children), key=_canonical_sort_key)
        return And(children)
    if isinstance(ast, Or):
        children = sorted((canonicalize_ast(c) for c in ast.children), key=_canonical_sort_key)
        return Or(children)
    if isinstance(ast, Xor):
        left = canonicalize_ast(ast.left)
        right = canonicalize_ast(ast.right)
        if _canonical_sort_key(right) < _canonical_sort_key(left):
            left, right = right, left
        return Xor(left, right)
    raise TypeError(f"Unknown AST node type: {type(ast).__name__}")


def structurally_equal(a: Node, b: Node) -> bool:
    """
        Return True iff `a` and `b` describe the same gates wired the same
        way -- same gate types, same fan-in, same nesting -- ignoring only
        the order each And/Or/Xor node's operands were written in.

        This is NOT Boolean-algebra equivalence: like `canonicalize_ast`
        (which this is built on), it never applies associativity,
        absorption, De Morgan, etc., so `A*B*C` (one 3-input And) and
        `(A*B)*C` (two nested 2-input Ands) are NOT structurally equal here,
        even though they compute the same function. To compare by function
        instead of by structure, use `evaluate_ast`/`truth_table_from_ast`.
    """
    return canonicalize_ast(a) == canonicalize_ast(b)


if __name__ == "__main__":
    ast = parse_boolean_expression("A(B+C) + A(BC)")
    ast2 = canonicalize_ast(ast)
    print("------------")
    print(ast)
    print(ast2)