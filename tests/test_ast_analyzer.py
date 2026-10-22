import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.ast_analyzer import canonicalize_ast, evaluate_ast, structurally_equal, truth_table_from_ast
from rysy_dl.boolean_expression_parser import And, Not, Or, Var, Xor, parse_boolean_expression


# ---------------------------------------------------------------------------
# evaluate_ast
# ---------------------------------------------------------------------------

def test_evaluate_var():
    assert evaluate_ast(Var('A'), {'A': True}) is True
    assert evaluate_ast(Var('A'), {'A': False}) is False


def test_evaluate_not():
    assert evaluate_ast(Not(Var('A')), {'A': True}) is False
    assert evaluate_ast(Not(Var('A')), {'A': False}) is True


@pytest.mark.parametrize('a,b,expected', [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, False),
])
def test_evaluate_and(a, b, expected):
    assert evaluate_ast(And([Var('A'), Var('B')]), {'A': a, 'B': b}) == expected


def test_evaluate_and_three_operands():
    node = And([Var('A'), Var('B'), Var('C')])
    assert evaluate_ast(node, {'A': True, 'B': True, 'C': True}) is True
    assert evaluate_ast(node, {'A': True, 'B': True, 'C': False}) is False


@pytest.mark.parametrize('a,b,expected', [
    (True, True, True),
    (True, False, True),
    (False, True, True),
    (False, False, False),
])
def test_evaluate_or(a, b, expected):
    assert evaluate_ast(Or([Var('A'), Var('B')]), {'A': a, 'B': b}) == expected


@pytest.mark.parametrize('a,b,expected', [
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (False, False, False),
])
def test_evaluate_xor(a, b, expected):
    assert evaluate_ast(Xor(Var('A'), Var('B')), {'A': a, 'B': b}) == expected


def test_evaluate_nested_expression():
    # A*B + C'
    node = Or([And([Var('A'), Var('B')]), Not(Var('C'))])
    assert evaluate_ast(node, {'A': True, 'B': True, 'C': True}) is True
    assert evaluate_ast(node, {'A': False, 'B': True, 'C': True}) is False
    assert evaluate_ast(node, {'A': False, 'B': True, 'C': False}) is True


def test_evaluate_repeated_variable_is_fine():
    # A*A is a well-defined expression (equivalent to A) even though it
    # can't be represented as a netlist.
    node = And([Var('A'), Var('A')])
    assert evaluate_ast(node, {'A': True}) is True
    assert evaluate_ast(node, {'A': False}) is False


# ---------------------------------------------------------------------------
# truth_table_from_ast
# ---------------------------------------------------------------------------

def test_truth_table_single_variable():
    assert truth_table_from_ast(Var('A')) == {'0': 0, '1': 1}


def test_truth_table_and():
    node = And([Var('A'), Var('B')])
    assert truth_table_from_ast(node) == {
        '00': 0,
        '01': 0,
        '10': 0,
        '11': 1,
    }


def test_truth_table_or():
    node = Or([Var('A'), Var('B')])
    assert truth_table_from_ast(node) == {
        '00': 0,
        '01': 1,
        '10': 1,
        '11': 1,
    }


def test_truth_table_xor():
    node = Xor(Var('A'), Var('B'))
    assert truth_table_from_ast(node) == {
        '00': 0,
        '01': 1,
        '10': 1,
        '11': 0,
    }


def test_truth_table_has_one_row_per_assignment():
    node = And([Var('A'), Var('B'), Var('C')])
    table = truth_table_from_ast(node)
    assert len(table) == 8
    assert set(table.keys()) == {format(i, 'b').zfill(3) for i in range(8)}


def test_truth_table_variable_order_defaults_to_first_appearance():
    # C appears before A in the expression, so columns should be ordered
    # C, A (MSB first) by default -- not alphabetically.
    node = And([Var('C'), Not(Var('A'))])
    table = truth_table_from_ast(node)
    assert table == {
        '00': 0,  # C=0, A=0
        '01': 0,  # C=0, A=1
        '10': 1,  # C=1, A=0
        '11': 0,  # C=1, A=1
    }


def test_truth_table_accepts_explicit_variable_order():
    node = And([Var('C'), Not(Var('A'))])
    table = truth_table_from_ast(node, variable_order=['A', 'C'])
    assert table == {
        '00': 0,  # A=0, C=0
        '01': 1,  # A=0, C=1
        '10': 0,  # A=1, C=0
        '11': 0,  # A=1, C=1
    }


def test_truth_table_from_parsed_expression_matches_manual_construction():
    parsed = parse_boolean_expression("A*B + C'")
    manual = Or([And([Var('A'), Var('B')]), Not(Var('C'))])
    assert truth_table_from_ast(parsed) == truth_table_from_ast(manual)


# ---------------------------------------------------------------------------
# canonicalize_ast: leaves and structure preservation
# ---------------------------------------------------------------------------

def test_canonicalize_var_is_equal_but_a_new_object():
    node = Var('A')
    result = canonicalize_ast(node)
    assert result == Var('A')


def test_canonicalize_not_recurses():
    assert canonicalize_ast(Not(Var('A'))) == Not(Var('A'))


def test_canonicalize_does_not_mutate_input():
    node = And([Var('B'), Var('A')])
    canonicalize_ast(node)
    assert node == And([Var('B'), Var('A')])  # original order untouched


def test_canonicalize_returns_a_new_object_not_the_same_reference():
    node = And([Var('A'), Var('B')])
    result = canonicalize_ast(node)
    assert result is not node


# ---------------------------------------------------------------------------
# canonicalize_ast: order of children doesn't matter
# ---------------------------------------------------------------------------

def test_canonicalize_and_ignores_operand_order():
    assert canonicalize_ast(And([Var('A'), Var('B')])) == canonicalize_ast(And([Var('B'), Var('A')]))


def test_canonicalize_or_ignores_operand_order():
    assert canonicalize_ast(Or([Var('A'), Var('B'), Var('C')])) == \
        canonicalize_ast(Or([Var('C'), Var('A'), Var('B')]))


def test_canonicalize_xor_ignores_operand_order():
    assert canonicalize_ast(Xor(Var('A'), Var('B'))) == canonicalize_ast(Xor(Var('B'), Var('A')))


def test_canonicalize_ignores_order_at_every_nesting_level():
    # (B+A)*C  vs  C*(A+B)
    left = And([Or([Var('B'), Var('A')]), Var('C')])
    right = And([Var('C'), Or([Var('A'), Var('B')])])
    assert canonicalize_ast(left) == canonicalize_ast(right)


def test_canonicalize_from_parsed_equivalent_expressions():
    a = parse_boolean_expression('A*B + C')
    b = parse_boolean_expression('C + B*A')
    assert canonicalize_ast(a) == canonicalize_ast(b)


# ---------------------------------------------------------------------------
# canonicalize_ast: structural fidelity (no algebraic simplification)
# ---------------------------------------------------------------------------

def test_canonicalize_preserves_operand_multiplicity():
    three = And([Var('A'), Var('A'), Var('B')])
    two = And([Var('A'), Var('B')])
    assert canonicalize_ast(three) != canonicalize_ast(two)


def test_canonicalize_does_not_flatten_nested_gates():
    flat = parse_boolean_expression('A*B*C')
    nested = parse_boolean_expression('(A*B)*C')
    assert canonicalize_ast(flat) != canonicalize_ast(nested)


def test_canonicalize_distinguishes_gate_types():
    assert canonicalize_ast(And([Var('A'), Var('B')])) != canonicalize_ast(Or([Var('A'), Var('B')]))


def test_canonicalize_distinguishes_different_variables():
    assert canonicalize_ast(Var('A')) != canonicalize_ast(Var('B'))


# ---------------------------------------------------------------------------
# structurally_equal
# ---------------------------------------------------------------------------

def test_structurally_equal_identical_expressions():
    assert structurally_equal(Var('A'), Var('A')) is True


def test_structurally_equal_ignores_and_operand_order():
    assert structurally_equal(And([Var('A'), Var('B')]), And([Var('B'), Var('A')])) is True


def test_structurally_equal_ignores_or_operand_order():
    assert structurally_equal(
        Or([Var('A'), Var('B'), Var('C')]),
        Or([Var('C'), Var('A'), Var('B')]),
    ) is True


def test_structurally_equal_ignores_xor_operand_order():
    assert structurally_equal(Xor(Var('A'), Var('B')), Xor(Var('B'), Var('A'))) is True


def test_structurally_equal_ignores_order_at_every_nesting_level():
    # (B+A)*C  vs  C*(A+B)
    left = And([Or([Var('B'), Var('A')]), Var('C')])
    right = And([Var('C'), Or([Var('A'), Var('B')])])
    assert structurally_equal(left, right) is True


def test_structurally_equal_from_parsed_equivalent_expressions():
    a = parse_boolean_expression('A*B + C')
    b = parse_boolean_expression('C + B*A')
    assert structurally_equal(a, b) is True


def test_structurally_equal_rejects_different_variables():
    assert structurally_equal(Var('A'), Var('B')) is False


def test_structurally_equal_rejects_different_gate_types():
    assert structurally_equal(And([Var('A'), Var('B')]), Or([Var('A'), Var('B')])) is False


def test_structurally_equal_rejects_different_multiplicity():
    three = And([Var('A'), Var('A'), Var('B')])
    two = And([Var('A'), Var('B')])
    assert structurally_equal(three, two) is False


def test_structurally_equal_rejects_flattened_vs_nested_gates():
    # A*B*C (one 3-input And) is NOT the same circuit as (A*B)*C (two
    # 2-input Ands), even though they compute the same function.
    flat = parse_boolean_expression('A*B*C')
    nested = parse_boolean_expression('(A*B)*C')
    assert structurally_equal(flat, nested) is False


def test_structurally_equal_does_not_mutate_its_arguments():
    a = And([Var('B'), Var('A')])
    b = And([Var('A'), Var('B')])
    structurally_equal(a, b)
    assert a == And([Var('B'), Var('A')])
    assert b == And([Var('A'), Var('B')])
