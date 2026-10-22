import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl.boolean_expression_parser import (
    And,
    BooleanExpressionError,
    Not,
    Or,
    Var,
    Xor,
    parse_boolean_expression,
)


# ---------------------------------------------------------------------------
# Terminals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['A', 'z', 'A1', 'B23', 'A_1', 'B_23'])
def test_single_variable(name):
    assert parse_boolean_expression(name) == Var(name)


def test_variable_letter_and_digits_without_underscore():
    assert parse_boolean_expression('A12') == Var('A12')


# ---------------------------------------------------------------------------
# AND: '*' and juxtaposition, without parentheses (single gate, no flattening
# needed because no nested And node is ever created)
# ---------------------------------------------------------------------------

def test_and_with_star_two_operands():
    assert parse_boolean_expression('A*B') == And([Var('A'), Var('B')])


def test_and_with_star_three_operands_is_one_gate():
    node = parse_boolean_expression('A*B*C')
    assert node == And([Var('A'), Var('B'), Var('C')])
    assert not any(isinstance(c, And) for c in node.children)


def test_and_with_juxtaposition():
    assert parse_boolean_expression('ABC') == And([Var('A'), Var('B'), Var('C')])


def test_and_juxtaposition_and_star_mixed():
    assert parse_boolean_expression('AB*C') == And([Var('A'), Var('B'), Var('C')])


# ---------------------------------------------------------------------------
# OR: '+'
# ---------------------------------------------------------------------------

def test_or_two_operands():
    assert parse_boolean_expression('A+B') == Or([Var('A'), Var('B')])


def test_or_three_operands_is_one_gate():
    node = parse_boolean_expression('A+B+C')
    assert node == Or([Var('A'), Var('B'), Var('C')])
    assert not any(isinstance(c, Or) for c in node.children)


# ---------------------------------------------------------------------------
# XOR: binary only
# ---------------------------------------------------------------------------

def test_xor_two_operands():
    assert parse_boolean_expression('A^B') == Xor(Var('A'), Var('B'))


def test_xor_chained_without_parens_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('A^B^C')


def test_xor_chained_error_position_is_second_caret():
    with pytest.raises(BooleanExpressionError) as exc_info:
        parse_boolean_expression('A^B^C')
    assert exc_info.value.position == 3


def test_xor_chained_with_parens_produces_two_gates():
    node = parse_boolean_expression('(A^B)^C')
    assert node == Xor(Xor(Var('A'), Var('B')), Var('C'))


# ---------------------------------------------------------------------------
# NOT: '!' prefix and "'" postfix
# ---------------------------------------------------------------------------

def test_not_prefix():
    assert parse_boolean_expression('!A') == Not(Var('A'))


def test_not_postfix():
    assert parse_boolean_expression("A'") == Not(Var('A'))


def test_not_postfix_stacks():
    assert parse_boolean_expression("A''") == Not(Not(Var('A')))


def test_not_prefix_stacks():
    assert parse_boolean_expression('!!A') == Not(Not(Var('A')))


def test_not_prefix_on_parenthesized_group():
    assert parse_boolean_expression('!(A*B)') == Not(And([Var('A'), Var('B')]))


def test_not_postfix_on_parenthesized_group():
    assert parse_boolean_expression("(A*B)'") == Not(And([Var('A'), Var('B')]))


# ---------------------------------------------------------------------------
# Precedence: NOT > AND > XOR > OR
# ---------------------------------------------------------------------------

def test_and_binds_tighter_than_or():
    assert parse_boolean_expression('A+B*C') == Or([Var('A'), And([Var('B'), Var('C')])])


def test_not_binds_tighter_than_and():
    assert parse_boolean_expression('!A*B') == And([Not(Var('A')), Var('B')])


def test_xor_binds_tighter_than_or_but_looser_than_and():
    assert parse_boolean_expression('A+B^C*D') == Or(
        [Var('A'), Xor(Var('B'), And([Var('C'), Var('D')]))]
    )


# ---------------------------------------------------------------------------
# Parentheses: explicit grouping is preserved, not flattened, even when the
# operator is the same as the enclosing expression -- this is the core
# correctness property described in the module's AST design notes.
# ---------------------------------------------------------------------------

def test_parens_same_operator_are_not_flattened_and():
    node = parse_boolean_expression('(A*B)*C')
    assert node == And([And([Var('A'), Var('B')]), Var('C')])


def test_mixed_and_juxtaposition_remains_flat_left():
    node = parse_boolean_expression('AB*C')
    assert node == And([Var('A'), Var('B'), Var('C')])

def test_mixed_and_juxtaposition_remains_flat_right():
    node = parse_boolean_expression('A*BC')
    assert node == And([Var('A'), Var('B'), Var('C')])

def test_parens_same_operator_are_not_flattened_and_different_grouping():
    node = parse_boolean_expression('A*(B*C)')
    assert node == And([Var('A'), And([Var('B'), Var('C')])])


def test_parens_same_operator_are_not_flattened_or():
    node = parse_boolean_expression('(A+B)+C')
    assert node == Or([Or([Var('A'), Var('B')]), Var('C')])


def test_flattened_and_grouped_versions_are_not_equal():
    flat = parse_boolean_expression('A*B*C')
    grouped = parse_boolean_expression('(A*B)*C')
    assert flat != grouped


def test_redundant_parens_around_single_variable_are_transparent():
    assert parse_boolean_expression('(A)') == Var('A')


def test_nested_redundant_parens():
    assert parse_boolean_expression('((A*B))') == And([Var('A'), Var('B')])


# ---------------------------------------------------------------------------
# Repeated variables: separate Var nodes (fan-out), not a shared object
# ---------------------------------------------------------------------------

def test_repeated_variable_produces_equal_but_distinct_nodes():
    node = parse_boolean_expression('A*A')
    assert node == And([Var('A'), Var('A')])
    assert node.children[0] is not node.children[1]


# ---------------------------------------------------------------------------
# Whitespace is ignored
# ---------------------------------------------------------------------------

def test_whitespace_is_ignored():
    assert parse_boolean_expression(' A * B + C ') == parse_boolean_expression('A*B+C')


def test_extra_whitespace_is_ignored():
    assert parse_boolean_expression(' A *     B + \t\n   C ') == parse_boolean_expression('A*B+C')

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_empty_expression_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('')


def test_whitespace_only_expression_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('   ')


def test_unexpected_character_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('A & B')


def test_unclosed_paren_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('(A*B')


def test_unmatched_close_paren_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('A*B)')


def test_trailing_operator_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('A*')


def test_leading_operator_is_an_error():
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression('*A')


@pytest.mark.parametrize('expr', ['A_', '_1', '2', 'A__1'])
def test_malformed_terminal_is_an_error(expr):
    with pytest.raises(BooleanExpressionError):
        parse_boolean_expression(expr)


def test_juxtaposed_letters_are_two_separate_terminals_not_an_error():
    # 'A' immediately followed by 'B' tokenizes as two IDENTs ('A', 'B')
    # combined by juxtaposition, not a single invalid multi-letter terminal.
    assert parse_boolean_expression('AB') == And([Var('A'), Var('B')])


def test_error_message_includes_position():
    with pytest.raises(BooleanExpressionError) as exc_info:
        parse_boolean_expression('A & B')
    assert 'position 2' in str(exc_info.value)
    assert exc_info.value.position == 2


def test_boolean_expression_error_without_position_omits_position_text():
    err = BooleanExpressionError('something went wrong')
    assert err.position is None
    assert 'position' not in str(err)
