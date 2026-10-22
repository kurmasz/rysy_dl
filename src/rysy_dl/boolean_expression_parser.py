"""
boolean_expression_parser.py

Parses boolean expressions into a tree structure that can be evaluated,
analyzed, and used to unambiguously generate a circuit diagram.

The parser supports the following operators:
- AND: represented by * (A*B*C) or juxtaposition (ABC)
- OR: represented by + (A+B+C)
- XOR: represented by ^ (A^B) --- XOR is limited to two inputs.
- NOT: represented by either a ! prefix (!A or !(ABC)) or a ' postfix
  (A' or (ABC)').

Precedence (highest to lowest): NOT, AND, XOR, OR.

Terminals must be either
- single letters (A, B, C, etc.), or
- a single letter followed by an integer (A1, B2, C3, etc.), or
- a single letter followed by an underscore and an integer (A_1, B_2, C_3, etc.)

Mixing * and juxtaposition is allowed, e.g. A*B*C or AB*C or A*BC or ABC. These will
all produce a single And node with three children (A, B, C). If you want two distinct 
2-input AND gates, use parentheses.

AST design notes
-----------------
And/Or nodes take a *list* of children rather than being strictly binary.
This directly represents unlimited fan-in gates: `A*B*C` is one And node
with three children, i.e. one 3-input AND gate -- not two nested 2-input
gates. Explicit parenthesization is preserved rather than flattened, because
it changes the circuit: `(A*B)*C` produces And(And(A, B), C) -- two distinct
2-input AND gates, with the first gate's output feeding the second -- while
`A*B*C` produces a single And(A, B, C), one 3-input AND gate. The two
expressions are logically equivalent but describe different circuits.

Xor is fixed at exactly two operands (`left`, `right`) because the spec
limits XOR to two inputs. `A^B^C` is a parse error; chain explicitly with
parentheses, e.g. `(A^B)^C`, which produces two distinct 2-input XOR gates.

Repeated variable names produce separate Var nodes (by design) -- that's
signal fan-out to multiple gate inputs, not a single shared input. A
diagram/netlist generator should group Var nodes by `.name` to find the
common source, and should key gate nodes by `id(node)`, not by value
equality, since two structurally-identical subtrees (e.g. from
`(A*B) + (A*B)'`) are still two distinct physical gates.


First pass written by Claude, Sep. 2026
"""

from dataclasses import dataclass
from typing import List
import re


class BooleanExpressionError(Exception):
    """Raised when a boolean expression cannot be parsed."""

    def __init__(self, message: str, position: int = None):
        self.position = position
        if position is not None:
            message = f"{message} (position {position})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class Node:
    pass


@dataclass
class Var(Node):
    name: str


@dataclass
class Not(Node):
    child: Node


@dataclass
class And(Node):
    children: List[Node]


@dataclass
class Or(Node):
    children: List[Node]


@dataclass
class Xor(Node):
    left: Node
    right: Node


def _make_and(children: List[Node]) -> Node:
    """Build an And node from operands gathered at one precedence level.

    Does NOT flatten nested And children -- a nested And came from an
    explicit parenthesization and represents a separate gate.
    """
    return children[0] if len(children) == 1 else And(children)


def _make_or(children: List[Node]) -> Node:
    """Build an Or node from operands gathered at one precedence level.

    Does NOT flatten nested Or children -- a nested Or came from an
    explicit parenthesization and represents a separate gate.
    """
    return children[0] if len(children) == 1 else Or(children)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("IDENT", r"[A-Za-z](?:_?[0-9]+)?"),
    ("AND", r"\*"),
    ("OR", r"\+"),
    ("XOR", r"\^"),
    ("NOT", r"!"),
    ("PRIME", r"'"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP", r"[ \t\r\n]+"),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC)
)


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def _tokenize(expression: str) -> List[Token]:
    tokens = []
    for m in _TOKEN_RE.finditer(expression):
        kind = m.lastgroup
        assert kind is not None # not possible with the regex above
        value = m.group()
        pos = m.start()
        if kind == "SKIP":
            continue
        if kind == "MISMATCH":
            raise BooleanExpressionError(f"Unexpected character {value!r}", pos)
        tokens.append(Token(kind, value, pos))
    tokens.append(Token("END", "", len(expression)))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
#
#   or_expr   := xor_expr ('+' xor_expr)*
#   xor_expr  := and_expr ('^' and_expr)?      # binary only
#   and_expr  := unary ( ('*' unary) | unary )*  # '*' or juxtaposition
#   unary     := '!' unary | primary "'"*
#   primary   := IDENT | '(' or_expr ')'
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._i = 0

    def _peek(self) -> Token:
        return self._tokens[self._i]

    def _advance(self) -> Token:
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise BooleanExpressionError(
                f"Expected {kind!r} but found {tok.value!r}", tok.pos
            )
        return self._advance()

    def parse(self) -> Node:
        if self._peek().kind == "END":
            raise BooleanExpressionError("Empty expression", 0)
        node = self._parse_or()
        if self._peek().kind != "END":
            tok = self._peek()
            raise BooleanExpressionError(f"Unexpected token {tok.value!r}", tok.pos)
        return node

    def _parse_or(self) -> Node:
        children = [self._parse_xor()]
        while self._peek().kind == "OR":
            self._advance()
            children.append(self._parse_xor())
        return _make_or(children)

    def _parse_xor(self) -> Node:
        left = self._parse_and()
        if self._peek().kind == "XOR":
            self._advance()
            right = self._parse_and()
            node = Xor(left, right)
            if self._peek().kind == "XOR":
                tok = self._peek()
                raise BooleanExpressionError(
                    "XOR only supports two operands; use parentheses to "
                    "chain, e.g. (A^B)^C instead of A^B^C",
                    tok.pos,
                )
            return node
        return left

    def _parse_and(self) -> Node:
        children = [self._parse_unary()]
        while True:
            if self._peek().kind == "AND":
                self._advance()
                children.append(self._parse_unary())
            elif self._peek().kind in ("IDENT", "NOT", "LPAREN"):
                children.append(self._parse_unary())  # juxtaposition
            else:
                break
        return _make_and(children)

    def _parse_unary(self) -> Node:
        if self._peek().kind == "NOT":
            self._advance()
            return Not(self._parse_unary())
        node = self._parse_primary()
        while self._peek().kind == "PRIME":
            self._advance()
            node = Not(node)
        return node

    def _parse_primary(self) -> Node:
        tok = self._peek()
        if tok.kind == "IDENT":
            self._advance()
            return Var(tok.value)
        if tok.kind == "LPAREN":
            self._advance()
            node = self._parse_or()
            self._expect("RPAREN")
            return node
        raise BooleanExpressionError(
            f"Expected a variable or '(' but found {tok.value!r}", tok.pos
        )


def parse_boolean_expression(expression: str) -> Node:
    """
    Parses a boolean expression into a tree structure.

    Args:
        expression (str): The boolean expression to parse.

    Returns:
        Node: The root of the parsed AST (Var, Not, And, Or, or Xor).

    Raises:
        BooleanExpressionError: If the expression is malformed.
    """
    tokens = _tokenize(expression)
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Student-facing syntax reference
#
# HTML describing the syntax above, meant to be dropped into a PrairieLearn
# question's params (e.g. data["params"]["syntax_help"]) and rendered with a
# triple-mustache tag ({{{params.syntax_help}}}) inside a collapsible
# <details> block so students can show/hide it as desired. Kept in this file
# (rather than a separate module) so a change to the grammar above and the
# text describing it are easy to keep in sync.
# ---------------------------------------------------------------------------

SYNTAX_HELP_HTML = """
<table class="table table-sm table-bordered mb-3">
  <thead>
    <tr><th>Operation</th><th>Symbol</th><th>Example</th></tr>
  </thead>
  <tbody>
    <tr><td>NOT</td><td><code>!</code> (prefix) or <code>'</code> (postfix)</td><td><code>!A</code>, <code>A'</code></td><td><code>!(AB)</code>, <code>(AB)'</code></td></tr>
    <tr><td>AND</td><td><code>*</code>, or just writing variables next to each other</td><td><code>A*B</code>, <code>AB</code></td><td><code>(A + B)*C</code>, <code>(A + B)C</code></td></tr>
    <tr><td>OR</td><td><code>+</code></td><td><code>A + B</code></td><td><code>AB + C</code>, <code>(AB) + C</code></td></tr>
    <tr><td>XOR</td><td><code>^</code> (exactly two operands)</td><td><code>A ^ B</code></td><td><code>(A^B)^C</code></td></tr>
  </tbody>
</table>
<p class="mb-3">
  Parentheses can determine the structure of the circuit described. 
  For example,  <code>A*B*C</code> is a single 3-input AND gate, 
  while <code>(A*B)*C</code> is two 2-input AND
  gates chained together -- these compute the same result but are different
  circuits.
  </p>
<p class="mb-3">
  By definition, XOR (<code>^</code>) only takes two operands, so a
  chain of XORs must be parenthesized explicitly, e.g. <code>(A^B)^C</code>
  rather than <code>A^B^C</code>.
</p>
""".strip()

HIDABLE_SYNTAX_HELP_HTML = f"""
  <details class="mb-2">
    <summary class="small text-muted">Boolean Expression Syntax</summary>
    {SYNTAX_HELP_HTML}
  </details>
"""