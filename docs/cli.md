# Rysy_dl Command Line Interface

## `generate-circuits`

This feature generates a library of 
### Edge limits: `--max-edges` and `--max-extra-edges`

An edge is a wire from an input or gate into a gate, including the output gate's inputs (NOT gates
are not counted). Every gate has at least two inputs, so a circuit with g gates has at least 2g
edges.

* `--max-edges N` leaves out circuits with more than N edges.
* `--max-extra-edges N` leaves out circuits with more than N edges beyond two per gate: 0 allows
  only 2-input gates, 1 allows one 3-input gate (or nothing wider), and so on. Because it is
  relative to the number of gates, one value suits every circuit size in a run.

Both can be given; a circuit must satisfy both. They are checked on each circuit's structure
before any gates are assigned, and `geng` is told the most edges a DAG can have and still pass, so
leaving circuits out this way costs almost nothing.

Why: almost all generation time goes to assigning gates and NOTs to graphs with many edges, and
very few of those circuits pass `--reject-redundant-inputs`. For 4 inputs and 4 gates, circuits
with at most 12 edges (`--max-extra-edges 4`) are 92% of what passes but about 7.5% of the work.
Passing circuits do not thin out steadily as edges increase, though (4 inputs and 4 gates: 928
with 13 edges, 576 with 14, 2,496 with 15), so a limit can leave out circuits a larger limit
would find.
