# Databases

`scripts/generate_dbs.sh` generates several databases. In general, databases (specifically, `.sqlite` files) are excluded from the git repo 
because of their size; but, there are a few exceptions.

Names end in `_nA-B`: the database holds circuits of A to B nodes, where a circuit's nodes are its inputs plus its
core gates (not NOT gates). A is the smallest `--num-inputs` plus the smallest `--num-gates` passed to
`generate-circuits`, and B is its `--max-nodes`. (A size in that range can still be empty if no circuit of that
size passes the filters.)


`concrete_circuits_n4-8.sqlite` (34MB): This is `rysy_dl`'s default database (the one used unless you specify something different).
 It contains concrete (fully specified) circuits of 4 to 8 nodes (2-4 inputs, 2-4 core gates): every gate
 assignment and every reduced NOT pattern for each DAG, so a question can use a circuit exactly as stored, with no
 gates or NOTs assigned afterward.


`circuit_templates_n3-8.sqlite` (2.1MB): Circuit templates. It contains one example circuit for each DAG shape with 3 to 8 nodes (2-7 inputs, 1-6 core gates). These example circuits contain only `AND` gates.
 A typical workflow is for a PrairieLearn question's `server.py` to do the following:
    * Choose one example graph at random from the database (possibly with constraints)
    * Randomly assign gates to nodes
    * Randomly assign NOT gates to edges connected to sources.


`circuit_templates_n3-9.sqlite` (144MB): Similar to `circuit_templates_n3-8.sqlite` except it also contains 9-node circuits.


## Gate counts

Each circuit stores two gate counts:

* `num_gates_core`: every gate except NOT gates. This is what `generate-circuits --num-gates` counts.
* `num_gates_total`: every gate, including NOT gates.

The number of NOT gates is `num_gates_total - num_gates_core`. Both counts can be used as filters,
e.g. `get_random_circuit(num_inputs=3, num_gates_core=(3, 4))` or `view-circuits --num-gates-total 5`.

