# Databases

`scripts/generate_dbs.sh` generates several databases. In general, databases (specifically, `.sqlite` files) are excluded from the git repo 
because of their size; but, there are a few exceptions.


`rysy_and_only_4-7.sqlite` (2.1MB): This is `rysy_dl`'s default database (the one used unless you specify something different). 
 It contains one example circuit for each dag containing 4 to 7 nodes (inclusive). These example circuits contain only `AND` gates.
 A typical workflow is for a PrairieLearn question's `server.py` to do the following:
    * Choose one example graph at random from the database (possibly with constraints)
    * Randomly assign gates to nodes
    * Randomly assign NOT gates to edges connected to sources.


`rysy_and_only_4-8.sqlite` (144MB): Similar to `rysy_and_only_4-8.sqlite` except it also contains circuits built from 8-node DAGs.


## Gate counts

Each circuit stores two gate counts:

* `num_gates_core`: every gate except NOT gates. This is what `generate-circuits --num-gates` counts.
* `num_gates_total`: every gate, including NOT gates.

The number of NOT gates is `num_gates_total - num_gates_core`. Both counts can be used as filters,
e.g. `get_random_circuit(num_inputs=3, num_gates_core=(3, 4))` or `view-circuits --num-gates-total 5`.

