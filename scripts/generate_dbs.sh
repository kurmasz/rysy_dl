#!/usr/bin/env bash
set -euo pipefail

# Locate the database directory relative to this script, so it works from any working directory.
DB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/rysy_dl/db" && pwd)"

# Names end in _nA-B: circuits of A to B nodes (inputs + core gates), where A is the smallest
# --num-inputs plus the smallest --num-gates and B is --max-nodes (see docs/db.md). One
# generate-circuits run covers every DAG size it needs. --if-exists is not given, so the run
# prompts if the database already exists (insert, overwrite, or quit; quitting stops this script).
#
# circuit_templates_n3-8: and-only templates of 3 to 8 nodes.
db="$DB_DIR/circuit_templates_n3-8.sqlite"
rysy_dl generate-circuits --num-inputs 2-7 --num-gates 1-6 --max-nodes 8 --gate-assignment and-only "$db"

# concrete_circuits_n4-8 (the default database; see db/__init__.py): 2-4 inputs and 2-4 core gates, up
# to 8 nodes. Every gate assignment, and every reduced NOT pattern for each assignment, so NOT
# gates are part of the stored circuits (num_gates_total > num_gates_core).
#
# --max-extra-edges 7 leaves out nothing: built without a limit (2026-11-20), the most edges beyond
# two per gate in any passing circuit was 7 (4 inputs, 4 gates, 15 edges). It skips the denser
# graphs, about half of the work, none of which produce a circuit that passes the filters. If the
# filters or ranges change, rebuild once without the limit and check that it is still lossless.
db="$DB_DIR/concrete_circuits_n4-8.sqlite"
rysy_dl generate-circuits --num-inputs 2-4 --num-gates 2-4 --max-nodes 8 --gate-assignment all --not-gates all \
    --output-connections any --reject-redundant-inputs --max-extra-edges 7 "$db"

# circuit_templates_n3-9: and-only templates of 3 to 9 nodes.
db="$DB_DIR/circuit_templates_n3-9.sqlite"
rysy_dl generate-circuits --num-inputs 2-7 --num-gates 1-7 --max-nodes 9 --gate-assignment and-only "$db"
