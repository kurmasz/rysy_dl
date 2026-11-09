#!/usr/bin/env bash
set -euo pipefail

# Locate the database directory relative to this script, so it works from any working directory.
DB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/rysy_dl/db" && pwd)"

# and-only databases. Each circuit is inputs + gates nodes; one generate-circuits run covers
# every DAG size it needs. --if-exists is not given, so the run prompts if the database already
# exists (insert, overwrite, or quit; quitting stops this script).
#
# rysy_and_only_4-7: up to 8 nodes (the circuits built from DAGs of 4-7 nodes, plus the few
# smaller ones with 3-4 nodes).
db="$DB_DIR/rysy_and_only_4-7.sqlite"
rysy_dl generate-circuits --num-inputs 2-7 --num-gates 1-6 --max-nodes 8 --gate-assignment and-only "$db"

# rysy_and_only_4-8: up to 9 nodes.
db="$DB_DIR/rysy_and_only_4-8.sqlite"
rysy_dl generate-circuits --num-inputs 2-7 --num-gates 1-7 --max-nodes 9 --gate-assignment and-only "$db"
