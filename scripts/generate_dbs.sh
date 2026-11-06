#!/usr/bin/env bash
set -euo pipefail

# Locate the database directory relative to this script, so it works from any working directory.
DB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/rysy_dl/db" && pwd)"

# and-only, dag sizes 4 - 7
# The first run prompts if the database already exists (insert, overwrite, or quit; quitting
# stops this script). The remaining runs add to the database the first run created or kept.

db="$DB_DIR/rysy_and_only_4-7.sqlite"
rysy_dl generate-circuits --dag-size 4 --gate-assignment and-only "$db"
for i in 5 6 7; do
	rysy_dl generate-circuits --dag-size $i --gate-assignment and-only --if-exists insert "$db"
done


db="$DB_DIR/rysy_and_only_4-8.sqlite"
rysy_dl generate-circuits --dag-size 4 --gate-assignment and-only "$db"
for i in 5 6 7 8; do
	rysy_dl generate-circuits --dag-size $i --gate-assignment and-only --if-exists insert "$db"
done