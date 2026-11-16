#!/usr/bin/env bash
#
# Copy the stable rysy_dl package source into the PrairieLearn course repo's
# serverFilesCourse/rysy_dl. That directory must stay a real, plain directory
# (not a symlink) so it works in production, so this script periodically
# overwrites it with a fresh copy of the working package.
#
# This copies the working tree of this checkout, so it refuses to run unless
# main is checked out (other branches may change the database schema).
#
# Usage:
#   scripts/copy_rysy_dl_to_pl.sh               # perform the copy
#   scripts/copy_rysy_dl_to_pl.sh --dry-run     # show what would change, copy nothing
#   scripts/copy_rysy_dl_to_pl.sh --any-branch  # allow copying from a branch other than main

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/src/rysy_dl/"
DEST="/Users/kurmasz/Courses/CIS351/pl-gvsu-cis351/serverFilesCourse/rysy_dl/"

RSYNC_OPTS=(-av --delete
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude 'temp_out/'
    # The course's db/.gitignore keeps large databases out of the course repo; leave it alone.
    --exclude '.gitignore'
    # Ship only the default database. Excluded files are also left untouched in DEST.
    --include 'db/rysy_dag_only_4-7.sqlite'
    --exclude 'db/*.sqlite'
)

any_branch=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) RSYNC_OPTS+=(--dry-run) ;;
        --any-branch) any_branch=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -d "$SRC" ]; then
    echo "Error: source directory not found: $SRC" >&2
    exit 1
fi

if [ ! -d "$DEST" ]; then
    echo "Error: destination directory not found: $DEST" >&2
    exit 1
fi

branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" && "$any_branch" == false ]]; then
    echo "Error: $REPO_ROOT is on branch '$branch', not main. Use --any-branch to copy it anyway." >&2
    exit 1
fi

uncommitted="$(git -C "$REPO_ROOT" status --porcelain -- src/rysy_dl)"
if [ -n "$uncommitted" ]; then
    echo "Warning: copying uncommitted changes:"
    echo "$uncommitted"
    echo
fi

echo "Syncing (branch $branch):"
echo "  $SRC"
echo "  -> $DEST"
echo

rsync "${RSYNC_OPTS[@]}" "$SRC" "$DEST"

echo
echo "Done. DEST is a git repo -- review with 'git status' / 'git diff' before committing."
