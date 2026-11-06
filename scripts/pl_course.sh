#!/usr/bin/env bash
#
# Install rysy_dl into a PrairieLearn course, or run a local PrairieLearn against a local
# rysy_dl checkout. See the "PrairieLearn" section of README.md.
#
# Requires Docker. Commands run inside the PrairieLearn image so installs match the
# environment questions run in. Set PL_IMAGE to use a different image.

set -euo pipefail

IMAGE="${PL_IMAGE:-prairielearn/prairielearn}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<USAGE
Usage:
  $(basename "$0") install <course-dir> [<source>]
      Install rysy_dl into <course-dir>/serverFilesCourse, replacing any previous copy.
      <source> may be:
        - a PyPI requirement (default: rysy_dl), e.g. rysy_dl==0.2.0
        - a path to a rysy_dl checkout (must contain a '/', e.g. . or ../rysy_dl)
        - a git URL, e.g. git+https://github.com/<you>/rysy_dl.git@<tag>

  $(basename "$0") dev <course-dir> [<rysy_dl-checkout>]
      Run PrairieLearn at http://localhost:3000 for <course-dir>, importing rysy_dl directly
      from <rysy_dl-checkout>/src/rysy_dl (default: the checkout containing this script).
USAGE
  exit 1
}

abspath() { (cd "$1" && pwd); }

[[ $# -ge 2 ]] || usage
cmd="$1"
[[ -d "$2" ]] || { echo "Course directory not found: $2" >&2; exit 1; }
course="$(abspath "$2")"
shift 2

case "$cmd" in
  install)
    [[ $# -le 1 ]] || usage
    source="${1:-rysy_dl}"
    target="$course/serverFilesCourse"
    mounts=(-v "$course:/course")

    if [[ ( "$source" == */* || "$source" == "." ) && "$source" != *://* ]]; then
      [[ -d "$source" ]] || { echo "Directory not found: $source" >&2; exit 1; }
      [[ -f "$source/pyproject.toml" ]] || { echo "No pyproject.toml in $source" >&2; exit 1; }
      mounts+=(-v "$(abspath "$source"):/src")
      source=/src
    fi

    # Install into a staging directory, then swap it in, so a failed install leaves any existing
    # copy untouched. (uv's --target also wouldn't delete files left by a previous version.)
    staging="$course/.rysy_dl_install"
    rm -rf "$staging"
    mkdir -p "$staging" "$target"
    trap 'rm -rf "$staging"' EXIT

    # --no-deps: PrairieLearn's image already provides rysy_dl's runtime dependencies.
    docker run --rm "${mounts[@]}" "$IMAGE" \
      uv pip install --target /course/.rysy_dl_install --no-deps "$source"

    # Move in only the package and its metadata (not uv's lock file or the command-line launcher).
    rm -rf "$target/rysy_dl" "$target"/rysy_dl-*.dist-info
    mv "$staging/rysy_dl" "$staging"/rysy_dl-*.dist-info "$target/"

    echo "Installed rysy_dl into $target. Commit serverFilesCourse/rysy_dl and serverFilesCourse/rysy_dl-*.dist-info."
    ;;

  dev)
    [[ $# -le 1 ]] || usage
    checkout="${1:-$REPO_ROOT}"
    [[ -d "$checkout" ]] || { echo "Checkout not found: $checkout" >&2; exit 1; }
    pkg="$(abspath "$checkout")/src/rysy_dl"
    [[ -f "$pkg/__init__.py" ]] || { echo "No rysy_dl package at $pkg" >&2; exit 1; }

    if [[ ! -d "$course/serverFilesCourse/rysy_dl" ]]; then
      echo "Note: Docker will create an empty serverFilesCourse/rysy_dl in the course as a mount point. Don't commit it." >&2
    fi

    # The second mount hides any installed copy of rysy_dl in the course without modifying it.
    docker run -it --rm -p 3000:3000 \
      -v "$course:/course" \
      -v "$pkg:/course/serverFilesCourse/rysy_dl" \
      "$IMAGE"
    ;;

  *)
    usage
    ;;
esac
