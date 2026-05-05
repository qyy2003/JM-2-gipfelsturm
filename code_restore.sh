#!/bin/bash
# code_restore.sh
#
# Verify Megatron-LM is clean, then apply every patch in patches/.
# This puts the submodule in the same state launch.sh would: vanilla pin
# + all patches applied to the working tree.
#
# Usage:
#   ./code_restore.sh           # refuse if dirty
#   ./code_restore.sh -f        # discard dirty changes, then apply
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SUBMODULE="$ROOT/Megatron-LM"
PATCH_DIR="$ROOT/patches"

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force) FORCE=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$SUBMODULE"

if [[ -n "$(git status --porcelain)" ]]; then
    if [[ $FORCE -eq 1 ]]; then
        echo "Force: discarding uncommitted changes."
        git checkout -- .
    else
        echo "Megatron-LM has uncommitted changes:"
        git status --short
        echo
        echo "Save first:  ./code_store.sh [-N name]"
        echo "Or force:    ./code_restore.sh -f"
        exit 1
    fi
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
    echo "No patches in $PATCH_DIR — nothing to apply."
    exit 0
fi

echo "Checking patches..."
if ! git apply --check "${patches[@]}"; then
    echo "Patches don't apply cleanly. Aborting." >&2
    exit 1
fi

echo "Applying ${#patches[@]} patch(es):"
for p in "${patches[@]}"; do echo "  $(basename "$p")"; done
git apply "${patches[@]}"
echo "Done."
