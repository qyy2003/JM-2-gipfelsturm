#!/bin/bash
# code_store.sh
#
# Save the user's WIP edits in Megatron-LM/ as a patch.
#
#   no args:    overwrite (or create) 1000-cache.patch with current WIP.
#               The cache contains "everything the user changed beyond the
#               other (non-cache) patches" — i.e. the diff against the state
#               produced by all non-cache patches. The fixed slot 1000 sorts
#               after all named patches (0001, 0002, ...) so it is always
#               applied last by launch.sh.
#
#   -N name:    save WIP to cache, then move it to NNNN-name.patch where
#               NNNN is the next free 4-digit slot below 1000 (e.g. 0002 if
#               only 0001 exists). Fails if there is no WIP.
#
# After running, the working tree is left in the same state as before, so you
# can keep editing without re-running code_restore.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SUBMODULE="$ROOT/Megatron-LM"
PATCH_DIR="$ROOT/patches"

NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -N)
            [[ $# -lt 2 ]] && { echo "Error: -N requires a name" >&2; exit 1; }
            NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$SUBMODULE"
mkdir -p "$PATCH_DIR"

# Split patches into cache (one) and non-cache (rest).
shopt -s nullglob
non_cache=()
existing_cache=""
for p in "$PATCH_DIR"/*.patch; do
    if [[ "$(basename "$p")" == *-cache.patch ]]; then
        existing_cache="$p"
    else
        non_cache+=("$p")
    fi
done

# Cache lives at a fixed slot so it always sorts after named patches.
cache_path="$PATCH_DIR/1000-cache.patch"

# Migrate any old-style cache filename (NNNN-cache.patch where NNNN != 1000).
if [[ -n "$existing_cache" && "$existing_cache" != "$cache_path" ]]; then
    mv "$existing_cache" "$cache_path"
    echo "Migrated $(basename "$existing_cache") -> 1000-cache.patch"
fi

# Reverse-apply non-cache patches (in reverse order) so the working tree
# becomes "vanilla + cache + new edits" — that delta IS the new cache.
non_cache_reversed=()
for ((i=${#non_cache[@]}-1; i>=0; i--)); do
    non_cache_reversed+=("${non_cache[i]}")
done

restore_working_tree() {
    [[ ${#non_cache[@]} -eq 0 ]] && return 0
    git apply "${non_cache[@]}" 2>/dev/null || true
}

if [[ ${#non_cache_reversed[@]} -gt 0 ]]; then
    if ! git apply -R --check "${non_cache_reversed[@]}" 2>/dev/null; then
        echo "Error: cannot reverse-apply non-cache patches." >&2
        echo "Working tree state is unexpected. Run ./code_restore.sh first." >&2
        exit 1
    fi
    git apply -R "${non_cache_reversed[@]}"
    trap 'restore_working_tree' EXIT
fi

diff_content=$(git diff)

if [[ -z "$diff_content" ]]; then
    echo "No changes to save."
    if [[ -n "$NAME" ]]; then
        echo "Cannot promote — there is no WIP and no existing cache." >&2
        exit 1
    fi
    exit 0
fi

git diff > "$cache_path"
lines=$(wc -l < "$cache_path")
echo "Saved cache: $(basename "$cache_path") ($lines lines)"

if [[ -n "$NAME" ]]; then
    # Pick next free slot below the cache slot (1000).
    max_num=0
    for p in "${non_cache[@]}"; do
        n=$((10#$(basename "$p" | grep -oE '^[0-9]{4}' || echo 0)))
        (( n < 1000 && n > max_num )) && max_num=$n
    done
    promote_num=$(printf "%04d" $((max_num + 1)))
    new_path="$PATCH_DIR/$promote_num-$NAME.patch"
    if [[ -e "$new_path" ]]; then
        echo "Error: $(basename "$new_path") already exists." >&2
        exit 1
    fi
    mv "$cache_path" "$new_path"
    echo "Promoted to: $(basename "$new_path")"
fi

restore_working_tree
trap - EXIT
