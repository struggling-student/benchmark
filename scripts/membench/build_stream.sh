#!/usr/bin/env bash
# Compile the vendored STREAM benchmark (third_party/stream/stream.c) at several
# array sizes, chosen relative to CRESCO8's 64 GiB/socket HBM capacity:
#
#   small        ~2.0 GiB total  -- quick smoke test
#   below_cliff  ~31.3 GiB total -- well under 64 GiB; should ride entirely in
#                                   the HBM cache on a cache-mode node
#   above_cliff  ~96.1 GiB total -- clearly exceeds 64 GiB; use DDR binding only
#                                   on the flat node (it will not fit in one HBM node)
#
# Total footprint = STREAM_ARRAY_SIZE * 24 bytes (three `double` arrays).
#
#   scripts/membench/build_stream.sh [--out-dir DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STREAM_SRC="${REPO_ROOT}/third_party/stream/stream.c"
OUT_DIR="${SCRIPT_DIR}/build"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ ! -r "${STREAM_SRC}" ]]; then
    printf 'ERROR: missing %s -- see third_party/stream/NOTES.md for provenance.\n' \
        "${STREAM_SRC}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

SIZES=(
    "small:90000000:10"
    "below_cliff:1400000000:10"
    "above_cliff:4300000000:6"
)

CC="${CC:-cc}"
command -v "${CC}" >/dev/null || {
    printf 'ERROR: no C compiler (%s) found. Load one via MODULE_COMMANDS.\n' "${CC}" >&2
    exit 2
}

for entry in "${SIZES[@]}"; do
    IFS=':' read -r name array_size ntimes <<<"${entry}"
    out="${OUT_DIR}/stream_${name}"
    printf 'Building %s (STREAM_ARRAY_SIZE=%s, NTIMES=%s) -> %s\n' \
        "${name}" "${array_size}" "${ntimes}" "${out}"
    # -mcmodel=medium: the below_cliff/above_cliff arrays exceed 2 GiB, which
    # overflows the default small code model's 32-bit .bss relocations
    # ("relocation truncated to fit: R_X86_64_32S against `.bss'").
    "${CC}" -O3 -march=native -mcmodel=medium -fopenmp \
        -DSTREAM_ARRAY_SIZE="${array_size}" -DNTIMES="${ntimes}" \
        -o "${out}" "${STREAM_SRC}"
done

printf 'Done. Binaries in %s\n' "${OUT_DIR}"
