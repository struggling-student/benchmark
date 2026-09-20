#!/usr/bin/env bash
# Fetch Intel Memory Latency Checker (MLC). MLC is proprietary -- Intel's own
# license forbids redistributing it, so it is never vendored in this repo.
# This script downloads it directly from Intel at run time instead.
#
# The URL below was confirmed working by hand (curl + `tar tzf`) on 2026-09-20,
# pulled from the "Linux" download link on Intel's own MLC download page
# (https://www.intel.com/content/www/us/en/download/736633/). Intel does not
# publish a stable/versionless URL, and download-portal links do change across
# releases -- if this 404s, open that page, find the current "Linux" link
# (right-click -> copy link), and pass it via --url, e.g.:
#
#   scripts/membench/build_mlc.sh --url https://downloadmirror.intel.com/<id>/mlc_v<ver>.tgz
#
# If the node has no outbound internet, download the tarball on a machine that
# does and place it manually as ${MLC_DIR}/mlc (or point run_mlc.sh at it via
# --bin), then skip this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_URL="https://downloadmirror.intel.com/926327/mlc_v3.13.tgz"
URL="${DEFAULT_URL}"
MLC_DIR="${SCRIPT_DIR}/vendor/mlc"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --dest) MLC_DIR="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

command -v curl >/dev/null || { printf 'ERROR: curl is required.\n' >&2; exit 2; }
command -v tar >/dev/null || { printf 'ERROR: tar is required.\n' >&2; exit 2; }

mkdir -p "${MLC_DIR}"
ARCHIVE="$(mktemp -t mlc-download.XXXXXX.tgz)"
trap 'rm -f "${ARCHIVE}"' EXIT

printf 'Downloading Intel MLC from %s\n' "${URL}"
if ! curl -fsSL --max-time 60 -A "Mozilla/5.0" -o "${ARCHIVE}" "${URL}"; then
    printf 'ERROR: download failed. This node may lack outbound internet, or the\n' >&2
    printf 'URL above may be stale. See this script'"'"'s header comment for how to\n' >&2
    printf 'find the current link or supply a manually transferred tarball.\n' >&2
    exit 1
fi

if ! tar tzf "${ARCHIVE}" >/dev/null 2>&1; then
    printf 'ERROR: downloaded file is not a valid gzip tarball (likely an HTML\n' >&2
    printf 'error/EULA page rather than the archive itself). Inspect it at %s\n' \
        "${ARCHIVE}" >&2
    exit 1
fi

tar xzf "${ARCHIVE}" -C "${MLC_DIR}"

BIN="$(find "${MLC_DIR}" -type f -name mlc -path '*Linux*' | head -n1)"
if [[ -z "${BIN}" ]]; then
    printf 'ERROR: extracted archive did not contain Linux/mlc.\n' >&2
    exit 1
fi
chmod +x "${BIN}"
printf 'Installed MLC binary at %s\n' "${BIN}"
printf 'Point run_mlc.sh at it with --bin %s, or export MLC_BIN=%s\n' "${BIN}" "${BIN}"
