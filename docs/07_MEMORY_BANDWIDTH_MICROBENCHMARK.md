# 07 — Memory-bandwidth microbenchmark (STREAM + Intel MLC)

This is a standalone hardware probe, not part of the LLM inference harness. It exists
to answer one question before investing in a vLLM-side HBM/DDR placement split: does
CRESCO8's Xeon Max 9480 flat mode actually let HBM and DDR be driven *concurrently* at
combined bandwidth above either tier alone, and how much faster is HBM than DDR here in
the first place. Neither number existed anywhere in this repo before this tool —
`docs/05_CPU_HBM_ROADMAP.md` explicitly flagged that gap.

## Read this first: what cache mode can and cannot show you

Confirmed CRESCO8 topology (from live `numa_nodes()` output, see `hardware.py`):

| Node | Cores | Capacity | Present in |
| --- | --- | --- | --- |
| 0 | socket 0 | ~504 GiB DDR5 | cache mode, flat mode |
| 1 | socket 1 | ~504 GiB DDR5 | cache mode, flat mode |
| 2 | none (memory-only) | 64 GiB HBM2e | flat mode only |
| 3 | none (memory-only) | 64 GiB HBM2e | flat mode only |

**On a cache-mode node, nodes 2/3 do not exist.** HBM is a transparent, hardware-managed
cache in front of DDR there — there is no NUMA target for it and no way to bind to "DDR
without the HBM cache." The cache-mode run in this tool measures OS-visible bandwidth to
whichever DDR node you bind to, and the *array-size sweep* is how the HBM cache's effect
still shows up indirectly: a working set well under 64 GiB should ride almost entirely in
the transparent cache (fast), while one well above 64 GiB cannot fit and degrades toward
plain DDR. That contrast — not a single number — is the cache-mode result.

On the flat-mode node (`cresco8-hbm14`), nodes 2/3 are real, explicit NUMA targets, so
you can measure HBM and DDR separately, and — the reason this tool exists — drive both
*at the same time* to see whether their bandwidth adds up.

## What's in `scripts/membench/`

| Script | Purpose |
| --- | --- |
| `build_stream.sh` | Compiles the vendored STREAM (`third_party/stream/stream.c`) at three array sizes: `small` (~2 GiB, smoke test), `below_cliff` (~31 GiB, under the 64 GiB HBM budget), `above_cliff` (~96 GiB, clearly over it). |
| `run_stream.sh` | Runs one STREAM binary under `numactl --cpunodebind=N --membind=N`, with the same `OMP_PROC_BIND=close`/`OMP_PLACES=cores` pinning the project already proved necessary for llama.cpp (see `docs/investigations/llamacpp-placement-instability.md`), and writes a result JSON. |
| `run_concurrent_stream.sh` | Launches two `run_stream.sh` instances at once with independent cpu/mem binds — the direct test of "does HBM+DDR concurrently beat either alone." |
| `build_mlc.sh` | Downloads Intel MLC directly from Intel at run time (never vendored — see Licensing below). |
| `run_mlc.sh` | Runs MLC's `--bandwidth_matrix` and `--latency_matrix`, which sweep every NUMA node pair internally (no `numactl` wrapping needed, unlike STREAM). |
| `run_all.sh` | Single entry point: full STREAM sweep (+ concurrent test on flat) + MLC matrices, for one `--target cache\|flat`. |
| `submit_cresco8_membench.sh` | Cluster driver: `sbatch` to a cache-mode node, or direct SSH to the flat node (same staging mechanism as `scripts/run_cresco8_bf16_flat_campaign.sh`). |

Parsing and result-writing live in `src/llm_bench/membench.py` (tested in
`tests/test_membench.py`), reusing `hardware.py`'s NUMA/HBM detection and
`metadata.py`'s run-naming helpers. It deliberately does **not** reuse `results.py`'s
LLM-run summary schema — that schema is built around model/backend/token fields a raw
memory probe doesn't have.

## Running it

```bash
# One-time, on the cluster (or wherever it will run):
scripts/membench/build_stream.sh
scripts/membench/build_mlc.sh   # see Licensing below if this fails

# Cache-mode node (Slurm) and the flat-mode node (hbm14), from the login node:
scripts/membench/submit_cresco8_membench.sh --target cache
scripts/membench/submit_cresco8_membench.sh --target flat --mlc-bin scripts/membench/vendor/mlc/Linux/mlc
```

Results land as one JSON file per run under `$RESULTS_ROOT/membench-results/`, each with
`tool`, `target`, `cpu_bind`/`mem_bind`, `results` (STREAM's Copy/Scale/Add/Triad
MB/s, or MLC's per-node-pair matrix), a `topology` snapshot, and a `warnings` list —
warnings are surfaced explicitly rather than silently dropped (same convention as
`docs/04_METRICS_AND_RESULTS.md`).

## Licensing

- **STREAM** (`third_party/stream/stream.c`) is vendored verbatim with its original
  license header. See `third_party/stream/NOTES.md` for provenance. These runs are
  `numactl`-bound NUMA microbenchmarks, not STREAM Run Rules submissions — don't call
  results "STREAM benchmark results" without following those rules.
- **Intel MLC** is proprietary and cannot be redistributed, so it is never committed to
  this repo. `build_mlc.sh` downloads it directly from Intel at run time; if the node has
  no outbound internet, download it elsewhere and place the binary manually (the
  script's header comment explains where). `--latency_matrix` may need root; if it isn't
  available, that failure is recorded as a warning in the result JSON rather than
  aborting the run.
