# 07 — KV-cache capacity-cliff experiments

The `membench` microbenchmark (STREAM/MLC, see the `feature/stream-mlc-memory-bandwidth`
branch) found that Llama-3.2-1B's real footprint (~2.5 GiB weights + 8 GiB capped KV
cache ≈ 10.5 GiB) never approaches the 64 GiB/socket Xeon Max HBM budget, and that a
synthetic STREAM sweep shows a real bandwidth cliff once a working set exceeds that
budget: cache mode's Triad rate falls from ~515-568 GB/s (under 64 GiB) to ~202 GB/s
(over it), a ~64% drop.

That cliff was measured with a synthetic memory probe, not real inference. This
experiment stresses it with actual vLLM serving on the *same* Llama-3.2-1B model — no
model change — by raising the KV-cache reservation and driving enough concurrent
long-context load that real, resident KV usage crosses 64 GiB. Two questions:

- **A (cache mode):** does the STREAM-measured cliff show up as a real vLLM throughput
  cliff, or does something else (attention compute, scheduling overhead) dominate
  first?
- **B (flat mode, HBM-only):** unlike cache mode's graceful degradation, binding
  strictly to one 64 GiB HBM node should hit a hard capacity wall. Soft cliff vs. hard
  wall is the contrast worth having on record before considering any real placement
  code in vLLM.

This is config/workload-only. No vLLM source is touched here.

## Why tensor_parallel_size: 1

The shipped vLLM profiles use `tensor_parallel_size: 2` (one rank per socket), and
`memory_binding: hbm` resolves symbolically to **both** HBM nodes machine-wide
(`hardware.py`'s `resolve_memory_binding` returns `"2,3"` for the `hbm` tier — every
memory-only node on the box, not one socket's). At TP=2 with a large total KV budget,
each rank's shard only needs to stay under half of the *combined* ~128 GiB HBM pool to
avoid failure, so a naive "raise the budget on the existing flat profile" would not
reliably hit a clean 64 GiB/*socket* wall — and would reintroduce the cross-socket
placement confound the membench work already characterized (a ~5x bandwidth penalty),
which isn't what this experiment is testing.

Running at `tensor_parallel_size: 1` — one rank, one socket — matches the STREAM/MLC
microbenchmark's own scope exactly (56 cores, one NUMA node pair) and keeps the result
clean. Multi-rank placement is exactly the kind of thing a real placement PoC would
need to solve; it's deliberately out of scope here.

## KV-cache allocation is lazy (confirmed)

Whether vLLM's CPU KV-cache allocation is eager (touches every page at model-load
time) or lazy (physical pages faulted in only as blocks are actually used during
serving) wasn't verifiable from source on this machine, so it was checked empirically
on `cresco8-hbm14`: the B profile's 80 GiB reservation against a tiny workload
(`fixed_32_32`, ~8 MiB of real KV demand) completed a full run — warm-up plus 3
repetitions — without any allocation failure. Eager allocation would have failed at
model-load time regardless of workload size; it didn't, so allocation is lazy. This is
exactly why B needs genuine concurrent demand (the sizing math below), not just a large
configured budget, to actually reach the 64 GiB wall.

## CPUWorker silently ignores memory_binding (found running B, fixed)

The first real run of the B profile did not hit any capacity wall, but not because the
budget was too small — the memory never landed on HBM at all. `numastat -p` on the
actual worker process (not the top-level CLI process, but the `VLLM::Worker` child
vLLM's engine spawns) showed ~85 GiB resident on **node 0 (DDR)** and ~200 MiB on node
2 (the HBM node the profile requested).

Traced into vLLM 0.29.0's own source (`vllm/v1/worker/cpu_worker.py`):

```python
allowed_cpu_list = get_allowed_cpu_list()
cpu_core = allowed_cpu_list[0]
...
memory_node = cpu_core.numa_node   # the NUMA node of the first allowed CPU core
torch.ops._C.init_cpu_memory_env([memory_node])
```

`CPUWorker.__init__` derives its own memory node from the NUMA node of the *first
allowed CPU core*, and force-rebinds memory to it internally — completely independent
of any external `numactl --membind` policy (`memory_binding` in a profile only ever
controls the external `numactl`/`--cpuset-mems` wrapping; it never reaches this code
path). In flat mode this can never pick an HBM node: nodes 2/3 have zero CPUs, so
`cpu_core.numa_node` is structurally always a DDR node (0 or 1) whenever CPU affinity
isn't restricted — which no profile in this project restricts (`cpu_mask: null`
everywhere, including the original campaign's flat profiles). **This means the original
`bf16-xbackend-flat` campaign's vLLM results plausibly never used HBM at all** — the
`memory_binding: hbm` profile setting was silently a no-op for vLLM specifically, for
its whole history. That reframes the "-9%" flat-vs-cache result from that campaign as
possibly DDR-vs-DDR noise, not a real HBM-vs-cache-mode effect; it doesn't retract that
result on its own, but it's no longer safe to cite as an HBM measurement.

**Fix**: vLLM ships a purpose-built override for exactly this
(`vllm/utils/cpu_resource_utils.py`, modeled on `CUDA_VISIBLE_DEVICES`):
`CPU_VISIBLE_MEMORY_NODES`. Setting it restricts `get_visible_memory_node()`'s result,
so `cpu_core.numa_node` (0) is no longer in the allowed set, and `CPUWorker.__init__`'s
own existing fallback branch (`memory_node = allowed_memory_nodes[0]`) fires instead —
now correctly landing on the named node. Confirmed empirically: with
`CPU_VISIBLE_MEMORY_NODES=2` set, the same diagnostic workload showed the Worker
process with ~4.6 GiB on node 2 and ~0 on node 0 (`numa_maps` entirely `bind:2`, zero
`bind:0`) — the exact reverse of the unfixed run.

This is now a first-class profile field, `vllm_cpu_visible_memory_nodes` (wired through
`config.py`, `providers.py`'s `_profile_environment()`, and `metadata.py`/`results.py`
for provenance, mirroring the existing `vllm_cpu_kvcache_space_gib` pattern), set
alongside `memory_binding` on the B profile. Options considered and rejected:
cgroup-level `cpuset.mems` (same effect, more operationally complex, needs cgroup
delegation on a shared node) and patching `cpu_worker.py` directly (unnecessary — this
is a supported, in-source mechanism, not a workaround).

## Files

| Path | Role |
| --- | --- |
| `configs/profiles/experiments/vllm_cpu_amx_hbm_cache_kvstress_control.yaml` | A1 — cache mode, 20 GiB KV budget (well under the cliff). |
| `configs/profiles/experiments/vllm_cpu_amx_hbm_cache_kvstress.yaml` | A2 — cache mode, 70 GiB KV budget (past the cliff). |
| `configs/profiles/experiments/vllm_cpu_amx_hbm_flat_hbm_only_kvstress.yaml` | B — flat mode, explicit single-node `memory_binding: "2"` **and** `vllm_cpu_visible_memory_nodes: "2"` (the latter is what actually works — see below), 80 GiB budget. |
| `configs/workloads/experiments/kv_stress_30000_10000.yaml` | Same 30K-in/10K-out shape as the literature `fixed_30000_10000.yaml`, `number_of_prompts: 200` so vLLM's scheduler is forced to hold many requests concurrently. `repetitions: 1`, no warmup — a capacity/cliff demonstration doesn't need statistical precision, and 3x repeating an already-heavy 200-request batch would multiply an expensive run for no benefit. |

These live under an `experiments/` subdirectory, not directly in `configs/workloads/`
or `configs/profiles/`, on purpose: `tests/test_config.py` treats those two
directories as a closed, fully-cataloged "shipped matrix" (exact model × workload ×
profile counts, and every workload's shape asserted against
`docs/06_LITERATURE_RESULTS.md`). This workload isn't literature-matched — it's a
synthetic capacity stress test — so it belongs outside that matrix rather than forcing
a misleading literature citation for it. The subdirectory sits below the tests'
non-recursive `glob("*.yaml")`, so the shipped-matrix tests are unaffected.

A1 and A2 share everything except the KV budget, which is the point: a single A2 run
alone couldn't distinguish "slow because of the cliff" from "just what this workload
costs." B uses the same workload shape so its result is directly comparable to A2's.

## Sizing the workload

This model's KV cache is ~32 KiB/token (16 layers × 8 KV heads × 64 head_dim × 2 bytes
× 2 for K+V). At the full 40,000-token length (30K prompt + 10K decode), that's
`32 KiB × 40,000 ≈ 1.22 GiB` per fully-grown request. Rough concurrent-capacity
estimates:

| Budget | ~Concurrent capacity at full length |
| --- | --- |
| 20 GiB (A1) | ~16 requests |
| 70 GiB (A2) | ~57 requests |
| 80 GiB (B) | ~65 requests |

`number_of_prompts: 200` is several multiples of even the largest of these, so the
scheduler should stay saturated for a real measurement window rather than a brief
spike — but this is an estimate, not a guarantee: a request's resident footprint grows
from ~30,000 tokens (prompt only, at the start of decode) up to the full 40,000, and
vLLM's own admission behavior isn't otherwise controlled by this harness (see "Gaps"
below). Treat 200 as a starting point to validate via a short, cheap smoke run before
committing a full 200-prompt job to Slurm.

## Gaps in this harness surfaced while designing this

- There's no `max_num_seqs` field anywhere in `ExecutionProfile`/`backends.py`, so
  vLLM's own default (256) governs the hard ceiling on concurrent sequences. That's
  fine for this experiment (256 > our target concurrency), but means true concurrency
  can't be capped or forced from a profile if a future experiment needs to.
- Confirming that resident KV cache actually crossed 64 GiB requires reading
  `numastat`/telemetry from the run, not just trusting the configured budget — a
  budget merely *allows* crossing 64 GiB, it doesn't guarantee real demand got there.
