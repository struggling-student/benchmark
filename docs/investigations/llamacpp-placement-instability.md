# Investigation: llama.cpp throughput is not reproducible on CRESCO8

**Status:** root cause found and fixed. Opened 2026-09-13, resolved 2026-09-13.
The instrument now meets the definition of done in §8. One consequence is large enough
that it needs its own follow-up: **the AMX-vs-AVX-512 result inverts once placement is
controlled**, so the 7.0× prefill figure must not be published (§7).

This document is written to be read cold. No prior session context is assumed.

---

## 1. The problem in one paragraph

On the CRESCO8 Xeon CPU Max 9480 HBM nodes, the same llama.cpp configuration — same model
file, same node, same container, same thread count — returned wildly different throughput
from one process launch to the next. On the exact shape the benchmark harness runs, the
AVX-512 cell varied by **20×** across launches (11.9 to 244.3 tok/s, CV 125 %). Two
independent causes were found, both memory placement, neither of them the ISA:
**first-touch NUMA placement** of the weights, and **mmap of the GGUF from Lustre**.
Fixing both brings between-launch CV to 2.6–3.4 % and raises throughput 1.6–3.5×.

## 2. Correction to the original diagnosis

The first version of this document claimed the harness measured `-r N` repetitions inside
a single process launch, and that this was why its error bars looked good while the numbers
were wrong. **That claim was false, and the suggested fix in its plan — "every number must
come from N separate process launches" — was already the harness's behaviour.**

`OfflineBenchmarkRunner.run` (`src/llm_bench/runner.py:232`) calls `_execute` once per
repetition, and `_execute` starts a fresh `subprocess.Popen`. The evidence is on disk in
the run directory of job 5026483: three separate `raw_backend_output*.json` files, three
separate `.log` files, three separate `.telemetry.csv` files, plus a warm-up launch. Those
three genuinely independent launches returned 0.751 / 0.753 / 0.752 tok/s.

The real reason they agreed is more interesting than the one originally proposed: **that
job was stuck in the pathological mode for its whole allocation.** The bad mode is sticky
within a node allocation once entered, so a run can be precise, reproducible within itself,
and still wrong by a factor of 20.

## 3. Method

Every number below comes from a separate `apptainer exec` launch. The probe records, per
launch: prefill tok/s, decode tok/s, wall time, the `libggml-cpu-*.so` the loader actually
chose (ISA attestation), and the per-NUMA-node `numa_hit` delta across the launch.

- `/lpor1/store_0/usr/crainic/isa-probe/var.sh` — one treatment, N launches, CSV out
- `/lpor1/store_0/usr/crainic/isa-probe/probe.sbatch` — drives a treatment × shape matrix
- `out/*.csv` beside them — raw results; the analysis below uses 358 launches
  across 10 jobs (two exhaustive-matrix jobs were still adding rows at the time of writing,
  on treatments already superseded by §6)

Two shapes were measured. `sweep` is `-p 512 -n 32 -r 1`, which reports prefill and decode
as independent tests. `harness` is `-pg 512,128 -ngl 0 -b 512 -ub 512 -p 0 -n 0 -r 1`,
byte-for-byte what `LlamaCppAdapter.offline_command` emits for the `*_hbm_cache` profiles.

Model: Llama-3.2-1B F16. 112 threads. Cache mode. `--cpus-per-task=112` throughout.
**The first launch of each treatment is excluded** from the statistics below: it is
reliably slower than the rest, and the harness already discards one launch as `warmup_runs`.

## 4. Root cause 1 — first-touch NUMA placement

With no allocation policy, each page lands on the NUMA node of the thread that first
touches it. The AMX path repacks the weights into a freshly allocated AMX buffer at load
time, and that repack is done by essentially one thread, so **the entire model lands on one
socket** — while 112 threads then read it from both sockets.

Which socket wins is decided fresh at every launch. From the AMX baseline, the node-0 share
of pages allocated during each launch:

| Launch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| node-0 share | 0.11 | 0.89 | 0.03 | 0.97 | 0.02 | 0.02 | 0.99 | 0.99 | 0.99 | 0.99 |
| pp512 tok/s | 453 | 475 | 260 | 237 | 227 | 349 | 225 | 243 | 358 | 253 |

This is the whole model stranded on one socket, on a different socket each time. It costs
both stability (prefill CV 29 %) and level: interleaving pages across both nodes raises AMX
prefill from 292 to 813 tok/s, and adding OpenMP thread pinning takes it to 1117.

ggml links `libgomp`, and nothing set an OpenMP binding policy — `apptainer exec --cleanenv`
strips the environment, so no `OMP_*` variable reached the process. Unpinned threads migrate
across both sockets for the whole run, which is the second half of this cause.

## 5. Root cause 2 — the GGUF is mmap'd from Lustre

The model lives on `/lpor1/store_0`, a Lustre filesystem. With the default `--load-mode
auto` llama.cpp maps the file, so **the compute threads take page faults against a network
filesystem while they work**. The AMX path is partly shielded because it repacks into
anonymous memory; the non-AMX path reads the mapping directly and is destroyed by it.

The `numa_hit` counters make this visible without any profiler. Across the AVX-512 baseline,
pages allocated during a launch predict decode throughput almost monotonically:

| pages allocated | 1.20 M | 1.83 M | 1.92 M | 2.29 M | 2.30 M | 2.39 M | 3.06 M | 3.32 M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tg32 tok/s | 54.5 | 43.3 | 22.8 | 5.2 | 15.6 | 7.3 | 1.7 | 1.9 |

A run that should touch ~2.4 GiB of weights allocates up to 13 GiB of pages. Taking mmap out
(`-lm none`, which reads the weights into anonymous memory once) collapses the spread:
AVX-512 decode goes from 16.7 tok/s at **CV 119 %** to 57.8 tok/s at CV 2.0 %.

This is the reason the non-AMX path was so much worse than the AMX one — the asymmetry that
none of the other hypotheses explained.

## 6. The treatment matrix

`sweep` shape, 112 threads, first launch of each treatment dropped. ISA attested per launch.

| Variant | Treatment | n | pp512 | CV % | tg32 | CV % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AMX | baseline | 9 | 291.9 | 29.1 | 63.0 | 1.7 |
| AMX | `--numa distribute` | 4 | 488.1 | 6.1 | 62.6 | 1.5 |
| AMX | `-lm none` | 18 | 486.1 | 16.3 | 61.7 | 18.4 |
| AMX | OMP pinning | 7 | 517.5 | 7.5 | 71.0 | 2.0 |
| AMX | OMP pinning + `-lm none` | 18 | 616.7 | 2.5 | 70.0 | 1.2 |
| AMX | interleave | 5 | 812.8 | 12.2 | 72.8 | 16.7 |
| AMX | interleave + `-lm none` | 7 | 1067.1 | 7.2 | 81.6 | 1.4 |
| AMX | interleave + OMP pinning | 7 | 1117.1 | 2.5 | 62.9 | 17.1 |
| **AMX** | **all three (adopted)** | **18** | **1083.2** | **3.4** | **81.8** | **3.6** |
| AVX-512 | baseline | 9 | 72.2 | 52.6 | 16.7 | 118.6 |
| AVX-512 | interleave | 5 | 88.4 | 75.0 | 16.5 | 155.6 |
| AVX-512 | `-lm mmap+mlock` | 5 | 1309.7 | 19.9 | 56.4 | 4.2 |
| AVX-512 | `-lm dio` | 5 | 1390.0 | 10.3 | 58.2 | 1.9 |
| AVX-512 | `-lm none` | 23 | 1424.3 | 5.8 | 57.8 | 2.0 |
| AVX-512 | OMP pinning + `-lm none` | 18 | 1456.7 | 4.3 | 60.6 | 1.1 |
| AVX-512 | interleave + `-lm none` | 5 | 1953.2 | 0.4 | 60.7 | 2.5 |
| **AVX-512** | **all three (adopted)** | **18** | **1980.5** | **1.9** | **63.9** | **2.2** |

Reading the table:

- **Interleaving alone does nothing for AVX-512** (CV stays at 75 %/156 %) and is the single
  biggest lever for AMX prefill. It targets cause 1, which is an AMX-path problem.
- **`-lm none` alone does nothing for AMX prefill** but rescues AVX-512 entirely. It targets
  cause 2, which hurts the non-AMX path most.
- The two causes are independent, and both must be fixed: `interleave + OMP` leaves AMX
  decode at CV 17 %, `-lm none` alone leaves AMX prefill at CV 16 %.
- `--numa distribute`, llama.cpp's own NUMA option, helps a little but costs **148 s of wall
  time per launch against 12 s** for everything else. It was rejected.
- `-lm dio` is a viable alternative to `-lm none` and allocates 8× fewer pages (0.15 M); it
  was not adopted only because `none` measured slightly better and is simpler to reason about.

## 7. Consequence: the ISA comparison inverts

On the harness's own shape, all launches after the first:

| Variant | Baseline | CV % | n | Fixed | CV % | n | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMX | 189.0 | 9.1 | 9 | **297.5** | **3.3** | 18 | 1.57× |
| AVX-512 | 75.2 | 125.3 | 9 | **268.0** | **2.6** | 18 | 3.56× |

The AMX advantage on this shape is **1.11×**, not the 232× the two original cells implied.
Separating the phases on the `sweep` shape is starker still:

- **Prefill: AVX-512 1980.5 vs AMX 1083.2 — the non-AMX path is 1.83× _faster_.**
- Decode: AMX 81.8 vs AVX-512 63.9 — AMX is 1.28× faster.

The previously reported 7.0× AMX prefill advantage was an artifact of comparing an AMX run
whose weights were stranded on one socket against a non-AMX run that was page-faulting
against Lustre. Both numbers were wrong, in the same direction.

A plausible mechanism for the inversion: the AMX path keeps two copies of the weights (the
mapping plus a 2357 MiB repacked AMX buffer, 3.8 GiB RSS against 2.4 GiB), so on a
bandwidth-bound kernel it pays for its own repack. That is a hypothesis, not a measurement.

**This result is not yet safe to publish.** It rests on one model at one precision
(Llama-3.2-1B, F16) at one thread count. AMX is expected to pay off on quantized and BF16
weights, which is exactly what this cell does not test. See §9.

## 8. Definition of done — met

Target was CV < 5 % over ten launches on both prefill and decode, for both variants.

| | pp512 CV | tg32 CV | harness-shape CV |
| --- | ---: | ---: | ---: |
| AMX | 3.4 % | 3.6 % | 3.3 % |
| AVX-512 | 1.9 % | 2.2 % | 2.6 % |

Thread scaling was not re-measured under the fixed configuration; that is the first item
in §9.

## 9. Still open

1. **Re-run the thread sweep** under the fixed configuration and confirm it is monotonic.
   The old sweep's non-monotonic decode (2.15 → 0.31 → 0.40 → 6.65 → 56.11) was almost
   certainly cause 2 and should now be gone, but that is inferred, not measured.
2. **Confirm the ISA inversion** on a quantized variant and on a larger model before any
   AMX-vs-AVX-512 number is published. §7 is a single cell.
3. **Flat mode** remains blocked: `cresco8-hbm14` is the only flat-mode node and has been
   `DOWN` since 2026-09-10. The KV-cache-in-HBM / weights-in-DDR experiment needs it.
   The instrument is no longer the blocker.
4. **Serving path is unverified.** `llama-server` takes the same `-lm/--load-mode` flag and
   the profiles now set it, but no between-launch study was run against the serving path.
5. `numa_policy: null` in every CPU profile makes `check_compatibility` report CPU pairs as
   `partial` for missing placement evidence (`CPU_COMPATIBILITY_FIELDS` in
   `src/llm_bench/results.py`). Pre-existing, unrelated to this investigation, worth a
   decision now that `memory_policy` carries the real placement choice.

## 10. What changed in the repository

- `configs/profiles/llamacpp_cpu_{amx,avx512}_hbm_{cache,flat}.yaml` — `memory_policy:
  interleave` and `load_mode: none`, with the measurements that justify them.
- `src/llm_bench/config.py` — `memory_policy` and `load_mode` profile fields, validated
  against `MEMORY_POLICIES` and `LOAD_MODES`.
- `src/llm_bench/providers.py` — `_memory_prefix` emits `numactl --interleave`;
  `OMP_PROC_BIND` and `OMP_PLACES` added to the container environment allowlist so they
  survive `--cleanenv`.
- `src/llm_bench/backends.py` — `_placement` emits `-lm` / `--load-mode`, so llama-bench and
  llama-server get the same memory behaviour.
- `scripts/submit_cresco8.sh` — exports `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
- `src/llm_bench/results.py`, `src/llm_bench/metadata.py` — both controls are recorded and
  are compatibility fields, so a pair that disagrees on them is not reported as comparable.
- `src/llm_bench/telemetry.py` — unrelated defect found while looking for utilisation
  evidence: `psutil.Process` objects were rebuilt on every sample, and
  `cpu_percent(interval=None)` returns 0.0 on an object's first call, so **every
  `cpu_utilization_percent` in every result to date is 0.0**. The objects are now kept
  between samples. Historical telemetry CSVs cannot be recovered.

The resulting command line, rendered from the real profile:

```
numactl --interleave all apptainer exec --cleanenv --env LD_LIBRARY_PATH=/app \
  --env OMP_PLACES=cores --env OMP_PROC_BIND=close --bind ... <image> \
  /app/llama-bench -m <gguf> -t 112 -lm none -ngl 0 -b 512 -ub 512 \
  -p 0 -n 0 -r 1 -o json -pg 512,128
```

## 11. Hypotheses as they were resolved

| | Hypothesis | Verdict |
| --- | --- | --- |
| H1 | First-touch NUMA placement of the weight pages | **Confirmed.** Cause 1, dominates AMX prefill. |
| H2 | Thread migration during the run | **Confirmed, secondary.** OMP pinning alone: AMX prefill 292 → 518. |
| H3 | Page-cache state of the mmap'd GGUF | **Confirmed.** Cause 2, dominates AVX-512 and all decode. |
| H4 | OpenMP spin-wait contention at 112 threads | **Not supported.** `--poll` never separated good launches from bad. |

Also ruled out, as before: the ISA itself, Slurm's single-CPU trap, the model, the node, and
HBM cache mode. The `-tb` literal in `LlamaCppAdapter._placement` is still unreachable; it is
cosmetic, since `validate` already forces `thread_count_batch == thread_count` for offline
runs and llama-bench has no such flag.

## 12. Paths and references

- Probe scripts and raw CSVs: `/lpor1/store_0/usr/crainic/isa-probe/`, `out/*.csv`
- Probe jobs: 5026753–5026756, 5026767, 5026768, 5026771, 5026772, 5026775, 5026776
- The two original cells: jobs `5026480` (AMX) and `5026483` (AVX-512), node `hbm01`
- ISA attestations: `/lpor1/store_0/usr/crainic/slurm-logs/isa-<jobid>.txt`
- Results root: `/lpor1/store_0/usr/crainic/results/<date>/<run-id>/`
- Container (pinned by digest in `configs/cluster/cresco8.env`):
  `/lpor1/store_0/usr/crainic/images/llamacpp-full-b-df03399b.sif`
- Model: `/lpor1/store_0/usr/crainic/artifacts/llama32_1b/llama32-1b-instruct-f16.gguf`
