# 01 — Cluster prerequisites

Start here with an empty Sapienza cluster account. The purpose of this guide is discovery: commands below are examples to run, not claims about the cluster's configuration. Record the actual values you observe or receive from the administrators, then place them in a private copy of `configs/cluster/sapienza.example.env`.

Do not guess a partition, account, QoS, module name, GPU type, Python path, or filesystem path. Slurm sites differ, and values can change.

## Step 0: log in and establish a workspace

Follow the cluster's current access documentation to connect to a login node. Check where you are and whether the repository is on a filesystem intended for source code:

```bash
hostname
pwd
id
git --version
```

Ask the cluster documentation or support team which locations are intended for home files, project data, scratch data, software environments, model caches, and benchmark results. Model caches and weights must be outside this Git repository. Run output may either use an external `RESULTS_ROOT` or the checkout's ignored `results/` directory; it must not be tracked by Git.

## Step 1: verify Slurm and inspect site policy

Check whether the basic Slurm commands are available and review their local help:

```bash
command -v sbatch srun sinfo squeue sacct scontrol
sbatch --version
sinfo --help
srun --help
```

Inspect partitions without assuming which one permits GPUs:

```bash
sinfo
sinfo --format='%P %a %l %D %G'
scontrol show partition
```

The `%G` column and `Gres` fields can reveal generic resources, but the exact GPU request syntax is site-defined. Also determine:

- which partition(s) permit GPU jobs;
- whether a Slurm account is required;
- whether a QoS is required or optional;
- time, memory, CPU, and GPU limits;
- whether interactive jobs are permitted;
- the correct GPU generic-resource syntax;
- how to inspect job accounting after completion.

Useful read-only commands, where the site permits them, include:

```bash
sacctmgr show associations user="$USER" format=Cluster,Account,Partition,QOS
sacctmgr show qos format=Name,MaxWall,MaxTRESPU,MaxJobsPU
squeue -u "$USER"
```

`sacctmgr` output may be restricted. Lack of access is a reason to consult site documentation, not to invent values.

## Step 2: discover GPU resources

Slurm may describe accelerators as generic resources (GRES) or trackable resources (TRES). Inspect the scheduler view first:

```bash
sinfo --Node --long
sinfo --format='%N %P %t %G'
scontrol show nodes
```

Do not assume a GPU model from a partition name. To learn what a job actually receives, request an interactive allocation using values confirmed in Step 1. Replace every angle-bracket placeholder before running:

```bash
srun \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<SHORT_TIME_LIMIT> \
  --pty bash -l
```

Replace `<SITE_APPROVED_ONE_GPU_OPTION>` with the complete one-GPU option documented by the site; the option family and resource spelling are cluster-specific. Omit `--account` or `--qos` only if the site says it is unnecessary. Inside the allocation, inspect what is visible:

```bash
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unset}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi && nvidia-smi
```

Record the accelerator name and count from the allocated node. Do not infer that all nodes in a partition are identical.

## Step 3: inspect environment modules

First determine whether a module command is initialized in the login shell:

```bash
type module
module --version
module list
module avail
```

If `module` is unavailable, consult the site documentation; it is often a shell function initialized by a profile script rather than a standalone executable. If it is available, look for Python and CUDA-related modules without assuming their names:

```bash
module spider python 2>/dev/null || true
module spider cuda 2>/dev/null || true
```

`module spider` is not supported by every module system. `module avail` and the site documentation are the fallback. Record the exact commands needed to recreate the chosen environment; those commands become `MODULE_COMMANDS` in the local cluster configuration.

## Step 3a: discover CPU, ISA, and HBM resources

Do this inside the CPU allocation that will run inference, not on a login node:

```bash
lscpu
numactl --hardware
```

Record the CPU model, sockets, cores, CPU flags, NUMA node CPU lists, and memory capacity per
node. For Xeon CPU Max, flat mode exposes HBM and DDR as separate NUMA memory nodes, while cache
mode exposes DDR and uses HBM as a transparent cache. NUMA node numbers depend on firmware and
SNC configuration, so never copy them from another machine or from this repository's examples.

The benchmark verifies `avx2`, `avx512`, and `amx` profile requirements against `lscpu` flags.
For a flat-mode run, use `memory_binding: hbm` or `memory_binding: ddr`; preflight resolves the
symbolic tier to the addressable NUMA nodes discovered inside the allocation and records both the
requested and numeric binding. An explicit numeric node list remains supported when needed. Cache
mode has no separately addressable HBM nodes and therefore uses no HBM binding.

## Step 4: inspect Python, CUDA, and the NVIDIA driver

Perform these checks both before and after loading any candidate modules:

```bash
command -v python3
python3 --version
python3 -c 'import platform, sys; print(sys.executable); print(platform.platform())'

command -v nvcc && nvcc --version
command -v nvidia-smi && nvidia-smi
```

The CUDA toolkit reported by `nvcc`, the maximum CUDA capability reported by the driver, and the CUDA runtime bundled with a Python package are related but not interchangeable. Capture what is present before selecting a vLLM or CUDA-enabled llama.cpp build. Check compatibility against the current upstream documentation during setup.

Login nodes may not expose a GPU or `nvidia-smi`. Repeat GPU and driver checks inside a real GPU allocation before concluding that NVIDIA support is unavailable.

## Step 5: inspect disk areas and quotas

Identify available filesystems and their policies:

```bash
df -h
df -ih
du -sh "$HOME" 2>/dev/null
quota -s 2>/dev/null || true
```

Sites may provide separate quota commands; use the cluster's documentation. Ask about capacity, inode quota, backup policy, purge policy, performance, and whether compute nodes can see each location.

Plan separate locations for:

- the Git checkout;
- the Python virtual environment;
- the Hugging Face cache and downloaded model snapshots;
- benchmark results and logs.

Both selected models require model weights, tokenizer/configuration files, and temporary cache space. The 8B model ordinarily requires substantially more storage and accelerator memory than the 1B model, but exact requirements depend on precision, quantization, vLLM version, context/KV-cache settings, and file format. Measure free space and successful allocations instead of embedding a hardware-specific estimate.

## Step 6: test network access separately

Test from the login node using a harmless HTTPS request:

```bash
python3 - <<'PY'
from urllib.request import urlopen

with urlopen("https://huggingface.co", timeout=10) as response:
    print(response.status)
PY
```

Then repeat inside a short compute allocation. A login node may have outbound internet access while compute nodes do not. A failed request might also reflect a proxy, certificate policy, or temporary outage; check the site's rules before troubleshooting around it.

If compute nodes are offline, pre-download both model snapshots from a permitted network location into a cache visible to compute nodes. Do not start a large download on a login node unless cluster policy permits it.

## Step 7: create the local cluster configuration

Copy the documented example; the destination is intentionally ignored by Git:

```bash
cp configs/cluster/sapienza.example.env configs/cluster/sapienza.env
```

Fill it only with verified values for:

- `MODULE_COMMANDS`
- `PYTHON_EXECUTABLE`
- `BENCH_REPO_ROOT`
- `VENV_PATH`
- `HF_HOME`
- `MODEL_CACHE_DIR`
- `MODEL_ARTIFACT_ROOT`
- `RESULTS_ROOT`
- native backend executable/converter paths or pinned Docker/Apptainer images
- `SLURM_PARTITION`
- `SLURM_ACCOUNT`
- `SLURM_QOS`
- `SLURM_GPU_REQUEST`
- `SLURM_TIME_LIMIT`

`BENCH_REPO_ROOT` is the absolute path to this checkout as seen by compute nodes. It is required because Slurm may copy the submitted job script to a spool directory, from which the repository cannot be inferred. Record the site's complete one-GPU request syntax in `SLURM_GPU_REQUEST`, but continue to pass the site-approved option explicitly to `srun`/`sbatch`; the provided job files do not invent or embed a resource request.

Keep placeholders until each value is known. Do not store `HF_TOKEN`, passwords, private keys, or other secrets in this file. Before continuing, verify that `HF_HOME`, `MODEL_CACHE_DIR`, and `MODEL_ARTIFACT_ROOT` are outside the repository and visible from compute nodes. `RESULTS_ROOT` may be an external path or the checkout's ignored `results/` directory; never commit generated runs.

Continue with [02 — Environment and model setup](02_ENVIRONMENT_AND_MODEL_SETUP.md).
