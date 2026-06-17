# SLURM templates for the SOAP LLM-judge

Reusable, **cluster-agnostic** scripts to run `btc-eval soap-judge` data-parallel
across a SLURM array, then merge the shards with `btc-eval soap-aggregate`.
Nothing here hardcodes a cluster, account, or model — all cluster-specific values
arrive as environment variables, so the same files run on OSC Ascend, a generic
university `--partition=gpu` cluster, or a cloud SLURM.

> For the BeTraC 2026 organizer eval set, **don't call these directly** — use the
> wrapper in [`eval2026/judge/run_eval_judge.sh`](../../betrac-private/eval2026/judge/run_eval_judge.sh),
> which fills in the OSC Ascend + `gemma4:31b` values and runs all three
> sub-datasets. These templates are the engine it drives.

## Files

| File | Role |
|------|------|
| `submit_soap_judge.sh`   | Orchestrator: even-chunks predictions (`split`, exactly `SAMPLES_PER_SHARD` dialogs/chunk), submits the judge array + the aggregate (chained `afterany`, on `AGG_PARTITION` if set). |
| `run_soap_judge.slurm`   | Array task: starts a per-task Ollama, pre-warms the model, judges its chunk file (`chunk_<i>.jsonl`). |
| `run_soap_aggregate.slurm` | CPU task: pools per-shard `soap_judge_per_dialog.jsonl` + `soap_judge_status.jsonl` and **recomputes** the aggregate (and the known-error report). |

## How the parallelism works

`submit_soap_judge.sh` **even-chunks** the predictions with `split` — exactly
`SAMPLES_PER_SHARD` dialogs per chunk file (`chunk_0000.jsonl`, …) — and runs one
array task per chunk (`N` derived from the chunk count, not an input). Even
chunking gives predictable per-job size/walltime, unlike the older hash-shard
(`--shard i/N`) which left jobs with 0–N dialogs. Each task writes its own
`OUTPUT_DIR/shard_<i>/`, so per-task files never race. `soap-aggregate` pools the
per-dialog rows across shards and recomputes mean/std over the union of dialogs
(the correct merge — not averaging per-shard means), and consolidates the
per-shard status into a known-error report. The id-keyed, **model-keyed**
raw-response cache (point shards at one `--cache-dir`) makes a failed shard cheap
to re-run (resume on by default); a different model never reuses another's cache.

## Prerequisites

1. **A venv with the judge installed** (`BTC_VENV`):
   ```bash
   cd betrac-metrics-private
   make setup                      # uv venv (py3.10) + btc-eval[...]; ignores the model
   uv pip install '.[llm-judge]'   # the Ollama backend needs `requests`
   ```
2. **Ollama on the compute nodes** — a module (`module load ollama/0.13.1`) or a
   binary on `PATH`. Each task starts its **own** server on a unique port. On OSC
   the `ollama` module provides an `ollama_start_helper` (sets `OLLAMA_PORT`,
   backgrounds `ollama serve`, defines `ollama_pull`). If your cluster has no such
   helper, the template falls back to a single `ollama serve` on port 11434; for
   multiple tasks per node, point `OLLAMA_START_HELPER` at your own helper script.
3. **The model, pre-pulled once** — compute nodes usually can't reach the network.
   Pull on a login node (`PREPULL=1`, or by hand) into the shared `OLLAMA_MODELS`
   cache **before** the jobs run.

## Environment variables

**Required**

| Var | Meaning |
|-----|---------|
| `BTC_VENV`    | venv with `btc-eval` installed (`.../.venv`) |
| `PREDICTIONS` | predictions JSONL — `{"id","summary"}` (summary = the SOAP note) |
| `TRANSCRIPTS` | transcripts JSONL — `{"id","transcript"}` (extra ids are harmless) |
| `JUDGE_MODEL` | model id for the Ollama backend, e.g. `gemma4:31b` |
| `OUTPUT_DIR`  | run root; shards land in `$OUTPUT_DIR/shard_<i>/`, result in `$OUTPUT_DIR/aggregated/` |

**Sizing**

| Var | Meaning |
|-----|---------|
| `SAMPLES_PER_SHARD` | dialogs per chunk = per array task; `N` is derived from the chunk count (default **4** for the Ollama/1-req-per-GPU path; vLLM batches, so use 24–48). `NUM_SHARDS` is no longer an input. |
| `MAX_CONCURRENT`    | cap on simultaneously-running array tasks (`--array=…%MAX_CONCURRENT`); default 24 |

**Cluster** (each added to `sbatch` only if set)

| Var | Maps to |
|-----|---------|
| `CLUSTER`    | `--cluster=` (multi-cluster sites like OSC) |
| `ACCOUNT`    | `--account=` |
| `PARTITION`  | `--partition=` (single-cluster sites) |
| `JUDGE_TIME` | `--time=` override (array task walltime) |
| `GPUS`       | `--gpus-per-task=` override (default 1) |

**Backend / tuning** (sensible defaults)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_MODULE`         | `ollama`        | module to `module load` for Ollama |
| `PYTHON_MODULE`         | `python/3.10`   | module to `module load` for Python |
| `OLLAMA_MODELS`         | `~/.ollama/models` | shared model cache, **writable** (login + compute must share it). On OSC the module clobbers it to a read-only path — the template re-applies your value after `module load`. |
| `OLLAMA_MODEL`          | `$JUDGE_MODEL`  | tag to pull/pre-warm if it differs from the judge id |
| `OLLAMA_START_HELPER`   | `ollama_start_helper` | path/name of the per-task server helper |
| `OLLAMA_SIF`            | (none)          | bring-your-own Ollama Apptainer image; overrides the module's `$OLLAMA_IMG`. Use when the site's module is too old for the model (e.g. OSC + `gemma4:31b`). See the eval2026 OSC guide. |
| `OLLAMA_CONTEXT_LENGTH` | `32768`         | context window. Judge stage needs ~20k (transcript+note+claims+output). **On OSC, `apptainer run --cleanenv` strips this** — the server auto-picks from VRAM (32k on A100-40GB). |
| `MAX_TOKENS`            | `16000`         | max output tokens/call — **reasoning models need 20000+** |
| `WORKERS`               | `4`             | concurrent in-task requests. No benefit when only one slot fits (`NUM_PARALLEL=1`, e.g. a 31B model on a 40 GB card), since the server serializes. |
| `PREPULL`               | `0`             | `1` = pull the model on the submit (login) node first |

## Run it (generic cluster)

```bash
BTC_VENV=$PWD/.venv \
PREDICTIONS=preds.jsonl TRANSCRIPTS=transcripts.jsonl \
JUDGE_MODEL=gemma4:31b OUTPUT_DIR=out/run1 \
PARTITION=gpu OLLAMA_MODULE=ollama PREPULL=1 \
bash slurm/submit_soap_judge.sh
```

## Run it (OSC Ascend)

```bash
BTC_VENV=$PWD/.venv \
PREDICTIONS=preds.jsonl TRANSCRIPTS=transcripts.jsonl \
JUDGE_MODEL=gemma4:31b OUTPUT_DIR=out/run1 \
CLUSTER=ascend ACCOUNT=PAS2138 \
OLLAMA_MODULE=ollama/0.13.1 OLLAMA_MODELS=$HOME/.ollama/models \
PREPULL=1 SAMPLES_PER_SHARD=4 \
bash slurm/submit_soap_judge.sh
```

Outputs:

- `OUTPUT_DIR/shard_<i>/` — per-shard `soap_judge_per_dialog.jsonl`, `summary.json`,
  `soap_eval_summary.csv`, raw caches (`extract_raw/`, `judge_raw/`), `gpu_util.log`.
- `OUTPUT_DIR/aggregated/summary.json` — the merged result: `num_dialogs` and
  `mean_/std_` for faithfulness, structure, coverage, conciseness (+ error rates).
- `OUTPUT_DIR/logs/` — `judge_<jobid>_<task>.{out,err}`, `aggregate_<jobid>.{out,err}`.

## Tuning notes

- **Shard size vs. walltime.** Each task pays the model-load cost once, then does
  `2 × dialogs` LLM calls (extract + judge). Size shards so a task finishes under
  your cluster's preferential-scheduling window (often 1 h). Start at
  `SAMPLES_PER_SHARD=4` and adjust from the per-shard timing in `summary.json`.
- **`WORKERS`.** Concurrency against the single per-task Ollama server. 4 is safe;
  raise it only if the GPU is underused (watch `gpu_util.log`).
- **Re-running.** Resume is on; just resubmit (or `sbatch --array=<failed ids>`
  the array script). Cached raws make completed dialogs free.
- **Scores are part of the metric.** Judge scores are **not comparable across
  models or backends** — fix one judge model when comparing systems.
