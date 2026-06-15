#!/bin/bash
#
# Orchestrator for the SOAP LLM-judge on SLURM (cluster-agnostic).
#
# Splits one predictions file into fixed-size CHUNK files (exactly
# SAMPLES_PER_SHARD dialogs each) and runs one array task per chunk, then a CPU
# aggregate chained with afterok. Even chunking (vs the engine's hash --shard)
# guarantees predictable per-job size and no empty/oversized jobs. Concurrency is
# throttled (--array=...%MAX_CONCURRENT) to avoid saturating shared storage.
#
# Run once per submission — and, for the BeTraC eval set, once per sub-dataset
# (see eval2026/judge/run_eval_judge.sh, which wraps this for OSC + gemma4:31b).
# Cluster-specific bits are env vars; nothing here hardcodes a cluster/model.
#
#   BTC_VENV=/path/.venv PREDICTIONS=preds.jsonl TRANSCRIPTS=transcripts.jsonl \
#   JUDGE_MODEL=gemma4:31b OUTPUT_DIR=out/run1 \
#   CLUSTER=ascend ACCOUNT=PAS2138 OLLAMA_SIF=$HOME/.../ollama.sif \
#   SAMPLES_PER_SHARD=4 MAX_CONCURRENT=24 \
#   bash slurm/submit_soap_judge.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Required -------------------------------------------------------------
: "${BTC_VENV:?set BTC_VENV to the venv with btc-eval installed}"
: "${PREDICTIONS:?set PREDICTIONS to a {id,summary} JSONL}"
: "${TRANSCRIPTS:?set TRANSCRIPTS to a {id,transcript} JSONL}"
: "${JUDGE_MODEL:?set JUDGE_MODEL (e.g. gemma4:31b)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for this run}"

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"; mkdir -p "${LOG_DIR}"

# --- Even chunking: exactly SAMPLES_PER_SHARD dialogs per array task -------
# Deterministic given (predictions, K): the predictions file is id-ordered, so
# re-running with the same K reuses each dialog's raw cache.
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-4}"
CHUNK_DIR="${OUTPUT_DIR}/chunks"
rm -rf "${CHUNK_DIR}"; mkdir -p "${CHUNK_DIR}"
grep -c . "${PREDICTIONS}" >/dev/null || { echo "ERROR: cannot read ${PREDICTIONS}" >&2; exit 1; }
split -d -a 4 -l "${SAMPLES_PER_SHARD}" --additional-suffix=.jsonl \
    "${PREDICTIONS}" "${CHUNK_DIR}/chunk_"
NUM_CHUNKS=$(find "${CHUNK_DIR}" -name 'chunk_*.jsonl' | wc -l)
[ "${NUM_CHUNKS}" -ge 1 ] || { echo "ERROR: no chunks produced from ${PREDICTIONS}" >&2; exit 1; }
MAX_TASK=$(( NUM_CHUNKS - 1 ))

# Throttle concurrent array tasks (shared-storage protection). 0/empty = no cap.
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"
if [ "${MAX_CONCURRENT}" -gt 0 ] 2>/dev/null; then
    ARRAY_SPEC="0-${MAX_TASK}%${MAX_CONCURRENT}"
else
    ARRAY_SPEC="0-${MAX_TASK}"
fi

# --- Optional cluster flags (only added when set) -------------------------
SBATCH_FLAGS=()
[ -n "${CLUSTER:-}" ]    && SBATCH_FLAGS+=(--cluster="${CLUSTER}")
[ -n "${ACCOUNT:-}" ]    && SBATCH_FLAGS+=(--account="${ACCOUNT}")
[ -n "${PARTITION:-}" ]  && SBATCH_FLAGS+=(--partition="${PARTITION}")
[ -n "${JUDGE_TIME:-}" ] && SBATCH_FLAGS+=(--time="${JUDGE_TIME}")
[ -n "${GPUS:-}" ]       && SBATCH_FLAGS+=(--gpus-per-task="${GPUS}")
[ -n "${JOB_MEM:-}" ]    && SBATCH_FLAGS+=(--mem="${JOB_MEM}")

# --- Pass-through env for the array tasks ---------------------------------
# Each task derives its predictions file from CHUNK_DIR + its array index.
EXPORT="ALL,BTC_VENV=${BTC_VENV},TRANSCRIPTS=${TRANSCRIPTS}"
EXPORT+=",JUDGE_MODEL=${JUDGE_MODEL},OUTPUT_DIR=${OUTPUT_DIR},CHUNK_DIR=${CHUNK_DIR},NUM_CHUNKS=${NUM_CHUNKS}"
EXPORT+=",OLLAMA_MODEL=${OLLAMA_MODEL:-${JUDGE_MODEL}}"
for v in OLLAMA_MODULE PYTHON_MODULE OLLAMA_MODELS OLLAMA_START_HELPER OLLAMA_SIF \
         OLLAMA_CONTEXT_LENGTH MAX_TOKENS WORKERS OLLAMA_WARMUP_SECONDS \
         OLLAMA_TIMEOUT STAGE_MODEL; do
    [ -n "${!v:-}" ] && EXPORT+=",${v}=${!v}"
done

# --- Optional login-node model pre-pull (compute nodes can't download) -----
# Generic (non-OSC) path. On OSC the model is staged from OLLAMA_MODELS, cached
# by setup_ollama_image.sh; run with PREPULL=0.
if [ "${PREPULL:-0}" = "1" ]; then
    echo ">>> Pre-pulling ${OLLAMA_MODEL:-${JUDGE_MODEL}} on this (login) node ..."
    command -v module >/dev/null 2>&1 && [ -n "${OLLAMA_MODULE:-}" ] && module load "${OLLAMA_MODULE}" || true
    export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
    ollama pull "${OLLAMA_MODEL:-${JUDGE_MODEL}}"
fi

echo "========================================================"
echo "SOAP LLM-judge submission"
echo "  venv:        ${BTC_VENV}"
echo "  predictions: ${PREDICTIONS}  ($(grep -c . "${PREDICTIONS}") dialogs)"
echo "  transcripts: ${TRANSCRIPTS}"
echo "  model:       ${JUDGE_MODEL}   stage=${STAGE_MODEL:-1}  sif=${OLLAMA_SIF:-<module>}"
echo "  output:      ${OUTPUT_DIR}"
echo "  jobs:        ${NUM_CHUNKS}  (array ${ARRAY_SPEC}, exactly ${SAMPLES_PER_SHARD} dialogs/job)"
echo "  cluster:     ${CLUSTER:-<none>}  account: ${ACCOUNT:-<none>}  mem: ${JOB_MEM:-<default>}"
echo "========================================================"

# --- 1. Judge array -------------------------------------------------------
ARRAY_OUT=$(sbatch \
    "${SBATCH_FLAGS[@]}" \
    --array="${ARRAY_SPEC}" \
    --export="${EXPORT}" \
    --output="${LOG_DIR}/judge_%A_%a.out" \
    --error="${LOG_DIR}/judge_%A_%a.err" \
    "${HERE}/run_soap_judge.slurm")
echo "${ARRAY_OUT}"
ARRAY_JOB_ID=$(echo "${ARRAY_OUT}" | awk '{print $4}')

# --- 2. Aggregate (afterANY: always run, even if some tasks fail/timeout) --
# Pools whatever per-dialog rows exist and reports coverage. With the shared
# resume cache, re-running the array then finishes any gaps cheaply and a second
# aggregate completes the picture. (afterok would yield NOTHING on any failure.)
AGG_OUT=$(sbatch \
    "${SBATCH_FLAGS[@]}" \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --export="ALL,BTC_VENV=${BTC_VENV},OUTPUT_DIR=${OUTPUT_DIR},NUM_SHARDS=${NUM_CHUNKS}" \
    --output="${LOG_DIR}/aggregate_%j.out" \
    --error="${LOG_DIR}/aggregate_%j.err" \
    "${HERE}/run_soap_aggregate.slurm")
echo "${AGG_OUT}"
AGG_JOB_ID=$(echo "${AGG_OUT}" | awk '{print $4}')

echo ""
echo "Submitted:"
echo "  judge array: ${ARRAY_JOB_ID}   (${NUM_CHUNKS} jobs, ${ARRAY_SPEC})"
echo "  aggregate:   ${AGG_JOB_ID}   (afterok:${ARRAY_JOB_ID})"
echo ""
echo "Result:  ${OUTPUT_DIR}/aggregated/summary.json"
echo "Monitor: squeue -u \$USER ${CLUSTER:+-M ${CLUSTER}}"
