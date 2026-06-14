#!/bin/bash
#
# Orchestrator for the SOAP LLM-judge on SLURM (cluster-agnostic).
#
# Submits ONE judge run (one predictions file) as a data-parallel array, then a
# CPU aggregate job chained with afterok. Run it once per submission — and, for
# the BeTraC eval set, once per sub-dataset (see eval2026/judge/run_eval_judge.sh,
# which wraps this with the OSC Ascend + gemma4:31b values).
#
# Cluster-specific bits are environment variables so this script stays portable;
# nothing here hardcodes a cluster, account, or model. See slurm/README.md.
#
#   BTC_VENV=/path/.venv \
#   PREDICTIONS=preds.jsonl TRANSCRIPTS=transcripts.jsonl \
#   JUDGE_MODEL=gemma4:31b OUTPUT_DIR=out/run1 \
#   CLUSTER=ascend ACCOUNT=PAS2138 OLLAMA_MODULE=ollama/0.13.1 \
#   OLLAMA_MODELS=$HOME/.ollama/models PREPULL=1 \
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

# --- Sizing ---------------------------------------------------------------
# NUM_SHARDS explicit, or derived from the prediction count / SAMPLES_PER_SHARD.
# Local models carry per-task model-load overhead, so prefer fewer/larger shards.
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-40}"
if [ -z "${NUM_SHARDS:-}" ]; then
    PRED_COUNT=$(grep -c . "${PREDICTIONS}")
    NUM_SHARDS=$(( (PRED_COUNT + SAMPLES_PER_SHARD - 1) / SAMPLES_PER_SHARD ))
    [ "${NUM_SHARDS}" -lt 1 ] && NUM_SHARDS=1
fi
MAX_TASK=$(( NUM_SHARDS - 1 ))

# --- Optional cluster flags (only added when set) -------------------------
SBATCH_FLAGS=()
[ -n "${CLUSTER:-}" ]   && SBATCH_FLAGS+=(--cluster="${CLUSTER}")
[ -n "${ACCOUNT:-}" ]   && SBATCH_FLAGS+=(--account="${ACCOUNT}")
[ -n "${PARTITION:-}" ] && SBATCH_FLAGS+=(--partition="${PARTITION}")
[ -n "${JUDGE_TIME:-}" ] && SBATCH_FLAGS+=(--time="${JUDGE_TIME}")
[ -n "${GPUS:-}" ]      && SBATCH_FLAGS+=(--gpus-per-task="${GPUS}")

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# --- Pass-through env for the jobs ---------------------------------------
EXPORT="ALL,BTC_VENV=${BTC_VENV},PREDICTIONS=${PREDICTIONS},TRANSCRIPTS=${TRANSCRIPTS}"
EXPORT+=",JUDGE_MODEL=${JUDGE_MODEL},OUTPUT_DIR=${OUTPUT_DIR},NUM_SHARDS=${NUM_SHARDS}"
EXPORT+=",OLLAMA_MODEL=${OLLAMA_MODEL:-${JUDGE_MODEL}}"
for v in OLLAMA_MODULE PYTHON_MODULE OLLAMA_MODELS OLLAMA_START_HELPER OLLAMA_SIF \
         OLLAMA_CONTEXT_LENGTH MAX_TOKENS WORKERS OLLAMA_WARMUP_SECONDS; do
    [ -n "${!v:-}" ] && EXPORT+=",${v}=${!v}"
done

# --- Optional login-node model pre-pull (compute nodes can't download) ----
if [ "${PREPULL:-0}" = "1" ]; then
    echo ">>> Pre-pulling ${OLLAMA_MODEL:-${JUDGE_MODEL}} on this (login) node ..."
    if command -v module >/dev/null 2>&1 && [ -n "${OLLAMA_MODULE:-}" ]; then
        module load "${OLLAMA_MODULE}" || true
    fi
    export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
    ollama pull "${OLLAMA_MODEL:-${JUDGE_MODEL}}"
    echo ">>> Pre-pull complete (cached at ${OLLAMA_MODELS})"
fi

echo "========================================================"
echo "SOAP LLM-judge submission"
echo "  venv:        ${BTC_VENV}"
echo "  predictions: ${PREDICTIONS}"
echo "  transcripts: ${TRANSCRIPTS}"
echo "  model:       ${JUDGE_MODEL}"
echo "  output:      ${OUTPUT_DIR}"
echo "  shards:      ${NUM_SHARDS}  (array 0-${MAX_TASK}, ~${SAMPLES_PER_SHARD}/shard)"
echo "  cluster:     ${CLUSTER:-<none>}  account: ${ACCOUNT:-<none>}  partition: ${PARTITION:-<none>}"
echo "========================================================"

# --- 1. Judge array -------------------------------------------------------
ARRAY_OUT=$(sbatch \
    "${SBATCH_FLAGS[@]}" \
    --array=0-${MAX_TASK} \
    --export="${EXPORT}" \
    --output="${LOG_DIR}/judge_%A_%a.out" \
    --error="${LOG_DIR}/judge_%A_%a.err" \
    "${HERE}/run_soap_judge.slurm")
echo "${ARRAY_OUT}"
ARRAY_JOB_ID=$(echo "${ARRAY_OUT}" | awk '{print $4}')

# --- 2. Aggregate (afterok the whole array) -------------------------------
AGG_OUT=$(sbatch \
    "${SBATCH_FLAGS[@]}" \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export="ALL,BTC_VENV=${BTC_VENV},OUTPUT_DIR=${OUTPUT_DIR},NUM_SHARDS=${NUM_SHARDS}" \
    --output="${LOG_DIR}/aggregate_%j.out" \
    --error="${LOG_DIR}/aggregate_%j.err" \
    "${HERE}/run_soap_aggregate.slurm")
echo "${AGG_OUT}"
AGG_JOB_ID=$(echo "${AGG_OUT}" | awk '{print $4}')

echo ""
echo "Submitted:"
echo "  judge array: ${ARRAY_JOB_ID}   (tasks 0-${MAX_TASK})"
echo "  aggregate:   ${AGG_JOB_ID}   (afterok:${ARRAY_JOB_ID})"
echo ""
echo "Result:  ${OUTPUT_DIR}/aggregated/summary.json"
echo "Monitor: squeue -u \$USER ${CLUSTER:+-M ${CLUSTER}}  |  tail -f ${LOG_DIR}/judge_${ARRAY_JOB_ID}_*.out"
