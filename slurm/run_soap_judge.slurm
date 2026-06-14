#!/bin/bash
#SBATCH --job-name=soap-judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Reusable, cluster-agnostic SLURM array template for the SOAP LLM-judge.
#
# Each array task starts its own Ollama instance, pre-warms the model, and judges
# ONE shard of the predictions via `btc-eval soap-judge --shard i/N` (i =
# SLURM_ARRAY_TASK_ID, N = NUM_SHARDS). Per-shard outputs never race, and the raw
# response cache makes a failed shard cheap to re-run.
#
# DO NOT submit directly — use submit_soap_judge.sh, which sizes the --array,
# adds the cluster flags (--cluster/--account/--partition), and chains the
# aggregate job. Cluster-specific bits come in as environment variables so this
# file stays portable; see slurm/README.md.
#
# Required env (exported by the submit script via --export=):
#   BTC_VENV        Path to the venv with btc-eval installed (…/.venv)
#   PREDICTIONS     Predictions JSONL  ({"id","summary"})
#   TRANSCRIPTS     Transcripts JSONL  ({"id","transcript"})
#   JUDGE_MODEL     Model id for the backend           (e.g. gemma4:31b)
#   OUTPUT_DIR      Per-run output root (shards land in $OUTPUT_DIR/shard_<i>/)
#   NUM_SHARDS      Total array size N (must equal the sbatch --array width)
# Optional env (sensible defaults below):
#   OLLAMA_MODEL    Ollama tag to pull/pre-warm        (default: $JUDGE_MODEL)
#   OLLAMA_MODULE   Module to load for Ollama          (default: ollama)
#   PYTHON_MODULE   Module to load for Python          (default: python/3.10)
#   OLLAMA_MODELS   Shared model cache dir             (default: ~/.ollama/models)
#   MAX_TOKENS      Max output tokens per LLM call     (default: 16000)
#   WORKERS         Parallel workers within the task   (default: 4)
#   OLLAMA_CONTEXT_LENGTH   Context window             (default: 16384)
#   OLLAMA_WARMUP_SECONDS   Wait after `ollama serve`  (default: 45)

set -u

OLLAMA_MODEL="${OLLAMA_MODEL:-${JUDGE_MODEL}}"
OLLAMA_MODULE="${OLLAMA_MODULE:-ollama}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
WORKERS="${WORKERS:-4}"
OLLAMA_WARMUP_SECONDS="${OLLAMA_WARMUP_SECONDS:-45}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_OUT="${OUTPUT_DIR}/shard_${TASK_ID}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job:            ${SLURM_JOB_NAME:-soap-judge}"
echo "Array job ID:   ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Shard:          ${TASK_ID}/${NUM_SHARDS}"
echo "Node:           $(hostname)"
echo "Started at:     $(date)"
echo "Judge model:    ${JUDGE_MODEL}  (ollama tag: ${OLLAMA_MODEL})"
echo "Predictions:    ${PREDICTIONS}"
echo "Transcripts:    ${TRANSCRIPTS}"
echo "Shard output:   ${SHARD_OUT}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
for f in "${PREDICTIONS}" "${TRANSCRIPTS}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: input not found: ${f}" >&2
        exit 1
    fi
done
if [ ! -x "${BTC_VENV}/bin/python" ]; then
    echo "ERROR: btc-eval venv not found: ${BTC_VENV}/bin/python" >&2
    echo "       (create it with: cd <betrac-metrics-private> && make setup && uv pip install '.[llm-judge]')" >&2
    exit 1
fi

mkdir -p "${SHARD_OUT}"

# ---------------------------------------------------------------------------
# Load modules and start a per-task Ollama instance
# ---------------------------------------------------------------------------
# Capture the desired (writable) model cache before 'module load' — the OSC
# ollama module overrides OLLAMA_MODELS with a read-only shared path.
WANT_OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"

# `module` may be a function on login shells but absent in batch; guard it.
if command -v module >/dev/null 2>&1; then
    module purge
    # Order matters on some clusters: ollama before python.
    module load "${OLLAMA_MODULE}"
    module load "${PYTHON_MODULE}"
fi

# Re-apply the writable cache the module just clobbered.
export OLLAMA_MODELS="${WANT_OLLAMA_MODELS}"
export OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-10m}"
# Judge stage needs ~20k tokens (transcript + note + extracted claims + output).
# NOTE: OSC's ollama_start_helper uses `apptainer run --cleanenv`, which STRIPS
# this var — the server then auto-picks a context from VRAM (32k on A100-40GB).
# This value only takes effect on clusters whose server inherits the env.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
export PYTHONUNBUFFERED=1

# Use a newer Ollama image if provided (overrides the module's $OLLAMA_IMG); the
# helper below reads $OLLAMA_IMG at runtime. See eval2026/judge/setup_ollama_image.sh.
if [ -n "${OLLAMA_SIF:-}" ]; then
    export OLLAMA_IMG="${OLLAMA_SIF}"
    echo ">>> Using custom Ollama image: ${OLLAMA_IMG}"
fi

# ollama_start_helper (provided by the OSC ollama module) assigns a unique
# OLLAMA_PORT per task and starts `ollama serve` in the background, and provides
# ollama_pull(). On clusters without it, start your own server here (see README).
if command -v ollama_start_helper >/dev/null 2>&1 || \
   [ -f "${OLLAMA_START_HELPER:-/nonexistent}" ]; then
    source "${OLLAMA_START_HELPER:-ollama_start_helper}"
    sleep "${OLLAMA_WARMUP_SECONDS}"
    echo ">>> Pulling model: ${OLLAMA_MODEL}"
    ollama_pull "${OLLAMA_MODEL}"
    export OLLAMA_HOST="http://localhost:${OLLAMA_PORT}"
else
    # Fallback: single shared server on the default port.
    export OLLAMA_PORT="${OLLAMA_PORT:-11434}"
    export OLLAMA_HOST="http://localhost:${OLLAMA_PORT}"
    ollama serve > "${SHARD_OUT}/ollama_serve.log" 2>&1 &
    sleep "${OLLAMA_WARMUP_SECONDS}"
    echo ">>> Pulling model: ${OLLAMA_MODEL}"
    ollama pull "${OLLAMA_MODEL}"
fi
export OLLAMA_SERVER_URL="${OLLAMA_HOST}"

# Pre-warm: force the model into GPU VRAM before timed inference.
echo ">>> Pre-warming ${OLLAMA_MODEL} ..."
curl -sf "${OLLAMA_SERVER_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${OLLAMA_MODEL}"'","messages":[{"role":"user","content":"hi"}],"stream":false}' \
    > /dev/null 2>&1 || true
echo ">>> Ollama ready at ${OLLAMA_SERVER_URL}"

# ---------------------------------------------------------------------------
# GPU info + background utilization log
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
nvidia-smi dmon -s um -d 10 -o DT > "${SHARD_OUT}/gpu_util.log" 2>/dev/null &
NVMON_PID=$!

# ---------------------------------------------------------------------------
# Judge this shard
# ---------------------------------------------------------------------------
echo ">>> soap-judge shard ${TASK_ID}/${NUM_SHARDS}"
"${BTC_VENV}/bin/python" -m btc_eval.cli soap-judge \
    --predictions "${PREDICTIONS}" \
    --transcripts "${TRANSCRIPTS}" \
    --backend ollama \
    --model "${JUDGE_MODEL}" \
    --host "${OLLAMA_SERVER_URL}" \
    --shard "${TASK_ID}/${NUM_SHARDS}" \
    --workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --output "${SHARD_OUT}"
EXIT_STATUS=$?

kill "${NVMON_PID}" 2>/dev/null

echo ""
echo "=========================================="
if [ ${EXIT_STATUS} -eq 0 ]; then
    N=$(wc -l < "${SHARD_OUT}/soap_judge_per_dialog.jsonl" 2>/dev/null || echo 0)
    echo "Shard ${TASK_ID} SUCCESS: ${N} judged -> ${SHARD_OUT}/"
else
    echo "Shard ${TASK_ID} FAILED (exit ${EXIT_STATUS}); re-run is resume-safe (cached raws kept)."
fi
echo "Finished at: $(date)"
echo "=========================================="
exit ${EXIT_STATUS}
