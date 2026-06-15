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
# submit_soap_judge.sh splits predictions into fixed-size CHUNK files; each array
# task judges exactly one chunk (CHUNK_DIR/chunk_<i>.jsonl, i = SLURM_ARRAY_TASK_ID)
# -> OUTPUT_DIR/shard_<i>/ (no races; id-keyed raw cache makes re-runs resume-safe).
#
# To survive scale, the task STAGES the model to node-local disk (STAGE_MODEL=1)
# so inference doesn't read 19 GB over shared NFS, and the submit script THROTTLES
# concurrency (--array=...%MAX_CONCURRENT).
#
# DO NOT submit directly — use submit_soap_judge.sh. See slurm/README.md.
#
# Required env (from submit_soap_judge.sh): BTC_VENV TRANSCRIPTS JUDGE_MODEL
#   OUTPUT_DIR CHUNK_DIR NUM_CHUNKS
# Optional: OLLAMA_MODEL($JUDGE_MODEL) OLLAMA_MODULE(ollama) PYTHON_MODULE(python/3.10)
#   OLLAMA_MODELS(~/.ollama/models) OLLAMA_SIF MAX_TOKENS(16000) WORKERS(4)
#   OLLAMA_TIMEOUT(1800) STAGE_MODEL(1) STAGE_DIR($TMPDIR/btc-ollama-...)
#   OLLAMA_CONTEXT_LENGTH(32768) OLLAMA_WARMUP_SECONDS(45)

set -u

OLLAMA_MODEL="${OLLAMA_MODEL:-${JUDGE_MODEL}}"
OLLAMA_MODULE="${OLLAMA_MODULE:-ollama}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
WORKERS="${WORKERS:-4}"
OLLAMA_WARMUP_SECONDS="${OLLAMA_WARMUP_SECONDS:-45}"
STAGE_MODEL="${STAGE_MODEL:-1}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-1800}"   # read by btc-eval's ollama backend
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PREDICTIONS="${CHUNK_DIR:?CHUNK_DIR not set — use submit_soap_judge.sh}/chunk_$(printf '%04d' "${TASK_ID}").jsonl"
SHARD_OUT="${OUTPUT_DIR}/shard_${TASK_ID}"

echo "=========================================="
echo "Job:            ${SLURM_JOB_NAME:-soap-judge}  task ${TASK_ID}/${NUM_CHUNKS:-?}"
echo "Node:           $(hostname)   Started: $(date)"
echo "Judge model:    ${JUDGE_MODEL}   stage=${STAGE_MODEL}"
echo "Predictions:    ${PREDICTIONS}  ($(grep -c . "${PREDICTIONS}" 2>/dev/null || echo 0) dialogs)"
echo "Output:         ${SHARD_OUT}"
echo "=========================================="

for f in "${PREDICTIONS}" "${TRANSCRIPTS}"; do
    [ -f "${f}" ] || { echo "ERROR: input not found: ${f}" >&2; exit 1; }
done
[ -x "${BTC_VENV}/bin/python" ] || { echo "ERROR: btc-eval venv not found: ${BTC_VENV}/bin/python" >&2; exit 1; }
mkdir -p "${SHARD_OUT}"

# --- Modules + writable model cache (OSC module clobbers OLLAMA_MODELS) ----
WANT_OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
if command -v module >/dev/null 2>&1; then
    module purge
    module load "${OLLAMA_MODULE}"
    module load "${PYTHON_MODULE}"
fi
export OLLAMA_MODELS="${WANT_OLLAMA_MODELS}"
export OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-15m}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"   # stripped by OSC --cleanenv; server auto-picks
export PYTHONUNBUFFERED=1
[ -n "${OLLAMA_SIF:-}" ] && { export OLLAMA_IMG="${OLLAMA_SIF}"; echo ">>> Using custom Ollama image: ${OLLAMA_IMG}"; }

# --- Stage the model to NODE-LOCAL disk (avoids shared-NFS contention) -----
# Copies just this model's manifest + blobs, binds the dir into Apptainer, and
# points OLLAMA_MODELS at it. Falls back to the shared cache on any failure.
stage_model() {
    local src="$1" dst="$2" model="$3" ref tag mpath manifest digests d f
    ref="${model%%:*}"; tag="${model##*:}"
    case "${ref}" in
        */*) mpath="registry.ollama.ai/${ref}/${tag}" ;;
        *)   mpath="registry.ollama.ai/library/${ref}/${tag}" ;;
    esac
    manifest="${src}/manifests/${mpath}"
    [ -f "${manifest}" ] || { echo "  stage: manifest not found: ${manifest}" >&2; return 1; }
    mkdir -p "${dst}/manifests/$(dirname "${mpath}")" "${dst}/blobs" || return 1
    cp -f "${manifest}" "${dst}/manifests/${mpath}" || return 1
    digests=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['config']['digest']);[print(l['digest']) for l in m.get('layers',[])]" "${manifest}") || return 1
    for d in ${digests}; do
        f="sha256-${d#sha256:}"
        [ -f "${dst}/blobs/${f}" ] || cp -f "${src}/blobs/${f}" "${dst}/blobs/${f}" || return 1
    done
    return 0
}

if [ "${STAGE_MODEL}" = "1" ]; then
    STAGE_DIR="${STAGE_DIR:-${TMPDIR:-/tmp}/btc-ollama-${SLURM_ARRAY_JOB_ID:-x}-${TASK_ID}}"
    echo ">>> Staging ${OLLAMA_MODEL}: ${OLLAMA_MODELS} -> ${STAGE_DIR}"
    t0=$(date +%s)
    if stage_model "${OLLAMA_MODELS}" "${STAGE_DIR}" "${OLLAMA_MODEL}"; then
        # Make sure Apptainer (the OSC helper) bind-mounts the node-local dir.
        export APPTAINER_BIND="${STAGE_DIR}${APPTAINER_BIND:+,${APPTAINER_BIND}}"
        export OLLAMA_MODELS="${STAGE_DIR}"
        echo ">>> Staged in $(( $(date +%s) - t0 ))s -> OLLAMA_MODELS=${OLLAMA_MODELS}"
    else
        echo ">>> WARNING: staging failed; using shared cache ${OLLAMA_MODELS}" >&2
        STAGE_DIR=""
    fi
fi

# --- Start a per-task Ollama server ---------------------------------------
if command -v ollama_start_helper >/dev/null 2>&1 || [ -f "${OLLAMA_START_HELPER:-/nonexistent}" ]; then
    source "${OLLAMA_START_HELPER:-ollama_start_helper}"
    sleep "${OLLAMA_WARMUP_SECONDS}"
    echo ">>> Pulling model: ${OLLAMA_MODEL}"
    ollama_pull "${OLLAMA_MODEL}"
    export OLLAMA_HOST="http://localhost:${OLLAMA_PORT}"
else
    export OLLAMA_PORT="${OLLAMA_PORT:-11434}"
    export OLLAMA_HOST="http://localhost:${OLLAMA_PORT}"
    ollama serve > "${SHARD_OUT}/ollama_serve.log" 2>&1 &
    sleep "${OLLAMA_WARMUP_SECONDS}"
    ollama pull "${OLLAMA_MODEL}"
fi
export OLLAMA_SERVER_URL="${OLLAMA_HOST}"

echo ">>> Pre-warming ${OLLAMA_MODEL} ..."
curl -sf "${OLLAMA_SERVER_URL}/api/chat" -H "Content-Type: application/json" \
    -d '{"model":"'"${OLLAMA_MODEL}"'","messages":[{"role":"user","content":"hi"}],"stream":false}' >/dev/null 2>&1 || true
echo ">>> Ollama ready at ${OLLAMA_SERVER_URL}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
nvidia-smi dmon -s um -d 10 -o DT > "${SHARD_OUT}/gpu_util.log" 2>/dev/null &
NVMON_PID=$!

# --- Judge this chunk (the chunk file IS the work; no --shard) -------------
echo ">>> soap-judge chunk ${TASK_ID} ($(grep -c . "${PREDICTIONS}") dialogs)"
"${BTC_VENV}/bin/python" -m btc_eval.cli soap-judge \
    --predictions "${PREDICTIONS}" \
    --transcripts "${TRANSCRIPTS}" \
    --backend ollama \
    --model "${JUDGE_MODEL}" \
    --host "${OLLAMA_SERVER_URL}" \
    --workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --cache-dir "${OUTPUT_DIR}/cache" \
    --output "${SHARD_OUT}"
EXIT_STATUS=$?

kill "${NVMON_PID}" 2>/dev/null
[ "${STAGE_MODEL}" = "1" ] && [ -n "${STAGE_DIR:-}" ] && rm -rf "${STAGE_DIR}" 2>/dev/null || true

echo ""
if [ ${EXIT_STATUS} -eq 0 ]; then
    N=$(wc -l < "${SHARD_OUT}/soap_judge_per_dialog.jsonl" 2>/dev/null || echo 0)
    echo "Task ${TASK_ID} SUCCESS: ${N} judged -> ${SHARD_OUT}/"
else
    echo "Task ${TASK_ID} FAILED (exit ${EXIT_STATUS}); re-run is resume-safe."
fi
echo "Finished at: $(date)"
exit ${EXIT_STATUS}
