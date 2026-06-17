#!/bin/bash
#SBATCH --job-name=soap-judge-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# vLLM variant of the SOAP-judge array task (cluster-agnostic).
#
# Same orchestration as run_soap_judge.slurm (submit_soap_judge.sh splits the
# predictions into chunk files; this task judges one chunk), but it serves the
# model with vLLM instead of Ollama. vLLM does continuous batching, so the win
# comes from feeding it MANY concurrent requests: use BIG chunks
# (SAMPLES_PER_SHARD ~24-100) and high WORKERS, the opposite of the Ollama path.
#
# DO NOT submit directly — use submit_soap_judge.sh with JUDGE_SCRIPT pointed
# here. See eval2026/judge/VLLM_SETUP.md for the full, step-by-step setup.
#
# Required env (from submit_soap_judge.sh): BTC_VENV TRANSCRIPTS JUDGE_MODEL
#   OUTPUT_DIR CHUNK_DIR NUM_CHUNKS
# vLLM-specific:
#   VLLM_SIF        Apptainer image with vLLM (docker://vllm/vllm-openai). REQUIRED.
#   VLLM_MODEL_DIR  Local/shared HF model dir to serve (e.g. the downloaded
#                   gemma-4-31B-it-qat-w4a16-ct). REQUIRED.
#   JUDGE_MODEL     The --served-model-name clients use (e.g. gemma-4-31B-w4a16).
# Optional (defaults):
#   VLLM_MAX_LEN(24000)  VLLM_GPU_UTIL(0.92)  VLLM_TP(1)  VLLM_KV_CACHE_DTYPE(auto)
#   VLLM_EXTRA_ARGS()    VLLM_READY_TIMEOUT(900)  WORKERS(24)  MAX_TOKENS(16000)
#   OLLAMA_TIMEOUT(1800) STAGE_MODEL(1)  STAGE_DIR($TMPDIR/btc-vllm-...)
#   PYTHON_MODULE()      APPTAINER_MODULE()
#   VLLM_CACHE_DIR(persistent big-quota dir recommended; else node-local $TMPDIR)

set -u

JUDGE_MODEL="${JUDGE_MODEL:?set JUDGE_MODEL (the --served-model-name)}"
VLLM_SIF="${VLLM_SIF:?set VLLM_SIF to the vLLM Apptainer image}"
VLLM_MODEL_DIR="${VLLM_MODEL_DIR:?set VLLM_MODEL_DIR to the HF model directory}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"   # judge prompt (~11-17k) + judge output must fit
VLLM_JUDGE_MAX_TOKENS="${VLLM_JUDGE_MAX_TOKENS:-8000}"  # judgment is ~5k; keeps prompt+max under max-len
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
WORKERS="${WORKERS:-24}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
STAGE_MODEL="${STAGE_MODEL:-1}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-1800}"   # the vllm backend reuses this
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PREDICTIONS="${CHUNK_DIR:?CHUNK_DIR not set — use submit_soap_judge.sh}/chunk_$(printf '%04d' "${TASK_ID}").jsonl"
SHARD_OUT="${OUTPUT_DIR}/shard_${TASK_ID}"

echo "=========================================="
echo "Job:          ${SLURM_JOB_NAME:-soap-judge-vllm}  task ${TASK_ID}/${NUM_CHUNKS:-?}"
echo "Node:         $(hostname)   Started: $(date)"
echo "Model:        ${JUDGE_MODEL}  (weights: ${VLLM_MODEL_DIR})"
echo "vLLM:         tp=${VLLM_TP} max_len=${VLLM_MAX_LEN} gpu_util=${VLLM_GPU_UTIL} workers=${WORKERS}"
echo "Predictions:  ${PREDICTIONS}  ($(grep -c . "${PREDICTIONS}" 2>/dev/null || echo 0) dialogs)"
echo "=========================================="

for f in "${PREDICTIONS}" "${TRANSCRIPTS}"; do
    [ -f "${f}" ] || { echo "ERROR: input not found: ${f}" >&2; exit 1; }
done
[ -x "${BTC_VENV}/bin/python" ] || { echo "ERROR: btc-eval venv not found: ${BTC_VENV}/bin/python" >&2; exit 1; }
[ -f "${VLLM_SIF}" ] || { echo "ERROR: vLLM image not found: ${VLLM_SIF}" >&2; exit 1; }
[ -d "${VLLM_MODEL_DIR}" ] || { echo "ERROR: model dir not found: ${VLLM_MODEL_DIR}" >&2; exit 1; }
mkdir -p "${SHARD_OUT}"

# --- Optional modules (apptainer is usually system-wide on OSC) ------------
if command -v module >/dev/null 2>&1; then
    [ -n "${APPTAINER_MODULE:-}" ] && module load "${APPTAINER_MODULE}" || true
    [ -n "${PYTHON_MODULE:-}" ] && module load "${PYTHON_MODULE}" || true
fi
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not on PATH" >&2; exit 1; }

# --- Stage the HF model dir to NODE-LOCAL disk (avoid shared-NFS reads) -----
SERVE_PATH="${VLLM_MODEL_DIR}"
if [ "${STAGE_MODEL}" = "1" ]; then
    STAGE_DIR="${STAGE_DIR:-${TMPDIR:-/tmp}/btc-vllm-${SLURM_ARRAY_JOB_ID:-x}-${TASK_ID}}"
    echo ">>> Staging model dir -> ${STAGE_DIR}"
    t0=$(date +%s)
    if mkdir -p "${STAGE_DIR}" && cp -r "${VLLM_MODEL_DIR}/." "${STAGE_DIR}/"; then
        SERVE_PATH="${STAGE_DIR}"
        echo ">>> Staged in $(( $(date +%s) - t0 ))s"
    else
        echo ">>> WARNING: staging failed; serving from shared ${VLLM_MODEL_DIR}" >&2
        STAGE_DIR=""
    fi
fi

# --- Start the vLLM OpenAI server (in the Apptainer container) -------------
# Pick a free-ish random port so multiple tasks can share a node.
PORT=$(( (RANDOM % 16000) + 49152 ))
SERVER_URL="http://127.0.0.1:${PORT}"
echo ">>> Starting vLLM on ${SERVER_URL} ..."
# --cleanenv keeps the host environment (the OSC Intel-compiler module's icc on
# PATH/CC, plus the --export=ALL job env) from leaking into the container and
# breaking torch.compile. Forward only the GPU selection + a sane compiler.
VLLM_ENVS="CC=gcc,CXX=g++"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && VLLM_ENVS="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},${VLLM_ENVS}"
# vLLM requires an explicit chat template when the tokenizer doesn't embed one;
# Gemma-4 ships it as a separate chat_template.jinja (staged alongside the model).
CHAT_TEMPLATE_ARG=""
[ -f "${SERVE_PATH}/chat_template.jinja" ] && CHAT_TEMPLATE_ARG="--chat-template ${SERVE_PATH}/chat_template.jinja"
# torch.compile / triton / inductor write a sizable cache. vLLM's default lives
# under $HOME/.cache, which is frequently OVER QUOTA on HPC homes ("Disk quota
# exceeded"). Resolve a writable cache dir WITHOUT assuming any site-specific
# scratch path (there is no portable ~/scratch):
#   1) explicit VLLM_CACHE_DIR -> a PERSISTENT big-quota dir compiles ONCE and
#      reuses across jobs (set it directly, or BTC_WORK in run_eval_judge.sh).
#   2) else node-local $TMPDIR -> always writable, but vLLM recompiles each job.
# Either way we redirect ALL cache roots there and point HOME at it too, to catch
# XDG-unaware writers.
if [ -z "${VLLM_CACHE_DIR:-}" ]; then
    VLLM_CACHE_DIR="${TMPDIR:-/tmp}/btc-vllm-cache-${USER:-x}"
    echo ">>> NOTE: VLLM_CACHE_DIR unset -> node-local ${VLLM_CACHE_DIR} (vLLM recompiles" >&2
    echo ">>>       each job). Set VLLM_CACHE_DIR, or BTC_WORK in run_eval_judge.sh, to a" >&2
    echo ">>>       persistent big-quota dir to compile once and reuse." >&2
fi
mkdir -p "${VLLM_CACHE_DIR}"
VLLM_CACHE_REAL="$(readlink -f "${VLLM_CACHE_DIR}")"
VLLM_ENVS="${VLLM_ENVS},HOME=${VLLM_CACHE_REAL},XDG_CACHE_HOME=${VLLM_CACHE_REAL}/.cache"
VLLM_ENVS="${VLLM_ENVS},TRITON_CACHE_DIR=${VLLM_CACHE_REAL}/triton,TORCHINDUCTOR_CACHE_DIR=${VLLM_CACHE_REAL}/inductor"
VLLM_ENVS="${VLLM_ENVS},VLLM_CACHE_ROOT=${VLLM_CACHE_REAL}/vllm,HF_HOME=${VLLM_CACHE_REAL}/hf"
# --nv exposes the GPU; bind the stage dir + the cache dir so the container can read/write them.
apptainer exec --nv --cleanenv --env "${VLLM_ENVS}" \
    --bind "${VLLM_CACHE_REAL}" ${SERVE_PATH:+--bind "${SERVE_PATH}"} "${VLLM_SIF}" \
    vllm serve "${SERVE_PATH}" \
        --served-model-name "${JUDGE_MODEL}" \
        --host 127.0.0.1 --port "${PORT}" \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --max-model-len "${VLLM_MAX_LEN}" \
        --tensor-parallel-size "${VLLM_TP}" \
        ${CHAT_TEMPLATE_ARG} \
        ${VLLM_KV_CACHE_DTYPE:+--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}"} \
        ${VLLM_EXTRA_ARGS:-} \
        > "${SHARD_OUT}/vllm_serve.log" 2>&1 &
VLLM_PID=$!

# --- Wait until the server is ready (vLLM loads + compiles; can take minutes) -
ready=0
for _ in $(seq 1 $(( VLLM_READY_TIMEOUT / 5 )) ); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited during startup — see ${SHARD_OUT}/vllm_serve.log" >&2
        tail -30 "${SHARD_OUT}/vllm_serve.log" >&2; exit 1
    fi
    if curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
done
[ "${ready}" = "1" ] || { echo "ERROR: vLLM not ready after ${VLLM_READY_TIMEOUT}s" >&2; tail -30 "${SHARD_OUT}/vllm_serve.log" >&2; kill "${VLLM_PID}" 2>/dev/null; exit 1; }
echo ">>> vLLM ready at ${SERVER_URL}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

# --- Judge this chunk (vLLM batches across --workers) ----------------------
echo ">>> soap-judge chunk ${TASK_ID} ($(grep -c . "${PREDICTIONS}") dialogs) via vLLM"
"${BTC_VENV}/bin/python" -m btc_eval.cli soap-judge \
    --predictions "${PREDICTIONS}" \
    --transcripts "${TRANSCRIPTS}" \
    --backend vllm \
    --model "${JUDGE_MODEL}" \
    --host "${SERVER_URL}" \
    --workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --judge-max-tokens "${VLLM_JUDGE_MAX_TOKENS}" \
    --cache-dir "${OUTPUT_DIR}/cache" \
    --output "${SHARD_OUT}"
EXIT_STATUS=$?

# --- Cleanup ---------------------------------------------------------------
kill "${VLLM_PID}" 2>/dev/null
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
