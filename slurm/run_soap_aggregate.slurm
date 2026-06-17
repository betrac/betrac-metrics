#!/bin/bash
#SBATCH --job-name=soap-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Reusable combine step: pool the per-shard per-dialog rows and RECOMPUTE the
# aggregate (the correct way to merge shards — not averaging per-shard means).
# Runs on CPU only; chained after the judge array with --dependency=afterany
# (always runs, pooling whatever per-dialog rows exist — afterok would yield
# nothing if any array task failed/timed out).
#
# DO NOT submit directly — use submit_soap_judge.sh.
#
# Required env (exported by the submit script):
#   BTC_VENV     Path to the venv with btc-eval installed
#   OUTPUT_DIR   Same run root passed to the array job (shards in $OUTPUT_DIR/shard_*/)
#   NUM_SHARDS   Number of shards (for the completeness check)

set -u

AGG_OUT="${OUTPUT_DIR}/aggregated"

echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME:-soap-aggregate}"
echo "Run root:   ${OUTPUT_DIR}"
echo "Shards:     ${NUM_SHARDS}"
echo "Aggregate:  ${AGG_OUT}"
echo "Started at: $(date)"
echo "=========================================="

# Warn (don't fail) on any shard that produced no per-dialog rows — soap-aggregate
# pools whatever it finds, so a missing shard would silently shrink coverage.
MISSING=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ ! -s "${OUTPUT_DIR}/shard_${i}/soap_judge_per_dialog.jsonl" ]; then
        echo "  WARNING: shard ${i} has no per-dialog output (${OUTPUT_DIR}/shard_${i}/)" >&2
        MISSING=$((MISSING + 1))
    fi
done
[ "${MISSING}" -gt 0 ] && echo "  ${MISSING}/${NUM_SHARDS} shards missing — aggregate will undercount; re-run them." >&2

"${BTC_VENV}/bin/python" -m btc_eval.cli soap-aggregate \
    --inputs "${OUTPUT_DIR}/shard_*/soap_judge_per_dialog.jsonl" \
    --output "${AGG_OUT}"
EXIT_STATUS=$?

echo ""
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "Aggregate written -> ${AGG_OUT}/summary.json"
else
    echo "Aggregate FAILED (exit ${EXIT_STATUS})" >&2
fi
echo "Finished at: $(date)"
exit ${EXIT_STATUS}
