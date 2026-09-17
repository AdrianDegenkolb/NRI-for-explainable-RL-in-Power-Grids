#!/bin/bash

# N-1 Teacher smoke test — ONE non-array job on ONE chronic — HoreKa.
#
# run-teacher-chronics's underlying path (env.set_id(chronics_id) -> NMinusOneTeacher.
# n_minus_one_agent) is not exercised by slurm/horeka/benchmark_teacher_cost.sh, which uses a
# separate hand-rolled loop. Run this once — and check the resulting shard CSV has sane rows —
# before launching slurm/horeka/teacher_n1_array.sh's full (and much longer) array.
#
# Note: "smoke testing" by setting max_concurrent=1 on the full array script does NOT work —
# that still creates one array task per chronic-batch and, before chunking was added, exceeded
# BWUniCluster's SLURM MaxArraySize outright ("Invalid job array specification"). This script
# submits a genuine single job instead.
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/horeka/teacher_n1_smoketest.sh <grid> <experiment_name> <lines_to_attack...>
#
# Example:
#   slurm/horeka/teacher_n1_smoketest.sh case36 2026_09_17_smoketest 20 13 39 33 35 32 11 34 23 22

GRID="${1:?Usage: teacher_n1_smoketest.sh <grid> <experiment_name> <lines_to_attack...>}"
EXPERIMENT="${2:?Usage: teacher_n1_smoketest.sh <grid> <experiment_name> <lines_to_attack...>}"
shift 2
LINES_TO_ATTACK=("$@")
if [[ ${#LINES_TO_ATTACK[@]} -eq 0 ]]; then
    echo "Usage: teacher_n1_smoketest.sh <grid> <experiment_name> <lines_to_attack...>" >&2
    exit 1
fi

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

OUT_DIR="results/teacher/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

# Pick just the first chronic — this is only checking that the pipeline runs and writes sane
# output, not measuring anything about chronic difficulty.
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate curriculum
CHRONIC_ID=$(PYTHONPATH="$(pwd)/src" python experiments/build_action_space.py list-chronics --grid "${GRID}" | head -n 1)
if [[ -z "$CHRONIC_ID" ]]; then
    echo "Could not list any chronics for grid ${GRID} — aborting." >&2
    exit 1
fi
echo "Smoke-testing chronic: ${CHRONIC_ID}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=teacher_n1_smoketest_${GRID}
#SBATCH --output=${OUT_DIR}/smoketest.%j.log
#SBATCH --error=${OUT_DIR}/smoketest.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-pai00074

source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate curriculum

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Chronic:      ${CHRONIC_ID}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/build_action_space.py run-teacher-chronic \
    --grid "${GRID}" \
    --lines-to-attack ${LINES_TO_ATTACK[*]} \
    --chronic-id "${CHRONIC_ID}" \
    --save-path "${OUT_DIR}/shard_smoketest.csv"

echo "========================================"
echo "Result:"
cat "${OUT_DIR}/shard_smoketest.csv" 2>/dev/null || echo "(no shard_smoketest.csv written — check the log above)"
EOF
