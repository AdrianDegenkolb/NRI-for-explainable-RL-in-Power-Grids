#!/bin/bash

# Phase 0: benchmark N-1 Teacher cost-per-interaction on one grid.
#
# Runs experiments/benchmark_teacher_cost.py for a handful of real teacher interactions
# (default 25) to measure wall-clock cost of the N-1-search and greedy-fallback branches.
# Combine the output with the interact-rate estimates pulled from existing training logs
# (see the Obsidian note "Teacher Action Space Reduction for IEEE-36 and IEEE-118") to size
# the SLURM array walltime for the full teacher run (slurm/horeka/teacher_n1_array.sh).
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/horeka/benchmark_teacher_cost.sh <env_name> <experiment_name> <lines_to_attack...>
#
# Example:
#   slurm/horeka/benchmark_teacher_cost.sh l2rpn_wcci_2020_train phase0_case36 3 12 27 41 58

ENV_NAME="${1:?Usage: benchmark_teacher_cost.sh <env_name> <experiment_name> <lines_to_attack...>}"
EXPERIMENT="${2:?Usage: benchmark_teacher_cost.sh <env_name> <experiment_name> <lines_to_attack...>}"
shift 2
LINES_TO_ATTACK=("$@")
if [[ ${#LINES_TO_ATTACK[@]} -eq 0 ]]; then
    echo "Usage: benchmark_teacher_cost.sh <env_name> <experiment_name> <lines_to_attack...>" >&2
    exit 1
fi

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

OUT_DIR="results/teacher/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=benchmark_teacher_cost
#SBATCH --output=${OUT_DIR}/benchmark.%j.log
#SBATCH --error=${OUT_DIR}/benchmark.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-pai00074

source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate curriculum

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$(hostname)"
echo "Env:          ${ENV_NAME}"
echo "Lines:        ${LINES_TO_ATTACK[*]}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/benchmark_teacher_cost.py \
    --env-name "${ENV_NAME}" \
    --lines-to-attack ${LINES_TO_ATTACK[*]} \
    --n-interactions 25 \
    --out "${OUT_DIR}/benchmark_result.json"
EOF
