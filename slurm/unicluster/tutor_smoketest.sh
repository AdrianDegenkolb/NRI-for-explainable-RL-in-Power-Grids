#!/bin/bash

# GeneralTutor smoke test — ONE non-array job on ONE chronic — BWUniCluster.
#
# Checks that run-tutor-chronics writes a sane imitation-data .npy and measures the per-chronic
# cost needed to size slurm/unicluster/tutor_array.sh's walltime. The log ends with the
# package's "game over at step-X" line: X equal to the chronic length means the Tutor survived.
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/unicluster/tutor_smoketest.sh <grid> <experiment_name> <action_space_npy>
#
# Example:
#   slurm/unicluster/tutor_smoketest.sh case36 2026_10_07_tutor_smoketest_case36_k208 \
#       data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.npy

GRID="${1:?Usage: tutor_smoketest.sh <grid> <experiment_name> <action_space_npy>}"
EXPERIMENT="${2:?Usage: tutor_smoketest.sh <grid> <experiment_name> <action_space_npy>}"
ACTION_SPACE="${3:?Usage: tutor_smoketest.sh <grid> <experiment_name> <action_space_npy>}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
if [[ ! -f "$ACTION_SPACE" ]]; then
    echo "Action space ${ACTION_SPACE} not found (create it with build_action_space.py to-npy)." >&2
    exit 1
fi

OUT_DIR="results/tutor/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate curriculum
CHRONIC_ID=$(PYTHONPATH="$(pwd)/src" python experiments/build_action_space.py list-chronics --grid "${GRID}" | head -n 1)
if [[ -z "$CHRONIC_ID" ]]; then
    echo "Could not list any chronics for grid ${GRID} — aborting." >&2
    exit 1
fi
echo "Smoke-testing chronic: ${CHRONIC_ID}"

# The Tutor only searches when rho >= 0.95 (~2.4 s per search over 208 actions on IEEE-36), so a
# chronic should take well under the Teacher's ~11 h. Override for bigger grids / action sets.
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tutor_smoketest_${GRID}
#SBATCH --output=${OUT_DIR}/smoketest.%j.log
#SBATCH --error=${OUT_DIR}/smoketest.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=${SBATCH_TIME}
#SBATCH --mem=16G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate curriculum

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Action space: ${ACTION_SPACE}"
echo "Chronic:      ${CHRONIC_ID}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/build_action_space.py run-tutor-chronics \
    --grid "${GRID}" \
    --action-space "${ACTION_SPACE}" \
    --chronic-ids "${CHRONIC_ID}" \
    --save-dir "${OUT_DIR}/experience"
EOF
