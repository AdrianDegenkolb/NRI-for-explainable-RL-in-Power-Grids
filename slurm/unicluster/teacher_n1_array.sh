#!/bin/bash

# N-1 Teacher action-space generation — SLURM array sharded one chronic per task — BWUniCluster.
#
# collect_n_minus_1_experience's own parallelism is a local multiprocessing.Pool (one node's
# core count); this shards at the chronic level instead (src/action_space_generation/
# teacher_runner.py:run_teacher_single_chronic), so chronics run across many array tasks/nodes.
# Each task appends its own shard CSV; aggregate them afterwards with:
#   python experiments/build_action_space.py aggregate --grid <grid> \
#       --experience "results/teacher/<experiment>/shard_*.csv" --top-k <k> --out <path>
#
# IMPORTANT: --time below is a placeholder. Run slurm/unicluster/benchmark_teacher_cost.sh first
# and size walltime off its p90, combined with the interact-rate estimate from existing
# training logs (see the Obsidian note "Teacher Action Space Reduction for IEEE-36 and
# IEEE-118") — budget for the tail (p90/p99 episode length), not the mean, since a killed task
# truncates towards the start of its chronic and biases the resulting action set. BWUniCluster's
# `cpu_il,cpu` partitions cap walltime at 72h; split a chronic across resubmissions if a single
# task would need longer (see slurm/horeka/resubmit_failed_seeds.sh for a retry-array pattern
# used elsewhere in this repo).
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/unicluster/teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <lines_to_attack...>
#
# Arguments:
#   grid              case36 | case118
#   experiment_name   Results sub-directory, e.g. 2026_09_20_teacher_n1_case36
#   max_concurrent    SLURM array throttle (--array=0-N%max_concurrent)
#   lines_to_attack   Line ids from: python experiments/build_action_space.py select-lines --grid <grid>
#
# Example:
#   slurm/unicluster/teacher_n1_array.sh case36 2026_09_20_teacher_n1_case36 50 3 12 27 41 58 60 71 88 95 101

GRID="${1:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <lines_to_attack...>}"
EXPERIMENT="${2:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <lines_to_attack...>}"
MAX_CONCURRENT="${3:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <lines_to_attack...>}"
shift 3
LINES_TO_ATTACK=("$@")
if [[ ${#LINES_TO_ATTACK[@]} -eq 0 ]]; then
    echo "Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <lines_to_attack...>" >&2
    exit 1
fi

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

OUT_DIR="results/teacher/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

# List chronics on the login node (cheap — just reads chronic subpaths, no simulation) so the
# array size is known before submission.
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate curriculum
PYTHONPATH="$(pwd)/src" python experiments/build_action_space.py list-chronics --grid "${GRID}" \
    > "${OUT_DIR}/chronic_ids.txt"
N_CHRONICS=$(wc -l < "${OUT_DIR}/chronic_ids.txt")
if [[ "$N_CHRONICS" -eq 0 ]]; then
    echo "No chronics listed for grid ${GRID} — aborting." >&2
    exit 1
fi
echo "Sharding ${N_CHRONICS} chronics across a ${N_CHRONICS}-task array (max ${MAX_CONCURRENT} concurrent)."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=teacher_n1_${GRID}
#SBATCH --output=${OUT_DIR}/shard_%a.%j.log
#SBATCH --error=${OUT_DIR}/shard_%a.%j.err
#SBATCH --array=0-$((N_CHRONICS - 1))%${MAX_CONCURRENT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate curriculum

CHRONIC_ID=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${OUT_DIR}/chronic_ids.txt")

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID (array task \$SLURM_ARRAY_TASK_ID)"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Chronic:      \$CHRONIC_ID"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/build_action_space.py run-teacher-chronic \
    --grid "${GRID}" \
    --lines-to-attack ${LINES_TO_ATTACK[*]} \
    --chronic-id "\$CHRONIC_ID" \
    --save-path "${OUT_DIR}/shard_\${SLURM_ARRAY_TASK_ID}.csv"
EOF
