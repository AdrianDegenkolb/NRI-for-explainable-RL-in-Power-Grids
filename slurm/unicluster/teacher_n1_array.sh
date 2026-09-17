#!/bin/bash

# N-1 Teacher action-space generation — SLURM array, chunked over chronics — BWUniCluster.
#
# collect_n_minus_1_experience's own parallelism is a local multiprocessing.Pool (one node's
# core count); this shards across many array tasks instead (src/action_space_generation/
# teacher_runner.py:run_teacher_chronics). Chronic counts can run into the thousands (e.g. 2592
# for l2rpn_wcci_2020_train), which exceeds a typical cluster's SLURM MaxArraySize if sharded
# one-chronic-per-task (confirmed: BWUniCluster rejected sbatch outright for a 2592-wide array
# with "Invalid job array specification") — so chronics are chunked into batches, each batch
# handled sequentially by one array task, all appending to the same shard CSV.
#
# Run slurm/unicluster/teacher_n1_smoketest.sh first to validate the pipeline on one chronic —
# run-teacher-chronics's underlying env.set_id()/n_minus_one_agent() path is not exercised by
# the Phase 0 benchmark script, so it should be checked once before committing to the full array.
#
# Aggregate shards afterwards with:
#   python experiments/build_action_space.py aggregate --grid <grid> \
#       --experience "results/teacher/<experiment>/shard_*.csv" --top-k <k> --out <path>
#
# IMPORTANT: --time=48h below assumes the worst-case per-chronic cost measured for IEEE-36
# Phase 0 (~7.3h, p90-per-step — see the Obsidian note "Teacher Action Space Reduction for
# IEEE-36 and IEEE-118") times a chunk size of 6 (the example's max_array_size=500 on 2592
# chronics -> chunk_size=6 -> ~44h worst case). This does NOT auto-scale with max_array_size —
# a smaller max_array_size means a bigger chunk_size and needs a bigger --time; recompute
# worst_case_hours = chunk_size * per_chronic_p90_hours before changing max_array_size, and
# budget the tail, not the mean, since a killed task truncates towards the start of its current
# chronic and biases the resulting action set. BWUniCluster's `cpu_il,cpu` partitions cap
# walltime at 72h — if worst_case_hours exceeds that, raise max_array_size instead (smaller
# chunks) rather than requesting more time than the partition allows.
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/unicluster/teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>
#
# Arguments:
#   grid              case36 | case118
#   experiment_name   Results sub-directory, e.g. 2026_09_20_teacher_n1_case36
#   max_concurrent    SLURM array throttle (--array=0-N%max_concurrent)
#   max_array_size    Upper bound on the number of array tasks to create. Chronics are split
#                     into ceil(n_chronics / max_array_size) chunks per task to stay under this.
#                     BWUniCluster's actual SLURM MaxArraySize is unconfirmed for this account —
#                     500 is a conservative guess; raise it (fewer, longer tasks) or lower it
#                     (more, shorter tasks) if sbatch still rejects the array. Check the real
#                     limit with: scontrol show config | grep -i MaxArraySize
#   lines_to_attack   Line ids from: python experiments/build_action_space.py select-lines --grid <grid>
#
# Example:
#   slurm/unicluster/teacher_n1_array.sh case36 2026_09_20_teacher_n1_case36 50 500 3 12 27 41 58 60 71 88 95 101

GRID="${1:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>}"
EXPERIMENT="${2:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>}"
MAX_CONCURRENT="${3:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>}"
MAX_ARRAY_SIZE="${4:?Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>}"
shift 4
LINES_TO_ATTACK=("$@")
if [[ ${#LINES_TO_ATTACK[@]} -eq 0 ]]; then
    echo "Usage: teacher_n1_array.sh <grid> <experiment_name> <max_concurrent> <max_array_size> <lines_to_attack...>" >&2
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

CHUNK_SIZE=$(( (N_CHRONICS + MAX_ARRAY_SIZE - 1) / MAX_ARRAY_SIZE ))
N_TASKS=$(( (N_CHRONICS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Sharding ${N_CHRONICS} chronics into ${N_TASKS} tasks of up to ${CHUNK_SIZE} chronics each (max ${MAX_CONCURRENT} concurrent)."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=teacher_n1_${GRID}
#SBATCH --output=${OUT_DIR}/shard_%a.%j.log
#SBATCH --error=${OUT_DIR}/shard_%a.%j.err
#SBATCH --array=0-$((N_TASKS - 1))%${MAX_CONCURRENT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate curriculum

CHUNK_SIZE=${CHUNK_SIZE}
START=\$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1 ))
END=\$(( START + CHUNK_SIZE - 1 ))
mapfile -t CHRONIC_IDS < <(sed -n "\${START},\${END}p" "${OUT_DIR}/chronic_ids.txt")

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID (array task \$SLURM_ARRAY_TASK_ID)"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Chronics:     \${CHRONIC_IDS[*]}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/build_action_space.py run-teacher-chronics \
    --grid "${GRID}" \
    --lines-to-attack ${LINES_TO_ATTACK[*]} \
    --chronic-ids "\${CHRONIC_IDS[@]}" \
    --save-path "${OUT_DIR}/shard_\${SLURM_ARRAY_TASK_ID}.csv"
EOF
