#!/bin/bash

# GeneralTutor imitation-data generation as a SLURM array — BWUniCluster.
#
# Each array task runs a strided set of chronics and writes one .npy per chronic to
# results/tutor/<experiment>/experience/ (rows: [action_idx, *obs.to_vect()]). Chronics whose
# file already exists are skipped, so re-submitting the same experiment only redoes unfinished
# chronics. Survival per chronic is in each task log ("game over at step-X").
#
# Chronics are assigned with a stride (task t gets lines t, t+N_TASKS, t+2*N_TASKS, ...), not in
# contiguous chunks: chronic_ids.txt is sorted by name, so each month is one block of ~215
# chronics, and contiguous chunks would give a task 13 chronics of the same month. Winter chronics
# need far more Tutor searches than summer ones (in the teacher array, which used contiguous
# chunks, ~50% of Nov-Mar tasks hit the walltime vs ~0% of Apr-Aug). Striding gives every task
# roughly one chronic per month, so tasks have similar runtimes and the walltime can be sized from
# the average chronic rather than an all-winter task.
#
# Afterwards, build the Junior's dataset:
#   python experiments/build_action_space.py build-tutor-dataset \
#       --experience-dir results/tutor/<experiment>/experience \
#       --out-dir results/tutor/<experiment>/dataset --name <name>
#
# Walltime does not auto-scale: set SBATCH_TIME to chunk_size * per_chronic_mean_hours plus a
# margin for the tail (measure
# the per-chronic cost with slurm/unicluster/tutor_smoketest.sh first), capped at 72h.
# MaxArraySize on BWUniCluster is 1001 (see teacher_n1_array.sh).
#
# Runs in the separate `curriculum` conda environment (environment_curriculum.yaml) — never L2RPN.
#
# Usage:
#   slurm/unicluster/tutor_array.sh <grid> <experiment_name> <action_space_npy> <max_concurrent> <max_array_size>
#
# Environment:
#   SBATCH_TIME       Walltime per array task (default 12:00:00).
#   ARRAY_INDICES     Run only these array tasks (sbatch --array syntax, e.g. "6,7,10-11") instead
#                     of all of them — to redo tasks that hit the walltime. Pass the same
#                     max_array_size as the original submission so the chronic stride is unchanged.
#
# Example:
#   slurm/unicluster/tutor_array.sh case36 2026_10_07_tutor_case36_k208 \
#       data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.npy 40 1000

USAGE="Usage: tutor_array.sh <grid> <experiment_name> <action_space_npy> <max_concurrent> <max_array_size>"
GRID="${1:?$USAGE}"
EXPERIMENT="${2:?$USAGE}"
ACTION_SPACE="${3:?$USAGE}"
MAX_CONCURRENT="${4:?$USAGE}"
MAX_ARRAY_SIZE="${5:?$USAGE}"

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
PYTHONPATH="$(pwd)/src" python experiments/build_action_space.py list-chronics --grid "${GRID}" \
    > "${OUT_DIR}/chronic_ids.txt"
N_CHRONICS=$(wc -l < "${OUT_DIR}/chronic_ids.txt")
if [[ "$N_CHRONICS" -eq 0 ]]; then
    echo "No chronics listed for grid ${GRID} — aborting." >&2
    exit 1
fi

CHUNK_SIZE=$(( (N_CHRONICS + MAX_ARRAY_SIZE - 1) / MAX_ARRAY_SIZE ))
N_TASKS=$(( (N_CHRONICS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
ARRAY_INDICES="${ARRAY_INDICES:-0-$((N_TASKS - 1))}"
echo "Sharding ${N_CHRONICS} chronics into ${N_TASKS} tasks of up to ${CHUNK_SIZE} chronics each (max ${MAX_CONCURRENT} concurrent), --time=${SBATCH_TIME}."

MAX_SLURM_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F'= *' '/^MaxArraySize/ {print $2}')
if [[ -n "$MAX_SLURM_ARRAY_SIZE" && "$N_TASKS" -gt "$MAX_SLURM_ARRAY_SIZE" ]]; then
    echo "N_TASKS=${N_TASKS} exceeds this cluster's MaxArraySize=${MAX_SLURM_ARRAY_SIZE}." >&2
    echo "Lower max_array_size (bigger chunks, fewer tasks) and raise SBATCH_TIME to match." >&2
    exit 1
fi

sbatch <<EOF
#!/bin/bash -l
# --export=NONE: a conda env active in the submitting shell would otherwise put its python
# first on PATH and break the 'conda' launcher (ModuleNotFoundError: No module named 'conda').
#SBATCH --export=NONE
#SBATCH --job-name=tutor_${GRID}
#SBATCH --output=${OUT_DIR}/task_%a.%j.log
#SBATCH --error=${OUT_DIR}/task_%a.%j.err
#SBATCH --array=${ARRAY_INDICES}%${MAX_CONCURRENT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=${SBATCH_TIME}
#SBATCH --mem=16G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate curriculum

mapfile -t CHRONIC_IDS < <(awk -v n=${N_TASKS} -v t=\$SLURM_ARRAY_TASK_ID '(NR - 1) % n == t' "${OUT_DIR}/chronic_ids.txt")

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID (array task \$SLURM_ARRAY_TASK_ID)"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Action space: ${ACTION_SPACE}"
echo "Chronics:     \${CHRONIC_IDS[*]}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/build_action_space.py run-tutor-chronics \
    --grid "${GRID}" \
    --action-space "${ACTION_SPACE}" \
    --chronic-ids "\${CHRONIC_IDS[@]}" \
    --save-dir "${OUT_DIR}/experience"
EOF
