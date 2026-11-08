#!/bin/bash

# Re-run TopologyVariabilityAnalyzer on existing trial directories.
#
# PPO mode (default): iterates over CustomPPO_* subdirs in <results_dir>/<obs_space>/.
# Junior mode (--junior): submits one job per obs_space run dir directly.
#
# Usage (from the repo root):
#   # PPO
#   slurm/unicluster/rerun_variability_analysis.sh <results_dir>
#   # Junior IL
#   slurm/unicluster/rerun_variability_analysis.sh --junior <junior_base_dir>
#
# Examples:
#   slurm/unicluster/rerun_variability_analysis.sh \
#       results/2026_10_28_compare_graph_obs_spaces_IEEE14
#   slurm/unicluster/rerun_variability_analysis.sh --junior \
#       results/junior/2026_11_05_topology_variability_study/36bus_junior
#
# Environment:
#   OBS_SPACES       Space-separated list (default: no_edges substation)
#   ENV_SPLIT        [Junior] val or test (default: test)
#   SBATCH_TIME      Walltime per job (default 02:00:00)
#   DRY_RUN=1        Print scripts instead of submitting

JUNIOR=0
if [[ "${1:-}" == "--junior" ]]; then
    JUNIOR=1
    shift
fi

USAGE="Usage: rerun_variability_analysis.sh [--junior] <results_dir>"
RESULTS_DIR="${1:?$USAGE}"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-no_edges substation}"
ENV_SPLIT="${ENV_SPLIT:-test}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

LOG_DIR="${RESULTS_DIR}/reanalysis_logs"

if [[ "$JUNIOR" -eq 1 ]]; then
    # --- Junior mode: one job per obs_space run dir ---
    for obs_space in "${OBS_SPACES[@]}"; do
        trial_dir="${RESULTS_DIR}/${obs_space}"
        [[ -d "$trial_dir" ]] || { echo "WARN: ${trial_dir} not found, skipping"; continue; }

        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=reanalyse_jr_${obs_space}
#SBATCH --output=${LOG_DIR}/junior_${obs_space}.%j.log
#SBATCH --error=${LOG_DIR}/junior_${obs_space}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  obs_space: ${obs_space}  split: ${ENV_SPLIT}"

PYTHONPATH="\$(pwd)/src" python experiments/rerun_analysis.py \
    --junior \
    --trial-dir "${trial_dir}" \
    --env-split "${ENV_SPLIT}"
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "----- junior / ${obs_space}"
            echo "${SCRIPT}"
        else
            mkdir -p "$LOG_DIR"
            JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
            echo "submitted  junior  obs_space=${obs_space}  -> job ${JOB_ID}"
        fi
    done

else
    # --- PPO mode: one job per CustomPPO_* trial dir ---
    for obs_space in "${OBS_SPACES[@]}"; do
        for trial_dir in "${RESULTS_DIR}/${obs_space}"/CustomPPO_*/; do
            [[ -d "$trial_dir" ]] || continue
            trial_name="$(basename "$trial_dir")"

            SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=reanalyse_${trial_name}
#SBATCH --output=${LOG_DIR}/${obs_space}_${trial_name}.%j.log
#SBATCH --error=${LOG_DIR}/${obs_space}_${trial_name}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  trial: ${trial_dir}"

PYTHONPATH="\$(pwd)/src" python experiments/rerun_analysis.py \
    --trial-dir "${trial_dir%/}"
EOF
)
            if [[ -n "${DRY_RUN:-}" ]]; then
                echo "----- ${obs_space} / ${trial_name}"
                echo "${SCRIPT}"
            else
                mkdir -p "$LOG_DIR"
                JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
                echo "submitted  obs_space=${obs_space}  trial=${trial_name}  -> job ${JOB_ID}"
            fi
        done
    done
fi
