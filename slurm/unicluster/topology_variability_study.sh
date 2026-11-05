#!/bin/bash

# Topology variability study — compare no_edges vs substation across settings,
# measure topology variability and correlate with graph-method advantage.
#
# Part 1: 14-bus + opponent — 2 methods × 5 seeds PPO training (100k steps)
# Part 2: 36-bus Junior IL  — 2 pretraining jobs + rollout arrays
# Part 3: 14-bus reanalysis — add TopologyVariabilityAnalyzer to existing results
#
# Usage (from the repo root): slurm/unicluster/topology_variability_study.sh
#
# Environment:
#   EXPERIMENT_NAME   Results folder (default <today>_topology_variability_study)
#   SEEDS             Seeds for Part 1 PPO (default "0 1 2 3 4")
#   N_WORKERS         Rollout workers per PPO run (default 48)
#   DEVICE            gpu (default) or cpu
#   TUTOR_DATASET     Path to the 36-bus tutor dataset
#   REANALYSIS_DIR    Existing 14-bus graph comparison results to reanalyse
#   SBATCH_TIME_PPO   Walltime for Part 1 PPO jobs (default 06:00:00)
#   SBATCH_TIME_IL    Walltime for Part 2 pretraining jobs (default 08:00:00)
#   SBATCH_TIME_ROLLOUT  Walltime for Part 2 rollout jobs (default 04:00:00)
#   SBATCH_TIME_REANALYSIS  Walltime for Part 3 reanalysis jobs (default 02:00:00)
#   DRY_RUN=1         Print sbatch scripts instead of submitting

set -euo pipefail

experiment_name="${EXPERIMENT_NAME:-$(date +%Y_%m_%d)_topology_variability_study}"

read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
N_WORKERS="${N_WORKERS:-48}"
DEVICE="${DEVICE:-gpu}"
TUTOR_DATASET="${TUTOR_DATASET:-data/imitation/case36_k208_greedy_soft_t0.01.npz}"
REANALYSIS_DIR="${REANALYSIS_DIR:-results/2026_10_28_compare_graph_obs_spaces_IEEE14}"
SBATCH_TIME_PPO="${SBATCH_TIME_PPO:-06:00:00}"
SBATCH_TIME_IL="${SBATCH_TIME_IL:-08:00:00}"
SBATCH_TIME_ROLLOUT="${SBATCH_TIME_ROLLOUT:-04:00:00}"
SBATCH_TIME_REANALYSIS="${SBATCH_TIME_REANALYSIS:-02:00:00}"

OBS_SPACES=(no_edges substation)

case "$DEVICE" in
    gpu) PARTITION_GPU="gpu_h100,gpu_a100_il,gpu_h100_vip1"; GRES_GPU="#SBATCH --gres=gpu:1"; DEVICE_OVERRIDES="rollouts.num_gpus=1" ;;
    cpu) PARTITION_GPU="cpu_il,cpu"; GRES_GPU=""; DEVICE_OVERRIDES="rollouts.num_gpus=0" ;;
    *) echo "DEVICE must be gpu or cpu, got '${DEVICE}'" >&2; exit 1 ;;
esac

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

RESULTS_DIR="results/${experiment_name}"

echo "============================================"
echo "Topology variability study: ${experiment_name}"
echo "Results: ${RESULTS_DIR}"
echo "============================================"

# ============================================================================
# Part 1: 14-bus + opponent — PPO training
# ============================================================================
echo ""
echo "--- Part 1: 14-bus + opponent PPO (${#OBS_SPACES[@]} methods × ${#SEEDS[@]} seeds) ---"

COMMON_PPO="training=ppo relation_awareness=disabled experiment.nb_timesteps=100000 \
rollouts.num_rollout_workers=${N_WORKERS} rollouts.num_gpus_per_learner_worker=0 ${DEVICE_OVERRIDES} \
experiment.post_training_evaluation.enabled=True reward.normalize_by_episode_length=true \
training.kl_coeff=0.0 opponent=random_line"

for seed in "${SEEDS[@]}"; do
    for obs_space in "${OBS_SPACES[@]}"; do
        OUT_DIR="${RESULTS_DIR}/14bus_opponent/${obs_space}/out"

        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=topo_14opp_${obs_space}_${seed}
#SBATCH --output=${OUT_DIR}/${obs_space}_${seed}.%j.log
#SBATCH --error=${OUT_DIR}/${obs_space}_${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=${SBATCH_TIME_PPO}
#SBATCH --partition=${PARTITION_GPU}
${GRES_GPU}

export RAY_gcs_rpc_server_reconnect_timeout_s=300
module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="\$(pwd)/src"
echo "Node: \$(hostname)  seed: ${seed}  obs_space: ${obs_space}"

python experiments/train.py ${COMMON_PPO} model=gcn_substation_readout \
    obs_space="graph_comparison/${obs_space}" \
    experiment.name="${experiment_name}/14bus_opponent/${obs_space}" \
    experiment.seed=${seed}
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "----- 14bus_opp ${obs_space} seed ${seed}"
            echo "${SCRIPT}"
        else
            mkdir -p "$OUT_DIR"
            JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
            echo "14bus_opp  seed=${seed}  obs_space=${obs_space}  -> job ${JOB_ID}"
        fi
    done
done

# ============================================================================
# Part 2: 36-bus Junior IL — pretraining + rollout
# ============================================================================
echo ""
echo "--- Part 2: 36-bus Junior IL (${#OBS_SPACES[@]} methods) ---"

declare -A PRETRAIN_JOBS  # obs_space -> job_id

for obs_space in "${OBS_SPACES[@]}"; do
    IL_NAME="${experiment_name}/36bus_junior/${obs_space}"
    IL_OUT_DIR="results/junior/${IL_NAME}"

    SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=junior_${obs_space}
#SBATCH --output=${IL_OUT_DIR}/job.%j.log
#SBATCH --error=${IL_OUT_DIR}/job.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME_IL}
#SBATCH --partition=gpu_h100,gpu_a100_il

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  obs_space: ${obs_space}"

PYTHONPATH="\$(pwd)/src" python experiments/pretrain_policy.py \
    experiment.name="${IL_NAME}" \
    env=case36 \
    model=gcn_substation_readout \
    obs_space="graph_comparison/${obs_space}" \
    obs_space.use_forecasts=false \
    relation_awareness=disabled \
    imitation=default \
    imitation.dataset=${TUTOR_DATASET} \
    imitation.device=cuda
EOF
)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "----- junior ${obs_space}"
        echo "${SCRIPT}"
        PRETRAIN_JOBS[${obs_space}]="DRY"
    else
        mkdir -p "$IL_OUT_DIR"
        JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
        echo "junior  obs_space=${obs_space}  -> job ${JOB_ID}"
        PRETRAIN_JOBS[${obs_space}]="${JOB_ID}"
    fi
done

# Rollout arrays: val and test splits, dependent on pretraining
N_TASKS=8
ROLLOUT_LOG_DIR="${RESULTS_DIR}/rollout_logs"

for obs_space in "${OBS_SPACES[@]}"; do
    IL_RUN_DIR="results/junior/${experiment_name}/36bus_junior/${obs_space}"
    DEP_JOB="${PRETRAIN_JOBS[${obs_space}]}"

    for env_split in val test; do
        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=rollout_${obs_space}_${env_split}
#SBATCH --output=${ROLLOUT_LOG_DIR}/${obs_space}_${env_split}_%a.%A.log
#SBATCH --error=${ROLLOUT_LOG_DIR}/${obs_space}_${env_split}_%a.%A.err
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=${SBATCH_TIME_ROLLOUT}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  obs_space: ${obs_space}  split: ${env_split}"

PYTHONPATH="\$(pwd)/src" python experiments/imitation_rollout.py \
    --run-dir ${IL_RUN_DIR} \
    --env-split ${env_split} \
    --agent junior_${obs_space} \
    --task-id \$SLURM_ARRAY_TASK_ID \
    --n-tasks ${N_TASKS}
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "----- rollout ${obs_space} ${env_split} (dep: ${DEP_JOB})"
            echo "${SCRIPT}"
        else
            mkdir -p "$ROLLOUT_LOG_DIR"
            JOB_ID=$(sbatch --parsable --dependency=afterok:${DEP_JOB} <<< "${SCRIPT}")
            echo "rollout  obs_space=${obs_space}  split=${env_split}  -> job ${JOB_ID} (after ${DEP_JOB})"
        fi
    done
done

# ============================================================================
# Part 3: 14-bus reanalysis of existing checkpoints
# ============================================================================
echo ""
echo "--- Part 3: 14-bus reanalysis (${REANALYSIS_DIR}) ---"

for obs_space in "${OBS_SPACES[@]}"; do
    OBS_DIR="${REANALYSIS_DIR}/${obs_space}"
    if [[ ! -d "$OBS_DIR" ]]; then
        echo "WARN: ${OBS_DIR} not found, skipping reanalysis for ${obs_space}"
        continue
    fi

    for trial_dir in "${OBS_DIR}"/CustomPPO_*/; do
        [[ -d "$trial_dir" ]] || continue
        trial_name="$(basename "$trial_dir")"
        REANALYSIS_LOG_DIR="${RESULTS_DIR}/reanalysis_logs"

        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=reanalyse_${obs_space}_${trial_name}
#SBATCH --output=${REANALYSIS_LOG_DIR}/${obs_space}_${trial_name}.%j.log
#SBATCH --error=${REANALYSIS_LOG_DIR}/${obs_space}_${trial_name}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=${SBATCH_TIME_REANALYSIS}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  trial: ${trial_dir}"

PYTHONPATH="\$(pwd)/src" python experiments/rerun_analysis.py \
    --trial-dir "${trial_dir}"
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "----- reanalyse ${obs_space} ${trial_name}"
            echo "${SCRIPT}"
        else
            mkdir -p "$REANALYSIS_LOG_DIR"
            JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
            echo "reanalyse  obs_space=${obs_space}  trial=${trial_name}  -> job ${JOB_ID}"
        fi
    done
done

echo ""
echo "============================================"
echo "All jobs submitted. Results in: ${RESULTS_DIR}"
echo "============================================"
