#!/bin/bash

# Retrain the best HPO trial of each 36-bus graph comparison obs_space with multiple seeds,
# then roll each out (argmax + top-k) on the val env.
#
# Prerequisites: the HPO studies from graph_comparison_36bus_il.sh must be complete (each
# results/junior/graph36_<obs>/best/config.yaml must exist).
#
# Usage (from repo root):
#   slurm/unicluster/graph_comparison_36bus_multiseed.sh
#
# Environment:
#   SEEDS          Space-separated seeds (default: "0 1 2 3 4")
#   OBS_SPACES     Space-separated obs_spaces (default: all 12)
#   PREFIX         Study name prefix (default: graph36)
#   TOP_KS         Top-k values for simulation rollouts (default: "20 50")
#   SBATCH_TIME    Walltime for pretraining (default: 02:00:00)
#   SBATCH_TIME_ROLLOUT  Walltime per rollout task (default: 04:00:00)
#   DRY_RUN=1      Print scripts instead of submitting

set -euo pipefail

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
PREFIX="${PREFIX:-graph36}"
read -r -a TOP_KS <<< "${TOP_KS:-20 50}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"
SBATCH_TIME_ROLLOUT="${SBATCH_TIME_ROLLOUT:-04:00:00}"

DEFAULT_OBS_SPACES="no_edges substation complete elements heterogeneous lodf ptdf zbus elements_lodf substation_ptdf substation_zbus flat"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-$DEFAULT_OBS_SPACES}"

OUT_ROOT="results/junior/${PREFIX}_multiseed"
LOG_DIR="${OUT_ROOT}/logs"

echo "============================================"
echo "36-bus Graph Comparison — Multi-seed retrain"
echo "Seeds:      ${SEEDS[*]}"
echo "Obs spaces: ${OBS_SPACES[*]}"
echo "Top-k:      ${TOP_KS[*]}"
echo "Output:     ${OUT_ROOT}"
echo "============================================"
echo ""

for obs_space in "${OBS_SPACES[@]}"; do
    study="${PREFIX}_${obs_space}"
    best_config="results/junior/${study}/best/config.yaml"

    if [[ -z "${DRY_RUN:-}" && ! -f "$best_config" ]]; then
        echo "SKIP ${obs_space}: ${best_config} not found"
        continue
    fi

    echo "--- ${obs_space} ---"

    for seed in "${SEEDS[@]}"; do
        RUN_DIR="${OUT_ROOT}/${obs_space}/seed_${seed}"

        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=ms_${obs_space}_${seed}
#SBATCH --output=${LOG_DIR}/${obs_space}_${seed}.%j.log
#SBATCH --error=${LOG_DIR}/${obs_space}_${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=gpu_h100,gpu_a100_il

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  obs_space: ${obs_space}  seed: ${seed}"

PYTHONPATH="\$(pwd)/src" python experiments/pretrain_policy.py \
    --config-path "\$(pwd)/results/junior/${study}/best" --config-name config \
    imitation.out_dir=${RUN_DIR} imitation.device=cuda \
    experiment.seed=${seed} \
    hydra.run.dir=${RUN_DIR}/hydra
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "  [DRY] pretrain ${obs_space} seed=${seed}"
            echo "$SCRIPT"
        else
            mkdir -p "$LOG_DIR"
            mkdir -p "$RUN_DIR"
            TRAIN_JOB=$(sbatch --parsable <<< "${SCRIPT}")
            echo "  pretrain seed=${seed} -> job ${TRAIN_JOB}"
        fi

        # Rollouts: argmax + top-k, each depending on the pretraining job
        for mode in argmax "${TOP_KS[@]/#/top}"; do
            if [[ "$mode" == "argmax" ]]; then
                EXTRA_ARGS=""
                ROLLOUT_TAG="${obs_space}_s${seed}"
            else
                k="${mode#top}"
                EXTRA_ARGS="--top-k ${k}"
                ROLLOUT_TAG="${obs_space}_s${seed}_${mode}"
            fi

            ROLLOUT_SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=ro_${ROLLOUT_TAG}
#SBATCH --output=${LOG_DIR}/rollout_${ROLLOUT_TAG}_%a.%A.log
#SBATCH --error=${LOG_DIR}/rollout_${ROLLOUT_TAG}_%a.%A.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=${SBATCH_TIME_ROLLOUT}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)  rollout: ${ROLLOUT_TAG}  task: \$SLURM_ARRAY_TASK_ID"

PYTHONPATH="\$(pwd)/src" python experiments/imitation_rollout.py \
    --run-dir ${RUN_DIR} --env-split val \
    --agent ${ROLLOUT_TAG} \
    --task-id \$SLURM_ARRAY_TASK_ID --n-tasks 8 \
    ${EXTRA_ARGS}
EOF
)
            if [[ -n "${DRY_RUN:-}" ]]; then
                echo "  [DRY] rollout ${ROLLOUT_TAG}"
            else
                ROLLOUT_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} <<< "${ROLLOUT_SCRIPT}")
                echo "    rollout ${mode} -> job ${ROLLOUT_JOB} (after ${TRAIN_JOB})"
            fi
        done
    done
    echo ""
done

echo "============================================"
echo "Submitted. Results in: ${OUT_ROOT}"
echo "============================================"
