#!/bin/bash

# Retrain the best trial of Junior HPO studies (pretrain_hpo.sh) for experiments/junior_training.ipynb
# — BWUniCluster.
#
# One GPU job per study. Each reruns experiments/pretrain_policy.py with the study's best/config.yaml
# (same data, hyperparameters and seed) and writes the run directory (see src/imitation/pretrain_run.py)
# to results/pretraining/junior_training/<study>/. All studies land in that one directory, which is all
# the notebook reads. GPU training is not bit-for-bit reproducible, so the curves differ slightly from
# the HPO trial's.
#
# The best trials took ~27 min on one GPU, so the 30 min dev partitions are too tight by default.
#
# Usage:
#   slurm/unicluster/junior_retrain.sh [study ...]
#   default studies: junior_hard junior_soft_t0.005 junior_soft_t0.01 junior_soft_t0.02
#
# Environment:
#   SBATCH_PARTITION   Partition(s) (default gpu_h100,gpu_a100_il).
#   SBATCH_TIME        Walltime (default 01:00:00).
#   DRY_RUN=1          Print the sbatch scripts instead of submitting.
#
# Example:
#   SBATCH_PARTITION=dev_gpu_h100 SBATCH_TIME=00:30:00 slurm/unicluster/junior_retrain.sh junior_hard

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
STUDIES=("$@")
if [[ ${#STUDIES[@]} -eq 0 ]]; then
    STUDIES=(junior_hard junior_soft_t0.005 junior_soft_t0.01 junior_soft_t0.02)
fi
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu_h100,gpu_a100_il}"
SBATCH_TIME="${SBATCH_TIME:-01:00:00}"
OUT_ROOT="results/pretraining/junior_training"

for STUDY in "${STUDIES[@]}"; do
    BEST_DIR="results/pretraining/${STUDY}/best"
    if [[ ! -f "${BEST_DIR}/config.yaml" ]]; then
        echo "${BEST_DIR}/config.yaml not found; run pretrain_hpo.sh ${STUDY} first." >&2
        exit 1
    fi
    OUT_DIR="${OUT_ROOT}/${STUDY}"
    SCRIPT=$(cat <<EOF
#!/bin/bash -l
# --export=NONE: see imitation_tutor_array.sh.
#SBATCH --export=NONE
#SBATCH --job-name=retrain_${STUDY}
#SBATCH --output=${OUT_DIR}/job.%j.log
#SBATCH --error=${OUT_DIR}/job.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=${SBATCH_PARTITION}

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)"

PYTHONPATH="\$(pwd)/src" python experiments/pretrain_policy.py \
    --config-path "$(realpath "$BEST_DIR")" --config-name config \
    imitation.out_dir=${OUT_DIR} imitation.device=cuda hydra.run.dir=${OUT_DIR}/hydra
EOF
)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "----- sbatch (${STUDY})"
        echo "$SCRIPT"
    else
        mkdir -p "$OUT_DIR"
        echo "$SCRIPT" | sbatch
    fi
done
