#!/bin/bash

# Imitation pretraining of the RL policy (experiments/pretrain_policy.py) on one GPU — BWUniCluster.
#
# All arguments are passed to the script as Hydra overrides. The first run of a dataset/converter
# pair converts all states (CPU, a few minutes per 100k) and caches them in imitation.cache_dir.
#
# Usage:
#   slurm/unicluster/pretrain_policy.sh <experiment_name> [hydra overrides...]
#
# Environment:
#   SBATCH_TIME   Walltime (default 08:00:00).
#
# Example:
#   slurm/unicluster/pretrain_policy.sh pretrain_gnn_graph_soft env=case36 model=gnn obs_space=default \
#       obs_space.use_forecasts=false relation_awareness=disabled imitation=default \
#       imitation.dataset=data/imitation/case36_k208_greedy_soft_t0.01.npz

USAGE="Usage: pretrain_policy.sh <experiment_name> [hydra overrides...]"
EXPERIMENT="${1:?$USAGE}"
shift
REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
OUT_DIR="results/junior/${EXPERIMENT}"
mkdir -p "$OUT_DIR"
SBATCH_TIME="${SBATCH_TIME:-08:00:00}"

sbatch <<EOF
#!/bin/bash -l
# --export=NONE: see imitation_tutor_array.sh.
#SBATCH --export=NONE
#SBATCH --job-name=pretrain_${EXPERIMENT}
#SBATCH --output=${OUT_DIR}/job.%j.log
#SBATCH --error=${OUT_DIR}/job.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=gpu_h100,gpu_a100_il

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)"

PYTHONPATH="\$(pwd)/src" python experiments/pretrain_policy.py \
    experiment.name=${EXPERIMENT} imitation.device=cuda $*
EOF
