#!/bin/bash
# DQN + MLP baseline (flat observations, no graph structure)
experiment_name=$(date +%Y_%m_%d)_dqn_mlp
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=dqn_mlp_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/dqn_mlp_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_dqn_mlp_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_mi300
#SBATCH --gres=gpu:4

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=dqn \
    model=mlp \
    obs_space=flat \
    relation_awareness=disabled \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
