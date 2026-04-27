#!/bin/bash
# RADQN: Relation-Aware Rainbow DQN (NRI encoder + RAGNN)
experiment_name=$(date +%Y_%m_%d)_radqn
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=radqn_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/radqn_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_radqn_s${seed}.%j.log
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
    model=ragnn \
    obs_space=graph \
    relation_awareness=default \
    experiment=long \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
