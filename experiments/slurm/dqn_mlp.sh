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
#SBATCH --cpus-per-task=111
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu,cpu_il

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=dqn \
    model=mlp \
    obs_space=flat \
    relation_awareness=disabled \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
