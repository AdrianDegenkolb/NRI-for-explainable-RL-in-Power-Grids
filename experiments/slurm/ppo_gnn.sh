#!/bin/bash
# PPO + GNN baseline (fixed graph edges, no NRI encoder)
experiment_name=$(date +%Y_%m_%d)_ppo_gnn
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ppo_gnn_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/ppo_gnn_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_ppo_gnn_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=111
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu,cpu_il

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo \
    model=gnn \
    obs_space=graph \
    relation_awareness=disabled \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
