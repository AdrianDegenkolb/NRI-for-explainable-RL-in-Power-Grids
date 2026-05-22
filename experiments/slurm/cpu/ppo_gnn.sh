#!/bin/bash
# PPO + GNN baseline (fixed graph edges, no NRI encoder) — CPU variant
experiment_name=$(date +%Y_%m_%d)_IEEE14/ppo_gnn
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ppo_gnn_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/ppo_gnn_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_ppo_gnn_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu_il,cpu

export RAY_gcs_rpc_server_reconnect_timeout_s=300
module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo \
    model=gnn \
    obs_space=graph \
    relation_awareness=disabled \
    rollouts.num_gpus_per_learner_worker=0 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}
EOF
done
