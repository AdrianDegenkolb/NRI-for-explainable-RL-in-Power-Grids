#!/bin/bash
# PPO + MLP baseline (flat observations, no graph structure) — CPU variant
# Updated 2026-07-09: 200k steps, 24 workers, default sgd_minibatch_size (256)
experiment_name=$(date +%Y_%m_%d)_IEEE36/ppo_mlp
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0 1 2 3 4; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ppo_mlp_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/ppo_mlp_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_ppo_mlp_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=249G
#SBATCH --partition=cpu_il,cpu

export RAY_gcs_rpc_server_reconnect_timeout_s=300
module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo \
    model=mlp \
    obs_space=flat \
    experiment.nb_timesteps=200000 \
    relation_awareness=disabled \
    rollouts.num_gpus_per_learner_worker=0 \
    rollouts.num_rollout_workers=24 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name} \
    env=case36
EOF
done
