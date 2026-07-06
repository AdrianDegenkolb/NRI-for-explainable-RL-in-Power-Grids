#!/bin/bash
# RAPPO case36 — seed 0, 200k timesteps, top-K sparsification (multiplier=7)
# K_budget = 7 × (1+0.5) × 118 = 1,239 edges (vs 31,152 FC for N=177)
experiment_name=$(date +%Y_%m_%d)_IEEE36/rappo
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=rappo_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/rappo_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_rappo_s${seed}.%j.log
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
    model=ragnn \
    obs_space=graph \
    experiment.nb_timesteps=200000 \
    relation_awareness=default \
    relation_awareness.sparsification.top_k_multiplier=7 \
    rollouts.num_gpus_per_learner_worker=0 \
    rollouts.num_rollout_workers=24 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name} \
    env=case36
EOF
done
