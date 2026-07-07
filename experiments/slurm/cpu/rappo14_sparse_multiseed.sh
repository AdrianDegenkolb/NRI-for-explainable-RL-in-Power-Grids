#!/bin/bash
# RAPPO case14 — 5 seeds, 200k timesteps, top-K sparsification (multiplier=7) + static prior
# K_budget = 7 × (1+0.5) × 40 = 420 edges (vs 3,192 FC for N=57)
experiment_name=$(date +%Y_%m_%d)_IEEE14/rappo_sparse_multiseed
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0 1 2 3 4; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=rappo14sp_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/rappo14sp_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_rappo14sp_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu_il,cpu

export RAY_gcs_rpc_server_reconnect_timeout_s=300
module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: \$(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo \
    model=ragnn \
    obs_space=graph \
    relation_awareness=default \
    relation_awareness.sparsification.top_k_multiplier=7 \
    experiment.nb_timesteps=200000 \
    rollouts.num_gpus_per_learner_worker=0 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}
EOF
done
