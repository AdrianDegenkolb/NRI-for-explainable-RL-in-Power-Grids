#!/bin/bash
# RAPPO: Relation-Aware PPO (NRI encoder + RAGNN) — CPU variant
experiment_name=$(date +%Y_%m_%d)_rappo118
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
#SBATCH --time=70:00:00
#SBATCH --mem=249G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo \
    model=ragnn \
    obs_space=graph \
    relation_awareness=default \
    rollouts.num_gpus_per_learner_worker=0 \
    rollouts.num_rollout_workers=1 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed} \
    training.sgd_minibatch_size=1 \
    env=case118 rollouts=default \
    model.encoder.max_degree=20 \
    model.encoder.max_path_distance=50
EOF
done
