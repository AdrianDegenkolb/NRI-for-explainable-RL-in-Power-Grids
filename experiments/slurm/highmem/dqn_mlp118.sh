#!/bin/bash
# DQN + MLP baseline (flat observations, no graph structure) — highmem partition
experiment_name=$(date +%Y_%m_%d)_IEEE118/dqn_mlp
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
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
#SBATCH --time=72:00:00
#SBATCH --mem=500G
#SBATCH --partition=highmem

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=dqn \
    model=mlp \
    obs_space=flat \
    relation_awareness=disabled \
    rollouts.num_gpus_per_learner_worker=0 \
    experiment=long \
    experiment.seed=${seed} \
    experiment.name=${experiment_name} \
    rollouts.num_rollout_workers=1 \
    training.train_batch_size=1 \
    env=case118 \
    experiment.nb_timesteps=20000000
EOF
done
