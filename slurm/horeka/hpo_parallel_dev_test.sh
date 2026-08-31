#!/bin/bash
# Minimal smoke test for parallel Optuna/Tune trials on a single multi-GPU node.
#
# Ray Tune packs trials automatically based on each trial's resource request
# (rollouts.num_rollout_workers CPUs + rollouts.num_gpus GPUs). This script
# requests a full 4-GPU dev node and shrinks the per-trial footprint so 4
# trials fit and run concurrently. Not meant to produce a usable model —
# nb_timesteps is tiny, this only verifies the resource packing works.
#
# Usage: bash slurm/horeka/hpo_parallel_dev_test.sh
experiment_name=$(date +%Y_%m_%d)_hpo_parallel_dev_test
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/${experiment_name}/out

G2OP_ENV=l2rpn_case14_sandbox

sbatch << EOF
#!/bin/bash
#SBATCH --job-name=hpo_dev_test
#SBATCH --output=results/${experiment_name}/out/%j.log
#SBATCH --error=results/${experiment_name}/out/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "Node: \$(hostname)"
echo "GPUs visible to job:"
nvidia-smi -L

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=ppo model=ragnn obs_space=graph relation_awareness=default \
    optimization=optuna optimization.num_trials=4 \
    experiment.seed=0 \
    experiment.nb_timesteps=5000 \
    experiment.checkpoint_freq=1 \
    experiment.post_training_evaluation.enabled=False \
    rollouts.num_rollout_workers=16 \
    rollouts.num_gpus=1 \
    model.gnn.sparsify_threshold=0.05 \
    model.gnn.diagnose_every=0 \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    experiment.name=${experiment_name}/trial
EOF
