#!/bin/bash
# CPU Efficiency Experiment v4 — 4 variants, single seed, IEEE14 RAPPO
# Identical to v3 but runs full training (300k steps) on production partitions.
#
# V1: chronics in TMPDIR (baseline)
# V2: TMPDIR + TruncatedEpisodes
# V3: TMPDIR + GPU
# V4: TMPDIR + TruncatedEpisodes + GPU
#
# Usage: bash experiments/slurm/horeka/cpu_efficiency_v4.sh
experiment_name=$(date +%Y_%m_%d)_IEEE14_cpu_efficiency_v4
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/${experiment_name}/out

G2OP_ENV=l2rpn_case14_sandbox

# ---------------------------------------------------------------------------
# Shared PPO args (single seed, 48 rollout workers)
# ---------------------------------------------------------------------------
BASE_ARGS="training=ppo model=ragnn obs_space=graph relation_awareness=default experiment.seed=0 experiment.nb_timesteps=300000 rollouts.num_rollout_workers=48 experiment.post_training_evaluation.enabled=False model.gnn.sparsify_threshold=0.05 model.gnn.diagnose_every=0"
BASE_ARGS_CPU="${BASE_ARGS} rollouts.num_gpus_per_learner_worker=0"
BASE_ARGS_GPU="${BASE_ARGS} rollouts.num_gpus=1 rollouts.num_gpus_per_learner_worker=1 rollouts.num_learner_workers=1"

# ---------------------------------------------------------------------------
# V1 — TMPDIR baseline
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff4_v1
#SBATCH --output=results/${experiment_name}/out/v1.%j.log
#SBATCH --error=results/${experiment_name}/out/v1.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS_CPU} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    experiment.name=${experiment_name}/v1_tmpdir
EOF

# ---------------------------------------------------------------------------
# V2 — TMPDIR + TruncatedEpisodes
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff4_v2
#SBATCH --output=results/${experiment_name}/out/v2.%j.log
#SBATCH --error=results/${experiment_name}/out/v2.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS_CPU} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    rollouts.batch_mode=truncate_episodes \
    experiment.name=${experiment_name}/v2_tmpdir_truncated
EOF

# ---------------------------------------------------------------------------
# V3 — TMPDIR + GPU
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff4_v3
#SBATCH --output=results/${experiment_name}/out/v3.%j.log
#SBATCH --error=results/${experiment_name}/out/v3.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=accelerated,accelerated-h100
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS_GPU} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    experiment.name=${experiment_name}/v3_tmpdir_gpu
EOF

# ---------------------------------------------------------------------------
# V4 — TMPDIR + TruncatedEpisodes + GPU
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff4_v4
#SBATCH --output=results/${experiment_name}/out/v4.%j.log
#SBATCH --error=results/${experiment_name}/out/v4.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=accelerated,accelerated-h100
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS_GPU} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    rollouts.batch_mode=truncate_episodes \
    experiment.name=${experiment_name}/v4_tmpdir_truncated_gpu
EOF
