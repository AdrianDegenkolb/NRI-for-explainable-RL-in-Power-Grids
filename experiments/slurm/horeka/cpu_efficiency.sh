#!/bin/bash
# CPU Efficiency Experiment — 5 variants, single seed, IEEE14 RAPPO
# Each variant builds on the previous to isolate individual speedup contributions.
#
# V1: baseline (current config)
# V2: chronics copied to $TMPDIR (eliminates network filesystem I/O)
# V3: V2 + MultifolderWithCache (chronics loaded into RAM after first episode)
# V4: V3 + truncate_episodes rollout mode (eliminates worker straggler wait)
# V5: V4 + GPU learner workers
#
# Usage: bash experiments/slurm/horeka/cpu_efficiency.sh
experiment_name=$(date +%Y_%m_%d)_IEEE14_cpu_efficiency
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/${experiment_name}/out

G2OP_ENV=l2rpn_case14_sandbox

# ---------------------------------------------------------------------------
# Shared PPO args (no GPU, single seed)
# ---------------------------------------------------------------------------
BASE_ARGS="training=ppo model=ragnn obs_space=graph relation_awareness=default rollouts.num_gpus_per_learner_worker=0 experiment.seed=0 experiment.nb_timesteps=50000"

# ---------------------------------------------------------------------------
# V1 — Baseline
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff_v1
#SBATCH --output=results/${experiment_name}/out/v1.%j.log
#SBATCH --error=results/${experiment_name}/out/v1.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly

export RAY_gcs_rpc_server_reconnect_timeout_s=300
conda activate L2RPN

echo "Node: \$(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS} \
    experiment.name=${experiment_name}/v1_baseline
EOF

# ---------------------------------------------------------------------------
# V2 — Chronics in TMPDIR
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff_v2
#SBATCH --output=results/${experiment_name}/out/v2.%j.log
#SBATCH --error=results/${experiment_name}/out/v2.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly

export RAY_gcs_rpc_server_reconnect_timeout_s=300
conda activate L2RPN

echo "Node: \$(hostname)"

# Copy chronics to fast local storage
mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    experiment.name=${experiment_name}/v2_tmpdir
EOF

# ---------------------------------------------------------------------------
# V3 — TMPDIR + MultifolderWithCache
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff_v3
#SBATCH --output=results/${experiment_name}/out/v3.%j.log
#SBATCH --error=results/${experiment_name}/out/v3.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly

export RAY_gcs_rpc_server_reconnect_timeout_s=300
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    env.use_chronics_cache=true \
    experiment.name=${experiment_name}/v3_tmpdir_cache
EOF

# ---------------------------------------------------------------------------
# V4 — TMPDIR + Cache + truncate_episodes
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff_v4
#SBATCH --output=results/${experiment_name}/out/v4.%j.log
#SBATCH --error=results/${experiment_name}/out/v4.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpuonly

export RAY_gcs_rpc_server_reconnect_timeout_s=300
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    env.use_chronics_cache=true \
    rollouts.batch_mode=truncate_episodes \
    experiment.name=${experiment_name}/v4_truncated
EOF

# ---------------------------------------------------------------------------
# V5 — TMPDIR + Cache + truncate_episodes + GPU
# ---------------------------------------------------------------------------
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ceff_v5
#SBATCH --output=results/${experiment_name}/out/v5.%j.log
#SBATCH --error=results/${experiment_name}/out/v5.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --partition=accelerated,accelerated-h100

export RAY_gcs_rpc_server_reconnect_timeout_s=300
conda activate L2RPN

echo "Node: \$(hostname)"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    ${BASE_ARGS} \
    env.chronics_dir=\$TMPDIR/data_grid2op \
    env.use_chronics_cache=true \
    rollouts.batch_mode=truncate_episodes \
    rollouts.num_gpus_per_learner_worker=1 \
    experiment.name=${experiment_name}/v5_gpu
EOF
