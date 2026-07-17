#!/bin/bash
# RAPPO case36 — 5 seeds, 200k timesteps, top-K sparsification (multiplier=7).
# After all seed jobs complete, auto-submits cross_seed_analysis.py.
#
# Usage:
#   bash experiments/slurm/cpu/rappo36_multiseed_with_analysis.sh
#
# Optional env override (before running):
#   export SEEDS="0 1 2 3 4"      # default: 0 1 2 3 4

experiment_name=$(date +%Y_%m_%d)_IEEE36/rappo_multiseed
export experiment_name

SEEDS=${SEEDS:-"0 1 2 3 4"}

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p "results/experiments/${experiment_name}/out"

# ── submit seed training jobs ──────────────────────────────────────────────────
job_ids=()

for seed in $SEEDS; do
    jid=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=rappo36_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/rappo36_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_rappo36_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=249G
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
    rollouts.num_rollout_workers=24 \
    experiment.seed=${seed} \
    experiment.name=${experiment_name} \
    env=case36
EOF
    )
    echo "Submitted seed=${seed} → job ${jid}"
    job_ids+=("${jid}")
done

# ── build dependency string ───────────────────────────────────────────────────
dep=$(printf ":%s" "${job_ids[@]}")
dep="afterok${dep}"   # afterok:j0:j1:j2:j3:j4

# ── submit analysis job ───────────────────────────────────────────────────────
analysis_jid=$(sbatch --parsable --dependency="${dep}" << EOF
#!/bin/bash
#SBATCH --job-name=rappo36_xseed
#SBATCH --output=results/experiments/${experiment_name}/out/cross_seed_analysis.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_cross_seed_analysis.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: \$(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/cross_seed_analysis.py \
    --experiment-dir "results/experiments/${experiment_name}" \
    --env l2rpn_wcci_2020 \
    --n-obs 100 \
    --rho-thresh 0.7 \
    --out-dir "results/experiments/${experiment_name}/cross_seed"
EOF
)

echo ""
echo "Analysis job ${analysis_jid} will start after: ${dep}"
echo "Results → results/experiments/${experiment_name}/cross_seed/"
