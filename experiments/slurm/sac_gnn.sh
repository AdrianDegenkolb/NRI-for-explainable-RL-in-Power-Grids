#!/bin/bash
# SAC + GNN baseline (fixed graph edges, no NRI encoder)
experiment_name=$(date +%Y_%m_%d)_sac_gnn
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=sac_gnn_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/sac_gnn_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_sac_gnn_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --gres=gpu:4

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname), GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none found')"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=sac \
    model=gnn \
    obs_space=graph \
    relation_awareness=disabled \
    experiment=long \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
