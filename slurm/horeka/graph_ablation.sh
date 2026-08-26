#!/bin/bash

# Graph ablation evaluation.
# Runs evaluate_agent on each method with shuffled edge indices (GraphAblationWrapper).
# Results → experiments/survival/obs_spaces/ablation/

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

OUT_DIR="experiments/survival/observation_spaces/ablation_fixed/out"
mkdir -p "$OUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=graph_ablation
#SBATCH --output=${OUT_DIR}/graph_ablation.%j.log
#SBATCH --error=${OUT_DIR}/graph_ablation.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:30:00
#SBATCH --mem=80G
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-pai00074

source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "========================================"
echo "Job ID:   \$SLURM_JOB_ID"
echo "Node:     \$(hostname)"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/graph_ablation.py --workers 10
EOF
