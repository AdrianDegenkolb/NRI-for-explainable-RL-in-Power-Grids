#!/bin/bash
# RASAC: Relation-Aware SAC (NRI encoder + RAGNN)
experiment_name=$(date +%Y_%m_%d)_rasac
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=rasac_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/rasac_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_rasac_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=111
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu,cpu_il

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=sac \
    model=ragnn \
    obs_space=graph \
    relation_awareness=default \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
