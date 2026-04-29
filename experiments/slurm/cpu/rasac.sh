#!/bin/bash
# RASAC: Relation-Aware SAC (NRI encoder + RAGNN) — CPU variant
experiment_name=$(date +%Y_%m_%d)_rasac
export experiment_name

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$REPO_ROOT"

mkdir -p results/experiments/${experiment_name}/out

for seed in 0; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=rasac_s${seed}
#SBATCH --output=results/experiments/${experiment_name}/out/rasac_s${seed}.%j.log
#SBATCH --error=results/experiments/${experiment_name}/out/error_rasac_s${seed}.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=dev_cpu_il,dev_cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

echo "Node: $(hostname)"

PYTHONPATH=\$(pwd)/src python experiments/train.py \
    training=sac \
    model=ragnn \
    obs_space=graph \
    relation_awareness=default \
    rollouts.num_gpus_per_learner_worker=0 \
    experiment=long \
    experiment.seed=${seed} \
    experiment.name=${experiment_name}_s${seed}
EOF
done
