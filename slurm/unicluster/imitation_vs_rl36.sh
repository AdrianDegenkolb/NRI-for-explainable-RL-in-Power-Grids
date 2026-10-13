#!/bin/bash

# What imitation buys and what RL buys after it — IEEE-36, BWUniCluster.
#
# Three agents with the same network (GNN, GraphObsSpace without forecasts), evaluated by the same
# rollout code on the same unseen chronics with the same per-chronic seeds:
#   junior     imitation only: the pretrained policy (experiments/pretrain_policy.py), no RL
#   senior     PPO started from the junior's weights (critic warm-up, ImitationInitCallback)
#   baseline   PPO from random weights, otherwise identical to senior
# junior vs baseline = what imitation buys over RL from scratch; senior vs junior = what RL adds
# on top of imitation; senior vs baseline (also over training, in the val-env eval curves of Tune)
# = what pretraining buys RL.
#
# Jobs per seed: senior + baseline RL runs (CPU, 64 cores) -> export of the final checkpoint's
# policy (experiments/export_policy_weights.py) -> rollout array on the Grid2Op test env
# (experiments/imitation_rollout.py --env-split test). The junior's rollout array starts at once.
# The test env (144 chronics) was never seen by Teacher, Tutor, junior or RL; RL evaluates on the
# val env during training.
#
# Layout (results/<experiment_name>/):
#   senior_s<seed>/, baseline_s<seed>/   train.py output (Hydra config + Tune trial with checkpoints)
#   eval/<agent>/                        policy_weights.pt, config.yaml, rollout/episodes_testenv_*.csv
#   out/                                 SLURM logs
#
# Usage:
#   slurm/unicluster/imitation_vs_rl36.sh <experiment_name> <junior_run_dir> [seed ...]
#
# Environment:
#   NB_TIMESTEPS      RL agent steps per run (default 200000, as the GNN PPO baselines)
#   N_WORKERS         Rollout workers per RL run (default 24)
#   SBATCH_TIME_RL    Walltime per RL run (default 72:00:00)
#   N_EVAL_TASKS      Rollout array tasks per agent, chronics assigned with a stride (default 8)
#   DRY_RUN=1         Print the sbatch scripts instead of submitting
#
# Example:
#   slurm/unicluster/imitation_vs_rl36.sh 2026_10_14_IEEE36_imitation \
#       results/pretraining/pretrain_gnn_graph_soft 0 1 2

USAGE="Usage: imitation_vs_rl36.sh <experiment_name> <junior_run_dir> [seed ...]"
EXPERIMENT="${1:?$USAGE}"
JUNIOR_DIR="${2:?$USAGE}"
shift 2
SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2)

NB_TIMESTEPS="${NB_TIMESTEPS:-200000}"
N_WORKERS="${N_WORKERS:-24}"
SBATCH_TIME_RL="${SBATCH_TIME_RL:-72:00:00}"
N_EVAL_TASKS="${N_EVAL_TASKS:-8}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
for f in policy_weights.pt config.yaml; do
    if [[ ! -f "${JUNIOR_DIR}/${f}" ]]; then
        echo "${JUNIOR_DIR}/${f} not found — pass the output dir of experiments/pretrain_policy.py." >&2
        exit 1
    fi
done

OUT_DIR="results/${EXPERIMENT}"
mkdir -p "${OUT_DIR}/out" "${OUT_DIR}/eval/junior"
cp "${JUNIOR_DIR}/policy_weights.pt" "${JUNIOR_DIR}/config.yaml" "${OUT_DIR}/eval/junior/"
echo "${JUNIOR_DIR}" > "${OUT_DIR}/eval/junior/source_run_dir.txt"

# Must match the junior's pretraining config: ImitationInitCallback loads its weights strictly.
COMMON_OVERRIDES="training=ppo env=case36 model=gnn obs_space=default obs_space.use_forecasts=false \
relation_awareness=disabled imitation=default rollouts.num_gpus_per_learner_worker=0 \
rollouts.num_rollout_workers=${N_WORKERS} experiment.nb_timesteps=${NB_TIMESTEPS}"

ENV_SETUP='module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="$(pwd)/src"
echo "Node: $(hostname)"'

# submit <dependency or ""> <script on stdin> -> prints the job id
submit() {
    local dependency="$1" script
    script="$(cat)"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "----- sbatch ${dependency:+--dependency=afterok:${dependency}}" >&2
        echo "${script}" >&2
        echo "DRY_RUN"
        return
    fi
    sbatch --parsable ${dependency:+--dependency=afterok:${dependency}} <<< "${script}"
}

# rollout_array <agent> <dependency or ""> -> evaluates eval/<agent> on the test env
rollout_array() {
    local agent="$1" dependency="$2"
    submit "${dependency}" <<EOF
#!/bin/bash -l
# --export=NONE: see imitation_tutor_array.sh.
#SBATCH --export=NONE
#SBATCH --job-name=rollout_${agent}
#SBATCH --output=${OUT_DIR}/out/rollout_${agent}_%a.%A.log
#SBATCH --error=${OUT_DIR}/out/rollout_${agent}_%a.%A.err
#SBATCH --array=0-$((N_EVAL_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_il,cpu
${ENV_SETUP}
python experiments/imitation_rollout.py --run-dir ${OUT_DIR}/eval/${agent} --env-split test \
    --agent ${agent} --task-id \$SLURM_ARRAY_TASK_ID --n-tasks ${N_EVAL_TASKS}
EOF
}

JUNIOR_EVAL=$(rollout_array junior "")
echo "junior: rollout ${JUNIOR_EVAL}"

for seed in "${SEEDS[@]}"; do
    for variant in senior baseline; do
        name="${variant}_s${seed}"
        weights=""
        [[ "${variant}" == "senior" ]] && weights="imitation.weights_path=${OUT_DIR}/eval/junior/policy_weights.pt"

        RL_JOB=$(submit "" <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=${name}
#SBATCH --output=${OUT_DIR}/out/${name}.%j.log
#SBATCH --error=${OUT_DIR}/out/${name}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=249G
#SBATCH --time=${SBATCH_TIME_RL}
#SBATCH --partition=cpu_il,cpu
export RAY_gcs_rpc_server_reconnect_timeout_s=300
${ENV_SETUP}
python experiments/train.py ${COMMON_OVERRIDES} ${weights} \
    experiment.seed=${seed} experiment.name=${EXPERIMENT}/${name}
EOF
)
        EXPORT_JOB=$(submit "${RL_JOB}" <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=export_${name}
#SBATCH --output=${OUT_DIR}/out/export_${name}.%j.log
#SBATCH --error=${OUT_DIR}/out/export_${name}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_il,cpu
${ENV_SETUP}
python experiments/export_policy_weights.py --experiment-dir ${OUT_DIR}/${name} --out ${OUT_DIR}/eval/${name}
EOF
)
        EVAL_JOB=$(rollout_array "${name}" "${EXPORT_JOB}")
        echo "${name}: RL ${RL_JOB} -> export ${EXPORT_JOB} -> rollout ${EVAL_JOB}"
    done
done

echo "Episode tables: ${OUT_DIR}/eval/<agent>/rollout/episodes_testenv_*.csv"
