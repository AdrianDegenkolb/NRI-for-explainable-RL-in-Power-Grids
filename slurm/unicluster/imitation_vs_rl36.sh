#!/bin/bash

# Senior experiment: what imitation buys and what RL buys after it — IEEE-36, BWUniCluster.
#
# Four agents with the same network: the Junior's architecture and observation space, read from its
# config.yaml (experiments/architecture_overrides.py, dropout 0 in RL). RL rewards are normalized by the
# episode length. All are evaluated by the same rollout code on the same chronics with the same per-chronic seeds:
#   junior            imitation only: the pretrained policy, no RL
#   senior_warmup     PPO from the junior's weights; for the first WARMUP_ITERS iterations only the value
#                     branch trains (ImitationInitCallback)
#   senior_nowarmup   PPO from the junior's weights, actor and critic trained from the start
#   baseline          PPO from random weights
#
# Jobs per seed and RL agent: RL run (CPU, 64 cores) -> export of the final checkpoint's policy
# (experiments/export_policy_weights.py) -> rollout arrays (slurm/unicluster/imitation_rollout_array.sh).
# Every agent is rolled out on the Grid2Op val and test envs (144 chronics each), greedy (argmax) and with
# top-k simulation for every K in TOP_KS. The junior's rollouts start at once, as do the references:
# do-nothing and the Tutor (top-k over all TUTOR_K actions). RL evaluates on the val env during training.
#
# Layout (results/senior/<experiment_name>/):
#   <agent>_s<seed>/                    train.py output (Hydra config + Tune trial with checkpoints)
#   eval/<agent>[_s<seed>]/             policy_weights.pt, config.yaml,
#                                       rollout[_top<K>]/episodes_{valenv,testenv}_*.csv
#   eval/do_nothing/, eval/tutor/       references, rollout/episodes_{valenv,testenv}_*.csv
#   out/                                SLURM logs
#
# Usage (from an activated L2RPN env; the architecture is read on the login node):
#   slurm/unicluster/imitation_vs_rl36.sh <experiment_name> <junior_run_dir> [seed ...]
#
# Environment:
#   NB_TIMESTEPS      RL agent steps per run (default 200000, as the GNN PPO baselines)
#   N_WORKERS         Rollout workers per RL run (default 24)
#   SBATCH_TIME_RL    Walltime per RL run (default 72:00:00)
#   WARMUP_ITERS      Critic warm-up iterations of senior_warmup (default 10)
#   RL_OVERRIDES      Extra Hydra overrides for every RL run, e.g. "training.vf_clip_param=1000"
#   EVAL_SPLITS       Grid2Op envs to roll out in (default "val test")
#   TOP_KS            K of the top-k rollouts (default "1 5 20 50")
#   TUTOR_K           Actions incl. do-nothing; top-k over all of them = the Tutor (default 209)
#   N_EVAL_TASKS      Rollout array tasks per agent, split and mode (default 8)
#   DRY_RUN=1         Print the sbatch scripts instead of submitting
#
# Example:
#   slurm/unicluster/imitation_vs_rl36.sh 2026_10_24_IEEE36_senior results/junior/junior2_hard/best 0 1 2

USAGE="Usage: imitation_vs_rl36.sh <experiment_name> <junior_run_dir> [seed ...]"
EXPERIMENT="${1:?$USAGE}"
JUNIOR_DIR="${2:?$USAGE}"
shift 2
SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2)

NB_TIMESTEPS="${NB_TIMESTEPS:-200000}"
N_WORKERS="${N_WORKERS:-24}"
SBATCH_TIME_RL="${SBATCH_TIME_RL:-72:00:00}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
RL_OVERRIDES="${RL_OVERRIDES:-}"
read -r -a EVAL_SPLITS <<< "${EVAL_SPLITS:-val test}"
TOP_KS="${TOP_KS:-1 5 20 50}"
TUTOR_K="${TUTOR_K:-209}"
N_EVAL_TASKS="${N_EVAL_TASKS:-8}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
for f in policy_weights.pt config.yaml; do
    if [[ ! -f "${JUNIOR_DIR}/${f}" ]]; then
        echo "${JUNIOR_DIR}/${f} not found — pass the output dir of experiments/pretrain_policy.py." >&2
        exit 1
    fi
done

# Must match the junior's network: ImitationInitCallback loads its weights strictly.
if ! ARCH_OVERRIDES=$(PYTHONPATH="$(pwd)/src" python experiments/architecture_overrides.py "${JUNIOR_DIR}/config.yaml"); then
    echo "Reading the junior's architecture failed — is the L2RPN env active?" >&2
    exit 1
fi
echo "Architecture: ${ARCH_OVERRIDES}"

OUT_DIR="results/senior/${EXPERIMENT}"
mkdir -p "${OUT_DIR}/out" "${OUT_DIR}/eval/junior"
cp "${JUNIOR_DIR}/policy_weights.pt" "${JUNIOR_DIR}/config.yaml" "${OUT_DIR}/eval/junior/"
echo "${JUNIOR_DIR}" > "${OUT_DIR}/eval/junior/source_run_dir.txt"

# Reward 1/max_step per step: value targets are the share of the chronic still to survive, in [0, 1]. Unscaled,
# they are hundreds of steps per decision and vf_clip_param zeroes the critic's gradient.
# kl_coeff=0: RLlib's old API stack ignores use_kl_loss=false and adds an adaptive KL penalty (default 0.2,
# grew to 58 in the pilot); PPO's clipping alone limits the updates. freeze_batch_norm: RLlib samples in eval
# mode but computes the loss in train mode, and the batch statistics made these two policies differ by a KL of
# 0.3-0.8 after RL (ImitationInitCallback).
COMMON_OVERRIDES="training=ppo env=case36 model=gnn obs_space=default relation_awareness=disabled imitation=default \
reward=constant reward.normalize_by_episode_length=true +training.kl_coeff=0.0 imitation.freeze_batch_norm=true \
rollouts.num_gpus_per_learner_worker=0 rollouts.num_rollout_workers=${N_WORKERS} \
experiment.nb_timesteps=${NB_TIMESTEPS} ${ARCH_OVERRIDES} ${RL_OVERRIDES}"
JUNIOR_WEIGHTS="imitation.weights_path=${OUT_DIR}/eval/junior/policy_weights.pt"
declare -A VARIANT_OVERRIDES=(
    [senior_warmup]="${JUNIOR_WEIGHTS} imitation.critic_warmup_iters=${WARMUP_ITERS}"
    [senior_nowarmup]="${JUNIOR_WEIGHTS} imitation.critic_warmup_iters=0"
    [baseline]=""
)

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

# rollout_array <run dir> <agent> <env split> <dependency or ""> [imitation_rollout.py args...] -> job id
rollout_array() {
    local run_dir="$1" agent="$2" split="$3" dependency="$4"
    shift 4
    DEPENDENCY="${dependency}" LOG_DIR="${OUT_DIR}/out" N_TASKS="${N_EVAL_TASKS}" \
        slurm/unicluster/imitation_rollout_array.sh "${run_dir}" "${split}" "${agent}" "$@"
}

# rollouts <agent> <dependency or ""> -> job ids of eval/<agent>'s greedy and top-k arrays on every split
rollouts() {
    local agent="$1" dependency="$2" split k ids=()
    for split in "${EVAL_SPLITS[@]}"; do
        ids+=("$(rollout_array "${OUT_DIR}/eval/${agent}" "${agent}" "${split}" "${dependency}")")
        for k in ${TOP_KS}; do
            ids+=("$(rollout_array "${OUT_DIR}/eval/${agent}" "${agent}" "${split}" "${dependency}" --top-k "${k}")")
        done
    done
    echo "${ids[*]}"
}

echo "junior: rollouts $(rollouts junior "")"
for split in "${EVAL_SPLITS[@]}"; do
    DN=$(rollout_array "${OUT_DIR}/eval/junior" do_nothing "${split}" "" --do-nothing --out-dir "${OUT_DIR}/eval/do_nothing/rollout")
    TUTOR=$(rollout_array "${OUT_DIR}/eval/junior" tutor "${split}" "" --top-k "${TUTOR_K}" --out-dir "${OUT_DIR}/eval/tutor/rollout")
    echo "references (${split}): do-nothing ${DN}, tutor ${TUTOR}"
done

for seed in "${SEEDS[@]}"; do
    for variant in senior_warmup senior_nowarmup baseline; do
        name="${variant}_s${seed}"

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
python experiments/train.py ${COMMON_OVERRIDES} ${VARIANT_OVERRIDES[${variant}]} \
    experiment.seed=${seed} experiment.name=senior/${EXPERIMENT}/${name}
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
        echo "${name}: RL ${RL_JOB} -> export ${EXPORT_JOB} -> rollouts $(rollouts "${name}" "${EXPORT_JOB}")"
    done
done

echo "Episode tables: ${OUT_DIR}/eval/<agent>/rollout*/episodes_{valenv,testenv}_*.csv"
