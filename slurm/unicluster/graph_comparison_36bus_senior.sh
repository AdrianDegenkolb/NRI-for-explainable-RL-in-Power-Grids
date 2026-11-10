#!/bin/bash

# 36-bus graph comparison with RL fine-tuning — BWUniCluster.
#
# For each graph representation, the best Junior HPO trial of graph_comparison_36bus_il.sh is retrained
# with new seeds, each Junior is fine-tuned with PPO (Senior with critic warm-up, settings of
# imitation_vs_rl36.sh), and Junior and Senior are rolled out greedily (argmax) on the val and test envs.
#
# Jobs per obs_space and seed:
#   junior  GPU: pretrain_policy.py with results/junior/<PREFIX>_<obs>/best/config.yaml, only the seed changed
#     ├─> junior rollouts  CPU arrays, one per split
#     └─> senior  one train.py run per RL_DEVICES entry, all from the Junior's weights (default: CPU and GPU
#                 in parallel). The first to finish exports its policy and cancels the others, so the GPU run
#                 gives speed and the CPU run is the fallback if the GPU run fails or never starts.
#           └─> senior rollouts  CPU arrays, one per split, after whichever Senior finishes first
#
# The RL model is built from the representation's base configs (model=gcn_substation_readout,
# obs_space=graph_comparison/<obs>) plus every model/obs_space value of the Junior's config
# (experiments/architecture_overrides.py, dropout 0), so ImitationInitCallback can load the weights strictly.
#
# Layout (results/senior/<experiment>/):
#   <obs>/junior_s<seed>/                 pretrain_policy.py run dir: policy_weights.pt, config.yaml, rollout/
#   <obs>/senior_s<seed>_<cpu|gpu>/       train.py output: Hydra config, Tune trial (checkpoints, analysis/)
#   <obs>/eval/senior_s<seed>/            exported Senior policy: policy_weights.pt, config.yaml, rollout/,
#                                         trained_on.txt (cpu or gpu), source_checkpoint.txt
#   <obs>/eval/senior_s<seed>.claim/      created by the Senior that exported first
#   <obs>/eval/senior_s<seed>.jobs        job ids of this seed's Seniors, for the winner to cancel the others
#   out/                                  SLURM logs
#
# Usage (from the repo root with the L2RPN env active; the architecture is read on the login node):
#   slurm/unicluster/graph_comparison_36bus_senior.sh
#
# Environment:
#   EXPERIMENT_NAME      Results folder (default <today>_IEEE36_graph_comparison_senior)
#   OBS_SPACES           Representations (default "no_edges ptdf substation substation_ptdf")
#   SEEDS                Seeds of Junior and Senior (default "0 1 2 3 4")
#   PREFIX               HPO study prefix (default graph36)
#   RL_DEVICES           Seniors per seed, raced (default "cpu gpu"; "cpu" or "gpu" alone runs one)
#   NB_TIMESTEPS         RL agent steps per Senior (default 200000)
#   N_WORKERS            RL rollout workers (default 24; the PPO updates, not sampling, dominate the run time)
#   WARMUP_ITERS         Critic warm-up iterations (default 10)
#   RL_CPUS, RL_MEM      Cores and memory per Senior (default 40, 100G: Ray uses 33 cores, the CPU Seniors of
#                        2026_10_26 used 68 GB)
#   EVAL_SPLITS          Grid2Op envs of the rollouts (default "val test")
#   N_ROLLOUT_TASKS      Array tasks per rollout (default 2, ~30-50 min each)
#   SBATCH_TIME_IL       Walltime per Junior (default 00:45:00; the best HPO trials trained 2-10 min)
#   SBATCH_TIME_RL_CPU   Walltime per CPU Senior (default 48:00:00; the CPU Seniors of 2026_10_26 took 22-37 h)
#   SBATCH_TIME_RL_GPU   Walltime per GPU Senior (default 24:00:00; must stay below gpu_a100_il's 48 h)
#   SBATCH_TIME_ROLLOUT  Walltime per rollout task (default 01:30:00)
#   DRY_RUN=1            Print the sbatch scripts instead of submitting

set -euo pipefail

EXPERIMENT="${EXPERIMENT_NAME:-$(date +%Y_%m_%d)_IEEE36_graph_comparison_senior}"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-no_edges ptdf substation substation_ptdf}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
read -r -a RL_DEVICES <<< "${RL_DEVICES:-cpu gpu}"
read -r -a EVAL_SPLITS <<< "${EVAL_SPLITS:-val test}"
PREFIX="${PREFIX:-graph36}"
NB_TIMESTEPS="${NB_TIMESTEPS:-200000}"
N_WORKERS="${N_WORKERS:-24}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
RL_CPUS="${RL_CPUS:-40}"
RL_MEM="${RL_MEM:-100G}"
N_ROLLOUT_TASKS="${N_ROLLOUT_TASKS:-2}"
SBATCH_TIME_IL="${SBATCH_TIME_IL:-00:45:00}"
SBATCH_TIME_ROLLOUT="${SBATCH_TIME_ROLLOUT:-01:30:00}"

declare -A RL_PARTITION=([cpu]="cpu_il,cpu" [gpu]="gpu_h100,gpu_a100_il,gpu_h100_vip1")
declare -A RL_GRES=([cpu]="" [gpu]="#SBATCH --gres=gpu:1")
declare -A RL_NUM_GPUS=([cpu]=0 [gpu]=1)
declare -A RL_TIME=([cpu]="${SBATCH_TIME_RL_CPU:-48:00:00}" [gpu]="${SBATCH_TIME_RL_GPU:-24:00:00}")
for device in "${RL_DEVICES[@]}"; do
    if [[ -z "${RL_PARTITION[$device]:-}" ]]; then
        echo "RL_DEVICES entries must be cpu or gpu, got '${device}'" >&2
        exit 1
    fi
done

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
OUT_DIR="results/senior/${EXPERIMENT}"
LOG_DIR="${OUT_DIR}/out"

# Fail before submitting anything if a study is missing, its architecture cannot be read, or a Senior of
# this experiment was already exported (its claim would make every new Senior cancel itself).
declare -A BEST_DIRS ARCH_OVERRIDES
for obs in "${OBS_SPACES[@]}"; do
    best_dir="results/junior/${PREFIX}_${obs}/best"
    if [[ ! -f "${best_dir}/config.yaml" ]]; then
        echo "${best_dir}/config.yaml not found: run graph_comparison_36bus_il.sh first." >&2
        exit 1
    fi
    BEST_DIRS[$obs]="$(realpath "$best_dir")"
    if ! ARCH_OVERRIDES[$obs]=$(PYTHONPATH="$(pwd)/src" python experiments/architecture_overrides.py "${best_dir}/config.yaml"); then
        echo "Reading the architecture of ${best_dir} failed: is the L2RPN env active?" >&2
        exit 1
    fi
    for seed in "${SEEDS[@]}"; do
        if [[ -d "${OUT_DIR}/${obs}/eval/senior_s${seed}.claim" ]]; then
            echo "${OUT_DIR}/${obs}/eval/senior_s${seed}.claim exists: that Senior was already exported." >&2
            exit 1
        fi
    done
done

# Reward 1/max_step per step: value targets stay in [0, 1], so vf_clip_param does not zero the critic's gradient.
# kl_coeff=0: RLlib's old API stack ignores use_kl_loss=false. freeze_batch_norm: RLlib samples in eval mode
# but computes the loss in train mode (see imitation_vs_rl36.sh).
COMMON_RL="training=ppo env=case36 relation_awareness=disabled imitation=default \
reward=constant reward.normalize_by_episode_length=true training.kl_coeff=0.0 imitation.freeze_batch_norm=true \
imitation.critic_warmup_iters=${WARMUP_ITERS} experiment.nb_timesteps=${NB_TIMESTEPS} \
rollouts.num_rollout_workers=${N_WORKERS} rollouts.num_gpus_per_learner_worker=0"

ENV_SETUP='module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="$(pwd)/src"
echo "Node: $(hostname)"'

# submit <job id to wait for (afterok) or ""> <script on stdin> -> prints the job id
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

# rollouts <run dir> <agent> <dependency> -> job ids of the argmax arrays on every split.
# imitation_rollout_array.sh submits with --dependency=afterok:<dependency>, so "A?afterok:B" waits for A or B.
rollouts() {
    local run_dir="$1" agent="$2" dependency="$3" split ids=()
    for split in "${EVAL_SPLITS[@]}"; do
        ids+=("$(DEPENDENCY="${dependency}" LOG_DIR="${LOG_DIR}" N_TASKS="${N_ROLLOUT_TASKS}" SBATCH_TIME="${SBATCH_TIME_ROLLOUT}" \
            slurm/unicluster/imitation_rollout_array.sh "${run_dir}" "${split}" "${agent}")")
    done
    echo "${ids[*]}"
}

echo "Experiment: ${OUT_DIR}"
echo "Obs spaces: ${OBS_SPACES[*]}   seeds: ${SEEDS[*]}   Seniors: ${RL_DEVICES[*]}"
if [[ -z "${DRY_RUN:-}" ]]; then
    mkdir -p "${LOG_DIR}"
fi

for obs in "${OBS_SPACES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        junior_dir="${OUT_DIR}/${obs}/junior_s${seed}"
        senior_eval="${OUT_DIR}/${obs}/eval/senior_s${seed}"
        claim="${senior_eval}.claim"
        jobs_file="${senior_eval}.jobs"

        IL_JOB=$(submit "" <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=junior_${obs}_s${seed}
#SBATCH --output=${LOG_DIR}/junior_${obs}_s${seed}.%j.log
#SBATCH --error=${LOG_DIR}/junior_${obs}_s${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME_IL}
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_vip1
${ENV_SETUP}
python experiments/pretrain_policy.py --config-path "${BEST_DIRS[$obs]}" --config-name config \
    experiment.seed=${seed} imitation.device=cuda imitation.out_dir=${junior_dir} hydra.run.dir=${junior_dir}/hydra
EOF
)

        RL_JOBS=()
        for device in "${RL_DEVICES[@]}"; do
            senior_dir="${obs}/senior_s${seed}_${device}"
            # The claim (mkdir is atomic on the shared filesystem) decides which Senior exports. The loser
            # cancels itself and sleeps, so it never ends with exit 0 and satisfies the rollouts' dependency.
            RL_JOBS+=("$(submit "${IL_JOB}" <<EOF
#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --job-name=senior_${device}_${obs}_s${seed}
#SBATCH --output=${LOG_DIR}/senior_${device}_${obs}_s${seed}.%j.log
#SBATCH --error=${LOG_DIR}/senior_${device}_${obs}_s${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${RL_CPUS}
#SBATCH --mem=${RL_MEM}
#SBATCH --time=${RL_TIME[$device]}
#SBATCH --partition=${RL_PARTITION[$device]}
${RL_GRES[$device]}
export RAY_gcs_rpc_server_reconnect_timeout_s=300
${ENV_SETUP}
set -e
python experiments/train.py ${COMMON_RL} rollouts.num_gpus=${RL_NUM_GPUS[$device]} \
    model=gcn_substation_readout obs_space=graph_comparison/${obs} ${ARCH_OVERRIDES[$obs]} \
    imitation.weights_path=${junior_dir}/policy_weights.pt \
    experiment.seed=${seed} experiment.name=senior/${EXPERIMENT}/${senior_dir}
mkdir -p ${OUT_DIR}/${obs}/eval
if ! mkdir ${claim} 2>/dev/null; then
    echo "Another Senior of this seed was exported first: cancelling this job."
    scancel \$SLURM_JOB_ID || true
    sleep 120
    exit 1
fi
if ! python experiments/export_policy_weights.py --experiment-dir ${OUT_DIR}/${senior_dir} --out ${senior_eval}; then
    rmdir ${claim}
    exit 1
fi
echo ${device} > ${senior_eval}/trained_on.txt
for job in \$(cat ${jobs_file}); do
    if [[ "\$job" != "\$SLURM_JOB_ID" ]]; then
        scancel "\$job" || true
    fi
done
exit 0
EOF
)")
        done
        if [[ -z "${DRY_RUN:-}" ]]; then
            mkdir -p "${OUT_DIR}/${obs}/eval"
            echo "${RL_JOBS[*]}" > "${jobs_file}"
        fi

        any_senior="${RL_JOBS[0]}"
        for job in "${RL_JOBS[@]:1}"; do
            any_senior+="?afterok:${job}"
        done
        echo "${obs} s${seed}: junior ${IL_JOB} -> rollouts $(rollouts "${junior_dir}" "junior_${obs}_s${seed}" "${IL_JOB}")"
        echo "${obs} s${seed}: seniors ${RL_JOBS[*]} (${RL_DEVICES[*]}) -> rollouts $(rollouts "${senior_eval}" "senior_${obs}_s${seed}" "${any_senior}")"
    done
done

echo "Episode tables: ${OUT_DIR}/<obs>/{junior_s<seed>,eval/senior_s<seed>}/rollout/episodes_{valenv,testenv}_*.csv"
