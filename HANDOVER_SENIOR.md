# Handover: run the Senior experiment on bwUniCluster

**Written:** 2026-10-24, for a Claude session on the cluster. **Branch:** `GraphComp`.
**Repo on the cluster:** `/pfs/data6/home/ka/ka_iai/ka_hw6998/dev/NRI-for-explainable-RL-in-Power-Grids`.
**Background:** Adrian's Obsidian notes *Junior*, *Junior Continued* and *Senior* (design and "How to run"), plus
`CLAUDE.md` (Senior paragraph under "Key Source Modules").

## Goal
Fine-tune the pretrained Junior (`results/junior/junior2_hard/best`) with PPO on IEEE-36 (`l2rpn_wcci_2020`). Then compare
four agents by survival in rollouts on the val and test envs, with argmax and top-k simulation (k = 1, 5, 20, 50):

| Agent | What it is | Runs |
|---|---|---|
| `junior` | the pretrained policy, no RL | rollouts only |
| `senior_warmup` | PPO from the Junior; the first 10 iterations train only the critic | seeds 0, 1, 2 |
| `senior_nowarmup` | PPO from the Junior; actor and critic train from the start | seeds 0, 1, 2 |
| `baseline` | PPO from random weights | seeds 0, 1, 2 |

`slurm/unicluster/imitation_vs_rl36.sh` submits everything: the RL runs, the export of each final policy, rollout arrays
for every agent, and the do-nothing and Tutor references. Your job is to launch it, check it and report back. You don't
analyse the results: Adrian does that locally in `experiments/senior_results.ipynb`.

## Rules
- Don't change the experiment's settings (PPO hyperparameters, reward, architecture, seeds, splits) without asking Adrian.
  The settings are fixed by the script and explained at the end.
- Don't delete or overwrite other experiments' results. Put scratch files in `/tmp` and delete them afterwards.
- Don't write to the Obsidian vault.

## 0. Setup (login node)
```bash
cd /pfs/data6/home/ka/ka_iai/ka_hw6998/dev/NRI-for-explainable-RL-in-Power-Grids
git checkout GraphComp && git pull
module load devel/miniforge && eval "$(conda shell.bash hook)" && conda activate L2RPN  # the script calls python at submission

# The code must be there (if not, the commit was not pushed yet: ask Adrian)
ls src/imitation/senior.py experiments/architecture_overrides.py src/analysis/training_curves.py
grep -c senior_nowarmup slurm/unicluster/imitation_vs_rl36.sh              # > 0
# The Junior must be there, with CompleteGraphObsSpace
ls results/junior/junior2_hard/best/{policy_weights.pt,config.yaml}
grep observation_space results/junior/junior2_hard/best/config.yaml      # CompleteGraphObsSpace
# Tests (CPU, ~2 min). test_rappo_policy.py::test_stats_fn fails on the old code too; ignore it.
PYTHONPATH=$(pwd)/src python -m pytest src/tests/imitation src/tests/rarl_rllib src/tests/grid2op_env src/tests/analysis -q
```

## 1. Pilot (one seed, short)
The full chain has never run with this Junior and with the normalized reward, so start with a short pilot.

```bash
# ~25 PPO iterations per agent; rollouts only on val, argmax and top-20. Run with DRY_RUN=1 first to see the jobs.
NB_TIMESTEPS=30000 SBATCH_TIME_RL=08:00:00 EVAL_SPLITS=val TOP_KS=20 \
    slurm/unicluster/imitation_vs_rl36.sh 2026_10_24_IEEE36_senior_pilot results/junior/junior2_hard/best 0
```

The submission prints the architecture overrides and the job ids; note them. When the three RL jobs have finished
(`sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode`), print the training scalars:

```bash
PYTHONPATH=$(pwd)/src python - results/senior/2026_10_24_IEEE36_senior_pilot <<'EOF'
import sys
from pathlib import Path
from analysis.training_curves import SENIOR_TAGS, load_scalars
for run in sorted(Path(sys.argv[1]).glob("*_s0")):
    trial = sorted(e.parent for e in run.glob("*/events.out.tfevents.*"))[-1]
    t = load_scalars(trial, SENIOR_TAGS)
    print(run.name)
    print(t[["iteration", "kl", "vf_loss", "vf_explained_var", "entropy", "train_survival"]].round(4).to_string(index=False))
EOF
```

Check these, and **stop and report to Adrian if one fails**:
1. **Weights loaded.** In iteration 1, entropy is clearly below log(209) ≈ 5.34 for both `senior_*` runs, and close to
   5.34 for `baseline` (the old run: senior 4.24, baseline 5.34).
2. **Warm-up holds the policy.** `senior_warmup`: `kl` is exactly 0 in iterations 1–10.
3. **The critic learns.** `vf_loss` is far below 1, since value targets are now in [0, 1] (all earlier runs sat at
   50–100). `vf_explained_var` rises above the ≤ 0.07 that every earlier PPO run had.
4. **KL after the warm-up.** Report `kl` for iterations 11–15 of `senior_warmup` and 1–5 of `senior_nowarmup`. A jump is
   expected: in updates, BatchNorm normalizes with the batch's statistics, but sampling uses its running statistics.
   It is a problem if `kl` stays above 1, or is `inf`, after the first 3 iterations.
5. **Export and rollouts worked.** Each of `eval/{junior,senior_warmup_s0,senior_nowarmup_s0,baseline_s0}/{rollout,rollout_top20}/`
   holds 8 `episodes_valenv_*.csv` files (completeness snippet in step 3), and the `out/*.err` files end without a traceback.

If all pass, report the tables briefly and go on to step 2.

## 2. Full run
```bash
slurm/unicluster/imitation_vs_rl36.sh 2026_10_24_IEEE36_senior results/junior/junior2_hard/best 0 1 2
```
- Keep the experiment name `2026_10_24_IEEE36_senior`: the notebook and Adrian's note use it.
- Jobs:
  - 9 RL runs of 64 cores each. About 15 h each in the old run (200k agent steps); walltime 72 h.
  - 9 export jobs.
  - 104 rollout arrays of 8 CPU tasks each (walltime 4 h).
- A job waits for its predecessor with `afterok`, so the arrays of the junior and the references start at once.

## 3. Monitoring and completion
```bash
squeue -u $USER
sacct -S today --format=JobID,JobName%30,State,Elapsed,ExitCode
```
- **Failed RL or export job.** Its dependent jobs stay pending forever (`DependencyNeverSatisfied`). `scancel` them, read
  `results/senior/<exp>/out/<agent>_s<seed>.<jobid>.err` and report. Don't resubmit RL runs on your own.
- **Failed or timed-out rollout array.** Resubmit only that array. It writes to the same folder, and the seeds are fixed,
  so rerun tasks reproduce identical files:
  ```bash
  LOG_DIR=results/senior/<exp>/out slurm/unicluster/imitation_rollout_array.sh \
      results/senior/<exp>/eval/<agent> <val|test> <agent> [--top-k K]
  # References: run dir eval/junior, agent do_nothing (--do-nothing) or tutor (--top-k 209),
  #             plus --out-dir results/senior/<exp>/eval/<do_nothing|tutor>/rollout
  ```
- **Completeness.** Every folder needs `val=8 test=8`, including `eval/tutor/rollout` and `eval/do_nothing/rollout`:
  ```bash
  for d in results/senior/2026_10_24_IEEE36_senior/eval/*/rollout*; do
    echo "$d val=$(ls $d/episodes_valenv_*.csv 2>/dev/null | wc -l) test=$(ls $d/episodes_testenv_*.csv 2>/dev/null | wc -l)"
  done
  ```

## 4. Report back to Adrian
- Job ids, failures and reruns.
- The pilot checks, and the completeness listing.
- A survival overview (numbers only, no interpretation):
```bash
PYTHONPATH=$(pwd)/src python - results/senior/2026_10_24_IEEE36_senior <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from analysis.episodes import load_episodes, survival_metrics
rows = []
for folder in sorted(Path(sys.argv[1], "eval").glob("*/rollout*")):
    for split in ("valenv", "testenv"):
        files = sorted(folder.glob(f"episodes_{split}_*.csv"))
        if files:
            m = survival_metrics(load_episodes(*files))
            rows.append({"agent": folder.parent.name, "mode": folder.name, "split": split, "chronics": m.chronics,
                         "full %": round(m.full_survival_pct, 1), "mean %": round(m.mean_survival_pct, 1)})
print(pd.DataFrame(rows).to_string(index=False))
EOF
```
Adrian copies the results with the rsync command in the header of `experiments/senior_results.ipynb`.

## Why the settings are what they are (don't change them)
- **Junior:** `junior2_hard` survived best under top-k simulation in *Junior Continued*. It uses CompleteGraphObsSpace and
  has no voltage angles, which were broken in the old Juniors' training data.
- **One network for all agents:** the architecture and observation space come from the Junior's `config.yaml`
  (`src/imitation/senior.py`), so the weights load strictly and the baseline is the same network.
- **Dropout 0 in RL:** RLlib samples in eval mode but updates in train mode, so dropout would make PPO's ratio noisy.
- **PPO:** `configs/rllib/training/ppo.yaml`, entropy coefficient 0, so the Junior's policy isn't pushed towards uniform.
- **`reward.normalize_by_episode_length=true`:** the RL agent collects every grid step's reward until its next decision.
  Unscaled, value targets were in the hundreds, and RLlib's `vf_clip_param=100` (a cap on each sample's squared error)
  zeroed the critic's gradient: explained variance stayed ≈ 0 in every earlier PPO run. Divided by the chronic's
  length, the targets lie in [0, 1].
- **Warm-up:** `set_policy_trainable` (`src/rarl_rllib/callback.py`) freezes the actor and the GNN, and keeps the GNN's
  BatchNorm layers in eval mode. Before this fix, the sampling policy drifted during the warm-up (KL 2.2 in the old run).
- **Val and test:** val is comparable with the Junior tables. Test is untouched, since the Junior was picked by its
  val survival.
