# N-1 Teacher Action-Space Generation — IEEE-36 & IEEE-118

Run log / decision record for building reduced topology action spaces with the Fraunhofer
`curriculumagent` N-1 Teacher. Substitutes for the Obsidian note
"Teacher Action Space Reduction for IEEE-36 and IEEE-118" — fold back in when convenient.

Started 2026-09-17. Branch `GraphComp`. All commands run in the `curriculum` conda env.

**Status: IN PROGRESS, checked 2026-09-30.** case118 array finished; case36 array ~85% through,
ETA ~2026-10-02/03. **Decision needed on 159 case36 walltime kills** (see snapshot).
To resume, see **`results/teacher/HANDOFF.md`** (partly stale: its `lines_to_attack` for case118
is the retired v1 list — use §2) and [Open items](#open-items) at the bottom.

### Snapshot 2026-09-30 (session 4)

| Job | What | State |
|---|---|---|
| `7009596` | case36 array, 864 tasks | **540 COMPLETED / 159 TIMEOUT / 40 running / 125 pending** (738-863 = rest of october + all of september); 708 shards, 13,251 rows (10,835 good), 3,989 unique good actions |
| `7036197` | case118 array, 748 tasks | **DONE** (last task ended 2026-09-28): 743 COMPLETED / 5 TIMEOUT; 617 shards, 3,533 rows (2,829 good), 1,724 unique good actions |

Remaining case36 tasks are mild months (3-16 h/task), so the array should drain in ~2-3 days.

**⚠ case36 timeouts got much worse than the 09-21 projection (27 → 159, 22% of finished tasks)**, and
none of the options (a)/(b)/(c) from 09-21 was taken. They sit exactly in the months with the
most rows, so the data is seasonally biased:

| case36 month | timeout share of finished tasks | rows per finished task |
|---|---|---|
| january / february / march | 49% / 53% / 52% | 31 / 31 / 30 |
| november / december | 54% / 29% | 28 / 32 |
| april - august | 0-1% | 1.5-11 |

A killed task keeps rows up to the kill, but loses the rest of its last chronic (sometimes two).
case118 is unaffected (≤ 3% per month). **Recommendation:** once 7009596 finishes, resubmit the
chronics of the 159 TIMEOUT tasks as a follow-up array at `SBATCH_TIME=72:00:00` (option (b)).
Rows from the killed runs would then be duplicated for the already-finished part of each
chronic, so the rerun should either replace those shards or be de-duplicated by shard.
Per-task states are now in `results/teacher/aggregated/task_states_<grid>.csv`.

**Analysis notebook re-run** (`experiments/teacher_action_space_analysis.ipynb`, on 09-30 data;
each section now states whether it is needed, what it measures and how to read it, followed by a
short conclusion). Key findings for choosing k:

- **No usable knee on either grid** (heuristic knee at rank 782 / 516); top-200 covers 43% of
  case36 picks, 34% of case118's. k has to be chosen by coverage + stability.
- **case118's frequency ranking is mostly noise:** split-half top-k Jaccard 0.16-0.25 (case36:
  0.45-0.70), top action picked only 15×. The array is complete, so more data won't come from
  this run. A frequency top-k beyond a few dozen actions is weakly supported.
- **case36 concentrates on sub 16** (17 objects): 49.5% of picks, effective substations 3.7.
  At the action level the top-100 still spans 23 substations. A per-substation cap of 10 costs
  5 pp coverage and is worth considering (notebook §10).
- Picks are tied to the attacked lines: case118's attacked-line endpoints are 12% of substations
  but get 67% of picks.
- Fixed in the notebook: case118 `LINES_TO_ATTACK` was still the retired v1 list (with bridge
  line 72); §7 now uses v2.

### Snapshot 2026-09-21 10:20 (session 3)

| Job | What | State |
|---|---|---|
| `7009596` | case36 array, 864 tasks | **207 COMPLETED / 27 TIMEOUT / 40 running / 590 pending** (next: 274-863, i.e. february→january→…); 266 shards, 4,427 rows, 2,024 unique actions; 0 real errors |
| `7036197` | case118 array, 748 tasks | **210 COMPLETED / 2 TIMEOUT / 40 running / 496 pending** (next: 252-747, i.e. ~May onward); 228 shards, 1,414 rows, 998 unique actions; 0 real errors |
| `7035681` | case118 smoke test v2 | Done (3 h 48): chronic `2050-01-03_1` ended at step 361 (teacher game-over), **5 sane rows** |

**⚠ case36 winter chronics blow the 36 h walltime.** All 27 TIMEOUTs are december/february tasks
(144-228); april/august tasks took 3-16 h, but winter chronics take ~12-18 h *each* because the
grid sits in the stress band far more often and the teacher survives longer (many run to 8062).
Completed-task elapsed is now p50 7.2 h / p90 29.6 h / max 35.4 h — the 36 h budget (sized off
the april smoke test) has no margin for winter. Killed tasks had finished 1-2 of their 3 chronics
and were part-way (200-6,000 steps) through the last one; the partial rows are kept, but the
third chronic of each is truncated. **january (tasks 286-357) is next in the queue and will
time out at a similar ~30% rate**; november/march may too. Options: (a) accept truncation,
(b) after the array ends, resubmit the unfinished chronics as a follow-up array at 72 h,
(c) cancel the pending range and resubmit it at `SBATCH_TIME=72:00:00` now (SLURM won't let a
user raise the limit of already-queued tasks). case118's 2 TIMEOUTs (tasks 29, 70, both january)
are the same pattern but at 2/212 it is not a problem there.

Throughput so far: case36 ≈ 16 slot-h/task → remaining 590 tasks ≈ **~10 days → ~2026-10-01**;
case118 ≈ 13.6 slot-h/task → remaining 536 tasks ≈ **~7.6 days → ~2026-09-29**. Both at
concurrency 40.

Note on case118 ranking: its frequency curve is very flat (top action seen 9× of 1,414 rows,
998 unique) vs. case36 (134× of 4,427). With ~5× more rows to come it may sharpen, but choosing
k by "knee" on case118 may need a coarser key (e.g. substation) — revisit at aggregation.

### Snapshot 2026-09-18 09:30 (session 2)

| Job | What | State |
|---|---|---|
| `7009596` | case36 array, 864 tasks | **127 COMPLETED / 40 running / 697 pending**, 0 failures; 151 shards, ~1,000 rows, 546 unique actions |
| `7010778` / `7010779` | N-1 conversion diagnostics | Done — case118 **0/8**, case36 8/8. Root cause found, see §4b |
| `7007452` | case118 smoke test (old lines) | Done after 723 steps: **16 N-1 hits → 0 rows**. Confirms the diagnostic |
| `7035679` | case118 diagnostic **v2** (bridge-free lines) | Done (54 min): **8/8 would-save**, conversion 1.0, 23-100 finite candidates per probe |
| `7035681` | case118 smoke test **v2** (bridge-free lines) | Running, `results/teacher/2026_09_18_smoketest_case118_v2/` |
| `7036197` | **case118 array**, 748 tasks, chunk 2, 44 h, `%40` | Launched 2026-09-18 10:15 → `results/teacher/2026_09_18_teacher_n1_case118/` |

case36 array is finishing **much faster than sized**: completed tasks take 3-16 h (median ~9 h),
not the ~22 h budgeted, because the teacher's own actions frequently end a chronic early (game
over, "Search stopped at N/8062"; median stop step 3279, only 26% of chronics run to 8062). Ops
rate ≈ 5.8 tasks/h at concurrency 40 → the whole array should finish in **~6 days (≈ 2026-09-23)**,
not 20.

### Snapshot at handoff (2026-09-17 12:18)

| Job | What | State |
|---|---|---|
| `7009596` | case36 array, 864 tasks | 40 running / 824 pending; **24 shards, 40 rows** so far |
| `7010778` | case118 N-1 conversion diagnostic | Running (started 12:15, ~1 h) |
| `7010779` | case36 N-1 conversion diagnostic (control) | Running |
| `7007231` | case36 smoke test | Running, step 3786/8062, 9 N-1 hits → **6 rows** |
| `7007452` | case118 smoke test | Running, step 410/2017, 2 N-1 hits → **0 rows** |

Refined conversion baseline: case36 converts N-1 hits to rows at **~67% (6/9)**, not the 100%
its first two hits suggested. That is the realistic figure to judge case118 against.

---

## 1. Cluster limits (BWUniCluster, confirmed 2026-09-17)

```
$ scontrol show config | grep -i MaxArraySize
MaxArraySize            = 1001
```

So the highest usable array index is 1000 → **at most 1001 array tasks**. The previously-guessed
`max_array_size=500` in `teacher_n1_array.sh`'s example was unnecessarily conservative; the
script's docs and example now record the real limit.

`cpu_il,cpu` partitions cap walltime at **72 h**.

Chronic counts (train split):

| Grid    | Grid2Op env            | Chronics |
|---------|------------------------|----------|
| case36  | `l2rpn_wcci_2020_train` | 2592     |
| case118 | `l2rpn_wcci_2022_train` | 1496 (1497 dir entries incl. `errors.json`) |

## 2. `lines_to_attack`

| Grid    | Lines                            | Source |
|---------|----------------------------------|--------|
| case36  | `20 13 39 33 35 32 11 34 23 22`  | Pre-existing (reused, not recomputed) |
| case118 (v1, **retired**) | `146 136 147 156 135 149 72 20 154 12` | `select-lines --grid case118` (2026-09-17). Contains bridge line 72 → N-1 search can never succeed, see §4b |
| case118 (v2) | `146 136 147 135 156 149 20 154 41 12` | `select-lines --grid case118` after bridge exclusion (2026-09-18) |

case118's selection runs in well under a minute on the login node and is saved at
`results/teacher/phase0_case118/lines_to_attack.txt` (v1 kept as `lines_to_attack_v1_with_bridge72.txt`).

## 3. Phase 0 — cost per teacher interaction

### case36 (`results/teacher/phase0_case36/benchmark_result.json`)

- 66,810 unitary actions; 1 chronic consumed; 25 timed interactions.
- **greedy fallback**: n=24, mean 109.4 s, p50 99.0 s, **p90 232.7 s**, max 239.9 s
- **N-1 search**: n=1, 235.8 s
- Job wall time: 50 min (job 6974764).

Confirms known-fact #3: with `active_search=True`, 24 of 25 timed interactions took the greedy
branch, only 1 the true N-1 search. Not re-run — still representative (same env, same lines,
same teacher config).

### case118 (`results/teacher/phase0_case118/benchmark_result.json`)

- **72,957 unitary actions** (vs 66,810 on case36); 1 chronic consumed; 25 timed interactions.
- **greedy fallback**: n=25, mean 237.0 s, p50 316.4 s, **p90 340.1 s**, max 350.0 s
- **N-1 search**: n=0 — the branch did not fire once in 25 interactions (case36 got 1/25).
- Job wall time: 1 h 40 min (job 7007449).

Greedy is ~1.5× more expensive per interaction than case36 at p90 (340 s vs 233 s). Note mean
(237 s) sits *below* p50 (316 s) on both grids — the branch is bimodal, with a cluster of cheap
calls (~13-15 s) pulling the mean down. **p90 is the right statistic for walltime sizing**, as
the script header already advises.

## 4. Smoke tests

Per known-fact #2, one real non-array job per grid on one chronic, exercising the
`run-teacher-chronics` → `env.set_id()` → `n_minus_one_agent` path that Phase 0 does *not* cover.

| Grid    | Job     | Experiment dir | Chronic | Status |
|---------|---------|----------------|---------|--------|
| case36  | 7007231 | `2026_09_17_smoketest` | `Scenario_april_000` (8062 steps) | **PASSED** — sane rows, see below |
| case118 (v1 lines) | 7007452 | `2026_09_17_smoketest_case118` | `2050-01-03_1` (2017 steps) | Done, 8 h: **16 N-1 hits → 0 rows**. Explained by §4b |
| case118 (v2 lines) | 7035681 | `2026_09_18_smoketest_case118_v2` | `2050-01-03_1` | Running since 2026-09-18 09:18 |

### ⚠ case118 risk: N-1 branch fires but finds nothing to save

At 11:52 the case118 smoke test had reached step 402/2017 with **2 N-1-branch hits, 34 greedy,
and still 0 CSV rows**. case36 for comparison converted its first 2 N-1 hits into 2 rows (100%).

This is a *second*, distinct failure mode on top of the known rarity. Hitting the N-1 branch is
necessary but not sufficient — `search_best_n_minus_one_action` (line 143) only sets
`n_1_action_found = True`, and hence only saves, if it gets past **both** of these:

1. `action_set` — the top-100 greedy candidates filtered to those that actually pull
   `rho` back below `rho_n0 = 0.95`. Empty if no single rewiring is enough.
2. At least one surviving candidate must score **finite** in `calculate_attacked_max_rho`, i.e.
   survive the disconnection of all 10 `lines_to_attack` without a game over. Any candidate that
   blacks out under some contingency scores `inf` and is disqualified; if *every* candidate
   does, `n_1_action_found` stays `False` and nothing is written.

On a larger, more fragile 118-bus grid, (2) failing for every candidate is entirely plausible —
as is (1), since one substation rewiring has proportionally less influence on a bigger network.

Sample size is only 2, so this is not yet conclusive. But it is the reason the case118 array
stays held: if case118 converts N-1 hits to rows at ~0%, its ~18,000 CPU-h would produce an
empty or near-empty action space. **Confirm a non-zero conversion rate before launching it.**

If the rate does turn out to be ~0, the levers (all scope decisions, none taken unilaterally)
would be: widen the `rho_n0`/`rho_max` band, shrink `lines_to_attack` from 10 to fewer/less
severe contingencies so fewer candidates get disqualified, or set `save_greedy=True` and accept
greedy actions into the ranking.

### Diagnostic built to settle this (jobs 7010778 / 7010779)

Waiting for the smoke test to accumulate enough N-1 hits to estimate a conversion rate would
take days, so the two failure modes are measured directly instead:
`src/action_space_generation/n1_conversion_diagnostic.py`, exposed as
`build_action_space.py diagnose-n1`.

It steps a **do-nothing** agent (milliseconds/step) until the grid naturally enters the
`[0.95, 1.0)` band with all lines in service — the teacher's own N-1 gate — then runs the two
filters once and counts survivors at each, mirroring `search_best_n_minus_one_action`
lines 154-180. Skipping the teacher's ~340 s greedy sweep *per step* is what turns this from a
multi-day question into a ~1 h one.

```bash
PYTHONPATH=$(pwd)/src python experiments/build_action_space.py diagnose-n1 \
    --grid case118 --lines-to-attack 146 136 147 156 135 149 72 20 154 12 \
    --n-probes 8 --out results/teacher/n1_diag_case118/diagnostic.json
```

Reported per grid: `n_would_save` / `conversion_rate`, plus the split between
`n_failed_stage1_no_candidate_fixes_rho` and `n_failed_stage2_all_candidates_blackout`. The two
have **opposite fixes** — Stage 1 points at `rho_n0`/the action set, Stage 2 at
`lines_to_attack` being too severe — so distinguishing them prevents tuning the wrong knob and
burning another multi-day run to discover it.

**case36 is run as a control.** It is known to convert (2 N-1 hits → 2 rows), so its Stage-1 /
Stage-2 counts define what "healthy" looks like and make case118's numbers interpretable rather
than merely alarming.

Caveat recorded in the module docstring: do-nothing reaches a somewhat different distribution of
in-band states than the teacher's own trajectory, so the output answers "can these filters pass
at all on this grid, and roughly how often" — not the teacher's exact conversion rate.

### 4b. Root cause of case118's 0% conversion: a bridge line in `lines_to_attack` (2026-09-18)

Both diagnostics finished (`results/teacher/n1_diag_case118|case36/diagnostic.json`):

| Grid | probes | would_save | Stage-1 failures | Stage-2 failures |
|---|---|---|---|---|
| case36 (control) | 8 | **8** (100%) | 0 | 0 — 84-95 of 100 candidates finite |
| case118 | 8 | **0** (0%) | 0 | **8** — 0 of 26-100 candidates finite |

Stage 1 is healthy on case118 (26-100 candidates pull rho under 0.95). Stage 2 rejects **every**
candidate, every time — which smells like one contingency that is lethal regardless of the
topology action rather than "the grid is fragile". Checked directly by simulating each
`lines_to_attack` disconnection under do-nothing at the initial observation, plus a graph
connectivity check on the substation graph:

| line | name | subs | bridge? | simulate → done? |
|---|---|---|---|---|
| 146 136 147 156 135 149 20 154 12 | … | … | no | no (rho_max 0.82-1.46) |
| **72** | `109_110_72` | 109-110 | **yes** | **yes** — `Divergence of DC powerflow (non connected grid)` |

**Line 72 is the only connection of substation 110.** Disconnecting it islands the substation, so
`calculate_attacked_max_rho` returns `inf` for *every* action, and `search_best_n_minus_one_action`
can never set `n_1_action_found`. No `rho_n0`/`rho_max`/`save_greedy` change would have helped;
this was a bug in `select_lines_to_attack`, which ranked lines purely by do-nothing stress and had
no notion of grid topology. (Radial lines to a generator bus are naturally "stressed" — that is
exactly how 72 got picked.) case36's ten lines are all non-bridges, which is why it was fine.

The old smoke test (7007452) independently confirms: 16 N-1-branch hits over 723 steps, 0 rows.

**Fix:** `line_selection.py` now has `find_bridge_lines(env)` (removes each line from the
substation multigraph and checks connectivity; parallel lines are handled) and
`select_lines_to_attack` excludes them. IEEE-118 has **9 bridges**
(6, 26, 72, 73, 80, 129, 140, 176, 177); IEEE-14 has 1 (bus 8's line), tested in
`src/tests/action_space_generation/test_line_selection.py`.

Re-running `select-lines --grid case118` (21 s, login node):

```
146 136 147 135 156 149 20 154 41 12
```

Same set as before minus 72, plus `41` (`79_95_41`, non-bridge, simulate rho_max 0.89). Saved to
`results/teacher/phase0_case118/lines_to_attack.txt`; the old list kept as
`lines_to_attack_v1_with_bridge72.txt`. Slightly different ordering vs. v1 is chronic-sampling
noise (the rollout is unseeded) and irrelevant — the teacher iterates the whole list.

Re-verification before launching the array (both submitted 09:18, running in parallel):

- `7035679` diagnostic v2 → `results/teacher/n1_diag_case118_v2/diagnostic.json` (~45 min)
- `7035681` smoke test v2 → `results/teacher/2026_09_18_smoketest_case118_v2/shard_smoketest.csv`
  (24 h walltime; needed to see *sane rows* on case118 per known-fact #2, not just conversion)

**Diagnostic v2 result (`results/teacher/n1_diag_case118_v2/diagnostic.json`): 8/8 would-save,
conversion 1.0**, Stage-2 survivors 23-100 per probe — on the *same* chronics/steps
(`2050-01-03_10@149-151`, `2050-01-03_13@84-89`) that scored 0 finite with line 72. That is the
launch criterion from HANDOFF, so the case118 array was launched (job 7036197, §6). The v2 smoke
test keeps running as the sanity check on row *contents*; if its first rows look wrong the array
is cancelled (`scancel 7036197`) with little lost.

**First case118 row (array shard_4, ~1 h after launch) — sane.** `rho_max_old 0.9617 →
rho_max_new 0.9237` (improvement 0.038, above `filter_good_experience`'s 2%), `best_action ==
top_0_action`, decodes to a single-substation bus split at **sub 76** — the substation where
attacked lines 20 (`76_81_20`) and 12 (`68_76_12`) terminate, i.e. a plausible N-1 hardening
move. `verify_round_trip` passes. Known-fact #2 is therefore satisfied for case118 as well.

### case36 smoke test — PASSED

First rows appeared at steps 941 and 948 (both N-1-branch hits). Contents are sane:

| rho_max_old | rho_max_new | rho_improvement | best_action == top_0_action |
|-------------|-------------|-----------------|------------------------------|
| 0.9557      | 0.6901      | 0.2656          | yes |
| 0.9677      | 0.7083      | 0.2593          | yes |

Both are large genuine improvements (~26%), far above `filter_good_experience`'s 2% threshold,
and `best_action` decodes as a valid base64 `EncodedTopologyAction`.

**The whole downstream path was then validated end-to-end on these two rows** (dry run to
`/tmp/dryrun_case36.json`, not committed):

```
2 unique actions across 1 experience file(s)
Top-2 actions touch 2 distinct substations:
4     1
23    1
Wrote 2 actions to /tmp/dryrun_case36.json
```

`aggregate --out` runs `verify_round_trip` before writing and did not raise, so
`load_experience` → `filter_good_experience` → `rank_actions_simple` → `substation_distribution`
→ `verify_round_trip` → `export_action_space` all work on real teacher output. Step 5 is
de-risked ahead of the array finishing.

One format note: the exported dicts address elements by **name** (`"4_6_5"`, `"load_23_24"`)
where the hand-curated case14 spaces use integer ids. Both are accepted by
`env.action_space(dict)` — the round-trip check confirms it — so this is cosmetic.

### Measured per-chronic cost (sustained rate, from the smoke-test logs)

Measured over the *active* region (excluding startup and the fast pre-stress steps), which is
the rate that should be extrapolated:

| Grid    | Step range measured | sec/step | sec/interaction | interactions/step | Chronic length | **Projected h/chronic** |
|---------|---------------------|----------|-----------------|-------------------|----------------|--------------------------|
| case36  | 246 → 2302          | 3.25     | 113.3           | 0.029             | 8062           | **7.3 h** |
| case118 | 38 → 388            | 16.83    | 184.0           | 0.091             | 2017           | **9.4 h** |

case36's 7.3 h independently reproduces the figure quoted in `teacher_n1_array.sh`'s header —
a good consistency check on the pre-existing cost model.

case118 chronics are ~4× shorter but ~5× slower per step *and* hit the teacher ~3× more often
per step, so they land slightly more expensive per chronic than case36.

## 5. Note on row yield (`save_greedy=False`)

Reading `curriculumagent/teacher/teachers/teacher_n_minus_1.py:247-295`: within
`n_minus_one_agent`, a row is written **only** when the N-1-search branch fires *and* finds an
action (`save_action = True`). The greedy-fallback branch sets `save_action = save_greedy`, and
`teacher_runner.run_teacher_chronics` hardcodes `save_greedy=False` — so greedy steps, which
known-fact #3 says dominate under `active_search=True`, contribute **no rows at all**.

Mechanism, from `do_nothing_and_run_through_lines_action` (line 82) and the step loop:

1. `rho_max < 0.95` and all lines in service → active search simulates disconnecting each
   `lines_to_attack` line and returns the disconnect that pushes rho *highest within*
   `(0.95, 1.0)` without a game over; `{}` if none qualifies. It deliberately stresses the grid
   into the N-1 band.
2. Next step, rho is in the band — but a line is now out, so
   `elif rho_max < rho_threshold and all(obs.line_status)` **fails** and control falls to greedy,
   which saves nothing.
3. The N-1 branch therefore only fires when rho drifts into `[0.95, 1.0)` with *all* lines still
   in service — the case active search works to pre-empt.

So the low N-1 rate is structural, not a misconfiguration. **Yield is still adequate**, though:
case36 Phase 0 saw 1 of 25 stressed interactions take the N-1 branch (~4%), and the case36 smoke
test is running ~9 stressed interactions per 500 steps over an 8062-step chronic → on the order
of 140 stressed interactions/chronic → roughly 5 saved rows/chronic → ~10-13k rows over 2592
chronics. Comfortably enough to rank a top-k in the hundreds. Confirmed against the actual smoke
test CSVs in section 4.

## 5b. Compatibility check: no explicit do-nothing in the teacher export

`grid2op_env/action_converters.load_actions` passes each JSON dict straight to
`env.action_space(dict)` and `setup_converter` feeds the result to `IdToAct.init_converter(
all_actions=...)`, which uses exactly the supplied list — nothing is prepended. The export
format therefore matches (`as_serializable_dict()` is the same shape), but note:

- Every `rl2grid_*` action space — including the two currently configured,
  `rl2grid_bus36-M_1` (302 actions) and `rl2grid_bus118-M_1` (308) — has `{}` (do-nothing) at
  index 0.
- The hand-curated case14 spaces (`medha`, `tennet`, `assym`, `d3qn2022`) do **not**.
- The teacher export ranks *selected topology actions* by frequency, so it will **not** contain
  a do-nothing entry.

Left as-is rather than silently prepending `{}`, since that would change what "top-k" means.
This repo's multi-agent setup routes do-nothing through a separate `DO_NOTHING_AGENT` policy
gated by `HIGH_LEVEL_AGENT`, so the RL agent's own list not containing it is most likely fine —
but it is a behavioural difference from the action spaces these configs point at today, and
worth a conscious confirmation before training on the result.

Incidental sanity check: `rl2grid_bus118-M_4.json` contains 72,461 actions, consistent with the
~72k unitary action set expected for IEEE-118.

## 6. Array sizing

`chunk_size = ceil(n_chronics / max_array_size)`, `n_tasks = ceil(n_chronics / chunk_size)`,
and `n_tasks` must stay ≤ 1001.

| Grid    | Chronics | max_array_size | chunk_size | n_tasks | h/chronic (p90 basis) | worst-case/task | `SBATCH_TIME` | concurrency |
|---------|----------|----------------|------------|---------|------------------------|------------------|----------------|-------------|
| case36  | 2592     | 1000           | 3          | **864** | ~7.3 h                 | ~21.9 h          | **36:00:00**   | 40 |
| case118 | 1497     | 1000           | 2          | **749** | ~17.4 h                | ~34.8 h          | **44:00:00**   | 40 |

Rationale:

- **`max_array_size=1000`**, not the script's old `500` guess, because the real `MaxArraySize`
  is 1001. Bigger max_array_size → smaller chunks → shorter, more schedulable tasks.
- **case36's `chunk_size=3` is forced, not chosen.** `chunk_size=2` would need
  `ceil(2592/2) = 1296` tasks, which exceeds the 1001 cap. 3 is the smallest legal chunk.
- **case118's `chunk_size=2`** gives 749 tasks, comfortably under the cap.
- **Walltimes** budget the tail, not the mean, since a task killed mid-chronic truncates towards
  the start of that chronic and biases the action set. case36: 3 × 7.3 h ≈ 22 h → 36 h (~60%
  margin). case118: its p90 greedy cost (340 s) × ~184 interactions/chronic ≈ 17.4 h, × 2 ≈
  35 h → 44 h (~26% margin). Both stay under the partition's 72 h cap.
- **Concurrency 40** on both, per the "tens, not hundreds" guardrail on a shared cluster.

A pre-flight check was added to `teacher_n1_array.sh` so an over-wide array now fails locally
with a message naming `MaxArraySize`, instead of being rejected by `sbatch`.

### Launched

| Grid    | Job     | Experiment dir | Submitted | Status |
|---------|---------|----------------|-----------|--------|
| case36  | 7009596 | `results/teacher/2026_09_17_teacher_n1_case36` | 2026-09-17 11:37 | 864 tasks, running; 127 done by 2026-09-18 09:00, 0 failures |
| case118 | 7036197 | `results/teacher/2026_09_18_teacher_n1_case118` | 2026-09-18 10:15 | **748 tasks** (1496 chronics — the 1497 above counted `errors.json`; chunk 2), `SBATCH_TIME=44:00:00`, `%40`, lines v2 |

case118's array is deliberately **not** submitted yet: known-fact #2 requires confirming sane
rows per grid first, and its smoke test had produced 0 rows as of 11:34.

**case36 array health check (11:44, 40 tasks started):**

- Chunk assignment is correct and non-overlapping: task 0 → chronics 1-3, task 1 → 4-6,
  task 2 → 7-9; across all started tasks, 120 chronics assigned / 120 unique (no double work).
- All 40 tasks are stepping the teacher normally.
- **Benign-error note for future triage:** every task's `.err` opens with
  `Traceback ... ModuleNotFoundError: No module named 'conda'`. That comes from the
  `module load devel/miniforge` shim calling its `condabin/conda` wrapper; the subsequent
  `eval "$(conda shell.bash hook)"` + `conda activate curriculum` works fine and the job runs.
  It is noise, not a failure — grepping these 864 `.err` files for `Error` will match all of
  them, so filter it out when hunting real failures.

## 6b. ⚠ Total resource cost — needs a decision

This is the headline number and it is large:

| Grid    | CPU-hours              | Wall-clock at concurrency 40 |
|---------|------------------------|-------------------------------|
| case36  | 2592 × 7.3 h ≈ **18,900** | 864/40 × 21.9 h ≈ 473 h ≈ **20 days** |
| case118 | 1497 × ~12 h ≈ **18,000** | 749/40 × 34.8 h ≈ 652 h ≈ **27 days** |
| **Total** | **≈ 37,000 CPU-h**    | ~27 days if both run concurrently (80 cores) |

Per the task's guardrail, scope was **not** silently shrunk to make this fit — no chronics were
cut. Reporting it instead. Points for the decision:

- The footprint is steady, not spiky: 40 cores per grid held for weeks, which backfills
  reasonably on a shared cluster, but it *is* weeks.
- **Concurrency is the lever and it scales linearly.** 40 → 100 takes case36 from ~20 days to
  ~8. The "tens, not hundreds" guardrail is what capped it at 40; raising it is a one-argument
  change and needs no re-sizing of chunks or walltime.
- Cutting chronic count is the other lever, and it trades directly against action-set coverage.
  That is a scope decision, so it is flagged here rather than taken.
- Nothing is lost by deciding late: `save_sample_new` appends row-by-row, so the arrays can be
  cancelled at any point and the partial shards still aggregate — just over fewer chronics.

## 7. Chosen k and final artifacts

_Pending._

## Open items

- [x] Confirm real `MaxArraySize` (1001)
- [x] case118 `lines_to_attack`
- [x] case118 Phase 0 benchmark (job 7007449)
- [x] case36 smoke test sane rows (job 7007231)
- [x] Validate aggregate → round-trip → export end-to-end on real teacher rows
- [x] Size + launch case36 array (job 7009596)
- [x] case118 smoke test v1 (job 7007452) — 0 rows; root-caused to bridge line 72 (§4b)
- [x] Fix `select_lines_to_attack` to exclude bridges; new case118 lines
- [x] case118 diagnostic v2 (job 7035679): 8/8, conversion 1.0
- [x] case118 sane rows — first array row verified (sub-76 split, rho 0.962→0.924, round-trip OK);
      smoke test v2 (7035681) still running as extra check
- [x] Launch case118 array (job 7036197, 2026-09-18 10:15)
- [x] case118 array finished (2026-09-28): 743 COMPLETED / 5 TIMEOUT
- [x] Preliminary aggregation (`aggregate_shards.py`) + analysis notebook re-run on 09-30 data
- [ ] case36 array to finish (125 pending, ETA ~2026-10-02/03)
- [ ] **Decision needed:** rerun the 159 case36 TIMEOUT tasks' chronics at 72 h (recommended), or
      accept the winter under-sampling (see 2026-09-30 snapshot)
- [ ] **Decision needed:** how to pick k for case118 given its noisy ranking (small k, coverage
      target, or substation-stratified selection — notebook §3, §8, §10)
- [ ] Aggregate + choose k per grid
- [ ] Point `configs/rllib/env/case36.yaml` / `case118.yaml` at the new files

### Files changed so far

- `slurm/unicluster/{benchmark_teacher_cost,teacher_n1_smoketest,teacher_n1_array}.sh` —
  walltime made overridable via `SBATCH_TIME` (previously hardcoded); real `MaxArraySize`
  documented; pre-flight array-width check added.
- `src/action_space_generation/n1_conversion_diagnostic.py` — **new**, see §4.
- `src/action_space_generation/line_selection.py` — bridge lines excluded from `lines_to_attack`
  (`find_bridge_lines`), see §4b; tests added in `src/tests/action_space_generation/test_line_selection.py`.
- `experiments/build_action_space.py` — added the `diagnose-n1` subcommand wiring it up.
- `results/teacher/SUMMARY.md` — this file.
- `results/teacher/aggregate_shards.py` — concatenates shards into `aggregated/experience_<grid>.csv`;
  now also writes per-task SLURM states (`aggregated/task_states_<grid>.csv`, via `sacct`).
- `experiments/teacher_action_space_analysis.ipynb` — k-selection analysis; restructured and
  re-run 2026-09-30 (see snapshot).
- Configs and `data/action_spaces/` are **untouched** so far; they change in steps 5-6.
- Nothing committed, per instructions.
