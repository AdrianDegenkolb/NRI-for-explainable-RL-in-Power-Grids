# Results

This folder is the output root for all experiment artefacts. It is not committed to git
(see `.gitignore`). The structure below reflects what the **current codebase** produces.

---

## `experiments/`

Written by `train.py` (via Ray Tune). Each SLURM script creates a dated subdirectory:

```
experiments/
└── <YYYY_MM_DD>_<grid>/       # e.g. 2026_05_28_IEEE14
    └── <variant>/             # e.g. rappo, ppo_mlp, radqn
        ├── <trial_dir>/       # Ray Tune trial directory (one per seed)
        │   ├── checkpoint_<N>/    # RLlib checkpoint (policy weights)
        │   └── result.json        # per-iteration metrics
        ├── checkpoint_results.json  # summary of best checkpoints + eval metrics
        └── out/               # SLURM stdout/stderr logs (*.log)
```

The variant name and date come from the `experiment.name` Hydra key set inside each SLURM
script.

## `evaluations/`

Written by the post-training evaluation step in `train.py` and by
`evaluate_heuristic_agents.py`. Organised by model name and environment:

```
evaluations/
└── <model_name>/
    └── <checkpoint_id>/
        └── <env_name>/     # e.g. l2rpn_case14_sandbox_test
            └── *.json / *.npy   # per-episode metrics
```

## `cross_validation/`

Written by `compare_models.py`. Contains the raw data arrays and figures produced by the
cross-validation analysis:

```
cross_validation/
├── cross_validate_models.json   # raw results for all agent pairs
├── cross_validate_models.{svg,png}   # rescue heatmap figure
├── failing_edges_connectivity.{svg,png}
├── failing_edges_rho.{svg,png}
├── cv_*.npy                     # intermediate numpy arrays (backup agents, maps, etc.)
└── reconfiguration_frequency_*.{svg,png}
```

## `hypothesis_electrical_coupling/`, `hypothesis_risk_coupling/`, `hypothesis_action_effect/`

Written by `analyze_latent_graphs.py` — one folder per hypothesis (H1, H2, H3). Each
contains intermediate `.npy` data arrays and the final figure files (`.png` / `.svg`)
produced by the corresponding verifier.
