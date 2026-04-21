import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score

from analysis.analyze_latent_graphs.build_coupling_matrices import get_risk_vector
from grid2op_env.observation_converter import GraphObservationConverter, EDGE_INDEX
from analysis.analyze_latent_graphs.agent_analysis_framework import PosteriorAnalyzer
from visualization import visualize_graph, PlottingArgs, get_node_styles

logger = logging.getLogger(__name__)

class Hypothesis2verifier(PosteriorAnalyzer):
    """
    Computes per-timestep alignment metrics between posterior edge existence probabilities
    and risk coupling (edge-aligned, fully-connected ordering).

    Metrics per timestep:
      - Spearman rho
      - Pearson r
      - Kendall tau
      - Top-k overlap (default: top 10%)
      - ROC-AUC for predicting "strong Risk coupling" edges (threshold: top 10% of coupling)
      - Average Precision (AP) for same labeling (often more informative for imbalanced labels)
      - Mutual information MI(P -> C)

    Plus edge-wise temporal correlation (Pearson) over time:
      corr_e = corr_t(P_t,e, C_t,e) for each edge e (ignoring NaNs)
    """

    def __init__(
        self,
        out_dir: Path = Path("results/hypothesis_risk_coupling"),
        topk_frac: float = 0.05,
        strong_label_percentile: float = 90.0,
        min_samples_per_timestep: int = 10,
        min_timesteps_per_edge: int = 5,
        use_abs_coupling_for_metrics: bool = False,
        random_state_mi: int = 0,
    ):
        self.topk_frac = float(topk_frac)
        self.strong_label_percentile = float(strong_label_percentile)
        self.min_samples_per_timestep = int(min_samples_per_timestep)
        self.min_timesteps_per_edge = int(min_timesteps_per_edge)
        self.use_abs_coupling_for_metrics = bool(use_abs_coupling_for_metrics)
        self.random_state_mi = int(random_state_mi)
        # Results go into a mode-specific subdirectory so both modes can coexist
        self.outdir = out_dir / "risk_coupling"

        # Accumulated over time: risk vectors r(s_t) of shape [n_nodes] each
        self._risk_vectors: list[npt.NDArray] = []   # list of [n_nodes]
        # Accumulated over time: posterior and prior edge-existence probabilities shape [E] each
        self._posterior_over_time: list[npt.NDArray] = []  # list of [E]
        self._prior_over_time: list[npt.NDArray] = []      # list of [E]  (baseline)

        # Stored on first on_rl_step call for use in generate_plots
        self._environment: Environment | None = None
        self._powerline_edge_index: npt.NDArray | None = None

        # Optional: episode boundaries (if you want later)
        self._episode_ids = []
        self._t_global = 0

    @property
    def _coupling_label(self) -> str:
        """Human-readable label for what coupling is being compared against."""
        return "Risk coupling"

    @property
    def _coupling_short(self) -> str:
        """Short token used in file names."""
        return "risk_coupling"

    @staticmethod
    def _nan_pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson with NaN filtering; returns np.nan if insufficient samples or zero variance."""
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        x0 = x[m]
        y0 = y[m]
        sx = x0.std()
        sy = y0.std()
        if sx < 1e-12 or sy < 1e-12:
            return np.nan
        return float(np.corrcoef(x0, y0)[0, 1])

    @staticmethod
    def _topk_overlap(c: np.ndarray, p: np.ndarray, frac: float) -> float:
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        n = c.shape[0]
        if n < 2:
            return np.nan
        k = max(1, int(frac * n))
        # indices of top-k
        top_c = np.argpartition(c, -k)[-k:]
        top_p = np.argpartition(p, -k)[-k:]
        overlap = np.intersect1d(top_c, top_p).size / k
        return float(overlap)

    @staticmethod
    def _auc_metrics(c: np.ndarray, p: np.ndarray, percentile: float):
        """
        Binarize coupling by percentile, compute ROC-AUC and AP. Returns (auc, ap) possibly NaN.
        """
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        if c.size < 10:
            return np.nan, np.nan

        thr = np.percentile(c, percentile)
        y = (c >= thr).astype(np.int32)
        # Need both classes present
        if y.min() == y.max():
            return np.nan, np.nan
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = np.nan
        try:
            ap = float(average_precision_score(y, p))
        except ValueError:
            ap = np.nan
        return auc, ap

    def _mi(self, c: np.ndarray, p: np.ndarray) -> float:
        """
        Mutual information between p (as feature) and c (target).
        Uses sklearn mutual_info_regression.
        """
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        if c.size < 10:
            return np.nan
        # MI expects X: (n_samples, n_features)
        X = p.reshape(-1, 1)
        try:
            mi = mutual_info_regression(X, c, random_state=self.random_state_mi)
            return float(mi[0])
        except Exception:
            return np.nan

    def on_rl_step(
        self,
        posterior: npt.NDArray,
        prior: npt.NDArray,
        powergrid_graph: npt.NDArray,
        observation: BaseObservation,
        environment: Environment,
        action: BaseAction,
    ):
        # Capture env and powerline edge index once for later graph visualization
        if self._environment is None:
            self._environment = environment
            obs_space = GraphObservationConverter(g2op_obs_space=environment.observation_space)
            self._powerline_edge_index = obs_space.to_gym(environment.current_obs)[EDGE_INDEX]

        # Accumulate per-node risk vector r(s_t): shape [n_nodes]
        r = get_risk_vector(environment).astype(np.float64)
        self._risk_vectors.append(r)

        # Accumulate posterior and prior edge-existence probability: shape [E]
        self._posterior_over_time.append(posterior[:, 0].astype(np.float64))
        self._prior_over_time.append(prior[:, 0].astype(np.float64))

        self._t_global += 1

    def on_heuristic_step(self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    def on_new_episode(self, chronic_id: str):
        self._episode_ids.append(chronic_id)

    @staticmethod
    def _save_array(arr: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    @staticmethod
    def _save_hist(values: np.ndarray, title: str, xlabel: str, outpath: Path,
                   vline: float | None = None, vline_label: str | None = None):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        finite_values = values[np.isfinite(values)]
        sns.histplot(finite_values, kde=False)
        mean_val = float(np.mean(finite_values)) if finite_values.size > 0 else None
        if mean_val is not None:
            plt.axvline(mean_val, color="blue", linestyle="-.", label=f"Mean = {mean_val:.3f}")
        if vline is not None:
            plt.axvline(vline, color="red", linestyle="--",
                        label=vline_label if vline_label else f"x = {vline:.3f}")
        if vline is not None or mean_val is not None:
            plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outpath.parent / (outpath.stem + ".png"))
        plt.savefig(outpath.parent / (outpath.stem + ".svg"))
        plt.show()

    @staticmethod
    def _save_scatter(values: np.ndarray, title: str, xlabel: str, ylabel: str, outpath: Path):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        x, y = values
        mask = np.isfinite(x) & np.isfinite(y)
        plt.figure()
        plt.scatter(x=x[mask], y=y[mask], s=4, alpha=0.2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outpath.parent / (outpath.stem + ".png"))
        plt.savefig(outpath.parent / (outpath.stem + ".svg"))
        plt.show()

    def on_evaluation_end(self):
        T = self._t_global
        if T == 0:
            print("No timesteps recorded.")
            return

        # --- Build C_{ij}^{risk} = spearman_t(r_i(s_t), r_j(s_t)) ---
        R = np.stack(self._risk_vectors, axis=0).astype(np.float64)  # [T, N]
        N = R.shape[1]

        R_std = R.std(axis=0)
        zero_var = R_std < 1e-12

        C_node = np.full((N, N), np.nan, dtype=np.float64)
        for i in range(N):
            if zero_var[i]:
                continue
            for j in range(i, N):
                if zero_var[j]:
                    continue
                ri, rj = R[:, i], R[:, j]
                valid = np.isfinite(ri) & np.isfinite(rj)
                if valid.sum() < self.min_timesteps_per_edge:
                    continue
                rho = spearmanr(ri[valid], rj[valid]).correlation
                c = float(rho) if rho is not None else np.nan
                C_node[i, j] = c
                C_node[j, i] = c

        edge_index = fully_connected_edge_index(num_nodes=N)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        E = src.shape[0]
        C_edge = C_node[src, dst]
        C = np.abs(C_edge) if self.use_abs_coupling_for_metrics else C_edge

        # --- Time-averaged posterior and prior [E] ---
        P_T  = np.stack(self._posterior_over_time, axis=0)  # [T, E]
        PR_T = np.stack(self._prior_over_time, axis=0)      # [T, E]
        P_mean  = np.nanmean(P_T,  axis=0)                  # [E]
        PR_mean = np.nanmean(PR_T, axis=0)                  # [E]

        print(f"\n=== Hypothesis 2: Risk Coupling vs Posterior/Prior (T={T}, N={N}, E={E}) ===")

        def _scalar_metrics(c_v: np.ndarray, p_v: np.ndarray) -> dict:
            if c_v.size < self.min_samples_per_timestep:
                return {}
            results = {}
            rho = spearmanr(c_v, p_v).correlation
            results["Spearman rho"] = float(rho) if rho is not None else np.nan
            results["Pearson r"] = self._nan_pearson(c_v, p_v)
            tau = kendalltau(c_v, p_v).correlation
            results["Kendall tau"] = float(tau) if tau is not None else np.nan
            results[f"Top-{int(self.topk_frac*100)}% overlap"] = self._topk_overlap(c_v, p_v, self.topk_frac)
            auc, ap = self._auc_metrics(c_v, p_v, self.strong_label_percentile)
            results[f"ROC-AUC (p{int(self.strong_label_percentile)})"] = auc
            results[f"Avg Precision (p{int(self.strong_label_percentile)})"] = ap
            results["Mutual information"] = self._mi(c_v, p_v)
            return results

        valid_post = np.isfinite(C) & np.isfinite(P_mean)
        metrics_post = _scalar_metrics(C[valid_post], P_mean[valid_post])
        valid_prior = np.isfinite(C) & np.isfinite(PR_mean)
        metrics_prior = _scalar_metrics(C[valid_prior], PR_mean[valid_prior])

        # Derived distributions: edges removed p(1-q) and added q(1-p)
        removed_mean = PR_mean * (1.0 - P_mean)   # p(1-q): [E]
        added_mean   = P_mean  * (1.0 - PR_mean)  # q(1-p): [E]
        valid_removed = np.isfinite(C) & np.isfinite(removed_mean)
        metrics_removed = _scalar_metrics(C[valid_removed], removed_mean[valid_removed])
        valid_added = np.isfinite(C) & np.isfinite(added_mean)
        metrics_added = _scalar_metrics(C[valid_added], added_mean[valid_added])

        self.print_summary(metrics_post, metrics_prior, metrics_removed, metrics_added)

        # --- Save ---
        cs = self._coupling_short
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._save_array(R,        self.outdir / "risk_vectors_over_time.npy")
        self._save_array(C_node,   self.outdir / "C_risk_node_matrix.npy")
        self._save_array(C_edge,   self.outdir / f"{cs}_edge_vector.npy")
        self._save_array(P_mean,   self.outdir / "posterior_mean_edge.npy")
        self._save_array(PR_mean,  self.outdir / "prior_mean_edge.npy")
        self._save_array(P_T,      self.outdir / "posterior_over_time.npy")
        self._save_array(PR_T,     self.outdir / "prior_over_time.npy")
        np.save(self.outdir / "scalar_metrics_posterior.npy", metrics_post, allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_prior.npy",     metrics_prior, allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_removed.npy",   metrics_removed, allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_added.npy",     metrics_added, allow_pickle=True)

        self.generate_plots(C_edge, P_mean, PR_mean, P_T, PR_T, metrics_post, metrics_prior,
                            metrics_removed, metrics_added)

    @staticmethod
    def print_summary(metrics_post: dict, metrics_prior: dict,
                      metrics_removed: dict | None = None, metrics_added: dict | None = None):
        all_dicts = [d for d in [metrics_post, metrics_prior, metrics_removed, metrics_added] if d]
        metric_names = list(dict.fromkeys(k for d in all_dicts for k in d.keys()))
        col_w = max((len(n) for n in metric_names), default=20) + 2
        print(f"\n=== Hypothesis 2: Risk Coupling ===")
        header = (f"\n  {'Metric':<{col_w}}  {'Posterior':>18}  {'Prior (baseline)':>18}"
                  f"  {'Removed p(1-q)':>18}  {'Added q(1-p)':>18}")
        print(header)
        print("  " + "-" * (col_w + 80))
        for name in metric_names:
            def _fmt(d, k):
                v = d.get(k, float("nan")) if d else float("nan")
                return f"{v:>18.4f}" if np.isfinite(v) else f"{'NaN':>18}"
            print(f"  {name:<{col_w}}"
                  f"{_fmt(metrics_post, name)}{_fmt(metrics_prior, name)}"
                  f"{_fmt(metrics_removed, name)}{_fmt(metrics_added, name)}")

    def print_summary_from_saved(self):
        """Loads saved metric dicts and prints the summary table without regenerating plots."""
        m_post    = np.load(self.outdir / "scalar_metrics_posterior.npy", allow_pickle=True).item()
        m_prior   = np.load(self.outdir / "scalar_metrics_prior.npy",     allow_pickle=True).item()
        def _try_load_dict(path):
            return np.load(path, allow_pickle=True).item() if path.exists() else None
        m_removed = _try_load_dict(self.outdir / "scalar_metrics_removed.npy")
        m_added   = _try_load_dict(self.outdir / "scalar_metrics_added.npy")
        self.print_summary(m_post, m_prior, m_removed, m_added)

    def repaint(self):
        """Reads all previously saved arrays from outdir and regenerates all plots."""
        cs = self._coupling_short
        C_edge  = np.load(self.outdir / f"{cs}_edge_vector.npy")
        P_mean  = np.load(self.outdir / "posterior_mean_edge.npy")
        PR_mean = np.load(self.outdir / "prior_mean_edge.npy")
        P_T     = np.load(self.outdir / "posterior_over_time.npy")
        PR_T    = np.load(self.outdir / "prior_over_time.npy")
        m_post  = np.load(self.outdir / "scalar_metrics_posterior.npy", allow_pickle=True).item()
        m_prior = np.load(self.outdir / "scalar_metrics_prior.npy",     allow_pickle=True).item()
        def _try_load_dict(path):
            return np.load(path, allow_pickle=True).item() if path.exists() else None
        m_removed = _try_load_dict(self.outdir / "scalar_metrics_removed.npy")
        m_added   = _try_load_dict(self.outdir / "scalar_metrics_added.npy")
        self.generate_plots(C_edge, P_mean, PR_mean, P_T, PR_T, m_post, m_prior, m_removed, m_added)

    def _visualize_coupling_graph(
        self,
        C_edge: npt.NDArray,
        P_mean: npt.NDArray,
        PR_mean: npt.NDArray,
    ):
        """
        Renders and saves three graph visualisations:
          1. C_{ij}^{risk} (risk-based coupling, normalised)
          2. Time-averaged posterior existence probability
          3. Time-averaged prior existence probability  (baseline)
        """
        env = self._environment
        obs_space = GraphObservationConverter(g2op_obs_space=env.observation_space)
        node_styles = get_node_styles(env, obs_space.__class__)
        powerline_edge_index = self._powerline_edge_index
        N = 2 * env.n_line + env.n_gen + env.n_load
        cs = self._coupling_short

        C_finite = np.where(np.isfinite(C_edge), C_edge, np.nan)
        c_min, c_max = np.nanmin(C_finite), np.nanmax(C_finite)
        denom = c_max - c_min if (c_max - c_min) > 1e-12 else 1.0
        C_norm = np.where(np.isfinite(C_finite), (C_finite - c_min) / denom, 0.0)

        for probs, title, fname in [
            (np.stack([C_norm, 1.0 - C_norm], axis=1),
             r"Risk-based coupling $C_{ij}^{\mathrm{risk}}$ (normalised)",
             f"graph_{cs}.png"),
            (np.stack([P_mean, 1.0 - P_mean], axis=1),
             r"Mean posterior $\bar{q}_\phi(z_{ij})$",
             "graph_mean_posterior.png"),
            (np.stack([PR_mean, 1.0 - PR_mean], axis=1),
             r"Mean prior $\bar{p}_\phi(z_{ij})$ (baseline)",
             "graph_mean_prior.png"),
        ]:
            fig = visualize_graph(PlottingArgs(
                num_nodes=N,
                node_styles=node_styles,
                latent_edge_probs=probs,
                powerline_edge_index=powerline_edge_index,
            ))
            #fig.suptitle(title, fontsize=14)
            fig.savefig(self.outdir / fname, bbox_inches="tight")
            fig.savefig(self.outdir / (Path(fname).stem + ".svg"), bbox_inches="tight")
            plt.show()

    def generate_plots(
        self,
        C_edge: npt.NDArray,      # [E]  – C_{ij}^{risk}
        P_mean: npt.NDArray,      # [E]  – mean posterior
        PR_mean: npt.NDArray,     # [E]  – mean prior  (baseline)
        P_T: npt.NDArray,         # [T, E] – posterior over time
        PR_T: npt.NDArray,        # [T, E] – prior over time  (baseline)
        metrics_post: dict | None = None,
        metrics_prior: dict | None = None,
        metrics_removed: dict | None = None,
        metrics_added: dict | None = None,
    ):
        sns.reset_orig()
        matplotlib.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        })

        cl = self._coupling_label
        cs = self._coupling_short

        # Derived mean distributions
        removed_mean = PR_mean * (1.0 - P_mean)   # p(1-q): [E]
        added_mean   = P_mean  * (1.0 - PR_mean)  # q(1-p): [E]

        # --- KDE: coupling conditioned on high/low posterior (left) and prior (right) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=True)
        for ax, mean_val, suffix in [
            (axes[0], P_mean,  "posterior"),
            (axes[1], PR_mean, "prior"),
        ]:
            valid = np.isfinite(C_edge) & np.isfinite(mean_val)
            C_v, X_v = C_edge[valid], mean_val[valid]
            sns.kdeplot(C_v[X_v > 0.5],  label=f"High {suffix} node pairs", ax=ax)
            sns.kdeplot(C_v[X_v <= 0.5], label=f"Low {suffix} node pairs",  ax=ax)
            ax.set_xlabel(r"$C_{ij}^{\mathrm{risk}}$")
            ax.set_ylabel("Density")
            ax.set_title(f"{cl} vs {suffix}")
            ax.legend()
        #fig.suptitle(r"$C_{ij}^{\mathrm{risk}}$", y=0.88)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_prior.png")
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_prior.svg")
        plt.show()

        # --- KDE: coupling conditioned on high/low for removed and added ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=True)
        for ax, mean_val, suffix in [
            (axes[0], removed_mean, r"removed $p(1-q)$"),
            (axes[1], added_mean,   r"added $q(1-p)$"),
        ]:
            valid = np.isfinite(C_edge) & np.isfinite(mean_val)
            C_v, X_v = C_edge[valid], mean_val[valid]
            med = np.median(X_v) if X_v.size > 0 else 0.5
            sns.kdeplot(C_v[X_v > med],  label=f"High {suffix} node pairs", ax=ax)
            sns.kdeplot(C_v[X_v <= med], label=f"Low {suffix} node pairs",  ax=ax)
            ax.set_xlabel(r"$C_{ij}^{\mathrm{risk}}$")
            ax.set_ylabel("Density")
            ax.set_title(f"{cl} vs {suffix}")
            ax.legend()
        #fig.suptitle(r"$C_{ij}^{\mathrm{risk}}$", y=0.88)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_removed_added.png")
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_removed_added.svg")
        plt.show()

        # --- Scatter: mean edge probability vs coupling (posterior / prior / removed / added) ---
        for mean_val, suffix in [
            (P_mean,       "posterior"),
            (PR_mean,      "prior"),
            (removed_mean, "removed_p1mq"),
            (added_mean,   "added_q1mp"),
        ]:
            valid = np.isfinite(C_edge) & np.isfinite(mean_val)
            C_v, X_v = C_edge[valid], mean_val[valid]
            label_suffix = suffix.replace("_", " ")
            self._save_scatter(
                values=(X_v, C_v),
                title=f"Scatter: {cl} vs {label_suffix}",
                xlabel=f"Mean {label_suffix} existence probability",
                ylabel=r"$C_{ij}^{\mathrm{risk}}$",
                outpath=self.outdir / f"scatter_{suffix}_vs_{cs}.png",
            )

        # --- Graph visualisation ---
        if self._environment is not None and self._powerline_edge_index is not None:
            self._visualize_coupling_graph(C_edge, P_mean, PR_mean)

        print(f"Saved metrics and plots to: {self.outdir.absolute()}")
