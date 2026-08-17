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

from analysis.analyze_latent_graphs.agent_analysis_framework import PosteriorAnalyzer
from analysis.analyze_latent_graphs.build_coupling_matrices import get_PTDF_based_coupling_index

logger = logging.getLogger(__name__)

class Hypothesis1verifier(PosteriorAnalyzer):
    """
    Computes per-timestep alignment metrics between posterior edge existence probabilities
    and PTDF-based coupling (edge-aligned, fully-connected ordering).

    Metrics per timestep:
      - Spearman rho
      - Pearson r
      - Kendall tau
      - Top-k overlap (default: top 10%)
      - ROC-AUC for predicting "strong PTDF coupling" edges (threshold: top 10% of coupling)
      - Average Precision (AP) for same labeling (often more informative for imbalanced labels)
      - Mutual information MI(P -> C)

    Plus edge-wise temporal correlation (Pearson) over time:
      corr_e = corr_t(P_t,e, C_t,e) for each edge e (ignoring NaNs)
    """

    def __init__(
        self,
        out_dir: Path = Path("results/hypothesis_electrical_coupling"),
        topk_frac: float = 0.05,
        strong_label_percentile: float = 90.0,
        min_samples_per_timestep: int = 10,
        min_timesteps_per_edge: int = 5,
        use_abs_coupling_for_metrics: bool = False,
        random_state_mi: int = 0,
        compare_to_mean_coupling: bool = False,
    ):
        self.topk_frac = float(topk_frac)
        self.strong_label_percentile = float(strong_label_percentile)
        self.min_samples_per_timestep = int(min_samples_per_timestep)
        self.min_timesteps_per_edge = int(min_timesteps_per_edge)
        self.use_abs_coupling_for_metrics = bool(use_abs_coupling_for_metrics)
        self.random_state_mi = int(random_state_mi)
        self.compare_to_mean_coupling = bool(compare_to_mean_coupling)
        self.mean_coupling = np.load(out_dir / "ptdf_coupling/coupling_mean.npy") if compare_to_mean_coupling else None
        # Results go into a mode-specific subdirectory so both modes can coexist
        self.outdir = out_dir / ("mean_coupling" if compare_to_mean_coupling else "ptdf_coupling")

        # Per-timestep scalars – posterior vs coupling
        self.spearman_rho = []
        self.pearson_r = []
        self.kendall_tau = []
        self.topk_overlap = []
        self.roc_auc = []
        self.avg_precision = []
        self.mutual_info = []

        # Per-timestep scalars – prior vs coupling  (baseline)
        self.prior_spearman_rho = []
        self.prior_pearson_r = []
        self.prior_kendall_tau = []
        self.prior_topk_overlap = []
        self.prior_roc_auc = []
        self.prior_avg_precision = []
        self.prior_mutual_info = []

        # Per-timestep scalars – edges removed by posterior p(1-q) vs coupling
        self.removed_spearman_rho = []
        self.removed_pearson_r = []
        self.removed_kendall_tau = []
        self.removed_topk_overlap = []
        self.removed_roc_auc = []
        self.removed_avg_precision = []
        self.removed_mutual_info = []

        # Per-timestep scalars – edges added by posterior q(1-p) vs coupling
        self.added_spearman_rho = []
        self.added_pearson_r = []
        self.added_kendall_tau = []
        self.added_topk_overlap = []
        self.added_roc_auc = []
        self.added_avg_precision = []
        self.added_mutual_info = []

        # For edge-wise temporal correlation
        self._P_over_time = []   # list of [E] – posterior
        self._prior_over_time = []  # list of [E] – prior  (baseline)
        self._C_over_time = []   # list of [E] with NaNs for invalid

        # Optional: episode boundaries (if you want later)
        self._episode_ids = []
        self._t_global = 0

    @property
    def _coupling_label(self) -> str:
        """Human-readable label for what coupling is being compared against."""
        return "avg PTDF coupling" if self.compare_to_mean_coupling else "PTDF coupling"

    @property
    def _coupling_short(self) -> str:
        """Short token used in file names."""
        return "mean_coupling" if self.compare_to_mean_coupling else "ptdf_coupling"

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
        # Coupling aligned with fully-connected edge ordering
        if self.compare_to_mean_coupling:
            coupling = self.mean_coupling
        else:
            coupling = get_PTDF_based_coupling_index(environment).astype(np.float64)  # [E]

        posterior_existence = posterior[:, :-1].sum(axis=-1).astype(np.float64)  # [E]
        prior_existence = prior[:, :-1].sum(axis=-1).astype(np.float64)        # [E]
        assert coupling.shape == posterior_existence.shape == prior_existence.shape, f"Shape mismatch: coupling {coupling.shape}, posterior {posterior_existence.shape}, prior {prior_existence.shape}"

        valid = np.isfinite(coupling) & np.isfinite(posterior_existence)

        # Optional: compare against |coupling|
        C = np.abs(coupling) if self.use_abs_coupling_for_metrics else coupling
        P = posterior_existence
        PR = prior_existence

        # Store full vectors (NaN for invalid) for edge-wise temporal correlation
        C_full = np.full_like(C, np.nan, dtype=np.float64)
        P_full = np.full_like(P, np.nan, dtype=np.float64)
        PR_full = np.full_like(P, np.nan, dtype=np.float64)

        C_full[valid] = C[valid]
        P_full[valid] = P[valid]
        PR_full[valid] = PR[valid]

        self._C_over_time.append(C_full)
        self._P_over_time.append(P_full)
        self._prior_over_time.append(PR_full)

        # Per-timestep metrics on valid subset
        C_valid = C[valid]
        P_valid = P[valid]
        PR_valid = PR[valid]

        # Derived distributions: edges removed p(1-q) and added q(1-p)
        removed_valid = PR_valid * (1.0 - P_valid)   # p(1-q)
        added_valid   = P_valid  * (1.0 - PR_valid)  # q(1-p)

        def _run_metrics(c_v, x_v, spearman_list, pearson_list, kendall_list,
                         topk_list, roc_list, ap_list, mi_list):
            """
            Computes all metrics for the given coupling vector c_v and existence probabilities or prior x_v, appends results to the provided lists.
            """
            if c_v.size < self.min_samples_per_timestep:
                spearman_list.append(np.nan); pearson_list.append(np.nan)
                kendall_list.append(np.nan); topk_list.append(np.nan)
                roc_list.append(np.nan); ap_list.append(np.nan); mi_list.append(np.nan)
                return
            rho = spearmanr(c_v, x_v).correlation
            spearman_list.append(float(rho) if rho is not None else np.nan)
            pearson_list.append(self._nan_pearson(c_v, x_v))
            tau = kendalltau(c_v, x_v).correlation
            kendall_list.append(float(tau) if tau is not None else np.nan)
            topk_list.append(self._topk_overlap(c_v, x_v, self.topk_frac))
            auc, ap = self._auc_metrics(c_v, x_v, self.strong_label_percentile)
            roc_list.append(auc); ap_list.append(ap)
            mi_list.append(self._mi(c_v, x_v))

        _run_metrics(C_valid, P_valid,
                     self.spearman_rho, self.pearson_r, self.kendall_tau,
                     self.topk_overlap, self.roc_auc, self.avg_precision, self.mutual_info)
        _run_metrics(C_valid, PR_valid,
                     self.prior_spearman_rho, self.prior_pearson_r, self.prior_kendall_tau,
                     self.prior_topk_overlap, self.prior_roc_auc, self.prior_avg_precision,
                     self.prior_mutual_info)
        _run_metrics(C_valid, removed_valid,
                     self.removed_spearman_rho, self.removed_pearson_r, self.removed_kendall_tau,
                     self.removed_topk_overlap, self.removed_roc_auc, self.removed_avg_precision,
                     self.removed_mutual_info)
        _run_metrics(C_valid, added_valid,
                     self.added_spearman_rho, self.added_pearson_r, self.added_kendall_tau,
                     self.added_topk_overlap, self.added_roc_auc, self.added_avg_precision,
                     self.added_mutual_info)

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
        # Convert lists -> arrays
        spearman_rho = np.array(self.spearman_rho, dtype=np.float64)
        pearson_r = np.array(self.pearson_r, dtype=np.float64)
        kendall_tau_arr = np.array(self.kendall_tau, dtype=np.float64)
        topk_overlap_arr = np.array(self.topk_overlap, dtype=np.float64)
        roc_auc_arr = np.array(self.roc_auc, dtype=np.float64)
        ap_arr = np.array(self.avg_precision, dtype=np.float64)
        mi_arr = np.array(self.mutual_info, dtype=np.float64)

        prior_spearman_rho = np.array(self.prior_spearman_rho, dtype=np.float64)
        prior_pearson_r = np.array(self.prior_pearson_r, dtype=np.float64)
        prior_kendall_tau_arr = np.array(self.prior_kendall_tau, dtype=np.float64)
        prior_topk_overlap_arr = np.array(self.prior_topk_overlap, dtype=np.float64)
        prior_roc_auc_arr = np.array(self.prior_roc_auc, dtype=np.float64)
        prior_ap_arr = np.array(self.prior_avg_precision, dtype=np.float64)
        prior_mi_arr = np.array(self.prior_mutual_info, dtype=np.float64)

        removed_spearman_rho = np.array(self.removed_spearman_rho, dtype=np.float64)
        removed_pearson_r = np.array(self.removed_pearson_r, dtype=np.float64)
        removed_kendall_tau_arr = np.array(self.removed_kendall_tau, dtype=np.float64)
        removed_topk_overlap_arr = np.array(self.removed_topk_overlap, dtype=np.float64)
        removed_roc_auc_arr = np.array(self.removed_roc_auc, dtype=np.float64)
        removed_ap_arr = np.array(self.removed_avg_precision, dtype=np.float64)
        removed_mi_arr = np.array(self.removed_mutual_info, dtype=np.float64)

        added_spearman_rho = np.array(self.added_spearman_rho, dtype=np.float64)
        added_pearson_r = np.array(self.added_pearson_r, dtype=np.float64)
        added_kendall_tau_arr = np.array(self.added_kendall_tau, dtype=np.float64)
        added_topk_overlap_arr = np.array(self.added_topk_overlap, dtype=np.float64)
        added_roc_auc_arr = np.array(self.added_roc_auc, dtype=np.float64)
        added_ap_arr = np.array(self.added_avg_precision, dtype=np.float64)
        added_mi_arr = np.array(self.added_mutual_info, dtype=np.float64)

        num_timesteps = len(spearman_rho)

        # Edge-wise temporal correlation (Pearson + Spearman across time, per edge)
        P_T  = np.stack(self._P_over_time, axis=0)     # [T, E]
        PR_T = np.stack(self._prior_over_time, axis=0) # [T, E]
        C_T  = np.stack(self._C_over_time, axis=0)     # [T, E]
        mean_c = np.nanmean(C_T, axis=0)  # [E]
        T, E = P_T.shape

        # Derived distributions over time
        Removed_T = PR_T * (1.0 - P_T)   # p(1-q): [T, E]
        Added_T   = P_T  * (1.0 - PR_T)  # q(1-p): [T, E]

        def _edgewise_corr(X_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Pearson and Spearman correlation over time for each edge."""
            ec = np.full(E, np.nan, dtype=np.float64)
            es = np.full(E, np.nan, dtype=np.float64)
            for e in range(E):
                x_e = X_T[:, e]
                c_e = C_T[:, e]
                m = np.isfinite(x_e) & np.isfinite(c_e)
                if m.sum() < self.min_timesteps_per_edge:
                    continue
                xe, ce = x_e[m], c_e[m]
                if xe.std() < 1e-12 or ce.std() < 1e-12:
                    continue
                ec[e] = np.corrcoef(xe, ce)[0, 1]
                rho_e = spearmanr(xe, ce).correlation
                es[e] = float(rho_e) if rho_e is not None else np.nan
            return ec, es

        edge_corr,          edge_spearman          = _edgewise_corr(P_T)
        prior_edge_corr,    prior_edge_spearman     = _edgewise_corr(PR_T)
        removed_edge_corr,  removed_edge_spearman   = _edgewise_corr(Removed_T)
        added_edge_corr,    added_edge_spearman     = _edgewise_corr(Added_T)

        self.print_summary(
            spearman_rho, pearson_r, kendall_tau_arr, topk_overlap_arr,
            roc_auc_arr, ap_arr, mi_arr, edge_corr, edge_spearman,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_overlap_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr, prior_edge_corr, prior_edge_spearman,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_overlap_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr, removed_edge_corr, removed_edge_spearman,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_overlap_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr, added_edge_corr, added_edge_spearman,
        )

        # Save arrays
        cs = self._coupling_short
        self._save_array(spearman_rho,       self.outdir / f"spearman_{cs}_posterior.npy")
        self._save_array(pearson_r,           self.outdir / f"pearson_{cs}_posterior.npy")
        self._save_array(kendall_tau_arr,     self.outdir / f"kendall_{cs}_posterior.npy")
        self._save_array(topk_overlap_arr,    self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(roc_auc_arr,         self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        self._save_array(ap_arr,              self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        self._save_array(mi_arr,              self.outdir / f"mutual_info_{cs}_posterior.npy")
        self._save_array(edge_corr,           self.outdir / f"edgewise_temporal_pearson_{cs}_posterior.npy")
        self._save_array(edge_spearman,       self.outdir / f"edgewise_temporal_spearman_{cs}_posterior.npy")
        self._save_array(prior_spearman_rho,    self.outdir / f"spearman_{cs}_prior.npy")
        self._save_array(prior_pearson_r,        self.outdir / f"pearson_{cs}_prior.npy")
        self._save_array(prior_kendall_tau_arr,  self.outdir / f"kendall_{cs}_prior.npy")
        self._save_array(prior_topk_overlap_arr, self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_prior.npy")
        self._save_array(prior_roc_auc_arr,      self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        self._save_array(prior_ap_arr,           self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        self._save_array(prior_mi_arr,           self.outdir / f"mutual_info_{cs}_prior.npy")
        self._save_array(prior_edge_corr,        self.outdir / f"edgewise_temporal_pearson_{cs}_prior.npy")
        self._save_array(prior_edge_spearman,    self.outdir / f"edgewise_temporal_spearman_{cs}_prior.npy")
        self._save_array(removed_spearman_rho,      self.outdir / f"spearman_{cs}_removed.npy")
        self._save_array(removed_pearson_r,          self.outdir / f"pearson_{cs}_removed.npy")
        self._save_array(removed_kendall_tau_arr,    self.outdir / f"kendall_{cs}_removed.npy")
        self._save_array(removed_topk_overlap_arr,   self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_removed.npy")
        self._save_array(removed_roc_auc_arr,        self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        self._save_array(removed_ap_arr,             self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        self._save_array(removed_mi_arr,             self.outdir / f"mutual_info_{cs}_removed.npy")
        self._save_array(removed_edge_corr,          self.outdir / f"edgewise_temporal_pearson_{cs}_removed.npy")
        self._save_array(removed_edge_spearman,      self.outdir / f"edgewise_temporal_spearman_{cs}_removed.npy")
        self._save_array(added_spearman_rho,         self.outdir / f"spearman_{cs}_added.npy")
        self._save_array(added_pearson_r,             self.outdir / f"pearson_{cs}_added.npy")
        self._save_array(added_kendall_tau_arr,       self.outdir / f"kendall_{cs}_added.npy")
        self._save_array(added_topk_overlap_arr,      self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_added.npy")
        self._save_array(added_roc_auc_arr,           self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        self._save_array(added_ap_arr,                self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        self._save_array(added_mi_arr,                self.outdir / f"mutual_info_{cs}_added.npy")
        self._save_array(added_edge_corr,             self.outdir / f"edgewise_temporal_pearson_{cs}_added.npy")
        self._save_array(added_edge_spearman,         self.outdir / f"edgewise_temporal_spearman_{cs}_added.npy")
        # Save a random subsample of flattened (C, P, PR) triples for KDE repaint.
        # The full arrays would be ~2.5 GB each; 50K samples are sufficient for KDE.
        _rng = np.random.default_rng(0)
        _n = min(50_000, P_T.size)
        _idx = _rng.choice(P_T.size, size=_n, replace=False)
        np.savez_compressed(
            self.outdir / "kde_sample.npz",
            P=P_T.ravel()[_idx].astype(np.float32),
            PR=PR_T.ravel()[_idx].astype(np.float32),
            C=C_T.ravel()[_idx].astype(np.float32),
        )
        self._save_array(mean_c, self.outdir / "coupling_mean.npy")

        self.generate_plots(
            C_T, P_T, PR_T,
            ap_arr, edge_corr, edge_spearman, kendall_tau_arr, mi_arr,
            pearson_r, roc_auc_arr, spearman_rho, topk_overlap_arr,
            prior_ap_arr, prior_edge_corr, prior_edge_spearman, prior_kendall_tau_arr,
            prior_mi_arr, prior_pearson_r, prior_roc_auc_arr, prior_spearman_rho,
            prior_topk_overlap_arr,
            removed_ap_arr, removed_edge_corr, removed_edge_spearman, removed_kendall_tau_arr,
            removed_mi_arr, removed_pearson_r, removed_roc_auc_arr, removed_spearman_rho,
            removed_topk_overlap_arr,
            added_ap_arr, added_edge_corr, added_edge_spearman, added_kendall_tau_arr,
            added_mi_arr, added_pearson_r, added_roc_auc_arr, added_spearman_rho,
            added_topk_overlap_arr,
        )

    def print_summary(
        self,
        spearman_rho, pearson_r, kendall_tau_arr, topk_overlap_arr,
        roc_auc_arr, ap_arr, mi_arr, edge_corr, edge_spearman,
        prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_overlap_arr,
        prior_roc_auc_arr, prior_ap_arr, prior_mi_arr, prior_edge_corr, prior_edge_spearman,
        removed_spearman_rho=None, removed_pearson_r=None, removed_kendall_tau_arr=None,
        removed_topk_overlap_arr=None, removed_roc_auc_arr=None, removed_ap_arr=None,
        removed_mi_arr=None, removed_edge_corr=None, removed_edge_spearman=None,
        added_spearman_rho=None, added_pearson_r=None, added_kendall_tau_arr=None,
        added_topk_overlap_arr=None, added_roc_auc_arr=None, added_ap_arr=None,
        added_mi_arr=None, added_edge_corr=None, added_edge_spearman=None,
    ):
        num_timesteps = len(spearman_rho)

        def summarize(name: str, x: np.ndarray):
            xf = x[np.isfinite(x)]
            if xf.size == 0:
                print(f"{name}: all NaN")
                return
            print(f"{name} over t={num_timesteps}: mean={xf.mean():.4f}, std={xf.std():.4f}, "
                  f"median={np.median(xf):.4f}, n_valid={xf.size}")

        print(f"\n=== Hypothesis 1: Electrical Coupling ===")
        print(f"  Correlating C_ij^elec(t) vs edge-existence probability across all E edges, "
              f"computed at each of t={num_timesteps} timesteps (distribution reported).")
        print(f"  C_ij^elec(t) = PTDF-derived electrical coupling at step t  "
              f"(how much power injected at node i flows through edge ij)")
        print(f"  Edge-wise temporal: Spearman_t(P_t(edge_ij), C_ij^elec(t)) per edge "
              f"(one value per edge, mean over edges reported)")
        print(f"\n--- Posterior P(edge_ij | obs_t) vs C_ij^elec(t) ---")
        summarize("Spearman rho", spearman_rho)
        summarize("Pearson r", pearson_r)
        summarize("Kendall tau", kendall_tau_arr)
        summarize(f"Top-{int(self.topk_frac*100)}% overlap", topk_overlap_arr)
        summarize(f"ROC-AUC (labels: coupling >= p{self.strong_label_percentile})", roc_auc_arr)
        summarize(f"Avg Precision (labels: coupling >= p{self.strong_label_percentile})", ap_arr)
        summarize("Mutual information", mi_arr)
        for arr, lbl in [(edge_corr, "Edge-wise temporal Pearson"),
                         (edge_spearman, "Edge-wise temporal Spearman")]:
            f = arr[np.isfinite(arr)]
            if f.size:
                print(f"{lbl}: mean={f.mean():.4f}, std={f.std():.4f}, median={np.median(f):.4f}, n={f.size}")

        print("\n--- Prior P(edge_ij) vs C_ij^elec(t)  [baseline] ---")
        summarize("Spearman rho", prior_spearman_rho)
        summarize("Pearson r", prior_pearson_r)
        summarize("Kendall tau", prior_kendall_tau_arr)
        summarize(f"Top-{int(self.topk_frac*100)}% overlap", prior_topk_overlap_arr)
        summarize(f"ROC-AUC (labels: coupling >= p{self.strong_label_percentile})", prior_roc_auc_arr)
        summarize(f"Avg Precision (labels: coupling >= p{self.strong_label_percentile})", prior_ap_arr)
        summarize("Mutual information", prior_mi_arr)
        for arr, lbl in [(prior_edge_corr, "Edge-wise temporal Pearson (prior)"),
                         (prior_edge_spearman, "Edge-wise temporal Spearman (prior)")]:
            f = arr[np.isfinite(arr)]
            if f.size:
                print(f"{lbl}: mean={f.mean():.4f}, std={f.std():.4f}, median={np.median(f):.4f}, n={f.size}")

        if removed_spearman_rho is not None:
            print("\n--- Edges removed p(1-q) = p_prior*(1-p_post) vs C_ij^elec(t) ---")
            summarize("Spearman rho", removed_spearman_rho)
            summarize("Pearson r", removed_pearson_r)
            summarize("Kendall tau", removed_kendall_tau_arr)
            summarize(f"Top-{int(self.topk_frac*100)}% overlap", removed_topk_overlap_arr)
            summarize(f"ROC-AUC (labels: coupling >= p{self.strong_label_percentile})", removed_roc_auc_arr)
            summarize(f"Avg Precision (labels: coupling >= p{self.strong_label_percentile})", removed_ap_arr)
            summarize("Mutual information", removed_mi_arr)
            for arr, lbl in [(removed_edge_corr, "Edge-wise temporal Pearson (removed)"),
                             (removed_edge_spearman, "Edge-wise temporal Spearman (removed)")]:
                f = arr[np.isfinite(arr)]
                if f.size:
                    print(f"{lbl}: mean={f.mean():.4f}, std={f.std():.4f}, median={np.median(f):.4f}, n={f.size}")

        if added_spearman_rho is not None:
            print("\n--- Edges added q(1-p) = p_post*(1-p_prior) vs C_ij^elec(t) ---")
            summarize("Spearman rho", added_spearman_rho)
            summarize("Pearson r", added_pearson_r)
            summarize("Kendall tau", added_kendall_tau_arr)
            summarize(f"Top-{int(self.topk_frac*100)}% overlap", added_topk_overlap_arr)
            summarize(f"ROC-AUC (labels: coupling >= p{self.strong_label_percentile})", added_roc_auc_arr)
            summarize(f"Avg Precision (labels: coupling >= p{self.strong_label_percentile})", added_ap_arr)
            summarize("Mutual information", added_mi_arr)
            for arr, lbl in [(added_edge_corr, "Edge-wise temporal Pearson (added)"),
                             (added_edge_spearman, "Edge-wise temporal Spearman (added)")]:
                f = arr[np.isfinite(arr)]
                if f.size:
                    print(f"{lbl}: mean={f.mean():.4f}, std={f.std():.4f}, median={np.median(f):.4f}, n={f.size}")


    def print_summary_from_saved(self):
        """Loads saved metric arrays and prints the summary table without regenerating plots."""
        cs = self._coupling_short
        spearman_rho      = np.load(self.outdir / f"spearman_{cs}_posterior.npy")
        pearson_r         = np.load(self.outdir / f"pearson_{cs}_posterior.npy")
        kendall_tau_arr   = np.load(self.outdir / f"kendall_{cs}_posterior.npy")
        topk_overlap_arr  = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        roc_auc_arr       = np.load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        ap_arr            = np.load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        mi_arr            = np.load(self.outdir / f"mutual_info_{cs}_posterior.npy")
        edge_corr         = np.load(self.outdir / f"edgewise_temporal_pearson_{cs}_posterior.npy")
        edge_spearman     = np.load(self.outdir / f"edgewise_temporal_spearman_{cs}_posterior.npy")
        prior_spearman_rho     = np.load(self.outdir / f"spearman_{cs}_prior.npy")
        prior_pearson_r        = np.load(self.outdir / f"pearson_{cs}_prior.npy")
        prior_kendall_tau_arr  = np.load(self.outdir / f"kendall_{cs}_prior.npy")
        prior_topk_overlap_arr = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_prior.npy")
        prior_roc_auc_arr      = np.load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        prior_ap_arr           = np.load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        prior_mi_arr           = np.load(self.outdir / f"mutual_info_{cs}_prior.npy")
        prior_edge_corr        = np.load(self.outdir / f"edgewise_temporal_pearson_{cs}_prior.npy")
        prior_edge_spearman    = np.load(self.outdir / f"edgewise_temporal_spearman_{cs}_prior.npy")

        def _try_load(path):
            return np.load(path) if path.exists() else None

        removed_spearman_rho     = _try_load(self.outdir / f"spearman_{cs}_removed.npy")
        removed_pearson_r        = _try_load(self.outdir / f"pearson_{cs}_removed.npy")
        removed_kendall_tau_arr  = _try_load(self.outdir / f"kendall_{cs}_removed.npy")
        removed_topk_overlap_arr = _try_load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_removed.npy")
        removed_roc_auc_arr      = _try_load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        removed_ap_arr           = _try_load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        removed_mi_arr           = _try_load(self.outdir / f"mutual_info_{cs}_removed.npy")
        removed_edge_corr        = _try_load(self.outdir / f"edgewise_temporal_pearson_{cs}_removed.npy")
        removed_edge_spearman    = _try_load(self.outdir / f"edgewise_temporal_spearman_{cs}_removed.npy")
        added_spearman_rho       = _try_load(self.outdir / f"spearman_{cs}_added.npy")
        added_pearson_r          = _try_load(self.outdir / f"pearson_{cs}_added.npy")
        added_kendall_tau_arr    = _try_load(self.outdir / f"kendall_{cs}_added.npy")
        added_topk_overlap_arr   = _try_load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_added.npy")
        added_roc_auc_arr        = _try_load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        added_ap_arr             = _try_load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        added_mi_arr             = _try_load(self.outdir / f"mutual_info_{cs}_added.npy")
        added_edge_corr          = _try_load(self.outdir / f"edgewise_temporal_pearson_{cs}_added.npy")
        added_edge_spearman      = _try_load(self.outdir / f"edgewise_temporal_spearman_{cs}_added.npy")

        self.print_summary(
            spearman_rho, pearson_r, kendall_tau_arr, topk_overlap_arr,
            roc_auc_arr, ap_arr, mi_arr, edge_corr, edge_spearman,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_overlap_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr, prior_edge_corr, prior_edge_spearman,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_overlap_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr, removed_edge_corr, removed_edge_spearman,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_overlap_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr, added_edge_corr, added_edge_spearman,
        )

    def repaint(self):
        """
        Reads all previously saved arrays from outdir and regenerates all plots
        by calling generate_plots with the loaded data.
        """
        cs = self._coupling_short
        spearman_rho      = np.load(self.outdir / f"spearman_{cs}_posterior.npy")
        pearson_r         = np.load(self.outdir / f"pearson_{cs}_posterior.npy")
        kendall_tau_arr   = np.load(self.outdir / f"kendall_{cs}_posterior.npy")
        topk_overlap_arr  = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        roc_auc_arr       = np.load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        ap_arr            = np.load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        mi_arr            = np.load(self.outdir / f"mutual_info_{cs}_posterior.npy")
        edge_corr         = np.load(self.outdir / f"edgewise_temporal_pearson_{cs}_posterior.npy")
        edge_spearman     = np.load(self.outdir / f"edgewise_temporal_spearman_{cs}_posterior.npy")

        prior_spearman_rho     = np.load(self.outdir / f"spearman_{cs}_prior.npy")
        prior_pearson_r        = np.load(self.outdir / f"pearson_{cs}_prior.npy")
        prior_kendall_tau_arr  = np.load(self.outdir / f"kendall_{cs}_prior.npy")
        prior_topk_overlap_arr = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_prior.npy")
        prior_roc_auc_arr      = np.load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        prior_ap_arr           = np.load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_prior.npy")
        prior_mi_arr           = np.load(self.outdir / f"mutual_info_{cs}_prior.npy")
        prior_edge_corr        = np.load(self.outdir / f"edgewise_temporal_pearson_{cs}_prior.npy")
        prior_edge_spearman    = np.load(self.outdir / f"edgewise_temporal_spearman_{cs}_prior.npy")

        def _try_load(path):
            return np.load(path) if path.exists() else None

        removed_spearman_rho     = _try_load(self.outdir / f"spearman_{cs}_removed.npy")
        removed_pearson_r        = _try_load(self.outdir / f"pearson_{cs}_removed.npy")
        removed_kendall_tau_arr  = _try_load(self.outdir / f"kendall_{cs}_removed.npy")
        removed_topk_overlap_arr = _try_load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_removed.npy")
        removed_roc_auc_arr      = _try_load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        removed_ap_arr           = _try_load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_removed.npy")
        removed_mi_arr           = _try_load(self.outdir / f"mutual_info_{cs}_removed.npy")
        removed_edge_corr        = _try_load(self.outdir / f"edgewise_temporal_pearson_{cs}_removed.npy")
        removed_edge_spearman    = _try_load(self.outdir / f"edgewise_temporal_spearman_{cs}_removed.npy")
        added_spearman_rho       = _try_load(self.outdir / f"spearman_{cs}_added.npy")
        added_pearson_r          = _try_load(self.outdir / f"pearson_{cs}_added.npy")
        added_kendall_tau_arr    = _try_load(self.outdir / f"kendall_{cs}_added.npy")
        added_topk_overlap_arr   = _try_load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct_added.npy")
        added_roc_auc_arr        = _try_load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        added_ap_arr             = _try_load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}_added.npy")
        added_mi_arr             = _try_load(self.outdir / f"mutual_info_{cs}_added.npy")
        added_edge_corr          = _try_load(self.outdir / f"edgewise_temporal_pearson_{cs}_added.npy")
        added_edge_spearman      = _try_load(self.outdir / f"edgewise_temporal_spearman_{cs}_added.npy")

        _kde = np.load(self.outdir / "kde_sample.npz")
        P_T  = _kde["P"].reshape(1, -1).astype(np.float64)
        PR_T = _kde["PR"].reshape(1, -1).astype(np.float64)
        C_T  = _kde["C"].reshape(1, -1).astype(np.float64)

        self.generate_plots(
            C_T, P_T, PR_T,
            ap_arr, edge_corr, edge_spearman, kendall_tau_arr, mi_arr,
            pearson_r, roc_auc_arr, spearman_rho, topk_overlap_arr,
            prior_ap_arr, prior_edge_corr, prior_edge_spearman, prior_kendall_tau_arr,
            prior_mi_arr, prior_pearson_r, prior_roc_auc_arr, prior_spearman_rho,
            prior_topk_overlap_arr,
            removed_ap_arr, removed_edge_corr, removed_edge_spearman, removed_kendall_tau_arr,
            removed_mi_arr, removed_pearson_r, removed_roc_auc_arr, removed_spearman_rho,
            removed_topk_overlap_arr,
            added_ap_arr, added_edge_corr, added_edge_spearman, added_kendall_tau_arr,
            added_mi_arr, added_pearson_r, added_roc_auc_arr, added_spearman_rho,
            added_topk_overlap_arr,
        )

    def generate_plots(
        self,
        C_T: npt.NDArray,           # [T, E]
        P_T: npt.NDArray,           # [T, E] – posterior
        PR_T: npt.NDArray,          # [T, E] – prior  (baseline)
        # posterior metrics
        ap_arr: npt.NDArray,
        edge_corr: npt.NDArray,
        edge_spearman: npt.NDArray,
        kendall_tau_arr: npt.NDArray,
        mi_arr: npt.NDArray,
        pearson_r: npt.NDArray,
        roc_auc_arr: npt.NDArray,
        spearman_rho: npt.NDArray,
        topk_overlap_arr: npt.NDArray,
        # prior metrics (baseline)
        prior_ap_arr: npt.NDArray,
        prior_edge_corr: npt.NDArray,
        prior_edge_spearman: npt.NDArray,
        prior_kendall_tau_arr: npt.NDArray,
        prior_mi_arr: npt.NDArray,
        prior_pearson_r: npt.NDArray,
        prior_roc_auc_arr: npt.NDArray,
        prior_spearman_rho: npt.NDArray,
        prior_topk_overlap_arr: npt.NDArray,
        # edges removed p(1-q) metrics
        removed_ap_arr: npt.NDArray | None = None,
        removed_edge_corr: npt.NDArray | None = None,
        removed_edge_spearman: npt.NDArray | None = None,
        removed_kendall_tau_arr: npt.NDArray | None = None,
        removed_mi_arr: npt.NDArray | None = None,
        removed_pearson_r: npt.NDArray | None = None,
        removed_roc_auc_arr: npt.NDArray | None = None,
        removed_spearman_rho: npt.NDArray | None = None,
        removed_topk_overlap_arr: npt.NDArray | None = None,
        # edges added q(1-p) metrics
        added_ap_arr: npt.NDArray | None = None,
        added_edge_corr: npt.NDArray | None = None,
        added_edge_spearman: npt.NDArray | None = None,
        added_kendall_tau_arr: npt.NDArray | None = None,
        added_mi_arr: npt.NDArray | None = None,
        added_pearson_r: npt.NDArray | None = None,
        added_roc_auc_arr: npt.NDArray | None = None,
        added_spearman_rho: npt.NDArray | None = None,
        added_topk_overlap_arr: npt.NDArray | None = None,
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

        # ---- KDE: coupling conditioned on high/low posterior (left) and prior (right) ----
        C_all = C_T.flatten()
        P_all = P_T.flatten()
        PR_all = PR_T.flatten()

        c_xlabel = (f"{cl} " + r"$\bar{C}_{ij}^{PTDF}$") if self.compare_to_mean_coupling \
                   else (f"{cl} " + r"$C_{ij}^{PTDF}$")
        c_sym = r"$\bar{C}_{ij}^{PTDF}$" if self.compare_to_mean_coupling else r"$C_{ij}^{PTDF}$"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        for ax, X_all, suffix in [
            (axes[0], P_all,  "posterior"),
            (axes[1], PR_all, "prior"),
        ]:
            valid = np.isfinite(C_all) & np.isfinite(X_all)
            C_v, X_v = C_all[valid], X_all[valid]
            sns.kdeplot(C_v[X_v > 0.5],  label=f"High {suffix} node pairs", ax=ax)
            sns.kdeplot(C_v[X_v <= 0.5], label=f"Low {suffix} node pairs",  ax=ax)
            ax.set_xlabel(c_xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"{cl} vs {suffix}")
            ax.legend()
        #fig.suptitle(c_sym, y=0.98)
        plt.tight_layout()
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_prior.png")
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_prior.svg")
        plt.show()

        # ---- KDE: coupling conditioned on high/low for removed p(1-q) and added q(1-p) ----
        Removed_all = PR_all * (1.0 - P_all)  # p(1-q)
        Added_all   = P_all  * (1.0 - PR_all)  # q(1-p)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        for ax, X_all, suffix in [
            (axes[0], Removed_all, r"removed $p(1-q)$"),
            (axes[1], Added_all,   r"added $q(1-p)$"),
        ]:
            valid = np.isfinite(C_all) & np.isfinite(X_all)
            C_v, X_v = C_all[valid], X_all[valid]
            med = np.median(X_v)
            sns.kdeplot(C_v[X_v > med],  label=f"High {suffix} node pairs", ax=ax)
            sns.kdeplot(C_v[X_v <= med], label=f"Low {suffix} node pairs",  ax=ax)
            ax.set_xlabel(c_xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"{cl} vs {suffix}")
            ax.legend()
        #fig.suptitle(c_sym, y=0.98)
        plt.tight_layout()
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_removed_added.png")
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_removed_added.svg")
        plt.show()

        # ---- Side-by-side histograms helper ----
        def _save_hist_overlaid(
            post_vals: np.ndarray,
            prior_vals: np.ndarray,
            metric: str,
            xlabel: str,
            outpath: Path,
            vline: float | None = None,
            vline_label: str | None = None,
        ):
            outpath.parent.mkdir(parents=True, exist_ok=True)
            # Shared bins computed from both panels combined so bar widths are equal
            combined = np.concatenate([post_vals[np.isfinite(post_vals)],
                                       prior_vals[np.isfinite(prior_vals)]])
            if combined.size > 1:
                bins = np.linspace(combined.min(), combined.max(), 31)
            else:
                bins = 30
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
            for ax, vals, vs_label in [
                (axes[0], post_vals, "posterior"),
                (axes[1], prior_vals, "prior"),
            ]:
                finite = vals[np.isfinite(vals)]
                sns.histplot(finite, kde=False, bins=bins, ax=ax)
                if finite.size > 0:
                    m = float(np.mean(finite))
                    ax.axvline(m, color="blue", linestyle="-.", label=f"Mean = {m:.3f}")
                if vline is not None:
                    ax.axvline(vline, color="red", linestyle="--",
                               label=vline_label if vline_label else f"x = {vline:.3f}")
                ax.legend(fontsize=12)
                ax.set_title(f"{cl} vs {vs_label}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
            #fig.suptitle(metric, y=0.88)
            plt.tight_layout()
            plt.savefig(outpath.parent / (outpath.stem + ".png"))
            plt.savefig(outpath.parent / (outpath.stem + ".svg"))
            plt.show()

        # ---- 4-panel histograms: posterior | prior | removed p(1-q) | added q(1-p) ----
        def _save_hist_4way(
            post_vals: np.ndarray,
            prior_vals: np.ndarray,
            removed_vals: np.ndarray | None,
            added_vals: np.ndarray | None,
            metric: str,
            xlabel: str,
            outpath: Path,
            vline: float | None = None,
            vline_label: str | None = None,
        ):
            if removed_vals is None or added_vals is None:
                return
            outpath.parent.mkdir(parents=True, exist_ok=True)
            all_vals = [post_vals, prior_vals, removed_vals, added_vals]
            labels = ["posterior", "prior (baseline)", r"removed $p(1-q)$", r"added $q(1-p)$"]
            combined = np.concatenate([v[np.isfinite(v)] for v in all_vals])
            bins = np.linspace(combined.min(), combined.max(), 31) if combined.size > 1 else 30
            fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)
            for ax, vals, vs_label in zip(axes, all_vals, labels):
                finite = vals[np.isfinite(vals)]
                sns.histplot(finite, kde=False, bins=bins, ax=ax)
                if finite.size > 0:
                    m = float(np.mean(finite))
                    ax.axvline(m, color="blue", linestyle="-.", label=f"Mean = {m:.3f}")
                if vline is not None:
                    ax.axvline(vline, color="red", linestyle="--",
                               label=vline_label if vline_label else f"x = {vline:.3f}")
                ax.legend(fontsize=10)
                ax.set_title(vs_label)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
            #fig.suptitle(metric, y=0.88)
            plt.tight_layout()
            stem = outpath.stem + "_4way"
            plt.savefig(outpath.parent / (stem + ".png"))
            plt.savefig(outpath.parent / (stem + ".svg"))
            plt.show()

        _save_hist_overlaid(
            spearman_rho, prior_spearman_rho,
            metric="Spearman",
            xlabel=r"Spearman $\rho$",
            outpath=self.outdir / "hist_spearman_rho.png",
        )
        _save_hist_4way(
            spearman_rho, prior_spearman_rho, removed_spearman_rho, added_spearman_rho,
            metric="Spearman", xlabel=r"Spearman $\rho$",
            outpath=self.outdir / "hist_spearman_rho.png",
        )
        _save_hist_overlaid(
            topk_overlap_arr, prior_topk_overlap_arr,
            metric=f"Top-{int(self.topk_frac * 100)}% overlap",
            xlabel=f"Top-{int(self.topk_frac * 100)}% overlap fraction",
            outpath=self.outdir / f"hist_topk_overlap_{int(self.topk_frac * 100)}pct.png",
            vline=self.topk_frac,
            vline_label=f"Random baseline ({self.topk_frac:.0%})",
        )
        _save_hist_4way(
            topk_overlap_arr, prior_topk_overlap_arr, removed_topk_overlap_arr, added_topk_overlap_arr,
            metric=f"Top-{int(self.topk_frac * 100)}% overlap",
            xlabel=f"Top-{int(self.topk_frac * 100)}% overlap fraction",
            outpath=self.outdir / f"hist_topk_overlap_{int(self.topk_frac * 100)}pct.png",
            vline=self.topk_frac,
            vline_label=f"Random baseline ({self.topk_frac:.0%})",
        )
        _save_hist_overlaid(
            edge_spearman, prior_edge_spearman,
            metric="Edge-wise temporal Spearman",
            xlabel=r"Temporal Spearman $\rho$",
            outpath=self.outdir / "hist_edgewise_temporal_spearman.png",
        )
        _save_hist_4way(
            edge_spearman, prior_edge_spearman, removed_edge_spearman, added_edge_spearman,
            metric="Edge-wise temporal Spearman",
            xlabel=r"Temporal Spearman $\rho$",
            outpath=self.outdir / "hist_edgewise_temporal_spearman.png",
        )
        _save_hist_overlaid(
            roc_auc_arr, prior_roc_auc_arr,
            metric=f"ROC-AUC (p{int(self.strong_label_percentile)})",
            xlabel="ROC-AUC",
            outpath=self.outdir / f"hist_roc_auc_p{int(self.strong_label_percentile)}.png",
        )
        _save_hist_4way(
            roc_auc_arr, prior_roc_auc_arr, removed_roc_auc_arr, added_roc_auc_arr,
            metric=f"ROC-AUC (p{int(self.strong_label_percentile)})",
            xlabel="ROC-AUC",
            outpath=self.outdir / f"hist_roc_auc_p{int(self.strong_label_percentile)}.png",
        )
        _save_hist_overlaid(
            ap_arr, prior_ap_arr,
            metric=f"Avg Precision (p{int(self.strong_label_percentile)})",
            xlabel="Average Precision",
            outpath=self.outdir / f"hist_avg_precision_p{int(self.strong_label_percentile)}.png",
        )
        _save_hist_4way(
            ap_arr, prior_ap_arr, removed_ap_arr, added_ap_arr,
            metric=f"Avg Precision (p{int(self.strong_label_percentile)})",
            xlabel="Average Precision",
            outpath=self.outdir / f"hist_avg_precision_p{int(self.strong_label_percentile)}.png",
        )
        _save_hist_overlaid(
            mi_arr, prior_mi_arr,
            metric="Mutual information",
            xlabel="MI",
            outpath=self.outdir / "hist_mutual_info.png",
        )
        _save_hist_4way(
            mi_arr, prior_mi_arr, removed_mi_arr, added_mi_arr,
            metric="Mutual information",
            xlabel="MI",
            outpath=self.outdir / "hist_mutual_info.png",
        )

        print(f"Saved metrics and plots to: {self.outdir.absolute()}")
