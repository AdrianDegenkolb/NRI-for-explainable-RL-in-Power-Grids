import logging
from pathlib import Path

import matplotlib as mpl
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
from tabulate import tabulate

from analysis.analyze_latent_graphs.agent_analysis_framework import PosteriorAnalyzer
from analysis.analyze_latent_graphs.build_coupling_matrices import get_ptdf_from_env, compute_node_risk_vector
from grid2op_env.observation_converter import GraphObservationConverter, EDGE_INDEX
from rarl import fully_connected_edge_index
from visualization import visualize_graph, PlottingArgs, get_node_styles

logger = logging.getLogger(__name__)


def get_reconfigured_nodes(action: BaseAction, obs: BaseObservation) -> set[int]:
    """
    Returns the set of node indices V(a_t) that are directly reconfigured by the action.

    Node ordering matches build_coupling_matrices.py:
      [line_or(0..n_line-1), line_ex(0..n_line-1), gen(0..n_gen-1), load(0..n_load-1)]

    We consider a node reconfigured if the action changes the bus assignment of the
    corresponding element (set_bus or change_bus) or changes the line status (reconnect /
    disconnect) for a line endpoint node.

    :param action: the grid2op action taken at timestep t
    :param obs: the observation at timestep t (s_t), used for topology context
    :return: set of node indices in the fully-connected node ordering
    """
    n_line = obs.n_line
    n_gen = obs.n_gen

    reconfigured: set[int] = set()

    # --- bus assignment changes (set_bus / change_bus) ---
    # action._set_topo_vect[k] != 0  →  element at position k has its bus set
    # action._change_bus_vect[k]      →  element at position k has its bus toggled
    set_topo = action._set_topo_vect      # (dim_topo,), 0 = no-op, 1/2 = target bus
    change_topo = action._change_bus_vect  # (dim_topo,), bool

    def _topo_pos_touched(pos: int) -> bool:
        return int(set_topo[pos]) != 0 or bool(change_topo[pos])

    # Line origins
    for l in range(n_line):
        if _topo_pos_touched(obs.line_or_pos_topo_vect[l]):
            reconfigured.add(l)

    # Line extremities
    for l in range(n_line):
        if _topo_pos_touched(obs.line_ex_pos_topo_vect[l]):
            reconfigured.add(n_line + l)

    # Generators
    for g in range(n_gen):
        if _topo_pos_touched(obs.gen_pos_topo_vect[g]):
            reconfigured.add(2 * n_line + g)

    # Loads
    for d in range(obs.n_load):
        if _topo_pos_touched(obs.load_pos_topo_vect[d]):
            reconfigured.add(2 * n_line + n_gen + d)

    # --- line status changes (reconnect / disconnect) ---
    # A status change touches both endpoints of the line.
    set_status = action._set_line_status    # (n_line,), -1 / 0 / +1
    change_status = action._switch_line_status  # (n_line,), bool

    for l in range(n_line):
        if int(set_status[l]) != 0 or bool(change_status[l]):
            reconfigured.add(l)          # line origin
            reconfigured.add(n_line + l)  # line extremity

    return reconfigured


class Hypothesis3verifier(PosteriorAnalyzer):
    """
    Tests whether latent edges capture action-effect coupling.

    At each timestep t the agent takes action a_t in state s_t, transitioning to s_{t+1}.
    We compute:

        V(a_t)        – set of node indices directly reconfigured by a_t
        Δr_j(t)       = |r_j(s_{t+1}) - r_j(s_t)|   for all j
        C_{ij}^{effect}(t) = Δr_j(t)  if i ∈ V(a_t), else 0

    This gives a [E]-shaped coupling vector (fully-connected edge ordering) for each
    timestep where an action was taken.  We then compare this against the posterior
    edge-existence probability q_φ(z_{ij} | s_t) via:

    Per-timestep metrics (over the E edge pairs at each step with a non-trivial action):
      - Spearman rho
      - Pearson r
      - Kendall tau
      - Top-k overlap
      - ROC-AUC / Average Precision (strong coupling label: top percentile of C^{effect})
      - Mutual information

    And the per-edge mean coupling vs mean posterior for a global summary.
    """

    def __init__(
        self,
        out_dir: Path = Path("results/hypothesis_action_effect"),
        topk_frac: float = 0.05,
        strong_label_percentile: float = 90.0,
        min_samples_per_timestep: int = 10,
        min_action_steps: int = 5,
        random_state_mi: int = 0,
    ):
        self.topk_frac = float(topk_frac)
        self.strong_label_percentile = float(strong_label_percentile)
        self.min_samples_per_timestep = int(min_samples_per_timestep)
        self.min_action_steps = int(min_action_steps)
        self.random_state_mi = int(random_state_mi)
        self.outdir = out_dir / "action_effect"

        # Per-timestep (only timesteps where V(a_t) is non-empty and s_{t+1} is observed)
        self.spearman_rho: list[float] = []
        self.pearson_r: list[float] = []
        self.kendall_tau: list[float] = []
        self.topk_overlap: list[float] = []
        self.roc_auc: list[float] = []
        self.avg_precision: list[float] = []
        self.mutual_info: list[float] = []

        # For edge-mean summary: accumulate C^{effect} and posterior/prior over action-steps [E] each
        self._C_over_time: list[npt.NDArray] = []   # [E] per action-step
        self._P_over_time: list[npt.NDArray] = []   # [E] per action-step – posterior
        self._PR_over_time: list[npt.NDArray] = []  # [E] per action-step – prior (baseline)

        # Per-timestep scalars – prior vs coupling  (baseline)
        self.prior_spearman_rho: list[float] = []
        self.prior_pearson_r: list[float] = []
        self.prior_kendall_tau: list[float] = []
        self.prior_topk_overlap: list[float] = []
        self.prior_roc_auc: list[float] = []
        self.prior_avg_precision: list[float] = []
        self.prior_mutual_info: list[float] = []

        # Per-timestep scalars – edges removed p(1-q) vs coupling
        self.removed_spearman_rho: list[float] = []
        self.removed_pearson_r: list[float] = []
        self.removed_kendall_tau: list[float] = []
        self.removed_topk_overlap: list[float] = []
        self.removed_roc_auc: list[float] = []
        self.removed_avg_precision: list[float] = []
        self.removed_mutual_info: list[float] = []

        # Per-timestep scalars – edges added q(1-p) vs coupling
        self.added_spearman_rho: list[float] = []
        self.added_pearson_r: list[float] = []
        self.added_kendall_tau: list[float] = []
        self.added_topk_overlap: list[float] = []
        self.added_roc_auc: list[float] = []
        self.added_avg_precision: list[float] = []
        self.added_mutual_info: list[float] = []

        # State carried between consecutive steps:
        self._prev_risk: npt.NDArray | None = None
        self._prev_posterior: npt.NDArray | None = None
        self._prev_prior: npt.NDArray | None = None   # prior at s_t  [E]
        self._prev_V: set[int] | None = None

        # Captured on first step for graph visualisation
        self._environment: Environment | None = None
        self._powerline_edge_index: npt.NDArray | None = None

        # Per-node count of how many times each node was directly reconfigured by an action.
        # Initialised lazily to shape [N] once N is known.
        self._node_action_counts: npt.NDArray | None = None  # [N]

        self._episode_ids: list[str] = []
        self._t_global: int = 0

    # ------------------------------------------------------------------
    # Static metric helpers (identical to hypo1 / hypo2)
    # ------------------------------------------------------------------

    @staticmethod
    def _nan_pearson(x: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        x0, y0 = x[m], y[m]
        if x0.std() < 1e-12 or y0.std() < 1e-12:
            return np.nan
        return float(np.corrcoef(x0, y0)[0, 1])

    @staticmethod
    def _topk_overlap(c: np.ndarray, p: np.ndarray, frac: float) -> float:
        m = np.isfinite(c) & np.isfinite(p)
        c, p = c[m], p[m]
        n = c.shape[0]
        if n < 2:
            return np.nan
        k = max(1, int(frac * n))
        top_c = np.argpartition(c, -k)[-k:]
        top_p = np.argpartition(p, -k)[-k:]
        return float(np.intersect1d(top_c, top_p).size / k)

    @staticmethod
    def _auc_metrics(c: np.ndarray, p: np.ndarray, percentile: float):
        m = np.isfinite(c) & np.isfinite(p)
        c, p = c[m], p[m]
        if c.size < 10:
            return np.nan, np.nan
        thr = np.percentile(c, percentile)
        y = (c >= thr).astype(np.int32)
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
        m = np.isfinite(c) & np.isfinite(p)
        c, p = c[m], p[m]
        if c.size < 10:
            return np.nan
        try:
            mi = mutual_info_regression(p.reshape(-1, 1), c, random_state=self.random_state_mi)
            return float(mi[0])
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # PosteriorAnalyzer interface
    # ------------------------------------------------------------------

    def on_rl_step(
        self,
        posterior: npt.NDArray,       # [E, K]
        prior: npt.NDArray,           # [E, K]
        powergrid_graph: npt.NDArray,
        observation: BaseObservation,  # s_t  (before action a_t is applied)
        environment: Environment,
        action: BaseAction,
    ):
        # Capture env / powerline edge index once
        if self._environment is None:
            self._environment = environment
            obs_space = GraphObservationConverter(
                g2op_obs_space=environment.observation_space
            )
            self._powerline_edge_index = obs_space.to_gym(environment.current_obs)[EDGE_INDEX]

        PTDF = get_ptdf_from_env(environment)
        r_t = compute_node_risk_vector(observation, PTDF).astype(np.float64)  # [N]
        posterior_t = posterior[:, 0].astype(np.float64)                       # [E]
        prior_t     = prior[:, 0].astype(np.float64)                           # [E]

        # --- Use the previous step's state to compute Δr and C^{effect} ---
        if (
            self._prev_risk is not None
            and self._prev_V is not None
            and len(self._prev_V) > 0
        ):
            self._record_action_step(
                r_prev=self._prev_risk,
                r_curr=r_t,
                posterior_prev=self._prev_posterior,
                prior_prev=self._prev_prior,
                V=self._prev_V,
            )

        if action is not None:
            V_t = get_reconfigured_nodes(action, observation)
        else:
            V_t = set()

        self._prev_risk = r_t
        self._prev_posterior = posterior_t
        self._prev_prior = prior_t
        self._prev_V = V_t
        self._t_global += 1

    def _record_action_step(
        self,
        r_prev: npt.NDArray,
        r_curr: npt.NDArray,
        posterior_prev: npt.NDArray,
        prior_prev: npt.NDArray,
        V: set[int],
    ):
        """Compute C^{effect}(t) and run per-timestep metrics for posterior and prior."""
        N = r_prev.shape[0]
        delta_r = np.abs(r_curr - r_prev)

        if self._node_action_counts is None:
            self._node_action_counts = np.zeros(N, dtype=np.int64)
        for node_idx in V:
            if 0 <= node_idx < N:
                self._node_action_counts[node_idx] += 1

        edge_index = fully_connected_edge_index(num_nodes=N)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        E = src.shape[0]

        src_in_V = np.array([s in V for s in src], dtype=bool)
        C_effect = np.where(src_in_V, delta_r[dst], 0.0)

        self._C_over_time.append(C_effect)
        self._P_over_time.append(posterior_prev)
        self._PR_over_time.append(prior_prev)

        valid = np.isfinite(C_effect)
        C_v = C_effect[valid]

        def _run(x_arr, sp_list, pe_list, kt_list, tk_list, rc_list, ap_list, mi_list):
            x_v = x_arr[valid & np.isfinite(x_arr)]
            c_v_ = C_v[np.isfinite(x_arr[valid])]
            if c_v_.size < self.min_samples_per_timestep:
                sp_list.append(np.nan); pe_list.append(np.nan); kt_list.append(np.nan)
                tk_list.append(np.nan); rc_list.append(np.nan); ap_list.append(np.nan)
                mi_list.append(np.nan)
                return
            rho = spearmanr(c_v_, x_v).correlation
            sp_list.append(float(rho) if rho is not None else np.nan)
            pe_list.append(self._nan_pearson(c_v_, x_v))
            tau = kendalltau(c_v_, x_v).correlation
            kt_list.append(float(tau) if tau is not None else np.nan)
            tk_list.append(self._topk_overlap(c_v_, x_v, self.topk_frac))
            auc, ap = self._auc_metrics(c_v_, x_v, self.strong_label_percentile)
            rc_list.append(auc); ap_list.append(ap)
            mi_list.append(self._mi(c_v_, x_v))

        _run(posterior_prev,
             self.spearman_rho, self.pearson_r, self.kendall_tau,
             self.topk_overlap, self.roc_auc, self.avg_precision, self.mutual_info)
        _run(prior_prev,
             self.prior_spearman_rho, self.prior_pearson_r, self.prior_kendall_tau,
             self.prior_topk_overlap, self.prior_roc_auc, self.prior_avg_precision,
             self.prior_mutual_info)

        # Derived distributions: edges removed p(1-q) and added q(1-p)
        removed_prev = prior_prev * (1.0 - posterior_prev)   # p(1-q)
        added_prev   = posterior_prev * (1.0 - prior_prev)   # q(1-p)
        _run(removed_prev,
             self.removed_spearman_rho, self.removed_pearson_r, self.removed_kendall_tau,
             self.removed_topk_overlap, self.removed_roc_auc, self.removed_avg_precision,
             self.removed_mutual_info)
        _run(added_prev,
             self.added_spearman_rho, self.added_pearson_r, self.added_kendall_tau,
             self.added_topk_overlap, self.added_roc_auc, self.added_avg_precision,
             self.added_mutual_info)

    def on_heuristic_step(
        self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment
    ):
        # Reset pending state: we cannot compute Δr across an RL→heuristic boundary
        #self._prev_risk = None
        #self._prev_posterior = None
        #self._prev_prior = None
        #self._prev_V = None
        pass

    def on_new_episode(self, chronic_id: str):
        # Reset carry-over state at episode boundaries
        self._prev_risk = None
        self._prev_posterior = None
        self._prev_prior = None
        self._prev_V = None
        self._episode_ids.append(chronic_id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_array(arr: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    @staticmethod
    def _save_hist(
        values: np.ndarray,
        title: str,
        xlabel: str,
        outpath: Path,
        vline: float | None = None,
        vline_label: str | None = None,
    ):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        finite_values = values[np.isfinite(values)]
        sns.histplot(finite_values, kde=False)
        mean_val = float(np.mean(finite_values)) if finite_values.size > 0 else None
        if mean_val is not None:
            plt.axvline(mean_val, color="blue", linestyle="-.", label=f"Mean = {mean_val:.3f}")
        if vline is not None:
            plt.axvline(
                vline,
                color="red",
                linestyle="--",
                label=vline_label if vline_label else f"x = {vline:.3f}",
            )
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


    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_aggregated_coupling(
        C_T: npt.NDArray,   # [T_action, E]
        num_nodes: int,
    ) -> npt.NDArray:
        """
        Aggregate sparse per-timestep C^{effect} into a single [E] vector.

        Steps
        -----
        1. Sum over time → raw totals per edge (i,j):  how much cumulative Δr_j
           was observed across all timesteps where i ∈ V(a_t).
        2. Scatter into an [N, N] node-pair matrix.
        3. Row-normalise: divide each row i by its row sum so the row sums to 1
           (or stays 0 for nodes that were never reconfigured).  Interpretation:
           "given that node i was reconfigured, what fraction of the total induced
           effect went to node j?"
        4. Flatten back to the fully-connected edge ordering [E].

        :param C_T: [T_action, E] per-timestep C^{effect} values
        :param num_nodes: N
        :return: C_agg [E] – row-normalised aggregate coupling
        """
        edge_index = fully_connected_edge_index(num_nodes=num_nodes)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        E   = src.shape[0]

        # 1. Sum over time
        C_sum_edge = np.nansum(C_T, axis=0)  # [E]

        # 2. Scatter into [N, N]
        C_sum_node = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for e in range(E):
            C_sum_node[src[e], dst[e]] = C_sum_edge[e]

        # 3. Row-normalise
        row_sums = C_sum_node.sum(axis=1, keepdims=True)           # [N, 1]
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)       # avoid /0
        C_norm_node = C_sum_node / row_sums                         # [N, N]

        # 4. Flatten back to [E]
        C_agg = C_norm_node[src, dst]  # [E]
        return C_agg

    def print_summary(
        self,
        spearman_rho, pearson_r, kendall_tau_arr, topk_arr,
        roc_auc_arr, ap_arr, mi_arr,
        prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_arr,
        prior_roc_auc_arr, prior_ap_arr, prior_mi_arr,
        global_post: dict, global_prior: dict,
        agg_post: dict, agg_prior: dict,
        removed_spearman_rho=None, removed_pearson_r=None, removed_kendall_tau_arr=None,
        removed_topk_arr=None, removed_roc_auc_arr=None, removed_ap_arr=None, removed_mi_arr=None,
        added_spearman_rho=None, added_pearson_r=None, added_kendall_tau_arr=None,
        added_topk_arr=None, added_roc_auc_arr=None, added_ap_arr=None, added_mi_arr=None,
        global_removed: dict | None = None, global_added: dict | None = None,
        agg_removed: dict | None = None, agg_added: dict | None = None,
    ):
        all_global = [d for d in [global_post, global_prior, global_removed, global_added] if d]
        metric_names = list(dict.fromkeys(
            list(k for d in all_global for k in d.keys())
        ))

        dist_arrs = {
            "Spearman rho":        (spearman_rho,    prior_spearman_rho,    removed_spearman_rho,    added_spearman_rho),
            "Pearson r":           (pearson_r,        prior_pearson_r,        removed_pearson_r,        added_pearson_r),
            "Kendall tau":         (kendall_tau_arr,  prior_kendall_tau_arr,  removed_kendall_tau_arr,  added_kendall_tau_arr),
            f"Top-{int(self.topk_frac*100)}% overlap": (topk_arr, prior_topk_arr, removed_topk_arr, added_topk_arr),
            f"ROC-AUC (p{int(self.strong_label_percentile)})": (roc_auc_arr, prior_roc_auc_arr, removed_roc_auc_arr, added_roc_auc_arr),
            f"Avg Precision (p{int(self.strong_label_percentile)})": (ap_arr, prior_ap_arr, removed_ap_arr, added_ap_arr),
            "Mutual information":  (mi_arr,           prior_mi_arr,           removed_mi_arr,           added_mi_arr),
        }

        headers = [
            "Metric",
            "Per-step post", "Per-step prior", "Per-step removed", "Per-step added",
            "Global post", "Global prior", "Global removed", "Global added",
            "Agg post", "Agg prior", "Agg removed", "Agg added",
        ]

        def _fv_plain(d, k):
            if d is None:
                return "N/A"
            v = d.get(k, float("nan"))
            return f"{v:.4f}" if np.isfinite(v) else "NaN"

        def _dist_mean_plain(arr):
            if arr is None:
                return "N/A"
            f = arr[np.isfinite(arr)]
            return f"{float(np.nanmean(f)):.4f}" if f.size else "NaN"

        rows = []
        for name in metric_names:
            dp, dpr, drem, dadd = dist_arrs.get(name, (None, None, None, None))
            rows.append([
                name,
                _dist_mean_plain(dp), _dist_mean_plain(dpr),
                _dist_mean_plain(drem), _dist_mean_plain(dadd),
                _fv_plain(global_post, name), _fv_plain(global_prior, name),
                _fv_plain(global_removed, name), _fv_plain(global_added, name),
                _fv_plain(agg_post, name), _fv_plain(agg_prior, name),
                _fv_plain(agg_removed, name), _fv_plain(agg_added, name),
            ])

        print(f"\n=== Hypothesis 3: Action-Effect Coupling ===\n")
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".4f"))

    def on_evaluation_end(self):
        n_action_steps = len(self._C_over_time)
        if n_action_steps == 0:
            print("No action steps recorded – nothing to evaluate.")
            return

        print(
            f"\n=== Hypothesis 3: Action-Effect Coupling vs Posterior "
            f"(T_total={self._t_global}, T_action={n_action_steps}) ==="
        )

        spearman_rho    = np.array(self.spearman_rho, dtype=np.float64)
        pearson_r       = np.array(self.pearson_r, dtype=np.float64)
        kendall_tau_arr = np.array(self.kendall_tau, dtype=np.float64)
        topk_arr        = np.array(self.topk_overlap, dtype=np.float64)
        roc_auc_arr     = np.array(self.roc_auc, dtype=np.float64)
        ap_arr          = np.array(self.avg_precision, dtype=np.float64)
        mi_arr          = np.array(self.mutual_info, dtype=np.float64)

        prior_spearman_rho    = np.array(self.prior_spearman_rho, dtype=np.float64)
        prior_pearson_r       = np.array(self.prior_pearson_r, dtype=np.float64)
        prior_kendall_tau_arr = np.array(self.prior_kendall_tau, dtype=np.float64)
        prior_topk_arr        = np.array(self.prior_topk_overlap, dtype=np.float64)
        prior_roc_auc_arr     = np.array(self.prior_roc_auc, dtype=np.float64)
        prior_ap_arr          = np.array(self.prior_avg_precision, dtype=np.float64)
        prior_mi_arr          = np.array(self.prior_mutual_info, dtype=np.float64)

        removed_spearman_rho    = np.array(self.removed_spearman_rho, dtype=np.float64)
        removed_pearson_r       = np.array(self.removed_pearson_r, dtype=np.float64)
        removed_kendall_tau_arr = np.array(self.removed_kendall_tau, dtype=np.float64)
        removed_topk_arr        = np.array(self.removed_topk_overlap, dtype=np.float64)
        removed_roc_auc_arr     = np.array(self.removed_roc_auc, dtype=np.float64)
        removed_ap_arr          = np.array(self.removed_avg_precision, dtype=np.float64)
        removed_mi_arr          = np.array(self.removed_mutual_info, dtype=np.float64)

        added_spearman_rho    = np.array(self.added_spearman_rho, dtype=np.float64)
        added_pearson_r       = np.array(self.added_pearson_r, dtype=np.float64)
        added_kendall_tau_arr = np.array(self.added_kendall_tau, dtype=np.float64)
        added_topk_arr        = np.array(self.added_topk_overlap, dtype=np.float64)
        added_roc_auc_arr     = np.array(self.added_roc_auc, dtype=np.float64)
        added_ap_arr          = np.array(self.added_avg_precision, dtype=np.float64)
        added_mi_arr          = np.array(self.added_mutual_info, dtype=np.float64)

        # Stack over action-steps: [T_action, E]
        C_T  = np.stack(self._C_over_time,  axis=0)
        P_T  = np.stack(self._P_over_time,  axis=0)
        PR_T = np.stack(self._PR_over_time, axis=0)

        C_mean  = np.nanmean(C_T,  axis=0)  # [E]
        P_mean  = np.nanmean(P_T,  axis=0)  # [E]
        PR_mean = np.nanmean(PR_T, axis=0)  # [E]

        E = C_T.shape[1]
        N_infer = int(round((1 + (1 + 4 * E) ** 0.5) / 2))
        C_agg      = self._build_aggregated_coupling(C_T,  num_nodes=N_infer)
        C_agg_prior = self._build_aggregated_coupling(C_T, num_nodes=N_infer)  # same C_T, same agg

        # Derived mean distributions
        removed_mean = PR_mean * (1.0 - P_mean)   # p(1-q): [E]
        added_mean   = P_mean  * (1.0 - PR_mean)  # q(1-p): [E]

        def summarize(name: str, post: np.ndarray, prior: np.ndarray):
            def _s(x):
                xf = x[np.isfinite(x)]
                return f"mean={xf.mean():.4f}, n={xf.size}" if xf.size else "all NaN"
            print(f"  {name}: posterior [{_s(post)}]  |  prior [{_s(prior)}]")

        summarize("Spearman rho", spearman_rho, prior_spearman_rho)
        summarize("Pearson r",    pearson_r,    prior_pearson_r)
        summarize("Kendall tau",  kendall_tau_arr, prior_kendall_tau_arr)
        summarize(f"Top-{int(self.topk_frac*100)}% overlap", topk_arr, prior_topk_arr)
        summarize(f"ROC-AUC (p{self.strong_label_percentile})", roc_auc_arr, prior_roc_auc_arr)
        summarize(f"Avg Precision (p{self.strong_label_percentile})", ap_arr, prior_ap_arr)
        summarize("Mutual information", mi_arr, prior_mi_arr)

        # Helper: compute all scalar metrics for a (c_v, p_v) pair
        def _all_scalar_metrics(c_v: np.ndarray, p_v: np.ndarray) -> dict[str, float]:
            if c_v.size < self.min_samples_per_timestep:
                return {}
            out: dict[str, float] = {}
            rho = spearmanr(c_v, p_v).correlation
            out["Spearman rho"] = float(rho) if rho is not None else np.nan
            out["Pearson r"] = self._nan_pearson(c_v, p_v)
            tau = kendalltau(c_v, p_v).correlation
            out["Kendall tau"] = float(tau) if tau is not None else np.nan
            out[f"Top-{int(self.topk_frac*100)}% overlap"] = self._topk_overlap(c_v, p_v, self.topk_frac)
            auc, ap = self._auc_metrics(c_v, p_v, self.strong_label_percentile)
            out[f"ROC-AUC (p{int(self.strong_label_percentile)})"] = auc
            out[f"Avg Precision (p{int(self.strong_label_percentile)})"] = ap
            out["Mutual information"] = self._mi(c_v, p_v)
            return out

        # Global (mean C vs mean P/PR/removed/added)
        valid_post = np.isfinite(C_mean) & np.isfinite(P_mean)
        global_post  = _all_scalar_metrics(C_mean[valid_post], P_mean[valid_post])
        valid_pr     = np.isfinite(C_mean) & np.isfinite(PR_mean)
        global_prior = _all_scalar_metrics(C_mean[valid_pr], PR_mean[valid_pr])
        valid_rem    = np.isfinite(C_mean) & np.isfinite(removed_mean)
        global_removed = _all_scalar_metrics(C_mean[valid_rem], removed_mean[valid_rem])
        valid_add    = np.isfinite(C_mean) & np.isfinite(added_mean)
        global_added   = _all_scalar_metrics(C_mean[valid_add], added_mean[valid_add])

        # Aggregated (row-norm C vs mean P/PR/removed/added)
        valid_agg_post  = np.isfinite(C_agg) & np.isfinite(P_mean)
        agg_post  = _all_scalar_metrics(C_agg[valid_agg_post], P_mean[valid_agg_post])
        valid_agg_prior = np.isfinite(C_agg) & np.isfinite(PR_mean)
        agg_prior = _all_scalar_metrics(C_agg[valid_agg_prior], PR_mean[valid_agg_prior])
        valid_agg_rem   = np.isfinite(C_agg) & np.isfinite(removed_mean)
        agg_removed = _all_scalar_metrics(C_agg[valid_agg_rem], removed_mean[valid_agg_rem])
        valid_agg_add   = np.isfinite(C_agg) & np.isfinite(added_mean)
        agg_added   = _all_scalar_metrics(C_agg[valid_agg_add], added_mean[valid_agg_add])

        self.print_summary(
            spearman_rho, pearson_r, kendall_tau_arr, topk_arr,
            roc_auc_arr, ap_arr, mi_arr,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr,
            global_post, global_prior, agg_post, agg_prior,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr,
            global_removed, global_added, agg_removed, agg_added,
        )

        # Merged metrics dict for saving
        metrics: dict[str, float] = {}
        for k, v in global_post.items():     metrics[f"Global Post {k}"] = v
        for k, v in global_prior.items():    metrics[f"Global Prior {k}"] = v
        for k, v in agg_post.items():        metrics[f"Agg Post {k}"] = v
        for k, v in agg_prior.items():       metrics[f"Agg Prior {k}"] = v
        for k, v in global_removed.items():  metrics[f"Global Removed {k}"] = v
        for k, v in global_added.items():    metrics[f"Global Added {k}"] = v
        for k, v in agg_removed.items():     metrics[f"Agg Removed {k}"] = v
        for k, v in agg_added.items():       metrics[f"Agg Added {k}"] = v

        # --- Save ---
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._save_array(C_T,                self.outdir / "C_effect_over_time.npy")
        self._save_array(P_T,                self.outdir / "posterior_over_time.npy")
        self._save_array(PR_T,               self.outdir / "prior_over_time.npy")
        self._save_array(C_mean,             self.outdir / "C_effect_mean_edge.npy")
        self._save_array(C_agg,              self.outdir / "C_effect_agg_edge.npy")
        self._save_array(P_mean,             self.outdir / "posterior_mean_edge.npy")
        self._save_array(PR_mean,            self.outdir / "prior_mean_edge.npy")
        self._save_array(spearman_rho,       self.outdir / "spearman_rho.npy")
        self._save_array(pearson_r,          self.outdir / "pearson_r.npy")
        self._save_array(kendall_tau_arr,    self.outdir / "kendall_tau.npy")
        self._save_array(topk_arr,           self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(roc_auc_arr,        self.outdir / f"roc_auc_p{int(self.strong_label_percentile)}.npy")
        self._save_array(ap_arr,             self.outdir / f"avg_precision_p{int(self.strong_label_percentile)}.npy")
        self._save_array(mi_arr,             self.outdir / "mutual_info.npy")
        self._save_array(prior_spearman_rho,    self.outdir / "prior_spearman_rho.npy")
        self._save_array(prior_pearson_r,        self.outdir / "prior_pearson_r.npy")
        self._save_array(prior_kendall_tau_arr,  self.outdir / "prior_kendall_tau.npy")
        self._save_array(prior_topk_arr,         self.outdir / f"prior_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(prior_roc_auc_arr,      self.outdir / f"prior_roc_auc_p{int(self.strong_label_percentile)}.npy")
        self._save_array(prior_ap_arr,           self.outdir / f"prior_avg_precision_p{int(self.strong_label_percentile)}.npy")
        self._save_array(prior_mi_arr,           self.outdir / "prior_mutual_info.npy")
        self._save_array(removed_spearman_rho,    self.outdir / "removed_spearman_rho.npy")
        self._save_array(removed_pearson_r,       self.outdir / "removed_pearson_r.npy")
        self._save_array(removed_kendall_tau_arr, self.outdir / "removed_kendall_tau.npy")
        self._save_array(removed_topk_arr,        self.outdir / f"removed_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(removed_roc_auc_arr,     self.outdir / f"removed_roc_auc_p{int(self.strong_label_percentile)}.npy")
        self._save_array(removed_ap_arr,          self.outdir / f"removed_avg_precision_p{int(self.strong_label_percentile)}.npy")
        self._save_array(removed_mi_arr,          self.outdir / "removed_mutual_info.npy")
        self._save_array(added_spearman_rho,      self.outdir / "added_spearman_rho.npy")
        self._save_array(added_pearson_r,         self.outdir / "added_pearson_r.npy")
        self._save_array(added_kendall_tau_arr,   self.outdir / "added_kendall_tau.npy")
        self._save_array(added_topk_arr,          self.outdir / f"added_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(added_roc_auc_arr,       self.outdir / f"added_roc_auc_p{int(self.strong_label_percentile)}.npy")
        self._save_array(added_ap_arr,            self.outdir / f"added_avg_precision_p{int(self.strong_label_percentile)}.npy")
        self._save_array(added_mi_arr,            self.outdir / "added_mutual_info.npy")
        np.save(self.outdir / "scalar_metrics_dict.npy",           metrics,        allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_global_post.npy",    global_post,    allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_global_prior.npy",   global_prior,   allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_agg_post.npy",       agg_post,       allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_agg_prior.npy",      agg_prior,      allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_global_removed.npy", global_removed, allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_global_added.npy",   global_added,   allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_agg_removed.npy",    agg_removed,    allow_pickle=True)
        np.save(self.outdir / "scalar_metrics_agg_added.npy",      agg_added,      allow_pickle=True)
        node_counts = self._node_action_counts if self._node_action_counts is not None else np.array([])
        self._save_array(node_counts, self.outdir / "node_action_counts.npy")

        # --- Plots ---
        self.generate_plots(
            C_T, P_T, PR_T, C_mean, C_agg, P_mean, PR_mean,
            spearman_rho, pearson_r, kendall_tau_arr, topk_arr, roc_auc_arr, ap_arr, mi_arr,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr,
            metrics, node_counts,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr,
        )

    def print_summary_from_saved(self):
        """Loads saved metric arrays and prints the summary table without regenerating plots."""
        spearman_rho     = np.load(self.outdir / "spearman_rho.npy")
        pearson_r        = np.load(self.outdir / "pearson_r.npy")
        kendall_tau_arr  = np.load(self.outdir / "kendall_tau.npy")
        topk_arr         = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        roc_auc_arr      = np.load(self.outdir / f"roc_auc_p{int(self.strong_label_percentile)}.npy")
        ap_arr           = np.load(self.outdir / f"avg_precision_p{int(self.strong_label_percentile)}.npy")
        mi_arr           = np.load(self.outdir / "mutual_info.npy")
        prior_spearman_rho    = np.load(self.outdir / "prior_spearman_rho.npy")
        prior_pearson_r       = np.load(self.outdir / "prior_pearson_r.npy")
        prior_kendall_tau_arr = np.load(self.outdir / "prior_kendall_tau.npy")
        prior_topk_arr        = np.load(self.outdir / f"prior_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        prior_roc_auc_arr     = np.load(self.outdir / f"prior_roc_auc_p{int(self.strong_label_percentile)}.npy")
        prior_ap_arr          = np.load(self.outdir / f"prior_avg_precision_p{int(self.strong_label_percentile)}.npy")
        prior_mi_arr          = np.load(self.outdir / "prior_mutual_info.npy")

        def _try_load(path):
            return np.load(path) if path.exists() else None
        def _try_load_dict(path):
            return np.load(path, allow_pickle=True).item() if path.exists() else {}

        removed_spearman_rho    = _try_load(self.outdir / "removed_spearman_rho.npy")
        removed_pearson_r       = _try_load(self.outdir / "removed_pearson_r.npy")
        removed_kendall_tau_arr = _try_load(self.outdir / "removed_kendall_tau.npy")
        removed_topk_arr        = _try_load(self.outdir / f"removed_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        removed_roc_auc_arr     = _try_load(self.outdir / f"removed_roc_auc_p{int(self.strong_label_percentile)}.npy")
        removed_ap_arr          = _try_load(self.outdir / f"removed_avg_precision_p{int(self.strong_label_percentile)}.npy")
        removed_mi_arr          = _try_load(self.outdir / "removed_mutual_info.npy")
        added_spearman_rho      = _try_load(self.outdir / "added_spearman_rho.npy")
        added_pearson_r         = _try_load(self.outdir / "added_pearson_r.npy")
        added_kendall_tau_arr   = _try_load(self.outdir / "added_kendall_tau.npy")
        added_topk_arr          = _try_load(self.outdir / f"added_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        added_roc_auc_arr       = _try_load(self.outdir / f"added_roc_auc_p{int(self.strong_label_percentile)}.npy")
        added_ap_arr            = _try_load(self.outdir / f"added_avg_precision_p{int(self.strong_label_percentile)}.npy")
        added_mi_arr            = _try_load(self.outdir / "added_mutual_info.npy")

        _load = lambda f: _try_load_dict(self.outdir / f)
        global_post     = _load("scalar_metrics_global_post.npy")
        global_prior    = _load("scalar_metrics_global_prior.npy")
        agg_post        = _load("scalar_metrics_agg_post.npy")
        agg_prior       = _load("scalar_metrics_agg_prior.npy")
        global_removed  = _load("scalar_metrics_global_removed.npy")
        global_added    = _load("scalar_metrics_global_added.npy")
        agg_removed     = _load("scalar_metrics_agg_removed.npy")
        agg_added       = _load("scalar_metrics_agg_added.npy")

        self.print_summary(
            spearman_rho, pearson_r, kendall_tau_arr, topk_arr,
            roc_auc_arr, ap_arr, mi_arr,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr,
            global_post, global_prior, agg_post, agg_prior,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr,
            global_removed, global_added, agg_removed, agg_added,
        )

    # ------------------------------------------------------------------
    # Repaint
    # ------------------------------------------------------------------

    def repaint(self):
        """Re-loads all saved arrays and regenerates all plots."""
        C_T    = np.load(self.outdir / "C_effect_over_time.npy")
        P_T    = np.load(self.outdir / "posterior_over_time.npy")
        PR_T   = np.load(self.outdir / "prior_over_time.npy")
        C_mean = np.load(self.outdir / "C_effect_mean_edge.npy")
        P_mean = np.load(self.outdir / "posterior_mean_edge.npy")
        PR_mean = np.load(self.outdir / "prior_mean_edge.npy")
        agg_path = self.outdir / "C_effect_agg_edge.npy"
        if agg_path.exists():
            C_agg = np.load(agg_path)
        else:
            E = C_T.shape[1]
            N_infer = int(round((1 + (1 + 4 * E) ** 0.5) / 2))
            C_agg = self._build_aggregated_coupling(C_T, num_nodes=N_infer)

        spearman_rho     = np.load(self.outdir / "spearman_rho.npy")
        pearson_r        = np.load(self.outdir / "pearson_r.npy")
        kendall_tau_arr  = np.load(self.outdir / "kendall_tau.npy")
        topk_arr         = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        roc_auc_arr      = np.load(self.outdir / f"roc_auc_p{int(self.strong_label_percentile)}.npy")
        ap_arr           = np.load(self.outdir / f"avg_precision_p{int(self.strong_label_percentile)}.npy")
        mi_arr           = np.load(self.outdir / "mutual_info.npy")

        prior_spearman_rho    = np.load(self.outdir / "prior_spearman_rho.npy")
        prior_pearson_r       = np.load(self.outdir / "prior_pearson_r.npy")
        prior_kendall_tau_arr = np.load(self.outdir / "prior_kendall_tau.npy")
        prior_topk_arr        = np.load(self.outdir / f"prior_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        prior_roc_auc_arr     = np.load(self.outdir / f"prior_roc_auc_p{int(self.strong_label_percentile)}.npy")
        prior_ap_arr          = np.load(self.outdir / f"prior_avg_precision_p{int(self.strong_label_percentile)}.npy")
        prior_mi_arr          = np.load(self.outdir / "prior_mutual_info.npy")

        def _try_load(path):
            return np.load(path) if path.exists() else None

        removed_spearman_rho    = _try_load(self.outdir / "removed_spearman_rho.npy")
        removed_pearson_r       = _try_load(self.outdir / "removed_pearson_r.npy")
        removed_kendall_tau_arr = _try_load(self.outdir / "removed_kendall_tau.npy")
        removed_topk_arr        = _try_load(self.outdir / f"removed_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        removed_roc_auc_arr     = _try_load(self.outdir / f"removed_roc_auc_p{int(self.strong_label_percentile)}.npy")
        removed_ap_arr          = _try_load(self.outdir / f"removed_avg_precision_p{int(self.strong_label_percentile)}.npy")
        removed_mi_arr          = _try_load(self.outdir / "removed_mutual_info.npy")
        added_spearman_rho      = _try_load(self.outdir / "added_spearman_rho.npy")
        added_pearson_r         = _try_load(self.outdir / "added_pearson_r.npy")
        added_kendall_tau_arr   = _try_load(self.outdir / "added_kendall_tau.npy")
        added_topk_arr          = _try_load(self.outdir / f"added_topk_overlap_{int(self.topk_frac*100)}pct.npy")
        added_roc_auc_arr       = _try_load(self.outdir / f"added_roc_auc_p{int(self.strong_label_percentile)}.npy")
        added_ap_arr            = _try_load(self.outdir / f"added_avg_precision_p{int(self.strong_label_percentile)}.npy")
        added_mi_arr            = _try_load(self.outdir / "added_mutual_info.npy")

        metrics_path = self.outdir / "scalar_metrics_dict.npy"
        metrics = np.load(metrics_path, allow_pickle=True).item() if metrics_path.exists() else None
        counts_path = self.outdir / "node_action_counts.npy"
        node_counts = np.load(counts_path) if counts_path.exists() else None

        self.generate_plots(
            C_T, P_T, PR_T, C_mean, C_agg, P_mean, PR_mean,
            spearman_rho, pearson_r, kendall_tau_arr, topk_arr, roc_auc_arr, ap_arr, mi_arr,
            prior_spearman_rho, prior_pearson_r, prior_kendall_tau_arr, prior_topk_arr,
            prior_roc_auc_arr, prior_ap_arr, prior_mi_arr,
            metrics, node_counts,
            removed_spearman_rho, removed_pearson_r, removed_kendall_tau_arr, removed_topk_arr,
            removed_roc_auc_arr, removed_ap_arr, removed_mi_arr,
            added_spearman_rho, added_pearson_r, added_kendall_tau_arr, added_topk_arr,
            added_roc_auc_arr, added_ap_arr, added_mi_arr,
        )

    # ------------------------------------------------------------------
    # Graph visualisation
    # ------------------------------------------------------------------

    def _shrink_axis_box(self, ax, left=0.0, right=0.0, bottom=0.0, top=0.0):
        """
        Shrink an axis *inside its allocated gridspec cell* by fractions of its size.
        Fractions are relative to the current axis width/height.
        """
        pos = ax.get_position()
        new_x0 = pos.x0 + pos.width * left
        new_y0 = pos.y0 + pos.height * bottom
        new_w = pos.width * (1.0 - left - right)
        new_h = pos.height * (1.0 - bottom - top)
        ax.set_position([new_x0, new_y0, new_w, new_h])

    def _visualize_coupling_graph(
            self,
            C_mean,
            P_mean,
            PR_mean,
            node_counts=None,
            timesteps_total=1,
    ):
        env = self._environment
        obs_space = GraphObservationConverter(g2op_obs_space=env.observation_space)
        node_styles = get_node_styles(env, obs_space.__class__)
        N = 2 * env.n_line + env.n_gen + env.n_load

        have_counts = node_counts is not None and node_counts.shape[0] == N
        denom = float(max(int(timesteps_total), 1))

        # --- Percent counts (do NOT mutate node_counts) ---
        counts_pct = None
        if have_counts:
            counts_pct = (node_counts.astype(np.float64) / denom) * 100.0

        # --- Shared colormap / normalization ---
        cmap = plt.cm.get_cmap("coolwarm")
        zero_color = (0.85, 0.85, 0.85, 1.0)

        if have_counts:
            vmax = float(max(counts_pct.max(), 1e-12))  # avoid zero vmax
        else:
            vmax = 1.0
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        # --- Apply colors to graph nodes from percent counts ---
        if have_counts:
            for i, c in enumerate(counts_pct):
                node_styles[i].color = zero_color if c <= 0.0 else cmap(norm(c))

        for probs, title, fname in [
            (np.stack([C_mean, 1.0 - C_mean], axis=1),
             r"Mean effect coupling $\bar{C}_{ij}^{\mathrm{effect}}$",
             "graph_C_effect_mean.png"),
            (np.stack([P_mean, 1.0 - P_mean], axis=1),
             r"Mean posterior $\bar{q}_\phi(z_{ij})$",
             "graph_mean_posterior.png"),
            (np.stack([PR_mean, 1.0 - PR_mean], axis=1),
             r"Mean prior $\bar{p}_\phi(z_{ij})$ (baseline)",
             "graph_mean_prior.png"),
        ]:
            # Layout (turn off constrained_layout because we manually position/shrink axes)
            fig = plt.figure(figsize=(12, 4), constrained_layout=False)
            gs = fig.add_gridspec(
                nrows=1, ncols=3,
                width_ratios=[2.5, 1.5, 0.10],
                wspace=0.25,  # <-- ensures ylabel doesn't clip/overlap the graph
            )

            ax_graph = fig.add_subplot(gs[0, 0])
            ax_hist = fig.add_subplot(gs[0, 1])
            cax = fig.add_subplot(gs[0, 2])

            # Global title centered over everything
            st = fig.suptitle("Node reconfiguration frequency (%) across time", x=0.5, y=0.98, ha="center")

            # Draw graph
            visualize_graph(
                PlottingArgs(
                    num_nodes=N,
                    node_styles=node_styles,
                    latent_edge_probs=probs,
                    powerline_edge_index=self._powerline_edge_index,
                    show_legend=False,
                ),
                ax=ax_graph,
            )

            # Shrink axes to mimic the graph's internal padding
            self._shrink_axis_box(ax_hist, left=0.03, right=0.03, bottom=0.06, top=0.06)
            self._shrink_axis_box(cax, left=0.0, right=0.0, bottom=0.06, top=0.06)

            # --- IMPORTANT: make colorbar match histogram height exactly ---
            hist_pos = ax_hist.get_position()
            cax_pos = cax.get_position()
            cax.set_position([cax_pos.x0, hist_pos.y0, cax_pos.width, hist_pos.height])

            # Histogram
            if have_counts and counts_pct.size > 0:
                x = np.arange(N)
                bar_colors = [zero_color if c <= 0.0 else cmap(norm(c)) for c in counts_pct]
                ax_hist.bar(x, counts_pct, color=bar_colors, edgecolor="none")

                ax_hist.set_xlabel("Node index")
                ax_hist.set_ylabel("Reconfiguration frequency", labelpad=8)  # small pad
                ax_hist.set_xlim(-0.5, N - 0.5)
                ax_hist.margins(x=0.02, y=0.05)
                ax_hist.set_ylim(0.0, vmax)

            # Colorbar
            if have_counts:
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cb = fig.colorbar(sm, cax=cax)
                cb.set_label("Reconfiguration frequency")

            # Leave room for suptitle
            fig.subplots_adjust(top=0.86)

            # Save (include suptitle reliably when using tight bbox)
            fig.savefig(self.outdir / fname, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=[st])
            fig.savefig(self.outdir / (Path(fname).stem + ".svg"), bbox_inches="tight", pad_inches=0.05,
                        bbox_extra_artists=[st])
            plt.show()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def generate_plots(
        self,
        C_T: npt.NDArray,               # [T_action, E]
        P_T: npt.NDArray,               # [T_action, E] – posterior
        PR_T: npt.NDArray,              # [T_action, E] – prior  (baseline)
        C_mean: npt.NDArray,            # [E]
        C_agg: npt.NDArray,             # [E] sum-over-time then row-normalised
        P_mean: npt.NDArray,            # [E]
        PR_mean: npt.NDArray,           # [E]
        # posterior per-timestep distributions
        spearman_rho: npt.NDArray,
        pearson_r: npt.NDArray,
        kendall_tau_arr: npt.NDArray,
        topk_arr: npt.NDArray,
        roc_auc_arr: npt.NDArray,
        ap_arr: npt.NDArray,
        mi_arr: npt.NDArray,
        # prior per-timestep distributions  (baseline)
        prior_spearman_rho: npt.NDArray,
        prior_pearson_r: npt.NDArray,
        prior_kendall_tau_arr: npt.NDArray,
        prior_topk_arr: npt.NDArray,
        prior_roc_auc_arr: npt.NDArray,
        prior_ap_arr: npt.NDArray,
        prior_mi_arr: npt.NDArray,
        metrics: dict | None = None,
        node_counts: npt.NDArray | None = None,
        # edges removed p(1-q) per-timestep distributions
        removed_spearman_rho: npt.NDArray | None = None,
        removed_pearson_r: npt.NDArray | None = None,
        removed_kendall_tau_arr: npt.NDArray | None = None,
        removed_topk_arr: npt.NDArray | None = None,
        removed_roc_auc_arr: npt.NDArray | None = None,
        removed_ap_arr: npt.NDArray | None = None,
        removed_mi_arr: npt.NDArray | None = None,
        # edges added q(1-p) per-timestep distributions
        added_spearman_rho: npt.NDArray | None = None,
        added_pearson_r: npt.NDArray | None = None,
        added_kendall_tau_arr: npt.NDArray | None = None,
        added_topk_arr: npt.NDArray | None = None,
        added_roc_auc_arr: npt.NDArray | None = None,
        added_ap_arr: npt.NDArray | None = None,
        added_mi_arr: npt.NDArray | None = None,
    ):
        sns.reset_orig()
        mpl.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        })

        cl = r"Effect coupling"

        # Derived mean distributions
        removed_mean = PR_mean * (1.0 - P_mean)   # p(1-q): [E]
        added_mean   = P_mean  * (1.0 - PR_mean)  # q(1-p): [E]

        # ---- Side-by-side histograms: posterior (left) | prior (right), shared axes ----
        def _hist2(post_vals, prior_vals, metric, xlabel, outpath, vline=None, vline_label=None):
            outpath.parent.mkdir(parents=True, exist_ok=True)
            combined = np.concatenate([post_vals[np.isfinite(post_vals)],
                                       prior_vals[np.isfinite(prior_vals)]])
            if combined.size > 1:
                bins = np.linspace(combined.min(), combined.max(), 31)
            else:
                bins = 30
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
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
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(outpath.parent / (outpath.stem + ".png"))
            plt.savefig(outpath.parent / (outpath.stem + ".svg"))
            plt.show()

        # ---- 4-panel histograms: posterior | prior | removed p(1-q) | added q(1-p) ----
        def _hist4(post_vals, prior_vals, rem_vals, add_vals,
                   metric, xlabel, outpath, vline=None, vline_label=None):
            if rem_vals is None or add_vals is None:
                return
            outpath.parent.mkdir(parents=True, exist_ok=True)
            all_vals = [post_vals, prior_vals, rem_vals, add_vals]
            labels = ["posterior", "prior (baseline)", r"removal prob", r"addition prob"]
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
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            stem = outpath.stem + "_4way"
            plt.savefig(outpath.parent / (stem + ".png"))
            plt.savefig(outpath.parent / (stem + ".svg"))
            plt.show()

        # ---- Graph visualisation ----
        # if self._environment is None or self._powerline_edge_index is None:
        #     import grid2op
        #     from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
        #     self._environment = grid2op.make("l2rpn_case14_sandbox")
        #     obs_space = BusConnectivityGraphObsSpace(grid2op_observation_space=self._environment.observation_space)
        #     self._powerline_edge_index = obs_space.to_gym(self._environment.reset())[EDGE_INDEX]
        #
        # self._visualize_coupling_graph(C_mean, P_mean, PR_mean, node_counts, timesteps_total = C_T.shape[0])
        #
        # if node_counts is not None and node_counts.size > 0:
        #     plt.figure(figsize=(max(8, len(node_counts) // 3), 4))
        #     plt.bar(range(len(node_counts)), node_counts)
        #     plt.xlabel("Node index")
        #     plt.ylabel("Reconfiguration count")
        #     plt.title(r"Per-node reconfiguration frequency ($V(a_t)$ membership count)")
        #     plt.tight_layout()
        #     plt.savefig(self.outdir / "bar_node_reconfiguration_counts.png")
        #     plt.savefig(self.outdir / "bar_node_reconfiguration_counts.svg")
        #     plt.show()

        # ---- KDE: mean C conditioned on posterior (left) and prior (right) ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)
        for ax, mean_val, suffix in [
            (axes[0], P_mean,  "posterior"),
            (axes[1], PR_mean, "prior"),
            #(axes[1,0], added_mean, "addition prob"),
            #(axes[1,1], removed_mean, "removal prob"),
        ]:
            valid = np.isfinite(C_mean) & np.isfinite(mean_val)
            C_v, X_v = C_mean[valid], mean_val[valid]
            sns.kdeplot(C_v[X_v > 0.8],  label=f"High {suffix} node pairs", ax=ax)
            sns.kdeplot(C_v[X_v <= 0.8], label=f"Low {suffix} node pairs",  ax=ax)
            ax.set_xlabel(r"Mean $\bar{C}_{ij}^{\mathrm{effect}}$")
            ax.set_ylabel("Density")
            ax.set_title(f"{cl} vs {suffix}")
            ax.legend()
        #fig.suptitle(r"$\bar{C}_{ij}^{\mathrm{effect}}$", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.outdir / "kde_C_effect_conditioned_on_posterior_prior.png")
        plt.savefig(self.outdir / "kde_C_effect_conditioned_on_posterior_prior.svg")
        plt.show()

        # # ---- KDE: mean C conditioned on removed p(1-q) and added q(1-p) ----
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # for ax, mean_val, suffix in [
        #     (axes[0], removed_mean, r"removal prob"),
        #     (axes[1], added_mean,   r"addition prob"),
        # ]:
        #     valid = np.isfinite(C_mean) & np.isfinite(mean_val)
        #     C_v, X_v = C_mean[valid], mean_val[valid]
        #     med = 0.8#np.median(X_v) if X_v.size > 0 else 0.5
        #     sns.kdeplot(C_v[X_v > med],  label=f"High {suffix} node pairs", ax=ax, bw_adjust=0.5)
        #     sns.kdeplot(C_v[X_v <= med], label=f"Low {suffix} node pairs",  ax=ax, bw_adjust=0.5)
        #     #ax.set_xlim(0, 1)
        #     ax.set_xlabel(r"Mean $\bar{C}_{ij}^{\mathrm{effect}}$")
        #     ax.set_ylabel("Density")
        #     ax.set_title(f"{cl} vs {suffix}")
        #     ax.legend()
        # #fig.suptitle(r"$\bar{C}_{ij}^{\mathrm{effect}}$ – removed/added", y=0.98)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(self.outdir / "kde_C_effect_conditioned_on_removed_added.png")
        # plt.savefig(self.outdir / "kde_C_effect_conditioned_on_removed_added.svg")
        # plt.show()

        # # Scatter: posterior / prior / removed / added vs mean C
        # for mean_val, suffix in [
        #     (P_mean,       "posterior"),
        #     (PR_mean,      "prior"),
        #     (removed_mean, "removed_p1mq"),
        #     (added_mean,   "added_q1mp"),
        # ]:
        #     valid = np.isfinite(C_mean) & np.isfinite(mean_val)
        #     C_v, X_v = C_mean[valid], mean_val[valid]
        #     label_suffix = suffix.replace("_", " ")
        #     self._save_scatter(
        #         values=(X_v, C_v),
        #         title=f"Scatter: {cl} vs {label_suffix}",
        #         xlabel=f"Mean {label_suffix} existence probability",
        #         ylabel=r"Mean $C_{ij}^{\mathrm{effect}}$",
        #         outpath=self.outdir / f"scatter_{suffix}_vs_C_effect.png",
        #     )

        # ---- KDE: aggregated C conditioned on posterior (left) and prior (right) ----
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # for ax, mean_val, suffix in [
        #     (axes[0], P_mean,  "posterior"),
        #     (axes[1], PR_mean, "prior"),
        # ]:
        #     valid_agg = np.isfinite(C_agg) & np.isfinite(mean_val)
        #     C_agg_v, X_agg_v = C_agg[valid_agg], mean_val[valid_agg]
        #     sns.kdeplot(C_agg_v[X_agg_v > 0.8],  label=f"High {suffix} node pairs", ax=ax)
        #     sns.kdeplot(C_agg_v[X_agg_v <= 0.8], label=f"Low {suffix} node pairs",  ax=ax)
        #     ax.set_xlabel(r"Aggregated $\tilde{C}_{ij}^{\mathrm{effect}}$ (row-normalised)")
        #     ax.set_ylabel("Density")
        #     ax.set_title(f"{cl} vs {suffix}")
        #     ax.legend()
        # #fig.suptitle(r"Aggregated $\tilde{C}_{ij}^{\mathrm{effect}}$", y=0.98)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(self.outdir / "kde_C_effect_agg_conditioned_on_posterior_prior.png")
        # plt.savefig(self.outdir / "kde_C_effect_agg_conditioned_on_posterior_prior.svg")
        # plt.show()
        #
        # # ---- KDE: aggregated C conditioned on removed and added ----
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        # for ax, mean_val, suffix in [
        #     (axes[0], removed_mean, r"addition prob"),
        #     (axes[1], added_mean,   r"removal prob"),
        # ]:
        #     valid_agg = np.isfinite(C_agg) & np.isfinite(mean_val)
        #     C_agg_v, X_agg_v = C_agg[valid_agg], mean_val[valid_agg]
        #     med = 0.8#np.median(X_agg_v) if X_agg_v.size > 0 else 0.5
        #     sns.kdeplot(C_agg_v[X_agg_v > med],  label=f"High {suffix} node pairs", ax=ax)
        #     sns.kdeplot(C_agg_v[X_agg_v <= med], label=f"Low {suffix} node pairs",  ax=ax)
        #     ax.set_xlabel(r"Aggregated $\tilde{C}_{ij}^{\mathrm{effect}}$ (row-normalised)")
        #     ax.set_ylabel("Density")
        #     ax.set_title(f"{cl} vs {suffix}")
        #     ax.legend()
        # #fig.suptitle(r"Aggregated $\tilde{C}_{ij}^{\mathrm{effect}}$ – removed/added", y=0.98)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(self.outdir / "kde_C_effect_agg_conditioned_on_removed_added.png")
        # plt.savefig(self.outdir / "kde_C_effect_agg_conditioned_on_removed_added.svg")
        # plt.show()

        # Scatter: removed / added vs aggregated C
        # for mean_val, suffix in [
        #     (P_mean,       "posterior"),
        #     (PR_mean,      "prior"),
        #     (removed_mean, "removed_p1mq"),
        #     (added_mean,   "added_q1mp"),
        # ]:
        #     valid_agg = np.isfinite(C_agg) & np.isfinite(mean_val)
        #     C_agg_v, X_agg_v = C_agg[valid_agg], mean_val[valid_agg]
        #     label_suffix = suffix.replace("_", " ")
        #     self._save_scatter(
        #         values=(X_agg_v, C_agg_v),
        #         title=f"Scatter: aggregated {cl} vs {label_suffix}",
        #         xlabel=f"Mean {label_suffix} existence probability",
        #         ylabel=r"Aggregated $\tilde{C}_{ij}^{\mathrm{effect}}$ (row-normalised)",
        #         outpath=self.outdir / f"scatter_{suffix}_vs_C_effect_agg.png",
        #     )

        # ---- Per-timestep metric histograms (2-panel and 4-panel) ----
        _hist2(spearman_rho, prior_spearman_rho,
               metric="Spearman", xlabel=r"Spearman $\rho$",
               outpath=self.outdir / "hist_spearman_rho.png")
        _hist4(spearman_rho, prior_spearman_rho, removed_spearman_rho, added_spearman_rho,
               metric="Spearman", xlabel=r"Spearman $\rho$",
               outpath=self.outdir / "hist_spearman_rho.png")
        #_hist2(pearson_r, prior_pearson_r,
         #      metric="Pearson r", xlabel="Pearson r",
          #     outpath=self.outdir / "hist_pearson_r.png")
        #_hist4(pearson_r, prior_pearson_r, removed_pearson_r, added_pearson_r,
        #       metric="Pearson r", xlabel="Pearson r",
        #       outpath=self.outdir / "hist_pearson_r.png")
        #_hist2(kendall_tau_arr, prior_kendall_tau_arr,
        #       metric="Kendall tau", xlabel=r"Kendall $\tau$",
        #       outpath=self.outdir / "hist_kendall_tau.png")
        #_hist4(kendall_tau_arr, prior_kendall_tau_arr, removed_kendall_tau_arr, added_kendall_tau_arr,
        #       metric="Kendall tau", xlabel=r"Kendall $\tau$",
        #       outpath=self.outdir / "hist_kendall_tau.png")
        #_hist2(topk_arr, prior_topk_arr,
        #       metric=f"Top-{int(self.topk_frac * 100)}% overlap",
        #       xlabel=f"Top-{int(self.topk_frac * 100)}% overlap fraction",
        #       outpath=self.outdir / f"hist_topk_overlap_{int(self.topk_frac * 100)}pct.png",
        #       vline=self.topk_frac, vline_label=f"Random baseline ({self.topk_frac:.0%})")
        #_hist4(topk_arr, prior_topk_arr, removed_topk_arr, added_topk_arr,
        #       metric=f"Top-{int(self.topk_frac * 100)}% overlap",
        #       xlabel=f"Top-{int(self.topk_frac * 100)}% overlap fraction",
        #       outpath=self.outdir / f"hist_topk_overlap_{int(self.topk_frac * 100)}pct.png",
        #       vline=self.topk_frac, vline_label=f"Random baseline ({self.topk_frac:.0%})")
        #_hist2(roc_auc_arr, prior_roc_auc_arr,
        #       metric=f"ROC-AUC (p{int(self.strong_label_percentile)})", xlabel="ROC-AUC",
        #       outpath=self.outdir / f"hist_roc_auc_p{int(self.strong_label_percentile)}.png")
        #_hist4(roc_auc_arr, prior_roc_auc_arr, removed_roc_auc_arr, added_roc_auc_arr,
        #       metric=f"ROC-AUC (p{int(self.strong_label_percentile)})", xlabel="ROC-AUC",
        #       outpath=self.outdir / f"hist_roc_auc_p{int(self.strong_label_percentile)}.png")
       # _hist2(ap_arr, prior_ap_arr,
       #        metric=f"Avg Precision (p{int(self.strong_label_percentile)})", xlabel="Average Precision",
       #        outpath=self.outdir / f"hist_avg_precision_p{int(self.strong_label_percentile)}.png")
       # _hist4(ap_arr, prior_ap_arr, removed_ap_arr, added_ap_arr,
       #        metric=f"Avg Precision (p{int(self.strong_label_percentile)})", xlabel="Average Precision",
       #        outpath=self.outdir / f"hist_avg_precision_p{int(self.strong_label_percentile)}.png")
       # _hist2(mi_arr, prior_mi_arr,
       #        metric="Mutual information", xlabel="MI",
        #       outpath=self.outdir / "hist_mutual_info.png")
        #_hist4(mi_arr, prior_mi_arr, removed_mi_arr, added_mi_arr,
        #       metric="Mutual information", xlabel="MI",
        #       outpath=self.outdir / "hist_mutual_info.png")

        print(f"Saved metrics and plots to: {self.outdir.absolute()}")
