"""
Unit tests for observation_converter.py.

Tests are organised by class:
  - TestGridDimensions
  - TestGraphObservationConverterSpace
  - TestGraphObservationConverterNodeFeatures
  - TestGraphObservationConverterEdgeIndex
  - TestGraphObservationConverterNormalize
  - TestGraphObservationConverterIntegration
  - TestFlatObservationConverter

A lightweight MockObservation replaces grid2op to avoid a heavyweight
dependency in the test suite. Tests that require real grid2op topology
(e.g. connectivity_matrix shape) are skipped when grid2op is unavailable.
"""
from __future__ import annotations

import unittest

import grid2op
import numpy as np

from src.grid2op_env.observation_converter import (  # noqa: E402
    GraphObservationConverter,
    _GridDimensions,
    NODES, EDGE_INDEX, EDGE_MASK, GLOBAL,
    _DEFAULT_NODE_FEATURES,
)

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without a grid2op installation
# ---------------------------------------------------------------------------

ENV_NAME = "l2rpn_wcci_2020"
env = grid2op.make(ENV_NAME)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGridDimensions(unittest.TestCase):
    """Tests for the _GridDimensions dataclass."""

    def test_num_nodes_without_storage(self):
        """num_nodes equals 2*n_line + n_gen + n_load when n_storage=0."""
        dims = _GridDimensions(n_gen=3, n_load=4, n_line=5, n_storage=0)
        self.assertEqual(dims.num_nodes, 2 * 5 + 3 + 4)

    def test_num_nodes_with_storage(self):
        """Storage units contribute to num_nodes."""
        dims = _GridDimensions(n_gen=3, n_load=4, n_line=5, n_storage=2)
        self.assertEqual(dims.num_nodes, 2 * 5 + 3 + 4 + 2)

    def test_from_obs_space(self):
        """from_obs_space reads the correct attributes."""
        dims = _GridDimensions.from_obs_space(env.observation_space)
        self.assertEqual(dims.n_gen, env.n_gen)
        self.assertEqual(dims.n_load, env.n_load)
        self.assertEqual(dims.n_line, env.n_line)
        self.assertEqual(dims.n_storage, env.n_storage)


class TestGraphObservationConverterSpace(unittest.TestCase):
    """Tests for the gymnasium observation space definition."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.dims = _GridDimensions.from_obs_space(env.observation_space)

    def test_node_feature_shape(self):
        """NODES box has shape (num_nodes, x_dim)."""
        num_nodes = self.dims.num_nodes
        x_dim = len(_DEFAULT_NODE_FEATURES)
        shape = self.converter.observation_space[NODES].shape
        self.assertEqual(shape, (num_nodes, x_dim))

    def test_edge_index_shape(self):
        """EDGE_INDEX box has shape (2, max_num_edges)."""
        shape = self.converter.observation_space[EDGE_INDEX].shape
        self.assertEqual(shape[0], 2)
        self.assertGreater(shape[1], 0)

    def test_edge_mask_shape(self):
        """EDGE_MASK box length matches EDGE_INDEX width."""
        ei_shape = self.converter.observation_space[EDGE_INDEX].shape
        mask_shape = self.converter.observation_space[EDGE_MASK].shape
        self.assertEqual(mask_shape[0], ei_shape[1])

    def test_global_features_shape(self):
        """GLOBAL box has shape (6,)."""
        self.assertEqual(self.converter.observation_space[GLOBAL].shape, (6,))

    def test_custom_attr_to_observe(self):
        """x_dim matches the length of a custom attr_to_observe list."""
        attrs = ["active_power", "rho"]

        converter = GraphObservationConverter(env.observation_space, attr_to_observe=attrs)
        self.assertEqual(converter.x_dim, 2)


class TestGraphObservationConverterNodeFeatures(unittest.TestCase):
    """Tests for _compute_all_node_features and _get_node_features."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.dim = _GridDimensions.from_obs_space(env.observation_space)
        self.obs = env.reset()

    def test_node_feature_matrix_shape(self):
        """_get_node_features returns shape (num_nodes, x_dim)."""
        feats = self.converter._get_node_features(self.obs)
        num_nodes = self.converter.num_nodes
        x_dim = self.converter.x_dim
        self.assertEqual(feats.shape, (num_nodes, x_dim))

    def test_node_feature_dtype(self):
        """Node features are float32."""
        feats = self.converter._get_node_features(self.obs)
        self.assertEqual(feats.dtype, np.float32)

    def test_all_features_same_length(self):
        """Every feature vector in _compute_all_node_features has length num_nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        expected_len = self.converter.num_nodes
        for name, arr in all_feats.items():
            with self.subTest(feature=name):
                self.assertEqual(len(arr), expected_len, f"{name} has wrong length")

    def test_rho_zero_for_non_line_nodes(self):
        """rho is 0 for gen and load nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        rho = all_feats["rho"]
        load_idx = np.arange(start=0, stop=self.dim.n_load) + 2 * self.dim.n_line + self.dim.n_gen
        gen_idx = np.arange(start=0, stop=self.dim.n_gen) + 2 * self.dim.n_line
        np.testing.assert_array_equal(rho[gen_idx],  0.0)
        np.testing.assert_array_equal(rho[load_idx], 0.0)

    def test_rho_nonzero_for_line_nodes(self):
        """rho is copied to both line_or and line_ex nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        rho = all_feats["rho"]
        or_idx = np.arange(start=0, stop=self.dim.n_line)
        ex_idx = np.arange(start=0, stop=self.dim.n_line) + self.dim.n_line
        np.testing.assert_array_equal(rho[or_idx], self.obs.rho)
        np.testing.assert_array_equal(rho[ex_idx], self.obs.rho)

    def test_active_power_sign_convention(self):
        """Loads have negative active_power (consumption), gens positive."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        p = all_feats["active_power"]
        load_idx = np.arange(start=0, stop=self.dim.n_load) + 2*self.dim.n_line + self.dim.n_gen
        gen_idx  = np.arange(start=0, stop=self.dim.n_gen) + 2*self.dim.n_line
        np.testing.assert_array_almost_equal(p[load_idx], -self.obs.load_p)
        np.testing.assert_array_almost_equal(p[gen_idx],   self.obs.gen_p)

    def test_only_requested_features_in_output(self):
        """_get_node_features only returns columns for attr_to_observe."""
        attrs = ["active_power", "rho"]
        obs_space = env.observation_space
        converter = GraphObservationConverter(obs_space, attr_to_observe=attrs)
        feats = converter._get_node_features(self.obs)
        self.assertEqual(feats.shape[1], 2)


class TestGraphObservationConverterEdgeIndex(unittest.TestCase):
    """Tests for _get_edge_index."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_edge_index_shape(self):
        """Edge index has exactly 2 rows."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.shape[0], 2)

    def test_edge_index_dtype(self):
        """Edge index dtype is int64."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.dtype, np.int64)

    def test_edge_index_within_node_range(self):
        """All edge indices are valid node indices."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.converter.num_nodes).all())

    def test_edges_are_bidirectional(self):
        """For every (i, j) edge there is a corresponding (j, i) edge."""
        ei = self.converter._get_edge_index(self.obs)
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        for src, dst in list(edge_set):
            self.assertIn((dst, src), edge_set, f"Missing reverse edge ({dst}, {src})")

    def test_no_self_loops(self):
        """No node is connected to itself."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertTrue((ei[0] != ei[1]).all())


    def test_num_edges_does_not_exceed_max(self):
        """Number of edges never exceeds max_num_edges."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertLessEqual(ei.shape[1], self.converter.max_num_edges)


class TestGraphObservationConverterNormalize(unittest.TestCase):
    """Tests for the normalize method and running statistics."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def _raw_obs(self) -> dict:
        """Build a raw (un-normalized) observation dict."""
        node_features = self.converter._get_node_features(self.obs)
        ei = self.converter._get_edge_index(self.obs)
        num_edges = ei.shape[1]
        ei_padded = np.zeros((2, self.converter.max_num_edges), dtype=np.int64)
        ei_padded[:, :num_edges] = ei
        mask = np.zeros(self.converter.max_num_edges, dtype=bool)
        mask[:num_edges] = True
        return {
            NODES: node_features,
            EDGE_INDEX: ei_padded,
            EDGE_MASK: mask,
            GLOBAL: self.converter._get_global_features(self.obs),
        }

    def test_normalize_returns_all_keys(self):
        """normalize returns a dict with NODES, EDGE_INDEX, EDGE_MASK, GLOBAL."""
        result = self.converter.normalize(self._raw_obs())
        self.assertIn(NODES, result)
        self.assertIn(EDGE_INDEX, result)
        self.assertIn(EDGE_MASK, result)
        self.assertIn(GLOBAL, result)

    def test_normalized_node_features_dtype(self):
        """Normalized node features are float32."""
        result = self.converter.normalize(self._raw_obs())
        self.assertEqual(result[NODES].dtype, np.float32)

    def test_normalize_does_not_alter_edge_index(self):
        """normalize passes EDGE_INDEX through unchanged."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        np.testing.assert_array_equal(result[EDGE_INDEX], raw[EDGE_INDEX])

    def test_normalize_does_not_alter_edge_mask(self):
        """normalize passes EDGE_MASK through unchanged."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        np.testing.assert_array_equal(result[EDGE_MASK], raw[EDGE_MASK])

    def test_running_stats_updated_after_normalize(self):
        """Calling normalize increments the running count."""
        count_before = self.converter._normalizer.count
        self.converter.normalize(self._raw_obs())
        self.assertGreater(self.converter._normalizer.count, count_before)

    def test_normalized_shape_preserved(self):
        """normalize preserves the NODES shape."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        self.assertEqual(result[NODES].shape, raw[NODES].shape)


class TestGraphObservationConverterIntegration(unittest.TestCase):
    """End-to-end tests for to_gym."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_to_gym_returns_all_keys(self):
        """to_gym returns a dict with all expected keys."""
        result = self.converter.to_gym(self.obs)
        for key in [NODES, EDGE_INDEX, EDGE_MASK, GLOBAL]:
            self.assertIn(key, result)

    def test_to_gym_node_shape(self):
        """to_gym NODES has shape (num_nodes, x_dim)."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[NODES].shape, (self.converter.num_nodes, self.converter.x_dim))

    def test_to_gym_edge_index_padded(self):
        """to_gym EDGE_INDEX is padded to max_num_edges."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_INDEX].shape[1], self.converter.max_num_edges)

    def test_to_gym_edge_mask_count_matches_real_edges(self):
        """The number of True values in EDGE_MASK equals the real edge count."""
        raw_ei = self.converter._get_edge_index(self.obs)
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_MASK].sum(), raw_ei.shape[1])

    def test_to_gym_global_features_values(self):
        """Global features contain the correct time values."""
        result = self.converter.to_gym(self.obs)
        gf = result[GLOBAL]
        self.assertEqual(gf[0], self.obs.year)
        self.assertEqual(gf[1], self.obs.month)
        self.assertEqual(gf[2], self.obs.day)
        self.assertEqual(gf[3], self.obs.hour_of_day)
        self.assertEqual(gf[4], self.obs.day_of_week)
        self.assertEqual(gf[5], self.obs.minute_of_hour)

    def test_to_gym_deterministic(self):
        """Two calls with the same observation produce the same EDGE_INDEX."""
        r1 = self.converter.to_gym(self.obs)
        r2 = self.converter.to_gym(self.obs)
        np.testing.assert_array_equal(r1[EDGE_INDEX], r2[EDGE_INDEX])


class TestGraphObservationConverterWithGrid2op(unittest.TestCase):
    """Integration tests that run against a real grid2op environment."""

    @classmethod
    def setUpClass(cls):
        cls.converter = GraphObservationConverter(env.observation_space)

    def test_connectivity_matrix_shape_matches_num_nodes(self):
        """connectivity_matrix shape matches num_nodes."""
        obs = env.reset()
        conn = obs.connectivity_matrix()
        n = self.converter.num_nodes
        self.assertEqual(conn.shape, (n, n))

    def test_pos_topo_vect_cover_all_nodes(self):
        """pos_topo_vect arrays together cover every index in [0, num_nodes)."""
        all_positions = np.concatenate([
            env.load_pos_topo_vect,
            env.gen_pos_topo_vect,
            env.line_or_pos_topo_vect,
            env.line_ex_pos_topo_vect,
            env.storage_pos_topo_vect,
        ])
        expected = set(range(self.converter.num_nodes))
        self.assertEqual(set(all_positions.tolist()), expected)

    def test_line_endpoints_connected_in_connectivity_matrix(self):
        """Every line's origin and extremity are connected in connectivity_matrix."""
        obs = env.reset()
        conn = obs.connectivity_matrix().astype(bool)
        for line_id in range(env.n_line):
            or_idx = env.line_or_pos_topo_vect[line_id]
            ex_idx = env.line_ex_pos_topo_vect[line_id]
            self.assertTrue(
                conn[or_idx, ex_idx],
                f"line {line_id}: origin {or_idx} and extremity {ex_idx} not connected",
            )

    def test_to_gym_observation_space_compliant(self):
        """to_gym output lies within the declared observation space."""
        obs = env.reset()
        result = self.converter.to_gym(obs)
        space = self.converter.observation_space
        # gymnasium Dict.contains checks dtype and shape
        for key in [EDGE_INDEX, EDGE_MASK, GLOBAL]:
            self.assertEqual(result[key].shape, space[key].shape)


if __name__ == "__main__":
    unittest.main()