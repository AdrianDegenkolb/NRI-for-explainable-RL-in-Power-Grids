"""
Test script to verify that BusConnectivityGraphObsSpace is correctly preserved
when loading from RLlib checkpoints.
"""

import pickle
import tempfile
from pathlib import Path
from pprint import pprint

from gymnasium.spaces import Dict, Box
import numpy as np

from src.common.observation_space import BusConnectivityGraphObsSpace

def fml():
    picklefile = "/home/adrian/Dev/NRI-for-explainable-RL-in-Power-Grids/results/experiments/test_minimal_run/CustomPPO_TEsTING_b8c191ee_2026-01-07_15-44-08/checkpoint_000000/policies/reinforcement_learning_policy/policy_state.pkl"
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
        pprint(data["policy_spec"]['observation_space'])

def test_pickle_serialization():
    """Test that pickle serialization preserves the custom class and attributes."""
    print("=" * 80)
    print("Test 1: Pickle Serialization")
    print("=" * 80)

    # Create a mock grid2op observation space
    class MockObsSpace:
        n_gen = 5
        n_load = 10
        n_line = 20
        sub_info = np.array([3, 4, 5, 6, 3])

    mock_obs_space = MockObsSpace()

    # Create the custom observation space
    print("\n1. Creating BusConnectivityGraphObsSpace...")
    original_space = BusConnectivityGraphObsSpace(
        grid2op_observation_space=mock_obs_space,
        normalization_boundaries={
            'active_power_forecast': (-100, 100),
            'reactive_power_forecast': (-50, 50),
            'active_power': (-100, 100),
            'reactive_power': (-50, 50),
            'voltage': (0, 200),
            'voltage_angle': (-180, 180),
            'current': (0, 1000),
            'rho': (0, 2),
        },
        verbose=False
    )

    print(f"   Type: {type(original_space).__name__}")
    print(f"   x_dim: {original_space.x_dim}")
    print(f"   num_nodes: {original_space.num_nodes}")
    print(f"   max_num_edges: {original_space.max_num_edges}")
    print(f"   Has normalization_min: {hasattr(original_space, 'normalization_min')}")
    print(f"   Has normalization_max: {hasattr(original_space, 'normalization_max')}")

    # Pickle and unpickle
    print("\n2. Pickling and unpickling...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        pickle.dump(original_space, f)
        temp_file = f.name

    with open(temp_file, 'rb') as f:
        restored_space = pickle.load(f)

    # Verify restoration
    print(f"\n3. Verifying restored space...")
    print(f"   Type: {type(restored_space).__name__}")
    print(f"   Is BusConnectivityGraphObsSpace: {isinstance(restored_space, BusConnectivityGraphObsSpace)}")
    print(f"   x_dim: {restored_space.x_dim}")
    print(f"   num_nodes: {restored_space.num_nodes}")
    print(f"   max_num_edges: {restored_space.max_num_edges}")
    print(f"   Has normalization_min: {hasattr(restored_space, 'normalization_min')}")
    print(f"   Has normalization_max: {hasattr(restored_space, 'normalization_max')}")

    # Verify attributes match
    assert isinstance(restored_space, BusConnectivityGraphObsSpace), \
        f"Expected BusConnectivityGraphObsSpace, got {type(restored_space).__name__}"
    assert restored_space.x_dim == original_space.x_dim, \
        f"x_dim mismatch: {restored_space.x_dim} != {original_space.x_dim}"
    assert restored_space.num_nodes == original_space.num_nodes, \
        f"num_nodes mismatch: {restored_space.num_nodes} != {original_space.num_nodes}"
    assert restored_space.max_num_edges == original_space.max_num_edges, \
        f"max_num_edges mismatch: {restored_space.max_num_edges} != {original_space.max_num_edges}"

    if original_space.normalization_min is not None:
        assert np.array_equal(restored_space.normalization_min, original_space.normalization_min), \
            "normalization_min mismatch"
        assert np.array_equal(restored_space.normalization_max, original_space.normalization_max), \
            "normalization_max mismatch"

    print("\n✓ Pickle serialization test PASSED!")

    # Cleanup
    Path(temp_file).unlink()

    return True


def test_multi_agent_dict_wrapper():
    """Test that the custom space works correctly when wrapped in a multi-agent Dict."""
    print("\n" + "=" * 80)
    print("Test 2: Multi-Agent Dict Wrapper")
    print("=" * 80)

    # Create a mock grid2op observation space
    class MockObsSpace:
        n_gen = 5
        n_load = 10
        n_line = 20
        sub_info = np.array([3, 4, 5, 6, 3])

    mock_obs_space = MockObsSpace()

    # Create the custom observation space
    print("\n1. Creating multi-agent observation space...")
    graph_obs_space = BusConnectivityGraphObsSpace(
        grid2op_observation_space=mock_obs_space,
        normalization_boundaries=None,
        verbose=False
    )

    # Wrap in multi-agent Dict (as done in training)
    from gymnasium.spaces import Discrete
    multi_agent_obs_space = Dict({
        "high_level_agent": Discrete(2),
        "reinforcement_learning_agent": graph_obs_space,
        "do_nothing_agent": Discrete(1),
    })

    print(f"   Multi-agent space type: {type(multi_agent_obs_space).__name__}")
    print(f"   RL agent space type: {type(multi_agent_obs_space.spaces['reinforcement_learning_agent']).__name__}")

    # Pickle and unpickle
    print("\n2. Pickling and unpickling multi-agent space...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        pickle.dump(multi_agent_obs_space, f)
        temp_file = f.name

    with open(temp_file, 'rb') as f:
        restored_multi_agent_space = pickle.load(f)

    # Verify restoration
    print(f"\n3. Verifying restored multi-agent space...")
    print(f"   Multi-agent space type: {type(restored_multi_agent_space).__name__}")
    rl_agent_space = restored_multi_agent_space.spaces['reinforcement_learning_agent']
    print(f"   RL agent space type: {type(rl_agent_space).__name__}")
    print(f"   Is BusConnectivityGraphObsSpace: {isinstance(rl_agent_space, BusConnectivityGraphObsSpace)}")

    if isinstance(rl_agent_space, BusConnectivityGraphObsSpace):
        print(f"   x_dim: {rl_agent_space.x_dim}")
        print(f"   num_nodes: {rl_agent_space.num_nodes}")
        print(f"   max_num_edges: {rl_agent_space.max_num_edges}")
        print("\n✓ Multi-agent Dict wrapper test PASSED!")
    else:
        print(f"\n✗ Multi-agent Dict wrapper test FAILED!")
        print(f"   Expected BusConnectivityGraphObsSpace, got {type(rl_agent_space).__name__}")
        return False

    # Cleanup
    Path(temp_file).unlink()

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing BusConnectivityGraphObsSpace Checkpoint Serialization")
    print("=" * 80)

    try:
        # Test 1: Basic pickle serialization
        test1_passed = test_pickle_serialization()

        # Test 2: Multi-agent wrapper
        test2_passed = test_multi_agent_dict_wrapper()

        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"Test 1 (Pickle Serialization): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
        print(f"Test 2 (Multi-Agent Wrapper): {'✓ PASSED' if test2_passed else '✗ FAILED'}")

        if test1_passed and test2_passed:
            print("\n✓ All tests PASSED!")
            return 0
        else:
            print("\n✗ Some tests FAILED!")
            return 1

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    #fml()
    exit(main())

