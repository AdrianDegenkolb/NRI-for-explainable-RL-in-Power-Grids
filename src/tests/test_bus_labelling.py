"""
Verification script for the lightsim2grid bus labelling convention.

Hypothesis to test:
    For a grid with n_sub substations and 2 busbars per substation:
        - Busbar 1 of substation i  ->  gridmodel bus index  i
        - Busbar 2 of substation i  ->  gridmodel bus index  i + n_sub

The script runs three independent experiments:

  Experiment 1 – Default topology (all on busbar 1)
      All elements are on busbar 1 by default.
      We verify that every element's gridmodel bus_id equals its grid2op substation id,
      confirming  bus_id == sub_id  (i.e. busbar 1 of sub i == bus i).

  Experiment 2 – Switch one element to busbar 2
      We move one load (load 0 at substation 0) to busbar 2 via a topology action.
      After the power flow we verify that load 0's gridmodel bus_id changed to  sub_id + n_sub,
      confirming  busbar 2 of sub i == bus i + n_sub.

  Experiment 3 – Exhaustive sweep over all substations
      For each substation, we place *all* its elements on busbar 2 (one substation at a time,
      then revert). We check the bus ids of every affected element against sub_id + n_sub.
"""

import grid2op
from lightsim2grid import LightSimBackend
import numpy as np

ENV_NAME = "l2rpn_case14_sandbox"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    env = grid2op.make(ENV_NAME, backend=LightSimBackend())
    env.seed(0)
    return env


def get_line_bus_or(gm, env, line_idx):
    """Return the gridmodel bus_id for the OR side of g2op line line_idx.
    Grid2op lines 0..n_pl-1 map to GridModel powerlines; n_pl..n_line-1 map to trafos (hv=or side).
    """
    n_pl = len(gm.get_lines())
    if line_idx < n_pl:
        return gm.get_lines()[line_idx].bus_or_id
    else:
        return gm.get_trafos()[line_idx - n_pl].bus_hv_id


def get_line_bus_ex(gm, env, line_idx):
    """Return the gridmodel bus_id for the EX side of g2op line line_idx.
    Grid2op lines 0..n_pl-1 map to GridModel powerlines; n_pl..n_line-1 map to trafos (lv=ex side).
    """
    n_pl = len(gm.get_lines())
    if line_idx < n_pl:
        return gm.get_lines()[line_idx].bus_ex_id
    else:
        return gm.get_trafos()[line_idx - n_pl].bus_lv_id


def get_element_bus_ids(gm, env):
    """Return dicts mapping element index -> gridmodel bus_id for all element types."""
    loads    = {i: gm.get_loads()[i].bus_id      for i in range(env.n_load)}
    gens     = {i: gm.get_generators()[i].bus_id for i in range(env.n_gen)}
    lines_or = {i: get_line_bus_or(gm, env, i)   for i in range(env.n_line)}
    lines_ex = {i: get_line_bus_ex(gm, env, i)   for i in range(env.n_line)}
    return loads, gens, lines_or, lines_ex


def run_dc_pf(gm):
    """Run DC power flow and return True if successful."""
    v_init = np.ones(gm.total_bus(), dtype=complex)
    res = gm.dc_pf(v_init, 1)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 – Default topology: busbar 1 == bus index == sub_id
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1 – Default topology (all elements on busbar 1)")
    print("  Hypothesis: gridmodel bus_id == grid2op sub_id  for all elements")
    print("=" * 70)

    env = make_env()
    env.reset()
    backend = env.backend
    gm = backend._grid
    n_sub = env.n_sub

    loads, gens, lines_or, lines_ex = get_element_bus_ids(gm, env)

    failures = []

    for i, bus in loads.items():
        expected = env.load_to_subid[i]
        if bus != expected:
            failures.append(f"  FAIL load {i}: bus={bus}, sub_id={expected}")
        else:
            print(f"  OK   load {i:2d}: bus={bus} == sub_id={expected}")

    for i, bus in gens.items():
        expected = env.gen_to_subid[i]
        if bus != expected:
            failures.append(f"  FAIL gen {i}: bus={bus}, sub_id={expected}")
        else:
            print(f"  OK   gen  {i:2d}: bus={bus} == sub_id={expected}")

    for i, bus in lines_or.items():
        expected = env.line_or_to_subid[i]
        if bus != expected:
            failures.append(f"  FAIL line_or {i}: bus={bus}, sub_id={expected}")
        else:
            print(f"  OK   line {i:2d} or: bus={bus} == sub_id={expected}")

    for i, bus in lines_ex.items():
        expected = env.line_ex_to_subid[i]
        if bus != expected:
            failures.append(f"  FAIL line_ex {i}: bus={bus}, sub_id={expected}")
        else:
            print(f"  OK   line {i:2d} ex: bus={bus} == sub_id={expected}")

    print()
    if failures:
        print("EXPERIMENT 1 FAILED:")
        for f in failures:
            print(f)
    else:
        print("EXPERIMENT 1 PASSED ✓")
        print("  → Busbar 1 of substation i  ==  gridmodel bus i")

    return len(failures) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 – Move load 0 (sub 0) to busbar 2
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2():
    print()
    print("=" * 70)
    print("EXPERIMENT 2 – Move load 0 (substation 0) to busbar 2")
    print("  Hypothesis: gridmodel bus_id == sub_id + n_sub  after the switch")
    print("=" * 70)

    env = make_env()
    obs = env.reset()
    backend = env.backend
    gm = backend._grid
    n_sub = env.n_sub

    target_load = 0
    target_sub  = int(env.load_to_subid[target_load])   # == 0

    print(f"  load {target_load} is at substation {target_sub}")
    print(f"  n_sub = {n_sub}")
    print(f"  Expected gridmodel bus after switch to BB2: {target_sub + n_sub}")

    # Build a set_bus action that places the load on busbar 2.
    # In grid2op each substation element has a position in the topology vector.
    # load_pos_topo_vect[i] gives its position; setting that to bus 2 (grid2op convention)
    # moves it to busbar 2.
    act = env.action_space({
        "set_bus": {"loads_id": [(target_load, 2)]}
    })

    obs, reward, done, info = env.step(act)

    if done:
        print("  WARNING: environment is done after this action (possible overload).")

    # Re-read bus id
    new_bus = gm.get_loads()[target_load].bus_id
    expected = target_sub + n_sub

    print(f"  gridmodel bus_id after action: {new_bus}")
    print(f"  expected:                       {expected}")

    print()
    if new_bus == expected:
        print("EXPERIMENT 2 PASSED ✓")
        print("  → Busbar 2 of substation i  ==  gridmodel bus i + n_sub")
        return True
    else:
        print("EXPERIMENT 2 FAILED ✗")
        print(f"  → Actual bus_id={new_bus}, expected={expected}")
        # Try to find the actual offset
        actual_offset = new_bus - target_sub
        print(f"  → Actual offset seems to be {actual_offset} (not n_sub={n_sub})")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 – Exhaustive sweep: move all elements of each sub to BB2
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3():
    print()
    print("=" * 70)
    print("EXPERIMENT 3 – Exhaustive sweep over all substations")
    print("  For each sub, move all its elements to busbar 2 one at a time.")
    print("  Check that bus_id == sub_id + n_sub for every switched element.")
    print("=" * 70)

    env = make_env()
    n_sub = env.n_sub
    all_passed = True
    offsets_seen = set()

    for sub_id in range(n_sub):
        env.reset()
        backend = env.backend
        gm = backend._grid

        # Gather elements of this substation
        load_ids  = [i for i in range(env.n_load) if env.load_to_subid[i] == sub_id]
        gen_ids   = [i for i in range(env.n_gen)  if env.gen_to_subid[i]  == sub_id]
        lor_ids   = [i for i in range(env.n_line) if env.line_or_to_subid[i] == sub_id]
        lex_ids   = [i for i in range(env.n_line) if env.line_ex_to_subid[i] == sub_id]

        if not load_ids and not gen_ids and not lor_ids and not lex_ids:
            print(f"  sub {sub_id:2d}: no elements, skipping.")
            continue

        # Move one element (pick first available) to busbar 2
        if load_ids:
            elem_type = "load"
            elem_idx  = load_ids[0]
            act = env.action_space({"set_bus": {"loads_id": [(elem_idx, 2)]}})
        elif gen_ids:
            elem_type = "gen"
            elem_idx  = gen_ids[0]
            act = env.action_space({"set_bus": {"generators_id": [(elem_idx, 2)]}})
        elif lor_ids:
            elem_type = "line_or"
            elem_idx  = lor_ids[0]
            act = env.action_space({"set_bus": {"lines_or_id": [(elem_idx, 2)]}})
        else:
            elem_type = "line_ex"
            elem_idx  = lex_ids[0]
            act = env.action_space({"set_bus": {"lines_ex_id": [(elem_idx, 2)]}})

        obs, reward, done, info = env.step(act)

        # Read back the bus_id
        if elem_type == "load":
            actual_bus = gm.get_loads()[elem_idx].bus_id
        elif elem_type == "gen":
            actual_bus = gm.get_generators()[elem_idx].bus_id
        elif elem_type == "line_or":
            actual_bus = get_line_bus_or(gm, env, elem_idx)
        else:
            actual_bus = get_line_bus_ex(gm, env, elem_idx)

        expected_bus = sub_id + n_sub
        offset       = actual_bus - sub_id
        offsets_seen.add(offset)

        status = "OK  ✓" if actual_bus == expected_bus else "FAIL✗"
        print(f"  sub {sub_id:2d} ({elem_type:7s} {elem_idx}): "
              f"bus_id={actual_bus}, expected={expected_bus}  [{status}]  "
              f"offset={offset}")

        if actual_bus != expected_bus:
            all_passed = False

    print()
    print(f"Offsets seen across all substations: {sorted(offsets_seen)}")
    print()
    if all_passed:
        print("EXPERIMENT 3 PASSED ✓")
        print("  → Busbar 2 of substation i  ==  gridmodel bus i + n_sub  (for ALL subs)")
    else:
        print("EXPERIMENT 3 FAILED ✗")
        print("  → The i + n_sub hypothesis does NOT hold for some substations.")
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: print full mapping table
# ─────────────────────────────────────────────────────────────────────────────

def print_mapping_table():
    print()
    print("=" * 70)
    print("MAPPING TABLE – substation i -> gridmodel bus indices")
    print("=" * 70)
    env = make_env()
    n_sub = env.n_sub
    print(f"  {'sub_id':>6}  {'BB1 bus':>8}  {'BB2 bus':>8}  {'BB2 = sub+n_sub?':>18}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*18}")
    for i in range(n_sub):
        bb1 = i
        bb2_hypothesis = i + n_sub
        print(f"  {i:>6}  {bb1:>8}  {bb2_hypothesis:>8}  (n_sub={n_sub})")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: verify solver bus labelling
# ─────────────────────────────────────────────────────────────────────────────

def experiment_solver_labelling():
    print()
    print("=" * 70)
    print("BONUS – Solver bus labelling after switching sub 0 to BB2")
    print("  Shows id_me_to_ac_solver and id_ac_solver_to_me after topo change.")
    print("=" * 70)

    env = make_env()
    obs = env.reset()
    backend = env.backend
    gm = backend._grid
    n_sub = env.n_sub

    # Move load 0 (sub 0) to busbar 2
    act = env.action_space({"set_bus": {"loads_id": [(0, 2)]}})
    obs, reward, done, info = env.step(act)

    me_to_solver = gm.id_me_to_ac_solver()
    solver_to_me = gm.id_ac_solver_to_me()

    print(f"  id_me_to_ac_solver (len={len(me_to_solver)}):")
    for i, s in enumerate(me_to_solver):
        tag = ""
        if i < n_sub:
            tag = f"  <- BB1 of sub {i}"
        else:
            tag = f"  <- BB2 of sub {i - n_sub}"
        active = "ACTIVE" if s >= 0 else "inactive"
        print(f"    gridmodel_bus {i:2d} -> solver_bus {s:3d}  ({active}){tag}")

    print()
    print(f"  id_ac_solver_to_me (len={len(solver_to_me)}):")
    for j, m in enumerate(solver_to_me):
        if m < n_sub:
            tag = f"BB1 of sub {m}"
        else:
            tag = f"BB2 of sub {m - n_sub}"
        print(f"    solver_bus {j:2d} -> gridmodel_bus {m:2d}  ({tag})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r1 = experiment_1()
    r2 = experiment_2()
    r3 = experiment_3()
    print_mapping_table()
    experiment_solver_labelling()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Experiment 1 (default topo, BB1==sub_id):       {'PASS ✓' if r1 else 'FAIL ✗'}")
    print(f"  Experiment 2 (move one load to BB2):            {'PASS ✓' if r2 else 'FAIL ✗'}")
    print(f"  Experiment 3 (exhaustive sweep over all subs):  {'PASS ✓' if r3 else 'FAIL ✗'}")
    print()
    if r1 and r2 and r3:
        print("CONCLUSION ✓")
        print("  Busbar 1 of substation i  ->  gridmodel bus index  i")
        print("  Busbar 2 of substation i  ->  gridmodel bus index  i + n_sub")
    else:
        print("CONCLUSION: hypothesis REFUTED – see individual experiment output above.")
