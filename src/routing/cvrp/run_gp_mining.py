import os
import sys
import json
import copy
import random
import glob
import pickle
import numpy as np
import traceback
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass

# ==============================================================================
# SETUP
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, '..', '..'))
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.utils import optimize_all_start_times, cleanup_inter_factory_routes
from routing.cvrp.alns_cvrp.destroy_operators import (
    random_removal, worst_removal_alpha_0, worst_removal_bigM,
    worst_removal_adaptive, time_worst_removal, shaw_spatial,
    shaw_hybrid, shaw_temporal, shaw_structural, trip_removal,
    historical_removal, update_solution_state_after_destroy
)
from routing.cvrp.alns_cvrp.repair_operators import (
    best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle,
    regret_3_position, regret_3_trip, regret_3_vehicle,
    regret_4_position, regret_4_trip, regret_4_vehicle
)
# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\AnKhoa\Project_Code\input_data_train_gp'
OUTPUT_FILE = 'discovered_macros_strategic.json'
NUM_TRAIN_INSTANCES = 50
NUM_WORKERS = 45
# GP Parameters
MAX_SEQ_LEN = 5 #5
POP_SIZE = 200 #200
N_GEN = 40 #40
CX_PROB = 0.8
MUT_PROB = 0.2
# Fitness penalties and bonuses (COMPLETE)
INFEASIBLE_DESTROY_PENALTY = 0.4 # Only when: D at end + infeasible
REPAIR_END_BONUS = 0.15 # Reward R at end (consolidation)
REPAIR_BEFORE_DESTROY_BONUS = 0.05 # Reward R→D pattern (preparation)
# Delayed Credit Configuration
DELAYED_CREDIT_ALPHA = 0.5 # Refund coefficient (0.2 = 20% of abs(I_raw))
# Operator Configurations
DESTROY_OPS = [
    random_removal, worst_removal_alpha_0, worst_removal_bigM,
    worst_removal_adaptive, time_worst_removal, shaw_spatial,
    shaw_hybrid, shaw_temporal, shaw_structural, trip_removal,
    historical_removal
]
# FIXED: Unified repair operators (NO soft/strong split)
REPAIR_OPS = [
    best_insertion,
    regret_2_position, regret_2_trip, regret_2_vehicle,
    regret_3_position, regret_3_trip, regret_3_vehicle,
    regret_4_position, regret_4_trip, regret_4_vehicle
]
# Removal fractions per destroy operator (operator personality)
DESTROY_CONFIG = {
    'random_removal': 0.18,
    'worst_removal_alpha_0': 0.08,
    'worst_removal_bigM': 0.08,
    'worst_removal_adaptive': 0.10,
    'time_worst_removal': 0.08,
    'shaw_spatial': 0.12,
    'shaw_hybrid': 0.14,
    'shaw_temporal': 0.12,
    'shaw_structural': 0.12,
    'trip_removal': 0.10,
    'historical_removal': 0.15,
}
def get_removal_fraction(destroy_op):
    """Get removal fraction for a destroy operator"""
    op_name = destroy_op.__name__ if hasattr(destroy_op, '__name__') else str(destroy_op)
    return DESTROY_CONFIG.get(op_name, 0.12)
def get_op_name(op):
    """Get operator name for logging"""
    if hasattr(op, '__name__'):
        return op.__name__
    if hasattr(op, 'func'):
        base_name = op.func.__name__
        if base_name == 'regret_k_insertion':
            return f"regret_{op.keywords.get('k_regret', '?')}"
        return base_name
    return str(op)
# ==============================================================================
# HELPER CLASSES
# ==============================================================================
class DotDict(dict):
    """Dictionary with dot notation access"""
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__.update(d)
@dataclass
class MacroRecord:
    """Record of macro execution for delayed credit tracking"""
    macro_id: int
    genome: list
    I_raw: float
    E_raw: float
    feasible: bool
    cost: float = None
    caused_infeasible: bool = False
    refunded: bool = False # Prevent double refund
def debug_solution_structure(solution, name="solution"):
    """
    Debug helper to inspect solution object structure
    """
    print(f"\n[DEBUG] Inspecting {name}:")
    print(f" Type: {type(solution)}")
   
    if hasattr(solution, '__dict__'):
        print(f" Attributes: {list(solution.__dict__.keys())}")
   
    # Try common attribute names
    for attr in ['routes', 'solution', 'route_list', 'tours', '_routes', '_solution']:
        if hasattr(solution, attr):
            val = getattr(solution, attr)
            print(f" .{attr}: type={type(val)}, len={len(val) if hasattr(val, '__len__') else 'N/A'}")
   
    # Check if iterable
    try:
        list(solution)
        print(f" Is iterable: YES")
    except TypeError:
        print(f" Is iterable: NO")
def load_data_from_pickle(file_path):
    """Load problem instance from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        instance = DotDict(data_dict)
        if 'farms' in instance:
            instance.farms = [DotDict(f) for f in instance.farms]
        if 'facilities' in instance:
            instance.facilities = [DotDict(f) for f in instance.facilities]
        if 'demands' in instance and 'demand' not in instance:
            instance.demand = instance.demands
        if hasattr(instance, 'farms'):
            for f in instance.farms:
                if 'service_time_params' in f:
                    f.service_time_params = [int(x) for x in f.service_time_params]
        return instance
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None
# ==============================================================================
# CUSTOMER COUNTING (FIXED BUG)
# ==============================================================================
def count_customers(solution):
    """
    Count number of unique farm customers in solution.
    Exclude TRANSFER nodes.
   
    FIXED: cvrpEnv uses .schedule attribute to store routes
   
    Returns: int
    """
    customers = set()
    routes = None
   
    # CRITICAL: cvrpEnv stores routes in .schedule (7-tuple format)
    if hasattr(solution, 'schedule'):
        routes = solution.schedule
    elif hasattr(solution, 'routes'):
        routes = solution.routes
    elif hasattr(solution, 'solution'):
        routes = solution.solution
    elif hasattr(solution, 'route_list'):
        routes = solution.route_list
    elif isinstance(solution, dict):
        if 'schedule' in solution:
            routes = solution['schedule']
        elif 'solution' in solution:
            routes = solution['solution']
        elif 'routes' in solution:
            routes = solution['routes']
   
    # If still None, try direct iteration
    if routes is None:
        try:
            routes = list(solution)
        except TypeError:
            if hasattr(solution, '__dict__'):
                sol_dict = solution.__dict__
                for key in ['schedule', 'routes', 'solution', 'route_list', '_schedule']:
                    if key in sol_dict:
                        routes = sol_dict[key]
                        break
   
    # If still None, raise helpful error
    if routes is None:
        available_attrs = dir(solution) if hasattr(solution, '__dict__') else type(solution)
        raise AttributeError(
            f"Cannot find routes in solution object. "
            f"Available attributes: {available_attrs}"
        )
   
    # Count customers from 7-tuple format
    # Format: (depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load)
    for route_info in routes:
        try:
            # Try to unpack 7-tuple
            if len(route_info) == 7:
                depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load = route_info
            elif len(route_info) >= 3:
                # Fallback: assume customer_list is at index 2
                customer_list = route_info[2]
            else:
                # Fallback: treat route_info as customer list directly
                customer_list = route_info
           
            # Count customers in this route
            for node in customer_list:
                node_id = str(node)
                if not node_id.startswith('TRANSFER_'):
                    customers.add(node_id)
        except (ValueError, TypeError, IndexError):
            # If unpacking fails, try to iterate route_info directly
            try:
                for node in route_info:
                    node_id = str(node)
                    if not node_id.startswith('TRANSFER_'):
                        customers.add(node_id)
            except:
                continue
   
    return len(customers)
# ==============================================================================
# FEASIBILITY & VIOLATION TRACKING (COMPLETE)
# ==============================================================================
def get_solution_violations(solution):
    """
    Extract ALL violation metrics from solution.
   
    Constraints:
    - Capacity: HARD (must be <= 1e-6 for feasible)
    - Time Window: SOFT (reward reduction)
    - Waiting Time: SOFT (reward reduction)
    - Multi-trip: COST-BASED (penalty for route cloning/duplicates)
    - Unassigned: HARD-ISH (must be 0 ideally)
   
    Returns:
        unassigned_count: Number of unassigned customers
        cap_pen: Capacity violation penalty (hard)
        tw_pen: Time window violation penalty (soft)
        waiting_pen: Waiting time penalty (soft)
        multi_trip_pen: Multi-trip/cloning penalty (cost-based)
        is_feasible: True if cap_pen <= 1e-6 (hard constraint met)
    """
    try:
        # Get objective components
        # From your objective: returns (total_cost, total_penalty_cost, total_waiting_cost, total_capacity_penalty)
        # Assume total_penalty_cost is TW penalty (time_penalty * 0.3)
        # Multi-trip not returned - compute separately
        total_cost, tw_pen, waiting_pen, cap_pen = solution.objective()
        multi_trip_pen = compute_multi_trip_penalty(solution)
       
        # Count unassigned customers
        unassigned = 0
        if hasattr(solution, 'schedule') and hasattr(solution, 'num_farms'):
            total_customers = solution.num_farms
            assigned = count_customers(solution)
            unassigned = max(0, total_customers - assigned)
       
        # Hard constraint: only capacity
        is_feasible = (cap_pen <= 1e-6)
       
        return unassigned, cap_pen, tw_pen, waiting_pen, multi_trip_pen, is_feasible
       
    except Exception as e:
        print(f"[WARN] get_solution_violations failed: {e}")
        return 0, 0.0, 0.0, 0.0, 0.0, True
def compute_multi_trip_penalty(solution):
    """
    Compute multi-trip penalty (route cloning/duplicates).
   
    Penalty for:
    - Same truck_id appearing multiple times in same shift
    - Overlapping time windows for same truck
   
    Returns: float (penalty value)
    """
    try:
        if not hasattr(solution, 'schedule') or not solution.schedule:
            return 0.0
       
        # Track truck usage per shift
        truck_usage = {} # (truck_id, shift) -> list of (start, end)
       
        for route_info in solution.schedule:
            if len(route_info) >= 7:
                _, truck_id, customer_list, shift, start_time_at_depot, finish_time_route, _ = route_info
               
                key = (truck_id, shift)
                if key not in truck_usage:
                    truck_usage[key] = []
                truck_usage[key].append((start_time_at_depot, finish_time_route))
       
        # Compute penalty
        penalty = 0.0
        CLONE_PENALTY_WEIGHT = 999999 # From your objective
       
        for key, time_windows in truck_usage.items():
            if len(time_windows) <= 1:
                continue
           
            # Sort by start time
            sorted_windows = sorted(time_windows, key=lambda x: x[0])
           
            # Check for overlaps
            for i in range(len(sorted_windows) - 1):
                current_end = sorted_windows[i][1]
                next_start = sorted_windows[i+1][0]
               
                if next_start < (current_end - 1e-6):
                    penalty += CLONE_PENALTY_WEIGHT
       
        return penalty
       
    except Exception as e:
        return 0.0
def compute_route_variance(solution):
    """
    Compute variance of route loads (structural stability metric).
    Lower variance = more balanced = more stable structure.
   
    Returns: float (variance of route loads)
    """
    try:
        if not hasattr(solution, 'schedule') or not solution.schedule:
            return 0.0
       
        loads = []
        for route_info in solution.schedule:
            if len(route_info) >= 7:
                route_load = route_info[6] # Index 6 is route_load
                loads.append(route_load)
       
        if len(loads) < 2:
            return 0.0
       
        return float(np.var(loads))
    except Exception as e:
        return 0.0
def sanitize_solution(solution, unassigned_pool, rnd_state):
    """
    Best-effort repair for remaining unassigned customers.
    NOT guaranteed to succeed (30% feasible is OK).
    Only fixes capacity violations (hard constraint).
   
    Returns: (solution, remaining_unassigned)
    """
    if not unassigned_pool:
        return solution, []
   
    remaining = list(unassigned_pool)
   
    try:
        # Attempt 1: regret_4 (strong)
        repaired, failed = regret_4_position(solution, rnd_state, unvisited_customers=remaining)
        if not failed:
            return repaired, []
        remaining = failed
       
        # Attempt 2: regret_3
        repaired, failed = regret_3_position(repaired, rnd_state, unvisited_customers=remaining)
        if not failed:
            return repaired, []
        remaining = failed
       
        # Attempt 3: best_insertion (last resort)
        repaired, failed = best_insertion(repaired, rnd_state, unvisited_customers=remaining)
        return repaired, failed if failed else []
       
    except Exception as e:
        print(f"[WARN] Sanitize failed: {e}")
        return solution, remaining
# ==============================================================================
# STRUCTURAL SIMILARITY (NO COST)
# ==============================================================================
def solution_similarity(sol_a, sol_b):
    """
    Compute structural similarity between two solutions.
    Uses Jaccard similarity on customer → route assignments.
    Returns value in [0, 1] where 1 = identical structure.
   
    FIXED: cvrpEnv uses .schedule with 7-tuple format
    """
    try:
        # FIXED: Get routes from .schedule (7-tuple format)
        def get_routes(sol):
            if hasattr(sol, 'schedule'):
                return sol.schedule
            elif hasattr(sol, 'routes'):
                return sol.routes
            elif hasattr(sol, 'solution'):
                return sol.solution
            elif isinstance(sol, dict):
                if 'schedule' in sol:
                    return sol['schedule']
                elif 'solution' in sol:
                    return sol['solution']
                elif 'routes' in sol:
                    return sol['routes']
           
            # Try direct iteration
            try:
                return list(sol)
            except TypeError:
                if hasattr(sol, '__dict__'):
                    for key in ['schedule', 'routes', 'solution', '_schedule']:
                        if key in sol.__dict__:
                            return sol.__dict__[key]
           
            return []
       
        routes_a = get_routes(sol_a)
        routes_b = get_routes(sol_b)
       
        assign_a = {}
        for route_idx, route_info in enumerate(routes_a):
            try:
                if len(route_info) == 7:
                    customer_list = route_info[2]
                elif len(route_info) >= 3:
                    customer_list = route_info[2]
                else:
                    customer_list = route_info
               
                for customer in customer_list:
                    customer_id = str(customer)
                    if not customer_id.startswith('TRANSFER_'):
                        assign_a[customer_id] = route_idx
            except (ValueError, TypeError, IndexError):
                try:
                    for customer in route_info:
                        customer_id = str(customer)
                        if not customer_id.startswith('TRANSFER_'):
                            assign_a[customer_id] = route_idx
                except:
                    continue
       
        assign_b = {}
        for route_idx, route_info in enumerate(routes_b):
            try:
                if len(route_info) == 7:
                    customer_list = route_info[2]
                elif len(route_info) >= 3:
                    customer_list = route_info[2]
                else:
                    customer_list = route_info
               
                for customer in customer_list:
                    customer_id = str(customer)
                    if not customer_id.startswith('TRANSFER_'):
                        assign_b[customer_id] = route_idx
            except (ValueError, TypeError, IndexError):
                try:
                    for customer in route_info:
                        customer_id = str(customer)
                        if not customer_id.startswith('TRANSFER_'):
                            assign_b[customer_id] = route_idx
                except:
                    continue
       
        if not assign_a or not assign_b:
            return 0.0
       
        same = sum(1 for cid in assign_a if assign_a[cid] == assign_b.get(cid, -1))
        return same / (len(assign_a) + 1e-6)
       
    except Exception as e:
        print(f"[WARN] solution_similarity failed: {e}")
        return 0.0
def solution_structural_distance(sol_a, sol_b):
    """
    Return structural distance between two solutions in [0, 1]
    """
    return 1.0 - solution_similarity(sol_a, sol_b)
def compute_exploration_score_structural(solutions):
    """
    E = average_pairwise_structural_distance(solutions)
    """
    if len(solutions) < 2:
        return 0.0
    distances = []
    for i in range(len(solutions)):
        for j in range(i+1, len(solutions)):
            d = solution_structural_distance(solutions[i], solutions[j])
            distances.append(d)
    E = np.mean(distances)
   
    # STEP 4: Diminishing return for low distance (similar structures)
    SMALL_THRESHOLD = 0.1
    if E < SMALL_THRESHOLD:
        E *= 0.7
   
    return float(E)
def snapshot_routes(solution):
    """
    Light snapshot: route structure only (no times)
    """
    return [
        tuple(route.customers)
        for route in solution.routes
        if route.customers
    ]
def compute_structural_delta(before, after):
    """
    Measure how much route structure changed
    Output ∈ [0, 1]
    """
    before_set = set(before)
    after_set = set(after)
    if not before_set:
        return 0.0
    diff = before_set.symmetric_difference(after_set)
    return len(diff) / max(1, len(before_set))
# ==============================================================================
# MACRO EXECUTION (SIMPLIFIED - NO SOFT/STRONG)
# ==============================================================================
def execute_macro_token_based(macro, solution, rnd_state, history_matrix):
    initial_customer_count = count_customers(solution)
    solutions = [copy.deepcopy(solution)]
    pool_history = [0]
    current = solution
    unassigned_pool = []
    # 🔥 NEW: collect destroy structural signals
    structural_deltas = []
    for token_idx, (token_type, op_idx) in enumerate(macro):
        try:
            if token_type == 'D':
                destroy_op = DESTROY_OPS[op_idx]
                remove_fraction = get_removal_fraction(destroy_op)
                # === BEFORE DESTROY SNAPSHOT ===
                snapshot_before = snapshot_routes(current)
                op_kwargs = {
                    'remove_fraction': remove_fraction,
                    'history_matrix': history_matrix
                }
                destroyed, unassigned = destroy_op(current, rnd_state, **op_kwargs)
                destroyed = update_solution_state_after_destroy(destroyed)
                # === AFTER DESTROY SNAPSHOT ===
                snapshot_after = snapshot_routes(destroyed)
                # 🔥 Structural delta
                delta_struct = compute_structural_delta(
                    snapshot_before,
                    snapshot_after
                )
                structural_deltas.append(delta_struct)
                farms = [c for c in unassigned if not str(c).startswith('TRANSFER_')]
                unassigned_pool.extend(farms)
                current = destroyed
            elif token_type == 'R':
                if unassigned_pool:
                    repair_op = REPAIR_OPS[op_idx]
                    repaired, failed = repair_op(
                        current,
                        rnd_state,
                        unvisited_customers=unassigned_pool
                    )
                    current = repaired
                    unassigned_pool = failed if failed else []
            solutions.append(copy.deepcopy(current))
            pool_history.append(len(unassigned_pool))
        except Exception as e:
            print(f"[WARN] Error at token {token_idx} ({token_type}-{op_idx}): {e}")
            solutions.append(copy.deepcopy(current))
            pool_history.append(len(unassigned_pool))
    # Cleanup & sanitize
    current = cleanup_inter_factory_routes(current)
    current = optimize_all_start_times(current)
    cost_before_sanitize = current.objective()[0]
    if unassigned_pool:
        current, remaining_unassigned = sanitize_solution(
            current, unassigned_pool, rnd_state
        )
        unassigned_pool = remaining_unassigned
    current = cleanup_inter_factory_routes(current)
    current = optimize_all_start_times(current)
    solutions[-1] = current
    pool_history[-1] = len(unassigned_pool)
    final_customer_count = count_customers(current)
    customers_lost = initial_customer_count - final_customer_count
    if customers_lost > 0:
        print(f"⚠️ Macro lost {customers_lost} customers! "
              f"({final_customer_count}/{initial_customer_count})")
    # 🔥 NEW: aggregate signal
    avg_structural_delta = (
        float(np.mean(structural_deltas))
        if structural_deltas else 0.0
    )
    return (
        current,
        solutions,
        pool_history,
        cost_before_sanitize,
        customers_lost,
        avg_structural_delta
    )
# ==============================================================================
# FITNESS EVALUATION
# ==============================================================================
def evaluate_macro_on_scenario(macro, scenario):
    """
    Evaluate one macro on one scenario.
    ALWAYS returns:
        (I, E, metadata)
    """
    try:
        solution = scenario['solution']
        initial_cost = scenario['cost']
        initial_customer_count = count_customers(solution)
        rnd_state = scenario['rnd_state']
        history_matrix = scenario['history_matrix']
        best_cost_before = scenario.get('best_cost_before', initial_cost)
        # Execute macro
        (final_solution, solutions, pool_history,
         cost_before_sanitize, customers_lost,structural_delta) = execute_macro_token_based(
            macro, solution, rnd_state, history_matrix
        )
        # ================================================================
        # Case 1: Customer loss → hard fail
        # ================================================================
        if customers_lost > 0:
            missing_ratio = customers_lost / max(1, initial_customer_count)
            I = -1.0 - missing_ratio
            E = 0.0
            metadata = {
                'I_raw': I,
                'E_raw': E,
                'feasible': False,
                'cost': None,
                'caused_infeasible': True,
                'best_cost_before': best_cost_before
            }
            return (I, E, metadata)
        # ================================================================
        # Compute violations BEFORE / AFTER
        # ================================================================
        unassigned_before, cap_before, tw_before, waiting_before, multi_trip_before, _ = get_solution_violations(solution)
        route_var_before = compute_route_variance(solution)
        N_customers = initial_customer_count
        unassigned_after, cap_after, tw_after, waiting_after, multi_trip_after, is_feasible = get_solution_violations(final_solution)
        route_var_after = compute_route_variance(final_solution)
        # ================================================================
        # Δ computation
        # ================================================================
        delta_unassigned = unassigned_before - unassigned_after
        delta_cap = cap_before - cap_after
        delta_tw = tw_before - tw_after
        delta_waiting = waiting_before - waiting_after
        delta_multi_trip = multi_trip_before - multi_trip_after
        delta_route_var = route_var_before - route_var_after
        def relative_tanh(before, delta):
            if before > 0:
                return np.tanh(delta / before)
            return 0.0 if delta >= 0 else np.tanh(delta / 1e-6)
        delta_cap = relative_tanh(cap_before, delta_cap)
        delta_multi_trip = relative_tanh(multi_trip_before, delta_multi_trip)
        delta_tw = relative_tanh(tw_before, delta_tw)
        delta_waiting = relative_tanh(waiting_before, delta_waiting)
        delta_route_var = relative_tanh(route_var_before, delta_route_var)
       
        # ================================================================
        # I components
        # ================================================================
        I_feas = (
            0.3 * (delta_unassigned / max(1, N_customers))
            + 0.3 * delta_cap
            + 0.1 * delta_tw
            + 0.1 * delta_waiting
            + 0.05 * delta_multi_trip
        )
        # STEP 5: Adjusted I_stab with relative_tanh (light weight α=0.1)
        I_stab = 0.1 * max(0, delta_route_var)
        # Repair placement bonus
        repair_bonus = 0.0
        if macro and macro[-1][0] == 'R':
            repair_bonus += REPAIR_END_BONUS
        for i in range(len(macro) - 1):
            if macro[i][0] == 'R' and macro[i + 1][0] == 'D':
                repair_bonus += REPAIR_BEFORE_DESTROY_BONUS
        I_raw = I_feas + I_stab + repair_bonus
        if macro and macro[-1][0] == 'D' and not is_feasible:
            if structural_delta < 0.3:
                I_raw -= INFEASIBLE_DESTROY_PENALTY
            else:
                I_raw -= 0.15
        # ================================================================
        # Exploration
        # ================================================================
        E = compute_exploration_score_structural(solutions)
        final_cost = final_solution.objective()[0] if is_feasible else None
        # STEP 7: Clip scales
        I_final = np.clip(I_raw, -2.0, 1.0)
        E_final = np.clip(E, 0.0, 1.0)
        metadata = {
            'I_raw': I_raw,
            'E_raw': E,
            'feasible': is_feasible,
            'cost': final_cost,
            'caused_infeasible': not is_feasible,
            'best_cost_before': best_cost_before
        }
        return (I_final, E_final, metadata)
    except Exception as e:
        print(f"[ERROR] evaluate_macro_on_scenario: {e}")
        traceback.print_exc()
        I = -2.0
        E = 0.0
        metadata = {
            'I_raw': I,
            'E_raw': E,
            'feasible': False,
            'cost': None,
            'caused_infeasible': True,
            'best_cost_before': None
        }
        return (I, E, metadata)
# ==============================================================================
# WORKER SETUP (MULTIPROCESSING)
# ==============================================================================
WORKER_SCENARIOS = []
def worker_initializer(file_paths):
    """Initialize worker with scenarios in multiple contexts"""
    global WORKER_SCENARIOS
    WORKER_SCENARIOS = []
   
    # DEBUG: Only for first file to inspect structure
    DEBUG_MODE = False
   
    for idx, f_path in enumerate(file_paths):
        try:
            problem_obj = load_data_from_pickle(f_path)
            if not problem_obj:
                continue
           
            seed = 42 + idx
            rnd = np.random.RandomState(seed)
           
            init_routes = compute_initial_solution(problem_obj, rnd)
            init_routes = [r for r in init_routes if len(r) > 0]
            if not init_routes:
                continue
           
            sol_initial = cvrpEnv(init_routes, problem_obj, seed=seed)
           
            # DEBUG: Inspect structure for first solution only
            if idx == 0 and DEBUG_MODE:
                debug_solution_structure(sol_initial, "sol_initial")
           
            sol_initial = cleanup_inter_factory_routes(sol_initial)
            sol_initial = optimize_all_start_times(sol_initial)
            cost_initial = sol_initial.objective()[0]
           
            # Context 1: Initial (25%)
            WORKER_SCENARIOS.append({
                'solution': copy.deepcopy(sol_initial),
                'cost': cost_initial,
                'rnd_state': np.random.RandomState(seed),
                'history_matrix': {},
                'context': 'initial'
            })
           
            # Context 2: Mid-search (50%)
            for warmup_iters in [100, 150]:
                sol_mid = copy.deepcopy(sol_initial)
                for _ in range(warmup_iters):
                    destroyed, unassigned = random_removal(sol_mid, rnd, remove_fraction=0.15, history_matrix={})
                    if unassigned:
                        farms = [c for c in unassigned if not str(c).startswith('TRANSFER_')]
                        if farms:
                            sol_mid, _ = best_insertion(destroyed, rnd, unvisited_customers=farms)
               
                sol_mid = optimize_all_start_times(sol_mid)
                cost_mid = sol_mid.objective()[0]
               
                WORKER_SCENARIOS.append({
                    'solution': copy.deepcopy(sol_mid),
                    'cost': cost_mid,
                    'rnd_state': np.random.RandomState(seed + warmup_iters),
                    'history_matrix': {},
                    'context': 'mid_search'
                })
           
            # Context 3: Stagnated (25%)
            sol_stuck = copy.deepcopy(sol_initial)
            for _ in range(500):
                destroyed, unassigned = random_removal(sol_stuck, rnd, remove_fraction=0.15, history_matrix={})
                if unassigned:
                    farms = [c for c in unassigned if not str(c).startswith('TRANSFER_')]
                    if farms:
                        sol_stuck, _ = best_insertion(destroyed, rnd, unvisited_customers=farms)
           
            sol_stuck = optimize_all_start_times(sol_stuck)
            cost_stuck = sol_stuck.objective()[0]
           
            WORKER_SCENARIOS.append({
                'solution': copy.deepcopy(sol_stuck),
                'cost': cost_stuck,
                'rnd_state': np.random.RandomState(seed + 500),
                'history_matrix': {},
                'context': 'stagnated'
            })
           
        except Exception as e:
            print(f"[INIT-ERROR] {f_path}: {e}")
            continue
   
    print(f"✅ Worker {os.getpid()} loaded {len(WORKER_SCENARIOS)} scenarios")
def evaluate_macro_worker(macro):
    """
    Worker function to evaluate macro on all scenarios.
   
    WITH DELAYED CREDIT:
    - Tracks last 2 macros per scenario
    - Applies refund if current macro recovers from previous infeasible
    """
    global WORKER_SCENARIOS
   
    if not WORKER_SCENARIOS:
        return (-1e9, -1e9)
   
    I_scores = []
    E_scores = []
   
    try:
        # Initialize macro_history buffer per scenario
        if not hasattr(evaluate_macro_worker, 'scenario_histories'):
            evaluate_macro_worker.scenario_histories = {}
       
        for scenario_idx, scenario in enumerate(WORKER_SCENARIOS):
            # Get or create history buffer for this scenario
            if scenario_idx not in evaluate_macro_worker.scenario_histories:
                evaluate_macro_worker.scenario_histories[scenario_idx] = {
                    'macro_history': deque(maxlen=2),
                    'best_cost': scenario['cost']
                }
           
            history_data = evaluate_macro_worker.scenario_histories[scenario_idx]
            macro_history = history_data['macro_history']
            best_cost = history_data['best_cost']
           
            # Update scenario with best_cost_before
            scenario['best_cost_before'] = best_cost
           
            # Evaluate current macro
            I_raw, E_raw, metadata = evaluate_macro_on_scenario(macro, scenario)
           
            # Create macro record
            record = MacroRecord(
                macro_id=len(macro_history),
                genome=list(macro),
                I_raw=I_raw,
                E_raw=E_raw,
                feasible=metadata['feasible'],
                cost=metadata['cost'],
                caused_infeasible=metadata['caused_infeasible'],
                refunded=False
            )
           
            # ================================================================
            # DELAYED CREDIT LOGIC
            # ================================================================
            I_final = I_raw # Start with raw I
           
            if len(macro_history) > 0:
                prev_record = macro_history[-1]
               
                # Check refund conditions
                if (prev_record.caused_infeasible and
                    not prev_record.refunded and
                    record.feasible and
                    record.cost is not None and
                    record.cost < best_cost):
                   
                    # Compute refund
                    refund = DELAYED_CREDIT_ALPHA * abs(prev_record.I_raw)
                   
                    # Apply refund to previous macro's I
                    prev_record.I_raw += refund
                    prev_record.refunded = True
                   
                    # NOTE: We can't retroactively change prev_record's fitness
                    # in the GP population, but we track it for logging
                    # The CURRENT macro benefits indirectly by not being penalized
                    # for the previous macro's setup work
                   
                    print(f"[DELAYED CREDIT] Scenario {scenario_idx}: "
                          f"Refund {refund:.3f} to previous macro "
                          f"(cost improved: {best_cost:.1f} → {record.cost:.1f})")
           
            # Update best cost if current macro is feasible and better
            if record.feasible and record.cost is not None:
                if record.cost < best_cost:
                    history_data['best_cost'] = record.cost
           
            # Append current record to history
            macro_history.append(record)
           
            # Store final scores
            I_scores.append(I_final)
            E_scores.append(E_raw)
       
        return (
            float(np.mean(I_scores)),
            float(np.mean(E_scores))
        )
       
    except Exception as e:
        print(f"[EVAL-ERROR] {e}")
        traceback.print_exc()
        return (-1e9, -1e9)
# ==============================================================================
# TOKEN GENERATION (SIMPLIFIED - NO SOFT/STRONG)
# ==============================================================================
def random_token_free():
    """
    NO PRIOR - GP is free to learn D/R ratios.
    SIMPLIFIED: Only 'D' and 'R' types
    """
    token_type = random.choice(['D', 'R'])
   
    if token_type == 'D':
        return ('D', random.randrange(len(DESTROY_OPS)))
    else:
        return ('R', random.randrange(len(REPAIR_OPS)))
def init_macro():
    """Initialize a random macro with variable length [2, MAX_SEQ_LEN]"""
    length = random.randint(2, MAX_SEQ_LEN)
    tokens = [random_token_free() for _ in range(length)]
    return creator.Individual(sanitize_sequence_enhanced(tokens))
def sanitize_sequence_enhanced(ind):
    """
    Enhanced sequence sanitization
    - Ensure has D and R
    - Ensure first token is D (macros start with destroy)
    - Maintain coherence
    """
    if len(ind) == 0:
        return [('D', 0), ('R', 0)]
   
    # Check token types
    has_D = any(t[0] == 'D' for t in ind)
    has_R = any(t[0] == 'R' for t in ind)
   
    # Ensure has both D and R
    if not has_D:
        ind[0] = ('D', random.randrange(len(DESTROY_OPS)))
    if not has_R:
        ind[-1] = ('R', random.randrange(len(REPAIR_OPS)))
   
    # Ensure first token is D
    if ind[0][0] != 'D':
        # Find first D and swap
        for i, (token_type, _) in enumerate(ind):
            if token_type == 'D':
                ind[0], ind[i] = ind[i], ind[0]
                break
        else:
            # No D found, force first to be D
            ind[0] = ('D', random.randrange(len(DESTROY_OPS)))
   
    return ind
def mutate_macro(ind):
    """Mutation operator - GP learns freely"""
    action = random.choice(['add', 'remove', 'change'])
   
    if action == 'add' and len(ind) < MAX_SEQ_LEN:
        pos = random.randint(0, len(ind))
        ind.insert(pos, random_token_free())
       
    elif action == 'remove' and len(ind) > 2:
        pos = random.randrange(len(ind))
        ind.pop(pos)
       
    elif action == 'change':
        pos = random.randrange(len(ind))
        if random.random() < 0.5:
            # Change operator (keep type)
            token_type = ind[pos][0]
            if token_type == 'D':
                ind[pos] = ('D', random.randrange(len(DESTROY_OPS)))
            else:
                ind[pos] = ('R', random.randrange(len(REPAIR_OPS)))
        else:
            # Flip type - NO PRIOR
            ind[pos] = random_token_free()
   
    ind[:] = sanitize_sequence_enhanced(ind)
    return (ind,)
# ==============================================================================
# STRUCTURAL NOVELTY FUNCTIONS (Thêm mới)
# ==============================================================================
def macro_distance(m1, m2):
    """
    Compute structural distance between two macros.
    - Uses Levenshtein (edit) distance on token strings (e.g., 'D0', 'R1')
    - Normalizes by max sequence length
    - Returns value in [0, 1]
    """
    def macro_to_tokens(m):
        return [f"{t}{i}" for t, i in m]
    
    a = macro_to_tokens(m1)
    b = macro_to_tokens(m2)
    
    m_len, n_len = len(a), len(b)
    if m_len == 0 and n_len == 0:
        return 0.0
    if m_len == 0 or n_len == 0:
        return 1.0
    
    # Levenshtein DP
    dp = np.zeros((m_len + 1, n_len + 1))
    for i in range(m_len + 1):
        dp[i][0] = i
    for j in range(n_len + 1):
        dp[0][j] = j
    for i in range(1, m_len + 1):
        for j in range(1, n_len + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    
    dist = dp[m_len][n_len]
    max_len = max(m_len, n_len)
    return dist / max_len

def compute_novelty(macro, archive, k=5):
    """
    Compute novelty score as average distance to k nearest neighbors in archive.
    - If archive small, average over all
    - Returns 1.0 if archive empty (max novelty)
    - Normalized in [0, 1]
    """
    if not archive:
        return 1.0
    
    distances = [macro_distance(macro, arch) for arch in archive]
    distances.sort()
    nearest = distances[:k] if len(distances) >= k else distances
    return float(np.mean(nearest))
# ==============================================================================
# CONFIGURATION (Thêm λ)
# ==============================================================================
LAMBDA_NOVELTY = 0.2  # Novelty weight
# ==============================================================================
# DEAP SETUP
# ==============================================================================
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, LAMBDA_NOVELTY, 0.2))  # I, E, Novelty, -len
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", init_macro)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_macro)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate_macro_worker)
# ==============================================================================
# MAIN GP LOOP
# ==============================================================================
def run_gp_mining():
    """Main GP loop with multiprocessing"""
    print("=" * 80)
    print("🚀 GP MACRO DISCOVERY - PROPER INTENSIFICATION MODE")
    print("=" * 80)
    print(f"📂 Data folder: {DATA_DIR}")
    print(f"⚡ Workers: {NUM_WORKERS}")
    print(f"🧬 Population: {POP_SIZE}, Generations: {N_GEN}")
    print(f"📏 Max sequence length: {MAX_SEQ_LEN}")
    print(f"🎯 Multi-objective: (I, E, Novelty, -len)")
    print(f"\n🔧 INTENSIFICATION (I) Components:")
    print(f" • Feasibility (85% total):")
    print(f" - Unassigned: 30% (hard-ish, normalized)")
    print(f" - Capacity: 30% (HARD constraint)")
    print(f" - Time Window: 10% (soft)")
    print(f" - Waiting Time: 10% (soft)")
    print(f" - Multi-trip: 5% (cost-based)")
    print(f" • Stability: 20% (route variance reduction)")
    print(f" • Repair Bonus: R at end (+{REPAIR_END_BONUS}), R→D (+{REPAIR_BEFORE_DESTROY_BONUS})")
    print(f" • Terminal D Penalty: D + infeasible (-{INFEASIBLE_DESTROY_PENALTY})")
    print(f"\n🔍 EXPLORATION (E): Structure diversity + Pool reduction")
    print(f"\n🚫 NO cost-based I | NO sanitize credit | FREE D/R learning")
    print("=" * 80)
   
    all_files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
    if not all_files:
        print(f"❌ No pickle files found in {DATA_DIR}")
        return
   
    selected_files = random.sample(all_files, min(len(all_files), NUM_TRAIN_INSTANCES))
    print(f"📚 Selected {len(selected_files)} training instances")
   
    print("⏳ Initializing worker pool...")
    pool = Pool(processes=NUM_WORKERS, initializer=worker_initializer, initargs=(selected_files,))
    toolbox.register("map", pool.map)
    print("✅ Worker pool ready!")
   
    pop = toolbox.population(n=POP_SIZE)
   
    # THÊM: Khởi tạo archive
    archive = [] # List of macros (list of tuples)
    ARCHIVE_MAX_SIZE = 500
   
    print("\n🔄 Evaluating initial population...")
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid)
    for ind, fit in zip(invalid, fitnesses):
        I, E = fit
        novelty = compute_novelty(ind, archive)
        ind.fitness.values = (I, E, novelty, -len(ind))
   
    # THÊM: Add newly evaluated to archive
    for ind in invalid:
        archive.append(list(ind))
   
    # Cap archive (FIFO)
    if len(archive) > ARCHIVE_MAX_SIZE:
        archive = archive[-ARCHIVE_MAX_SIZE:]
   
    pop = toolbox.select(pop, len(pop))
   
    for gen in range(1, N_GEN + 1):
        if hasattr(evaluate_macro_worker, 'scenario_histories'):
            for scenario_idx in evaluate_macro_worker.scenario_histories:
                history_data = evaluate_macro_worker.scenario_histories[scenario_idx]
                history_data['macro_history'].clear()
        print(f"\n{'='*80}")
        print(f"Generation {gen}/{N_GEN}")
        print(f"{'='*80}")
       
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB)
       
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            I, E = fit
            novelty = compute_novelty(ind, archive)
            ind.fitness.values = (I, E, novelty, -len(ind))
       
        # THÊM: Add newly evaluated to archive AFTER computing novelty
        for ind in invalid:
            archive.append(list(ind))
       
        # Cap archive
        if len(archive) > ARCHIVE_MAX_SIZE:
            archive = archive[-ARCHIVE_MAX_SIZE:]
       
        pop = toolbox.select(pop + offspring, POP_SIZE)
       
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
       
        I_values = [ind.fitness.values[0] for ind in pareto_front]
        E_values = [ind.fitness.values[1] for ind in pareto_front]
        novelty_values = [ind.fitness.values[2] for ind in pareto_front]
        len_values = [-ind.fitness.values[3] for ind in pareto_front]
       
        print(f"📊 Pareto front: {len(pareto_front)} macros")
        print(f" I: min={min(I_values):.3f}, avg={np.mean(I_values):.3f}, max={max(I_values):.3f}")
        print(f" E: min={min(E_values):.3f}, avg={np.mean(E_values):.3f}, max={max(E_values):.3f}")
        print(f" Novelty: min={min(novelty_values):.3f}, avg={np.mean(novelty_values):.3f}, max={max(novelty_values):.3f}")
        print(f" Len: min={min(len_values)}, avg={np.mean(len_values):.1f}, max={max(len_values)}")
   
    pool.close()
    pool.join()
   
    return pop
# ==============================================================================
# EXPORT MACROS
# ==============================================================================
def export_macros(pop, k=30):
    """Export top k macros from Pareto front"""
    print("\n" + "="*80)
    print("📤 EXPORTING DISCOVERED MACROS")
    print("="*80)
   
    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    pareto = sorted(pareto, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2]), reverse=True)
    selected = pareto[:k]
   
    macros = []
    for rank, ind in enumerate(selected, 1):
        sequence_readable = []
        for token_type, op_idx in ind:
            if token_type == 'D':
                op_name = get_op_name(DESTROY_OPS[op_idx])
                removal_pct = get_removal_fraction(DESTROY_OPS[op_idx]) * 100
                sequence_readable.append(f"D:{op_name}({removal_pct:.0f}%)")
            else: # 'R'
                op_name = get_op_name(REPAIR_OPS[op_idx])
                sequence_readable.append(f"R:{op_name}")
       
        macro_data = {
            "rank": rank,
            "I": round(ind.fitness.values[0], 4),
            "E": round(ind.fitness.values[1], 4),
            "novelty": round(ind.fitness.values[2], 4),
            "length": len(ind),
            "sequence_tokens": list(ind),
            "sequence_readable": sequence_readable
        }
       
        macros.append(macro_data)
        print(f"\n#{rank} | I={macro_data['I']:.3f}, E={macro_data['E']:.3f}, Novelty={macro_data['novelty']:.3f}, Len={macro_data['length']}")
        print(f" {' → '.join(sequence_readable)}")
   
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(macros, f, indent=2)
   
    print(f"\n✅ Exported {len(macros)} macros to {OUTPUT_FILE}")
   
    # Statistics
    print("\n" + "="*80)
    print("📈 MACRO STATISTICS")
    print("="*80)
   
    # D/R ratio analysis
    total_D = sum(1 for m in macros for t, _ in m['sequence_tokens'] if t == 'D')
    total_R = sum(1 for m in macros for t, _ in m['sequence_tokens'] if t == 'R')
    print(f"🔄 D/R Ratio: {total_D}:{total_R} ({total_D/(total_D+total_R)*100:.1f}% destroy)")
   
    # Operator frequency
    destroy_freq = {}
    repair_freq = {}
   
    for m in macros:
        for token_type, op_idx in m['sequence_tokens']:
            if token_type == 'D':
                op_name = get_op_name(DESTROY_OPS[op_idx])
                destroy_freq[op_name] = destroy_freq.get(op_name, 0) + 1
            else:
                op_name = get_op_name(REPAIR_OPS[op_idx])
                repair_freq[op_name] = repair_freq.get(op_name, 0) + 1
   
    print("\n🔨 Top Destroy Operators:")
    for op, count in sorted(destroy_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f" {op}: {count} times")
   
    print("\n🔧 Top Repair Operators:")
    for op, count in sorted(repair_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f" {op}: {count} times")
   
    # Sequence pattern analysis
    print("\n📋 Sequence Patterns:")
    patterns = {}
    for m in macros:
        pattern = '-'.join([t for t, _ in m['sequence_tokens']])
        patterns[pattern] = patterns.get(pattern, 0) + 1
   
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f" {pattern}: {count} macros")
   
    # Length distribution
    lengths = [m['length'] for m in macros]
    print(f"\n📏 Length Distribution:")
    print(f" Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f}")
   
    return macros
# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Necessary for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    try:
        print("\n" + "="*80)
        print("🎬 STARTING GP MACRO MINING")
        print("="*80)
        print("\n📋 FITNESS COMPUTATION METHOD:")
        print(" I = Feasibility Improvement + Structural Stability + Repair Bonus")
        print(" - Terminal Destroy Penalty (if D at end + infeasible)")
        print(" E = Structure Diversity + Pool Reduction Bonus")
        print("\n🎯 EXPECTED MACRO TYPES:")
        print(" • High I: Feasibility improvers (refinement)")
        print(" • High E: Diverse explorers (escape)")
        print(" • GP learns which D→R pairings work best naturally")
       
        # 1. Run GP mining process
        print("\n⏰ Start time:", np.datetime64('now'))
        final_pop = run_gp_mining()
        print("\n⏰ End time:", np.datetime64('now'))
        # 2. Export best macros to JSON
        if final_pop:
            discovered_macros = export_macros(final_pop, k=30)
           
            # 3. Additional analysis
            print("\n" + "="*80)
            print("🔬 ADVANCED ANALYSIS")
            print("="*80)
           
            # Pareto front visualization (text-based)
            pareto = tools.sortNondominated(final_pop, k=len(final_pop), first_front_only=True)[0]
            print(f"\n📊 Pareto Front Size: {len(pareto)}")
           
            # I vs E scatter (simplified)
            print("\n🎯 Improvement (I) vs Exploration (E) Trade-off:")
            print(" (Each * represents one macro in Pareto front)")
           
            # Create simple ASCII scatter plot
            I_vals = [ind.fitness.values[0] for ind in pareto]
            E_vals = [ind.fitness.values[1] for ind in pareto]
           
            I_min, I_max = min(I_vals), max(I_vals)
            E_min, E_max = min(E_vals), max(E_vals)
           
            # Normalize to 0-20 for ASCII plot
            if I_max > I_min and E_max > E_min:
                plot_height = 10
                plot_width = 40
               
                grid = [[' ' for _ in range(plot_width)] for _ in range(plot_height)]
               
                for I, E in zip(I_vals, E_vals):
                    x = int((I - I_min) / (I_max - I_min) * (plot_width - 1))
                    y = plot_height - 1 - int((E - E_min) / (E_max - E_min) * (plot_height - 1))
                    if 0 <= x < plot_width and 0 <= y < plot_height:
                        grid[y][x] = '*'
               
                print(f"\n E ({E_max:.2f})")
                print(" │")
                for row in grid:
                    print(" │" + ''.join(row))
                print(" └" + "─" * plot_width + f"→ I ({I_max:.2f})")
                print(f" ({I_min:.2f})")
           
            # Operator pair analysis (D-R co-occurrence)
            print("\n" + "="*80)
            print("🔗 OPERATOR PAIRING ANALYSIS")
            print("="*80)
            print("Most common Destroy → Repair sequences:")
           
            pairs = {}
            for m in discovered_macros:
                seq = m['sequence_tokens']
                for i in range(len(seq) - 1):
                    if seq[i][0] == 'D' and seq[i+1][0] == 'R':
                        d_op = get_op_name(DESTROY_OPS[seq[i][1]])
                        r_op = get_op_name(REPAIR_OPS[seq[i+1][1]])
                        pair_key = f"{d_op} → {r_op}"
                        pairs[pair_key] = pairs.get(pair_key, 0) + 1
           
            for pair, count in sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f" {pair}: {count}x")
           
            # Success rate by context (if we tracked it)
            print("\n" + "="*80)
            print("📝 RECOMMENDATIONS")
            print("="*80)
            print("Based on discovered macros:")
           
            # Find best I macro
            best_I = max(discovered_macros, key=lambda m: m['I'])
            print(f"\n💪 Best Intensification Macro (I={best_I['I']:.3f}):")
            print(f" {' → '.join(best_I['sequence_readable'])}")
            print(f" → Use for: Refinement, feasibility improvement")
           
            # Find best E macro
            best_E = max(discovered_macros, key=lambda m: m['E'])
            print(f"\n🔍 Best Exploration Macro (E={best_E['E']:.3f}):")
            print(f" {' → '.join(best_E['sequence_readable'])}")
            print(f" → Use for: Escaping local optima, diversification")
           
            # Find most balanced
            balanced = min(discovered_macros, key=lambda m: abs(m['I'] - m['E']))
            print(f"\n⚖️ Most Balanced Macro (I={balanced['I']:.3f}, E={balanced['E']:.3f}):")
            print(f" {' → '.join(balanced['sequence_readable'])}")
            print(f" → Use for: General-purpose search")
           
            # Pattern analysis
            print("\n📊 MACRO PATTERN STATISTICS:")
           
            # R at end frequency
            r_end_count = sum(1 for m in discovered_macros if m['sequence_tokens'][-1][0] == 'R')
            d_end_count = len(discovered_macros) - r_end_count
            print(f"\n Terminal token distribution:")
            print(f" • R at end: {r_end_count}/{len(discovered_macros)} ({r_end_count/len(discovered_macros)*100:.1f}%)")
            print(f" • D at end: {d_end_count}/{len(discovered_macros)} ({d_end_count/len(discovered_macros)*100:.1f}%)")
            print(f" → GP learned: {'Consolidate with R' if r_end_count > d_end_count else 'Aggressive D OK if feasible'}")
           
            # R→D pattern frequency
            rd_pattern_count = 0
            for m in discovered_macros:
                seq = m['sequence_tokens']
                for i in range(len(seq) - 1):
                    if seq[i][0] == 'R' and seq[i+1][0] == 'D':
                        rd_pattern_count += 1
            print(f"\n R→D pattern occurrences: {rd_pattern_count}")
            print(f" → GP learned: Prepare before destroy = {rd_pattern_count / len(discovered_macros):.2f}x per macro")
           
            # Usage recommendations
            print("\n💡 USAGE RECOMMENDATIONS:")
            print(" • High I macros: Final refinement, feasibility push")
            print(" • High E macros: When stuck, need diversification")
            print(" • Balanced macros: General iterative improvement")
            print(" • Short macros (len=2-3): Fast iterations")
            print(" • Long macros (len=4-5): Deep transformations")
            print("\n🎯 INTEGRATION TIPS:")
            print(" • ALNS: Select macro based on search phase")
            print(" • PPO: Use I/E as state features + macro as action")
            print(" • Adaptive: Track macro success rates per context")
           
        else:
            print("\n❌ No population returned from GP mining")
           
        print("\n" + "="*80)
        print(f"🎉 DONE! Results saved to: {OUTPUT_FILE}")
        print("="*80)
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        traceback.print_exc()
    finally:
        print("\n👋 GP Mining completed!")