import sys
import os
import re
import traceback
import math
import json
import copy
import csv
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from numpy.random import RandomState

# --- SETUP ĐƯỜNG DẪN MODULE ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORT ---
try:    
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import (
        random_removal, time_worst_removal, worst_removal_alpha_0, 
        worst_removal_bigM, worst_removal_adaptive, shaw_spatial, 
        shaw_temporal, shaw_structural, shaw_hybrid, trip_removal, 
        historical_removal, update_solution_state_after_destroy
    )
    from routing.cvrp.alns_cvrp.repair_operators import (
        best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, 
        regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, 
        regret_4_trip, regret_4_vehicle
    )
    from routing.cvrp.alns_cvrp.utils import (
        _calculate_route_schedule_and_feasibility, find_truck_by_id, 
        optimize_all_start_times, update_history_matrix, 
        reconstruct_truck_finish_times, balance_depot_loads, 
        cleanup_inter_factory_routes
    )
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit()

# ==============================================================================
# CẤU HÌNH MACRO & ALNS
# ==============================================================================
# Danh sách Operators để ánh xạ với JSON (Phải khớp thứ tự bên PPO)
DESTROY_OPS = [random_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, time_worst_removal, shaw_spatial, shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, historical_removal]
REPAIR_OPS = [best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle]
REMOVE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Tham số Adaptive (Thưởng điểm)
SIGMA_1 = 33  # New Best Solution
SIGMA_2 = 9   # Improved Solution
SIGMA_3 = 13  # Accepted Solution (worse but accepted by SA)
RHO = 0.1     # Reaction factor (Hệ số cập nhật trọng số)
PU = 100      # Số vòng lặp cập nhật trọng số một lần (Segment length)

# Cấu hình đường dẫn
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
INSTANCE_FILE = r"K:\\Data Science\\SOS lab\\Project Code\\output_data\\train_inst_22_size_278.pkl"
JSON_MACRO_FILE = r"K:\Data Science\SOS lab\Project Code\src\rl\environments\macro_hybrid_final_xeon.json" # File chứa macro
SEED, ITER = 99013, 1000

# Cấu hình Simulated Annealing
start_temperature = 1000
end_temperature = 0.1   
cooling_rate = 0.999

# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def load_macros(filename):
    """Load macro operators từ file JSON"""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(json_path):
        json_path = filename # Thử tìm ở thư mục gốc
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            macros = json.load(f)
        print(f"✅ Loaded {len(macros)} Macro-Operators from {filename}")
        return macros
    else:
        print(f"⚠️ Warning: '{filename}' not found.")
        return []

def count_real_customers(solution):
    """Đếm số lượng khách hàng thực tế (Trừ điểm TRANSFER)"""
    count = 0
    if not solution.schedule: return 0
    for route in solution.schedule:
        if len(route) >= 3:
            count += sum(1 for cust_id in route[2] if not str(cust_id).startswith('TRANSFER_'))
    return count

def execute_macro(macro_idx, macros, current_sol, random_state, history_matrix):
    """
    Thực thi logic Macro-Operator (Chuỗi hành động).
    Bao gồm cơ chế Safety Check (Rollback nếu mất khách).
    """
    op_data = macros[macro_idx]
    sequence_indices = op_data['sequence_indices']
    
    # Tạo bản sao để làm việc
    temp_sol = copy.deepcopy(current_sol)
    
    # Đếm khách ban đầu để đối chiếu
    initial_count = count_real_customers(temp_sol)
    
    op_kwargs = {'history_matrix': history_matrix}

    for i, step_indices in enumerate(sequence_indices):
        # Giải mã tham số từ JSON
        if len(step_indices) == 2:
            d_idx, r_idx = step_indices
            p_idx = 2 # Default 15%
        else:
            d_idx, p_idx, r_idx = step_indices
        
        try:
            d_op = DESTROY_OPS[d_idx]
            op_kwargs['remove_fraction'] = REMOVE_LEVELS[p_idx]
            r_op = REPAIR_OPS[r_idx]
            
            # Cleanup rác trước khi phá hủy
            temp_sol = cleanup_inter_factory_routes(temp_sol)
            
            # 1. DESTROY
            destroyed, unvisited = d_op(temp_sol, random_state, **op_kwargs)
            destroyed = update_solution_state_after_destroy(destroyed)
            
            if unvisited:
                farms = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
                if farms:
                    # 2. REPAIR
                    repaired, failed_to_insert = r_op(destroyed, random_state, unvisited_customers=farms)
                    
                    if failed_to_insert:
                        # Nếu chèn thất bại -> Rollback về solution gốc
                        return current_sol
                    
                    temp_sol = repaired
                else:
                    temp_sol = destroyed
            else:
                temp_sol = destroyed
                
        except Exception as e:
            # Gặp lỗi -> Rollback
            return current_sol

    # [SAFETY CHECK FINAL]
    final_count = count_real_customers(temp_sol)
    if final_count < initial_count:
        # Mất khách -> Rollback
        return current_sol

    # Optimize lần cuối
    temp_sol = optimize_all_start_times(temp_sol)
    return temp_sol

def print_solution_summary(solution, title):
    results = solution.objective()
    print(f"--- {title} ---")
    print(f"   Objective: {results[0]:.2f} | TimePen: {results[1]:.2f} | Wait: {results[2]:.2f} | CapPen: {results[3]:.2f}")

# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

# 1. LOAD DỮ LIỆU
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)
macros = load_macros(JSON_MACRO_FILE)

if not macros:
    print("❌ Không có Macro nào để chạy. Dừng chương trình.")
    sys.exit()

# 2. KHỞI TẠO ALNS WEIGHTS
num_macros = len(macros)
weights = np.ones(num_macros, dtype=float)  # Trọng số chọn (xác suất)
scores = np.zeros(num_macros, dtype=float)  # Điểm tích lũy trong segment
counts = np.zeros(num_macros, dtype=int)    # Số lần chọn trong segment

# 3. TẠO INITIAL SOLUTION
print("\n🔄 Đang tạo lời giải ban đầu...")
initial_schedule = compute_initial_solution(problem, rand)

sim_seed = rand.randint(0, 1000000)
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=sim_seed)
env = cleanup_inter_factory_routes(env)
env = optimize_all_start_times(env)

int_obj = env.objective()[0]
print(f"✅ Initial Objective: {int_obj:.2f}")

# Setup biến ALNS
current_solution = env
best_solution = copy.deepcopy(env)
best_obj = int_obj

global_history_matrix = {}
update_history_matrix(global_history_matrix, current_solution)

temperature = start_temperature
start_time = time.time()

print("\n--- BẮT ĐẦU VÒNG LẶP MACRO-ALNS ---")

for i in range(ITER):
    # --- A. CHỌN MACRO (Roulette Wheel) ---
    prob = weights / np.sum(weights)
    macro_idx = rand.choice(range(num_macros), p=prob)
    
    counts[macro_idx] += 1
    
    # --- B. THỰC THI MACRO ---
    # (Đã bao gồm logic check mất khách bên trong hàm execute_macro)
    candidate_sol = execute_macro(macro_idx, macros, current_solution, rand, global_history_matrix)
    
    # --- C. ĐÁNH GIÁ & CHẤP NHẬN ---
    current_obj = current_solution.objective()[0]
    candidate_res = candidate_sol.objective()
    candidate_obj = candidate_res[0]
    
    accepted = False
    score_increment = 0
    
    # Case 1: New Best
    if candidate_obj < best_obj:
        print(f"Iter {i} [Macro {macro_idx}]: 🎉 NEW BEST {best_obj:.2f} -> {candidate_obj:.2f}")
        best_obj = candidate_obj
        best_solution = copy.deepcopy(candidate_sol)
        current_solution = candidate_sol
        update_history_matrix(global_history_matrix, best_solution)
        
        score_increment = SIGMA_1
        accepted = True
        
    # Case 2: Improved Current
    elif candidate_obj < current_obj:
        # print(f"Iter {i} [Macro {macro_idx}]: Improved {current_obj:.2f} -> {candidate_obj:.2f}")
        current_solution = candidate_sol
        update_history_matrix(global_history_matrix, current_solution)
        
        score_increment = SIGMA_2
        accepted = True
        
    # Case 3: Simulated Annealing Acceptance
    else:
        delta = candidate_obj - current_obj
        probability = math.exp(-delta / max(temperature, 1e-6))
        if rand.rand() < probability:
            # print(f"Iter {i} [Macro {macro_idx}]: SA Accepted (Δ={delta:.2f})")
            current_solution = candidate_sol
            update_history_matrix(global_history_matrix, current_solution)
            
            score_increment = SIGMA_3
            accepted = True
    
    # --- D. CẬP NHẬT ĐIỂM ---
    scores[macro_idx] += score_increment
    
    # Giảm nhiệt độ
    temperature = max(end_temperature, temperature * cooling_rate)
    
    # --- E. CẬP NHẬT TRỌNG SỐ (ADAPTIVE WEIGHTS) ---
    # Cứ sau PU vòng lặp thì cập nhật lại trọng số dựa trên thành tích
    if (i + 1) % PU == 0:
        print(f"\n[ALNS Update] Updating weights based on performance...")
        for m in range(num_macros):
            if counts[m] > 0:
                # Công thức ALNS chuẩn: w_new = rho * w_old + (1-rho) * (score / count)
                weights[m] = RHO * weights[m] + (1 - RHO) * (scores[m] / counts[m])
            else:
                # Nếu không được chọn lần nào, giữ nguyên hoặc giảm nhẹ (ở đây giữ nguyên)
                pass
        
        # Reset điểm và bộ đếm cho segment tiếp theo
        scores.fill(0)
        counts.fill(0)

# ==============================================================================
# POST-PROCESSING & KẾT QUẢ
# ==============================================================================
print("\n>>> Đang tối ưu hóa cuối cùng...")
best_solution.schedule = [r for r in best_solution.schedule if r[3] != 'INTER-FACTORY']
final_finish_times = reconstruct_truck_finish_times(best_solution)
best_solution = balance_depot_loads(best_solution, final_finish_times)

print_solution_summary(best_solution, "FINAL BEST SOLUTION")
print(f"Total Iterations: {ITER}")
print(f"Execution Time: {time.time() - start_time:.2f}s")