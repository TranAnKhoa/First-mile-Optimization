import sys
import os
import re
import traceback
import math
from datetime import timedelta
from numpy.random import RandomState
import numpy as np
import time
import copy
from collections import defaultdict
import csv
from pathlib import Path

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
    # from routing.cvrp.alns_cvrp.local_search_operators import apply_2_opt, apply_relocate
    from routing.cvrp.alns_cvrp.utils import (
        _calculate_route_schedule_and_feasibility, _get_farm_info, 
        find_truck_by_id, print_schedule, optimize_all_start_times, fmt, 
        update_history_matrix, reconstruct_truck_finish_times, 
        balance_depot_loads, cleanup_inter_factory_routes
    )
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# ==============================================================================
# --- CẤU HÌNH & HYPERPARAMETERS (Theo Paper ICAPS 2024) ---
# ==============================================================================
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
INSTANCE_FILE = r"K:\\Data Science\\SOS lab\\Project Code\\output_data\\train_inst_22_size_278.pkl"
# INSTANCE_FILE = r"K:\\Data Science\\SOS lab\\Project Code\\output_data\\CEL_instance.pkl"

SEED = 99013
ITER = 1000  # Số vòng lặp

# [cite_start]ALNS Specific Parameters [cite: 192, 197]
SCORE_SIGMA1 = 5  # New Global Best
SCORE_SIGMA2 = 3  # Better than Current
SCORE_SIGMA3 = 1  # Accepted
SCORE_SIGMA4 = 0  # Rejected
REACTION_FACTOR = 0.8  # Lambda (Decay factor)

# Simulated Annealing Settings
SA_WORSE_RATIO = 0.05   # Chấp nhận lời giải tệ hơn 5%...
SA_PROBABILITY = 0.5    # ...với xác suất 50% tại nhiệt độ khởi đầu [cite: 195]

print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")

# --- 1. ĐỌC DỮ LIỆU ---
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)
random_state = np.random.RandomState(seed=SEED) # Dùng cho Operators

# ==============================================================================
# --- 2. TẠO LỜI GIẢI BAN ĐẦU ---
# ==============================================================================
print("\n🔄 Đang tạo lời giải ban đầu (Heuristic)...")
initial_schedule = compute_initial_solution(problem, rand)

print("🔧 Optimizing initial solution...")
sim_seed = rand.randint(0, 1000000)
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=sim_seed)
env = cleanup_inter_factory_routes(env)
env = optimize_all_start_times(env)

initial_results = env.objective()
int_best_obj = initial_results[0]
ini_total_penalty = initial_results[1]
ini_wait_time = initial_results[2]
ini_capacity_redun = initial_results[3]

print(f"\n📊 Initial Solution (Optimized): {int_best_obj:.2f}")

# ==============================================================================
# --- 3. KHỞI TẠO BIẾN CHO ALNS ---
# ==============================================================================
int_solution = copy.deepcopy(env)
best_solution = copy.deepcopy(env)
current_solution = copy.deepcopy(env)

best_obj = int_best_obj
current_obj = int_best_obj

# Setup Operators
destroy_operators = [
    random_removal, worst_removal_alpha_0, worst_removal_bigM, 
    worst_removal_adaptive, time_worst_removal, shaw_spatial, 
    shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, 
    historical_removal
]
repair_operators = [
    best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, 
    regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, 
    regret_4_trip, regret_4_vehicle
]

# [NEW] Khởi tạo Trọng số (Weights)
d_weights = np.ones(len(destroy_operators)) # Khởi tạo bằng 1
r_weights = np.ones(len(repair_operators))

# [cite_start][NEW] Tính toán SA Start Temperature tự động [cite: 195]
if int_best_obj > 0:
    start_temperature = - (SA_WORSE_RATIO * int_best_obj) / math.log(SA_PROBABILITY)
else:
    start_temperature = 100
print(f"🌡️ Calculated Start Temperature: {start_temperature:.2f}")

# [cite_start]Linear Decay Step [cite: 177]
# End temp = 0
step_temp_drop = start_temperature / ITER 

# History tracking
best_obj_history = [best_obj]

# Global History Matrix (cho Historical Removal)
global_history_matrix = {}
update_history_matrix(global_history_matrix, current_solution)

# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def count_real_customers(solution):
    """Đếm khách hàng thực (trừ Transfer node) để đảm bảo không bị mất khách."""
    count = 0
    if not solution.schedule: return 0
    for route in solution.schedule:
        if len(route) >= 3:
            count += sum(1 for cust_id in route[2] if not str(cust_id).startswith('TRANSFER_'))
    return count

def get_op_name(op):
    """Lấy tên Operator để in log."""
    if hasattr(op, '__name__'): return op.__name__
    if hasattr(op, 'func'): return op.func.__name__
    return str(op)

def select_operator(operators, weights):

    total_weight = np.sum(weights)
    probs = weights / total_weight
    idx = random_state.choice(range(len(operators)), p=probs)
    return operators[idx], idx

# ==============================================================================
# --- 4. BẮT ĐẦU VÒNG LẶP ALNS ---
# ==============================================================================
print("\n🚀 --- BẮT ĐẦU ALNS (VANILLA CONFIG) ---")
start_time = time.time()
MAX_REMOVE_FRACTION = 0.4
MIN_REMOVE_FRACTION = 0.05

current_temperature = start_temperature

for i in range(ITER):
    try: 
        # 1. Chọn Operator dựa trên trọng số (Adaptive)
        destroy_op, d_idx = select_operator(destroy_operators, d_weights)
        repair_op, r_idx = select_operator(repair_operators, r_weights)
        
        # Tính toán mức độ phá hủy động
        progress = i / ITER
        remove_fraction = MAX_REMOVE_FRACTION - (MAX_REMOVE_FRACTION - MIN_REMOVE_FRACTION) * progress
        
        # Chuẩn bị tham số
        op_kwargs = {
            'remove_fraction': remove_fraction,
            'history_matrix': global_history_matrix
        }

        # 2. Thực thi Destroy & Repair
        temp_sol = copy.deepcopy(current_solution)
        temp_sol = cleanup_inter_factory_routes(temp_sol)
        
        target_customer_count = count_real_customers(temp_sol) # Safety Check
        
        # Destroy
        destroyed, unvisited = destroy_op(temp_sol, random_state, **op_kwargs)
        destroyed = update_solution_state_after_destroy(destroyed)
        
        if not unvisited:
            # Nếu destroy không xóa gì cả (hiếm), phạt nhẹ hoặc bỏ qua
            continue
            
        farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
        if not farms_to_reinsert: continue
        
        # Repair
        repaired, failed_to_insert = repair_op(destroyed, random_state, unvisited_customers=farms_to_reinsert)
        
        # 3. Đánh giá và Chấp nhận (Acceptance & Scoring)
        score = SCORE_SIGMA4 # Mặc định là Rejected (0)
        accepted = False
        
        if not failed_to_insert:
            refined_solution = repaired
            
            # Check an toàn: Mất khách hàng?
            if count_real_customers(refined_solution) < target_customer_count:
                print(f"Iter {i}: ⚠️ Safety Rollback (Lost customers)")
                continue # Skip update, score = 0
            
            # Optimize (Có thể comment dòng này nếu chạy quá chậm)
            refined_solution = optimize_all_start_times(refined_solution)
            
            refined_obj = refined_solution.objective()[0]
            
            # Logic so sánh
            delta = refined_obj - current_obj
            
            if refined_obj < best_obj:
            
                print(f"Iter {i} | T={current_temperature:.1f} | ⭐ NEW BEST: {best_obj:.2f} -> {refined_obj:.2f} ({get_op_name(destroy_op)} + {get_op_name(repair_op)})")
                best_obj = refined_obj
                best_solution = copy.deepcopy(refined_solution)
                current_solution = refined_solution
                current_obj = refined_obj
                
                update_history_matrix(global_history_matrix, best_solution)
                score = SCORE_SIGMA1
                accepted = True
                
            elif refined_obj < current_obj:
            
                # print(f"Iter {i} | 🟢 Improved: {current_obj:.2f} -> {refined_obj:.2f}")
                current_solution = refined_solution
                current_obj = refined_obj
                update_history_matrix(global_history_matrix, current_solution)
                score = SCORE_SIGMA2
                accepted = True
                
            else:
                # Simulated Annealing Criteria
                # Prob = exp(-(f(s') - f(s)) / T)
                prob = math.exp(-delta / max(current_temperature, 1e-6))
                if random_state.rand() < prob:
                
                    # print(f"Iter {i} | T={current_temperature:.1f} | 🟡 SA Accept (Δ={delta:.2f})")
                    current_solution = refined_solution
                    current_obj = refined_obj
                    update_history_matrix(global_history_matrix, current_solution)
                    score = SCORE_SIGMA3
                    accepted = True
                else:
                    # Rejected -> Score 0
                    score = SCORE_SIGMA4
        
        # 4. Cập nhật Trọng số (Adaptive Weight Update)
        # [cite_start]Công thức: weight = lambda * weight + (1 - lambda) * score [cite: 101]
        d_weights[d_idx] = REACTION_FACTOR * d_weights[d_idx] + (1 - REACTION_FACTOR) * score
        r_weights[r_idx] = REACTION_FACTOR * r_weights[r_idx] + (1 - REACTION_FACTOR) * score
        
        # [cite_start]5. Giảm nhiệt độ (Linear Decay) [cite: 177]
        current_temperature = max(0, current_temperature - step_temp_drop)
        
        # Logging định kỳ
        if i % 100 == 0:
            print(f"--- Iter {i}/{ITER} --- Best: {best_obj:.2f} --- Temp: {current_temperature:.2f}")

    except Exception as e:
        print(f"❌ Error at Iter {i}: {e}")
        # traceback.print_exc()

    best_obj_history.append(best_obj)

# ==============================================================================
# --- 5. POST-PROCESSING & KẾT QUẢ ---
# ==============================================================================
print("\n🏁 --- FINISHING UP ---")
print(">>> Final Cleanup & Balancing...")

# Xóa tuyến ảo cũ và cân bằng lại
best_solution.schedule = [r for r in best_solution.schedule if r[3] != 'INTER-FACTORY']
final_finish_times = reconstruct_truck_finish_times(best_solution)
best_solution = balance_depot_loads(best_solution, final_finish_times)
# Optimize lần cuối cùng cho chắc chắn
best_solution = optimize_all_start_times(best_solution)

final_res = best_solution.objective()
print(f"\n{'='*60}")
print(f"RESULT SUMMARY (SEED {SEED})")
print(f"{'='*60}")
print(f"► Initial Cost: {int_best_obj:.2f}")
print(f"► Final Cost:   {final_res[0]:.2f}")
print(f"► Improvement:  {((int_best_obj - final_res[0])/int_best_obj)*100:.2f}%")
print(f"► Runtime:      {time.time() - start_time:.2f}s")
print(f"{'='*60}")

