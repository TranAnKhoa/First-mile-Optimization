import sys
import os
import re
import math
from datetime import timedelta
from numpy.random import RandomState
import numpy as np
import time
# --- SETUP ĐƯỜNG DẪN MODULE ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORT ---
try:
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, worst_removal, shaw_removal, time_worst_removal
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret_2_insertion, time_shift_repair
    from routing.cvrp.alns_cvrp.local_search_operators import apply_2_opt, apply_relocate, apply_exchange
    # Import các hàm tiện ích cần thiết
    from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id, print_schedule
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# --- CẤU HÌNH ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
#INSTANCE_FILE = os.path.join(base_path, 'output_data', 'haiz.pkl')
INSTANCE_FILE = os.path.join(base_path, 'Project Code', 'output_data', 'Small_sample.pkl')
#INSTANCE_FILE = os.path.join(base_path, 'Project Code', 'output_data', 'CEL_instance.pkl')
SEED, ITER = 1234, 1000

# CẤU HÌNH SIMULATED ANNEALING
start_temperature = 1000
end_temperature = 0.1
cooling_rate = 0.999

print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")

# --- 1. ĐỌC DỮ LIỆU ---
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)

# --- 2. TẠO LỜI GIẢI BAN ĐẦU (Đã đơn giản hóa) ---
initial_schedule = compute_initial_solution(problem, rand)

# --- 3. TẠO MÔI TRƯỜNG ---
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=SEED)
best_solution, current_solution = env, env      
best_obj = best_solution.objective()[0]
best_total_penalty = best_solution.objective()[1]
best_wait_time = best_solution.objective()[2]

ini_total_penalty = best_total_penalty
int_solution = best_solution
int_best_obj = best_obj
ini_wait_time = best_wait_time
print(f"Initial Objective: {best_obj:.2f}")
destroy_operators = [random_removal, worst_removal, shaw_removal, time_worst_removal]
repair_operators = [best_insertion]
random_state = np.random.RandomState(seed=SEED)
# ==============================================================================
# HÀM MÔ PHỎNG VÀ CÁC HÀM HỖ TRỢ
def apply_full_local_search(repaired):
    
    print("      [LS] Running apply_relocate (Intra-route)...")
    repaired = apply_relocate(repaired) # O(M*K^3)
    
    print("      [LS] Running apply_2_opt (Intra-route)...")
    repaired = apply_2_opt(repaired) # O(M*K^3)
    
    # print("      [LS] SKIPPING apply_exchange (Inter-route)...")
    # repaired = apply_exchange(repaired) # (KHÔNG CHẠY CÁI NÀY)
    
    return repaired




# --- 4. CHẠY ALNS (Đã đơn giản hóa) ---
print("\n--- BẮT ĐẦU VÒNG LẶP ALNS ---")
temperature = start_temperature

for i in range(ITER):
    try: # <--- BỌC Ở ĐÂY
        destroy_op = random_state.choice(destroy_operators)
        repair_op = random_state.choice(repair_operators)
        
        # In ra để biết toán tử nào đang chạy
        print(f"\nIter {i}: Running {destroy_op.__name__}...")
        
        destroyed, unvisited = destroy_op(current_solution, random_state)
        print(unvisited)
        if not unvisited: continue
        
        farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
        if not farms_to_reinsert: continue
            
        print(f"Iter {i}: Running {repair_op.__name__}...")
        repaired, failed_to_insert = repair_op(destroyed, rand, unvisited_customers=farms_to_reinsert)
        
        if not failed_to_insert:
        
        # 1. GÁN TRỰC TIẾP (KHÔNG CHẠY LS)
        # Bỏ dòng: refined_solution = apply_full_local_search(repaired)
            refined_solution = repaired 

        # 2. TÍNH CHI PHÍ CỦA GIẢI PHÁP "THÔ" (CHƯA ĐÁNH BÓNG)
        current_obj = current_solution.objective()[0]
        refined_obj = refined_solution.objective()[0]

        # 3. KIỂM TRA XEM GIẢI PHÁP "THÔ" CÓ TỐT HƠN KHÔNG
        if refined_obj < best_obj:
            # 🔹 BẠN TÌM THẤY VÀNG 🔹
            # BÂY GIỜ MỚI CHẠY LS ĐỂ "ĐÁNH BÓNG" NÓ
            
            print(f"Iter {i}: New best found (Raw: {refined_obj:.2f}). Running Full Local Search to polish...")
            start_ls = time.time() # (Bạn cần import time)
            
            # 🔴 CHỈ CHẠY LS Ở ĐÂY 🔴
            refined_solution = apply_full_local_search(refined_solution) 
            
            refined_obj = refined_solution.objective()[0] # Tính lại obj sau khi đánh bóng
            print(f"Iter {i}: LS complete after {time.time() - start_ls:.2f}s. New polished obj = {refined_obj:.2f}")

            # Cập nhật giải pháp tốt nhất VÀ giải pháp hiện tại
            best_solution = refined_solution
            best_obj = refined_obj
            current_solution = refined_solution
        
        # 4. LOGIC SA (Simulated Annealing)
        elif random_state.random() < math.exp((current_obj - refined_obj) / temperature):
            # Chấp nhận giải pháp "thô" (không cần LS)
            current_solution = refined_solution
    except Exception as e:
        print(f"\n❌❌❌ LỖI NGHIÊM TRỌNG Ở ITERATION {i} ❌❌❌")
        print(f"Toán tử Destroy: {destroy_op.__name__}")
        print(f"Toán tử Repair: {repair_op.__name__}")
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc() # In ra toàn bộ traceback
        break # Dừng vòng lặp sau khi báo lỗi
    
    

    temperature = max(end_temperature, temperature * cooling_rate)


print(f"\n🏁 Initial Best Objective: {int_solution.objective()[0]:.2f}")
print(f"Tổng thời gian vi phạm Time Window: {ini_total_penalty}")
print(f"Tổng thời gian chờ: ",ini_wait_time)
print_schedule(int_solution)

print(f"\n🏁 Final Best Objective: {best_solution.objective()[0]:.2f}")
print(f"Tổng thời gian vi phạm Time Window: {best_total_penalty}")
print(f"Tổng thời gian chờ: ",best_wait_time)
print_schedule(best_solution)


