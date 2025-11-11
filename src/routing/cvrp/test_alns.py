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
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret_2_insertion, regret_3_insertion, regret_4_insertion
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

### <SỬA 1>: Khởi tạo TẤT CẢ các biến 'best' và 'history' ###

# Lấy tất cả giá trị ban đầu MỘT LẦN
initial_results = best_solution.objective()
best_obj = initial_results[0]
best_time_penalty = initial_results[1]
best_wait_time = initial_results[2]
best_cap_penalty = initial_results[3] # Index [3] là capacity

# Lưu lại giá trị ban đầu để in
int_solution = best_solution
int_best_obj = best_obj
ini_total_penalty = best_time_penalty
ini_wait_time = best_wait_time
ini_capacity_redun = best_cap_penalty # Giữ tên biến của bạn

print(f"Initial Objective: {best_obj:.2f}")
print(f"Initial Time Penalty: {ini_total_penalty:.2f}")
print(f"Initial Wait Time: {ini_wait_time:.2f}")
print(f"Initial Capacity Penalty: {ini_capacity_redun:.2f}")

# (Tùy chọn) Tạo danh sách "lịch sử" để theo dõi
best_obj_history = [best_obj]
best_time_penalty_history = [best_time_penalty]
best_wait_time_history = [best_wait_time]
best_cap_penalty_history = [best_cap_penalty]

# -----------------------------------------------------------------

destroy_operators = [random_removal, worst_removal, shaw_removal, time_worst_removal]
repair_operators = [best_insertion, regret_2_insertion, regret_3_insertion, regret_4_insertion]
random_state = np.random.RandomState(seed=SEED)
# ==============================================================================
# HÀM MÔ PHỎNG VÀ CÁC HÀM HỖ TRỢ
def apply_full_local_search(repaired):
    
    print("       [LS] Running apply_relocate (Intra-route)...")
    repaired = apply_relocate(repaired) # O(M*K^3)
    
    print("       [LS] Running apply_2_opt (Intra-route)...")
    repaired = apply_2_opt(repaired) # O(M*K^3)
    
    # print("       [LS] SKIPPING apply_exchange (Inter-route)...")
    # repaired = apply_exchange(repaired) # (KHÔNG CHẠY CÁI NÀY)
    
    return repaired


# --- 4. CHẠY ALNS (Đã đơn giản hóa) ---
print("\n--- BẮT ĐẦU VÒNG LẶP ALNS ---")
temperature = start_temperature

for i in range(ITER):
    try: 
        destroy_op = random_state.choice(destroy_operators)
        repair_op = random_state.choice(repair_operators)
        
        print(f"\nIter {i}: Running {destroy_op.__name__}...")
        
        destroyed, unvisited = destroy_op(current_solution, random_state)

        if not unvisited: continue
        
        farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
        if not farms_to_reinsert: continue
            
        print(f"Iter {i}: Running {repair_op.__name__}...")
        repaired, failed_to_insert = repair_op(destroyed, rand, unvisited_customers=farms_to_reinsert)
        
        if not failed_to_insert:
        
            refined_solution = repaired 

            ### <SỬA 2>: Lấy TẤT CẢ giá trị của giải pháp "thô" (gọi 1 lần) ###
            current_obj = current_solution.objective()[0]
            
            # Lấy kết quả của giải pháp "thô" (chưa đánh bóng)
            refined_results = refined_solution.objective()
            refined_obj = refined_results[0]
            # (Chúng ta chưa cần lưu 3 giá trị còn lại ở bước này)
            
            # 3. KIỂM TRA XEM GIẢI PHÁP "THÔ" CÓ TỐT HƠN KHÔNG
            if refined_obj < best_obj:
                print(f"Iter {i}: New best found (Raw: {refined_obj:.2f}). Running Full Local Search to polish...")
                start_ls = time.time()
                
                # 🔴 CHỈ CHẠY LS Ở ĐÂY 🔴
                refined_solution = apply_full_local_search(refined_solution) 
                
                ### <SỬA 3>: Lấy TẤT CẢ giá trị (SAU KHI ĐÁNH BÓNG) ###
                refined_results_polished = refined_solution.objective() 
                
                refined_obj = refined_results_polished[0]
                
                print(f"Iter {i}: LS complete after {time.time() - start_ls:.2f}s. New polished obj = {refined_obj:.2f}")

                best_solution = refined_solution
                current_solution = refined_solution

                ### <SỬA 4>: Cập nhật TẤT CẢ các biến 'best' ###
                best_obj = refined_results_polished[0]
                best_time_penalty = refined_results_polished[1]
                best_wait_time = refined_results_polished[2]
                best_cap_penalty = refined_results_polished[3]
                
            # 4. LOGIC SA (Simulated Annealing)
            elif random_state.random() < math.exp((current_obj - refined_obj) / temperature):
                current_solution = refined_solution
                
    except Exception as e:
        print(f"\n❌❌❌ LỖI NGHIÊM TRỌNG Ở ITERATION {i} ❌❌❌")
        print(f"Toán tử Destroy: {destroy_op.__name__}")
        print(f"Toán tử Repair: {repair_op.__name__}")
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc() # In ra toàn bộ traceback
        break # Dừng vòng lặp sau khi báo lỗi
    
    ### <SỬA 5>: (Tùy chọn) Lưu lại lịch sử của giải pháp TỐT NHẤT ###
    # (Để theo dõi sự thay đổi của các giá trị qua từng vòng lặp)
    best_obj_history.append(best_obj)
    best_time_penalty_history.append(best_time_penalty)
    best_wait_time_history.append(best_wait_time)
    best_cap_penalty_history.append(best_cap_penalty)

    temperature = max(end_temperature, temperature * cooling_rate)


### <SỬA 6>: In kết quả cuối cùng bằng các biến đã lưu ###
# (Không cần gọi lại solution.objective() nhiều lần)

print(f"\n🏁 Initial Best Objective: {int_best_obj:.2f}")
print(f"Tổng thời gian vi phạm Time Window: {ini_total_penalty:.2f}")
print(f"Tổng thời gian chờ: {ini_wait_time:.2f}")
print(f"Tổng số capacity bị vi phạm: {ini_capacity_redun:.2f}")
print_schedule(int_solution)

print(f"\n🏁 Final Best Objective: {best_obj:.2f}")
print(f"Tổng thời gian vi phạm Time Window: {best_time_penalty:.2f}")
print(f"Tổng thời gian chờ: {best_wait_time:.2f}")
print(f"Tổng số capacity bị vi phạm: {best_cap_penalty:.2f}")
print_schedule(best_solution)