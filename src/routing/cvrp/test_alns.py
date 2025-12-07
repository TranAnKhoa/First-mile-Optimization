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
# --- SETUP ĐƯỜNG DẪN MODULE ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
#! K:\Data Science\SOS lab\Project Code\src\routing\cvrp
# --- IMPORT ---
try:    
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, time_worst_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, shaw_spatial, shaw_temporal, shaw_structural, shaw_hybrid, trip_removal, historical_removal
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle
    from routing.cvrp.alns_cvrp.local_search_operators import apply_2_opt, apply_relocate
    # Import các hàm tiện ích cần thiết
    from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id, print_schedule,\
    optimize_all_start_times,fmt, update_history_matrix, reconstruct_truck_finish_times, balance_depot_loads, cleanup_inter_factory_routes
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# --- CẤU HÌNH ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
INSTANCE_FILE = os.path.join(base_path, 'Project Code', 'output_data', 'CEL_400.pkl')
#INSTANCE_FILE = os.path.join(base_path, 'Project Code', 'output_data', 'Small_sample.pkl')
#INSTANCE_FILE = os.path.join(base_path, 'Project Code', 'output_data', 'CEL_instance.pkl')
SEED, ITER = 99013, 1000

# CẤU HÌNH SIMULATED ANNEALING
start_temperature = 1000
end_temperature = 0.1
cooling_rate = 0.999

print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")

# --- 1. ĐỌC DỮ LIỆU ---
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)

# --- 2. TẠO LỜI GIẢI BAN ĐẦU ---
# (Bước này dùng trạng thái random đầu tiên của rand)
initial_schedule = compute_initial_solution(problem, rand)

# --- 3. TẠO MÔI TRƯỜNG ---
# [QUAN TRỌNG]: Mô phỏng lại logic của PPO:
# PPO gọi rand.randint() sau khi tạo lời giải để lấy seed cho môi trường.
# Ta cũng phải làm y hệt để dòng chảy Random đồng bộ.
sim_seed = rand.randint(0, 1000000)

# Dùng sim_seed thay vì SEED gốc
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=sim_seed)

# Cleanup (Bước này sẽ dùng sim_seed ở trên để xử lý nếu có random)
env = cleanup_inter_factory_routes(env)

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

destroy_operators = [random_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, time_worst_removal\
                     ,shaw_spatial, shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, historical_removal]
repair_operators = [best_insertion,regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle ]
random_state = np.random.RandomState(seed=SEED)
# ==============================================================================
# HÀM MÔ PHỎNG VÀ CÁC HÀM HỖ TRỢ
"""def apply_full_local_search(repaired):
    print("[LS] Running apply_relocate (Intra-route)...")
    repaired = apply_relocate(repaired) # O(M*K^3)
    print("[LS] Running apply_2_opt (Intra-route)...")
    repaired = apply_2_opt(repaired)    
    return repaired"""

def get_op_name(op):
    """
    Hàm lấy tên thông minh: Tự động nhận diện tham số của Partial 
    để in ra tên cụ thể (VD: regret_2_trip thay vì regret_partial).
    """
    # 1. Nếu hàm có tên chính chủ (Hàm thường hoặc đã gán __name__)
    if hasattr(op, '__name__'):
        return op.__name__

    # 2. Nếu là Partial (Biến thể dùng functools.partial)
    if hasattr(op, 'func'):
        base_name = op.func.__name__
        kwargs = op.keywords if op.keywords else {}

        # --- TỰ ĐỘNG ĐẶT TÊN CHO REGRET ---
        if base_name == 'regret_k_insertion':
            k = kwargs.get('k_regret', '?')
            mode = kwargs.get('mode', 'position') # Mặc định là position
            return f"regret_{k}_{mode}"

        # --- TỰ ĐỘNG ĐẶT TÊN CHO WORST REMOVAL ---
        if base_name == 'worst_removal':
            alpha = kwargs.get('alpha', 0)
            if alpha == 0: return "worst_alpha_0"
            if alpha > 1000: return "worst_bigM"
            # Nếu alpha được truyền động (adaptive), nó có thể không hiện ở đây
            # nên ta check mode wrapper nếu có
            return f"worst_removal_variant"

        # --- TỰ ĐỘNG ĐẶT TÊN CHO SHAW ---
        if base_name == 'shaw_removal':
            if kwargs.get('w_dist') == 1.0 and kwargs.get('w_tw') == 0: return "shaw_spatial"
            if kwargs.get('w_tw') == 1.0: return "shaw_temporal"
            if kwargs.get('w_depot') > 0 and kwargs.get('w_access') > 0: return "shaw_structural"
            if kwargs.get('w_dist') == 1.0 and kwargs.get('w_tw') == 0.5: return "shaw_hybrid"

        return f"{base_name}_partial"

    return str(op)

# --- 4. CHẠY ALNS (Đã đơn giản hóa) ---
print("\n--- BẮT ĐẦU VÒNG LẶP ALNS ---")
global_history_matrix = {} 

# "Dạy" cho nó biết về giải pháp khởi tạo ban đầu
# (Giả sử current_solution đã được khởi tạo ở trên)
update_history_matrix(global_history_matrix, current_solution)

temperature = start_temperature
start_time = time.time()
MAX_REMOVE_FRACTION = 0.4  # Tỷ lệ phá vỡ tối đa (lúc đầu)
MIN_REMOVE_FRACTION = 0.05  # Tỷ lệ phá vỡ tối thiểu (lúc cuối)
for i in range(ITER):
    try: 
        # 1. CHỌN TOÁN TỬ
        destroy_op = random_state.choice(destroy_operators)
        repair_op = random_state.choice(repair_operators)
        
        print(f"\nIter {i}: Running {get_op_name(destroy_op)}...")
        
        progress = i / ITER
        remove_fraction = MAX_REMOVE_FRACTION - (MAX_REMOVE_FRACTION - MIN_REMOVE_FRACTION) * progress
        
        # ----------------------------------------------------------------------
        # [SỬA 1]: Dùng dict để truyền tham số (Kwargs)
        # ----------------------------------------------------------------------
        op_kwargs = {
            'remove_fraction': remove_fraction,
            'history_matrix': global_history_matrix  # Truyền history vào đây
        }
        
        # Gọi hàm destroy với **op_kwargs (Hàm nào cần gì thì tự lấy)
        destroyed, unvisited = destroy_op(current_solution, random_state, **op_kwargs)
        if not unvisited: continue
        
        farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
        if not farms_to_reinsert: continue
            
        print(f"Iter {i}: Running {get_op_name(repair_op)}...")
        repaired, failed_to_insert = repair_op(destroyed, random_state, unvisited_customers=farms_to_reinsert)
        if not failed_to_insert:
            refined_solution = repaired
            
            # Lấy TẤT CẢ giá trị của giải pháp "thô"
            current_obj = current_solution.objective()[0]
            
            refined_results = refined_solution.objective()
            refined_obj = refined_results[0]
            print("New_objective: ", refined_obj)
            # 3. KIỂM TRA XEM GIẢI PHÁP "THÔ" CÓ TỐT HƠN KHÔNG
            if refined_obj < best_obj:
                print(f"Iter {i}: New potential best found (Raw: {refined_obj:.2f}). Running Full Local Search to polish...")
                start_ls = time.time()
                
                # ============================================================
                # 1. TẠO BẢN SAO LƯU (BACKUP)
                # ============================================================
                solution_backup = copy.deepcopy(refined_solution)

                # ============================================================
                # 2. CHẠY LOCAL SEARCH & TỐI ƯU HÓA
                # ============================================================
                try:
                    # Chạy LS (Thay đổi cấu trúc tuyến)
                    #refined_solution = apply_full_local_search(refined_solution)
                    
                    # [MỚI]: Tối ưu thời gian (Giảm Waiting Time)
                    # Hàm này phải là bản "Safe" (có check cost) như đã bàn
                    refined_solution = optimize_all_start_times(refined_solution)
                    
                    # ============================================================
                    # 3. KIỂM TRA TOÀN CỤC (NGƯỜI GIÁM SÁT)
                    # ============================================================
                    refined_results_polished = refined_solution.objective()
                    polished_obj = refined_results_polished[0]
                    
                    print(f"Iter {i}: LS complete after {time.time() - start_ls:.2f}s. Polished obj = {polished_obj:.2f}")

                    # ============================================================
                    # 4. RA QUYẾT ĐỊNH: CHẤP NHẬN HAY TỪ CHỐI?
                    # ============================================================
                    if polished_obj < best_obj and polished_obj < 1e9:
                        # --- TRƯỜNG HỢP THÀNH CÔNG ---
                        print(f" ✅Optimize start time success: {best_obj:.2f} -> {polished_obj:.2f}")
                        
                        best_solution = refined_solution
                        current_solution = refined_solution 
                        
                        # Cập nhật các biến thống kê
                        best_obj = polished_obj
                        best_time_penalty = refined_results_polished[1]
                        best_wait_time = refined_results_polished[2]
                        best_cap_penalty = refined_results_polished[3]
                        
                        # [SỬA 2]: Cập nhật History Matrix (Học từ Best mới)
                        update_history_matrix(global_history_matrix, best_solution)
                    
                    else:
                        # --- TRƯỜNG HỢP THẤT BẠI ---
                        print(f"   ⚠️ LS THẤT BẠI (Gây lỗi phân thân/Tăng cost). Hoàn tác về bản trước LS.")
                        
                        refined_solution = solution_backup
                        
                        # Vẫn cập nhật Best Solution nếu bản backup (chưa LS) vẫn tốt hơn Best cũ
                        if refined_obj < best_obj:
                             print(f"  ✅ Cập nhật Best (Bản Pre-LS): {best_obj:.2f} -> {refined_obj:.2f}")
                             best_solution = refined_solution
                             current_solution = refined_solution
                             
                             backup_res = refined_solution.objective()
                             best_obj = backup_res[0]
                             best_time_penalty = backup_res[1]
                             best_wait_time = backup_res[2]
                             best_cap_penalty = backup_res[3]
                             
                             # [SỬA 3]: Cập nhật History Matrix (Học từ bản Backup tốt)
                             update_history_matrix(global_history_matrix, best_solution)

                except Exception as e:
                    print(f"❌ Lỗi trong quá trình Local Search tại Iter {i}: {e}")
                    import traceback
                    traceback.print_exc() # In ra dòng code chính xác gây lỗi
                    
                    # Tùy chọn: Nếu lỗi này xảy ra, có thể giải pháp đầu vào đã bị sai (thiếu khách)
                    # Hãy kiểm tra lại backup
                    check_cust = sum(len(r) for r in solution_backup.routes)
                    print(f"Kiểm tra bản Backup khi lỗi: Tổng khách = {check_cust}")
                    
                    refined_solution = solution_backup
                    raise e
                
            # 4. LOGIC SA (Simulated Annealing)
            elif random_state.random() < math.exp((current_obj - refined_obj) / temperature):
                current_solution = refined_solution
                
                # [SỬA 4]: Cập nhật History Matrix (Học từ giải pháp được chấp nhận bởi SA)
                update_history_matrix(global_history_matrix, current_solution)

        # 7. GIẢM NHIỆT ĐỘ
        temperature *= cooling_rate
                
    except Exception as e:
        print(f"❌ Lỗi Nặng: {e}")
        traceback.print_exc() # <-- Dòng này sẽ chỉ ra chính xác lỗi ở dòng số mấy
        raise e
    
    ### <SỬA 5>: (Tùy chọn) Lưu lại lịch sử của giải pháp TỐT NHẤT ###
    # (Để theo dõi sự thay đổi của các giá trị qua từng vòng lặp)
    best_obj_history.append(best_obj)
    best_time_penalty_history.append(best_time_penalty)
    best_wait_time_history.append(best_wait_time)
    best_cap_penalty_history.append(best_cap_penalty)

    temperature = max(end_temperature, temperature * cooling_rate)

def print_full_solution_details(solution_env, title):
    """
    HÀM IN COMPACT (GỌN GÀNG) - ĐÃ SỬA ĐỂ DEBUG LỖI
    """
    print(f"\n\n{'='*60}")
    print(f"=== {title} ===")
    print(f"{'='*60}")

    try:
        problem_instance = solution_env.problem_instance
        available_trucks = problem_instance['fleet']['available_trucks']
    except AttributeError:
        print("LỖI: Đối tượng solution không hợp lệ.")
        return

    if not solution_env.schedule:
        print("  (Không có tuyến đường nào)")
        return

    # 1. NHÓM CÁC TUYẾN THEO TRUCK_ID
    truck_routes_map = defaultdict(list)
    for route_info in solution_env.schedule:
        # Unpack 7-tuple (Đảm bảo schedule của bạn đã là 7-tuple toàn bộ)
        try:
            depot_idx, truck_id, customer_list, shift, start, finish, load = route_info
            truck_routes_map[truck_id].append(route_info)
        except ValueError:
            print(f"❌ Lỗi dữ liệu Schedule: Không phải 7-tuple -> {route_info}")
            continue

    # 2. SẮP XẾP VÀ IN
    sorted_truck_ids = sorted(truck_routes_map.keys())

    for truck_id in sorted_truck_ids:
        routes = truck_routes_map[truck_id]
        routes.sort(key=lambda x: x[4]) # Sort theo start_time
        
        truck_info = find_truck_by_id(truck_id, available_trucks)
        truck_cap = truck_info.get('capacity', 0) if truck_info else 0
        truck_type = truck_info.get('type', 'Unknown') if truck_info else 'Unknown'
        
        print(f"🚚 Truck {truck_id} ({truck_type}) chạy {len(routes)} chuyến:")

        for trip_idx, route_data in enumerate(routes, 1):
            depot_idx, _, customer_list, shift, start, finish, load = route_data
            
            # --- TÍNH TOÁN CHỈ SỐ (STATS) ---
            try:
                if shift == 'INTER-FACTORY':
                    # Logic Inter-Factory (Giữ nguyên)
                    velocity = 1.0 if truck_type in ["Single", "Truck and Dog"] else 0.5
                    task_name = customer_list[0]
                    total_dist = (finish - start) * velocity
                    total_wait = 0.0
                    time_pen = max(0, finish - 1900)
                    cap_pen = 0.0
                    route_str = f"{task_name.replace('_', ' ')}"
                    icon = "🏭"
                    trip_name = "Chuyến đặc biệt"
                else:
                    # Logic Farm Visit
                    # [QUAN TRỌNG] Gọi hàm tính toán với ĐÚNG tham số
                    # Hàm này trả về 6 giá trị: (finish, feasible, dist, wait, time_pen, cap_pen)
                    # Chúng ta cần truyền đủ: finish_time_route, route_load
                    
                    calc_results = _calculate_route_schedule_and_feasibility(
                        depot_idx, customer_list, shift, start, finish, load, problem_instance, truck_info
                    )
                    
                    # Unpack kết quả (6 giá trị)
                    _, total_dist, total_wait, time_pen, cap_pen = calc_results
                    
                    route_str = f"Depot {depot_idx} → {' → '.join(map(str, customer_list))} → Depot {depot_idx}"
                    icon = "🧭"
                    trip_name = f"Chuyến {trip_idx}"

            except Exception as e:
                # ‼️ IN RA LỖI THỰC SỰ ĐỂ DEBUG ‼️
                print(f"   ❌ Lỗi Python: {e}")
                total_dist, total_wait, time_pen, cap_pen = 0, 0, 0, 0
                route_str = "Lỗi tính toán (Xem chi tiết ở trên)"
                icon = "⚠️"
                trip_name = f"Chuyến {trip_idx}"

            # --- IN KẾT QUẢ ---
            sh, sm = divmod(int(start), 60)
            eh, em = divmod(int(finish), 60)
            
            print(f"{icon} {trip_name} ({shift}) - Depot {depot_idx} (Xuất phát {sh:02d}:{sm:02d}): "
                  f"{route_str}, Kết thúc: {eh:02d}:{em:02d}")

            pen_flag = "⚠️ " if (time_pen > 0 or cap_pen > 0) else ""
            
            print(f"   📊 Tổng: Dist: {total_dist:.1f} km | Wait: {total_wait:.1f} min | "
                  f"Demand: {load:.0f}/{truck_cap:.0f} | "
                  f"{pen_flag}Time Pen: {time_pen:.1f} | Cap Pen: {cap_pen:.1f}")

# 2. HẬU XỬ LÝ (POST-PROCESSING) CHO GIẢI PHÁP TỐT NHẤT
# ==============================================================================
print("\n>>> Đang tối ưu hóa: Loại bỏ Inter-Factory cũ & Tính toán chuyển kho mới...")

# [BƯỚC A]: Xóa sạch các tuyến Inter-Factory cũ (từ Initial)
# Lý do: Sau khi ALNS chạy, demand tại các kho đã thay đổi, các tuyến cũ là rác.
best_solution.schedule = [r for r in best_solution.schedule if r[3] != 'INTER-FACTORY']

# [BƯỚC B]: Tái tạo lại bảng thời gian xe từ các tuyến Farm tối ưu
# (Hàm này bạn đã thêm vào utils.py)
final_finish_times = reconstruct_truck_finish_times(best_solution)

# [BƯỚC C]: Tính toán và Chèn tuyến Inter-Factory MỚI
# Hàm này sẽ tự động chèn các chuyến chuyển hàng cần thiết vào cuối danh sách
best_solution = balance_depot_loads(best_solution, final_finish_times)

print(f"{'='*60}\n")

print_full_solution_details(int_solution, "CHI TIẾT LỊCH TRÌNH BAN ĐẦU")
print(f"Initial Objective: {int_best_obj:.2f}")
print(f"Initial Time Penalty: {ini_total_penalty:.2f}")
print(f"Initial Wait Time: {ini_wait_time:.2f}")
print(f"Initial Capacity Penalty: {ini_capacity_redun:.2f}")


# (Đây là giải pháp tốt nhất sau khi chạy ALNS)
print_full_solution_details(best_solution, "CHI TIẾT LỊCH TRÌNH TỐT NHẤT (FINAL)")
print(f"Final Objective: {best_obj:.2f}")
print(f"Final Time Penalty: {best_time_penalty/0.3:.2f}")
print(f"Final Wait Time: {best_wait_time/0.2:.2f}")
print(f"Final Capacity Penalty: {best_cap_penalty:.2f}")
print(f"\n--- KẾT THÚC VÒNG LẶP ALNS SAU {time.time() - start_time:.2f} giây ---")