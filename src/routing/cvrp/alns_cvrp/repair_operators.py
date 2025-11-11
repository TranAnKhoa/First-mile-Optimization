import copy
import random
import numpy as np
import re
from collections import defaultdict
import itertools
import time
from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id, _check_insertion_efficiency, _check_insertion_delta, _calculate_route_schedule_WITH_SLACK, _check_accessibility
# ==============================================================================
# HÀM TIỆN ÍCH CHUNG (Không thay đổi)
# ==============================================================================

# --- HÀM TRỢ GIÚP: TÌM VỊ TRÍ TỐT NHẤT CHO MỘT FARM ---

# ==============================================================================
# TOÁN TỬ SỬA CHỮA CHÍNH (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================
def _find_all_inserts_for_visit(schedule_list, visit_id, problem_instance):
    """
    ## PHIÊN BẢN TỐI ƯU HÓA - DEBUG ##
    """
    all_insertions = []
    WAIT_COST_PER_MIN = 0.2 # (Nên lấy từ problem_instance)
    
    # In ra để xem hàm này có được gọi nhiều không
    # print(f"  [_find_all] Bắt đầu tìm chỗ cho {visit_id}...")
    
    M = len(schedule_list)
    
    # 1. Thử chèn vào các tuyến đường hiện có
    for route_idx, route_info in enumerate(schedule_list):
        if route_info[3] == 'INTER-FACTORY': continue
            
        # In định kỳ
        # if (route_idx + 1) % 50 == 0:
        #     print(f"    [_find_all] ...Đang check tuyến {route_idx + 1}/{M}...")

        # (Code O(K) của bạn... _calculate_route_schedule_WITH_SLACK)
        # ...
        
        depot_idx, truck_id, customer_list, shift, start_time_at_depot = route_info
        truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
        if not truck_info: continue
        current_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in customer_list)
        is_feasible_orig, orig_dist, orig_wait, original_schedule = \
            _calculate_route_schedule_WITH_SLACK(
                depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info
            )
        if not is_feasible_orig:
            continue
            
        # (Code O(K) * O(1) của bạn... _check_insertion_delta)
        for insert_pos in range(len(original_schedule) - 1):
            is_feasible, cost_increase = _check_insertion_delta(
                problem_instance, route_info, original_schedule, 
                insert_pos, visit_id, 
                truck_info, current_load
            )
            if is_feasible:
                all_insertions.append({
                    'cost': cost_increase, 'route_idx': route_idx, 
                    'pos': insert_pos, 'shift': shift, 'new_route_details': None
                })

    # 2. Thử tạo một tuyến đường mới
    # 🔴 NÚT THẮT CỔ CHAI RẤT NGHI NGỜ Ở ĐÂY 🔴
    # print(f"  [_find_all] ...Đang check 'Tạo tuyến mới' cho {visit_id}...")
    
    # (Logic "Tạo tuyến mới" O(T*S*O(K)) của bạn)
    # ... (sao chép y hệt logic cũ của bạn vào đây) ...
    farm_idx, farm_details, farm_demand = _get_farm_info(visit_id, problem_instance)
    facilities = problem_instance['facilities']
    closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx]))
    depot_region = facilities[closest_depot_idx].get('region', None)
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    suitable_trucks = []
    available_trucks = problem_instance['fleet']['available_trucks']
    for truck in available_trucks:
        if truck.get('region') != depot_region or truck['capacity'] < farm_demand: continue
        truck_type_idx = type_to_idx.get(truck['type']);
        if truck_type_idx is None: continue
        depot_details = facilities[closest_depot_idx]
        if _check_accessibility(truck, farm_details, depot_details):
             suitable_trucks.append(truck)

    if suitable_trucks:
        best_truck_for_new_route = min(suitable_trucks, key=lambda t: t['capacity'])
        var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
            (best_truck_for_new_route['type'], best_truck_for_new_route['region']), 1.0)
        
        for shift in ['AM', 'PM']:
            is_feasible, new_dist, new_wait, _ = _calculate_route_schedule_WITH_SLACK(
                closest_depot_idx, [visit_id], shift, 0, problem_instance, best_truck_for_new_route)
            
            if is_feasible:
                cost_of_new_route = (new_dist * var_cost_per_km) + (new_wait * WAIT_COST_PER_MIN)
                all_insertions.append({
                    'cost': cost_of_new_route, 'route_idx': -1, 'pos': 0, 'shift': shift,
                    'new_route_details': (closest_depot_idx, best_truck_for_new_route['id'], shift, 0)
                })
    
    all_insertions.sort(key=lambda x: x['cost'])
    return all_insertions

# ==============================================================================
# CÁC TOÁN TỬ SỬA CHỮA (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================

def best_insertion(current, random_state, **kwargs):
    """
    ## PHIÊN BẢN TỐI ƯU HÓA (O(N log N)) - DEBUG ##
    """
    print(f"[BestInsert] Bắt đầu. Tổng số khách cần chèn (N): {len(kwargs['unvisited_customers'])}")
    start = time.time()
    
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers_set = set(kwargs['unvisited_customers'])
    failed_customers = []
    
    all_best_insertions = []
    
    # --- PHASE 1: TÍNH TOÁN CHI PHÍ (Chạy N lần) ---
    print(f"[BestInsert] ... Bắt đầu Phase 1: Tính toán chi phí chèn (N={len(unserved_customers_set)})...")
    
    for idx, farm_id in enumerate(unserved_customers_set): 
        
        # In định kỳ để xem tiến độ
        if (idx + 1) % 10 == 0:
            print(f"[BestInsert] ... Phase 1: Đang tính toán cho khách {idx + 1}/{len(unserved_customers_set)} (ID: {farm_id})...")
            
        # 🔴 NÚT THẮT CỔ CHAI CÓ THỂ Ở ĐÂY 🔴
        insertions = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance) 
        
        if not insertions:
            continue
            
        best_insert_for_this_farm = insertions[0]
        all_best_insertions.append(
            (best_insert_for_this_farm['cost'], farm_id, best_insert_for_this_farm)
        )

    phase1_time = time.time()
    print(f"[BestInsert] >>> Đã xong Phase 1 sau {phase1_time - start:.2f} giây.")

    # --- PHASE 2: SẮP XẾP (Chạy 1 lần) ---
    print(f"[BestInsert] ... Bắt đầu Phase 2: Sắp xếp {len(all_best_insertions)} lựa chọn...")
    all_best_insertions.sort(key=lambda x: x[0])
    phase2_time = time.time()
    print(f"[BestInsert] >>> Đã xong Phase 2 sau {phase2_time - phase1_time:.2f} giây.")

    # --- PHASE 3: THỰC HIỆN CHÈN (Chạy N lần) ---
    print(f"[BestInsert] ... Bắt đầu Phase 3: Thực hiện chèn...")
    
    # (Code chèn của bạn y hệt như cũ)
    for cost, farm_id, details in all_best_insertions:
        if farm_id not in unserved_customers_set:
            continue
        
        if details['route_idx'] == -1:
            depot, truck_id, shift, start_time = details['new_route_details']
            repaired.schedule.append((depot, truck_id, [farm_id], details['shift'], start_time))
        else:
            route_idx = details['route_idx']
            pos = details['pos']
            if route_idx >= len(repaired.schedule):
                failed_customers.append(farm_id)
                unserved_customers_set.remove(farm_id)
                continue
            route_as_list = list(repaired.schedule[route_idx])
            if pos > len(route_as_list[2]):
                pos = len(route_as_list[2])
            route_as_list[2].insert(pos, farm_id)
            repaired.schedule[route_idx] = tuple(route_as_list)
        
        unserved_customers_set.remove(farm_id)

    # ... (Phần xử lý failed_customers) ...
    failed_customers.extend(list(unserved_customers_set))
    
    end_time = time.time()
    print(f"[BestInsert] >>> Hoàn thành. Tổng thời gian: {end_time - start:.2f} giây. Lỗi: {len(failed_customers)}")
    
    return repaired, failed_customers

def regret_k_insertion(current, random_state, **kwargs):
    """
    ## PHIÊN BẢN TỐI ƯU HÓA (O(N log N)) ##
    Tính toán regret MỘT LẦN, sắp xếp, và sau đó chèn tất cả.
    Nhanh hơn O(N^2) nhưng "lỗi thời" (stale) về chi phí.
    """
    
    print(f"[RegretInsert] Bắt đầu. Tổng số khách cần chèn (N): {len(kwargs['unvisited_customers'])}")
    start_time = time.time()
    
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    
    unserved_customers_set = set(kwargs.get('unvisited_customers', []))
    failed_customers = []
    # Lấy K từ kwargs, mặc định là 2
    K = kwargs.get('k_regret') 

    all_regret_options = []

    # --- PHASE 1: TÍNH TOÁN REGRET (Chạy N lần) ---
    # N (ví dụ 60) * O(M*K)
    print(f"[RegretInsert] ... Bắt đầu Phase 1: Tính toán Regret (N={len(unserved_customers_set)}, K={K})...")
    
    for farm_id in unserved_customers_set:
        
        # Gọi hàm _find_all TỐI ƯU của bạn
        insertions = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance) 
        
        if not insertions:
            continue
            
        best_insert = insertions[0]
        regret_value = 0

        # --- Logic tính K-Regret (y hệt code cũ của bạn) ---
        if len(insertions) >= K:
            for i in range(1, K):
                regret_value += (insertions[i]['cost'] - best_insert['cost'])
        elif len(insertions) > 1:
            for i in range(1, len(insertions)):
                regret_value += (insertions[i]['cost'] - best_insert['cost'])
        # (Nếu len(insertions) == 1, regret_value = 0, ưu tiên thấp nhất)

        all_regret_options.append(
            (regret_value, farm_id, best_insert) # (regret, id, details)
        )

    phase1_time = time.time()
    print(f"[RegretInsert] >>> Đã xong Phase 1 sau {phase1_time - start_time:.2f} giây.")

    # --- PHASE 2: SẮP XẾP (Chạy 1 lần) ---
    # O(N log N)
    print(f"[RegretInsert] ... Bắt đầu Phase 2: Sắp xếp {len(all_regret_options)} lựa chọn...")
    
    # Sắp xếp theo REGRET GIẢM DẦN (reverse=True)
    all_regret_options.sort(key=lambda x: x[0], reverse=True) 
    
    phase2_time = time.time()
    print(f"[RegretInsert] >>> Đã xong Phase 2 sau {phase2_time - phase1_time:.2f} giây.")

    # --- PHASE 3: THỰC HIỆN CHÈN (Chạy N lần) ---
    print(f"[RegretInsert] ... Bắt đầu Phase 3: Thực hiện chèn...")
    
    # (Sử dụng logic "lười" y hệt 'best_insertion' O(N log N))
    # (Cảnh báo: Logic này CÓ THỂ tạo ra giải pháp infeasible, 
    #  như chúng ta đã thảo luận, và cần được xử lý bằng 
    #  "penalty" trong objective_function hoặc "re-check")

    for regret, farm_id, details in all_regret_options:
        if farm_id not in unserved_customers_set:
            continue

        if details['route_idx'] == -1:
            # 🔹 Tạo route mới
            depot, truck_id, shift, route_start_time = details['new_route_details']
            repaired.schedule.append((depot, truck_id, [farm_id],
                                      details['shift'], route_start_time))
        else:
            # 🔹 Chèn vào route có sẵn
            route_idx = details['route_idx']
            pos = details['pos']
            
            if route_idx >= len(repaired.schedule):
                failed_customers.append(farm_id)
                unserved_customers_set.remove(farm_id)
                continue
                
            route_as_list = list(repaired.schedule[route_idx])
            
            if pos > len(route_as_list[2]):
                pos = len(route_as_list[2]) 
                
            route_as_list[2].insert(pos, farm_id)
            repaired.schedule[route_idx] = tuple(route_as_list)
        
        unserved_customers_set.remove(farm_id)

    failed_customers.extend(list(unserved_customers_set))
    if failed_customers:
         print(f"!!! REPAIR (RegretInsert) FAILED: Không thể chèn các khách hàng: {failed_customers}")

    end_time = time.time()
    print(f"[RegretInsert] >>> Hoàn thành. Tổng thời gian: {end_time - start_time:.2f} giây. Lỗi: {len(failed_customers)}")
    
    return repaired, failed_customers
def regret_2_insertion(current, random_state, **kwargs):
    """Hàm bao bọc: Luôn gọi hàm 'k' với k_regret=2"""
    # Bạn phải truyền **kwargs vào để 'unvisited_customers' được đi qua
    return regret_k_insertion(current, random_state, k_regret=2, **kwargs)

def regret_3_insertion(current, random_state, **kwargs):
    """Hàm bao bọc: Luôn gọi hàm 'k' với k_regret=3"""
    return regret_k_insertion(current, random_state, k_regret=3, **kwargs)

def regret_4_insertion(current, random_state, **kwargs):
    """Hàm bao bọc: Luôn gọi hàm 'k' với k_regret=4"""
    return regret_k_insertion(current, random_state, k_regret=4, **kwargs)


def time_shift_repair(current, random_state, **kwargs):
    # PARAMS — bạn có thể tinh chỉnh
    DEFAULT_START_SEARCH_MAX = 240   # tối đa dịch +240 phút (4 giờ) — tùy dữ liệu
    DEFAULT_START_SEARCH_STEP = 15   # bước 15 phút
    WAIT_COST_PER_MIN = 0.2
    TIME_PENALTY = 0.3          
    """
    Repair operator that:
    1) performs an insertion repair (regret or best) to reinsert unvisited_customers
    2) for every route in the repaired schedule, searches for an improved departure time
       (start_time_at_depot) that minimizes route waiting (or route cost).
    Returns repaired_env, failed_customers

    Expected kwargs:
      - unvisited_customers: list of farm IDs to insert
      - base_repair: function to use for insertion (default: regret_insertion)
      - start_search_max: int (minutes) max shift to try (default DEFAULT_START_SEARCH_MAX)
      - start_search_step: int (minutes) step size (default DEFAULT_START_SEARCH_STEP)
      - optimize_by: 'wait' or 'cost' (default 'cost')
      - wait_cost_per_min: float (default WAIT_COST_PER_MIN)
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited = list(kwargs.get('unvisited_customers', []))
    base_repair = kwargs.get('base_repair', regret_k_insertion)  # use your regret_insertion by default
    start_search_max = kwargs.get('start_search_max', DEFAULT_START_SEARCH_MAX)
    start_search_step = kwargs.get('start_search_step', DEFAULT_START_SEARCH_STEP)
    optimize_by = kwargs.get('optimize_by', 'cost')  # or 'wait'
    wait_cost_per_min = kwargs.get('wait_cost_per_min', WAIT_COST_PER_MIN)

    # 1) First, run the base repair to reinsert visits (this yields a schedule)
    kwargs.pop('unvisited_customers', None)

    # Gọi base repair (regret/best insertion)
    repaired, failed_customers = base_repair(
        repaired, random_state, unvisited_customers=unvisited, **kwargs
    )

    # If nothing was inserted and there are failures, return early
    if failed_customers:
        return repaired, failed_customers

    # 2) For each route, search candidate start times (0 .. start_search_max) with step
    new_schedule = []
    for route_idx, route in enumerate(repaired.schedule):
        # Route format before: (depot_idx, truck_id, customer_list, shift)
        # We'll support both formats: if route already has 5-tuple, keep its start as baseline
        if len(route) == 5:
            depot_idx, truck_id, cust_list, shift, existing_start = route
            baseline_start = int(existing_start)
        else:
            depot_idx, truck_id, cust_list, shift = route
            baseline_start = 0

        # If route empty or INTER-FACTORY => keep as is (no start optimization)
        if not cust_list or shift == 'INTER-FACTORY':
            new_schedule.append(route if len(route) == 5 else (depot_idx, truck_id, cust_list, shift, baseline_start))
            continue

        truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
        if truck_info is None:
            # keep original
            new_schedule.append(route if len(route) == 5 else (depot_idx, truck_id, cust_list, shift, baseline_start))
            continue

        best_metric = float('inf')
        best_start = baseline_start

        # candidate_start iterate from 0 up to start_search_max (inclusive)
        # optionally you could allow negative shifts (start earlier) if model supports it
        for s in range(0, start_search_max + 1, start_search_step):
            finish_time, is_feasible, total_dist, total_wait, opt_start, time_penalty, capacity_penalty = _calculate_route_schedule_and_feasibility(
                depot_idx, cust_list, shift, s, problem_instance, truck_info
            )
            if not is_feasible:
                continue

            if optimize_by == 'wait':
                metric = total_wait
            else:  # 'cost'
                # compute route variable cost
                var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                    (truck_info['type'], truck_info['region']), 1.0
                )
                metric = total_dist * var_cost_per_km + total_wait * wait_cost_per_min + time_penalty*TIME_PENALTY

            if metric < best_metric - 1e-6:
                best_metric = metric
                best_start = s

        # Append route with chosen start_time (extend tuple to length 5)
        new_schedule.append((depot_idx, truck_id, cust_list, shift, best_start))

    # Replace repaired schedule with new_schedule
    repaired.schedule = new_schedule

    return repaired, failed_customers
#! Mấy repairs dưới chưa đổi theo yếu tố multi-trip, cần sửa lại sau


"""
def cheapest_feasible_insertion(current, random_state, **kwargs):
    # Logic của cheapest_feasible rất giống best_insertion, chỉ khác ở cách lặp
    # Thay vì tìm vị trí tốt nhất cho tất cả rồi chọn 1, nó tìm và chèn ngay lập tức
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited_customers = list(kwargs['unvisited_customers'])

    # Lặp lại cho đến khi không còn khách hàng nào để chèn
    inserted_in_this_pass = True
    while inserted_in_this_pass:
        inserted_in_this_pass = False
        best_cost_this_pass = float('inf')
        best_details_this_pass = None
        farm_to_insert_this_pass = None
        
        if not unvisited_customers: break

        for farm_id in unvisited_customers:
            insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
            if insertions:
                best_for_farm = min(insertions, key=lambda x: x[0])
                if best_for_farm[0] < best_cost_this_pass:
                    best_cost_this_pass = best_for_farm[0]
                    best_details_this_pass = best_for_farm
                    farm_to_insert_this_pass = farm_id
        
        if farm_to_insert_this_pass:
            cost, day_idx, route_idx, pos, shift, truck_id = best_details_this_pass
            repaired.schedule[day_idx][route_idx][2].insert(pos, farm_to_insert_this_pass)
            unvisited_customers.remove(farm_to_insert_this_pass)
            inserted_in_this_pass = True
            
    # Xử lý các khách hàng còn lại không thể chèn vào tuyến có sẵn
    for farm_id in unvisited_customers:
        new_route_info = _create_new_route_for_farm(farm_id, problem_instance)
        if new_route_info:
            cost, depot_idx, truck_id, cust_list = new_route_info
            random_day = random_state.choice(list(repaired.schedule.keys()))
            repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])

    return repaired


def random_feasible_insertion(current, random_state, **kwargs):
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited_customers = list(kwargs['unvisited_customers'])
    random_state.shuffle(unvisited_customers)

    for farm_id in unvisited_customers:
        insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
        
        if insertions:
            # Chọn một vị trí chèn ngẫu nhiên từ các vị trí khả thi
            chosen_insertion = random_state.choice(insertions)
            cost, day_idx, route_idx, pos, shift, truck_id = chosen_insertion
            repaired.schedule[day_idx][route_idx][2].insert(pos, farm_id)
        else:
            # Nếu không chèn được, tạo tuyến mới
            new_route_info = _create_new_route_for_farm(farm_id, problem_instance)
            if new_route_info:
                cost, depot_idx, truck_id, cust_list = new_route_info
                random_day = random_state.choice(list(repaired.schedule.keys()))
                repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])
                
    return repaired

def regret_insertion(current, random_state, **kwargs):

    return _regret_k_insertion(current, random_state, k_regret=2, **kwargs)"""