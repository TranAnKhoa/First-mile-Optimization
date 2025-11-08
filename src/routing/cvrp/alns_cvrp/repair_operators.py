import copy
import random
import numpy as np
import re
from collections import defaultdict
import itertools
from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id, _check_insertion_feasibility
# ==============================================================================
# HÀM TIỆN ÍCH CHUNG (Không thay đổi)
# ==============================================================================

# --- HÀM TRỢ GIÚP: TÌM VỊ TRÍ TỐT NHẤT CHO MỘT FARM ---

# ==============================================================================
# TOÁN TỬ SỬA CHỮA CHÍNH (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================
def _find_all_inserts_for_visit(schedule_list, visit_id, problem_instance):
    """
    ## SIMPLIFIED ##
    Tìm TẤT CẢ các vị trí chèn khả thi cho một farm_id và trả về danh sách đã sắp xếp.
    """
    all_insertions = []

    # 1. Thử chèn vào các tuyến đường hiện có
    for route_idx, route_info in enumerate(schedule_list):
        if route_info[3] == 'INTER-FACTORY': continue
        
        for insert_pos in range(len(route_info[2]) + 1):
            is_feasible, cost_increase, _ = _check_insertion_feasibility(
                problem_instance, route_info, insert_pos, visit_id, route_info[3], start_time=0
            )
            if is_feasible:
                all_insertions.append({
                    'cost': cost_increase, 'route_idx': route_idx, 'pos': insert_pos,
                    'shift': route_info[3], 'new_route_details': None
                })

    # 2. Thử tạo một tuyến đường mới
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
        depot_access = facilities[closest_depot_idx].get('accessibility')
        farm_access = farm_details.get('accessibility')
        depot_ok = (depot_access is None or (len(depot_access) > truck_type_idx and depot_access[truck_type_idx] == 1))
        farm_ok = (farm_access is None or (len(farm_access) > truck_type_idx and farm_access[truck_type_idx] == 1))
        if depot_ok and farm_ok: suitable_trucks.append(truck)

    if suitable_trucks:
        best_truck_for_new_route = min(suitable_trucks, key=lambda t: t['capacity'])
        WAIT_COST_PER_MIN = 0.2
        var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
            (best_truck_for_new_route['type'], best_truck_for_new_route['region']), 1.0)
        
        for shift in ['AM', 'PM']:
            _, is_feasible, new_dist, new_wait, _ = _calculate_route_schedule_and_feasibility(
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
    TIME-AWARE VERSION
    Chèn lại các farm_id vào vị trí có chi phí thấp nhất, có xét đến start_time.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers = list(kwargs['unvisited_customers'])
    failed_customers = []

    while unserved_customers:
        best_customer_to_insert = None
        best_insertion_details = None
        min_insertion_cost = float('inf')

        # --- Xét tất cả farm để tìm vị trí chèn rẻ nhất ---
        for farm_id in unserved_customers:
            insertions = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance)
            if not insertions:
                continue
            best_insert_for_this_farm = insertions[0]
            if best_insert_for_this_farm['cost'] < min_insertion_cost:
                min_insertion_cost = best_insert_for_this_farm['cost']
                best_customer_to_insert = farm_id
                best_insertion_details = best_insert_for_this_farm

        # --- Thực hiện chèn ---new_route_info
        if best_customer_to_insert:
            if best_insertion_details['route_idx'] == -1:
                # 🔹 Tạo route mới
                depot, truck_id, shift, start_time = best_insertion_details['new_route_details']
                repaired.schedule.append((depot, truck_id, [best_customer_to_insert],
                                          best_insertion_details['shift'], start_time))
            else:
                # 🔹 Chèn vào route có sẵn
                route_idx = best_insertion_details['route_idx']
                pos = best_insertion_details['pos']
                route_as_list = list(repaired.schedule[route_idx])
                route_as_list[2].insert(pos, best_customer_to_insert)
                repaired.schedule[route_idx] = tuple(route_as_list)
            
            unserved_customers.remove(best_customer_to_insert)
        else:
            failed_customers = unserved_customers
            print(f"!!! REPAIR FAILED: Không thể chèn các khách hàng: {failed_customers}")
            break

    return repaired, failed_customers


def regret_insertion(current, random_state, **kwargs):
    """
    TIME-AWARE VERSION
    Chèn lại các farm_id có "regret" cao nhất (tức là nếu không chèn sớm thì sau này tốn nhiều chi phí hơn).
    Có xét đến start_time của từng route.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers = list(kwargs['unvisited_customers'])
    failed_customers = []
    K = kwargs.get('k_regret', 3)  # Sử dụng K=3 mặc định

    while unserved_customers:
        customer_regret_options = []

        # --- 1) Tính regret cho mỗi farm ---
        for farm_id in unserved_customers:
            insertions = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance)
            if not insertions:
                continue

            best_insert = insertions[0]
            regret_value = 0

            # Nếu có nhiều hơn 1 lựa chọn, tính regret giữa các lựa chọn đầu
            if len(insertions) >= K:
                for i in range(1, K):
                    regret_value += (insertions[i]['cost'] - best_insert['cost'])
            elif len(insertions) > 1:
                for i in range(1, len(insertions)):
                    regret_value += (insertions[i]['cost'] - best_insert['cost'])

            customer_regret_options.append({
                'regret': regret_value,
                'customer': farm_id,
                'best_insertion': best_insert
            })

        # --- 2) Nếu không còn farm khả thi ---
        if not customer_regret_options:
            failed_customers = unserved_customers
            print(f"!!! REPAIR FAILED: Không thể chèn các khách hàng còn lại: {failed_customers}")
            break

        # --- 3) Chọn farm có regret cao nhất ---
        best_regret_option = max(customer_regret_options, key=lambda x: x['regret'])
        customer_to_insert = best_regret_option['customer']
        insertion_details = best_regret_option['best_insertion']

        # --- 4) Thực hiện chèn ---
        if insertion_details['route_idx'] == -1:
            # 🔹 Tạo route mới
            depot, truck_id, shift, start_time = insertion_details['new_route_details']
            repaired.schedule.append((depot, truck_id, [customer_to_insert],
                                      insertion_details['shift'], start_time))
        else:
            # 🔹 Chèn vào route có sẵn
            route_idx = insertion_details['route_idx']
            pos = insertion_details['pos']
            route_as_list = list(repaired.schedule[route_idx])
            route_as_list[2].insert(pos, customer_to_insert)
            repaired.schedule[route_idx] = tuple(route_as_list)

        unserved_customers.remove(customer_to_insert)

    return repaired, failed_customers



def time_shift_repair(current, random_state, **kwargs):
    # PARAMS — bạn có thể tinh chỉnh
    DEFAULT_START_SEARCH_MAX = 240   # tối đa dịch +240 phút (4 giờ) — tùy dữ liệu
    DEFAULT_START_SEARCH_STEP = 15   # bước 15 phút
    WAIT_COST_PER_MIN = 0.2          
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
    base_repair = kwargs.get('base_repair', regret_insertion)  # use your regret_insertion by default
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
            baseline_start = int(existing_start or 0)
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
            finish_time, is_feasible, total_dist, total_wait, opt_start = _calculate_route_schedule_and_feasibility(
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
                metric = total_dist * var_cost_per_km + total_wait * wait_cost_per_min

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