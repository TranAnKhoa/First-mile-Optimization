import copy
import random
import numpy as np
import re
from collections import defaultdict
import itertools

def _clean_base_id(fid):
    """Remove suffixes like _onfly, _part, _d<number> to get the real farm id."""
    # Nếu fid không phải str (có thể là int), trả về thẳng (không cần xử lý suffix)
    if not isinstance(fid, str):
        return fid
    # Dùng regex split để loại bỏ các hậu tố thường dùng khi tách farm (ví dụ: '_onfly_part1', '_d2'...)
    # re.split(r'(...pattern...)', fid)[0] trả về phần trước phần match — tức là id "gốc"
    # Pattern giải thích:
    #   _onfly.*         : bắt đầu bằng '_onfly' và mọi thứ theo sau
    #   |_fallback_part.*: hoặc bắt đầu bằng '_fallback_part' và mọi thứ theo sau
    #   |_part.*         : hoặc '_part' và mọi thứ theo sau
    #   |_d\d+           : hoặc '_d' theo sau là ít nhất một chữ số (phần định danh chia)
    return re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', fid)[0]

def _get_farm_info(farm_id, problem_instance):
    """Hàm "thông dịch" ID, trả về thông tin chính xác cho cả farm thật và ảo."""
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']
    virtual_map = problem_instance.get('virtual_split_farms', {})
    farms = problem_instance['farms']
    
    base_id = _clean_base_id(farm_id)
    
    try:
        farm_idx = farm_id_to_idx_map[base_id]
    except KeyError:
        try:
            farm_idx = farm_id_to_idx_map[int(base_id)]
        except (KeyError, ValueError):
            raise KeyError(f"RepairOp: Không thể tìm thấy Farm ID '{base_id}' (từ '{farm_id}') trong map.")
            
    farm_details = farms[farm_idx]
    
    if farm_id in virtual_map:
        demand = virtual_map[farm_id]['portion']
    else:
        demand = farm_details['demand']
        
    return farm_idx, farm_details, demand

def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

# ==============================================================================
# HÀM TÍNH TOÁN NÂNG CAO (VỚI TỐI ƯU HÓA THỜI GIAN XUẤT PHÁT)
# ==============================================================================

def _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """
    ## FINAL VERSION ##
    Tính toán lịch trình, kiểm tra feasibility VÀ TỐI ƯU HÓA THỜI GIAN XUẤT PHÁT một cách chính xác.
    Trả về: (finish_time, is_feasible, total_dist, total_wait, optimal_start_time)
    """
    if not customer_list:
        return start_time_at_depot, True, 0, 0, start_time_at_depot

    # --- Các biến và hàm con (không đổi) ---
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    depot_end_time = 1440
    truck_name = truck_info['type']
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
    virtual_map = problem_instance.get('virtual_split_farms', {})

    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid)
        try: base_idx = farm_id_to_idx[base_id_str]
        except KeyError: base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        if isinstance(fid, str) and fid in virtual_map:
            return base_idx, virtual_map[fid]['portion'], base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

    # === BƯỚC 1: MÔ PHỎNG LẦN ĐẦU VỚI THỜI GIAN XUẤT PHÁT SỚM NHẤT ===
    timeline_sim = []
    current_time_sim1 = start_time_at_depot # Bắt đầu từ thời gian sớm nhất có thể

    # Depot -> Farm 1
    idx, demand, params, tw = _resolve_farm(customer_list[0])
    travel_time = depot_farm_dist[depot_idx, idx] / velocity
    arrival = current_time_sim1 + travel_time
    start_tw, end_tw = tw[shift]
    if arrival > end_tw: return -1, False, -1, -1, -1
    service_start = max(arrival, start_tw)
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    current_time_sim1 = service_start + service_duration
    timeline_sim.append({'arrival': arrival, 'start': service_start})

    # Giữa các farm
    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i+1])
        travel_time = dist_matrix[from_idx, to_idx] / velocity
        arrival = current_time_sim1 + travel_time
        start_tw, end_tw = to_tw[shift]
        if arrival > end_tw: return -1, False, -1, -1, -1
        service_start = max(arrival, start_tw)
        service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
        current_time_sim1 = service_start + service_duration
        timeline_sim.append({'arrival': arrival, 'start': service_start})
    
    # Quay về depot (để kiểm tra ràng buộc thời gian làm việc)
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity
    finish_time_sim1 = current_time_sim1 + travel_time_back
    if finish_time_sim1 > depot_end_time: return -1, False, -1, -1, -1

    # === BƯỚC 2: TÍNH TOÁN THỜI GIAN XUẤT PHÁT TỐI ƯU ===
    optimal_start_time = start_time_at_depot

    # === BƯỚC 3: MÔ PHỎNG LẠI VỚI THỜI GIAN TỐI ƯU ĐỂ LẤY KẾT QUẢ CUỐI CÙNG ===
    total_dist = 0
    total_wait = 0
    current_time_final = optimal_start_time # Bắt đầu từ thời gian đã được tối ưu hóa

    # Lặp lại logic tính toán một cách cẩn thận để có được các giá trị cuối cùng
    idx, demand, params, tw = _resolve_farm(customer_list[0])
    travel_dist = depot_farm_dist[depot_idx, idx]; total_dist += travel_dist
    travel_time = travel_dist / velocity; arrival = current_time_final + travel_time
    start_tw, _ = tw[shift]; wait_time = max(0, start_tw - arrival); total_wait += wait_time
    service_start = arrival + wait_time
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    current_time_final = service_start + service_duration

    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i+1])
        travel_dist = dist_matrix[from_idx, to_idx]; total_dist += travel_dist
        travel_time = travel_dist / velocity; arrival = current_time_final + travel_time
        start_tw, _ = to_tw[shift]; wait_time = max(0, start_tw - arrival); total_wait += wait_time
        service_start = arrival + wait_time
        service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
        current_time_final = service_start + service_duration
    
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_dist_back = depot_farm_dist[depot_idx, last_idx]; total_dist += travel_dist_back
    travel_time_back = travel_dist_back / velocity
    finish_time_final = current_time_final + travel_time_back
        
    return finish_time_final, True, total_dist, total_wait, optimal_start_time
# ==============================================================================
# HÀM KIỂM TRA TÍNH KHẢ THI (FEASIBILITY CHECKLIST)
# ==============================================================================

def _check_insertion_feasibility(problem_instance, route_info, insert_pos, farm_id_to_insert, shift, start_time=None):
    """Thực hiện The Feasibility Checklist và tính toán chi phí tăng thêm."""
    
    # Giải nén route_info an toàn (4 hoặc 5 phần tử)
    if len(route_info) == 5:
        depot_idx, truck_id, customer_list, shift_in_route, route_start_time = route_info
    else:
        depot_idx, truck_id, customer_list, shift_in_route = route_info
        route_start_time = 0

    # Nếu start_time được truyền ngoài thì ưu tiên, ngược lại dùng trong route
    if start_time is None:
        start_time = route_start_time

    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
    if not truck_info:
        return False, float('inf'), -1

    WAIT_COST_PER_MIN = 0.2
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
        (truck_info['type'], truck_info['region']), 1.0
    )

    # --- Accessibility + capacity check ---
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    truck_type_idx = type_to_idx.get(truck_info['type'])
    if truck_type_idx is None:
        return False, float('inf'), -1

    _, farm_details, farm_demand = _get_farm_info(farm_id_to_insert, problem_instance)
    farm_access = farm_details.get('accessibility')
    if farm_access is None or len(farm_access) <= truck_type_idx or farm_access[truck_type_idx] != 1:
        return False, float('inf'), -1

    current_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in customer_list)
    if current_load + farm_demand > truck_info['capacity']:
        return False, float('inf'), -1

    # --- Compute old route cost ---
    old_total_cost = 0
    if customer_list:
        _, is_feasible_old, old_dist, old_wait, _ = _calculate_route_schedule_and_feasibility(
            depot_idx, customer_list, shift_in_route, start_time, problem_instance, truck_info=truck_info
        )
        if not is_feasible_old:
            return False, float('inf'), -1
        old_total_cost = old_dist * var_cost_per_km + old_wait * WAIT_COST_PER_MIN

    # --- Compute new route cost after inserting this farm ---
    test_route = customer_list[:insert_pos] + [farm_id_to_insert] + customer_list[insert_pos:]
    _, is_feasible_new, new_dist, new_wait, _ = _calculate_route_schedule_and_feasibility(
        depot_idx, test_route, shift_in_route, start_time, problem_instance, truck_info=truck_info
    )

    if not is_feasible_new:
        return False, float('inf'), -1

    new_total_cost = new_dist * var_cost_per_km + new_wait * WAIT_COST_PER_MIN
    cost_increase = new_total_cost - old_total_cost
        
    return True, cost_increase, new_total_cost

def _find_all_inserts_for_visit(schedule_list, visit_id, problem_instance):
    """
    TIME-AWARE version:
    - Tìm tất cả vị trí chèn khả thi vào các route hiện có (với start_time đúng).
    - Khi tạo route mới, thử nhiều depot và truck, chọn phương án có cost nhỏ nhất.
    """
    all_insertions = []

    # --- 1) Chèn vào các tuyến hiện có ---
    for route_idx, route_info in enumerate(schedule_list):
        if route_info[3] == 'INTER-FACTORY':
            continue
        
        # Đọc start_time đúng từ route (nếu có)
        start_time = route_info[4]

        for insert_pos in range(len(route_info[2]) + 1):
            is_feasible, cost_increase, _ = _check_insertion_feasibility(
                problem_instance, route_info, insert_pos, visit_id, route_info[3], start_time=start_time
            )
            if is_feasible:
                all_insertions.append({
                    'cost': cost_increase,
                    'route_idx': route_idx,
                    'pos': insert_pos,
                    'shift': route_info[3],
                    'new_route_details': None
                })

    # --- 2) Tạo route mới (nâng cấp logic) ---
    farm_idx, farm_details, farm_demand = _get_farm_info(visit_id, problem_instance)
    facilities = problem_instance['facilities']
    dist_depots_farms = problem_instance['distance_depots_farms']
    available_trucks = problem_instance['fleet']['available_trucks']
    WAIT_COST_PER_MIN = 0.2

    num_depots = dist_depots_farms.shape[0]
    K_NEAREST_DEPOTS = min(3, num_depots)
    depot_dists = list(enumerate(dist_depots_farms[:, farm_idx]))
    depot_dists_sorted = sorted(depot_dists, key=lambda x: x[1])[:K_NEAREST_DEPOTS]
    candidate_depots = [d for d, _ in depot_dists_sorted]

    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}

    for depot_idx in candidate_depots:
        depot_region = facilities[depot_idx].get('region', None)
        depot_access = facilities[depot_idx].get('accessibility')

        suitable_trucks = []
        for truck in available_trucks:
            if truck.get('region') != depot_region or truck['capacity'] < farm_demand:
                continue
            truck_type_idx = type_to_idx.get(truck['type'])
            if truck_type_idx is None:
                continue
            farm_access = farm_details.get('accessibility')
            depot_ok = (depot_access is None or (len(depot_access) > truck_type_idx and depot_access[truck_type_idx] == 1))
            farm_ok = (farm_access is None or (len(farm_access) > truck_type_idx and farm_access[truck_type_idx] == 1))
            if depot_ok and farm_ok:
                suitable_trucks.append(truck)

        for truck in suitable_trucks:
            var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                (truck['type'], truck['region']), 1.0
            )
            for shift in ['AM', 'PM']:
                _, is_feasible, new_dist, new_wait, optimal_start_time = _calculate_route_schedule_and_feasibility(
                    depot_idx, [visit_id], shift, 0, problem_instance, truck
                )
                if not is_feasible:
                    continue
                cost_of_new_route = new_dist * var_cost_per_km + new_wait * WAIT_COST_PER_MIN
                all_insertions.append({
                'cost': cost_of_new_route,
                'route_idx': -1,
                'pos': 0,
                'shift': shift,
                'new_route_details': (depot_idx, truck['id'], shift, 0)  # thêm shift và start_time=0
            })

    all_insertions.sort(key=lambda x: x['cost'])
    return all_insertions




#!  DESTROY 

def _remove_customers_from_schedule(schedule, customers_to_remove):
    """
    Xóa danh sách khách hàng khỏi schedule hiện tại.
    Mỗi route_info bây giờ có 5 phần tử: (depot_idx, truck_id, customer_list, shift, start_time)
    """
    new_schedule = []
    for idx, route_info in enumerate(schedule):
        if len(route_info) != 5:
            print(f"⚠️ Route {idx} có {len(route_info)} phần tử: {route_info}")
    for route_info in schedule:
        depot_idx, truck_id, customer_list, shift, start_time = route_info
        
        # Giữ lại các khách hàng không bị xóa
        updated_customer_list = [c for c in customer_list if c not in customers_to_remove]
        
        if updated_customer_list:
            new_schedule.append((depot_idx, truck_id, updated_customer_list, shift, start_time))
    
    return new_schedule



#! Local Search

def get_route_cost(problem_instance, route_info):
    """
    Tính toán tổng chi phí (di chuyển + chờ) của một tuyến đường duy nhất.
    """
    depot_idx, truck_id, customer_list, shift, start_time = route_info
    
    if not customer_list:
        return 0

    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
    if not truck_info:
        return float('inf')

    WAIT_COST_PER_MIN = 0.2
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
        (truck_info['type'], truck_info['region']), 1.0
    )

    _, is_feasible, total_dist, total_wait, _ = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, 0, problem_instance, truck_info
    )

    if not is_feasible:
        return float('inf')

    return (total_dist * var_cost_per_km) + (total_wait * WAIT_COST_PER_MIN)

