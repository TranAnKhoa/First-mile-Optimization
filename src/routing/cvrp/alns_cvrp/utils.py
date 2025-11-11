import copy
import random
import numpy as np
import re
from collections import defaultdict
import itertools
import math


# ==============================================================================
# HÀM CHUNG
# ==============================================================================

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
def _get_service_time(farm_details, demand):
    """Học từ logic 'calculate_cost_repair' của bạn."""
    params = farm_details['service_time_params']
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    return service_duration

def _get_dist_and_time(from_loc_id, to_loc_id, from_is_depot, to_is_depot, truck_info, problem_instance):
    """Học từ logic 'calculate_cost_repair' và 'find_all_inserts'."""
    
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    
    # Lấy velocity từ logic của 'calculate_cost_repair'
    truck_name = truck_info['type']
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
    
    dist = 0
    if from_is_depot and not to_is_depot:
        # from_loc_id là depot_idx, to_loc_id là farm_idx
        dist = depot_farm_dist[from_loc_id, to_loc_id]
    elif not from_is_depot and to_is_depot:
        # from_loc_id là farm_idx, to_loc_id là depot_idx
        dist = depot_farm_dist[to_loc_id, from_loc_id]
    elif not from_is_depot and not to_is_depot:
        # from_loc_id là farm_idx, to_loc_id là farm_idx
        dist = dist_matrix[from_loc_id, to_loc_id]
    # else: (Depot -> Depot) dist = 0
        
    travel_time = dist / velocity
    return dist, travel_time

def _check_accessibility(truck_info, farm_details, depot_details):
    """Học từ logic 'find_all_inserts' (phần tạo tuyến mới)."""
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    truck_type_idx = type_to_idx.get(truck_info['type'])
    if truck_type_idx is None:
        return False # Loại xe không xác định

    # Kiểm tra Farm
    farm_access = farm_details.get('accessibility')
    farm_ok = (farm_access is None or (len(farm_access) > truck_type_idx and farm_access[truck_type_idx] == 1))
    if not farm_ok:
        return False
        
    # Kiểm tra Depot (nếu được cung cấp)
    if depot_details:
        depot_access = depot_details.get('accessibility')
        depot_ok = (depot_access is None or (len(depot_access) > truck_type_idx and depot_access[truck_type_idx] == 1))
        if not depot_ok:
            return False
            
    return True

def _get_shift_end_time(shift, problem_instance):
    """Học từ 'time_shift_repair' (giả định cấu trúc này tồn tại)."""
    # (GIẢ ĐỊNH) - Bạn cần xác nhận cấu trúc này
    # Nếu không, hãy thay thế bằng logic lấy end-time của bạn (ví dụ: 186?)
    shift_info = problem_instance.get('shifts', {}).get(shift, {'end': 1900})
    return shift_info['end']

def _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """ 
    ## FINAL VERSION (Tối ưu 1 vòng lặp) ##
    Tính toán LỊCH TRÌNH và TẢI TRỌNG chỉ trong MỘT vòng lặp.
    
    Trả về 7 giá trị, tách riêng (time_penalty) và (capacity_VIOLATION):
    (finish_time, is_feasible, total_dist, total_wait, optimal_start_time, time_penalty, capacity_violation)
    """
    
    # === BƯỚC 0: ROUTE RỖNG ===
    if not customer_list:
        # (finish, feasible, dist, wait, start, time_penalty, cap_violation)
        return start_time_at_depot, True, 0, 0, start_time_at_depot, 0, 0

    # === BƯỚC 1: KHỞI TẠO BIẾN ===
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    
    try:
        shift_end_time = problem_instance['shifts'][shift]['end']
    except (KeyError, TypeError):
        shift_end_time = 1900 # Fallback của bạn
            
    truck_capacity = truck_info.get('capacity', float('inf')) 
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    virtual_map = problem_instance.get('virtual_split_farms', {})

    # (Hàm _resolve_farm không đổi)
    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid)
        try: base_idx = farm_id_to_idx[base_id_str]
        except KeyError: base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        if isinstance(fid, str) and fid in virtual_map:
            return base_idx, virtual_map[fid]['portion'], base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

            
    # === BƯỚC 2: MÔ PHỎNG (TÍCH HỢP TÍNH TIME VÀ DEMAND) ===
    
    total_dist = 0
    total_wait = 0
    time_penalty = 0.0
    total_demand = 0.0     # <-- 1. Khởi tạo total_demand
    current_time_final = start_time_at_depot 

    try:
        # ---- Xử lý khách hàng đầu tiên (Depot -> C1) ----
        idx, demand, params, tw = _resolve_farm(customer_list[0])
        total_demand += demand # <-- 2. TÍNH DEMAND C1
        
        travel_dist = depot_farm_dist[depot_idx, idx]; total_dist += travel_dist
        travel_time = travel_dist / velocity; arrival = current_time_final + travel_time
        start_tw, end_tw = tw[shift] 
        wait_time = max(0, start_tw - arrival); total_wait += wait_time
        service_start = arrival + wait_time
        
        if service_start > end_tw + 1e-6: 
            time_penalty += (service_start - end_tw)
        
        service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
        current_time_final = service_start + service_duration

        # ---- Xử lý các khách hàng ở giữa (C(i) -> C(i+1)) ----
        for i in range(len(customer_list) - 1):
            from_idx, _, _, _ = _resolve_farm(customer_list[i])
            to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i+1])
            
            total_demand += to_demand # <-- 3. TÍNH DEMAND C(i+1)
            
            travel_dist = dist_matrix[from_idx, to_idx]; total_dist += travel_dist
            travel_time = travel_dist / velocity
            arrival = current_time_final + travel_time
            
            start_tw, end_tw = to_tw[shift] 
            wait_time = max(0, start_tw - arrival); total_wait += wait_time
            service_start = arrival + wait_time
            
            if service_start > end_tw + 1e-6:
                time_penalty += (service_start - end_tw)
                
            service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
            current_time_final = service_start + service_duration

        # ---- Xử lý quay về Depot (CLast -> Depot) ----
        last_idx, _, _, _ = _resolve_farm(customer_list[-1])
        travel_dist_back = depot_farm_dist[depot_idx, last_idx]; total_dist += travel_dist_back
        travel_time_back = travel_dist_back / velocity
        finish_time_final = current_time_final + travel_time_back
        
        if finish_time_final > shift_end_time + 1e-6:
             time_penalty += (finish_time_final - shift_end_time)
    
    except Exception as e:
        # Bắt lỗi nếu _resolve_farm thất bại (ví dụ: farm_id không tồn tại)
        print(f"LỖI NGHIÊM TRỌNG khi mô phỏng route: {e}. Trả về vi phạm lớn.")
        # Trả về chi phí vô hạn để ALNS tự động hủy route này
        return np.inf, True, np.inf, 0, start_time_at_depot, 9999999, 9999999
        
    # === BƯỚC 3: TÍNH TOÁN LƯỢNG VI PHẠM (VIOLATION) ===
    
    # Tính lượng vi phạm capacity (Đúng theo ý bạn, chỉ trả về lượng vi phạm)
    capacity_violation = 0.0
    if total_demand > truck_capacity:
        capacity_violation = total_demand - truck_capacity 
    
    is_feasible = True # Luôn là True vì ta dùng soft constraints

    # === BƯỚC 4: TRẢ VỀ 7 GIÁ TRỊ ===
    return finish_time_final, is_feasible, total_dist, total_wait, start_time_at_depot, time_penalty, capacity_violation
# ==============================================================================
# HÀM Repair 
# ==============================================================================

def _calculate_route_schedule_WITH_SLACK(depot_idx, customer_list, shift, 
                                           start_time_at_depot, problem_instance, truck_info):
    """
    ## PHIÊN BẢN NÂNG CẤP O(K) ##
    Tính toán lịch trình, chi phí, VÀ 'forward_slack'.
    Trả về: (is_feasible, total_dist, total_wait, detailed_schedule)
    """
    
    shift_end_time = _get_shift_end_time(shift, problem_instance)

    if not customer_list:
        depot_schedule = {
            'loc_id': depot_idx, 'loc_is_depot': True,
            'arrival': start_time_at_depot, 'wait': 0,
            'departure': start_time_at_depot, 'tw_close': shift_end_time, 
            'forward_slack': shift_end_time - start_time_at_depot
        }
        return True, 0, 0, [depot_schedule, depot_schedule]

    current_time = start_time_at_depot
    current_loc_id = depot_idx
    current_loc_is_depot = True
    total_dist = 0
    total_wait = 0
    
    detailed_schedule = [] 

    # Thêm điểm Depot vào đầu lịch trình
    detailed_schedule.append({
        'loc_id': depot_idx, 'loc_is_depot': True,
        'arrival': current_time, 'wait': 0, 'departure': current_time,
        'tw_close': shift_end_time 
    })

    # 2. Mô phỏng tiến (Forward Simulation)
    for cust_id in customer_list:
        loc_idx, farm_details, demand = _get_farm_info(cust_id, problem_instance)
        
        tw_open, tw_close = farm_details['time_windows'][shift]
        service_duration = _get_service_time(farm_details, demand)
        
        dist, travel_time = _get_dist_and_time(
            current_loc_id, loc_idx, current_loc_is_depot, False, 
            truck_info, problem_instance
        )
        
        total_dist += dist
        arrival_time = current_time + travel_time
        
        if arrival_time > tw_close + 1e-6: # Thêm epsilon
            return False, 0, 0, [] 

        wait_time = max(0, tw_open - arrival_time)
        total_wait += wait_time
        
        departure_time = arrival_time + wait_time + service_duration
        
        detailed_schedule.append({
            'loc_id': loc_idx, 'loc_is_depot': False,
            'arrival': arrival_time, 'wait': wait_time,
            'departure': departure_time, 'tw_close': tw_close
        })
        
        current_time = departure_time
        current_loc_id = loc_idx
        current_loc_is_depot = False

    # 3. Quay về Depot
    dist, travel_time_back = _get_dist_and_time(
        current_loc_id, depot_idx, current_loc_is_depot, True, 
        truck_info, problem_instance
    )
    
    total_dist += dist
    arrival_at_depot = current_time + travel_time_back
    
    if arrival_at_depot > shift_end_time + 1e-6: # Thêm epsilon
         return False, 0, 0, [] 

    detailed_schedule.append({
        'loc_id': depot_idx, 'loc_is_depot': True,
        'arrival': arrival_at_depot, 'wait': 0,
        'departure': arrival_at_depot, 'tw_close': shift_end_time
    })
    
    # 4. TÍNH TOÁN FORWARD SLACK (Mô phỏng ngược)
    last_slack = shift_end_time - detailed_schedule[-1]['arrival']
    detailed_schedule[-1]['forward_slack'] = last_slack

    for i in range(len(detailed_schedule) - 2, -1, -1):
        current_node = detailed_schedule[i]
        next_node = detailed_schedule[i+1]
        
        slack_via_next = next_node['forward_slack'] + next_node['wait']
        slack_via_tw = current_node['tw_close'] - current_node['arrival']
        
        current_node['forward_slack'] = min(slack_via_next, slack_via_tw)

    return True, total_dist, total_wait, detailed_schedule

def _check_insertion_delta(problem_instance, route_info, original_schedule, 
                           insert_pos, farm_id_to_insert, 
                           truck_info, current_load):
    """
    KIỂM TRA CHÈN SIÊU NHANH (O(1)) - Tính toán Delta.
    """
    
    depot_idx, truck_id, customer_list, shift, start_time_at_depot = route_info
    
    # 1. Lấy thông tin farm mới (Farm X)
    try:
        loc_X_idx, farm_X_details, demand_X = _get_farm_info(farm_id_to_insert, problem_instance)
    except Exception as e:
        return False, float('inf')
        
    tw_X_open, tw_X_close = farm_X_details['time_windows'][shift]
    service_X = _get_service_time(farm_X_details, demand_X)

    # 2. KIỂM TRA RÀNG BUỘC CỨNG (Tải trọng, Accessibility)
    # 2a. Tải trọng
    if current_load + demand_X > truck_info['capacity']:
        return False, float('inf') # Lỗi quá tải
    
    # 2b. Accessibility (Không kiểm tra depot, vì depot đã OK cho tuyến này)
    if not _check_accessibility(truck_info, farm_X_details, depot_details=None):
        return False, float('inf') # Lỗi accessibility

    # 3. Lấy các điểm lân cận (A và B)
    node_A = original_schedule[insert_pos]
    node_B = original_schedule[insert_pos + 1]
    
    loc_A_id = node_A['loc_id']
    loc_A_is_depot = node_A['loc_is_depot']
    loc_B_id = node_B['loc_id']
    loc_B_is_depot = node_B['loc_is_depot']

    # 4. TÍNH TOÁN THỜI GIAN DELTA (O(1))
    # (A -> X)
    dist_A_X, travel_A_X = _get_dist_and_time(
        loc_A_id, loc_X_idx, loc_A_is_depot, False, 
        truck_info, problem_instance
    )
    arrival_at_X = node_A['departure'] + travel_A_X
    
    if arrival_at_X > tw_X_close + 1e-6:
        return False, float('inf') 
        
    wait_at_X = max(0, tw_X_open - arrival_at_X)
    departure_at_X = arrival_at_X + wait_at_X + service_X

    # (X -> B)
    dist_X_B, travel_X_B = _get_dist_and_time(
        loc_X_idx, loc_B_id, False, loc_B_is_depot, 
        truck_info, problem_instance
    )
    new_arrival_at_B = departure_at_X + travel_X_B
    
    # 5. KIỂM TRA FORWARD SLACK (O(1))
    original_arrival_at_B = node_B['arrival']
    original_slack_at_B = node_B['forward_slack']
    delay_at_B = new_arrival_at_B - original_arrival_at_B
    
    if delay_at_B > original_slack_at_B + 1e-6: 
        return False, float('inf') 

    # 6. TÍNH TOÁN CHI PHÍ DELTA (O(1))
    dist_A_B, _ = _get_dist_and_time(
        loc_A_id, loc_B_id, loc_A_is_depot, loc_B_is_depot, 
        truck_info, problem_instance
    )
    dist_increase = (dist_A_X + dist_X_B) - dist_A_B

    original_wait_at_B = node_B['wait']
    new_wait_at_B = max(0, original_wait_at_B - delay_at_B)
    wait_increase = wait_at_X + (new_wait_at_B - original_wait_at_B)
    
    # Lấy chi phí từ problem_instance
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
        (truck_info['type'], truck_info['region']), 1.0)
    WAIT_COST_PER_MIN = 0.2 # (Hoặc lấy từ problem_instance)
    
    cost_increase = (dist_increase * var_cost_per_km) + (wait_increase * WAIT_COST_PER_MIN)

    return True, cost_increase

def _check_insertion_efficiency(problem_instance, route_info, insert_pos, farm_id_to_insert, shift, start_time):
    """Thực hiện The Feasibility Checklist và tính toán chi phí tăng thêm."""

    depot_idx, truck_id, customer_list, shift_in_route, route_start_time = route_info
    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks']) #Tìm truck_id của route rồi tra cứu ra toàn bộ dict của truck
    if not truck_info:
        return False, float('inf'), -1

    WAIT_COST_PER_MIN = 0.2
    TIME_PENALTY_COST = 0.3
    CAPACITY_PENALTY_COSt = 9999
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
        (truck_info['type'], truck_info['region']), 1.0
    )

    # --- Accessibility + capacity check ---
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    truck_type_idx = type_to_idx.get(truck_info['type']) #Lấy ra truck type của truck trong route đó
    if truck_type_idx is None: #Nếu k thấy trục_type --> cho False
        return False, float('inf'), -1

    _, farm_details, farm_demand = _get_farm_info(farm_id_to_insert, problem_instance) 
    farm_access = farm_details.get('accessibility')
    if farm_access is None or len(farm_access) <= truck_type_idx or farm_access[truck_type_idx] != 1: #Check lại logic accessibility
        return False, float('inf'), -1

    current_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in customer_list)
    if current_load + farm_demand > truck_info['capacity']:
        return False, float('inf'), -1

    # --- Compute old route cost ---    
    old_total_cost = 0
    if customer_list:
        _, is_feasible_old, old_dist, old_wait, _, time_penalty, capacity_penalty = _calculate_route_schedule_and_feasibility(
            depot_idx, customer_list, shift_in_route, start_time, problem_instance, truck_info=truck_info
        )
        if not is_feasible_old:
            return False, float('inf'), -1
        old_total_cost = old_dist * var_cost_per_km + old_wait * WAIT_COST_PER_MIN + time_penalty * TIME_PENALTY_COST + CAPACITY_PENALTY_COSt * capacity_penalty

    # --- Compute new route cost after inserting this farm ---
    test_route = customer_list[:insert_pos] + [farm_id_to_insert] + customer_list[insert_pos:]
    #Technique: Lấy mọi customer từ đầu tới vị trí insert_pos + chèn id farm mới vào + lấy phần còn lại
    _, is_feasible_new, new_dist, new_wait, _, time_penalty, capacity_penalty = _calculate_route_schedule_and_feasibility(
        depot_idx, test_route, shift_in_route, start_time, problem_instance, truck_info=truck_info
    )

    if not is_feasible_new:
        return False, float('inf'), -1

    new_total_cost = new_dist * var_cost_per_km + new_wait * WAIT_COST_PER_MIN + time_penalty * TIME_PENALTY_COST + capacity_penalty * CAPACITY_PENALTY_COSt
    cost_increase = new_total_cost - old_total_cost
        
    return True, cost_increase, new_total_cost


# ==============================================================================
# HÀM DESTROY
# ==============================================================================

def _remove_customers_from_schedule(schedule, customers_to_remove):
    """
    Xóa danh sách khách hàng khỏi schedule hiện tại.
    Mỗi route_info bây giờ có 5 phần tử: (depot_idx, truck_id, customer_list, shift, start_time)
    """
    new_schedule = []
    for route_info in schedule:#Duyệt qua từng phần tử trong array scheduling
        depot_idx, truck_id, customer_list, shift, start_time = route_info    
        # Giữ lại các khách hàng không bị xóa
        updated_customer_list = [c for c in customer_list if c not in customers_to_remove]
        
        if updated_customer_list:
            new_schedule.append((depot_idx, truck_id, updated_customer_list, shift, start_time))
    
    return new_schedule



# ==============================================================================
# HÀM LOCAL SEARCH
# ==============================================================================

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

    _, is_feasible, total_dist, total_wait, _, time_penalty, capacity_penalty = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, 0, problem_instance, truck_info
    )

    if not is_feasible:
        return float('inf')

    return (total_dist * var_cost_per_km) + (total_wait * WAIT_COST_PER_MIN)


# ==============================================================================


# ==============================================================================
# HÀM SIMULATE (ĐÃ SỬA LỖI)
# ==============================================================================
def simulate_route_and_get_timeline(problem_instance, depot_idx, customer_list, shift, truck_info, passed_start_time): 
    """Mô phỏng tuyến thực tế, BẮT ĐẦU TỪ THỜI GIAN ĐÃ TỐI ƯU (passed_start_time)."""
    if not customer_list:
        return 0, [], 0

    # ### SỬA 1: SỬ DỤNG THỜI GIAN ĐÚNG (thay vì 0) ###
    start_time_at_depot = passed_start_time 
    
    # Lấy thông tin
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms'] 
    farm_id_to_idx = problem_instance['farm_id_to_idx_map'] 
    virtual_map = problem_instance.get('virtual_split_farms', {}) 
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    
    # ### SỬA 2: SỬ DỤNG _resolve_farm (logic y hệt objective) ###
    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid) # Cần hàm _clean_base_id
        try: base_idx = farm_id_to_idx[base_id_str]
        except KeyError: base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        
        if isinstance(fid, str) and fid in virtual_map:
            # Trả về (idx, details, portion_demand)
            return base_idx, base_info, virtual_map[fid]['portion']
        else:
            # Trả về (idx, details, full_demand)
            return base_idx, base_info, base_info['demand']

    timeline = []
    current_time = start_time_at_depot # <-- BẮT ĐẦU TỪ THỜI GIAN ĐÚNG
    prev_idx = -1

    for i, fid in enumerate(customer_list):
        idx, details, demand = _resolve_farm(fid) # <-- SỬ DỤNG HÀM ĐÚNG
        
        travel_dist = depot_farm_dist[depot_idx, idx] if i == 0 else dist_matrix[prev_idx, idx]
        travel_time = travel_dist / velocity
        arrival = current_time + travel_time
        start_tw, _ = details['time_windows'][shift]
        
        # 'wait' SẼ ĐƯỢC TÍNH TOÁN DỰA TRÊN THỜI GIAN BẮT ĐẦU ĐÚNG
        wait = max(0, start_tw - arrival) 
        
        start_service = arrival + wait
        fix, var = details['service_time_params']
        service_duration = fix + (demand / var if var > 0 else 0)
        finish_service = start_service + service_duration
        
        timeline.append({
            'fid': fid,
            'arrival': arrival,
            'wait': wait,
            'start': start_service,
            'finish': finish_service
        })
        
        # ### SỬA 3: CẬP NHẬT THỜI GIAN (thay vì reset) ###
        current_time = finish_service 
        prev_idx = idx

    # Quay về depot
    travel_back = depot_farm_dist[depot_idx, prev_idx]
    travel_time_back = travel_back / velocity
    return_depot_time = current_time + travel_time_back

    return start_time_at_depot, timeline, return_depot_time

# ==============================================================================
# HÀM FMT (Giữ nguyên)
# ==============================================================================
def fmt(minutes):
    """Định dạng phút (float) sang chuỗi HH:MM, làm tròn LÊN phút gần nhất."""
    if minutes is None or not isinstance(minutes, (int, float)):
        return "00:00"
    total_rounded_minutes = math.ceil(minutes)
    hours, mins = divmod(total_rounded_minutes, 60)
    return f"{int(hours):02d}:{int(mins):02d}"

# ==============================================================================
# HÀM IN (Sửa lại 2 chỗ)
# ==============================================================================
def print_schedule(sol):
    """
    In ra lịch trình tối ưu, sử dụng optimal_start_time đã lưu.
    """
    prob = sol.problem_instance
    print("\n===== 🧭 LỊCH TRÌNH TỐI ƯU CHO NGÀY =====")
    
    # ### SỬA 4: Nhận "optimal_start_time" (từ schedule) ###
    for depot, truck_id, custs, shift, optimal_start_time in sol.schedule:
        if not custs and shift != 'INTER-FACTORY': continue

        if shift == 'INTER-FACTORY':
            print(f"  🏭 Truck {truck_id} ({shift}): {str(custs[0]).replace('_', ' ')}")
            continue

        # (Bạn phải đảm bảo hàm find_truck_by_id và _clean_base_id có thể được truy cập)
        truck_info = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
        if not truck_info: continue

        # ### SỬA 5: Gửi "optimal_start_time" vào hàm mô phỏng ###
        calc_start, timeline, return_depot_time = simulate_route_and_get_timeline(
            prob, depot, custs, shift, truck_info, optimal_start_time
        )
        
        if not timeline: continue

        print(f"  🚚 Truck {truck_id} ({shift}) - Depot {depot} (Xuất phát lúc {fmt(calc_start)}):")
        for stop in timeline:
            # Bây giờ "stop['wait']" sẽ được tính toán chính xác
            print(f"    🧭 Farm {stop['fid']}: Arrive {fmt(stop['arrival'])}, Wait {math.ceil(stop['wait'])} min, "
                  f"Start {fmt(stop['start'])}, Finish {fmt(stop['finish'])}")