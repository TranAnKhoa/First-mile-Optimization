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

def _calculate_route_schedule_and_feasibility(
    depot_idx, customer_list, shift, 
    start_time_at_depot, finish_time_route, route_load, # Nhận đủ tham số từ 7-tuple
    problem_instance, truck_info
):
    """ 
    ## FINAL DETAILED VERSION ##
    
    1. Capacity Violation: Tính O(1) dựa trên route_load.
    2. Time Penalty: Tính O(N) bằng cách duyệt từng node để cộng dồn vi phạm TW.
    
    Output: (is_feasible, total_dist, total_wait, time_penalty, capacity_violation)
    """
    
    # === CHECK RỖNG ===
    if not customer_list:
        return True, 0, 0, 0, 0

    # === 1. TÍNH CAPACITY VIOLATION (O(1)) ===
    # Tận dụng route_load đã lưu, không cần cộng lại
    truck_capacity = truck_info.get('capacity', float('inf'))
    capacity_violation = max(0.0, route_load - truck_capacity)

    # === 2. KHỞI TẠO MÔ PHỎNG ===
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    
    shift_end_time = 1900 # Hoặc lấy từ config
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    
    total_dist = 0
    total_wait = 0
    time_penalty = 0.0
    current_time = start_time_at_depot #Mô phỏng lại từ thời điểm bắt đầu

    # Helper tra cứu nhanh
    def _get_node_data(fid):
        base_id = int(str(fid).split('_')[0]) if isinstance(fid, str) else fid
        idx = farm_id_to_idx.get(base_id)
        if idx is None: idx = farm_id_to_idx.get(str(base_id))
        info = farms[idx]
        return idx, info['time_windows'], info['service_time_params']

    try:
        # === 3. VÒNG LẶP TÍNH TOÁN CHI TIẾT ===
        # --- A. Depot -> Khách đầu tiên ---
        first_id = customer_list[0]
        to_idx, to_tw, to_params = _get_node_data(first_id)
        d = depot_farm_dist[depot_idx, to_idx]
        total_dist += d
        current_time += (d / velocity) # Giờ đến nơi
        
        # Check TW Khách đầu
        start_tw, end_tw = to_tw[shift]
        
        # Xử lý chờ (nếu đến sớm)
        wait = max(0, start_tw - current_time)
        total_wait += wait
        
        # Thời gian bắt đầu phục vụ (Service Start)
        # Nếu đến muộn (current_time > end_tw), service_start chính là current_time
        service_start = current_time + wait 
        # [QUAN TRỌNG] Tính phạt nếu đến muộn
        if service_start > end_tw + 1e-6:
            violation = service_start - end_tw
            time_penalty += violation

        # Cộng thời gian phục vụ
        # (params[0] là fixed time. Nếu muốn chính xác tuyệt đối có thể cộng thêm var time)
        current_time = service_start + to_params[0] 

        # --- B. Khách -> Khách ---
        prev_idx = to_idx
        for i in range(1, len(customer_list)):
            curr_id = customer_list[i]
            to_idx, to_tw, to_params = _get_node_data(curr_id)
            
            # Di chuyển
            d = dist_matrix[prev_idx, to_idx]
            total_dist += d
            current_time += (d / velocity)
            
            # Check TW
            start_tw, end_tw = to_tw[shift]
            wait = max(0, start_tw - current_time)
            total_wait += wait
            service_start = current_time + wait
            
            # [QUAN TRỌNG] Tính phạt TW từng khách
            if service_start > end_tw + 1e-6:
                violation = service_start - end_tw
                time_penalty += violation
            
            current_time = service_start + to_params[0]
            prev_idx = to_idx

        # --- C. Khách cuối -> Depot ---
        d_back = depot_farm_dist[depot_idx, prev_idx]
        total_dist += d_back
        current_time += (d_back / velocity) # Đây là finish_time thực tế sau khi tính toán lại

        # [QUAN TRỌNG] Tính phạt nếu về Depot muộn
        if current_time > shift_end_time + 1e-6:
            time_penalty += (current_time - shift_end_time)

    except Exception:
        # Fallback an toàn
        return False, float('inf'), 0, 1e9, 1e9

    # === 4. TRẢ VỀ KẾT QUẢ ===
    # is_feasible luôn True vì ta dùng Soft Constraints (đã chuyển thành penalty)
    return True, total_dist, total_wait, time_penalty, capacity_violation
# ==============================================================================
# HÀM Repair 
# ==============================================================================
def balance_depot_loads(repaired_solution, truck_finish_times):
    """
    Chạy sau khi Repair xong các tuyến Farm.
    Xử lý quá tải kho bằng cách tận dụng xe rảnh hoặc xe đã chạy xong.
    """
    problem_instance = repaired_solution.problem_instance
    facilities = problem_instance['facilities']
    available_trucks = problem_instance['fleet']['available_trucks']
    depot_capacity = [f['capacity'] for f in facilities]
    dist_matrix_depots = problem_instance.get('distance_matrix_depots')
    
    # 1. Tính tải trọng hiện tại của các kho (Chỉ tính hàng từ Farm về)
    depot_loads = defaultdict(float)
    for route in repaired_solution.schedule:
        # Unpack 7-tuple
        depot_idx, _, _, shift, _, _, route_load = route
        if shift != 'INTER-FACTORY':
            depot_loads[depot_idx] += route_load

    # 2. Xử lý từng kho bị quá tải
    for depot_idx, current_load in depot_loads.items():
        if current_load > depot_capacity[depot_idx]:
            
            excess_amount = current_load - depot_capacity[depot_idx]
            current_region = facilities[depot_idx]['region']
            
            # Tìm kho đích cùng vùng còn trống (Logic đơn giản hóa: chọn kho khác bất kỳ cùng vùng)
            # (Trong thực tế bạn nên check xem kho đích có bị đầy không, nhưng ở đây ta giả định kho đích nhận được)
            target_depots = [i for i, f in enumerate(facilities) 
                             if f.get('region') == current_region and i != depot_idx]
            
            if not target_depots: continue
            target_depot = target_depots[0] # Chọn kho đầu tiên tìm thấy
            
            if dist_matrix_depots is None: continue
            dist_one_way = dist_matrix_depots[depot_idx, target_depot]

            # --- CHIẾN THUẬT CHỌN XE (OPTION 2) ---
            # Duyệt qua TẤT CẢ các xe (đã dùng hoặc chưa dùng)
            # Sắp xếp xe theo Capacity giảm dần (để ưu tiên xe to chở cho nhanh hết)
            sorted_trucks = sorted(available_trucks, key=lambda t: t['capacity'], reverse=True)

            for truck in sorted_trucks:
                if excess_amount <= 0: break # Đã chuyển hết hàng
                
                if truck.get('region') != current_region: continue

                # Lấy thời gian rảnh của xe này
                # Nếu xe chưa chạy gì cả -> finish_time = 0
                # Nếu xe đã chạy farm -> finish_time = giờ về depot cuối cùng
                # (Lưu ý: truck_finish_times lưu theo key (truck_id, shift), ta cần lấy max finish time của xe đó)
                
                # Tìm thời gian kết thúc muộn nhất của xe này trong tất cả các ca
                current_finish_time = 0
                for (tid, shift), (ftime, d_loc) in truck_finish_times.items():
                    if tid == truck['id'] and ftime > current_finish_time:
                        current_finish_time = ftime
                
                # Tính toán chuyến đi Inter-Factory
                velocity = 1.0 if truck['type'] in ["Single", "Truck and Dog"] else 0.5
                travel_time = (dist_one_way / velocity) * 2 # Đi và về
                
                start_time = current_finish_time
                end_time = start_time + travel_time
                
                # Kiểm tra giờ đóng cửa (Ví dụ 19:00 = 1140 phút)
                # Nếu xe rảnh lúc 14:00, chuyến đi mất 3 tiếng -> Xong 17:00 -> OK
                if end_time <= 3000: 
                    
                    # Tính lượng hàng xe này chở được
                    amount_to_carry = min(excess_amount, truck['capacity'])
                    
                    # TẠO TUYẾN INTER-FACTORY
                    transfer_route = [f'TRANSFER_FROM_{depot_idx}_TO_{target_depot}']
                    
                    repaired_solution.schedule.append((
                        depot_idx,
                        truck['id'],
                        transfer_route,
                        'INTER-FACTORY',
                        start_time,
                        end_time,
                        amount_to_carry
                    ))
                    
                    # Cập nhật trạng thái
                    excess_amount -= amount_to_carry
                    truck_finish_times[(truck['id'], 'INTER-FACTORY')] = (end_time, depot_idx)
                    
                    # (Tùy chọn) In ra để debug
                    # print(f"   --> Chuyển {amount_to_carry} từ kho {depot_idx} bằng xe {truck['id']} (Rảnh lúc {start_time})")

    return repaired_solution

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
    detailed_schedule[-1]['forward_slack'] = last_slack #cập nhật forward_slack cuối thành giá trị mới

    for i in range(len(detailed_schedule) - 2, -1, -1): #i = 2, 1, 0
        current_node = detailed_schedule[i] #2
        before_node = detailed_schedule[i+1] #3
            
        slack_via_next = before_node['forward_slack'] + before_node['wait']
        slack_via_tw = current_node['tw_close'] - current_node['arrival']
        
        current_node['forward_slack'] = min(slack_via_next, slack_via_tw)

    return True, total_dist, total_wait, detailed_schedule
    
def _check_insertion_delta(problem_instance, route_info, original_schedule, 
                           insert_pos, farm_id_to_insert, 
                           truck_info, current_load):
    """
    KIỂM TRA CHÈN SIÊU NHANH (O(1)) - Tính toán Delta.
    """
    
    depot_idx, truck_id, customer_list, shift, start_time_at_depot,_,_ = route_info
    
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

def calculate_route_finish_time(depot_idx, customer_list, shift, start_time, problem_instance, truck_info):
    """
    Hàm chuyên biệt để tính toán thời gian kết thúc thực tế của một tuyến.
    Dùng để cập nhật 'finish_time' trong 7-tuple và 'truck_finish_times'.
    KHÔNG tính chi phí, KHÔNG kiểm tra penalty.
    """
    if not customer_list:
        return start_time

    # 1. Setup
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    current_time = start_time

    # Helper nội bộ lấy dữ liệu
    def _get_data(fid):
        base_id = int(str(fid).split('_')[0]) if isinstance(fid, str) else fid
        idx = farm_id_to_idx.get(base_id, farm_id_to_idx.get(str(base_id)))
        info = farms[idx]
        return idx, info['time_windows'], info['service_time_params']

    try:
        # 2. Depot -> Khách đầu tiên
        first_id = customer_list[0]
        to_idx, to_tw, to_params = _get_data(first_id)
        
        # Di chuyển
        travel_time = depot_farm_dist[depot_idx, to_idx] / velocity
        current_time += travel_time
        
        # Chờ (nếu đến sớm)
        start_tw, _ = to_tw[shift]
        wait = max(0, start_tw - current_time)
        current_time += wait
        
        # Phục vụ
        current_time += to_params[0] # Fixed service time

        # 3. Khách -> Khách
        prev_idx = to_idx
        for i in range(1, len(customer_list)):
            curr_id = customer_list[i]
            to_idx, to_tw, to_params = _get_data(curr_id)
            
            # Di chuyển
            travel_time = dist_matrix[prev_idx, to_idx] / velocity
            current_time += travel_time
            
            # Chờ
            start_tw, _ = to_tw[shift]
            wait = max(0, start_tw - current_time)
            current_time += wait
            
            # Phục vụ
            current_time += to_params[0]
            prev_idx = to_idx

        # 4. Khách cuối -> Depot
        travel_time_back = depot_farm_dist[depot_idx, prev_idx] / velocity
        current_time += travel_time_back

    except Exception as e:
        print(f"Lỗi tính finish time: {e}")
        return start_time # Fallback

    return current_time

# ==============================================================================
# HÀM DESTROY
# ==============================================================================
# --- HELPER CẦN THIẾT CHO HISTORICAL REMOVAL ---
def _get_dist_between_nodes(u, v, problem, depot_idx):
    """Hàm phụ lấy khoảng cách giữa 2 node (có thể là Depot hoặc Farm)"""
    farms_dist = problem['distance_matrix_farms']
    depots_dist = problem['distance_depots_farms']
    f_map = problem['farm_id_to_idx_map']
    
    def get_idx(nid):
        # Giả sử -1 là Depot
        if nid == -1: return -1 
        return f_map.get(nid, f_map.get(str(nid)))

    u_idx, v_idx = get_idx(u), get_idx(v)
    
    if u_idx == -1 and v_idx == -1: return 0 # Depot -> Depot
    if u_idx == -1: return depots_dist[depot_idx, v_idx] # Depot -> Farm
    if v_idx == -1: return depots_dist[depot_idx, u_idx] # Farm -> Depot
    return farms_dist[u_idx, v_idx] # Farm -> Farm

def update_history_matrix(history_matrix, solution):
    """
    Cập nhật ma trận lịch sử với các cạnh trong giải pháp hiện tại.
    Phiên bản fix lỗi NumPy array và hỗ trợ Virtual Split IDs.
    history_matrix: Dictionary {(u, v): min_cost}
    """
    problem = solution.problem_instance
    dist_matrix = problem['distance_matrix_farms']     # Ma trận Farm-Farm
    depot_dist = problem['distance_depots_farms']      # Ma trận Depot-Farm
    
    # Hàm nội bộ: Lấy Index trong ma trận của một node (xử lý cả ảo và thật)
    def _get_matrix_idx(node_id):
        # Nếu là marker Depot (-1)
        if node_id == -1: 
            return -1
        # Nếu là Farm (ảo hoặc thật), dùng hàm có sẵn để lấy index chuẩn
        # _get_farm_info trả về (idx, details, demand) -> lấy [0]
        return int(_get_farm_info(node_id, problem)[0])

    # Hàm nội bộ: Lấy ID gốc để làm Key cho dictionary (để 7678_part1 cũng tính là 7678)
    def _get_clean_key(node_id):
        if node_id == -1: return -1
        cleaned = _clean_base_id(node_id)
        # Cố gắng chuyển về int nếu ID gốc là số (để đồng bộ với key cũ trong dict)
        try: return int(cleaned)
        except: return cleaned

    for route_info in solution.schedule:
        # Unpack route (bảo vệ trường hợp thiếu phần tử)
        if len(route_info) < 3: continue
        depot_idx = route_info[0]
        customer_list = route_info[2]
        shift = route_info[3]

        if not customer_list or shift == 'INTER-FACTORY':
            continue
            
        # Xây dựng chuỗi node: [-1] + [c1, c2, ...] + [-1]
        nodes = [-1] + customer_list + [-1]
        
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i+1]
            
            try:
                # 1. Lấy Index chuẩn để tra ma trận
                u_idx = _get_matrix_idx(u)
                v_idx = _get_matrix_idx(v)
                
                # 2. Tính Cost (Distance)
                dist = 0.0
                
                # Trường hợp A: Depot -> Farm
                if u_idx == -1:
                    dist = float(depot_dist[depot_idx, v_idx])
                # Trường hợp B: Farm -> Depot
                elif v_idx == -1:
                    dist = float(depot_dist[depot_idx, u_idx])
                # Trường hợp C: Farm -> Farm
                else:
                    dist = float(dist_matrix[u_idx, v_idx])
                
                # 3. Lấy Key chuẩn (Clean ID) để lưu vào History
                # (Để xe chạy từ 7678_part1 qua 54 cũng được tính là từ 7678 qua 54)
                u_key = _get_clean_key(u)
                v_key = _get_clean_key(v)
                
                edge_key = (u_key, v_key)
                
                # 4. Cập nhật Min Cost
                # Sử dụng get với default là vô cùng
                current_best = history_matrix.get(edge_key, float('inf'))
                
                if dist < current_best:
                    history_matrix[edge_key] = dist
                    # Nếu bạn muốn ma trận đối xứng (cho undirected graph), mở dòng dưới:
                    # history_matrix[(v_key, u_key)] = dist 
                    
            except Exception:
                # Bỏ qua nếu có lỗi lookup ID (hiếm gặp)
                continue
                
    return history_matrix

def _remove_customers_from_schedule(schedule, customers_to_remove):
    """
    Xóa danh sách khách hàng khỏi schedule hiện tại.
    Mỗi route_info bây giờ có 5 phần tử: (depot_idx, truck_id, customer_list, shift, start_time)
    """
    new_schedule = []
    for route_info in schedule:#Duyệt qua từng phần tử trong array scheduling
        depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load = route_info    
        # Giữ lại các khách hàng không bị xóa
        updated_customer_list = [c for c in customer_list if c not in customers_to_remove]
        
        if updated_customer_list:
            new_schedule.append((depot_idx, truck_id, updated_customer_list, shift, start_time, finish_time, route_load))
    
    return new_schedule

#! Best_insertion:

# ==============================================================================
# HÀM LOCAL SEARCH
# ==============================================================================
def get_route_cost(problem_instance, route_info):
    """
    Tính toán tổng chi phí của MỘT tuyến đường duy nhất, sử dụng hàm mô phỏng "chân lý".
    Đây là phiên bản đúng để dùng trong các toán tử Local Search.
    """
    depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load = route_info
    
    # Bỏ qua các tuyến đặc biệt hoặc rỗng
    if not customer_list or shift == 'INTER-FACTORY':
        return 0.0

    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
    if not truck_info:
        return float('inf') # Trả về chi phí vô hạn nếu không tìm thấy xe

    # Lấy các hệ số chi phí từ problem_instance (để đảm bảo nhất quán)
    costs = problem_instance.get('costs', {})
    var_cost_per_km = costs.get('variable_cost_per_km', {}).get((truck_info['type'], truck_info['region']), 1.0)
    WAIT_COST_PER_MIN = costs.get('wait_cost_per_min', 0.2)
    TIME_PENALTY_COST = costs.get('time_penalty_cost', 0.3)
    CAPACITY_PENALTY_COST = costs.get('capacity_penalty_cost', 9999)

    # Gọi hàm mô phỏng duy nhất để lấy tất cả các chỉ số
    is_feasible, total_dist, total_wait, time_penalty, capacity_violation = \
        _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, 
    start_time, finish_time, route_load, # Nhận đủ tham số từ 7-tuple
    problem_instance, truck_info)

    if not is_feasible:
        return float('inf')

    # Tính toán tổng chi phí theo công thức của hàm mục tiêu
    total_cost = (total_dist * var_cost_per_km) + \
                 (total_wait * WAIT_COST_PER_MIN) + \
                 (time_penalty * TIME_PENALTY_COST) + \
                 (capacity_violation * CAPACITY_PENALTY_COST)
                 
    return total_cost
from datetime import timedelta, datetime
def fmt(minutes):
    """Hàm tiện ích chuyển đổi phút (float) sang định dạng HH:MM."""
    if minutes is None or minutes == float('inf') or minutes == float('-inf'):
        return "N/A"
    try:
        # Sử dụng timedelta để xử lý (an toàn hơn)
        return (datetime.min + timedelta(minutes=minutes)).strftime('%H:%M')
    except Exception:
        return f"{minutes:.2f} min"
# ==============================================================================
def print_schedule(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """
    HÀM IN MỚI (THEO STYLE CỦA BẠN):
    In ra lịch trình chi tiết của một tuyến đường theo định dạng 1 dòng/stop.
    Logic vẫn dựa trên hàm _calculate_route_schedule_and_feasibility.
    """
    
    # === BƯỚC 1: KHỞI TẠO BIẾN ===
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    
    shift_end_time = 1990 
    truck_capacity = truck_info.get('capacity', float('inf')) 
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    virtual_map = problem_instance.get('virtual_split_farms', {})

    # (Hàm _resolve_farm - lồng bên trong)
    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid) 
        try: base_idx = farm_id_to_idx[base_id_str]
        except KeyError: base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        if isinstance(fid, str) and fid in virtual_map:
            return base_idx, virtual_map[fid]['portion'], base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

    # === BƯỚC 2: MÔ PHỎNG VÀ IN ===
    total_dist = 0
    total_wait = 0
    time_penalty = 0.0
    total_demand = 0.0
    current_time = start_time_at_depot 
    
    try:
        # ---- Xử lý khách hàng đầu tiên (Depot -> C1) ----
        farm_id_c1 = customer_list[0]
        idx, demand, params, tw = _resolve_farm(farm_id_c1)
        total_demand += demand
        
        travel_dist = depot_farm_dist[depot_idx, idx]; total_dist += travel_dist
        travel_time = travel_dist / velocity; 
        arrival = current_time + travel_time
        start_tw, end_tw = tw[shift] 
        wait_time = max(0, start_tw - arrival); total_wait += wait_time
        service_start = arrival + wait_time
        
        if service_start > end_tw + 1e-6: 
            time_penalty += (service_start - end_tw)
        
        service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
        current_time = service_start + service_duration # Đây là departure time

        # In dòng đầu tiên
        print(f"    🧭 Farm {str(farm_id_c1).ljust(20)}: Arrive {fmt(arrival)}, Wait {math.ceil(wait_time):>2} min, "
              f"Start {fmt(service_start)}, Finish {fmt(current_time)}")

        # ---- Xử lý các khách hàng ở giữa (C(i) -> C(i+1)) ----
        for i in range(len(customer_list) - 1):
            from_idx, _, _, _ = _resolve_farm(customer_list[i])
            
            farm_id_next = customer_list[i+1]
            to_idx, to_demand, to_params, to_tw = _resolve_farm(farm_id_next)
            
            total_demand += to_demand
            
            travel_dist = dist_matrix[from_idx, to_idx]; total_dist += travel_dist
            travel_time = travel_dist / velocity
            arrival = current_time + travel_time
            
            start_tw, end_tw = to_tw[shift] 
            wait_time = max(0, start_tw - arrival); total_wait += wait_time
            service_start = arrival + wait_time
            
            if service_start > end_tw + 1e-6:
                time_penalty += (service_start - end_tw)
                
            service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
            current_time = service_start + service_duration # Departure time

            # In các dòng tiếp theo
            print(f"    🧭 Farm {str(farm_id_next).ljust(20)}: Arrive {fmt(arrival)}, Wait {math.ceil(wait_time):>2} min, "
                  f"Start {fmt(service_start)}, Finish {fmt(current_time)}")

        # ---- Xử lý quay về Depot (CLast -> Depot) ----
        last_idx, _, _, _ = _resolve_farm(customer_list[-1])
        travel_dist_back = depot_farm_dist[depot_idx, last_idx]; total_dist += travel_dist_back
        travel_time_back = travel_dist_back / velocity
        finish_time = current_time + travel_time_back
        
        if finish_time > shift_end_time + 1e-6:
               time_penalty += (finish_time - shift_end_time)
        
        # In dòng cuối cùng (về Depot)
        print(f"🏁 Về Depot {depot_idx}: Arrive {fmt(finish_time)}")

        # === BƯỚC 3: IN TỔNG KẾT ===
        print(f"    -----------------------------------------------------------------------")
        capacity_violation = max(0, total_demand - truck_capacity)
        print(f"📊 Tổng: Dist: {total_dist:.1f} km | Wait: {total_wait:.1f} min | Demand: {total_demand:.1f}/{truck_capacity:.1f} "
              f"| Time Pen: {time_penalty:.1f} | Cap Pen: {capacity_violation:.1f}")
    
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG khi in lịch trình: {e}.")
def _calculate_optimal_early_start(depot_idx, customer_list, shift, problem_instance, truck_info):
    """
    Tính toán thời gian bắt đầu tối ưu:
    Dời lịch trễ lại để giảm Wait Time, nhưng không được vượt quá Slack (để đảm bảo Feasible).
    """
    if not customer_list:
        # Trả về thời gian mặc định của ca nếu route rỗng
        return problem_instance['shifts'][shift]['start'], True

    # 1. Mô phỏng với start_time = 0 (hoặc thời gian start ca)
    # Để đo lường Total Wait và Slack tối đa
    initial_start_ref = 0 
    
    is_feasible, _, total_wait, detailed_schedule = _calculate_route_schedule_WITH_SLACK(
        depot_idx, 
        customer_list, 
        shift, 
        initial_start_ref, 
        problem_instance, 
        truck_info
    )

    if not is_feasible or not detailed_schedule:
        return -1, False

    # 2. Lấy Forward Slack tại Depot (Node đầu tiên trong schedule)
    # forward_slack này đã tính toán tất cả các ràng buộc TW phía sau (nhờ logic min trong hàm WITH_SLACK)
    max_safe_delay = detailed_schedule[0]['forward_slack'] #! Remind: forward slack là khoảng thời gian min có thể delay trong 1 route
    
    # 3. Tính toán lượng Delay tối ưu
    # - Muốn delay bằng total_wait để triệt tiêu thời gian chờ.
    # - Nhưng bị chặn trên bởi max_safe_delay để không vi phạm TW hẹp.
    optimal_delay = min(total_wait, max_safe_delay) #! Lượng delay tối ưu, tránh vi phạm TW, và tránh lùi quá giờ ca
    
    # 4. Thời gian bắt đầu mới
    new_start_time = initial_start_ref + optimal_delay #! Delay lại 1 khoảng 
    
    return new_start_time, True
# Đặt hàm này ở cấp độ cao trong code của bạn, ví dụ gần chỗ gọi ALNS

def optimize_all_start_times(solution_to_optimize):
    optimized_solution = copy.deepcopy(solution_to_optimize)
    problem_instance = optimized_solution.problem_instance
    
    # Gom nhóm các chuyến theo xe (Giữ nguyên logic cũ)
    truck_routes_map = {}
    for idx, route in enumerate(optimized_solution.schedule):
        truck_id = route[1]
        if truck_id not in truck_routes_map: truck_routes_map[truck_id] = []
        truck_routes_map[truck_id].append((route, idx))
        
    # Sắp xếp các chuyến của mỗi xe theo thời gian
    for tid in truck_routes_map:
        truck_routes_map[tid].sort(key=lambda x: x[0][4]) 

    final_schedule = [None] * len(optimized_solution.schedule) 

    for tid, routes_with_idx in truck_routes_map.items():
        for i, (route_info, original_idx) in enumerate(routes_with_idx):
            components = list(route_info)
            # Unpack 7 phần tử
            depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load = components

            # Bỏ qua nếu không có khách hoặc là chuyển kho
            if not customer_list or shift == 'INTER-FACTORY':
                final_schedule[original_idx] = route_info
                continue
            
            truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
            
            # -------------------------------------------------------
            # 1. TÍNH COST CŨ (Baseline)
            # -------------------------------------------------------
            metrics_old = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, 
                start_time_at_depot=start_time, finish_time_route=0, route_load=route_load,
                problem_instance=problem_instance, truck_info=truck_info
            )
            # metrics_old chỉ có 5 phần tử: (is_feasible, dist, wait, time_pen, cap_pen)
            cost_old = (metrics_old[2] * 1.0) + (metrics_old[3] * 1e9)

            # 2. XÁC ĐỊNH GIỚI HẠN DELAY
            next_trip_limit_delay = float('inf')
            if i < len(routes_with_idx) - 1:
                next_route_start = routes_with_idx[i+1][0][4]
                gap = next_route_start - finish_time
                next_trip_limit_delay = max(0, gap)

            # 3. TÌM GIỜ XUẤT PHÁT SỚM NHẤT CÓ THỂ (OPTIMAL START)
            optimal_start_time, is_feasible_opt = _calculate_optimal_early_start(
                depot_idx, customer_list, shift, problem_instance, truck_info
            )

            if is_feasible_opt:
                proposed_delay = optimal_start_time - start_time
                actual_delay = max(0, min(proposed_delay, next_trip_limit_delay))
                
                if actual_delay < 1e-3: 
                    final_schedule[original_idx] = route_info
                    continue

                final_start_time = start_time + actual_delay
                
                # 4. VALIDATION - KIỂM TRA COST VỚI GIỜ MỚI
                metrics_new = _calculate_route_schedule_and_feasibility(
                    depot_idx, customer_list, shift, 
                    start_time_at_depot=final_start_time, finish_time_route=0, route_load=route_load,
                    problem_instance=problem_instance, truck_info=truck_info
                )
                cost_new = (metrics_new[2] * 1.0) + (metrics_new[3] * 1e9)

                # 5. SO SÁNH VÀ CẬP NHẬT
                if cost_new <= cost_old and metrics_new[3] <= metrics_old[3] + 1e-6:
                    # --- [SỬA Ở ĐÂY] ---
                    # Thay vì lấy metrics_new[5] (gây lỗi), ta gọi hàm riêng để tính
                    new_finish_time = calculate_route_finish_time(
                        depot_idx, customer_list, shift, final_start_time,
                        problem_instance, truck_info
                    )
                    
                    components[4] = final_start_time    # Cập nhật Start Time
                    components[5] = new_finish_time     # Cập nhật Finish Time tính riêng
                    final_schedule[original_idx] = tuple(components)
                    # -------------------
                else:
                    final_schedule[original_idx] = route_info
            else:
                final_schedule[original_idx] = route_info

    final_schedule = [r for r in final_schedule if r is not None]
    optimized_solution.schedule = final_schedule
    return optimized_solution

def reconstruct_truck_finish_times(solution):
    """
    Quét qua schedule hiện tại để tìm thời gian rảnh (finish time) muộn nhất của từng xe.
    """
    finish_times = defaultdict(lambda: (0.0, -1)) # Mặc định (0.0, -1)
    
    for route in solution.schedule:
        # Unpack 7-tuple
        depot, truck_id, cust_list, shift, start, finish, load = route
        
        # Key theo truck và shift (để khớp với logic của balance_depot_loads)
        key = (truck_id, shift)
        
        # Lấy max finish time
        if finish > finish_times[key][0]:
            finish_times[key] = (finish, depot)
            
    return finish_times

def cleanup_inter_factory_routes(solution):
    """
    Lọc bỏ toàn bộ các tuyến INTER-FACTORY khỏi lịch trình.
    Trả xe về trạng thái sẵn sàng để PPO tối ưu hóa Farm.
    """
    if not solution or not solution.schedule:
        return solution
        
    # Chỉ giữ lại các tuyến Farm (Route có shift KHÁC 'INTER-FACTORY')
    filtered_schedule = [r for r in solution.schedule if r[3] != 'INTER-FACTORY']
    solution.schedule = filtered_schedule
    return solution