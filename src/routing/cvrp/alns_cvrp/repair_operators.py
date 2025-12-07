import copy
import random
import numpy as np
import re
from collections import defaultdict
from functools import partial
import itertools
import time
from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id, _check_insertion_efficiency,\
      _check_insertion_delta, _calculate_route_schedule_WITH_SLACK, _check_accessibility, balance_depot_loads, calculate_route_finish_time
# ==============================================================================
# HÀM TIỆN ÍCH CHUNG (Không thay đổi)
# ==============================================================================

# --- HÀM TRỢ GIÚP: TÌM VỊ TRÍ TỐT NHẤT CHO MỘT FARM ---

# ==============================================================================
# TOÁN TỬ SỬA CHỮA CHÍNH (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================
def _find_all_inserts_for_visit(schedule_list, visit_id, problem_instance, truck_finish_times):
    """
    Tìm tất cả vị trí chèn có thể cho 1 khách hàng.
    """
    all_insertions = []
    WAIT_COST_PER_MIN = 1.0 # Cần đồng bộ với config của bạn
    HUGE_PENALTY = 1e9
    
    farm_idx, farm_details, farm_demand = _get_farm_info(visit_id, problem_instance)
    if farm_idx is None: return []

    # ------------------------------------------------------------------
    # 1. CHÈN VÀO TUYẾN HIỆN CÓ
    # ------------------------------------------------------------------
    for route_idx, route_info in enumerate(schedule_list): 
        depot_idx, truck_id, customer_list, shift, start_time, finish_time, current_load = route_info
        
        if shift == 'INTER-FACTORY': continue
        
        truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
        if not truck_info: continue
        
        # Check nhanh Capacity
        if current_load + farm_demand > truck_info['capacity']: continue
        
        # Check nhanh Accessibility
        depot_details = problem_instance['facilities'][depot_idx]
        if not _check_accessibility(truck_info, farm_details, depot_details): continue

        # Tính toán Slack
        is_feasible_orig, _, _, original_schedule = _calculate_route_schedule_WITH_SLACK(
            depot_idx, customer_list, shift, start_time, problem_instance, truck_info
        )
        
        if not is_feasible_orig: continue
            
        # Thử chèn vào từng vị trí
        for insert_pos in range(len(original_schedule) - 1): 
            is_feasible, cost_increase = _check_insertion_delta(
                problem_instance, route_info, original_schedule, 
                insert_pos, visit_id, truck_info, current_load
            )
            if is_feasible:
                all_insertions.append({
                    'cost': cost_increase, 
                    'route_idx': route_idx, 
                    'pos': insert_pos, 
                    'shift': shift, 
                    'new_route_details': None
                })
    
    # ------------------------------------------------------------------
    # 2. TẠO TUYẾN MỚI (Smart Repair)
    # ------------------------------------------------------------------
    closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx]))
    depot_region = problem_instance['facilities'][closest_depot_idx].get('region', None)
    
    available_trucks = problem_instance['fleet']['available_trucks']
    
    # Tìm xe phù hợp nhất (Logic đơn giản hóa để chạy nhanh)
    best_truck = None
    min_cap_diff = float('inf')

    for truck in available_trucks:
        if truck.get('region') != depot_region: continue
        if truck['capacity'] < farm_demand: continue
        
        # Check accessibility
        if not _check_accessibility(truck, farm_details, problem_instance['facilities'][closest_depot_idx]):
            continue
            
        # Ưu tiên xe nhỏ nhất đủ tải
        diff = truck['capacity'] - farm_demand
        if diff < min_cap_diff:
            min_cap_diff = diff
            best_truck = truck

    if best_truck:
        var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get((best_truck['type'], best_truck['region']), 1.0)
        velocity = 1.0 if best_truck['type'] in ["Single", "Truck and Dog"] else 0.5
        dist_depot_farm = problem_instance['distance_depots_farms'][closest_depot_idx, farm_idx]
        travel_time = dist_depot_farm / velocity
        
        for shift in ['AM', 'PM']:
            
            # 1. Lấy thông tin chuyến trước của xe này
            key = (best_truck['id'], shift)
            
            # truck_finish_times lưu: (finish_time, finish_depot_idx)
            last_finish_time, last_finish_depot = truck_finish_times[key]
            
            # ------------------------------------------------------------------
            # [FIX] TÍNH THỜI GIAN DI CHUYỂN GIỮA CÁC KHO (INTER-DEPOT TRAVEL)
            # ------------------------------------------------------------------
            inter_depot_travel_time = 0.0
            
            # Điều kiện: Xe đã chạy chuyến trước (time > 0) VÀ Depot kết thúc khác Depot mới
            if last_finish_time > 0 and last_finish_depot != -1 and last_finish_depot != closest_depot_idx:
                
                # Lấy khoảng cách thực tế từ Matrix
                dist_between_depots = problem_instance['distance_matrix_depots'][last_finish_depot, closest_depot_idx]
                
                # Tính thời gian di chuyển
                inter_depot_travel_time = dist_between_depots / velocity
                
                # (Optional) Debug log nếu cần kiểm tra
                # print(f"Truck {best_truck['id']}: Moving Depot {last_finish_depot} -> {closest_depot_idx}. Time: {inter_depot_travel_time:.1f}m")

            # Thời gian xe thực sự sẵn sàng tại KHO MỚI
            # = Giờ xong chuyến trước + Thời gian chạy sang kho này + 1 phút buffer
            actual_vehicle_ready_time = last_finish_time + inter_depot_travel_time + 1

            # ------------------------------------------------------------------
            
            # 2. Tính thời gian bắt đầu dựa trên Time Window của khách hàng
            tw_open, _ = farm_details['time_windows'][shift]
            
            # Để đến kịp giờ mở cửa, xe phải xuất phát lúc:
            start_time_based_on_tw = tw_open - travel_time

            # 3. Start time thực tế = Max(Yêu cầu của Farm, Khả năng của Xe)
            start_time_at_depot = max(start_time_based_on_tw, actual_vehicle_ready_time)
            
            # 4. Check tính khả thi
            # Lưu ý: Hàm calculate của bạn đã sửa trả về 6 giá trị (có finish_time)
            is_feas, new_dist, new_wait, t_pen, c_pen= _calculate_route_schedule_and_feasibility(
                closest_depot_idx, 
                [visit_id], 
                shift, 
                start_time_at_depot, 
                0, 0, # Dummy finish/load
                problem_instance, 
                best_truck
            )

            if is_feas:
                # Cộng Penalty vào Cost để Regret so sánh công bằng
                HUGE_PENALTY = 1e9
                base_cost = (new_dist * var_cost_per_km) + (new_wait * WAIT_COST_PER_MIN)
                penalty_cost = (t_pen * HUGE_PENALTY) + (c_pen * HUGE_PENALTY)
                total_cost = base_cost + penalty_cost

                all_insertions.append({
                    'cost': total_cost, 
                    'route_idx': -1, 
                    'pos': start_time_at_depot, 
                    'shift': shift,
                    # Tuple đầy đủ thông tin để tạo tuyến
                    'new_route_details': (closest_depot_idx, best_truck['id'], shift, start_time_at_depot)
                })
    all_insertions.sort(key=lambda x: x['cost'])
    return all_insertions

# ==============================================================================
# CÁC TOÁN TỬ SỬA CHỮA (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================
#! BEST_INSERTION SẼ CHO RA KẾT QUẢ TỐT NHẤT
def best_insertion(current, random_state, **kwargs):
    """
    Best Insertion (Robust Load Fix).
    Sửa lỗi Phantom Load bằng cách tính lại tổng demand từ đầu mỗi khi chèn.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers_set = set(kwargs['unvisited_customers'])
    failed_customers = []
    
    # Lấy map ảo để tra cứu demand split
    virtual_map = problem_instance.get('virtual_split_farms', {})
    

    # ==========================================================
    # 1. KHỞI TẠO truck_finish_times
    # ==========================================================
    truck_finish_times = defaultdict(lambda: (0.0, -1))
    for route_info in repaired.schedule:
        depot, truck_id, cust_list, shift, start, finish, load = route_info
        if not cust_list or shift == 'INTER-FACTORY': continue
        key = (truck_id, shift)
        if finish > truck_finish_times[key][0]:
            truck_finish_times[key] = (finish, depot)

    # ==========================================================
    # 2. PHASE 1 & 2: TÍNH TOÁN VÀ SẮP XẾP
    # ==========================================================
    all_best_insertions = []
    for farm_id in unserved_customers_set: 
        insertions = _find_all_inserts_for_visit(
            repaired.schedule, farm_id, problem_instance, truck_finish_times
        ) 
        if not insertions:
            failed_customers.append(farm_id)
            continue
        
        best_insert = insertions[0]
        all_best_insertions.append((best_insert['cost'], farm_id, best_insert))

    # Clean up set
    for f in failed_customers:
        if f in unserved_customers_set: unserved_customers_set.remove(f)

    # Sort: Chi phí thấp nhất lên đầu
    all_best_insertions.sort(key=lambda x: x[0])

    # ==========================================================
    # 3. PHASE 3: CHÈN VÀ CẬP NHẬT (ĐÃ SỬA LOGIC LOAD)
    # ==========================================================
    dirty_routes = set() 
    
    for _, farm_id, initial_details in all_best_insertions:
        if farm_id not in unserved_customers_set: continue

        details_to_use = initial_details
        route_idx_targeted = initial_details['route_idx']

        # --- A. RE-EVALUATE IF DIRTY (Kiểm tra nếu tuyến đã bị đổi) ---
        if route_idx_targeted != -1 and route_idx_targeted in dirty_routes:
            new_insertions = _find_all_inserts_for_visit(
                repaired.schedule, farm_id, problem_instance, truck_finish_times
            )
            if not new_insertions: 
                failed_customers.append(farm_id)
                unserved_customers_set.remove(farm_id)
                continue 
            details_to_use = new_insertions[0]
            route_idx_targeted = details_to_use['route_idx']

        # --- B. CHUẨN BỊ DỮ LIỆU ---
        final_cust_list = []
        target_depot = None
        target_truck = None
        target_shift = None
        target_start = None
        actual_route_idx = -1

        if details_to_use['route_idx'] == -1:
            # Tạo tuyến mới
            target_depot, target_truck, target_shift, target_start = details_to_use['new_route_details']
            final_cust_list = [farm_id]
            actual_route_idx = -1 # Append
        else:
            # Chèn tuyến cũ
            route_data = repaired.schedule[route_idx_targeted]
            target_depot, target_truck, old_list, target_shift, target_start, _, _ = route_data
            
            final_cust_list = list(old_list)
            pos = details_to_use['pos']
            if pos > len(final_cust_list): pos = len(final_cust_list)
            final_cust_list.insert(pos, farm_id)
            actual_route_idx = route_idx_targeted

        # ==================================================================
        # 4. [QUAN TRỌNG] TÍNH LẠI LOAD TỪ CON SỐ 0 (FIX PHANTOM LOAD)
        # ==================================================================
        recalc_load = 0
        for c in final_cust_list:
            # 1. Ưu tiên lấy demand ảo (Split)
            if c in virtual_map:
                recalc_load += virtual_map[c]['portion']
            else:
                # 2. Lấy demand gốc
                base_id = str(c).split('_')[0]
                # Tìm index
                f_idx = problem_instance['farm_id_to_idx_map'].get(base_id)
                if f_idx is None: 
                    f_idx = problem_instance['farm_id_to_idx_map'].get(int(base_id))
                
                if f_idx is not None:
                    recalc_load += problem_instance['farms'][f_idx]['demand']
        # ==================================================================

        # Tính Finish Time chuẩn (để hàm optimize sau này có dữ liệu đúng mà chạy)
        truck_info = find_truck_by_id(target_truck, problem_instance['fleet']['available_trucks'])
        recalc_finish = calculate_route_finish_time(
            target_depot, final_cust_list, target_shift, target_start, problem_instance, truck_info
        )
        
        # Đóng gói tuple mới
        new_route_tuple = (target_depot, target_truck, final_cust_list, target_shift, target_start, recalc_finish, recalc_load)

        # Cập nhật vào Schedule
        if actual_route_idx == -1:
            repaired.schedule.append(new_route_tuple)
            current_idx = len(repaired.schedule) - 1
            dirty_routes.add(current_idx)
        else:
            repaired.schedule[actual_route_idx] = new_route_tuple
            dirty_routes.add(actual_route_idx)
        
        # Update Metadata
        truck_finish_times[(target_truck, target_shift)] = (recalc_finish, target_depot)
        unserved_customers_set.remove(farm_id)

    # Xử lý Failed
    if unserved_customers_set:
        failed_customers.extend(list(unserved_customers_set))
    
    return repaired, failed_customers
#! REGRET_K_INSERTION CHỈ CHO RA KẾT QUẢ XẤP XỈ --> CÓ THỂ VI PHẠM CONSTRAINTS
# --- HELPER: GOM NHÓM ỨNG VIÊN ---
def _filter_candidates_by_mode(all_insertions, mode, repaired_schedule):
    """
    Lọc danh sách chèn dựa trên chế độ Regret.
    mode: 'position' (default), 'trip', 'vehicle'
    """
    if mode == 'position':
        return all_insertions # Không lọc, trả về tất cả khe

    # Dictionary để lưu Best Cost cho mỗi nhóm
    # Key sẽ là (truck_id, shift) cho 'trip' hoặc (truck_id) cho 'vehicle'
    best_per_group = {} 

    for option in all_insertions:
        # 1. Trích xuất thông tin Truck và Shift từ option
        truck_id = None
        shift = None
        
        if option['route_idx'] != -1:
            # Tuyến có sẵn: Lấy từ schedule
            route_info = repaired_schedule[option['route_idx']]
            truck_id = route_info[1] # index 1 là truck_id
            shift = route_info[3]    # index 3 là shift
        else:
            # Tuyến mới: Lấy từ new_route_details
            # new_route_details: (depot, truck_id, shift, start_time)
            truck_id = option['new_route_details'][1]
            shift = option['new_route_details'][2]

        # 2. Xác định Key Gom Nhóm (Group Key)
        if mode == 'trip':
            group_key = (truck_id, shift)
        elif mode == 'vehicle':
            group_key = truck_id
        else:
            continue # Should not happen

        # 3. Giữ lại lựa chọn tốt nhất cho nhóm này
        if group_key not in best_per_group:
            best_per_group[group_key] = option
        else:
            if option['cost'] < best_per_group[group_key]['cost']:
                best_per_group[group_key] = option
    
    # Trả về danh sách các đại diện tốt nhất của từng nhóm
    return list(best_per_group.values())

# --- HÀM CHÍNH: REGRET K INSERTION (ĐA CHẾ ĐỘ) ---
def regret_k_insertion(current, random_state, k_regret=2, mode='position', **kwargs):
    """
    Regret Insertion Robust Version.
    Hỗ trợ mode: 'position', 'trip', 'vehicle'.
    Có cơ chế Fallback để không bao giờ bỏ rơi khách hàng nếu còn chỗ chèn.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers_set = set(kwargs.get('unvisited_customers', []))
    failed_customers = []
    # 1. Xây dựng bản đồ thời gian xe (cho Multi-trip)
    truck_finish_times = defaultdict(lambda: (0.0, -1))
    for route_info in repaired.schedule:
        depot, truck_id, cust_list, shift, start, finish, load = route_info
        if not cust_list or shift == 'INTER-FACTORY': continue
        key = (truck_id, shift)
        if finish > truck_finish_times[key][0]:
            truck_finish_times[key] = (finish, depot)

    # Helper lọc candidates (nhúng vào đây hoặc để ngoài đều được)
    def filter_candidates(candidates, mode, schedule):
        if mode == 'position': return candidates
        best_per_group = {}
        for opt in candidates:
            if opt['route_idx'] != -1:
                r = schedule[opt['route_idx']]
                grp = (r[1], r[3]) if mode == 'trip' else r[1] # (Truck, Shift) or Truck
            else:
                d = opt['new_route_details']
                grp = (d[1], d[2]) if mode == 'trip' else d[1]
            
            if grp not in best_per_group or opt['cost'] < best_per_group[grp]['cost']:
                best_per_group[grp] = opt
        return sorted(list(best_per_group.values()), key=lambda x: x['cost'])

    # -------------------------------------------------------
    # PHASE 1: TÍNH REGRET BAN ĐẦU
    # -------------------------------------------------------
    all_regret_options = []
    
    for farm_id in list(unserved_customers_set):
        raw_opts = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance, truck_finish_times)
        if not raw_opts: continue
        
        # Lọc theo mode
        final_opts = filter_candidates(raw_opts, mode, repaired.schedule)
        
        # Tính Regret-K
        best_opt = final_opts[0]
        regret_val = 0
        limit = min(len(final_opts), k_regret)
        
        if limit > 1:
            for i in range(1, limit):
                regret_val += (final_opts[i]['cost'] - best_opt['cost'])
        else:
            regret_val = float('inf') # Vô cực nếu chỉ có 1 lựa chọn (Khan hiếm)
            
        all_regret_options.append({'regret': regret_val, 'farm_id': farm_id, 'opt': best_opt})

    # Sort giảm dần theo Regret (Ưu tiên xử lý ca khó trước)
    all_regret_options.sort(key=lambda x: x['regret'], reverse=True)

    # -------------------------------------------------------
    # PHASE 2: CHÈN VÀ CẬP NHẬT
    # -------------------------------------------------------
    dirty_routes = set()
    
    # Duyệt qua danh sách Regret đã sort
    # Lưu ý: Ta dùng while loop hoặc copy list vì ta có thể phải tính lại
    queue = all_regret_options
    
    while queue:
        # Lấy ứng viên có Regret lớn nhất
        current_item = queue.pop(0)
        farm_id = current_item['farm_id']
        
        # Nếu đã được xử lý ở đâu đó rồi (hiếm gặp)
        if farm_id not in unserved_customers_set: continue
        
        target_opt = current_item['opt']
        target_route_idx = target_opt['route_idx']
        
        # KIỂM TRA DIRTY (Nếu tuyến đích đã bị thay đổi bởi bước chèn trước)
        # Logic: Nếu route_idx nằm trong dirty set HOẶC tạo tuyến mới (luôn check lại cho chắc với Multi-trip)
        is_dirty = False
        if target_route_idx != -1 and target_route_idx in dirty_routes:
            is_dirty = True
        elif target_opt['new_route_details']: 
            # Với tuyến mới, ta cần check xem xe đó có bị update thời gian chưa
            tid = target_opt['new_route_details'][1]
            shift = target_opt['new_route_details'][2]
            # Logic đơn giản: Luôn coi tạo tuyến mới là dirty để tính lại start_time cho chuẩn
            is_dirty = True 

        if is_dirty:
            # TÍNH LẠI TỪ ĐẦU CHO KHÁCH NÀY
            raw_opts = _find_all_inserts_for_visit(repaired.schedule, farm_id, problem_instance, truck_finish_times)
            
            if not raw_opts:
                failed_customers.append(farm_id)
                unserved_customers_set.remove(farm_id)
                continue
            
            # Lọc lại
            final_opts = filter_candidates(raw_opts, mode, repaired.schedule)
            
            # [FALLBACK QUAN TRỌNG]: Nếu lọc xong mà rỗng (do mode quá gắt), lấy raw
            if not final_opts:
                final_opts = raw_opts
            
            # Cập nhật target mới tốt nhất
            target_opt = final_opts[0]
            target_route_idx = target_opt['route_idx']

        # --- THỰC HIỆN CHÈN ---
        
        # A. Tạo tuyến mới
                # Biến tạm để lưu thông tin trước khi update
        final_depot = None
        final_truck = None
        final_cust_list = []
        final_shift = None
        final_start = None
        target_idx_in_schedule = -1

        # A. Chuẩn bị dữ liệu
        if target_route_idx == -1:
            # Trường hợp: TẠO TUYẾN MỚI
            final_depot, final_truck, final_shift, final_start = target_opt['new_route_details']
            final_cust_list = [farm_id]
            target_idx_in_schedule = -1 # Đánh dấu là append
        else:
            # Trường hợp: CHÈN VÀO TUYẾN CŨ
            route_data = list(repaired.schedule[target_route_idx])
            final_depot, final_truck, old_cust_list, final_shift, final_start, _, _ = route_data
            
            final_cust_list = list(old_cust_list)
            pos = target_opt['pos']
            # Bảo vệ index
            if pos > len(final_cust_list): pos = len(final_cust_list)
            final_cust_list.insert(pos, farm_id)
            target_idx_in_schedule = target_route_idx

        # ==================================================================
        # 🔧 [FIX LỖI LOAD]: TÍNH TỔNG LẠI TỪ ĐẦU (RESET = 0)
        # ==================================================================
        recalc_load = 0
        virtual_map = problem_instance.get('virtual_split_farms', {})
        
        for c in final_cust_list:
            # 1. Nếu là khách ảo (Split Demand)
            if c in virtual_map:
                recalc_load += virtual_map[c]['portion']
            else:
                # 2. Nếu là khách thường -> Lấy demand gốc
                # (Dùng try-except hoặc logic map an toàn)
                base_id = str(c).split('_')[0]
                # Tìm index trong map
                f_idx = problem_instance['farm_id_to_idx_map'].get(base_id)
                if f_idx is None: 
                    f_idx = problem_instance['farm_id_to_idx_map'].get(int(base_id))
                
                if f_idx is not None:
                    recalc_load += problem_instance['farms'][f_idx]['demand']
                else:
                    print(f"⚠️ Cảnh báo: Không tìm thấy demand cho {c}")
        
        # ==================================================================

        # Tính lại Finish Time (Cần thiết để update schedule)
        truck_info = find_truck_by_id(final_truck, problem_instance['fleet']['available_trucks'])
        recalc_finish = calculate_route_finish_time(
            final_depot, final_cust_list, final_shift, final_start, problem_instance, truck_info
        )

        # Đóng gói tuple mới (Lúc này recalc_load đã CHUẨN 100%)
        new_route_tuple = (final_depot, final_truck, final_cust_list, final_shift, final_start, recalc_finish, recalc_load)

        # Cập nhật vào Schedule
        if target_idx_in_schedule == -1:
            repaired.schedule.append(new_route_tuple)
            dirty_routes.add(len(repaired.schedule) - 1)
        else:
            repaired.schedule[target_idx_in_schedule] = new_route_tuple
            dirty_routes.add(target_idx_in_schedule)
            
        # Update metadata finish time
        truck_finish_times[(final_truck, final_shift)] = (recalc_finish, final_depot)
        
        # Đánh dấu xong khách hàng này
        unserved_customers_set.remove(farm_id)

    return repaired, failed_customers
regret_2_position = partial(regret_k_insertion, k_regret=2, mode='position')

# 2. Regret-2 Trip (Tốt cho Multi-trip)
regret_2_trip = partial(regret_k_insertion, k_regret=2, mode='trip')

# 3. Regret-2 Vehicle (Tốt cho khan hiếm xe/Region)
regret_2_vehicle = partial(regret_k_insertion, k_regret=2, mode='vehicle')

# 4. Regret-3 Trip (Nhìn xa hơn chút)
regret_3_position = partial(regret_k_insertion, k_regret=3, mode='position')

regret_3_trip = partial(regret_k_insertion, k_regret=3, mode='trip')

regret_3_vehicle = partial(regret_k_insertion, k_regret=3, mode='vehicle')

regret_4_position = partial(regret_k_insertion, k_regret=4, mode='position')

regret_4_trip = partial(regret_k_insertion, k_regret=4, mode='trip')

regret_4_vehicle = partial(regret_k_insertion, k_regret=4, mode='vehicle')




