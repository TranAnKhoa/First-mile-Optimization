import copy
import random
import numpy as np
import re
from collections import defaultdict
from .utils import _remove_customers_from_schedule, get_route_cost, _get_farm_info,_calculate_route_schedule_and_feasibility, find_truck_by_id, _get_dist_between_nodes, calculate_route_finish_time
# ==============================================================================
# HÀM TIỆN ÍCH CHUNG (Không thay đổi)
# ==============================================================================

# =============================================================================
# CÁC TOÁN TỬ PHÁ HỦY (VIẾT LẠI CHO SINGLE-DAY VRP)
# ==============================================================================

import copy
import numpy as np

# Tham số dùng chung (nếu bạn có config chung, bạn có thể chuyển xuống file config)
WAIT_COST_PER_MIN = 0.2
TIME_PENALTY = 0.3

# Trong file destroy_operators.py

# ... (các hàm khác không thay đổi) ...

def update_solution_state_after_destroy(solution):
    """
    Cập nhật lại Route Load và Finish Time cho toàn bộ Schedule.
    Dùng để gọi ngay sau khi Destroy xóa khách hàng.
    """
    problem = solution.problem_instance
    # Cache các biến hay dùng để tăng tốc độ truy xuất
    virtual_map = problem.get('virtual_split_farms', {})
    farm_map = problem['farm_id_to_idx_map']
    farms_data = problem['farms']
    all_trucks = problem['fleet']['available_trucks']
    
    # Tạo map truck_id -> truck_info để đỡ phải find trong vòng lặp
    truck_info_map = {t['id']: t for t in all_trucks}

    new_schedule = []

    for route in solution.schedule:
        # Unpack 7 phần tử (Load cũ và Finish cũ sẽ bị ghi đè)
        depot, truck_id, cust_list, shift, start, _, _ = route

        # Nếu route rỗng hoặc Inter-factory thì giữ nguyên (hoặc reset load về 0)
        if not cust_list or shift == 'INTER-FACTORY':
            # Load = 0 cho chắc chắn nếu không có khách
            # Finish time giữ nguyên start + duration (hoặc logic riêng), tạm thời giữ nguyên
            current_load = route[6] if shift == 'INTER-FACTORY' else 0
            current_finish = route[5]
            new_schedule.append((depot, truck_id, cust_list, shift, start, current_finish, current_load))
            continue

        # --- 1. TÍNH LẠI LOAD (Siêu nhanh) ---
        new_load = 0
        for c in cust_list:
            # Check nhanh trong virtual map trước
            if c in virtual_map:
                new_load += virtual_map[c]['portion']
            else:
                # Check farm thường
                # Dùng str.split để lấy base id nhanh
                base = str(c).split('_')[0]
                # Tra cứu index (thử str rồi thử int)
                f_idx = farm_map.get(base)
                if f_idx is None: f_idx = farm_map.get(int(base))
                
                if f_idx is not None:
                    new_load += farms_data[f_idx]['demand']

        # --- 2. TÍNH LẠI FINISH TIME ---
        truck_info = truck_info_map.get(truck_id)
        if truck_info:
            # Gọi hàm tính toán có sẵn của bạn
            new_finish = calculate_route_finish_time(
                depot, cust_list, shift, start, problem, truck_info
            )
        else:
            new_finish = route[5] # Fallback (hiếm)

        # Lưu lại tuple mới đã làm sạch
        new_schedule.append((depot, truck_id, cust_list, shift, start, new_finish, new_load))

    solution.schedule = new_schedule
    return solution

def random_removal(current, random_state, **kwargs):
    """
    Xóa ngẫu nhiên các farm_id khỏi lịch trình.
    *** PHIÊN BẢN ĐÃ SỬA LỖI ValueError (logic 5 phần tử) ***
    """
    destroyed = copy.deepcopy(current)
    
    # <<< SỬA LỖI Ở ĐÂY: GIẢI NÉN 5 PHẦN TỬ >>>
    # Lấy danh sách tất cả các visit có thể xóa
    all_visits = []  # 1. Tạo một danh sách rỗng
    # 2. Vòng lặp ngoài: Lặp qua từng chuyến xe trong lịch trình
    for route_info in destroyed.schedule:
        # 3. Giải nén 5 phần tử của chuyến xe
        _, _, cust_list, shift, _,_,_ = route_info
        # 4. Điều kiện lọc: Chỉ xử lý nếu không phải là chuyến 'INTER-FACTORY'
        if shift != 'INTER-FACTORY':
            # 5. Vòng lặp trong: Lặp qua từng farm_id trong chuyến xe này
            for fid in cust_list:
                # 6. Thêm farm_id vào danh sách kết quả
                all_visits.append(fid)
    

    if not all_visits:
        return destroyed, []
    
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_visits) * 0.15))) # Lấy ra 'num_to_move', nếu k có thì lấy phá 15%
    num_to_remove = min(num_to_remove, len(all_visits))
    
    customers_to_remove = random.sample(all_visits, num_to_remove)
    
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, customers_to_remove)
    
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, customers_to_remove



def worst_removal(current, random_state, alpha=0, **kwargs):
    """
    IMPROVED worst removal (Fixed & Adaptive Ready):
    - alpha: Trọng số phạt (Penalty Weight). 
             Nếu alpha=0: Chỉ quan tâm Distance.
             Nếu alpha cao: Ưu tiên xóa khách gây Waiting Time/Infeasible lớn.
    """
    destroyed = copy.deepcopy(current)
    problem_instance = destroyed.problem_instance
    
    # Lấy thông tin chi phí từ problem_instance (nếu không có thì dùng default)
    # Giả sử wait cost nhỏ vì ta sẽ dùng alpha để scale nó lên
    BASE_WAIT_COST = 1.0 
    HUGE_PENALTY = 1e9

    removed_customers = []

    # Đếm số lượng visits thực tế
    num_visits = sum(len(r[2]) for r in current.schedule if r[3] != 'INTER-FACTORY')
    if num_visits == 0:
        return destroyed, []

    default_frac = kwargs.get('remove_fraction', 0.20)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(num_visits * default_frac)))
    num_to_remove = min(num_to_remove, num_visits)
    
    power = kwargs.get('selection_power', 4)

    for _ in range(num_to_remove):
        savings_list = []
        
        # Duyệt qua từng tuyến đường để tìm ứng viên
        for route_idx, route_info in enumerate(destroyed.schedule):
            depot_idx, truck_id, customer_list, shift, start_time, finish_time, route_load = route_info
            
            # Bỏ qua tuyến rỗng hoặc inter-factory
            if not customer_list or shift == 'INTER-FACTORY':
                continue

            truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
            if truck_info is None:
                continue

            # --- BƯỚC 1: Tính Cost của tuyến CŨ (Baseline) ---
            # Chỉ tính 1 lần cho mỗi tuyến
            feasible_old, old_dist, old_wait, old_time_pen, old_cap_pen = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, start_time, finish_time, route_load, problem_instance, truck_info
            )
            
            if not feasible_old:
                # Nếu bản thân tuyến cũ đã lỗi, ta càng muốn sửa nó -> Coi cost cũ là rất lớn
                old_dist_cost = old_dist # Vẫn giữ dist
                old_penalty_cost = HUGE_PENALTY # Phạt nặng
            else:
                var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                    (truck_info['type'], truck_info['region']), 1.0
                )
                old_dist_cost = old_dist * var_cost_per_km
                # Penalty bao gồm: Waiting Time + Các loại phạt vi phạm
                old_penalty_cost = (old_wait * BASE_WAIT_COST) + (old_time_pen * HUGE_PENALTY) + (old_cap_pen * HUGE_PENALTY)

            # --- BƯỚC 2: Thử xóa từng khách hàng ---
            for pos, farm_to_remove in enumerate(customer_list):
                # Tạo danh sách mới (đã xóa farm_to_remove)
                temp_list = customer_list[:pos] + customer_list[pos+1:]
                
                if not temp_list:
                    # Nếu xóa xong mà tuyến rỗng -> Cost = 0
                    new_dist_cost = 0
                    new_penalty_cost = 0
                else:
                    # [FIX] Truyền temp_list vào hàm tính toán (Code cũ truyền customer_list -> Sai logic)
                    feasible_new, new_dist, new_wait, new_time_pen, new_cap_pen = _calculate_route_schedule_and_feasibility(
                        depot_idx, temp_list, shift, start_time, finish_time, route_load, problem_instance, truck_info
                    )
                    
                    # Tính toán Cost Mới
                    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get((truck_info['type'], truck_info['region']), 1.0)
                    new_dist_cost = new_dist * var_cost_per_km
                    
                    if not feasible_new:
                        # Xóa bớt khách mà lại thành infeasible (hiếm gặp) -> Phạt
                        new_penalty_cost = HUGE_PENALTY
                    else:
                        new_penalty_cost = (new_wait * BASE_WAIT_COST) + (new_time_pen * HUGE_PENALTY) + (new_cap_pen * HUGE_PENALTY)

                # --- BƯỚC 3: Tính Saving với ALPHA ---
                dist_saving = old_dist_cost - new_dist_cost
                penalty_saving = old_penalty_cost - new_penalty_cost
                
                # CÔNG THỨC QUAN TRỌNG:
                # Score = (Tiết kiệm Distance) + alpha * (Tiết kiệm Penalty/Wait)
                # Nếu alpha = 0: Chỉ quan tâm dist_saving
                # Nếu alpha lớn: penalty_saving sẽ chi phối
                total_saving = dist_saving + (alpha * penalty_saving)

                savings_list.append({
                    'saving': total_saving,
                    'farm_id': farm_to_remove,
                    'route_idx': route_idx
                })

        if not savings_list:
            break

        # --- BƯỚC 4: Chọn và Xóa ---
        # Sort giảm dần theo saving
        savings_list.sort(key=lambda x: x['saving'], reverse=True)

        # Chọn ngẫu nhiên có định hướng (Roulette Wheel / Power Bias)
        r = random_state.random()
        idx = int(len(savings_list) * (r ** power))
        idx = max(0, min(len(savings_list)-1, idx))

        chosen_farm_id = savings_list[idx]['farm_id']
        
        # Check trùng (đề phòng)
        if chosen_farm_id in removed_customers:
            continue

        # Thực hiện xóa trong destroyed.schedule
        destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, [chosen_farm_id])
        removed_customers.append(chosen_farm_id)
        
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, removed_customers

def worst_removal_alpha_0(current, random_state, **kwargs):
    return worst_removal(current, random_state, **kwargs)
def worst_removal_bigM(current, random_state, **kwargs):
    return worst_removal(current, random_state, penalty_alpha=1_000_000, **kwargs)
def worst_removal_adaptive(current, random_state, **kwargs):
    # Lấy số lượng khách cần xóa (đã được tính ở main loop)
    # Giả sử bạn truyền remove_fraction vào kwargs, hoặc tính trực tiếp
    remove_fraction = kwargs.get('remove_fraction', 0.1)
    
    # --- LOGIC CỦA BẠN Ở ĐÂY ---
    # Map ngược: Xóa càng nhiều (fraction lớn) -> Alpha càng nhỏ (để khám phá)
    # Ví dụ: fraction đi từ 0.3 (nhiều) về 0.05 (ít)
    # Alpha đi từ 0 (lỏng) lên 500 (chặt)
    
    min_frac, max_frac = 0.05, 0.3
    min_alpha, max_alpha = 10, 500
    
    # Normalize fraction về 0-1
    ratio = (remove_fraction - min_frac) / (max_frac - min_frac)
    ratio = max(0, min(1, ratio)) # Kẹp trong [0, 1]
    
    # Tính Alpha (nghịch đảo)
    # Ratio càng cao (xóa nhiều) -> (1-ratio) càng nhỏ -> Alpha nhỏ
    current_alpha = min_alpha + (max_alpha - min_alpha) * (1 - ratio)
    
    return worst_removal(current, random_state, penalty_alpha=int(current_alpha), **kwargs)

# ... (Các hàm khác không thay đổi) ...

def shaw_removal(current, random_state, w_dist=1.0, w_tw=0.5, w_depot=20.0, w_access=25.0, **kwargs):
    """
    Hàm CORE Shaw Removal (ĐÃ FIX LỖI INDEX).
    """
    destroyed = copy.deepcopy(current)
    problem = destroyed.problem_instance
    
    all_visits = [
        (cust, depot_idx, shift, start_time)
        for depot_idx, _, custs, shift, start_time, finish_time, route_load in destroyed.schedule
        for cust in custs if shift != 'INTER-FACTORY'
    ]

    if not all_visits:
        return destroyed, []
        
    default_frac = kwargs.get('remove_fraction', 0.15)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_visits) * default_frac)))
    num_to_remove = min(num_to_remove, len(all_visits))

    power = kwargs.get('selection_power', 3) 

    seed_visit = all_visits[random_state.randint(len(all_visits))]
    removed = {seed_visit}
    remaining = set(all_visits) - removed

    dist_mat = problem['distance_matrix_farms']
    
    def tw_overlap(f1_det, f2_det):
        overlap = 0
        for shift in ['AM', 'PM']:
            a1, b1 = f1_det['time_windows'][shift]
            a2, b2 = f2_det['time_windows'][shift]
            overlap += max(0, min(b1, b2) - max(a1, a2))
        return overlap

    while len(removed) < num_to_remove and remaining:
        ref_visit = random.choice(list(removed))
        ref_farm_id = ref_visit[0] # Đây là ID (VD: 8461)
        
        # --- [QUAN TRỌNG] LẤY INDEX TỪ HÀM HELPER ---
        # ref_idx mới là số thứ tự trong ma trận (0, 1, 2...)
        ref_idx, ref_det, _ = _get_farm_info(ref_farm_id, problem)
        
        scores = []
        for cand_visit in list(remaining):
            cand_farm_id = cand_visit[0] # ID
            
            # --- [QUAN TRỌNG] LẤY INDEX CHO ỨNG VIÊN ---
            cand_idx, cand_det, _ = _get_farm_info(cand_farm_id, problem)
            
            # 1. Distance (SỬA LỖI TẠI ĐÂY)
            term_dist = 0
            if w_dist > 0:
                # Dùng Index (ref_idx, cand_idx), KHÔNG DÙNG ID (ref_farm_id...)
                term_dist = dist_mat[ref_idx, cand_idx]
            
            # 2. Time Overlap
            term_tw = 0
            if w_tw > 0:
                term_tw = tw_overlap(ref_det, cand_det)
            
            # 3. Depot
            term_depot = 0
            if w_depot > 0:
                dep_ref = ref_visit[1]
                dep_cand = cand_visit[1]
                term_depot = 1.0 if dep_ref == dep_cand else 0.0

            # 4. Accessibility
            term_access = 0
            if w_access > 0:
                acc_ref = ref_det.get('accessibility', 'ANY')
                acc_cand = cand_det.get('accessibility', 'ANY')
                term_access = 1.0 if acc_ref == acc_cand else 0.0

            # Score càng thấp = Càng tương đồng
            score = (w_dist * term_dist) - (w_tw * term_tw) - (w_depot * term_depot) - (w_access * term_access)
            
            scores.append((score, cand_visit))

        if not scores: break

        scores.sort(key=lambda x: x[0])
        
        r = random_state.random()
        idx_pick = int(len(scores) * (r ** power))
        idx_pick = max(0, min(len(scores)-1, idx_pick))
        
        pick = scores[idx_pick][1]
        removed.add(pick)
        remaining.remove(pick)

    customers_to_remove_ids = [visit[0] for visit in removed]
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, customers_to_remove_ids)
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, customers_to_remove_ids

# --- 1. SPATIAL SHAW (Chỉ quan tâm Không Gian/Khoảng cách) ---
# Gom các khách hàng ở GẦN NHAU về mặt địa lý.
def shaw_spatial(current, random_state, **kwargs):
    return shaw_removal(current, random_state, 
                              w_dist=1.0, 
                              w_time=0.0, 
                              w_depot=0.0, 
                              w_access=0.0, **kwargs)

# --- 2. TEMPORAL SHAW (Chỉ quan tâm Thời Gian) ---
# Gom các khách hàng có KHUNG GIỜ TRÙNG NHAU.
def shaw_temporal(current, random_state, **kwargs):
    return shaw_removal(current, random_state, 
                              w_dist=0.0, 
                              w_time=1.0, # Chỉ bật Time
                              w_depot=0.0, 
                              w_access=0.0, **kwargs)

# --- 3. STRUCTURAL SHAW (Quan tâm Cấu trúc Hạ tầng) ---
# Gom các khách hàng CÙNG VÙNG (Depot) và CÙNG LOẠI XE (Access).
# Rất quan trọng để tối ưu việc ghép xe (Heterogeneous fleet).
def shaw_structural(current, random_state, **kwargs):
    return shaw_removal(current, random_state, 
                              w_dist=0.0, 
                              w_time=0.0, 
                              w_depot=10.0,  # Bật Depot (Bonus lớn chút vì là binary)
                              w_access=10.0, # Bật Access
                              **kwargs)

# --- 4. HYBRID SHAW (Tổng hợp - Cân bằng) ---
# Phiên bản Shaw "cổ điển" nhưng có thêm Accessibility.
def shaw_hybrid(current, random_state, **kwargs):
    return shaw_removal(current, random_state, 
                              w_dist=1.0, 
                              w_time=0.5,   # Giảm chút để không át Dist
                              w_depot=20.0, # Bonus lớn
                              w_access=20.0, 
                              **kwargs)



def time_worst_removal(current, random_state, **kwargs):
    """
    Remove the visits that have the largest individual waiting times.
    Returns (destroyed_copy, removed_list).
    """
    destroyed = copy.deepcopy(current)
    prob = destroyed.problem_instance
    visits_wait = []  # list of (wait_time, farm_id)

    # compute waiting per stop using current schedule and _calculate_route_schedule_and_feasibility
    for depot_idx, truck_id, custs, shift, start_time, finish_time, route_load in destroyed.schedule:
        if not custs or shift == 'INTER-FACTORY':
            continue
        truck = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
        if not truck:
            continue
        # get timeline by simulating (we already have helper but reuse _calculate for totals)
        # To get per-stop wait need to reconstruct timeline like simulate does:
        # We'll reuse the same logic as in _calculate_route_schedule_and_feasibility but per-stop.
        # Simpler: call helper simulate-like code here:
        current_time = start_time
        truck_name = truck['type']
        velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
        for i, fid in enumerate(custs):
            f_idx, f_det, f_dem = _get_farm_info(fid, prob)
            travel = prob['distance_depots_farms'][depot_idx, f_idx] if i == 0 else prob['distance_matrix_farms'][prev_idx, f_idx]
            travel_time = travel / velocity
            arrival = current_time + travel_time
            start_tw, _ = f_det['time_windows'][shift]
            wait = max(0, start_tw - arrival)
            visits_wait.append((wait, fid))
            # service
            fix, var = f_det['service_time_params']
            service_duration = fix + (f_dem / var if var > 0 else 0)
            current_time = arrival + wait + service_duration
            prev_idx = f_idx

    if not visits_wait:
        return destroyed, []

    # sort visits by wait desc
    visits_wait.sort(key=lambda x: x[0], reverse=True)
    default_frac = kwargs.get('remove_fraction', 0.15)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(custs
                                                               ) * default_frac)))
    to_remove = []
    while len(to_remove) < num_to_remove and visits_wait:
        # Chọn ngẫu nhiên nhưng thiên vị phần tử đầu (Wait cao)
        # r random từ 0-1. Mũ càng cao càng tập trung vào top đầu.
        r = random_state.random() 
        idx = int(len(visits_wait) * (r ** 6)) # Số 6 là độ mạnh của sự thiên vị (bias)
        idx = min(idx, len(visits_wait) - 1)
        
        item = visits_wait.pop(idx) # Lấy ra và xóa khỏi danh sách
        to_remove.append(item[1])
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, to_remove)
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, to_remove



def trip_removal(current, random_state, **kwargs):
    """
    TRIP REMOVAL (ROUTE REMOVAL)
    Chọn ngẫu nhiên các chuyến xe (Route) và xóa TOÀN BỘ khách hàng trong đó.
    Rất mạnh để thoát khỏi Local Optima trong bài toán Multi-trip.
    """
    destroyed = copy.deepcopy(current)
    
    # 1. Lấy danh sách các chuyến có thể xóa (Không rỗng, Không phải Inter-factory)
    # Lưu index của chuyến trong schedule để dễ truy cập
    candidate_trip_indices = []
    for idx, route_info in enumerate(destroyed.schedule):
        customer_list = route_info[2] # Index 2 là customer_list trong 7-tuple
        shift = route_info[3]
        if customer_list and shift != 'INTER-FACTORY':
            candidate_trip_indices.append(idx)
    
    if not candidate_trip_indices:
        return destroyed, []

    # 2. Xác định số lượng khách cần xóa mục tiêu
    # Lưu ý: Với Trip Removal, ta khó kiểm soát chính xác số lượng khách bị xóa.
    # Ta sẽ xóa từng chuyến cho đến khi đạt hoặc vượt số lượng này.
    num_visits = sum(len(r[2]) for r in destroyed.schedule if r[3] != 'INTER-FACTORY')
    default_frac = kwargs.get('remove_fraction', 0.15)
    target_remove_count = max(1, int(num_visits * default_frac))
    
    removed_customers = []
    removed_count = 0

    # Shuffle danh sách chuyến để chọn ngẫu nhiên
    random_state.shuffle(candidate_trip_indices)

    # 3. Thực hiện xóa chuyến
    for route_idx in candidate_trip_indices:
        if removed_count >= target_remove_count:
            break
            
        # Lấy thông tin chuyến
        route_data = list(destroyed.schedule[route_idx])
        cust_list = route_data[2]
        
        # Lưu lại khách hàng bị xóa
        removed_customers.extend(cust_list)
        removed_count += len(cust_list)
        
        # Xóa khách khỏi tuyến (làm rỗng list)
        route_data[2] = [] 
        
        # Cập nhật lại tuple rỗng vào schedule
        # Lưu ý: Cần tính lại metrics (start/finish) nếu cần, 
        # nhưng tuyến rỗng thường không ảnh hưởng feasibility, chỉ cần update load = 0
        # (Giả sử start/finish giữ nguyên hoặc reset về shift start tùy logic của bạn)
        # Ở đây ta chỉ cần list rỗng là đủ để Repair Operator biết.
        destroyed.schedule[route_idx] = tuple(route_data)

    # Gọi hàm helper để dọn dẹp sạch sẽ (tính lại load, time cho các tuyến rỗng nếu cần)
    # Hoặc đơn giản trả về danh sách ID để Repair chèn lại
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, removed_customers


def historical_removal(current, random_state, **kwargs):
    """
    HISTORICAL KNOWLEDGE REMOVAL (Fixed for Split Demand)
    Xóa các khách hàng có chi phí cạnh hiện tại tệ hơn nhiều so với 
    tốt nhất từng thấy trong lịch sử tìm kiếm.
    """
    history_matrix = kwargs.get('history_matrix')
    
    # Nếu chưa có lịch sử, fallback về Random
    if not history_matrix:
        from .destroy_operators import random_removal
        return random_removal(current, random_state, **kwargs)

    destroyed = copy.deepcopy(current)
    problem = destroyed.problem_instance
    dist_mat = problem['distance_matrix_farms']
    depot_dist = problem['distance_depots_farms']
    
    # Cần đảm bảo import _clean_base_id và _get_farm_info từ utils hoặc định nghĩa lại
    from .utils import _clean_base_id, _get_farm_info 

    # --- HELPER: Lấy index chuẩn (int) để tra ma trận ---
    def _get_safe_idx(node_id):
        if node_id == -1: return -1
        # _get_farm_info trả về (idx, details, demand). Lấy [0] là idx
        try:
            idx = _get_farm_info(node_id, problem)[0]
            return int(idx) # Ép kiểu int quan trọng
        except:
            return None

    # --- HELPER: Lấy key chuẩn (ID gốc) để tra history ---
    def _get_clean_key(node_id):
        if node_id == -1: return -1
        base = _clean_base_id(node_id)
        try: return int(base)
        except: return base

    all_visits = [] 
    
    # 1. Duyệt qua giải pháp
    for r_idx, route_info in enumerate(destroyed.schedule):
        # Unpack an toàn
        if len(route_info) < 3: continue
        depot_idx = route_info[0]
        customer_list = route_info[2]
        shift = route_info[3]
        
        if not customer_list or shift == 'INTER-FACTORY':
            continue
            
        # Hàm tính cost nội bộ (Thay thế get_costs cũ)
        def get_edge_costs(u, v):
            u_idx = _get_safe_idx(u)
            v_idx = _get_safe_idx(v)
            
            if u_idx is None or v_idx is None:
                return 0.0, 0.0
            
            # A. Tính Distance thực tế (Current)
            real_dist = 0.0
            if u_idx == -1:   real_dist = float(depot_dist[depot_idx, v_idx])
            elif v_idx == -1: real_dist = float(depot_dist[depot_idx, u_idx])
            else:             real_dist = float(dist_mat[u_idx, v_idx])
            
            # B. Lấy Distance lịch sử (History)
            u_key = _get_clean_key(u)
            v_key = _get_clean_key(v)
            
            hist_dist = float(history_matrix.get((u_key, v_key), float('inf')))
            
            # Nếu chưa từng thấy trong lịch sử, coi như bằng thực tế (deviation = 0)
            if hist_dist == float('inf'):
                hist_dist = real_dist
                
            return real_dist, hist_dist

        # Duyệt các khách hàng trong route
        for i, cust_id in enumerate(customer_list):
            prev_node = customer_list[i-1] if i > 0 else -1
            next_node = customer_list[i+1] if i < len(customer_list) - 1 else -1
            
            # Tính toán cho 2 cạnh nối với khách hàng này
            d1_curr, d1_hist = get_edge_costs(prev_node, cust_id)
            d2_curr, d2_hist = get_edge_costs(cust_id, next_node)
            
            current_cost = d1_curr + d2_curr
            historical_best = d1_hist + d2_hist
            
            # Tính độ lệch (Deviation)
            # Dùng float() bọc ngoài cùng để chắc chắn 100% là scalar
            deviation = float(current_cost - historical_best)
            
            all_visits.append((deviation, cust_id))

    if not all_visits:
        return destroyed, []

    # 2. Chọn lọc xóa
    default_frac = kwargs.get('remove_fraction', 0.15)
    num_to_remove = max(1, int(len(all_visits) * default_frac))
    
    # Sort an toàn (vì deviation đã là float)
    all_visits.sort(key=lambda x: x[0], reverse=True)
    
    # Chọn ngẫu nhiên bias
    power = kwargs.get('selection_power', 4)
    removed_ids = []
    
    candidates = list(all_visits)
    while len(removed_ids) < num_to_remove and candidates:
        r = random_state.random()
        idx = int(len(candidates) * (r ** power))
        idx = min(idx, len(candidates) - 1)
        
        pick = candidates.pop(idx)
        removed_ids.append(pick[1])

    # Thực hiện xóa
    # Lưu ý: _remove_customers_from_schedule phải nằm trong utils hoặc destroy_operators
    from .utils import _remove_customers_from_schedule
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, removed_ids)
    destroyed = update_solution_state_after_destroy(destroyed)
    return destroyed, removed_ids


def infeasible_constraint_removal(current, random_state, **kwargs):
    """
    Destroy operator chuyên biệt: Triệt tiêu các vi phạm Hard Constraints.
    1. Multi-trip Overlap: Xóa chuyến bị chồng lấn thời gian.
    2. Capacity Overload: Xóa các đơn hàng lớn trong chuyến quá tải.
    
    Returns (destroyed_copy, removed_list).
    """
    destroyed = copy.deepcopy(current)
    prob = destroyed.problem_instance
    
    # Dictionary lưu ứng viên cần xóa: {farm_id: score}
    # Score càng cao càng dễ bị xóa.
    # - Overlap: Score = 1,000,000 (Ưu tiên xóa tuyệt đối)
    # - Overload: Score = Demand (Xóa đơn to để giảm tải nhanh)
    candidates = {} 

    # --------------------------------------------------------------------------
    # BƯỚC 1: TỔ CHỨC DỮ LIỆU THEO XE ĐỂ CHECK OVERLAP
    # --------------------------------------------------------------------------
    # schedule item: (depot_idx, truck_id, custs, shift, start_time, finish_time, route_load)
    truck_trips = {}
    
    # Duyệt qua schedule để gom nhóm và check Capacity luôn
    for idx, row in enumerate(destroyed.schedule):
        depot_idx, truck_id, custs, shift, start_time, finish_time, route_load = row
        
        if not custs or shift == 'INTER-FACTORY':
            continue

        # Lấy thông tin xe để biết capacity
        truck = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
        if not truck: continue
        
        truck_capacity = truck['capacity']

        # --- CHECK 1: CAPACITY OVERLOAD ---
        if route_load > truck_capacity:
            # Route này bị quá tải -> Thêm tất cả khách vào danh sách ứng viên
            for fid in custs:
                _, _, f_dem = _get_farm_info(fid, prob)
                # Nếu farm này chưa có trong list hoặc điểm hiện tại thấp hơn Demand -> update
                # Điểm = Demand (Ưu tiên xóa đơn to)
                if fid not in candidates or candidates[fid] < f_dem:
                    candidates[fid] = f_dem

        # Gom nhóm để check Overlap (lưu index của trip trong schedule)
        if truck_id not in truck_trips:
            truck_trips[truck_id] = []
        truck_trips[truck_id].append({
            'start': start_time,
            'end': finish_time,
            'custs': custs,
            'idx': idx
        })

    # --------------------------------------------------------------------------
    # BƯỚC 2: CHECK MULTI-TRIP OVERLAP
    # --------------------------------------------------------------------------
    for t_id, trips in truck_trips.items():
        if len(trips) < 2:
            continue
        
        # Sắp xếp các chuyến của xe này theo thời gian bắt đầu
        trips.sort(key=lambda x: x['start'])
        
        # Kiểm tra chồng lấn
        for i in range(len(trips) - 1):
            trip_a = trips[i]
            trip_b = trips[i+1]
            
            # Logic Overlap: Chuyến sau bắt đầu trước khi chuyến trước kết thúc
            # (Thêm một lượng nhỏ epsilon nếu cần, ở đây so sánh trực tiếp)
            if trip_b['start'] < trip_a['end']:
                # PHÁT HIỆN OVERLAP!
                # Chiến thuật: Xóa toàn bộ khách của chuyến sau (Trip B)
                for fid in trip_b['custs']:
                    # Điểm = 1 Triệu (Cao hơn mọi demand -> Chắc chắn bị xóa trước)
                    candidates[fid] = 1_000_000

    # --------------------------------------------------------------------------
    # BƯỚC 3: THỰC HIỆN XÓA (BIAS RANDOM)
    # --------------------------------------------------------------------------
    if not candidates:
        return destroyed, []

    # Chuyển candidates thành list để sort: [(score, farm_id), ...]
    candidate_list = [(score, fid) for fid, score in candidates.items()]
    
    # Sort giảm dần theo score (Overlap lên đầu, sau đó đến Big Demand)
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    # Xác định số lượng cần xóa
    # Mặc định xóa khoảng 20% số khách, hoặc ít nhất là xóa hết các ca Overlap (score >= 1M)
    num_overlap = sum(1 for x in candidate_list if x[0] >= 1_000_000)
    
    default_frac = kwargs.get('remove_fraction', 0.2)
    # Tổng số khách trong schedule
    total_custs = sum(len(row[2]) for row in destroyed.schedule if row[2] and row[3] != 'INTER-FACTORY')
    target_remove = int(total_custs * default_frac)
    
    # Chúng ta phải xóa ÍT NHẤT là hết các ca Overlap
    num_to_remove = max(num_overlap, target_remove)
    # Không xóa quá số lượng ứng viên tìm được
    num_to_remove = min(num_to_remove, len(candidate_list))
    
    to_remove = []
    
    # Logic chọn ngẫu nhiên có trọng số (giống hàm mẫu của bạn)
    while len(to_remove) < num_to_remove and candidate_list:
        # r random từ 0-1. Mũ 6 để cực kỳ ưu tiên phần tử đầu danh sách
        r = random_state.random()
        idx = int(len(candidate_list) * (r ** 6))
        idx = min(idx, len(candidate_list) - 1)
        
        item = candidate_list.pop(idx)
        to_remove.append(item[1])

    # Thực hiện xóa trong solution
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, to_remove)
    destroyed = update_solution_state_after_destroy(destroyed)
    
    return destroyed, to_remove