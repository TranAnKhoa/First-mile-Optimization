import copy
import random
import numpy as np
import re
from collections import defaultdict
from .utils import _remove_customers_from_schedule, get_route_cost, _get_farm_info,_calculate_route_schedule_and_feasibility, find_truck_by_id
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

# Trong file destroy_operators.py

# ... (các hàm khác không thay đổi) ...

def random_removal(current, random_state, **kwargs):
    """
    Xóa ngẫu nhiên các farm_id khỏi lịch trình.
    *** PHIÊN BẢN ĐÃ SỬA LỖI ValueError (logic 5 phần tử) ***
    """
    destroyed = copy.deepcopy(current)
    
    # <<< SỬA LỖI Ở ĐÂY: GIẢI NÉN 5 PHẦN TỬ >>>
    # Lấy danh sách tất cả các visit có thể xóa
    all_visits = [
        fid for _, _, cust_list, shift, _ in destroyed.schedule 
        for fid in cust_list if shift != 'INTER-FACTORY'
    ]

    if not all_visits:
        return destroyed, []
    
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_visits) * 0.15)))
    num_to_remove = min(num_to_remove, len(all_visits))

    customers_to_remove = random.sample(all_visits, num_to_remove)
    
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, customers_to_remove)
    
    return destroyed, customers_to_remove


def worst_removal(current, random_state, **kwargs):
    """
    IMPROVED worst removal:
    - Tính chi tiết bằng _calculate_route_schedule_and_feasibility (dist + wait)
    - Loại các farm có 'cost_saving' lớn nhất khi xóa chúng.
    - Thêm yếu tố ngẫu nhiên khi chọn (không luôn pick top-1) bằng power bias.
    """
    destroyed = copy.deepcopy(current)
    problem_instance = destroyed.problem_instance

    removed_customers = []
    
    num_visits = sum(len(r[2]) for r in current.schedule if r[3] != 'INTER-FACTORY')
    if num_visits == 0:
        return destroyed, []
    default_frac = kwargs.get('remove_fraction', 0.20)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(num_visits * default_frac)))
    num_to_remove = min(num_to_remove, num_visits)

    power = kwargs.get('selection_power', 4)  # higher -> more bias to top items

    for _ in range(num_to_remove):
        savings_list = []
        # For each candidate farm, compute saving = old_cost - new_cost when removing it
        # We compute old_dist, old_wait via _calculate_route_schedule_and_feasibility for the route
        for route_idx, route_info in enumerate(destroyed.schedule):
            depot_idx, truck_id, customer_list, shift, start_time = route_info
            if not customer_list or shift == 'INTER-FACTORY':
                continue

            truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
            if truck_info is None:
                continue

            # Get old route metrics
            _, feasible_old, old_dist, old_wait, _ = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, 0, problem_instance, truck_info
            )
            if not feasible_old:
                continue
            var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                (truck_info['type'], truck_info['region']), 1.0
            )
            old_cost = old_dist * var_cost_per_km + old_wait * WAIT_COST_PER_MIN

            for pos in range(len(customer_list)):
                farm_to_remove = customer_list[pos]
                temp_list = customer_list[:pos] + customer_list[pos+1:]
                # If the route becomes empty, cost_after = 0 (route removed)
                if not temp_list:
                    new_cost = 0.0
                else:
                    _, feasible_new, new_dist, new_wait, _ = _calculate_route_schedule_and_feasibility(
                        depot_idx, temp_list, shift, 0, problem_instance, truck_info
                    )
                    if not feasible_new:
                        # If removing this farm causes infeasibility (shouldn't normally), penalize by skipping
                        continue
                    new_cost = new_dist * var_cost_per_km + new_wait * WAIT_COST_PER_MIN

                saving = old_cost - new_cost
                # Keep also some context to break ties / debugging
                savings_list.append({
                    'saving': saving,
                    'farm_id': farm_to_remove,
                    'route_idx': route_idx,
                    'old_cost': old_cost,
                    'new_cost': new_cost
                })

        if not savings_list:
            break

        # Sort by saving desc
        savings_list.sort(key=lambda x: x['saving'], reverse=True)

        # Randomized selection biased towards top using power distribution
        r = random_state.random()
        idx = int(len(savings_list) * (r ** power))
        # Safety clamp
        idx = max(0, min(len(savings_list)-1, idx))

        chosen = savings_list[idx]['farm_id']
        # Prevent removing same farm twice (in case duplicates)
        if chosen in removed_customers:
            # find next available not removed
            for entry in savings_list:
                if entry['farm_id'] not in removed_customers:
                    chosen = entry['farm_id']
                    break
            else:
                break

        # Apply removal
        destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, [chosen])
        removed_customers.append(chosen)

    return destroyed, removed_customers


# Trong file destroy_operators.py

# ... (Các hàm khác không thay đổi) ...

def shaw_removal(current, random_state, **kwargs):
    """
    IMPROVED Shaw-style related removal.
    *** PHIÊN BẢN ĐÃ SỬA LỖI ValueError (logic 5 phần tử) ***
    """
    destroyed = copy.deepcopy(current)
    problem = destroyed.problem_instance
    
    # <<< SỬA LỖI Ở ĐÂY: GIẢI NÉN 5 PHẦN TỬ >>>
    # Bây giờ all_visits sẽ là danh sách các tuple: (farm_id, depot_idx, shift, start_time)
    all_visits = [
        (cust, depot_idx, shift, start_time)
        # Giải nén 5 phần tử, bỏ qua truck_id không cần thiết bằng dấu gạch dưới '_'
        for depot_idx, _, custs, shift, start_time in destroyed.schedule
        for cust in custs if shift != 'INTER-FACTORY'
    ]
 
    if not all_visits:
        return destroyed, []
        
    default_frac = kwargs.get('remove_fraction', 0.15)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_visits) * default_frac)))
    num_to_remove = min(num_to_remove, len(all_visits))

    # Chọn một visit ngẫu nhiên làm "hạt giống"
    seed_visit = all_visits[random_state.randint(len(all_visits))]

    # 'removed' và 'remaining' bây giờ sẽ chứa các tuple (farm_id, depot_idx, shift, start_time)
    removed = {seed_visit}
    remaining = set(all_visits) - removed

    dist_mat = problem['distance_matrix_farms']
    
    w_dist = kwargs.get('w_dist', 1.0)
    w_tw = kwargs.get('w_tw', 2.0)
    w_depot = kwargs.get('w_depot', 1.5)

    def tw_overlap(f1_det, f2_det):
        overlap = 0
        for shift in ['AM', 'PM']:
            a1, b1 = f1_det['time_windows'][shift]
            a2, b2 = f2_det['time_windows'][shift]
            overlap += max(0, min(b1, b2) - max(a1, a2))
        return overlap

    while len(removed) < num_to_remove and remaining:
        ref_visit = random.choice(list(removed))
        ref_farm_id = ref_visit[0]
        ref_idx, ref_det, _ = _get_farm_info(ref_farm_id, problem)
        
        scores = []
        for cand_visit in list(remaining):
            cand_farm_id = cand_visit[0]
            cand_idx, cand_det, _ = _get_farm_info(cand_farm_id, problem)
            
            d = dist_mat[ref_idx, cand_idx]
            ov = tw_overlap(ref_det, cand_det)
            
            # So sánh depot gốc của visit
            dep_ref = ref_visit[1]
            dep_cand = cand_visit[1]
            same_depot_bonus = 1.0 if dep_ref == dep_cand else 0.0

            score = w_dist * d - w_tw * ov - w_depot * same_depot_bonus
            scores.append((score, cand_visit))

        if not scores: break

        scores.sort(key=lambda x: x[0])
        r = random_state.random()
        idx_pick = int(len(scores) * (r ** 1.8))
        idx_pick = max(0, min(len(scores)-1, idx_pick))
        
        pick = scores[idx_pick][1]

        removed.add(pick)
        remaining.remove(pick)

    # Trích xuất chỉ farm_id để xóa và trả về
    customers_to_remove_ids = [visit[0] for visit in removed]
    
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, customers_to_remove_ids)

    return destroyed, customers_to_remove_ids

def time_worst_removal(current, random_state, **kwargs):
    """
    Remove the visits that have the largest individual waiting times.
    Returns (destroyed_copy, removed_list).
    """
    destroyed = copy.deepcopy(current)
    prob = destroyed.problem_instance
    visits_wait = []  # list of (wait_time, farm_id)

    # compute waiting per stop using current schedule and _calculate_route_schedule_and_feasibility
    for depot_idx, truck_id, custs, shift, start_time in destroyed.schedule:
        if not custs or shift == 'INTER-FACTORY':
            continue
        truck = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
        if not truck:
            continue
        # get timeline by simulating (we already have helper but reuse _calculate for totals)
        # To get per-stop wait need to reconstruct timeline like simulate does:
        # We'll reuse the same logic as in _calculate_route_schedule_and_feasibility but per-stop.
        # Simpler: call helper simulate-like code here:
        current_time = 0
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
    num_visits = len(visits_wait)
    num_to_remove = kwargs.get('num_to_remove', max(1, int(num_visits * default_frac)))
    num_to_remove = min(num_to_remove, num_visits)

    to_remove = [fid for _, fid in visits_wait[:num_to_remove]]
    destroyed.schedule = _remove_customers_from_schedule(destroyed.schedule, to_remove)
    return destroyed, to_remove




def truck_day_removal(current, random_state, **kwargs):
    # Logic này phức tạp hơn vì nó xóa các xe, không phải khách hàng.
    # Tạm thời, để đảm bảo tính nhất quán, hãy để nó gọi random_removal.
    # Khi bạn sẵn sàng, bạn có thể viết lại nó để trả về danh sách khách hàng gốc.
    return random_removal(current, random_state, **kwargs)

def shaw_distance_removal(current, random_state, **kwargs):
    # Cần viết lại hoàn toàn theo logic "đơn vị khách hàng"
    return random_removal(current, random_state, **kwargs)

# Các hàm còn lại trỏ đến các phiên bản đã hoạt động
relatedness_removal = shaw_distance_removal
cost_outlier_removal = worst_removal
path_removal = random_removal
sequence_removal = random_removal
history_biased_removal = random_removal
two_route_exchange = random_removal
depot_shift_removal = random_removal
load_balancing_removal = random_removal