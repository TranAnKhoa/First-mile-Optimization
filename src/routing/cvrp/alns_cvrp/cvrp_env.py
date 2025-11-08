import numpy as np
import re

# ==============================================================================
# HÀM TIỆN ÍCH (Giữ nguyên)
# ==============================================================================
def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

def _clean_base_id(fid):
    """Làm sạch ID để lấy ID gốc của nông trại vật lý."""
    if not isinstance(fid, str):
        return fid
    return re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', fid)[0]

# QUAN TRỌNG: Import hoặc định nghĩa hàm tính toán cốt lõi ở đây.
# Tốt nhất là đưa hàm này vào một file utils.py chung và import từ đó.
# Dưới đây là định nghĩa lại để đảm bảo file này có thể tự chạy.
def _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """
    Hàm tính toán nâng cao với tối ưu hóa thời gian xuất phát.
    (Đây là phiên bản đầy đủ trả về 5 giá trị)
    """
    if not customer_list:
        return start_time_at_depot, True, 0, 0, start_time_at_depot

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

    timeline_sim = []
    current_time_sim1 = start_time_at_depot

    idx, demand, params, tw = _resolve_farm(customer_list[0])
    travel_time = depot_farm_dist[depot_idx, idx] / velocity
    arrival = current_time_sim1 + travel_time
    start_tw, end_tw = tw[shift]
    if arrival > end_tw: return -1, False, -1, -1, -1
    service_start = max(arrival, start_tw)
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    current_time_sim1 = service_start + service_duration
    timeline_sim.append({'arrival': arrival, 'start': service_start})

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
    
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity
    finish_time_sim1 = current_time_sim1 + travel_time_back
    if finish_time_sim1 > depot_end_time: return -1, False, -1, -1, -1

    slacks = [t['start'] - t['arrival'] for t in timeline_sim]
    min_slack = min(slacks) if slacks else 0
    optimal_start_time = start_time_at_depot + min_slack

    total_dist = 0; total_wait = 0
    current_time_final = optimal_start_time

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
# ĐỊNH NGHĨA CLASS cvrpEnv (Đã đơn giản hóa cho Single-Day)
# ==============================================================================

class cvrpEnv:
    def __init__(self, initial_schedule, problem_instance, seed, **kwargs):
        self.problem_instance = problem_instance
        self.schedule = initial_schedule # Bây giờ là một danh sách các tuyến
        self.seed = seed
        self.dist_matrix_data = problem_instance['distance_matrix_farms']
        self.dist_depot_data = problem_instance['distance_depots_farms']
        self.farm_id_to_idx = problem_instance['farm_id_to_idx_map']
        self.num_facilities = len(problem_instance['facilities'])
        self.num_farms = len(problem_instance['farms'])
        self.nb_customers = self.num_farms
        self.demands_data = [farm['demand'] for farm in problem_instance['farms']]
        self.customer_tw = [[farm['time_windows']['AM'][0], farm['time_windows']['PM'][1]] for farm in problem_instance['farms']]
        self.customer_st = [farm['service_time_params'][0] for farm in problem_instance['farms']]
        self.depot_tw = [[0, 24*60] for _ in problem_instance['facilities']]
        self.truck_capacity = problem_instance['fleet']['available_trucks'][0]['capacity'] if problem_instance['fleet']['available_trucks'] else 0

    def _get_farm_idx(self, farm_id):
        """Hàm tra cứu ID nông trại một cách "bền bỉ"."""
        try:
            return self.farm_id_to_idx[farm_id]
        except KeyError:
            try:
                return self.farm_id_to_idx[int(farm_id)]
            except (KeyError, ValueError):
                raise KeyError(f"Không thể tìm thấy Farm ID '{farm_id}' trong farm_id_to_idx map.")

    def objective(self):
        """
        ## FINAL VERSION for SINGLE-DAY ##
        Tính toán objective function một cách nhất quán với các toán tử repair.
        Sử dụng hàm tính toán nâng cao để có chi phí chính xác.
        """
        WAIT_COST_PER_MIN = 0.2
        
        total_variable_cost = 0.0
        total_fixed_cost = 0.0
        total_waiting_cost = 0.0
        
        unique_trucks_used = set()
        dist_depot_depot = self.problem_instance.get('distance_matrix_depots', None)

        if not self.schedule:
            return 0.0, 0.0

        for route_info in self.schedule:
            # support both 4-tuple (legacy) and 5-tuple (with start_time)
            if len(route_info) == 5:
                depot_idx, truck_id, customer_list, shift, start_time_at_depot = route_info
            else:
                depot_idx, truck_id, customer_list, shift = route_info
                start_time_at_depot = 0
            
            unique_trucks_used.add(truck_id)
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if not truck_details: continue

            var_cost_per_km = self.problem_instance['costs']['variable_cost_per_km'].get(
                (truck_details['type'], truck_details['region']), 1.0
            )

            if shift == 'INTER-FACTORY':
                if dist_depot_depot is not None and customer_list:
                    try:
                        parts = customer_list[0].split('_')
                        from_depot = int(parts[2])
                        to_depot = int(parts[4])
                        travel_dist = dist_depot_depot[from_depot, to_depot] * 2
                        total_variable_cost += travel_dist * var_cost_per_km
                    except (IndexError, ValueError): pass
                continue
            
            if not customer_list:
                continue

            # <<< THAY ĐỔI CỐT LÕI: SỬ DỤNG HÀM TÍNH TOÁN NÂNG CẤP >>>
            finish_time, is_feasible, total_dist, total_wait, _ = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, start_time_at_depot, self.problem_instance, truck_details)

            # Lời giải được đưa vào objective phải luôn khả thi
            if not is_feasible:
                # Trả về chi phí vô cùng lớn để loại bỏ lời giải này
                return float('inf'), float('inf')

            total_variable_cost += total_dist * var_cost_per_km
            total_waiting_cost += total_wait * WAIT_COST_PER_MIN

        # --- Chi phí thuê xe ---
        for truck_id in unique_trucks_used:
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if truck_details:
                lease_cost_per_day = truck_details.get('lease_cost_monthly', 0) / 30
                total_fixed_cost += lease_cost_per_day # Chỉ tính cho 1 ngày

        total_cost = total_variable_cost + total_fixed_cost + total_waiting_cost
        
        # Vì lời giải cuối cùng phải khả thi, không có penalty
        return total_cost, 0