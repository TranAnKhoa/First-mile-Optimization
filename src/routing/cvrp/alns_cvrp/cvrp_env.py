import numpy as np
import re
from .utils import _remove_customers_from_schedule, get_route_cost, _get_farm_info,_calculate_route_schedule_and_feasibility
from collections import defaultdict
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
        (Phiên bản TỐI ƯU - Tin tưởng 6-Tuple - Đã làm sạch Debug)
        Kiểm tra chồng lấn thời gian dựa trên 'start_time_at_depot' 
        và 'finish_time_route' đã được lưu trữ.
        """
        WAIT_COST_PER_MIN = 0.2
        TIME_PENALTY = 0.3
        total_variable_cost = 0.0
        total_fixed_cost = 0.0
        total_waiting_cost = 0.0
        total_penalty_cost = 0.0
        total_capacity_penalty = 0.0
        CAP_PENALTY_WEIGHT = 10000000
        unique_trucks_used = set()
        dist_depot_depot = self.problem_instance.get('distance_matrix_depots', None)
        
        # ==========================================================
        # 1. Khởi tạo bộ theo dõi
        # ==========================================================
        truck_time_windows_used = defaultdict(list)
        truck_cloning_penalty = 0.0
        CLONE_PENALTY_WEIGHT = 999999
        
        if not self.schedule:
            return 0.0, 0.0, 0.0, 0.0

        # ==========================================================
        # 2. Duyệt từng chuyến để tính Variable Cost & Thu thập Time
        # ==========================================================
        for route_info in self.schedule:
            # UNPACK 7-TUPLE
            try:
                depot_idx, truck_id, customer_list, shift, start_time_at_depot, finish_time_route, route_load = route_info
            except ValueError:
                raise ValueError(f"Lỗi Unpack! Cấu trúc route_info không phải là 7-tuple: {route_info}")
            
            unique_trucks_used.add(truck_id)
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if not truck_details: continue

            var_cost_per_km = self.problem_instance['costs']['variable_cost_per_km'].get(
                (truck_details['type'], truck_details['region']), 1.0
            )

            # Thu thập Time Windows
            key = (truck_id, shift)
            truck_time_windows_used[key].append((start_time_at_depot, finish_time_route))

            # Xử lý chuyến điều chuyển (INTER-FACTORY)
            if shift == 'INTER-FACTORY':
                if dist_depot_depot is not None and customer_list:
                    try:
                        velocity = 1.0 if truck_details['type'] in ["Single", "Truck and Dog"] else 0.5
                        travel_dist = (finish_time_route - start_time_at_depot) * velocity
                        total_variable_cost += travel_dist * var_cost_per_km
                    except Exception: pass
                continue
            
            if not customer_list:
                continue
            
            # Tính toán chi phí biến đổi cho chuyến giao hàng thường
            (_is_feasible, total_dist, total_wait, 
            time_penalty, capacity_penalty) = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, start_time_at_depot, finish_time_route, route_load, self.problem_instance, truck_details)
            
            total_capacity_penalty += capacity_penalty
            total_penalty_cost += time_penalty * TIME_PENALTY 
            total_variable_cost += total_dist * var_cost_per_km
            total_waiting_cost += total_wait * WAIT_COST_PER_MIN

        # ==========================================================
        # 3. TÍNH FIXED COST (Clean Version)
        # ==========================================================
        VIRTUAL_SUFFIXES = {222, 333, 444} 
        
        for raw_id in unique_trucks_used:
            try:
                truck_id = int(raw_id)
                suffix = truck_id % 1000 
            except ValueError:
                suffix = 0 # ID lạ -> coi như xe gốc

            if suffix in VIRTUAL_SUFFIXES:
                # --- XE ẢO (VIRTUAL) ---
                # Phải trả phí thuê vì ta đang thuê thêm ngoài
                truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
                if truck_details:
                    lease_cost = truck_details.get('lease_cost_monthly', 0.0)
                    total_fixed_cost += lease_cost / 30
            else:
                # --- XE GỐC (OWNED) ---
                # Sunk cost = 0. Khuyến khích tận dụng tối đa xe nhà.
                pass

        # ==========================================================
        # 4. KIỂM TRA CHỒNG LẤP (OVERLAP)
        # ==========================================================
        for key, time_list in truck_time_windows_used.items():
            if len(time_list) <= 1:
                continue 

            sorted_times = sorted(time_list, key=lambda x: x[0])
            
            for i in range(len(sorted_times) - 1):
                current_trip_finish = sorted_times[i][1]
                next_trip_start = sorted_times[i+1][0]
                
                # Check overlap với dung sai nhỏ
                if next_trip_start < (current_trip_finish - 1e-6): 
                    truck_cloning_penalty += CLONE_PENALTY_WEIGHT
                    break 
        
        # ==========================================================
        # 5. Tổng hợp chi phí
        # ==========================================================
        total_cost = (total_variable_cost + total_fixed_cost + 
                      total_waiting_cost + total_penalty_cost + 
                      (total_capacity_penalty * CAP_PENALTY_WEIGHT) +
                      truck_cloning_penalty)
        
        return total_cost, total_penalty_cost, total_waiting_cost, total_capacity_penalty
    