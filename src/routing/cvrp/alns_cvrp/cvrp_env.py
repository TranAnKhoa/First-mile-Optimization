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
        (Phiên bản TỐI ƯU - Tin tưởng 6-Tuple)
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
        # ‼️ THAY ĐỔI 1: Bộ theo dõi THỜI GIAN "phân thân" ‼️
        # ==========================================================
        # (Giữ nguyên logic)
        truck_time_windows_used = defaultdict(list)
        truck_cloning_penalty = 0.0
        CLONE_PENALTY_WEIGHT = 999999
        # ==========================================================
        if not self.schedule:
            return 0.0, 0.0, 0.0, 0.0

        for route_info in self.schedule:
            # ‼️ UNPACK 7-TUPLE (Theo thiết kế mới của bạn) ‼️
            try:
                depot_idx, truck_id, customer_list, shift, start_time_at_depot, finish_time_route, route_load = route_info
            except ValueError:
                # Báo lỗi nếu một hàm nào đó (ví dụ 'destroy') trả về 5-tuple
                raise ValueError(f"Lỗi Unpack! Cấu trúc route_info không phải là 7-tuple: {route_info}")
            unique_trucks_used.add(truck_id)
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if not truck_details: continue

            var_cost_per_km = self.problem_instance['costs']['variable_cost_per_km'].get(
                (truck_details['type'], truck_details['region']), 1.0
            )

            # ==========================================================
            # ‼️ THAY ĐỔI 2: Thu thập Time Windows (Từ 6-Tuple) ‼️
            # ==========================================================
            # Chúng ta tin tưởng 100% vào start/finish time đã lưu
            # Đây là cửa sổ "Depot-to-Depot" (thời gian xe bận)
            key = (truck_id, shift)
            truck_time_windows_used[key].append((start_time_at_depot, finish_time_route))
            # ==========================================================

            if shift == 'INTER-FACTORY':
                # (Logic INTER-FACTORY giữ nguyên)
                if dist_depot_depot is not None and customer_list:
                    try:
                        parts = customer_list[0].split('_')
                        from_depot = int(parts[2])
                        to_depot = int(parts[4])
                        # Lấy dist thực tế từ finish_time và start_time
                        # (Giả định chi phí đã được tính đúng ở repair)
                        velocity = 1.0 if truck_details['type'] in ["Single", "Truck and Dog"] else 0.5
                        travel_dist = (finish_time_route - start_time_at_depot) * velocity
                        total_variable_cost += travel_dist * var_cost_per_km
                    except Exception: pass
                continue
            
            if not customer_list:
                continue
            
            # ‼️ LƯU Ý VỀ HIỆU SUẤT ‼️
            # Chúng ta VẪN PHẢI gọi hàm này, không phải để lấy 'finish_time',
            # mà là để lấy các THÀNH PHẦN CHI PHÍ (dist, wait, penalties).
            (_is_feasible, total_dist, total_wait, 
            time_penalty, capacity_penalty) = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, start_time_at_depot, finish_time_route, route_load, self.problem_instance, truck_details)
            
            # (Bạn có thể thêm một 'assert' ở đây để kiểm tra
            # 'abs(_calc_finish - finish_time_route) < 1e-6' nếu muốn)

            # Cộng dồn chi phí (Giữ nguyên)
            total_capacity_penalty += capacity_penalty
            total_penalty_cost += time_penalty*TIME_PENALTY 
            total_variable_cost += total_dist * var_cost_per_km
            total_waiting_cost += total_wait * WAIT_COST_PER_MIN

        # --- Chi phí thuê xe --- (Giữ nguyên)
        for truck_id in unique_trucks_used:
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if truck_details:
                lease_cost_per_day = truck_details.get('lease_cost_monthly', 0) / 30
                total_fixed_cost += lease_cost_per_day

        # ==========================================================
        # ‼️ THAY ĐỔI 3: KIỂM TRA CHỒNG LẤP (OVERLAP) ‼️
        # ==========================================================
        # (Logic này giữ nguyên, nhưng bây giờ nó đang so sánh 
        # 'start_time_at_depot' và 'finish_time_route' rất rõ ràng)
        for key, time_list in truck_time_windows_used.items():
            if len(time_list) <= 1:
                continue # Xe này chỉ chạy 1 chuyến (ca này), không thể overlap

            # Sắp xếp các chuyến đi theo thời gian bắt đầu (start_time_at_depot)
            sorted_times = sorted(time_list, key=lambda x: x[0])
            
            # Kiểm tra chồng lấp
            for i in range(len(sorted_times) - 1):
                current_trip_finish = sorted_times[i][1] # finish_time_route của chuyến 1
                next_trip_start = sorted_times[i+1][0]   # start_time_at_depot của chuyến 2
                
                # (Sử dụng 1e-6 để tránh lỗi float)
                if next_trip_start < (current_trip_finish - 1e-6): 
                    # LỖI! Chuyến 2 bắt đầu TRƯỚC khi chuyến 1 kết thúc
                    truck_cloning_penalty += CLONE_PENALTY_WEIGHT
                    break # Chỉ phạt một lần cho mỗi xe
        
        # ==========================================================
        # ‼️ THAY ĐỔI 4: Cộng dồn chi phí (Giữ nguyên) ‼️
        # ==========================================================
        total_cost = (total_variable_cost + total_fixed_cost + 
                      total_waiting_cost + total_penalty_cost + 
                      (total_capacity_penalty * CAP_PENALTY_WEIGHT) +
                      truck_cloning_penalty)
        #print("Penalty from cloning trucks:", truck_cloning_penalty)
        return total_cost, total_penalty_cost, total_waiting_cost, total_capacity_penalty
    