import numpy as np
import re
from .utils import _remove_customers_from_schedule, get_route_cost, _get_farm_info,_calculate_route_schedule_and_feasibility
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
        ## FINAL VERSION for SINGLE-DAY ##
        Tính toán objective function một cách nhất quán với các toán tử repair.
        Sử dụng hàm tính toán nâng cao để có chi phí chính xác.
        
        <<< THAY ĐỔI: Chuyển từ hard constraint sang soft constraint (penalty) >>>
        """
        WAIT_COST_PER_MIN = 0.2
        TIME_PENALTY = 0.3
        total_variable_cost = 0.0
        total_fixed_cost = 0.0
        total_waiting_cost = 0.0
        total_penalty_cost = 0.0  # <<< THÊM MỚI: Biến lưu chi phí phạt >>>
        
        unique_trucks_used = set()
        dist_depot_depot = self.problem_instance.get('distance_matrix_depots', None)

        if not self.schedule:
            return 0.0, 0.0

        for route_info in self.schedule:
            # support both 4-tuple (legacy) and 5-tuple (with start_time)
            depot_idx, truck_id, customer_list, shift, start_time_at_depot = route_info
            
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
            # Giả định: hàm _calculate... *luôn* trả về (total_dist, total_wait)
            finish_time, is_feasible, total_dist, total_wait, start_time, route_penalty = _calculate_route_schedule_and_feasibility(
                depot_idx, customer_list, shift, start_time_at_depot, self.problem_instance, truck_details)
            print('route:', route_info)
            print('late: ',route_penalty)

            # <<< THAY ĐỔI: BỎ TRẢ VỀ 'inf', THAY BẰNG CỘNG DỒN PENALTY >>>
            # if not is_feasible:
            #     # Trả về chi phí vô cùng lớn để loại bỏ lời giải này
            #     return float('inf'), float('inf')
            
            # Thay vào đó, cộng dồn chi phí phạt (nếu có)
            total_penalty_cost += route_penalty*TIME_PENALTY 

            # Luôn cộng chi phí quãng đường và chi phí chờ,
            # ngay cả khi tuyến đường không khả thi (để so sánh)
            total_variable_cost += total_dist * var_cost_per_km
            total_waiting_cost += total_wait * WAIT_COST_PER_MIN

        # --- Chi phí thuê xe ---
        for truck_id in unique_trucks_used:
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if truck_details:
                lease_cost_per_day = truck_details.get('lease_cost_monthly', 0) / 30
                total_fixed_cost += lease_cost_per_day # Chỉ tính cho 1 ngày
        
        # <<< THAY ĐỔI: Thêm chi phí phạt vào tổng chi phí >>>
        total_cost = total_variable_cost + total_fixed_cost + total_waiting_cost + total_penalty_cost
        
        # Trả về tổng chi phí (để ALNS tối ưu) và chi phí phạt (để theo dõi)
        return total_cost, total_penalty_cost, total_waiting_cost