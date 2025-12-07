import copy
# Giả định các hàm này có thể được import từ repair_operators
# Hoặc bạn có thể tạo một file utils.py chung
from .utils import get_route_cost, _get_dist_and_time, _get_farm_info, _get_service_time, _get_shift_end_time, find_truck_by_id, calculate_route_finish_time

import copy

# Assumes get_route_cost(problem_instance, route_info) exists and accepts 4- or 5-tuple route_info.
def apply_2_opt(solution):
    """
    Phiên bản 2-Opt AN TOÀN (Feasibility-Safe).
    Đã cập nhật cho 7-Tuple và Smart Repair.
    """
    improved_solution = copy.deepcopy(solution)
    problem_instance = improved_solution.problem_instance
    available_trucks = problem_instance['fleet']['available_trucks']
    
    # Lặp qua TỪNG tuyến đường
    for route_idx, route_data in enumerate(list(improved_solution.schedule)):
        
        # ‼️ [SỬA] UNPACK 7-TUPLE ‼️
        depot_idx, truck_id, original_customer_list, shift, start_time, finish_time, route_load = route_data
        
        # Bỏ qua nếu không đủ điều kiện
        if len(original_customer_list) < 2 or shift == 'INTER-FACTORY':
            continue
            
        # Bắt đầu quá trình lặp
        improved_in_route = True
        best_list_for_route = original_customer_list
        
        # Tính chi phí hiện tại (dùng tuple 7 phần tử)
        best_cost_for_route = get_route_cost(problem_instance, route_data)
        
        while improved_in_route:
            improved_in_route = False
            current_list = list(best_list_for_route)
            
            # Vòng lặp 2-Opt
            # i đi từ 1 đến N-1 (vì cần ít nhất 1 điểm trước nó)
            for i in range(1, len(current_list)):
                # j đi từ i+2 đến N (cần cách i ít nhất 1 điểm để đảo ngược có ý nghĩa)
                for j in range(i + 1, len(current_list) + 1):
                    
                    # 2-Opt Swap: Đảo ngược đoạn từ i đến j
                    # [0...i-1] + [j-1...i] + [j...end]
                    new_list = current_list[:i] + current_list[i:j][::-1] + current_list[j:]
                    
                    # Nếu list không thay đổi (hiếm gặp nhưng có thể), bỏ qua
                    if new_list == current_list: continue

                    # ‼️ [SỬA] Tạo tuple tạm để tính chi phí ‼️
                    # Truyền dummy finish/load, hy vọng get_route_cost sẽ tính lại
                    new_route_info_temp = (
                        depot_idx, truck_id, new_list, shift, start_time, 
                        finish_time, route_load # (Giá trị cũ, dùng tạm)
                    )
                    
                    # Tính toán lại TOÀN BỘ chi phí
                    new_cost = get_route_cost(problem_instance, new_route_info_temp)
                    
                    # First Improvement
                    if new_cost < best_cost_for_route - 1e-6:
                        best_cost_for_route = new_cost
                        best_list_for_route = new_list
                        improved_in_route = True
                        break 
                
                if improved_in_route:
                    break
        
        # --- CẬP NHẬT TUYẾN ĐƯỜNG (NẾU CÓ THAY ĐỔI) ---
        if best_list_for_route != original_customer_list:
            truck_info = find_truck_by_id(truck_id, available_trucks)
            
            # 1. Tính Finish Time Mới (Bắt buộc)
            new_finish_time = calculate_route_finish_time(
                depot_idx, best_list_for_route, shift, start_time, problem_instance, truck_info
            )
            
            # 2. Tính Load Mới (Dù 2-Opt không đổi tổng load, nhưng cứ tính cho an toàn/nhất quán)
            # (Thực tế 2-Opt chỉ đảo thứ tự, tổng hàng không đổi -> có thể dùng lại route_load cũ)
            # Nhưng ta cứ tính lại để đảm bảo tính toàn vẹn dữ liệu
            new_route_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in best_list_for_route)
            
            # 3. ‼️ [SỬA] Lưu 7-TUPLE Mới ‼️
            improved_solution.schedule[route_idx] = (
                depot_idx, 
                truck_id, 
                best_list_for_route, 
                shift, 
                start_time, 
                new_finish_time, # Cập nhật
                new_route_load   # Cập nhật (hoặc giữ nguyên route_load cũ cũng được)
            )

    return improved_solution

def apply_relocate(solution, max_iterations_per_route=5):
    """ 
    GIẢI PHÁP INTRA_ROUTE NỘI TUYẾN (Đã cập nhật cho 7-Tuple)
    """
    improved_solution = copy.deepcopy(solution)
    problem_instance = improved_solution.problem_instance
    available_trucks = problem_instance['fleet']['available_trucks']
    
    # Lặp qua TỪNG tuyến đường
    # Lưu ý: Dùng enumerate trên list copy để an toàn
    for route_idx, route_data in enumerate(list(improved_solution.schedule)):
        
        # ‼️ [SỬA] UNPACK 7-TUPLE ‼️
        # (load và finish cũ sẽ bị thay thế nếu có cải thiện)
        depot_idx, truck_id, original_customer_list, shift, start_time, finish_time, route_load = route_data

        # Bỏ qua tuyến không thể tối ưu
        if len(original_customer_list) < 2 or shift == 'INTER-FACTORY':
            continue 

        # --- Bắt đầu tối ưu ---
        best_list_for_route = original_customer_list
        
        # Tính chi phí hiện tại (dùng tuple 7 phần tử, hàm get_route_cost phải hỗ trợ hoặc slice)
        # Lưu ý: Để gọi get_route_cost, ta tạo tuple tạm thời đúng format nó cần (thường là 5 hoặc 7)
        # Ở đây ta dùng lại 'route_data' vì nó đã đúng format
        best_cost_for_route = get_route_cost(problem_instance, route_data)
        
        iteration_count = 0
        
        while iteration_count < max_iterations_per_route:
            iteration_count += 1
            was_improved_in_this_pass = False 
            current_list = list(best_list_for_route)
            
            for i in range(len(current_list)):
                cust_to_move = current_list[i]
                temp_list = current_list[:i] + current_list[i+1:]
                
                for j in range(len(temp_list) + 1):
                    if i == j: continue
                    
                    new_list = temp_list[:j] + [cust_to_move] + temp_list[j:]
                    
                    # ‼️ [SỬA] Tạo tuple tạm để tính chi phí ‼️
                    # Lưu ý: finish_time và load trong tuple tạm này là SAI (của tuyến cũ).
                    # Nhưng hàm 'get_route_cost' thường sẽ TÍNH LẠI dựa trên danh sách khách hàng,
                    # nên việc truyền giá trị cũ vào đây thường không sao (tùy thuộc vào implementation của get_route_cost).
                    # Nếu get_route_cost của bạn tin tưởng vào finish/load truyền vào -> BẠN PHẢI TÍNH LẠI NGAY TẠI ĐÂY.
                    # Giả định: get_route_cost sẽ chạy mô phỏng lại từ đầu (như logic cũ).
                    new_route_info_temp = (
                        depot_idx, truck_id, new_list, shift, start_time, finish_time, route_load
                    )
                    
                    new_cost = get_route_cost(problem_instance, new_route_info_temp)

                    # First Improvement
                    if new_cost < best_cost_for_route - 1e-6:
                        best_cost_for_route = new_cost
                        best_list_for_route = new_list
                        was_improved_in_this_pass = True
                        break 
                
                if was_improved_in_this_pass:
                    break
            
            if not was_improved_in_this_pass:
                break
        
        # --- CẬP NHẬT TUYẾN ĐƯỜNG (NẾU CÓ THAY ĐỔI) ---
        # Nếu danh sách khách hàng thay đổi, ta PHẢI tính lại Finish Time và Load
        # để lưu vào 7-tuple cho đúng.
        
        if best_list_for_route != original_customer_list:
            truck_info = find_truck_by_id(truck_id, available_trucks)
            
            # 1. Tính Finish Time Mới
            new_finish_time = calculate_route_finish_time(
                depot_idx, best_list_for_route, shift, start_time, problem_instance, truck_info
            )
            
            # 2. Tính Load Mới
            new_route_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in best_list_for_route)
            
            # 3. ‼️ [SỬA] Lưu 7-TUPLE Mới ‼️
            improved_solution.schedule[route_idx] = (
                depot_idx, 
                truck_id, 
                best_list_for_route, 
                shift, 
                start_time, 
                new_finish_time, # Cập nhật
                new_route_load   # Cập nhật
            )

    return improved_solution
