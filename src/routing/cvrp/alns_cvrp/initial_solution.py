import numpy as np
from collections import defaultdict
import random
import re
import copy
import sys
from .utils import _clean_base_id
# ======================= HÀM TIỆN ÍCH =======================


def _calculate_route_schedule_and_feasibility_ini(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """Kiểm tra tính khả thi của route với time window, đã bao gồm velocity."""
    
    # Nếu danh sách khách rỗng -> kết thúc ngay
    if not customer_list:
        ### 1. <SỬA> Trả về 3 giá trị (thêm wait_time = 0) ###
        return start_time_at_depot, True, 0
    
    # Lấy các cấu trúc dữ liệu cần thiết từ problem_instance
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    depot_end_time = 1900 
    current_time = start_time_at_depot 
    truck_name = truck_info['type'] 
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
    virtual_map = problem_instance.get('virtual_split_farms', {})

    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid)
        try:
            base_idx = farm_id_to_idx[base_id_str]
        except KeyError:
            base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        if isinstance(fid, str) and fid in virtual_map:
            portion = virtual_map[fid].get('portion', 0)
            return base_idx, portion, base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

    # ============ xử lý khách đầu tiên (từ depot -> customer đầu) ============
    first_cust_id = customer_list[0]
    first_idx, first_demand, first_params, first_tw = _resolve_farm(first_cust_id)
    travel_time = depot_farm_dist[depot_idx, first_idx] / velocity
    arrival_time = current_time + travel_time 
    
    start_tw, end_tw = first_tw[shift]
    
    ### 2. <SỬA> Tính toán thời gian chờ của khách đầu tiên ###
    first_wait = max(0, start_tw - arrival_time)
    
    service_start = max(arrival_time, start_tw)
    
    # (Đây là logic bạn đã sửa đúng)
    if service_start > end_tw + 1e-6:
        return -1, False, 0 # Trả về 3 giá trị

    fix_time, var_param = first_params
    service_duration = fix_time + (first_demand / var_param if var_param > 0 else 0)
    current_time = service_start + service_duration 

    # ============ xử lý các khách tiếp theo (customer_list[1:] ) ============
    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i + 1])
        travel_time = dist_matrix[from_idx, to_idx] / velocity
        arrival_time = current_time + travel_time

        start_tw, end_tw = to_tw[shift]
        service_start = max(arrival_time, start_tw)
        
        # (Đây là logic bạn đã sửa đúng)
        if service_start > end_tw + 1e-6:
            return -1, False, 0 # Trả về 3 giá trị
        
        fix_time, var_param = to_params
        service_duration = fix_time + (to_demand / var_param if var_param > 0 else 0)
        current_time = service_start + service_duration

    # ============ sau khi phục vụ khách cuối, quay lại depot ============
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity
    finish_time_at_depot = current_time + travel_time_back
    
    if finish_time_at_depot > depot_end_time:
        return -1, False, 0 # Trả về 3 giá trị
    
    ### 3. <SỬA> Trả về 3 giá trị (thêm first_wait) ###
    return finish_time_at_depot, True, first_wait


#Hàm _calculate_route_schedule_and_feasibility_ini sẽ kiểm tra tất cả vị trí khả thi để chèn rồi output: return finish_time_at_depot, True, first_wait
# ==================== HÀM CHÍNH (SINGLE-DAY, NÂNG CẤP) ====================
def compute_initial_solution(problem_instance, random_state):
    print("\n--- BÊN TRONG COMPUTE_INITIAL_SOLUTION (AUTO-SPLIT ENABLED) ---")
    count = 0 
    
    # 1. KHỞI TẠO CÁC BIẾN CƠ BẢN
    farms = problem_instance['farms'] 
    facilities = problem_instance['facilities'] 
    available_trucks = problem_instance['fleet']['available_trucks'] 
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map'] 
    final_schedule = [] 
    
    depot_capacity = [f['capacity'] for f in facilities]
    
    # [MỚI] Tính Median Demand để dùng làm kích thước chuẩn khi chia nhỏ
    all_demands = [f["demand"] for f in farms]
    median_demand = np.median(all_demands) if all_demands else 10000
    
    depot_load = defaultdict(float) 
    depots_by_region = defaultdict(list) 
    rest = 10 # Thời gian nghỉ
    
    for i, facility in enumerate(facilities): 
        if 'region' in facility:
            depots_by_region[facility['region']].append(i)
    
    # Danh sách thăm viếng ban đầu
    all_required_visits = [farm['id'] for farm in farms]
    random_state.shuffle(all_required_visits)
    
    truck_finish_times = defaultdict(lambda: (0, -1))
    assigned_farms = set() 
    
    # [QUAN TRỌNG] Map lưu thông tin các farm ảo đã chia
    virtual_map = problem_instance.setdefault('virtual_split_farms', {})
    onfly_split_done = set() # Đánh dấu ID gốc nào đã bị chia rồi

    # Hàm helper nội bộ (giữ nguyên logic của bạn)
    def _resolve_farm_for_ci_local(fid):
        if isinstance(fid, str) and fid in virtual_map:
            base = virtual_map[fid]['base_id'] 
            portion = virtual_map[fid].get('portion', 0) 
            # Logic đệ quy nếu split nhiều tầng (an toàn)
            while base in virtual_map: 
                base = virtual_map[base]['base_id']
            base_clean = _clean_base_id(base) 
            idx = farm_id_to_idx_map.get(base_clean, farm_id_to_idx_map.get(int(base_clean)))
            base_info = farms[idx] 
            return base, portion, base_info, idx
        
        base_clean = _clean_base_id(fid)
        idx = farm_id_to_idx_map.get(base_clean, farm_id_to_idx_map.get(int(base_clean)))
        base_info = farms[idx]
        return fid, base_info['demand'], base_info, idx

    # ====================== MAIN LOOP (CHUYỂN SANG WHILE) ======================
    # [LÝ DO]: Dùng while để có thể append phần tử mới vào all_required_visits và duyệt tới nó
    idx_iter = 0
    while idx_iter < len(all_required_visits):
        i = all_required_visits[idx_iter]
        idx_iter += 1
        
        if i in assigned_farms:
            continue 
        
        # Lấy thông tin farm (có thể là farm ảo hoặc thật)
        effective_id, eff_demand, farm_details, farm_idx = _resolve_farm_for_ci_local(i) 
        
        closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx])) 
        depot_region = facilities[closest_depot_idx].get('region', None) 
        type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}

        # ==========================================================================
        # 🚀 [LOGIC MỚI]: ON-THE-FLY DEMAND SPLITTING
        # ==========================================================================
        # 1. Tìm max capacity của xe TRONG VÙNG này
        eligible_trucks_in_region = [t for t in available_trucks if t.get('region') == depot_region]
        if eligible_trucks_in_region:
            max_capacity_in_region = max(t['capacity'] for t in eligible_trucks_in_region)
        else:
            # Fallback nếu vùng này chưa có xe (hiếm), lấy max toàn bộ đội xe
            max_capacity_in_region = max(t['capacity'] for t in available_trucks) if available_trucks else float('inf')

        # 2. Kiểm tra điều kiện split
        # Chỉ split nếu demand > max_cap VÀ farm gốc chưa từng bị split
        clean_real_id = _clean_base_id(effective_id)
        
        if eff_demand > max_capacity_in_region and clean_real_id not in onfly_split_done:
            num_parts = int(np.ceil(eff_demand / median_demand))
            remaining = eff_demand
            
            print(f"⚠️ ON-THE-FLY SPLIT: Farm {i} (Demand {eff_demand}) > MaxCap {max_capacity_in_region} vùng {depot_region}. Chia thành {num_parts} phần.")
            
            for k in range(num_parts):
                # Lấy demand cho phần này (ưu tiên median, phần cuối lấy phần dư)
                part_qty = min(median_demand, remaining)
                if k == num_parts - 1: 
                     part_qty = remaining
                
                split_id = f"{clean_real_id}_onfly_part{k+1}"
                
                # Lưu vào virtual map
                virtual_map[split_id] = {'base_id': clean_real_id, 'portion': part_qty}
                
                # [QUAN TRỌNG]: Thêm vào cuối danh sách để vòng lặp while sẽ duyệt tới sau
                all_required_visits.append(split_id)
                
                remaining -= part_qty
            
            # Đánh dấu farm cha đã xử lý xong (được thay thế bởi các con)
            assigned_farms.add(i) 
            onfly_split_done.add(clean_real_id)
            
            # Bỏ qua vòng lặp hiện tại, đợi xử lý các phần con
            continue
        # ==========================================================================

        # [LOGIC CŨ CỦA BẠN]: PHÂN LOẠI XE
        suitable_trucks_IN_REGION = []
        suitable_trucks_OUT_OF_REGION = []

        for t in available_trucks:
            # Lọc 1: Accessibility
            t['type_idx'] = type_to_idx.get(t.get('type'), -1)
            if t['type_idx'] == -1: continue 

            depot_ok = facilities[closest_depot_idx].get('accessibility', [1]*4)[t['type_idx']] == 1
            farm_ok = farm_details.get('accessibility', [1]*4)[t['type_idx']] == 1
            
            # Lọc 2: Capacity (so với demand hiện tại - có thể là demand nhỏ đã split)
            capacity_ok = t['capacity'] >= eff_demand
            
            if not (depot_ok and farm_ok and capacity_ok):
                continue 

            # Lọc 3: Phân loại VÙNG
            if t.get('region') == depot_region:
                suitable_trucks_IN_REGION.append(t)
            else:
                suitable_trucks_OUT_OF_REGION.append(t)
        
        # -----------------------------------------------------------------
        best_option = (float('inf'), None) 

        # LƯỢT 1: TRONG VÙNG
        for truck_obj in suitable_trucks_IN_REGION:
            truck_id = truck_obj['id']
            last_finish_time, _ = truck_finish_times[truck_id] 
            start_time = last_finish_time + rest if last_finish_time > 0 else 0
            
            for shift in ['AM', 'PM']: 
                finish_time, feasible, first_wait = _calculate_route_schedule_and_feasibility_ini(
                    closest_depot_idx, [i], shift, start_time, problem_instance, truck_obj
                )
                if feasible and finish_time < best_option[0]:
                    best_option = (finish_time, (closest_depot_idx, truck_id, [i], shift, start_time, truck_obj, first_wait))

        # LƯỢT 2: NGOÀI VÙNG
        if best_option[1] is None:
            for truck_obj in suitable_trucks_OUT_OF_REGION:
                truck_id = truck_obj['id']
                last_finish_time, _ = truck_finish_times[truck_id] 
                start_time = last_finish_time + rest if last_finish_time > 0 else 0
                
                for shift in ['AM', 'PM']: 
                    finish_time, feasible, first_wait = _calculate_route_schedule_and_feasibility_ini(
                        closest_depot_idx, [i], shift, start_time, problem_instance, truck_obj
                    )
                    if feasible and finish_time < best_option[0]:
                        best_option = (finish_time, (closest_depot_idx, truck_id, [i], shift, start_time, truck_obj, first_wait))

        # -----------------------------------------------------------------
        if best_option[1] is None:
            print(f"!!! LỖI THỜI GIAN: Farm {i} (Demand {eff_demand}) không thể lên lịch (đã thử cả ngoài vùng).")
            count += 1
            continue

        # LƯU KẾT QUẢ
        new_finish_time, (depot, truck, cust_list, chosen_shift, base_start_time, truck_obj, first_wait) = best_option
        
        optimal_start_time = base_start_time + first_wait
        optimal_finish_time = new_finish_time + first_wait
        assigned_farms.update(cust_list)
        
        truck_finish_times[truck] = (optimal_finish_time, depot)
        
        # [SỬA]: Tính load chính xác (hỗ trợ ID ảo)
        # Vì cust_list ở đây chỉ có 1 phần tử [i], nhưng viết loop cho tổng quát
        current_route_load = 0
        for fid in cust_list:
            if fid in virtual_map:
                current_route_load += virtual_map[fid]['portion']
            else:
                # Lấy demand gốc nếu không phải ảo
                current_route_load += _resolve_farm_for_ci_local(fid)[1]

        depot_load[depot] += current_route_load

        final_schedule.append((depot, truck, cust_list, chosen_shift, optimal_start_time, optimal_finish_time, current_route_load))

        # --- Xử lý quá tải depot (Logic cũ của bạn giữ nguyên) ---
        if depot_load[depot] > depot_capacity[depot]:
            print(f"    -> 🏭 CẢNH BÁO QUÁ TẢI: Depot {depot} đạt {depot_load[depot]:.0f}/{depot_capacity[depot]}.")
            current_region = facilities[depot]['region']
            candidate_target_depots = [d_idx for d_idx in depots_by_region[current_region] if d_idx != depot]
            transfer_truck = None

            if candidate_target_depots:
                target_depot = min(candidate_target_depots, key=lambda d: depot_load[d])
                transfer_amount = depot_load[depot] - depot_capacity[depot] 

                for t in available_trucks:
                    if t.get('region') != depot_region: continue
                    type_idx = t.get('type_idx', 0)
                    src_acc = facilities[depot].get('accessibility', [1]*4)
                    dst_acc = facilities[target_depot].get('accessibility', [1]*4)
                    if (t['capacity'] >= transfer_amount and src_acc[type_idx] == 1 and dst_acc[type_idx] == 1):
                        transfer_truck = t
                        break

                if transfer_truck is None:
                    # Kế hoạch B: Multi-trip
                    dist_matrix_depots = problem_instance.get('distance_matrix_depots')
                    if dist_matrix_depots is not None:
                            dist_one_way = dist_matrix_depots[depot, target_depot]
                            for truck_id, (finish_time, depot_used) in truck_finish_times.items():
                                if facilities[depot_used]['region'] != depot_region: continue
                                temp_truck_info = next((t for t in available_trucks if t['id'] == truck_id), None)
                                if not temp_truck_info: continue 
                                velocity = 1.0 if temp_truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
                                time_one_way = dist_one_way / velocity
                                actual_travel_time_round_trip = time_one_way * 2
                                if finish_time + actual_travel_time_round_trip + 1 < 1900:
                                    transfer_truck = temp_truck_info
                                    print(f" 		✅ Dùng lại Truck {truck_id} (multi-trip) cho INTER-FACTORY transfer.")
                                    break 
                if transfer_truck:
                    transfer_route_customer = [f'TRANSFER_FROM_{depot}_TO_{target_depot}']
                    start_time = truck_finish_times.get(transfer_truck['id'], (0, depot))[0]
                    new_finish_time = start_time + 180 
                    dist_matrix_depots = problem_instance.get('distance_matrix_depots')
                    
                    if dist_matrix_depots is not None:
                        try:
                            velocity = 1.0 if transfer_truck['type'] in ["Single", "Truck and Dog"] else 0.5
                            dist_one_way = dist_matrix_depots[depot, target_depot]
                            new_finish_time = start_time + (dist_one_way / velocity) * 2
                        except Exception: pass

                    depot_end_time = 1900 
                    if new_finish_time > depot_end_time:
                            print(f" 		⚠️ [BỊ HỦY] INTER-FACTORY quá muộn.")
                    else:
                        final_schedule.append((depot, transfer_truck['id'], transfer_route_customer, 'INTER-FACTORY', start_time, new_finish_time, transfer_amount))
                        truck_finish_times[transfer_truck['id']] = (new_finish_time, target_depot) 
                        depot_load[depot] -= transfer_amount
                        depot_load[target_depot] += transfer_amount
                        print(f" 		-> 🚚 Tạo chuyến INTER-FACTORY ({depot}->{target_depot}) thành công.")
                else:
                    print(f" 		⚠️ Không có xe phù hợp cho INTER-FACTORY.")

    # ====================== IN KẾT QUẢ ======================
    print("\n📅 LỊCH TRÌNH CHO NGÀY:")
    if not final_schedule:
        print("(Không có tuyến nào)")
    else:
        truck_routes = defaultdict(list)
        for depot, truck, cust_list, shift, start_time, finish_time, route_load in final_schedule:
            truck_routes[truck].append((depot, cust_list, shift, start_time, finish_time, route_load))

        for truck, trips in truck_routes.items():
            print(f"🚚 Truck {truck} chạy {len(trips)} chuyến:")
            for trip_no, (depot, cust_list, shift, start_time, finish_time, route_load) in enumerate(trips, 1):
                route_str = " → ".join(str(c) for c in cust_list)
                h, m = divmod(int(start_time), 60)
                k, n = divmod(int(finish_time), 60)
                
                if shift == 'INTER-FACTORY':
                    print(f"🏭 Chuyến đặc biệt - Depot {depot} (XP {h:02d}:{m:02d}): {route_str.replace('_', ' ')} -> Kết thúc {k:02d}:{n:02d}")
                else:
                    print(f"🧭 Chuyến {trip_no} ({shift}) - Depot {depot} (XP {h:02d}:{m:02d}): {route_str} (Load: {route_load}) -> Kết thúc {k:02d}:{n:02d}")

    print("\n--- KẾT THÚC COMPUTE_INITIAL_SOLUTION ---")
    print(f"Số nông trại không thể lên lịch: {count}")
    
    # In thống kê xe (như cũ)
    all_truck_ids = {t['id'] for t in available_trucks}
    used_truck_ids = set(truck_finish_times.keys())
    unused_truck_ids = all_truck_ids - used_truck_ids
    print(f"Tổng số xe: {len(all_truck_ids)} | Đã dùng: {len(used_truck_ids)} | Chưa dùng: {len(unused_truck_ids)}")
    
    return final_schedule