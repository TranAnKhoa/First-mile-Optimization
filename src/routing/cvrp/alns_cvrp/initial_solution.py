import numpy as np
from collections import defaultdict
import random
import re
import copy
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
    print("\n--- BÊN TRONG COMPUTE_INITIAL_SOLUTION (SINGLE-DAY, NÂNG CẤP) ---")
    count = 0 
    onfly_split_done = set() 
    farms = problem_instance['farms'] 
    facilities = problem_instance['facilities'] 
    available_trucks = problem_instance['fleet']['available_trucks'] 
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map'] 
    final_schedule = [] 
    depot_capacity=[]
    farm_demand =[] 
    for i in problem_instance['facilities']:
        depot_capacity.append(i['capacity'])
    depot_load = defaultdict(float) 
    depots_by_region = defaultdict(list) 
    for i in farms:
        farm_demand.append(i["demand"])
    print(farm_demand)

    for i, facility in enumerate(facilities): 
        if 'region' in facility:
            depots_by_region[facility['region']].append(i)
    
    all_required_visits = [farm['id'] for farm in farms]
    random_state.shuffle(all_required_visits)
    truck_finish_times = defaultdict(lambda: (0, -1))
    assigned_farms = set() 
    virtual_map = problem_instance.setdefault('virtual_split_farms', {})

    def _resolve_farm_for_ci_local(fid):
        if isinstance(fid, str) and fid in virtual_map:
            base = virtual_map[fid]['base_id'] 
            portion = virtual_map[fid].get('portion', 0) 
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

    # ====================== MAIN LOOP ======================
    for i in all_required_visits: 
        if i in assigned_farms:
            continue 
        
        effective_id, eff_demand, farm_details, farm_idx = _resolve_farm_for_ci_local(i) 
        closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx])) 
        depot_region = facilities[closest_depot_idx].get('region', None) 
        type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
        eligible_trucks_in_region = []
        for t in available_trucks:
            if t.get('region') != depot_region:
                continue 
            
            t['type_idx'] = type_to_idx.get(t.get('type'), -1)
            depot_ok = facilities[closest_depot_idx].get('accessibility', [1]*4)[t['type_idx']] == 1
            farm_ok = farm_details.get('accessibility', [1]*4)[t['type_idx']] == 1
            if depot_ok and farm_ok:
                eligible_trucks_in_region.append(t) 
        
        if not eligible_trucks_in_region:
            print(f"!!! KHÔNG CÓ XE Ở VÙNG {depot_region} PHÙ HỢP để phục vụ Farm {i}")
            count += 1
            continue
        
        max_capacity_in_region = max(t['capacity'] for t in eligible_trucks_in_region)
        median_demand = np.median(farm_demand)
        
        if eff_demand > max_capacity_in_region and i not in onfly_split_done:
            num_parts = int(np.ceil(eff_demand / median_demand))
            remaining, true_base = eff_demand, _clean_base_id(effective_id) 
            print(f"⚠️ ON-THE-FLY SPLIT: {i} demand {eff_demand} > {median_demand}. Tạo {num_parts} phần.")
            for k in range(num_parts): 
                part_qty = min(median_demand, remaining) 
                split_id = f"{i}_onfly_part{k+1}" 
                virtual_map[split_id] = {'base_id': true_base, 'portion': part_qty} 
                all_required_visits.append(split_id)
                remaining -= part_qty
            assigned_farms.add(i) 
            onfly_split_done.add(i) 
            continue 
        
        suitable_trucks = [t for t in eligible_trucks_in_region if t['capacity'] >= eff_demand] 
        if not suitable_trucks:
            print(f"!!! LỖI TẢI TRỌNG: Không có xe đủ tải cho Farm {i} ở vùng {depot_region}.")
            count += 1
            continue
            
        best_option = (float('inf'), None) 
        for truck_obj in suitable_trucks:
            truck_id = truck_obj['id']
            last_finish_time, _ = truck_finish_times[truck_id] 
            start_time = last_finish_time + 30 if last_finish_time > 0 else 0
            
            for shift in ['AM', 'PM']: 
                
                ### 4. <SỬA> Nhận 3 giá trị (thêm first_wait) ###
                finish_time, feasible, first_wait = _calculate_route_schedule_and_feasibility_ini(
                    closest_depot_idx, [i], shift, start_time, problem_instance, truck_obj
                )
                
                if feasible and finish_time < best_option[0]:
                    ### 5. <SỬA> Lưu 7 giá trị (thêm first_wait) vào best_option ###
                    best_option = (finish_time, (closest_depot_idx, truck_id, [i], shift, start_time, truck_obj, first_wait))

        if best_option[1] is None:
            print(f"!!! LỖI THỜI GIAN: Farm {i} không thể lên lịch.")
            continue

        ### 6. <SỬA> Unpack 7 giá trị (lấy ra first_wait) ###
        new_finish_time, (depot, truck, cust_list, chosen_shift, base_start_time, truck_obj, first_wait) = best_option
        
        ### 7. <SỬA> Tính toán optimal_start_time ###
        # Thời gian xuất phát tối ưu = Thời gian gốc (0) + Thời gian chờ (471)
        optimal_start_time = base_start_time + first_wait
        
        assigned_farms.update(cust_list)
        # Cập nhật finish time cho truck (code này đã đúng)
        truck_finish_times[truck] = (new_finish_time, depot)
        
        route_total_demand = sum(_resolve_farm_for_ci_local(fid)[1] for fid in cust_list)
        depot_load[depot] += route_total_demand

        ### 8. <SỬA> Lưu `optimal_start_time` vào schedule ###
        final_schedule.append((depot, truck, cust_list, chosen_shift, optimal_start_time))

        # --- Xử lý quá tải depot (Không thay đổi) ---
        if depot_load[depot] > depot_capacity[depot]:
            print(f"    -> 🏭 CẢNH BÁO QUÁ TẢI: Depot {depot} đạt {depot_load[depot]:.0f}/{depot_capacity[depot]}.")
            current_region = facilities[depot]['region']
            candidate_target_depots = [d_idx for d_idx in depots_by_region[current_region] if d_idx != depot]
            transfer_truck = None

            if candidate_target_depots:
                target_depot = min(candidate_target_depots, key=lambda d: depot_load[d])
                transfer_amount = depot_load[depot] - depot_capacity[depot] 

                for t in available_trucks:
                    if t.get('region') != depot_region:
                        continue
                    type_idx = t.get('type_idx', 0)
                    src_acc = facilities[depot].get('accessibility', [1]*4)
                    dst_acc = facilities[target_depot].get('accessibility', [1]*4)
                    if (
                        t['capacity'] >= transfer_amount and
                        src_acc[type_idx] == 1 and dst_acc[type_idx] == 1
                    ):
                        transfer_truck = t
                        break

                if transfer_truck is None:
                    for truck_id, (finish_time, depot_used) in truck_finish_times.items():
                        if facilities[depot_used]['region'] == depot_region and finish_time + 180 < 1900:
                            transfer_truck = next((t for t in available_trucks if t['id'] == truck_id), None)
                            if transfer_truck:
                                print(f"        ✅ Dùng lại Truck {truck_id} (multi-trip) cho INTER-FACTORY transfer.")
                                break

                if transfer_truck:
                    transfer_route_customer = [f'TRANSFER_FROM_{depot}_TO_{target_depot}']
                    start_time = truck_finish_times.get(transfer_truck['id'], (0, depot))[0]
                    final_schedule.append(
                        (depot, transfer_truck['id'], transfer_route_customer, 'INTER-FACTORY', start_time)
                    )
                    truck_finish_times[transfer_truck['id']] = (start_time + 180, target_depot)
                    depot_load[depot] -= transfer_amount
                    depot_load[target_depot] += transfer_amount
                    print(f"        -> 🚚 Tạo chuyến INTER-FACTORY ({depot}->{target_depot}) thành công.")
                else:
                    print(f"        ⚠️ Không có xe phù hợp cho INTER-FACTORY transfer giữa {depot} và {target_depot}.")

    # ====================== In ra lịch trình kết quả (tổng quan) ======================
    print("\n📅 LỊCH TRÌNH CHO NGÀY:")
    if not final_schedule:
        print("  (Không có tuyến nào)")
    else:
        truck_routes = defaultdict(list)
        for depot, truck, cust_list, shift, start_time in final_schedule:
            truck_routes[truck].append((depot, cust_list, shift, start_time))
        
        for truck, trips in truck_routes.items():
            print(f"  🚚 Truck {truck} chạy {len(trips)} chuyến:")
            for trip_no, (depot, cust_list, shift, start_time) in enumerate(trips, 1):
                route_str = " → ".join(str(c) for c in cust_list)
                if shift == 'INTER-FACTORY':
                    print(f"    🏭 Chuyến đặc biệt ({shift}): {route_str.replace('_', ' ')}")
                else:
                    # In ra thời gian xuất phát đã được tối ưu
                    h, m = divmod(int(start_time), 60)
                    print(f"    🧭 Chuyến {trip_no} ({shift}) - Depot {depot} (Xuất phát {h:02d}:{m:02d}): Depot {depot} → {route_str} → Depot {depot}")

    print("\n--- KẾT THÚC COMPUTE_INITIAL_SOLUTION ---")
    print(f"Số nông trại không thể lên lịch: {count}")
    return final_schedule