import copy
# Giả định các hàm này có thể được import từ repair_operators
# Hoặc bạn có thể tạo một file utils.py chung
from .utils import get_route_cost, _get_dist_and_time, _get_farm_info, _get_service_time, _get_shift_end_time, find_truck_by_id

import copy

# Assumes get_route_cost(problem_instance, route_info) exists and accepts 4- or 5-tuple route_info.

def _unpack_route(route):
    """Trả về (depot_idx, truck_id, cust_list, shift, start_time) an toàn."""
    if len(route) == 5:
        depot_idx, truck_id, cust_list, shift, start_time = route
    else:
        depot_idx, truck_id, cust_list, shift = route
        start_time = 0
    return depot_idx, truck_id, cust_list, shift, start_time

def apply_2_opt(solution):
    """Phiên bản 2-Opt O(M*K^2) sử dụng Delta O(1)"""
    improved_solution = copy.deepcopy(solution)
    problem = improved_solution.problem_instance
    # (Giả sử bạn có hàm _get_dist_and_time)
    
    for route_idx, route_info in enumerate(list(improved_solution.schedule)):
        depot_idx, truck_id, customer_list, shift, start_time = _unpack_route(route_info)
        if len(customer_list) < 2: continue
            
        # Nối depot vào đầu/cuối để tính delta
        idx_list = [_get_farm_info(cid, problem)[0] for cid in customer_list]
        full_idx_list = [depot_idx] + idx_list + [depot_idx]
        is_depot_list = [True] + [False] * len(idx_list) + [True]

        truck_info = find_truck_by_id(truck_id, problem['fleet']['available_trucks'])

        improved_in_route = True
        while improved_in_route:
            improved_in_route = False
            n = len(full_idx_list) - 1 # (Số cạnh)
            
            for i in range(1, n - 2): # Không swap điểm depot đầu
                for j in range(i + 1, n): # j là điểm cuối (không phải depot)
                    
                    # Cạnh hiện tại: (i-1) -> (i) và (j) -> (j+1)
                    # Cạnh mới: (i-1) -> (j) và (i) -> (j+1)
                    
                    A, B = full_idx_list[i-1], full_idx_list[i]
                    C, D = full_idx_list[j], full_idx_list[j+1]
                    
                    is_A_depot = is_depot_list[i-1]
                    is_B_depot = is_depot_list[i]
                    is_C_depot = is_depot_list[j]
                    is_D_depot = is_depot_list[j+1]

                    # Tính chi phí (delta) O(1)
                    dist_AB, _ = _get_dist_and_time(A, B, is_A_depot, is_B_depot, truck_info, problem)
                    dist_CD, _ = _get_dist_and_time(C, D, is_C_depot, is_D_depot, truck_info, problem)
                    
                    dist_AC, _ = _get_dist_and_time(A, C, is_A_depot, is_C_depot, truck_info, problem)
                    dist_BD, _ = _get_dist_and_time(B, D, is_B_depot, is_D_depot, truck_info, problem)

                    cost_delta = (dist_AC + dist_BD) - (dist_AB + dist_CD)
                    
                    if cost_delta < -1e-6: # (Giả sử chỉ tối ưu quãng đường)
                        # Nếu cải thiện, ĐẢO NGƯỢC (reverse)
                        customer_list[i-1:j] = customer_list[i-1:j][::-1]
                        
                        # Cập nhật lại full_idx_list và customer_list cho vòng lặp while
                        idx_list = [_get_farm_info(cid, problem)[0] for cid in customer_list]
                        full_idx_list = [depot_idx] + idx_list + [depot_idx]
                        is_depot_list = [True] + [False] * len(idx_list) + [True]
                        
                        improved_in_route = True
                        break # Bắt đầu lại (first improvement)
                if improved_in_route:
                    break
            
            # Cập nhật schedule trong solution
            improved_solution.schedule[route_idx] = (depot_idx, truck_id, customer_list, shift, start_time)

    return improved_solution

def apply_relocate(solution):
    """
    Relocate (intra-route relocate): thử di chuyển 1 visit trong cùng route
    để tìm vị trí tốt hơn. Hỗ trợ route 4/5-tuple.
    """
    improved_solution = copy.deepcopy(solution)
    problem_instance = improved_solution.problem_instance

    for route_idx, route_info in enumerate(list(improved_solution.schedule)):
        depot_idx, truck_id, customer_list, shift, start_time = _unpack_route(route_info)

        if len(customer_list) < 2 or shift == 'INTER-FACTORY':
            continue

        # current best cost (route-level)
        current_route = improved_solution.schedule[route_idx]
        best_cost = get_route_cost(problem_instance, current_route)
        improved = False

        # try relocate each customer to best position inside same route
        for i in range(len(customer_list)):
            cust = customer_list[i]
            temp_list = customer_list[:i] + customer_list[i+1:]

            best_local_cost = best_cost
            best_local_list = customer_list

            for j in range(len(temp_list) + 1):
                # Insert at position j
                new_list = temp_list[:j] + [cust] + temp_list[j:]
                # If position j equals original position, skip (no-op)
                if new_list == customer_list:
                    continue
                new_route_info = (depot_idx, truck_id, new_list, shift, start_time)
                new_cost = get_route_cost(problem_instance, new_route_info)
                if new_cost < best_local_cost - 1e-6:
                    best_local_cost = new_cost
                    best_local_list = new_list

            # apply best local change for this customer
            if best_local_list is not customer_list:
                customer_list = best_local_list
                improved = True
                best_cost = best_local_cost

        if improved:
            improved_solution.schedule[route_idx] = (depot_idx, truck_id, customer_list, shift, start_time)

    return improved_solution


def apply_exchange(solution):
    """
    Inter-route exchange (swap 1-1) with re-insertion to best positions.
    Works with 4/5-tuple, preserves start_time for both routes.
    """
    improved_solution = copy.deepcopy(solution)
    problem = improved_solution.problem_instance

    was_improved = True
    while was_improved:
        was_improved = False
        schedule = improved_solution.schedule
        # iterate pairs of routes
        for r1_idx in range(len(schedule)):
            for r2_idx in range(r1_idx + 1, len(schedule)):
                route1 = schedule[r1_idx]
                route2 = schedule[r2_idx]

                d1, t1, custs1, s1, start1 = _unpack_route(route1)
                d2, t2, custs2, s2, start2 = _unpack_route(route2)

                if s1 == 'INTER-FACTORY' or s2 == 'INTER-FACTORY':
                    continue
                if not custs1 or not custs2:
                    continue

                # compute current cost
                cost_before = get_route_cost(problem, route1) + get_route_cost(problem, route2)

                # test all swaps cust1 <-> cust2
                improved_here = False
                for i in range(len(custs1)):
                    for j in range(len(custs2)):
                        cust1 = custs1[i]
                        cust2 = custs2[j]

                        # remove items
                        tmp1 = custs1[:i] + custs1[i+1:]
                        tmp2 = custs2[:j] + custs2[j+1:]

                        # find best insertion position for cust2 into tmp1
                        best_cost1 = float('inf')
                        best_new1 = None
                        for pos1 in range(len(tmp1) + 1):
                            cand1 = tmp1[:pos1] + [cust2] + tmp1[pos1:]
                            cand_info1 = (d1, t1, cand1, s1, start1)
                            c1 = get_route_cost(problem, cand_info1)
                            if c1 < best_cost1:
                                best_cost1 = c1
                                best_new1 = cand_info1

                        # best insertion for cust1 into tmp2
                        best_cost2 = float('inf')
                        best_new2 = None
                        for pos2 in range(len(tmp2) + 1):
                            cand2 = tmp2[:pos2] + [cust1] + tmp2[pos2:]
                            cand_info2 = (d2, t2, cand2, s2, start2)
                            c2 = get_route_cost(problem, cand_info2)
                            if c2 < best_cost2:
                                best_cost2 = c2
                                best_new2 = cand_info2

                        # If both insertions feasible and improve
                        if best_new1 is not None and best_new2 is not None:
                            cost_after = best_cost1 + best_cost2
                            if cost_after < cost_before - 1e-6:
                                # apply swap
                                schedule[r1_idx] = best_new1
                                schedule[r2_idx] = best_new2
                                # clean up empty routes if any
                                improved_solution.schedule = [r for r in schedule if r[2]]
                                was_improved = True
                                improved_here = True
                                break
                    if improved_here:
                        break
                if was_improved:
                    break
            if was_improved:
                break

    return improved_solution
