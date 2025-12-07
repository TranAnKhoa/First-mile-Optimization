import csv
import sys
import os
import time
import copy
import random
from pathlib import Path
from collections import Counter, defaultdict
#! python run_drl_alns_cvrp_gpseq.py
# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
# Lấy đường dẫn file hiện tại: .../src/routing/cvrp/run_drl_alns_cvrp_gpseq.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục 'src' vào path (để import được 'routing')
src_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'src'))
if src_path not in sys.path: sys.path.insert(0, src_path)

# Thêm thư mục gốc Project
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

# --- IMPORT MODULES ---
from stable_baselines3 import PPO
from src.rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP 
from src.routing.cvrp.alns_cvrp import cvrp_helper_functions
import helper_functions 

# Import Utils
try:
    from routing.cvrp.alns_cvrp.utils import (
        fmt, 
        reconstruct_truck_finish_times, 
        balance_depot_loads,
        _calculate_route_schedule_and_feasibility
    )
    PRINT_FUNC_LOADED = True
except ImportError as e:
    print(f"⚠️ CẢNH BÁO: Không thể import utils. Lỗi: {e}")
    PRINT_FUNC_LOADED = False

# --- CÁC HẰNG SỐ ---
DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = r'K:\Data Science\SOS lab\Project Code\src\routing\cvrp\configs\drl_alns_cvrp_debug.json'

# ==============================================================================
# 2. HÀM IN ẤN & TÌM KIẾM (HELPER)
# ==============================================================================
def find_truck_by_id(truck_id, truck_list):
    for truck in truck_list:
        if truck['id'] == truck_id:
            return truck
    return None

def print_full_solution_details(solution_env, title):
    """Hàm in kết quả Compact & Robust"""
    print(f"\n\n{'='*60}")
    print(f"=== {title} ===")
    print(f"{'='*60}")

    try:
        problem_instance = solution_env.problem_instance
        available_trucks = problem_instance['fleet']['available_trucks']
    except AttributeError:
        print("LỖI: Đối tượng solution không hợp lệ.")
        return

    if not solution_env.schedule:
        print("  (Không có tuyến đường nào)")
        return

    # Nhóm theo Truck ID
    truck_routes_map = defaultdict(list)
    for route_info in solution_env.schedule:
        try:
            if len(route_info) >= 7:
                depot_idx, truck_id, customer_list, shift, start, finish, load = route_info[:7]
            else:
                depot_idx, truck_id, customer_list, shift, start = route_info[:5]
                finish = 0
                load = 0
            truck_routes_map[truck_id].append((depot_idx, truck_id, customer_list, shift, start, finish, load))
        except ValueError:
            continue

    sorted_truck_ids = sorted(truck_routes_map.keys())

    for truck_id in sorted_truck_ids:
        routes = truck_routes_map[truck_id]
        routes.sort(key=lambda x: x[4]) 
        
        truck_info = find_truck_by_id(truck_id, available_trucks)
        truck_cap = truck_info.get('capacity', 0) if truck_info else 0
        truck_type = truck_info.get('type', 'Unknown') if truck_info else 'Unknown'
        
        print(f"🚚 Truck {truck_id} ({truck_type}) chạy {len(routes)} chuyến:")

        for trip_idx, route_data in enumerate(routes, 1):
            depot_idx, _, customer_list, shift, start, finish, load = route_data
            
            try:
                if shift == 'INTER-FACTORY':
                    velocity = 1.0 if truck_type in ["Single", "Truck and Dog"] else 0.5
                    task_name = str(customer_list[0])
                    if finish == 0: finish = start + 60 
                    
                    total_dist = (finish - start) * velocity
                    total_wait = 0.0
                    time_pen = max(0, finish - 1900)
                    cap_pen = 0.0
                    
                    route_str = f"{task_name.replace('_', ' ')}"
                    icon = "🏭"
                    trip_name = "Chuyến đặc biệt"
                else:
                    if PRINT_FUNC_LOADED:
                        calc_results = _calculate_route_schedule_and_feasibility(
                            depot_idx, customer_list, shift, start, finish, load, problem_instance, truck_info
                        )
                        _, total_dist, total_wait, time_pen, cap_pen = calc_results[:5]
                    else:
                        total_dist, total_wait, time_pen, cap_pen = 0, 0, 0, 0

                    route_str = f"Depot {depot_idx} → {' → '.join(map(str, customer_list))} → Depot {depot_idx}"
                    icon = "🧭"
                    trip_name = f"Chuyến {trip_idx}"

            except Exception as e:
                total_dist, total_wait, time_pen, cap_pen = 0, 0, 0, 0
                route_str = f"Route: {customer_list}"
                icon = "⚠️"
                trip_name = f"Chuyến {trip_idx}"

            sh, sm = divmod(int(start), 60)
            eh, em = divmod(int(finish), 60)
            
            print(f"   {icon} {trip_name} ({shift}) - {sh:02d}:{sm:02d} -> {eh:02d}:{em:02d}")
            pen_flag = "⚠️ " if (time_pen > 0 or cap_pen > 0) else ""
            print(f"      📊 Stats: Dist: {total_dist:.1f} km | Wait: {total_wait:.1f} min | "
                  f"Load: {load:.0f}/{truck_cap:.0f} | {pen_flag}TimePen: {time_pen:.1f} | CapPen: {cap_pen:.1f}")

# ==============================================================================
# 3. HÀM CHẠY ĐÁNH GIÁ (RUN EVALUATION)
# ==============================================================================
def run_evaluation(folder, exp_name, problem_instance, **kwargs):
    # 1. Trích xuất tham số
    instance_nr = kwargs['instance_nr']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # 2. Khởi tạo môi trường
    print("\n--- Khởi tạo môi trường PPO_ALNS_Env_GP ---")
    env = PPO_ALNS_Env_GP(problem_instance=problem_instance, max_iterations=iterations, buffer_size=1)

    # 3. Tải model PPO
    model_path = kwargs['model_directory'] 
    print(f"Đang tải model từ: {model_path}")
    model = PPO.load(model_path)
    
    # 4. KHỞI TẠO BIẾN BAN ĐẦU
    print(f"Resetting Env with Seed: {seed}")
    obs, _ = env.reset(seed=seed)
    
    if hasattr(env, 'initial_solution'):
        int_solution = copy.deepcopy(env.initial_solution)
    else:
        int_solution = copy.deepcopy(env.current_solution)

    # --- SỬA LẠI ĐOẠN NÀY ---
    # Lấy bộ giá trị mục tiêu (Total Cost, Time Penalty, Wait Time, Capacity Penalty)
    initial_objectives = int_solution.objective()
    
    print(f"\n📊 KẾT QUẢ BAN ĐẦU:")
    print(f"   ► Tổng Cost: {initial_objectives[0]:.2f}")
    print(f"   ► Phạt thời gian: {initial_objectives[1]:.2f}")
    print(f"   ► Thời gian chờ: {initial_objectives[2]:.2f}")
    # 5. VÒNG LẶP CHÍNH (LIVE TRACKING)
    print("\n--- Bắt đầu vòng lặp PPO/ALNS ---")
    start_time = time.time()
    
    action_history = [] 
    step_counter = 0
    done = False
    
    # In Header bảng
    print("\n" + "="*85)
    print(f"{'ITER':<6} | {'OP #':<6} | {'STATUS':<10} | {'OBJECTIVE':<12} | {'SEQUENCE DETAIL'}")
    print("="*85)

    while not done:
        step_counter += 1
        
        # PPO chọn hành động (False = Sáng tạo hơn)
        action, _states = model.predict(obs, deterministic=False)
        op_index = int(action)
        action_history.append(op_index)
        
        # Bước environment
        step_result = env.step(action)
        
        # Xử lý kết quả trả về linh hoạt
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            
        # Lấy thông tin từ Info để in ra
        is_accepted = info.get('accepted', False)
        current_best = info.get('best_objective', 0)
        status_str = "ACCEPTED" if is_accepted else "REJECTED"
        
        # Lấy mô tả tuyệt kỹ
        op_desc = ""
        if hasattr(env, 'macro_ops') and env.macro_ops:
            try:
                op_data = env.macro_ops[op_index]
                # Ưu tiên lấy sequence_pretty nếu có
                if 'sequence_pretty' in op_data:
                    raw_seq = op_data['sequence_pretty']
                    if isinstance(raw_seq, list): op_desc = " => ".join(raw_seq)
                    else: op_desc = str(raw_seq)
                else:
                    op_desc = str(op_data.get('sequence_indices', []))
            except: pass
            
        # Cắt bớt nếu chuỗi quá dài
        if len(op_desc) > 50: op_desc = op_desc[:47] + "..."

        # IN RA DÒNG LOG
        print(f"{step_counter:<6} | #{op_index:<4} | {status_str:<10} | {current_best:<12.2f} | {op_desc}")

        if done: break
            
    run_duration = time.time() - start_time
    print(f"\n--- Vòng lặp kết thúc sau {step_counter} bước. Thời gian: {run_duration:.2f}s ---")

    # ==========================================================================
    # 6. IN THỐNG KÊ CHIẾN THUẬT
    # ==========================================================================
    print("\n" + "="*60)
    print("📜 LỊCH SỬ CHIẾN THUẬT (ACTION STATS)")
    print("="*60)
    
    print("\n📊 TẦN SUẤT SỬ DỤNG (Top 10):")
    counts = Counter(action_history)
    for op_idx, count in counts.most_common(10):
        percentage = (count / len(action_history)) * 100
        op_name = "Unknown"
        if hasattr(env, 'macro_ops') and env.macro_ops:
            try:
                raw = env.macro_ops[op_idx].get('sequence_pretty', [])
                if isinstance(raw, list): op_name = " => ".join(raw)
                else: op_name = str(raw)
            except: pass
        print(f"   Op #{op_idx:<2}: {count:4d} lần ({percentage:5.1f}%) | 👉 {op_name}")

    # ==========================================================================
    # 7. POST-PROCESSING & IN KẾT QUẢ
    # ==========================================================================
    
    # Lấy Best Solution (Bản Farm Only)
    best_solution_farm_only = copy.deepcopy(env.best_solution)
    final_obj_farm_only = best_solution_farm_only.objective()
    
    print("\n" + "="*60)
    print(">>> BẮT ĐẦU POST-PROCESSING & IN KẾT QUẢ <<<")
    
    # [BƯỚC A]: Xóa sạch Inter-Factory cũ
    best_solution_farm_only.schedule = [r for r in best_solution_farm_only.schedule if r[3] != 'INTER-FACTORY']

    # [BƯỚC B]: Tạo bản sao Full (có Inter-Factory) để in đẹp
    best_solution_full = copy.deepcopy(best_solution_farm_only)
    
    try:
        from routing.cvrp.alns_cvrp.utils import reconstruct_truck_finish_times, balance_depot_loads
        final_finish_times = reconstruct_truck_finish_times(best_solution_full)
        best_solution_full = balance_depot_loads(best_solution_full, final_finish_times)
        print("✅ Đã tạo lịch trình đầy đủ (bao gồm chuyển kho).")
    except Exception as e:
        print(f"⚠️ Lỗi Post-processing: {e}")
        best_solution_full = best_solution_farm_only 

    print(f"{'='*60}\n")

    # --- IN KẾT QUẢ BAN ĐẦU ---
    if 'int_solution' in locals():
        print_full_solution_details(int_solution, "CHI TIẾT LỊCH TRÌNH BAN ĐẦU")
        ini_obj = int_solution.objective()
        print(f"Initial Objective: {ini_obj[0]:.2f}")
        print(f"Initial Time Penalty: {ini_obj[1]:.2f}")
        print(f"Initial Wait Time: {ini_obj[2]:.2f}")
        print(f"Initial Capacity Penalty: {ini_obj[3]:.2f}")

    # --- IN KẾT QUẢ FINAL (FULL) ---
    print_full_solution_details(best_solution_full, "CHI TIẾT LỊCH TRÌNH TỐT NHẤT (FINAL)")
    
    # --- TỔNG KẾT ---
    print(f"\n🏆 FINAL OBJECTIVE (Routing Only): {final_obj_farm_only[0]:.2f}")
    print(f"   - Time Penalty: {final_obj_farm_only[1]:.2f}")
    print(f"   - Wait Time:    {final_obj_farm_only[2]:.2f}")
    print(f"   - Cap Penalty:  {final_obj_farm_only[3]:.2f}")
    
    full_obj = best_solution_full.objective()
    print(f"\nℹ️  Total Logistic Cost (Inc. Inter-Factory): {full_obj[0]:.2f}")

    if 'start_time' in locals():
        print(f"\n⏱️ Tổng thời gian chạy: {time.time() - start_time:.2f} giây")

    # 8. Ghi CSV
    # 8. Ghi kết quả ra file CSV (Tạo file mới mỗi lần chạy)
    try:
        print(f"\n--- Đang ghi kết quả ra file CSV ---")
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        # [MỚI] Thêm timestamp vào tên file để không bị ghi đè
        # Ví dụ: drl_alns_eval_1_1234_20231027_153045.csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}{exp_name}_{timestamp}.csv"
        
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['problem_instance', 'rseed', 'iterations', 'best_objective', 'final_logistic_cost', 'solution_schedule', 'instance_file', 'timestamp'])
            
            # Data row (Thêm cả chi phí Logistic tổng để tiện so sánh sau này)
            writer.writerow([
                instance_nr, 
                seed, 
                iterations, 
                final_obj_farm_only[0],   # Routing Cost (Benchmark)
                full_obj[0],              # Logistic Cost (Thực tế)
                str(best_solution_full.schedule), 
                kwargs['instance_file'],
                timestamp
            ])
            
        print(f"✅ Đã ghi xong: {filename}")
        
    except Exception as e:
        print(f"❌ Lỗi ghi file CSV: {e}")

    return final_obj_farm_only[0]
# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================
def main(param_file=PARAMETERS_FILE):
    try:
        print(f"Đang đọc file tham số: {param_file}")
        parameters = helper_functions.readJSONFile(param_file)
        
        base_path = Path(__file__).parent.parent.parent
        instance_file = str(base_path.joinpath(parameters['instance_file']))
        
        print(f"Đang đọc dữ liệu instance từ: {instance_file}")
        (_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(instance_file)
        
        folder = DEFAULT_RESULTS_ROOT
        exp_name = 'drl_alns_eval_' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])
        
        run_evaluation(folder, exp_name, problem_instance=problem_obj, **parameters)

    except Exception as e:
        print(f"\n❌ LỖI TRONG HÀM MAIN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()