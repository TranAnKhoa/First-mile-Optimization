import os
import sys
import time
import random
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from stable_baselines3 import PPO
#! python run_benchmark.py
# ==============================================================================
# 1. SETUP ĐƯỜNG DẪN & IMPORT
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

try:
    from src.rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP
    from src.routing.cvrp.alns_cvrp import cvrp_helper_functions
    from src.routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from src.routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from src.routing.cvrp.alns_cvrp.utils import optimize_all_start_times, cleanup_inter_factory_routes, update_history_matrix
    from src.routing.cvrp.alns_cvrp.destroy_operators import (
        random_removal, worst_removal_alpha_0, worst_removal_bigM, 
        worst_removal_adaptive, time_worst_removal, shaw_spatial, 
        shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, 
        historical_removal, update_solution_state_after_destroy
    )
    from src.routing.cvrp.alns_cvrp.repair_operators import (
        best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, 
        regret_3_position, regret_3_trip, regret_3_vehicle, 
        regret_4_position, regret_4_trip, regret_4_vehicle
    )
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# ==============================================================================
# 2. CẤU HÌNH
# ==============================================================================
N_RUNS = 1                 
ITERATIONS = 1000          

INSTANCE_FILE = os.path.join(PROJECT_ROOT, "output_data", "CEL_instance.pkl")
# Đường dẫn model PPO của bạn
MODEL_PATH = r"K:\Data Science\SOS lab\Project Code\trained_model\ppo_macro_22_150000_steps.zip"

print(f"📂 Loading Data: {INSTANCE_FILE}")
if not os.path.exists(INSTANCE_FILE):
    print("❌ Data file not found!")
    sys.exit(1)
(_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

# ==============================================================================
# HÀM CHẠY 1: PPO MODEL (GIỮ NGUYÊN NHƯ BẠN YÊU CẦU)
# ==============================================================================
def run_ppo_session(seed):
    print(f"\n   🔴 [PPO AI Agent] Start Run (Seed {seed})...")
    
    # Init Env
    env = PPO_ALNS_Env_GP(problem_instance=problem_obj, max_iterations=ITERATIONS, buffer_size=1)
    
    # Load Model
    full_path = MODEL_PATH + ".zip" if not MODEL_PATH.endswith(".zip") else MODEL_PATH
    if not os.path.exists(full_path):
        print(f"⚠️ Model not found: {full_path}")
        return float('inf'), 0 

    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return float('inf'), 0

    obs, _ = env.reset(seed=seed)
    
    start_time = time.time()
    done = False
    step_cnt = 0
    
    while not done:
        step_cnt += 1
        # deterministic=False (Theo yêu cầu của bạn: giữ nguyên logic cũ)
        action, _ = model.predict(obs, deterministic=False)
        
        step_res = env.step(action)
        if len(step_res) == 5: obs, reward, done, trunc, info = step_res
        else: obs, reward, done, info = step_res
        
        if step_cnt % 200 == 0 or done:
            best_obj = env.best_objective
            curr_obj = env.current_solution.objective()[0]
            print(f"      Step {step_cnt:4d}: Best={best_obj:.2f} | Curr={curr_obj:.2f}")
            
    return env.best_objective, time.time() - start_time

# ==============================================================================
# HÀM CHẠY 2: ALNS ĐƠN THUẦN (SỬA LOGIC THEO FILE 1 BẠN GỬI)
# ==============================================================================
def run_atomic_alns_session(seed):
    """
    Tái tạo logic của file gốc, nhưng có thêm bước CLEANUP ban đầu
    để khớp Input Cost với môi trường PPO.
    """
    print(f"\n   🔵 [Pure ALNS] Start Run (Seed {seed})...")
    
    # Khởi tạo RandomState
    rand = np.random.RandomState(seed)
    
    # --- 1. TẠO LỜI GIẢI BAN ĐẦU ---
    initial_schedule = compute_initial_solution(problem_obj, rand)
    env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem_obj, seed=seed)
    
    # [QUAN TRỌNG]: Thêm dòng này để giống với PPO Env
    # Nó sẽ loại bỏ các tuyến thừa/rác ngay từ đầu -> Cost giảm từ 52k xuống 50k
    env = cleanup_inter_factory_routes(env)
    
    best_solution = env
    current_solution = env
    
    # Lấy giá trị mục tiêu ban đầu
    best_obj = best_solution.objective()[0]
    current_obj = best_obj
    
    print(f"      [Init] Cost ban đầu (Sau Cleanup): {best_obj:.2f}") # In ra để kiểm tra
    
    # Init History Matrix
    global_history_matrix = {}
    update_history_matrix(global_history_matrix, current_solution)

    # --- CẤU HÌNH SIMULATED ANNEALING ---
    start_temperature = 1000
    end_temperature = 0.1
    cooling_rate = 0.999
    temperature = start_temperature
    
    # --- DANH SÁCH TOÁN TỬ ---
    destroy_operators = [random_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, time_worst_removal,
                         shaw_spatial, shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, historical_removal]
    repair_operators = [best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, 
                        regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle]
    
    MAX_REMOVE_FRACTION = 0.4
    MIN_REMOVE_FRACTION = 0.05
    
    start_time = time.time()
    
    for i in range(ITERATIONS):
        try:
            # 1. CHỌN TOÁN TỬ (Uniform)
            destroy_op = rand.choice(destroy_operators)
            repair_op = rand.choice(repair_operators)
            
            progress = i / ITERATIONS
            remove_fraction = MAX_REMOVE_FRACTION - (MAX_REMOVE_FRACTION - MIN_REMOVE_FRACTION) * progress
            
            op_kwargs = {
                'remove_fraction': remove_fraction,
                'history_matrix': global_history_matrix
            }
            
            # 2. PHÁ HỦY & SỬA CHỮA
            destroyed, unvisited = destroy_op(current_solution, rand, **op_kwargs)
            if not unvisited: continue
            
            farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
            if not farms_to_reinsert: continue
                
            repaired, failed_to_insert = repair_op(destroyed, rand, unvisited_customers=farms_to_reinsert)
            
            if not failed_to_insert:
                refined_solution = repaired
                
                # Tối ưu Start Time nhẹ
                try:
                    refined_solution = optimize_all_start_times(refined_solution)
                except: pass

                refined_obj = refined_solution.objective()[0]
                
                # 3. CHẤP NHẬN (Greedy + SA)
                if refined_obj < best_obj:
                    best_solution = refined_solution
                    current_solution = refined_solution
                    best_obj = refined_obj
                    update_history_matrix(global_history_matrix, best_solution)
                
                elif rand.random() < math.exp((current_obj - refined_obj) / temperature):
                    current_solution = refined_solution
                    current_obj = refined_obj
                    update_history_matrix(global_history_matrix, current_solution)

            # 4. GIẢM NHIỆT ĐỘ
            temperature = max(end_temperature, temperature * cooling_rate)
            
            # Logging
            if i % 200 == 0 or i == ITERATIONS - 1:
                print(f"      Iter {i:4d}: Best={best_obj:.2f} | Curr={current_obj:.2f}")
                
        except Exception:
            continue

    duration = time.time() - start_time
    
    # Hậu xử lý cuối cùng
    try:
        best_solution = cleanup_inter_factory_routes(best_solution)
        best_solution = optimize_all_start_times(best_solution)
        best_obj = best_solution.objective()[0]
    except: pass
    
    return best_obj, duration

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"\n🚀 BẮT ĐẦU BENCHMARK (n={N_RUNS}, steps={ITERATIONS})")
    print(f"   Model: {os.path.basename(MODEL_PATH)}")
    print("-" * 60)

    results = []
    # Dùng seed cố định để kiểm tra
    seeds = [1234] 
    # seeds = np.random.randint(1000, 9999, size=N_RUNS) # Bỏ comment nếu chạy nhiều seed

    for i, seed in enumerate(seeds):
        print(f"\n🔹 ROUND {i+1}/{len(seeds)} (Seed: {seed})")
        
        # 1. Chạy Pure ALNS (Logic gốc)
        base_cost, base_time = run_atomic_alns_session(int(seed))
        
        # 2. Chạy PPO
        ppo_cost, ppo_time = run_ppo_session(int(seed))
        
        # Tính toán
        if base_cost > 0: 
            gap = ((base_cost - ppo_cost) / base_cost) * 100
        else: 
            gap = 0
            
        print(f"\n   🏁 KẾT QUẢ ROUND {i+1}:")
        print(f"   - Pure ALNS: {base_cost:.2f} (Time: {base_time:.1f}s)")
        print(f"   - PPO Agent: {ppo_cost:.2f} (Time: {ppo_time:.1f}s)")
        print(f"   👉 Gap: {gap:+.2f}% ({'PPO TỐT HƠN' if gap > 0 else 'ALNS TỐT HƠN'})")

        results.append({
            'Run': i+1, 'Seed': seed,
            'Baseline Cost': base_cost, 'PPO Cost': ppo_cost, 
            'Baseline Time': base_time, 'PPO Time': ppo_time,
            'Gap (%)': gap
        })

    # Export Report
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("📊 TỔNG HỢP KẾT QUẢ")
    print("="*70)
    print(df.to_string(index=False))
    
    csv_path = os.path.join(PROJECT_ROOT, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Đã lưu kết quả vào: {csv_path}")
    
    # Vẽ biểu đồ đơn giản
    plt.figure(figsize=(8, 5))
    df_melt = df.melt(id_vars=['Run'], value_vars=['Baseline Cost', 'PPO Cost'], var_name='Model', value_name='Cost')
    sns.barplot(data=df_melt, x='Run', y='Cost', hue='Model')
    plt.title("So sánh Cost: Pure ALNS vs PPO")
    plt.savefig(os.path.join(PROJECT_ROOT, "benchmark_chart.png"))

if __name__ == "__main__":
    main()