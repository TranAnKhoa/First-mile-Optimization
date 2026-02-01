import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import math
import random
import json
import os
import sys

# ==============================================================================
# [FIX PATH]: ÉP ĐƯỜNG DẪN SRC VÀO HỆ THỐNG
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_path not in sys.path: sys.path.insert(0, src_path)

# ==============================================================================
# 1. IMPORT MODULES
# ==============================================================================
try:
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    
    from routing.cvrp.alns_cvrp.destroy_operators import (
        random_removal, worst_removal_alpha_0, worst_removal_bigM, 
        worst_removal_adaptive, time_worst_removal, shaw_spatial, 
        shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, 
        historical_removal, update_solution_state_after_destroy
    )
    
    from routing.cvrp.alns_cvrp.repair_operators import (
        best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, 
        regret_3_position, regret_3_trip, regret_3_vehicle, 
        regret_4_position, regret_4_trip, regret_4_vehicle
    )
    
    from routing.cvrp.alns_cvrp.utils import (
        optimize_all_start_times, update_history_matrix, cleanup_inter_factory_routes
    )
except ImportError as e:
    print(f"❌ [Env] Lỗi Import: {e}")
    raise e

# ==============================================================================
# 2. CẤU HÌNH TOÁN TỬ
# ==============================================================================
DESTROY_OPS = [random_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, time_worst_removal, shaw_spatial, shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, historical_removal]
REPAIR_OPS = [best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle]
REMOVE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def get_op_name(op):
    if hasattr(op, '__name__'): return op.__name__
    if hasattr(op, 'func'): return op.func.__name__
    return str(op)

# ==============================================================================
# 3. CLASS PPO ALNS (MACRO-OP VERSION - ROBUST ROLLBACK)
# ==============================================================================
class PPO_ALNS_Env_GP(gym.Env):
    def __init__(self, problem_instance, max_iterations=200, buffer_size=1, **kwargs):
        super(PPO_ALNS_Env_GP, self).__init__()
        
        self.problem_instance = problem_instance
        self.random_state = np.random.RandomState()
        
        # --- LOAD TUYỆT KỸ TỪ JSON ---
        json_filename = 'macro_advanced_safety.json'
        # Tìm ở thư mục hiện tại hoặc thư mục gốc
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), json_filename)
        if not os.path.exists(json_path): json_path = json_filename 

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.macro_ops = json.load(f)
            print(f"✅ [Env] Loaded {len(self.macro_ops)} Macro-Operators from {json_filename}")
        else:
            print(f"⚠️ [Env] Warning: '{json_filename}' not found. Dummy mode activated.")
            self.macro_ops = []

        # --- ACTION SPACE ---
        self.num_actions = len(self.macro_ops) if self.macro_ops else 10
        self.action_space = spaces.Discrete(self.num_actions)
        
        # --- OBSERVATION SPACE ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.max_iterations = max_iterations
        self.buffer_size = buffer_size 

        # Init variables
        self.current_solution = None
        self.best_solution = None
        self.history_matrix = {}
        
        self.best_objective = float('inf')
        self.initial_objective = float('inf')
        self.best_time_penalty = 0
        self.best_wait_time = 0
        self.best_capacity_penalty = 0
        
        self.stag_count = 0
        self.current_iteration = 0
        
        # Thêm biến theo dõi lịch sử hành động để chống spam
        self.last_actions = []
        

    # ==========================================================================
    # HÀM HỖ TRỢ (HELPER METHODS)
    # ==========================================================================
    
    def _count_customers(self, solution):
        """Đếm tổng số khách hàng thực tế (trừ điểm TRANSFER)"""
        count = 0
        if not solution.schedule: return 0
        for route in solution.schedule:
            # route[2] là customer_list
            if len(route) >= 3:
                # Đếm tất cả trừ các node TRANSFER
                count += sum(1 for cust_id in route[2] if not str(cust_id).startswith('TRANSFER_'))
        return count

    def _sanitize_and_repair(self, solution):
        """
        [HARD CONSTRAINT FIX]: Chỉ sửa lỗi Capacity.
        Time Window là Soft -> Không cần sửa, cứ để ALNS tự tối ưu sau.
        """
        # 1. Kiểm tra nhanh: Nếu không vi phạm Capacity -> Valid ngay lập tức
        # (Bỏ qua time_pen vì đó là soft constraint)
        _, _, _, cap_pen = solution.objective()
        if cap_pen == 0:
            return solution

        # Lưu lại số lượng khách để đảm bảo không mất
        initial_count = self._count_customers(solution)
        
        # 2. Chiến thuật sửa Capacity:
        # Dùng 'worst_removal_adaptive' để loại bỏ những điểm "xấu" (gây tốn chi phí/quá tải)
        # Không dùng 'time_worst' vì nó không giải quyết vấn đề tải trọng.
        destroy_op = worst_removal_adaptive
        
        # Dùng 'best_insertion' (tham lam) để lấp chỗ trống nhanh nhất có thể
        repair_op = best_insertion 
        
        op_kwargs = {'remove_fraction': 0.15, 'history_matrix': self.history_matrix} # Xóa 15% để giảm tải

        try:
            # --- PHÁ HỦY ---
            destroyed, unvisited = destroy_op(solution, self.random_state, **op_kwargs)
            destroyed = update_solution_state_after_destroy(destroyed)
            
            if not unvisited: return solution # Không xóa được gì -> Bó tay

            farms = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
            if not farms: return solution

            # --- SỬA CHỮA ---
            repaired, failed_to_insert = repair_op(destroyed, self.random_state, unvisited_customers=farms)
            
            # --- KIỂM TRA KẾT QUẢ ---
            if failed_to_insert:
                return solution # Sửa thất bại -> Trả về cái cũ (chấp nhận phạt nặng để AI học tránh)
            
            if self._count_customers(repaired) < initial_count:
                return solution # Mất khách -> Rollback
            
            # Kiểm tra xem đã hết lỗi Capacity chưa?
            _, _, _, new_cap_pen = repaired.objective()
            
            # Nếu hết lỗi Capacity (new_cap_pen == 0) -> TUYỆT VỜI
            # Nếu vẫn còn lỗi -> Vẫn trả về 'repaired' vì hy vọng nó đã đỡ hơn cái cũ.
            return repaired

        except Exception:
            return solution # Gặp lỗi code -> An toàn trả về cũ

    def _execute_macro_op(self, op_index, solution):
        """
        [MODIFIED] Thực thi Tuyệt kỹ với cơ chế ROLLBACK TỪNG BƯỚC.
        Nếu bước i thành công -> Lưu lại.
        Nếu bước i+1 gây Infeasible/Lỗi -> Quay lại kết quả bước i và dừng ngay.
        """
        # Đây là solution "đang làm việc". Khởi đầu bằng solution gốc.
        working_sol = copy.deepcopy(solution)
        
        if not self.macro_ops: return working_sol 
        
        if op_index >= len(self.macro_ops):
            op_index = op_index % len(self.macro_ops)
            
        op_data = self.macro_ops[op_index]
        sequence_indices = op_data['sequence_indices'] 
        op_kwargs = {'history_matrix': self.history_matrix}
        
        # Đếm số khách gốc để đảm bảo không bao giờ bị mất khách
        base_customer_count = self._count_customers(solution)

        # Duyệt qua từng cặp (Destroy -> Repair) trong Macro
        for i, step_indices in enumerate(sequence_indices):
            
            # 1. TẠO CHECKPOINT (Lưu trạng thái tốt của bước trước)
            step_backup = copy.deepcopy(working_sol)
            
            # Giải mã tham số
            if len(step_indices) == 2:
                d_idx, r_idx = step_indices
                p_idx = 2 # Default 15%
            else:
                d_idx, p_idx, r_idx = step_indices
            
            try:
                d_op = DESTROY_OPS[d_idx]
                op_kwargs['remove_fraction'] = REMOVE_LEVELS[p_idx]
                r_op = REPAIR_OPS[r_idx]
                
                # Cleanup nhẹ trước khi destroy
                current_step_sol = cleanup_inter_factory_routes(working_sol)
                
                # --- A. EXECUTE DESTROY ---
                destroyed, unvisited = d_op(current_step_sol, self.random_state, **op_kwargs)
                destroyed = update_solution_state_after_destroy(destroyed)
                
                # --- B. EXECUTE REPAIR ---
                if unvisited:
                    farms = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
                    if farms:
                        repaired, failed_to_insert = r_op(destroyed, self.random_state, unvisited_customers=farms)
                        
                        # [CHECK 1]: Repair thất bại? -> ROLLBACK & BREAK
                        if failed_to_insert:
                            # print(f"   ⚠️ Step {i} failed to insert. Rolling back to previous step.")
                            working_sol = step_backup
                            break 
                        
                        current_step_sol = repaired
                    else: 
                        current_step_sol = destroyed
                else: 
                    current_step_sol = destroyed
                
                # --- C. SANITIZE & OPTIMIZE ---
                # Cố gắng sửa lỗi vi phạm (nếu có)
                current_step_sol = self._sanitize_and_repair(current_step_sol)
                current_step_sol = optimize_all_start_times(current_step_sol)

                # --- D. VALIDATION (QUAN TRỌNG NHẤT) ---
                
                # Kiểm tra 1: Mất khách hàng?
                current_count = self._count_customers(current_step_sol)
                if current_count < base_customer_count:
                    # print(f"   ⚠️ Step {i} lost customers. Rolling back.")
                    working_sol = step_backup
                    break

                # Kiểm tra 2: Infeasible (Vi phạm ràng buộc cứng)?
                # Theo yêu cầu: "tới cặp 3 làm lời giải infeasible thì giữ cặp 2"
                _, time_pen, _, cap_pen = current_step_sol.objective()
                if time_pen > 0 or cap_pen > 0:
                    # print(f"   ⚠️ Step {i} caused Infeasibility (Penalties). Rolling back.")
                    working_sol = step_backup
                    break

                # --- E. COMMIT ---
                # Nếu vượt qua mọi bài test, chấp nhận bước này làm nền tảng cho bước sau
                working_sol = current_step_sol
                
                # (Optional Debug): Nếu bước này tốt hơn Global Best, ta vẫn tiếp tục chạy
                # để xem có tốt hơn nữa không, nhưng working_sol đã lưu lại trạng thái tốt này rồi.

            except Exception as e:
                # print(f"   ❌ Error in Step {i}: {e}. Rolling back.")
                working_sol = step_backup
                break 

        # Kết thúc vòng lặp (hoặc do chạy hết, hoặc do break sớm)
        # Trả về working_sol (là kết quả của bước thành công cuối cùng)
        return working_sol

    # ==========================================================================
    # GYM INTERFACE
    # ==========================================================================

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
            
        print(f">>> [Env] Resetting PPO Environment (Seed: {seed})...")
        
        initial_schedule = compute_initial_solution(self.problem_instance, self.random_state)
        sim_seed = self.random_state.randint(0, 1000000)
        self.initial_solution = cvrpEnv(initial_schedule, self.problem_instance, seed=sim_seed)
        
        self.initial_solution = cleanup_inter_factory_routes(self.initial_solution)
        self.initial_solution = optimize_all_start_times(self.initial_solution)
        
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)
        
        self.history_matrix = {}
        update_history_matrix(self.history_matrix, self.current_solution)
        
        metrics = self.initial_solution.objective()
        self.initial_objective = metrics[0]
        self.best_objective = metrics[0]
        
        self.best_time_penalty = metrics[1]
        self.best_wait_time = metrics[2]
        self.best_capacity_penalty = metrics[3]
        
        count = self._count_customers(self.initial_solution)
        print(f"    -> Initial Objective: {self.initial_objective:.2f} | Farms: {count}")

        self.stag_count = 0
        self.current_iteration = 0
        self.last_actions = []
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Trả về trạng thái sạch (Pure Python Floats -> Numpy Array)."""
        metrics = self.current_solution.objective()
        
        def clean(val):
            try:
                if val is None: return 0.0
                if np.isnan(val): return 0.0
                if np.isinf(val): return 999999.0
                return float(val) 
            except:
                return 0.0

        current_obj = clean(metrics[0])
        time_penalty = clean(metrics[1]) if len(metrics) > 1 else 0.0
        wait_time = clean(metrics[2]) if len(metrics) > 2 else 0.0
        cap_penalty = clean(metrics[3]) if len(metrics) > 3 else 0.0
        
        epsilon = 1e-6
        iter_curr = clean(self.current_iteration)
        iter_max = clean(self.max_iterations)
        
        progress = iter_curr / (iter_max + epsilon)
        obj_init = clean(self.initial_objective)
        obj_best = clean(self.best_objective)
        current_temp = (obj_init * 0.05) * (1.0 - progress)
        stag = clean(self.stag_count)
        
        len_curr = float(len(self.current_solution.schedule))
        len_init = float(len(self.initial_solution.schedule)) if self.initial_solution else 1.0

        raw_state = [
            (current_obj - obj_best) / (obj_best + epsilon),
            stag / ((iter_max / 10.0) + epsilon),
            progress,
            current_temp / (obj_init + epsilon),
            current_obj / (obj_init + epsilon),
            time_penalty / (current_obj + epsilon),
            cap_penalty / (current_obj + epsilon),
            wait_time / (current_obj + epsilon),
            len_curr / (len_init + epsilon)
        ]
        
        return np.array(raw_state, dtype=np.float32)

    def step(self, action):
        op_index = int(action)
        self.current_iteration += 1
        
        self.last_actions.append(op_index)
        if len(self.last_actions) > 5: self.last_actions.pop(0)
        
        spam_penalty = 0
        if len(self.last_actions) >= 3 and all(x == op_index for x in self.last_actions[-3:]):
            spam_penalty = -5.0 

        op_data = self.macro_ops[op_index]
        seq_len = len(op_data['sequence_indices']) 

        objective_before = self.current_solution.objective()[0]
        
        # 1. Thực thi (Đã có logic Rollback từng bước)
        new_solution = self._execute_macro_op(op_index, self.current_solution)
        update_history_matrix(self.history_matrix, new_solution)
        
        final_results = new_solution.objective()
        objective_after = final_results[0]
        
        # 2. Tính Reward
        raw_improvement = (objective_before - objective_after) / (objective_before + 1e-6)
        clipped_improvement = max(-0.5, min(raw_improvement, 1.0))
        reward = clipped_improvement * 10 
        
        # 3. Acceptance (SA)
        accepted = False
        is_new_best = False
        
        if objective_after < objective_before:
            accepted = True
        else:
            progress = self.current_iteration / self.max_iterations
            current_temp = (self.initial_objective * 0.05) * (1 - progress)
            current_temp = max(current_temp, 1e-6)
            diff = objective_after - objective_before
            if diff > self.initial_objective * 0.5: probability = 0
            else: probability = math.exp(-diff / current_temp)
            if self.random_state.rand() < probability: accepted = True

        # 4. Cập nhật & Thưởng/Phạt
        if accepted:
            self.current_solution = new_solution
            
            if seq_len > 1:
                complexity_bonus = (seq_len - 1) * 3.0 
                reward += complexity_bonus

            if objective_after < self.best_objective:
                self.best_objective = objective_after
                self.best_solution = copy.deepcopy(new_solution)
                is_new_best = True
                
                self.best_time_penalty = final_results[1]
                self.best_wait_time = final_results[2]
                self.best_capacity_penalty = final_results[3]
                
                print(f"🎉 New Best (Op #{op_index}): {self.best_objective:.2f}")

        if is_new_best:
            self.stag_count = 0
            reward += 20.0 
        else:
            self.stag_count += 1
            if not accepted: reward -= 0.5 
            elif raw_improvement <= 0: reward -= 0.1

        reward += spam_penalty
        reward = max(reward, -10.0) 

        done = self.current_iteration >= self.max_iterations
        info = {'best_objective': self.best_objective}

        return self._get_state(), reward, done, False, info