import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import time
import gymnasium as gym
import random
from alns import ALNS
import numpy as np
import numpy.random as rnd
from pathlib import Path

from routing.cvrp.alns_cvrp import cvrp_helper_functions
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.destroy_operators import *
from routing.cvrp.alns_cvrp.repair_operators import *
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution

from .gp_sequence import GPSequence

class cvrpAlnsEnv_LSA1(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config["environment"]
        self.rnd_state = rnd.RandomState()  
    
        self.max_temperature = 5
        self.temperature = 5

        base_path = Path(__file__).resolve().parents[2]
        self.instance_file = str(base_path.joinpath(self.config["instance_file"]))
        
        self.instances = self.config["instance_nr"]
        self.instance = None
        self.best_routes = []

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        self.penalty_ratio = 0
        self.is_infeasible = 0

        self.reward = 0
        self.done = False
        self.episode = 0
        self.iteration = 0
        self.max_iterations = self.config["iterations"]

        # *** QUAY LẠI action_space CHỈ CHO MỘT HÀNH ĐỘNG DUY NHẤT ***
        self_action_sub_space = [13, 6, 10, 100] # [destroy_operator, repair_operator, factor, temperature]
        self.action_space = gym.spaces.MultiDiscrete(self_action_sub_space)
        
        # Observation space giữ nguyên
        self.observation_space = gym.spaces.Box(shape=(10,), low=0, high=100, dtype=np.float64)

        # Buffer để lưu trữ 8 hành động từ PPO
        self.actions_buffer = []
        self.num_actions_to_collect = 8 # Số lượng hành động cần thu thập trước khi chạy ALNS + GP
        
    #! Tạo ra state để agent biết được tiến độ học có tốt hay không    
    def make_observation(self):
        is_current_best = 0
        current_obj, _ = self.current_solution.objective()
        best_obj, _ = self.best_solution.objective()
        if current_obj == best_obj:
            is_current_best = 1
            
        state = np.array(
            [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
            self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved,
            self.penalty_ratio,  # <-- Feature mới
            self.is_infeasible], # <-- Feature mới
            dtype=np.float64).squeeze()

        return state

    def reset(self, seed=None, options=None):
        SEED = random.randint(0, 10000) 
        self.instance = random.choice(self.instances)

        # --- SỬA ĐỔI 1: Nhận thêm 'data' gốc ---
        (nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, 
        demands_data, customer_st, customer_tw, depot_tw, 
        problem_instance_data) = cvrp_helper_functions.read_input_cvrp(
            self.instance_file, self.instance
        )

        random_state = rnd.RandomState(SEED)

        initial_schedule = compute_initial_solution(problem_instance_data, random_state)

        # Khởi tạo env với lịch trình
        state = cvrpEnv(initial_schedule, problem_instance_data, SEED)

        self.initial_solution = state
        self.current_solution = copy.deepcopy(self.initial_solution) 
        self.best_solution = copy.deepcopy(self.initial_solution)

        self.dr_alns = ALNS(random_state)
        
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)
        self.dr_alns.add_destroy_operator(sequence_removal) 
        self.dr_alns.add_destroy_operator(shaw_distance_removal)
        self.dr_alns.add_destroy_operator(path_removal)
        self.dr_alns.add_destroy_operator(history_biased_removal)
       
        self.dr_alns.add_destroy_operator(worst_removal)
        self.dr_alns.add_destroy_operator(cluster_removal)
        self.dr_alns.add_destroy_operator(cost_outlier_removal)
        self.dr_alns.add_destroy_operator(two_route_exchange)
        self.dr_alns.add_destroy_operator(depot_shift_removal)
        self.dr_alns.add_destroy_operator(load_balancing_removal)
        
        self_dr_alns_repair_operators = [regret2_insertion, regret3_insertion, best_insertion, cheapest_feasible_insertion, random_feasible_insertion]
        for op in self_dr_alns_repair_operators:
            self.dr_alns.add_repair_operator(op)
        
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0  

        self.iteration = 0 # Iteration cho ALNS "bước lớn"
        self.reward = 0 # Tổng reward cho "bước lớn"
        self.done = False

        self.actions_buffer = [] # Reset buffer khi reset môi trường

        obs = self.make_observation()
        info = {}
        return obs, info

    
    def step(self, action_from_ppo_single): 
         # action_from_ppo_single là MỘT hành động [d_idx, r_idx, removal_level, temp_level] từ PPO
        
        # ... (Phần đầu giữ nguyên) ...
        # Thêm hành động vào buffer
        self.actions_buffer.append(action_from_ppo_single)
        
        if len(self.actions_buffer) < self.num_actions_to_collect:
            return self.make_observation(), 0, False, False, {}

        # ... (Phần xử lý buffer và tính toán tham số giữ nguyên) ...
        # --- Đã thu thập đủ 8 hành động, bây giờ xử lý chúng ---
        ppo_actions_list_for_gp = []
        removal_factors_raw = []
        temp_factors_raw = []

        for ppo_action in self.actions_buffer:
            d_idx, r_idx, removal_level, temp_level = ppo_action
            ppo_actions_list_for_gp.append((int(d_idx), int(r_idx)))
            removal_factors_raw.append(removal_level)
            temp_factors_raw.append(temp_level)

        avg_removal_factor_level = np.mean(removal_factors_raw)
        avg_temp_factor_level = np.mean(temp_factors_raw)

        removal_factor = (avg_removal_factor_level + 1) / 10.0
        temp_factor = (avg_temp_factor_level + 1) / 100.0
        
        self.temperature = self.max_temperature * (1 - temp_factor)
        self.temperature = max(0.1, self.temperature)

        print("PPO collected 8 distinct pairs for GP:", ppo_actions_list_for_gp, flush=True)
        print(f"Average Removal Factor: {removal_factor:.2f}, Average Temperature Factor: {temp_factor:.2f}", flush=True)

        current = self.current_solution
        best = self.best_solution

        def fitness_fn(sequence):
            temp_sol = copy.deepcopy(current)
            
            nr_nodes_to_remove_base = max(1, round(removal_factor * temp_sol.nb_customers))

            for d_idx, r_idx in sequence:
                d_name, d_op = self.dr_alns.destroy_operators[d_idx]
                r_name, r_op = self.dr_alns.repair_operators[r_idx]

                nr_nodes_to_remove = max(1, round(nr_nodes_to_remove_base))
                
                destroyed = d_op(temp_sol, self.rnd_state, nr_nodes_to_remove)
                temp_sol = r_op(destroyed, self.rnd_state)

            ### SỬA ĐỔI 1: Lấy giá trị đầu tiên từ hàm objective ###
            obj, _ = temp_sol.objective()
            return -obj

        gp = GPSequence(ppo_actions_list_for_gp, fitness_fn, ngen=5, pop_size=2)
        best_sequence = gp.run()
        
        if isinstance(best_sequence, tuple):
            best_sequence = [best_sequence]
        
        candidate = copy.deepcopy(current)
        oper_data = []
            
        for (d_idx, r_idx) in best_sequence:
            d_name, d_operator = self.dr_alns.destroy_operators[d_idx]
            r_name, r_operator = self.dr_alns.repair_operators[r_idx]
            oper_data.append(f"({d_name},{r_name})")
            nr_nodes_to_remove = max(1, round(removal_factor * candidate.nb_customers))
            destroyed = d_operator(candidate, self.rnd_state, nr_nodes_to_remove)
            candidate = r_operator(destroyed, self.rnd_state)
                
        ### SỬA ĐỔI 2: Tính toán objective MỘT LẦN và sử dụng lại ###
        best_obj, _ = best.objective()
        current_obj, _ = current.objective()
        candidate_obj, _ = candidate.objective()
        
        new_best, new_current = self.consider_candidate(best, current, candidate, best_obj, current_obj, candidate_obj)
        
        self.current_updated = 0
        self.current_improved = 0
        current_reward_for_this_meta_step = 0

        ### SỬA ĐỔI 3: Sử dụng các giá trị obj đã được tính toán ở trên ###
        if new_best is not None: # Điều kiện so sánh đã nằm trong consider_candidate
            best = new_best
            current = new_best
            current_reward_for_this_meta_step += 5
            self.stagcount = 0
            self.current_updated = 1
            self.current_improved = 1
        else: # new_best is None, chỉ xét new_current
            new_current_obj, _ = new_current.objective()
            if new_current_obj < current_obj:
                current = new_current
                current_reward_for_this_meta_step += 1
                self.stagcount = 0
                self.current_updated = 1
                self.current_improved = 1
            else: # Chấp nhận giải pháp tệ hơn
                current = new_current
                self.current_updated = 1
                self.stagcount += 1

        self.reward = current_reward_for_this_meta_step

        ### SỬA ĐỔI 4: Lấy giá trị cuối cùng để in ra và tính state ###
        final_current_obj, current_penalty = current.objective()
        final_best_obj, _ = best.objective()

        print(f"Iteration {self.iteration}: {' -> '.join(oper_data)}", flush=True)
        print(f"After sequence - Current objective: {final_current_obj}, Best: {final_best_obj}", flush=True)

        self.current_solution = current
        self.best_solution = best
        
        ### SỬA ĐỔI 5: Tính toán tất cả các state dựa trên giá trị cuối cùng ###
        # Tính các state mới cho observation
        if final_current_obj > 0:
            self.penalty_ratio = (current_penalty / final_current_obj) * 100
        else:
            self.penalty_ratio = 0
        self.is_infeasible = 1 if current_penalty > 0 else 0

        # Tính các state cũ dựa trên giá trị đã có
        if final_current_obj < final_best_obj:
            self.improvement = 1
        else:
            self.improvement = 0

        if final_best_obj != 0:
            self.cost_difference_from_best = ((final_current_obj - final_best_obj) / final_best_obj) * 100
        else:
            self.cost_difference_from_best = 0

        # Cập nhật graph với giá trị obj chính xác
        # self.current_solution.graph = cvrp_helper_functions.update_neighbor_graph(...) # Phần này có thể bỏ nếu không dùng
        # self.best_solution.graph = cvrp_helper_functions.update_neighbor_graph(...)

        self.actions_buffer = []
        self.iteration += 1

        if self.iteration >= self.max_iterations:
            self.done = True
            # ... (phần ghi file giữ nguyên) ...

        terminated = self.done
        truncated = False
        
        return self.make_observation(), self.reward, terminated, truncated, {}


    def consider_candidate(self, best, curr, cand, best_obj, curr_obj, cand_obj):
        # Sử dụng các giá trị obj đã được truyền vào, không gọi lại .objective()
        if cand_obj < best_obj:
            return cand, cand

        probability = np.exp((curr_obj - cand_obj) / self.temperature)

        if probability >= self.rnd_state.random():
            return None, cand
        else:
            return None, curr

    def run(self, model, episodes=1):
        """
        Use a trained model to select actions.
        In this mode, PPO is called multiple times per "meta-step" of the environment.
        """
        all_sequences_of_actions = []

        try:
            for episode in range(episodes):
                self.done = False
                state, _ = self.reset()
                
                while not self.done:
                    # Trong hàm run này, chúng ta sẽ lặp 8 lần gọi model.predict()
                    # để thu thập đủ 8 hành động cho một "bước lớn" của ALNS.
                    # Mỗi lần gọi model.predict() sẽ dẫn đến một lần gọi self.step()
                    # của môi trường, và env.step() sẽ tự quản lý buffer.
                    
                    # Chúng ta không cần một vòng lặp 8 lần ở đây nữa,
                    # vì self.step() bây giờ đã xử lý việc đó nội bộ bằng buffer.
                    # Chỉ cần gọi model.predict() và self.step() một lần cho mỗi "sub-step".
                    
                    # Cần lưu ý rằng 'state' ở đây là trạng thái sau mỗi 'sub-step'
                    # (khi buffer chưa đầy) hoặc trạng thái sau 'meta-step' (khi buffer đầy).
                    
                    # PPO.predict() sẽ trả về MỘT hành động raw (4 phần tử)
                    ppo_single_action, _ = model.predict(state, deterministic=True)
                    
                    # Truyền hành động này vào env.step()
                    state, reward, terminated, truncated, info = self.step(ppo_single_action) 

                    # self.done sẽ chỉ là True khi đã hết meta_iterations
                    self.done = terminated or truncated
                    
                    # Cập nhật tổng reward của episode (nếu bạn muốn)
                    # Hoặc reward từ self.step() đã là reward của meta-step.
                    
                    print(f"Episode {episode}, Current Buffer Size: {len(self.actions_buffer)}, State: {state}, Reward: {reward}, Done: {self.done}", flush=True)
                    
        except Exception as e:
            print(f"An error occurred during run: {e}", flush=True)
        
        return all_sequences_of_actions

    def run_time_limit(self, model, episodes=1):
        try:
            for episode in range(episodes):
                start_time = time.time()
                time_done = False
                state, _ = self.reset()
                while not time_done:
                    action, _states = model.predict(state, deterministic=True)
                    state, reward, terminated, truncated, _ = self.step(action)
                    current_time = time.time() - start_time
                    print(current_time)
                    if current_time > 30 or terminated or truncated:
                        time_done = True
        except KeyboardInterrupt:
            pass

    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(2):
            self.done = False
            state, _ = self.reset()
            print("start episode: ", episode, " with start state: ", state)
            while not self.done:
                action = self.action_space.sample() # action_space.sample() trả về một mảng 4 chiều
                state, reward, terminated, truncated, _ = self.step(action)
                self.done = terminated or truncated
                print(
                    "step {}, action: {}, New state: {}, Reward: {:2.3f}".format(
                        self.iteration, action, state, reward
                    )
                )


if __name__ == "__main__":
    from rl.baselines import get_parameters, Trainer

    env = cvrpAlnsEnv_LSA1(get_parameters("cvrpAlnsEnv_LSA1"))
    
    # print("Sampling random actions...")
    # env.sample()

    print('Start training')
    model = Trainer("cvrpAlnsEnv_LSA1", "models").create_model()
    model.learn(total_timesteps=100000)
    print("Training done")
    input("Run trained model (Enter)")
    env.run(model)