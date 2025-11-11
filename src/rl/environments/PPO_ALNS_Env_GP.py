import gym
from gym import spaces
import numpy as np
import copy
import math
import random
from deap import base, creator, tools

# ==============================================================================
# LỚP GPSequence (Từ mã nguồn của bạn)
# ==============================================================================

class GPSequence:
    def __init__(self, pairs, fitness_fn, ngen=10, pop_size=10):
        self.pairs = pairs 
        self.fitness_fn = fitness_fn
        self.ngen = ngen
        self.pop_size = pop_size

        if not hasattr(creator, "FitnessMin"): # Sửa thành FitnessMin vì objective càng thấp càng tốt
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.sample, range(len(self.pairs)), len(self.pairs))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        
        def eval_func(individual):
            sequence = [self.pairs[i] for i in individual]
            return (self.fitness_fn(sequence),)
        self.toolbox.register("evaluate", eval_func)

    def run(self):
        pop = self.toolbox.population(n=self.pop_size)
        
        # Thuật toán di truyền đơn giản (bạn có thể cải thiện phần này)
        for gen in range(self.ngen):
            offspring = tools.selTournament(pop, len(pop), tournsize=3)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]
        best_seq = [self.pairs[i] for i in best_ind]
        return best_seq

# ==============================================================================
# PHẦN IMPORT MODULE ALNS CỦA BẠN
# ==============================================================================
try:
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, worst_removal, shaw_removal, time_worst_removal
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret_2_insertion, regret_3_insertion, regret_4_insertion
    print("✅ Đã import thành công các module ALNS của bạn!")
except ImportError:
    print("❌ CẢNH BÁO: Không tìm thấy module ALNS. Sử dụng các lớp giả (dummy classes).")
    class cvrpEnv: pass
    def compute_initial_solution(problem, rand): return cvrpEnv()
    def random_removal(c, r, **k): return c, []
    def worst_removal(c, r, **k): return c, []
    def shaw_removal(c, r, **k): return c, []
    def time_worst_removal(c, r, **k): return c, []
    def best_insertion(c, r, **k): return c, []
    def regret_2_insertion(c, r, **k): return c, []
    def regret_3_insertion(c, r, **k): return c, []
    def regret_4_insertion(c, r, **k): return c, []

# ==============================================================================
# MÔI TRƯỜNG PPO TÍCH HỢP GP
# ==============================================================================
class PPO_ALNS_Env_GP(gym.Env):
    def __init__(self, problem_instance, max_iterations=125, buffer_size=8, **kwargs): # max_iter = 1000 / 8
        super(PPO_ALNS_Env_GP, self).__init__()
        
        self.problem_instance = problem_instance
        self.random_state = np.random.RandomState()
        self.destroy_operators = [random_removal, worst_removal, shaw_removal, time_worst_removal]
        self.repair_operators = [best_insertion, regret_2_insertion, regret_3_insertion, regret_4_insertion]
        
        self.action_space = spaces.MultiDiscrete([len(self.destroy_operators), len(self.repair_operators)])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.max_iterations = max_iterations # Số lần GP được gọi
        self.buffer_size = buffer_size
        self.action_buffer = []

        # Các biến theo dõi khác
        self.current_solution = None
        self.initial_solution = None
        self.best_solution = None
        self.initial_objective = float('inf')
        self.best_objective = float('inf')
        self.stag_count = 0
        self.current_iteration = 0
        self.start_temperature = kwargs.get('start_temperature', 1) # Không dùng SA nên temp chỉ để làm feature

    def _fitness_function_for_gp(self, sequence):
        """
        Hàm mô phỏng thực thi một chuỗi các toán tử và trả về objective.
        Càng thấp càng tốt.
        """
        temp_solution = copy.deepcopy(self.current_solution)
        for destroy_op, repair_op in sequence:
            destroyed, unvisited = destroy_op(temp_solution, self.random_state)
            if unvisited:
                repaired, _ = repair_op(destroyed, self.random_state, unvisited_customers=unvisited)
                temp_solution = repaired
        
        return temp_solution.objective()[0]

    def reset(self):
        print(">>> Môi trường được reset. Tạo lời giải ban đầu mới...")
        initial_schedule = compute_initial_solution(self.problem_instance, self.random_state)
        self.initial_solution = cvrpEnv(initial_schedule, self.problem_instance, seed=None)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)
        
        initial_results = self.initial_solution.objective()
        self.initial_objective = initial_results[0]
        self.best_objective = initial_results[0]
        
        self.stag_count = 0
        self.current_iteration = 0
        self.action_buffer = []
        
        return self._get_state()

    def _get_state(self):
        # (Giữ nguyên hàm _get_state đã viết ở lần trước)
        current_metrics = self.current_solution.objective()
        current_obj, time_penalty, wait_time, cap_penalty = current_metrics[:4]
        epsilon = 1e-6

        state = np.array([
            (current_obj - self.best_objective) / (self.best_objective + epsilon),
            self.stag_count / ((self.max_iterations / 10) + epsilon),
            self.current_iteration / self.max_iterations,
            (self.start_temperature * (0.999 ** (self.current_iteration * self.buffer_size))) / self.start_temperature,
            current_obj / (self.initial_objective + epsilon),
            time_penalty / (current_obj + epsilon),
            cap_penalty / (current_obj + epsilon),
            wait_time / (current_obj + epsilon),
            len(self.current_solution.schedule) / (len(self.initial_solution.schedule) + epsilon)
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        destroy_idx, repair_idx = action
        destroy_op = self.destroy_operators[destroy_idx]
        repair_op = self.repair_operators[repair_idx]
        self.action_buffer.append((destroy_op, repair_op))

        # --- KIỂM TRA BỘ ĐỆM ---
        if len(self.action_buffer) < self.buffer_size:
            # Nếu bộ đệm chưa đầy, không làm gì cả, chờ bước tiếp theo
            # Trả về reward = 0 và state không đổi
            done = self.current_iteration >= self.max_iterations
            return self._get_state(), 0, done, {'best_objective': self.best_objective}

        # --- BỘ ĐỆM ĐÃ ĐẦY -> GỌI GP VÀ THỰC THI ---
        self.current_iteration += 1
        print(f"\n--- Buffer đầy. Iter {self.current_iteration}/{self.max_iterations}. Gọi GPSequence... ---")

        # 1. Gọi GP để tìm thứ tự tốt nhất
        gp = GPSequence(self.action_buffer, self._fitness_function_for_gp)
        best_sequence = gp.run()
        
        # 2. Thực thi chuỗi tối ưu
        objective_before = self.current_solution.objective()[0]
        
        new_solution = copy.deepcopy(self.current_solution)
        for destroy_op, repair_op in best_sequence:
            destroyed, unvisited = destroy_op(new_solution, self.random_state)
            if unvisited:
                repaired, _ = repair_op(destroyed, self.random_state, unvisited_customers=unvisited)
                new_solution = repaired
        
        objective_after = new_solution.objective()[0]
        
        # 3. Tính toán Reward (cho cả chuỗi 8 bước)
        reward = 0.0
        epsilon = 1e-6
        improvement = (objective_before - objective_after) / (objective_before + epsilon)
        reward += improvement * 10
        
        if objective_after < self.best_objective:
            reward += 10.0
            print(f"🎉 New best found! Obj: {self.best_objective:.2f} -> {objective_after:.2f}")
            self.best_objective = objective_after
            self.best_solution = copy.deepcopy(new_solution)
            self.stag_count = 0
        else:
            self.stag_count += 1
        
        if abs(improvement) < epsilon:
            reward -= 0.1

        # 4. Cập nhật trạng thái và reset buffer
        self.current_solution = new_solution
        self.action_buffer = [] # Quan trọng: Xóa buffer để bắt đầu lại

        # 5. Trả về kết quả
        done = self.current_iteration >= self.max_iterations
        next_state = self._get_state()
        info = {'best_objective': self.best_objective}

        return next_state, reward, done, info

    def render(self, mode='human'):
        print(
            f"Iter (GP calls): {self.current_iteration}/{self.max_iterations} | "
            f"Buffer size: {len(self.action_buffer)}/{self.buffer_size} | "
            f"Current Obj: {self.current_solution.objective()[0]:.2f} | "
            f"Best Obj: {self.best_objective:.2f} | "
            f"Stag: {self.stag_count}"
        )