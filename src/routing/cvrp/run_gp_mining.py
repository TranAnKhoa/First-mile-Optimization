import os
import sys
import json
import copy
import random
import numpy as np
from deap import base, creator, tools, gp, algorithms

# =========================================================================
# IMPORT SYSTEM
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, '..', '..'))

from routing.cvrp.alns_cvrp import cvrp_helper_functions
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp.utils import optimize_all_start_times, cleanup_inter_factory_routes 

# Import Operators
from routing.cvrp.alns_cvrp.destroy_operators import (
    random_removal, worst_removal_alpha_0, worst_removal_bigM,
    worst_removal_adaptive, time_worst_removal, shaw_spatial,
    shaw_hybrid, shaw_temporal, shaw_structural, trip_removal,
    historical_removal,update_solution_state_after_destroy
)
from routing.cvrp.alns_cvrp.repair_operators import (
    best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle,
    regret_3_position, regret_3_trip, regret_3_vehicle,
    regret_4_position, regret_4_trip, regret_4_vehicle
)

# =========================================================================
# CONFIGURATION
# =========================================================================
INSTANCE_FILE = r'K:\Data Science\SOS lab\Project Code\output_data\CEL_instance.pkl'
OUTPUT_FILE   = 'macro_gp_tree_final.json'

TEST_SEEDS = [42, 101, 2024, 777, 999]
POPULATION_SIZE = 100  # 100 cây
GENERATIONS = 50       # 50 đời
MAX_DEPTH = 4          # Độ sâu cây tối đa (tránh cây quá bự)

# Kho tham số
DESTROY_OPS = [random_removal, worst_removal_alpha_0, worst_removal_bigM, worst_removal_adaptive, time_worst_removal, shaw_spatial, shaw_hybrid, shaw_temporal, shaw_structural, trip_removal, historical_removal]
REPAIR_OPS = [best_insertion, regret_2_position, regret_2_trip, regret_2_vehicle, regret_3_position, regret_3_trip, regret_3_vehicle, regret_4_position, regret_4_trip, regret_4_vehicle]
REMOVE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]

def get_op_name(op):
    if hasattr(op, '__name__'): return op.__name__
    if hasattr(op, 'func'): return op.func.__name__
    return str(op)

# =========================================================================
# 1. LOGIC SỬA CHỮA (SAFETY NET)
# =========================================================================
def sanitize_and_repair(solution, rnd_state):
    """Chạy khi chuỗi đã đi được > 2 bước"""
    solution = cleanup_inter_factory_routes(solution)
    try:
        _, time_pen, _, cap_pen = solution.objective()
    except: return solution

    # Nếu không vi phạm thì thôi
    if time_pen == 0 and cap_pen == 0:
        return solution

    # Nếu vi phạm: Đá bớt khách gây lỗi time/cap
    destroy_op = time_worst_removal
    op_kwargs = {'remove_fraction': 0.10, 'history_matrix': {}}
    
    destroyed, unvisited = destroy_op(solution, rnd_state, **op_kwargs)
    destroyed = update_solution_state_after_destroy(destroyed)

    if unvisited:
        farms = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
        if farms:
            # Dùng Regret-3 (cẩn thận) để nhét lại
            repaired, _ = regret_3_trip(destroyed, rnd_state, unvisited_customers=farms)
            return repaired
    return destroyed

# =========================================================================
# 2. TRÌNH BIÊN DỊCH CÂY (TREE INTERPRETER)
# =========================================================================
class TreeExecutor:
    def __init__(self, rnd_state):
        self.rnd = rnd_state
        self.step_count = 0 # Đếm số bước thực tế đã chạy
        self.execution_log = [] # Lưu lại vết (trace) để debug/in ấn

    def run_op(self, solution, d_idx, p_idx, r_idx):
        """Thực thi 1 nút lá (Operator)"""
        
        # --- SAFETY NET LOGIC ---
        # Nếu đã chạy được hơn 2 bước, kích hoạt sửa lỗi trước khi chạy bước tiếp theo
        if self.step_count >= 2:
            solution = sanitize_and_repair(solution, self.rnd)
            
        self.step_count += 1
        
        d_op = DESTROY_OPS[d_idx]
        r_op = REPAIR_OPS[r_idx]
        frac = REMOVE_LEVELS[p_idx]
        
        # Lưu log để biết cây đã chọn chạy cái gì
        self.execution_log.append(f"[{get_op_name(d_op)}({frac}) -> {get_op_name(r_op)}]")

        try:
            solution = cleanup_inter_factory_routes(solution)
            destroyed, unvisited = d_op(solution, self.rnd, remove_fraction=frac, history_matrix={})
            destroyed = update_solution_state_after_destroy(destroyed)

            if unvisited:
                farms = [c for c in unvisited if not str(c).startswith("TRANSFER_")]
                if farms:
                    repaired, _ = r_op(destroyed, self.rnd, unvisited_customers=farms)
                    return repaired
            return destroyed
        except:
            return solution # Nếu lỗi thì trả về nguyên vẹn

    def execute(self, node, solution):
        """Hàm đệ quy duyệt cây"""
        # Nút Lá: (d, p, r)
        if isinstance(node, tuple) and len(node) == 3 and isinstance(node[0], int):
            return self.run_op(solution, node[0], node[1], node[2])

        # Nút SEQ2: Chạy nhánh trái, rồi lấy kết quả chạy nhánh phải
        if node[0] == "SEQ2":
            _, left, right = node
            sol_after_left = self.execute(left, solution)
            return self.execute(right, sol_after_left)

        # Nút IF: Kiểm tra điều kiện rồi rẽ nhánh
        if node[0] == "IF":
            _, cond_type, true_branch, false_branch = node
            
            # Check điều kiện
            try:
                _, t_pen, _, c_pen = solution.objective()
            except: t_pen, c_pen = 0, 0
            
            # Cond 0: Check Capacity, Cond 1: Check Time
            is_violated = (c_pen > 0) if cond_type == 0 else (t_pen > 0)
            
            if is_violated:
                return self.execute(true_branch, solution)
            else:
                return self.execute(false_branch, solution)
        
        return solution

# =========================================================================
# 3. GP SETUP (DEAP)
# =========================================================================
pset = gp.PrimitiveSet("MAIN", 0)

# Hàm logic: Tên hiển thị sẽ là SEQ, IF...
pset.addPrimitive(lambda a, b: ("SEQ2", a, b), 2, name="SEQ")
pset.addPrimitive(lambda c, a, b: ("IF", c, a, b), 3, name="IF")

# Terminal điều kiện (0: Cap, 1: Time)
pset.addTerminal(0, name="CheckCap")
pset.addTerminal(1, name="CheckTime")

# Terminal hành động: Thay vì add 550 cái, ta add từng thành phần rồi để GP tự ghép (Advance hơn)
# Nhưng để giữ code bạn chạy được ngay, ta giữ cách add tuple (D, P, R)
# Mẹo: Chỉ add những combo phổ biến hoặc add full nếu máy chịu nổi.
# Ở đây mình add full (550) nhưng dùng vòng lặp gọn.
count = 0
for d in range(len(DESTROY_OPS)):
    for p in range(len(REMOVE_LEVELS)):
        for r in range(len(REPAIR_OPS)):
            # Tên ngắn gọn cho node: OP_id
            pset.addTerminal((d, p, r), name=f"OP_{count}")
            count += 1

# =========================================================================
# 4. EVALUATION
# =========================================================================
def evaluate_tree(individual, base_solutions):
    # Compile cây thành cấu trúc tuple lồng nhau
    tree_struct = gp.compile(individual, pset)
    
    feasible = 0
    total_cost = 0
    
    for i, base_sol in enumerate(base_solutions):
        rnd = np.random.RandomState(TEST_SEEDS[i])
        
        # Tạo bộ thực thi mới cho mỗi lần chạy
        executor = TreeExecutor(rnd)
        
        try:
            # Chạy cây
            final_sol = executor.execute(tree_struct, copy.deepcopy(base_sol))
            
            # Tối ưu nhẹ cuối cùng
            final_sol = optimize_all_start_times(final_sol)
            cost, tpen, _, cpen = final_sol.objective()
            
            if tpen == 0 and cpen == 0:
                feasible += 1
                total_cost += cost
        except:
            continue

    if feasible == 0:
        return (0, 1e9) # Phạt nặng

    return (feasible, total_cost / feasible)

# =========================================================================
# 5. MAIN RUN
# =========================================================================
def run_mining():
    print("🚀 GP TREE MINING START (Tree-based Hyper-heuristic)")
    
    # Init Data
    (_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)
    base_solutions = []
    for seed in TEST_SEEDS:
        rnd = np.random.RandomState(seed)
        init = compute_initial_solution(problem_obj, rnd)
        sol = cvrpEnv(init, problem_obj, seed=seed)
        sol = cleanup_inter_factory_routes(sol)
        base_solutions.append(sol)
    print("✅ Init Data Done.")

    # Setup GA/GP
    if hasattr(creator, "FitnessMulti"): del creator.FitnessMulti
    if hasattr(creator, "Individual"): del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(10.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    # Tạo cây ngẫu nhiên độ sâu 1-3
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", lambda ind: evaluate_tree(ind, base_solutions))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

    # Limit depth để tránh cây quá to (Bloat)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

    # Run
    import operator # Cần cho staticLimit
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Hall of Fame: Lưu các cá thể tốt nhất
    hof = tools.HallOfFame(30)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("feas", lambda vals: np.max([v[0] for v in vals]))
    stats.register("cost", lambda vals: np.min([v[1] for v in vals if v[1] < 1e9]))

    print("\n--- Start Evolution ---")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # Output Results
    print("\n🏆 TOP 30 GP TREES:")
    results = []
    for i, ind in enumerate(hof):
        tree_struct = gp.compile(ind, pset)
        
        # Chạy thử 1 lần để lấy log hành động (để in ra xem nó làm gì)
        rnd_debug = np.random.RandomState(42)
        executor = TreeExecutor(rnd_debug)
        executor.execute(tree_struct, copy.deepcopy(base_solutions[0]))
        trace = executor.execution_log
        
        print(f"   #{i+1} Feas:{ind.fitness.values[0]} | Cost:{ind.fitness.values[1]:.0f}")
        print(f"      Tree: {str(ind)}")
        print(f"      Trace Example: {' -> '.join(trace[:4])}...") # In 4 bước đầu

        results.append({
            "rank": i+1,
            "feasible": ind.fitness.values[0],
            "cost": ind.fitness.values[1],
            "tree_str": str(ind),
            "trace_example": trace
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Saved to", OUTPUT_FILE)

if __name__ == "__main__":
    run_mining()