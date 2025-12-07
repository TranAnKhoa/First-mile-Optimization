import random
from deap import base, creator, tools
import copy

class GP_Matching:
    """
    Genetic Algorithm để tìm cách GHÉP CẶP (Matching)
    và SẮP XẾP THỨ TỰ (Sequencing) tốt nhất.
    Input: 2 danh sách riêng biệt: [d1, d2...] và [r1, r2...]
    Output: sequence tốt nhất của các cặp (d, r)
    """

    def __init__(self, destroy_ops, repair_ops, fitness_fn, ngen=10, pop_size=10):
        """
        destroy_ops: list các toán tử [d1, d2, ...] (ví dụ: 8 ops)
        repair_ops: list các toán tử [r1, r2, ...] (ví dụ: 8 ops)
        fitness_fn: hàm fitness(sequence) -> score
        """
        # --- THAY ĐỔI: Nhận 2 danh sách riêng biệt ---
        if len(destroy_ops) != len(repair_ops):
            raise ValueError("Số lượng toán tử Destroy và Repair phải bằng nhau để ghép cặp 1-1")
            
        self.destroy_ops = destroy_ops 
        self.repair_ops = repair_ops
        self.num_pairs = len(destroy_ops) # Ví dụ: 8
        # --- HẾT THAY ĐỔI ---

        self.fitness_fn = fitness_fn
        self.ngen = ngen
        self.pop_size = pop_size

        # Setup DEAP (Giữ nguyên như code của bạn)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # --- THAY ĐỔI: Cá thể là hoán vị của CHỈ SỐ REPAIR ---
        # (Giả sử self.num_pairs = 8, nó sẽ tạo hoán vị của [0, 1, 2, 3, 4, 5, 6, 7])
        self.toolbox.register("indices", random.sample, range(self.num_pairs), self.num_pairs) 
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        # --- HẾT THAY ĐỔI ---
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Operators Crossover và Mutation giữ nguyên, vì chúng vẫn làm việc trên hoán vị
        self.toolbox.register("mate", tools.cxOrdered) 
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2) 
        
        # --- THAY ĐỔI: Hàm Evaluate dịch individual thành sequence matching ---
        def eval_func(individual):
            # individual là list các index của repair_ops, ví dụ: [3, 0, 1, ..., 7]
            # Ta sẽ tạo sequence: [(D0, R3), (D1, R0), (D2, R1), ..., (D7, R7)]
            sequence = []
            for i in range(self.num_pairs):
                d_op = self.destroy_ops[i]
                r_idx = individual[i] # Đây là logic matching
                r_op = self.repair_ops[r_idx]
                sequence.append((d_op, r_op))
                
            return (self.fitness_fn(sequence),) # Trả về tuple
        
        self.toolbox.register("evaluate", eval_func)
        # --- HẾT THAY ĐỔI ---

    def run(self):
        # --- HÀM RUN NÀY GIỮ NGUYÊN LOGIC CỦA BẠN ---
        # (Elitism 2, Crossover 6, Mutation 2)
        
        pop = self.toolbox.population(n=self.pop_size)

        for gen in range(self.ngen):
            # Đánh giá fitness cho toàn bộ cá thể
            for ind in pop:
                # Chỉ đánh giá nếu fitness chưa hợp lệ
                if not ind.fitness.valid: 
                    ind.fitness.values = self.toolbox.evaluate(ind) 
            
            # 1. Chọn ra 2 cá thể tốt nhất (elitism)
            best_two = tools.selBest(pop, 2)

            # 2. Lấy phần còn lại để crossover + mutation
            # (Giữ nguyên logic chọn của bạn)
            remaining = tools.selBest(pop, len(pop))[2:]

            # --- crossover ---
            selected_for_crossover = remaining[:6]  # lấy 6 con
            offspring = []
            for i in range(0, len(selected_for_crossover), 2):
                if i+1 < len(selected_for_crossover):
                    p1, p2 = selected_for_crossover[i], selected_for_crossover[i+1]
                    c1, c2 = self.toolbox.clone(p1), self.toolbox.clone(p2)
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
                    offspring.extend([c1, c2])
            
            # --- mutation ---
            selected_for_mutation = remaining[6:8]  # 2 con
            mutated = []
            for ind in selected_for_mutation:
                mutant = self.toolbox.clone(ind)
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
                mutated.append(mutant)

            # 3. Tạo quần thể mới (đúng 10 cá thể)
            pop = best_two + offspring + mutated

        # Đánh giá fitness lần cuối cho quần thể cuối cùng
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        # --- THAY ĐỔI: Dựng lại best_seq từ best_ind theo logic matching ---
        best_ind = tools.selBest(pop, 1)[0]
        
        best_seq = []
        for i in range(self.num_pairs):
            d_op = self.destroy_ops[i]
            r_idx = best_ind[i] # Lấy index repair từ cá thể tốt nhất
            r_op = self.repair_ops[r_idx]
            best_seq.append((d_op, r_op))
            
        return best_seq
        # --- HẾT THAY ĐỔI ---