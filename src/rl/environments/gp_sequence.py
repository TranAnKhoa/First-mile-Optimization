import random
from deap import base, creator, tools

class GPSequence:
    """
    Genetic Programming để tìm thứ tự thực hiện các operator (destroy, repair)
    Input: danh sách cặp [(d1,r1), (d2,r2), ...] từ PPO
    Output: sequence tốt nhất
    """

    def __init__(self, pairs, fitness_fn, ngen=10, pop_size=10):
        """
        pairs: list các cặp [(d1,r1), (d2,r2), ...] từ PPO
        fitness_fn: hàm fitness(sequence) -> score
        """
        self.pairs = pairs 
        self.fitness_fn = fitness_fn
        self.ngen = ngen #số thế hệ --> chạy tiến hóa bao nhiêu vòng
        self.pop_size = pop_size #số lượng thế hệ trong mỗi thế hệ

        # Setup DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #fitness để maximize (score càng cao càng tốt)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax) #cá thể là 1 list (chứa các index của pairs)
        # individaul là 1 list các index của pairs, ví dụ pairs có 3 cặp thì individual có thể là [2,0,1] (thứ tự thực hiện các cặp)

        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.sample, range(len(self.pairs)), len(self.pairs)) #shuffle các index
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices) #tạo 1 cá thể từ list index
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) #tạo quần thể là list các cá thể
        # khởi tạo cá thể bằng cách lấy 1 permutation của các index
         # ví dụ pairs có 3 cặp thì individual có thể là [2,0,1] (thứ tự thực hiện các cặp)
         # random.sample(range(3), 3) có thể trả về [2,0,1], [1,2,0], [0,1,2], ..
        # Operators
        self.toolbox.register("mate", tools.cxOrdered) #crossover giữ nguyên thứ tự
         # ví dụ [0,1,2] x [2,0,1] có thể ra [0,2,1] (giữ nguyên thứ tự các phần tử)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2) #đổi chỗ 2 phần tử với xác suất 0.2
            # ví dụ [0,1,2] có thể mutate thành [1,0,2] hoặc [0,2,1], ...
        # Fitness function
        def eval_func(individual):
            sequence = [self.pairs[i] for i in individual]
            return (self.fitness_fn(sequence),)
        self.toolbox.register("evaluate", eval_func) #đánh giá fitness của cá thể bằng cách lấy sequence từ individual rồi tính điểm bằng fitness_fn

    def run(self):
        # Khởi tạo population
        pop = self.toolbox.population(n=self.pop_size)

        for gen in range(self.ngen):
            # Đánh giá fitness cho toàn bộ cá thể, pop_size = 10
            for ind in pop:
                ind.fitness.values = self.toolbox.evaluate(ind) 
            # 1. Chọn ra 2 cá thể tốt nhất (elitism)
            best_two = tools.selBest(pop, 2)

            # 2. Lấy phần còn lại để crossover + mutation
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

        # Đánh giá fitness lần cuối
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # Trả về sequence tốt nhất
        best_ind = tools.selBest(pop, 1)[0]
        best_seq = [self.pairs[i] for i in best_ind]
        return best_seq