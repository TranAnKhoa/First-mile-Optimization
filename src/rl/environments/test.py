# Giả lập candidate solution
from gp_sequence import GPSequence
class DummySolution:
    def __init__(self, nb_customers):
        self.nb_customers = nb_customers
        self.obj = 100  # objective ban đầu
    def objective(self):
        return self.obj

# Giả lập destroy & repair operators
def destroy_operator(sol, rnd, nr_nodes):
    new_sol = DummySolution(sol.nb_customers)
    new_sol.obj = sol.obj - nr_nodes * rnd.random() * 5  # giảm objective 1 cách ngẫu nhiên
    return new_sol

def repair_operator(sol, rnd):
    return sol  # đơn giản: repair không thay đổi gì

import random

# Fitness function thử nghiệm
def test_fitness_fn(sequence):
    # sequence càng "sắp xếp" theo index tăng, fitness càng lớn
    score = 0
    for idx, (d, r) in enumerate(sequence):
        score += (d + r) * (idx+1)
    return score  # càng nhỏ càng tốt

# Test GPSequence
actions = [(3,2), (1,2), (0,3)]
gp = GPSequence(actions, test_fitness_fn, ngen=5, pop_size=6)
best_seq = gp.run()
print("Best sequence from GP:", best_seq)
