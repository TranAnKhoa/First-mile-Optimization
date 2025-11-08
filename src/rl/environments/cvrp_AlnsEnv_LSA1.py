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

from routing.cvrp.alns_cvrp import cvrp_helper_functions #các hàm hỗ trợ cho cvrp

from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv #môi trường cvrp cơ bản
from routing.cvrp.alns_cvrp.destroy_operators import neighbor_graph_removal, random_removal, relatedness_removal # các toán tử phá hủy
from routing.cvrp.alns_cvrp.repair_operators import regret_insertion # toán tử sửa chữa
from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution # hàm tạo lời giải ban đầu


class cvrpAlnsEnv_LSA1(gym.Env): # Môi trường CVRP sử dụng ALNS với tiêu chí chấp nhận là Simulated Annealing
    def __init__(self, config, **kwargs):

        # Parameters
        self.config = config["environment"] #*TODO lấy giá trị config gắn với key environment (config là 1 dict) --> cấu hình instance, số iteration,...
        self.rnd_state = rnd.RandomState() #tao trạng thái ngẫu nhiên
    
        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # LOAD INSTANCE
        base_path = Path(__file__).resolve().parents[2] #đường dẫn đến thư mục gốc của project
        self.instance_file = str(base_path.joinpath(self.config["instance_file"])) #đường dẫn đến file instance
         #lấy danh sách các instance từ config
         #nếu instance là 1 chuỗi thì chuyển thành list có 1 phần tử
        
        self.instances = self.config["instance_nr"] #lấy danh sách các instance từ config
        self.instance = None #instance hiện tại
        self.best_routes = [] #lưu trữ các tuyến đường tốt nhất trong quá trình huấn luyện

        self.initial_solution = None #lời giải ban đầu
        self.best_solution = None #lời giải tốt nhất tìm được
        self.current_solution = None #lời giải hiện tại

        self.improvement = None #biến theo dõi sự cải thiện
        self.cost_difference_from_best = None #biến theo dõi sự khác biệt về chi phí so với lời giải tốt nhất
        self.current_updated = None #biến theo dõi việc cập nhật lời giải hiện tại
        self.current_improved = None #biến theo dõi việc cải thiện lời giải hiện tại

        # Gym-related part
        self.reward = 0  # Total episode reward --> biến lưu lại tổng phần thưởng của episode
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations) --> biến lưu lại số episode
        self.iteration = 0  # Current gen in the episode --> biến lưu lại số iteration hiện tại trong episode
        self.max_iterations = self.config["iterations"]  # max number of generations in an episode --> số iteration tối đa trong một episode

        #*! Action and observation spaces, action spaces là không gian hành động --> Tập hợp cho các action thực hiện trong môi trường
        self.action_space = gym.spaces.MultiDiscrete([3, 1, 10, 100]) # action space: [destroy_operator, repair_operator, factor, temperature]
        #Chọn destroy operator (3 loại).#có 3 toán tử phá hủy. 
        #Chọn repair operator (1 loại). #chỉ có 1 toán tử sửa chữa.
        #Số lượng node bị xóa (10 mức). #số node bị phá hủy: từ 10% đến 100% số node hiện có.
        #Tham số chỉnh temperature (100 mức). #temperature: từ 0.01 đến 1.00 (tăng dần độ "nóng" của hệ thống, làm giảm khả năng chấp nhận lời giải kém hơn).
        
        #*! Observation space là không gian quan sát --> Mô tả tất cả các trạng thái mà môi trường trả về cho agent
        self.observation_space = gym.spaces.Box(shape=(8,), low=0, high=100, dtype=np.float64)
        #giả sử khi gọi state, reward, done, info = env.step(action) thì state phải thuộc vè observation_space
        #TODO Tùy thuộc vào việc định nghĩa ở make_observation 
        # state = [improvement, cost_difference_from_best, is_current_best, temperature, stagcount, iteration/max_iterations, current_updated, current_improved]
        
        
        #*! Action space là tập hợp tất cả hành động agent có thể làm còn ob_space là tập hợp tất cả trạng thái môi trường trả về 
    def make_observation(self): #*! Tạo những parameters của state cho agent học
        """
        Return the environment's current state
        """
        
        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
             self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
            dtype=np.float64).squeeze()

        return state

    #*! Hàm dùng để reset môi trường về trạng thái ban đầu
    def reset(self, seed=None, options=None):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        SEED = random.randint(0, 10000) 

        # randomly select problem instance
        self.instance = random.choice(self.instances)

        # Load instance and create initial solution
        nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data = cvrp_helper_functions.read_input_cvrp(self.instance_file, self.instance)

        random_state = rnd.RandomState(SEED)
        
        #TODO State ở đây là nền tảng của dữ liệu để tính initial_solution
        state = cvrpEnv([], nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data,
                                self.instance, SEED)

        self.initial_solution = compute_initial_solution(state, random_state)
        self.current_solution = copy.deepcopy(self.initial_solution) #Dùng copy để tránh việc thay đổi dữ liệu gốc
        self.best_solution = copy.deepcopy(self.initial_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state) #thư viện có sẵn
        
        # thêm các toán tử phá hủy và sửa chữa vào đối tượng ALNS
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)

        self.dr_alns.add_repair_operator(regret_insertion)

        # reset tracking values
        self.stagcount = 0 #biến đếm số lần không cải thiện
        self.current_improved = 0 #biến theo dõi việc cải thiện lời giải hiện tại
        self.current_updated = 0 #biến theo dõi việc cập nhật lời giải hiện tại
        self.episode += 1 #tăng số episode lên 1
        self.temperature = self.max_temperature #khởi tạo nhiệt độ ban đầu
        self.improvement = 0 #biến theo dõi sự cải thiện
        self.cost_difference_from_best = 0 #biến theo dõi sự khác biệt về chi phí so với lời giải tốt nhất  

        self.iteration, self.reward = 0, 0 #khởi tạo số iteration và phần thưởng về 0
        self.done = False #khởi tạo biến done về False

        #! Truyền vào những paramters ban đầu để trả về state
        return self.make_observation()  # Chỉ trả về obs, bỏ info
    
    #*! Hàm dùng để thực hiện một hành động (action) trong môi trường    
    def step(self, action, **kwargs):
        self.iteration += 1
        self.stagcount += 1 #tăng biến đếm số lần không cải thiện lên 1
        self.current_updated = 0 #biến theo dõi việc cập nhật lời giải hiện tại
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0 #biến theo dõi sự khác biệt về chi phí so với lời giải tốt nhất
        self.current_improved = 0

        current = self.current_solution
        best = self.best_solution

        #TODO Action là một mảng gồm 4 phần tử [destroy_operator, repair_operator, factor, temperature]
        d_idx, r_idx = action[0], action[1] #lấy chỉ số của toán tử phá hủy và sửa chữa từ action
        

        factors = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0} #bảng ánh xạ từ chỉ số đến tỷ lệ phần trăm số node bị phá hủy
        #(factors đã được quy định ở action_space)
        
        nr_nodes_to_remove = round(factors[action[2]] * current.nb_customers) #tính số node cần phá hủy dựa trên tỷ lệ phần trăm và số khách hàng hiện có   

        self.temperature = (1/(action[3]+1)) * self.max_temperature #tính nhiệt độ dựa trên chỉ số action[3] (temperature)

        # Nếu số node cần phá hủy bằng số khách hàng hiện có, giảm đi 1 để tránh việc phá hủy toàn bộ
        if nr_nodes_to_remove == current.nb_customers:
            nr_nodes_to_remove -= 1

        ##!! ALNS step
        
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx] #lấy tên và hàm của toán tử phá hủy từ chỉ số từ file alns.py
        
        destroyed = d_operator(current, self.rnd_state, nr_nodes_to_remove) #!Thực hiện toán tử phá hủy để tạo lời giải bị phá hủy
        #current là lời giản hiện tại, self.rnd_state là trạng thái ngẫu nhiên, nr_nodes_to_remove là số node cần phá hủy

        r_name, r_operator = self.dr_alns.repair_operators[r_idx] #lấy tên và hàm của toán tử sửa chữa từ chỉ số
        
        candidate = r_operator(destroyed, self.rnd_state) #!Thực hiện toán tử sửa chữa để tạo lời giải ứng viên từ lời giải bị phá hủy

        print(f"Destroy operator: {d_name},Repair operator: {r_name}")
        #! Hàm consider_candidate để quyết định có chấp nhận lời giải ứng viên hay không dựa trên tiêu chí chấp nhận Simulated Annealing
        new_best, new_current = self.consider_candidate(best, current, candidate) #Xem xét lời giải ứng viên để quyết định có chấp nhận nó hay không dựa trên tiêu chí chấp nhận Simulated Annealing


        # Update best and current solutions
        if new_best != best and new_best is not None: #Nếu có lời giải tốt hơn lời giải tốt nhất (best) hiện tại
            # found new best solution
            self.best_solution = new_best #cập nhật lời giải tốt nhất
            self.current_solution = new_best #cập nhật lời giải hiện tại
            self.current_updated = 1 #biến theo dõi việc cập nhật lời giải hiện tại
            self.reward += 5 #phần thưởng khi tìm được lời giải tốt nhất
            self.stagcount = 0 #reset biến đếm số lần không cải thiện về 0
            self.current_improved = 1 #biến theo dõi việc cải thiện lời giải hiện tại

        elif new_current != current and new_current.objective() > current.objective(): #Nếu new_current tốt hơn current nhưng không vượt qua best
            # solution accepted, because better than current, but not better than best
            self.current_solution = new_current #cập nhật lời giải hiện tại
            self.current_updated = 1 #biến theo dõi việc cập nhật lời giải hiện tại
            self.current_improved = 1 #biến theo dõi việc cải thiện lời giải hiện tại
            # self.reward += 3

        elif new_current != current and new_current.objective() <= current.objective(): #Nếu new_current kém hơn hoặc bằng current
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            # self.reward += 1

        if new_current.objective() > current.objective(): #Nếu new_current tốt hơn current
            self.improvement = 1 # gán biến theo dõi sự cải thiện = 1
        
        # Tính parameters for state: objective value của lời giải hiện tại so với lời giải tốt nhất
        self.cost_difference_from_best = (self.current_solution.objective() / self.best_solution.objective()) * 100

        
        # Cập nhật đồ thị lân cận của lời giải hiện tại và lời giải tốt nhất (trong ALNS)
        self.current_solution.graph = self.best_solution.graph = cvrp_helper_functions.update_neighbor_graph(candidate, candidate.routes, candidate.objective())

        state = self.make_observation() #!Tạo state mới sau khi thực hiện action
        self.best_routes.append(self.best_solution.objective()) #lưu lại giá trị objective của lời giải tốt nhất vào danh sách best_routes

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True

            import random, string, csv, os
            directory_path = 'E:/Syllabus for 2nd and 3rd year/cel/Code cel/DR-ALNS/output_trajectories_drl_10k/'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Generate random file name
            file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=100)) + '.csv'
            random_string = os.path.join(directory_path, file_name)

            # Write data to the file
            with open(random_string, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.best_routes)

        return state, self.reward, self.done, False, {} #! Trả về state, reward, done, truncated, info

    # --------------------------------------------------------------------------------------------------------------------

    #! Hàm xem xét lời giải ứng viên để quyết định có chấp nhận nó hay không dựa trên tiêu chí chấp nhận Simulated Annealing
    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
        probability = np.exp((curr.objective() - cand.objective()) / self.temperature)

        # best:
        if cand.objective() < best.objective(): #Nếu lời giải ứng viên tốt hơn lời giải tốt nhất hiện tại
            return cand, cand

        # accepted:
        elif probability >= rnd.random(): #Nếu lời giải ứng viên kém hơn lời giải hiện tại nhưng vẫn được chấp nhận dựa trên xác suất
            return None, cand

        else:
            return None, curr #Không chấp nhận lời giải ứng viên, giữ nguyên lời giải hiện tại

    # --------------------------------------------------------------------------------------------------------------------
#! Hàm dùng để chạy mô hình đã được huấn luyện
    def run(self, model, episodes=1):
        """
        Use a trained model to select actions
        """
        try:
            for episode in range(episodes): #chạy qua số episode
                self.done = False #khởi tạo biến done về False => True là kết thúc episode
                state = self.reset() #reset môi trường về trạng thái ban đầu và lấy state ban đầu
                while not self.done: #chạy đến khi done = True
                    #! Model sẽ được học cách chọn action dựa trên state hiện tại ==> chọn 4 parameters: destroy_operator, repair_operator, factor phá hủy note, temperature
                    action = model.predict(state) #dự đoán action dựa trên state hiện tại
                    
                    state, reward, terminated, truncated, info = self.step(action[0]) #thực hiện action trong môi trường và nhận về state mới, reward, done
                    #State có dạng: [improvement, cost_difference_from_best, is_current_best, temperature, stagcount, iteration/max_iterations, current_updated, current_improved]
                    self.done = terminated or truncated  # Kết hợp terminated và truncated
                    print(f"Episode {episode}, State: {state}, Reward: {reward}, Done: {self.done}")
        except KeyboardInterrupt:
            pass

    #! Hàm dùng để chạy mô hình đã được huấn luyện với giới hạn thời gian
    def run_time_limit(self, model, episodes=1):
        """
        Use a trained model to select actions
        """
        try:
            for episode in range(episodes):
                start_time = time.time()
                time_done = False
                state = self.reset()
                while not time_done:
                    action = model.predict(state)
                    state, reward, _, _ = self.step(action[0])
                    current_time = time.time() - start_time
                    print(current_time)
                    if current_time > 30: # giới hạn thời gian chạy là 30 giây
                        time_done = True
                    # print(state, reward, self.iteration)
        except KeyboardInterrupt:
            pass

    def sample(self): #! Hàm dùng để lấy mẫu các hành động ngẫu nhiên và chạy môi trường
        """
        Sample random actions and run the environment
        """
        for episode in range(2):
            self.done = False
            state = self.reset()
            print("start episode: ", episode, " with start state: ", state)
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print(
                    "step {}, action: {}, New state: {}, Reward: {:2.3f}".format(
                        self.iteration, action, state, reward
                    )
                )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from rl.baselines import get_parameters, Trainer

    env = cvrpAlnsEnv_LSA1(get_parameters("cvrpAlnsEnv_LSA1"))
    # print("Sampling random actions...")
    # env.sample()

    print('Start training')
    model = Trainer("cvrpAlnsEnv_LSA1", "models").create_model()
    # model._tensorboard()
    model.train()
    print("Training done")
    input("Run trained model (Enter)")
    env.run(model)
