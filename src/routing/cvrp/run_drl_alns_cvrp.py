import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pathlib import Path
from rl.environments.cvrp_AlnsEnv_LSA1 import cvrpAlnsEnv_LSA1
import helper_functions

from stable_baselines3 import PPO

DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = 'E:/Syllabus for 2nd and 3rd year/cel/Code cel/DR-ALNS/code/src/routing/cvrp/configs/drl_alns_cvrp_debug.json'

def run_algo(folder, exp_name, client=None, **kwargs): #*kwargs -> dictionary of parameters được truyền qua json (gồm nhiều parameters như instance_nr,rseed, iterations,...)
    instance_nr = kwargs['instance_nr'] #--> instance number: sử dụng instance nào trong file instance (ở đây là instance thứ 9999 trong 10000 instance)
    seed = kwargs['rseed'] #giá trị khởi tạo, rseed=1 --> tất cả lần khởi tạo đều có giá trị giống nhau
    iterations = kwargs['iterations'] #số vòng lặp của thuật toán ALNS (max iterations)

    base_path = Path(__file__).parent.parent.parent
    instance_file = str(base_path.joinpath(kwargs['instance_file']))
    model_path = base_path / kwargs['model_directory'] / 'model' #Tạo đường dẫn đến model đã train xong trong thư mục model_directory
    model = PPO.load(model_path) #Load model ppo đã train xong
    
    parameters = {'environment': {'iterations': iterations, 'instance_nr': [instance_nr], 'instance_file': instance_file}} #Tạo dictionary parameters để truyền vào môi trường
    env = cvrpAlnsEnv_LSA1(parameters) #Tạo môi trường rồi truyền parameters vào
    env.reset() #Reset môi trường, agent về lại trạng thái ban đầu   
    env.run(model)
    best_objective = env.best_solution.objective()
    print("best_obj", best_objective)
    
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'instance_file'])
        writer.writerow([instance_nr, seed, iterations, env.best_solution.routes, best_objective, kwargs['instance_file']])

    return [], best_objective


def main(param_file=PARAMETERS_FILE):
    try:
        print(f"Attempting to read file: {param_file}")
        parameters = helper_functions.readJSONFile(param_file)
        print("Parameters loaded:", parameters)

        folder = DEFAULT_RESULTS_ROOT
        print("Results folder:", folder)

        exp_name = 'drl_alns' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])
        print("Experiment name:", exp_name)

        best_objective = run_algo(folder, exp_name, **parameters)
        return best_objective
    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
    
    
