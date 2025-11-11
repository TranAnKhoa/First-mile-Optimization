import csv
import sys
import os
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đảm bảo Python có thể tìm thấy các module của bạn
# (Giữ nguyên cấu trúc của bạn)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- IMPORT CÁC THÀNH PHẦN CẦN THIẾT ---
# Môi trường PPO mới mà chúng ta đã tạo
from rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP 
# Các hàm tiện ích của bạn
import helper_functions 
# Agent PPO từ Stable Baselines3
from stable_baselines3 import PPO
# Hàm đọc dữ liệu của bạn
from routing.cvrp.alns_cvrp import cvrp_helper_functions


# --- CÁC HẰNG SỐ ---
DEFAULT_RESULTS_ROOT = "single_runs/"
# Thay đổi đường dẫn này cho phù hợp với cấu trúc project của bạn
PARAMETERS_FILE = r'K:\Data Science\SOS lab\Project Code\src\routing\cvrp\configs\drl_alns_cvrp_debug.json'


def run_evaluation(folder, exp_name, problem_instance, **kwargs):
    """
    Hàm này thực hiện việc đánh giá một agent PPO đã huấn luyện trên một instance cụ thể.
    Nó thay thế cho vòng lặp `env.run(model)` bằng vòng lặp chuẩn của Gym.
    """
    # 1. Trích xuất các tham số cần thiết
    instance_nr = kwargs['instance_nr']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # 2. Khởi tạo môi trường PPO_ALNS_Env
    # Môi trường cần 'problem_instance' và 'max_iterations'
    env = PPO_ALNS_Env_GP(problem_instance=problem_instance, max_iterations=iterations)

    # 3. Tải model PPO đã được huấn luyện
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / kwargs['model_directory'] / 'model'
    print(f"Đang tải model từ: {model_path}")
    model = PPO.load(model_path)
    
    # --- 4. VÒNG LẶP ĐÁNH GIÁ (THAY THẾ CHO env.run(model)) ---
    print("\n--- Bắt đầu vòng lặp đánh giá ---")
    obs = env.reset()
    for i in range(iterations):
        # Agent chọn hành động tốt nhất dựa trên state hiện tại
        # deterministic=True đảm bảo agent không khám phá ngẫu nhiên nữa
        action, _states = model.predict(obs, deterministic=True)
        
        # Môi trường thực hiện hành động đó
        obs, reward, done, info = env.step(action)
        
        # In ra tiến trình (tùy chọn)
        env.render()
        
        # Nếu episode (ALNS run) kết thúc, dừng vòng lặp
        if done:
            break
    print("--- Vòng lặp đánh giá kết thúc ---")

    # 5. Lấy kết quả cuối cùng từ môi trường
    # Các thuộc tính .best_solution và .best_objective được theo dõi bên trong môi trường
    best_solution_routes = env.best_solution.schedule # <-- XÁC NHẬN TÊN THUỘC TÍNH NÀY
    best_objective_value = env.best_objective
    
    print(f"\nKết quả tốt nhất đạt được: {best_objective_value}")
    
    # 6. Ghi kết quả ra file CSV (giữ nguyên logic của bạn)
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder + exp_name + ".csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution_schedule', 'best_objective', 'instance_file'])
        writer.writerow([instance_nr, seed, iterations, best_solution_routes, best_objective_value, kwargs['instance_file']])

    return best_objective_value


def main(param_file=PARAMETERS_FILE):
    try:
        print(f"Đang đọc file tham số: {param_file}")
        parameters = helper_functions.readJSONFile(param_file)
        print("Đã tải tham số:", parameters)

        # --- TẢI DỮ LIỆU INSTANCE TRƯỚC KHI CHẠY ---
        base_path = Path(__file__).parent.parent.parent
        instance_file = str(base_path.joinpath(parameters['instance_file']))
        
        print(f"Đang đọc dữ liệu instance từ: {instance_file}")
        # Sử dụng hàm đọc input của bạn để lấy ra đối tượng 'problem'
        (_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(instance_file)
        print("Đã đọc xong dữ liệu instance.")

        # Thiết lập thư mục và tên file kết quả
        folder = DEFAULT_RESULTS_ROOT
        exp_name = 'drl_alns_eval_' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])
        print("Tên file kết quả:", exp_name)

        # Gọi hàm đánh giá và truyền vào 'problem_instance' đã được tải
        best_objective = run_evaluation(folder, exp_name, problem_instance=problem_obj, **parameters)
        return best_objective

    except Exception as e:
        print(f"Lỗi trong hàm main: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()