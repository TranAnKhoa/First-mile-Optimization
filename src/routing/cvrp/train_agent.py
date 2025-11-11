import os
import sys
from pathlib import Path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Thêm thư mục 'src' vào sys.path nếu nó chưa có ở đó
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)
    
# === KẾT THÚC ĐOẠN MÃ THÊM VÀO ===

# Bây giờ các dòng import của bạn sẽ hoạt động bình thường
# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thêm thư mục gốc 'src' vào sys.path để có thể import các module khác
# (Giả sử file này được đặt trong thư mục `src/routing/cvrp/`)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- IMPORT CÁC THÀNH PHẦN CẦN THIẾT ---
from stable_baselines3 import PPO

# Import môi trường có tích hợp GP mà chúng ta đã tạo
from rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP

# Import các hàm tiện ích của bạn để tải dữ liệu
from routing.cvrp.alns_cvrp import cvrp_helper_functions

# --- CẤU HÌNH HUẤN LUYỆN ---
# Đường dẫn đến file instance bạn muốn dùng để huấn luyện
# Đường dẫn đến file instance bạn muốn dùng để huấn luyện
INSTANCE_FILE = r'K:\Data Science\SOS lab\Project Code\output_data\CEL_instance.pkl' # Sử dụng raw string 'r'

# Tổng số bước tương tác giữa agent và môi trường trong toàn bộ quá trình huấn luyện
# Con số này càng lớn, agent học được càng nhiều (nhưng tốn thời gian hơn)
# Ví dụ: 125 * 200 = 25,000. Tức là agent sẽ chạy qua 200 episodes.
TOTAL_TRAINING_STEPS = 2500

# Nơi lưu model đã huấn luyện và logs
# (Sẽ tạo ra file `model_directory/model.zip`)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "src", "routing", "cvrp", "model_directory", "model")
TENSORBOARD_LOG_PATH = os.path.join(PROJECT_ROOT, "src", "routing", "cvrp", "tensorboard_logs")


def train_new_agent():
    """
    Hàm chính để huấn luyện một agent PPO mới từ đầu.
    """
    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN AGENT MỚI ---")
    
    # --- 1. Tải dữ liệu bài toán ---
    print(f"Đang tải dữ liệu instance từ: {INSTANCE_FILE}")
    (_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)
    print("✅ Tải dữ liệu thành công.")

    # --- 2. Khởi tạo Môi trường ---
    # Sử dụng môi trường PPO_ALNS_Env_GP với state 9 chiều
    print("Đang khởi tạo môi trường PPO_ALNS_Env_GP...")
    # max_iterations = tổng số bước ALNS / kích thước buffer
    # Ví dụ: 1000 bước ALNS, buffer 8 -> max_iterations = 125
    env = PPO_ALNS_Env_GP(problem_instance=problem_obj, max_iterations=125)
    print("✅ Khởi tạo môi trường thành công.")

    # --- 3. Khởi tạo Agent PPO mới ---
    print("Đang khởi tạo agent PPO mới...")
    # 'MlpPolicy': Sử dụng mạng nơ-ron đa lớp (Multi-Layer Perceptron) cho state dạng vector.
    # policy_kwargs: Định nghĩa kiến trúc mạng nơ-ron.
    #                Đây là kiến trúc "thân chung, hai đầu ra" mà chúng ta đã thảo luận.
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,  # In ra thông tin huấn luyện
        tensorboard_log=TENSORBOARD_LOG_PATH
    )
    print("✅ Khởi tạo agent thành công.")

    # --- 4. Chạy Huấn luyện ---
    print(f"\nBắt đầu huấn luyện với {TOTAL_TRAINING_STEPS} bước...")
    # Hàm learn() sẽ tự động chạy nhiều episodes cho đến khi đủ số bước
    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, progress_bar=True)
    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")

    # --- 5. Lưu Model đã huấn luyện ---
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model đã được huấn luyện và lưu tại: {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    train_new_agent()