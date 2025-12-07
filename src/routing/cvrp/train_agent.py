import os
import sys
from pathlib import Path
import glob # <-- Thêm import này

# === ĐOẠN MÃ THÊM VÀO SYS.PATH (Giữ nguyên) ===
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# === KẾT THÚC ===

from stable_baselines3 import PPO
# --- THÊM IMPORT CALLBACK ---
from stable_baselines3.common.callbacks import CheckpointCallback 

from rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP
from routing.cvrp.alns_cvrp import cvrp_helper_functions

# --- CẤU HÌNH (Giữ nguyên) ---
INSTANCE_FILE = r'K:\Data Science\SOS lab\Project Code\output_data\CEL_instance.pkl'
TOTAL_TRAINING_STEPS = 2500

# --- CẤU HÌNH ĐƯỜNG DẪN ĐÃ CẬP NHẬT ---
# Nơi lưu model *cuối cùng*
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "src", "routing", "cvrp", "model_directory", "ankhoa_model_1")
# Nơi lưu các file backup (checkpoints)
CHECKPOINT_DIR = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "checkpoints")
CHECKPOINT_NAME_PREFIX = "ppo_alns_checkpoint" # Tên file backup

TENSORBOARD_LOG_PATH = os.path.join(PROJECT_ROOT, "src", "routing", "cvrp", "tensorboard_logs")

# --- HÀM HELPER MỚI: Tìm checkpoint mới nhất ---
def get_latest_checkpoint(checkpoint_dir, prefix):
    """Tìm file checkpoint mới nhất trong thư mục."""
    try:
        # Tìm tất cả các file .zip khớp với prefix
        list_of_files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.zip"))
        if not list_of_files:
            return None # Không tìm thấy checkpoint
        
        # Tìm file có số bước (steps) cao nhất
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Lỗi khi tìm checkpoint: {e}")
        return None

# --- HÀM TRAIN ĐÃ CẬP NHẬT ---
def train_agent(): # Đổi tên hàm một chút
    """
    Hàm chính để huấn luyện hoặc tiếp tục huấn luyện agent.
    """
    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")
    
    # --- 1. Tải dữ liệu bài toán ---
    print(f"Đang tải dữ liệu instance từ: {INSTANCE_FILE}")
    (_, _, _, _, _, _, _, _, problem_obj) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)
    print("✅ Tải dữ liệu thành công.")

    # --- 2. Khởi tạo Môi trường ---
    print("Đang khởi tạo môi trường PPO_ALNS_Env_GP...")
    env = PPO_ALNS_Env_GP(problem_instance=problem_obj, max_iterations=125)
    print("✅ Khởi tạo môi trường thành công.")

    # --- 3. KIỂM TRA CHECKPOINT VÀ TẢI MODEL ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Tạo thư mục checkpoint nếu chưa có
    latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR, CHECKPOINT_NAME_PREFIX)

    if latest_checkpoint:
        print(f"🔥 Tìm thấy checkpoint! Đang tải từ: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        # Đảm bảo model tiếp tục log vào đúng nơi
        model.set_tensorboard_log(TENSORBOARD_LOG_PATH)
        print("✅ Tải model từ checkpoint thành công. Tiếp tục huấn luyện...")
    
    else:
        print("🌱 Không tìm thấy checkpoint. Khởi tạo agent PPO mới...")
        # 'MlpPolicy': (Giữ nguyên)
        policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
        
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
        print("✅ Khởi tạo agent mới thành công.")

    # --- 4. TẠO CALLBACK ĐỂ TỰ ĐỘNG LƯU ---
    # Tự động lưu sau mỗi 500 bước
    checkpoint_callback = CheckpointCallback(
        save_freq=500, # <-- LƯU SAU MỖI 500 BƯỚC
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_NAME_PREFIX
    )

    # --- 5. Chạy Huấn luyện ---
    print(f"\nBắt đầu huấn luyện với {TOTAL_TRAINING_STEPS} bước...")
    
    model.learn(
        total_timesteps=TOTAL_TRAINING_STEPS,
        progress_bar=True,
        callback=checkpoint_callback, # <-- THÊM CALLBACK VÀO ĐÂY
        reset_num_timesteps=False # <-- Quan trọng: Không reset số bước khi resume
    )
    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")

    # --- 6. Lưu Model (CUỐI CÙNG) ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model *cuối cùng* đã được huấn luyện và lưu tại: {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    train_agent()