import os
import sys
import glob
import random
import numpy as np
import gymnasium as gym
import contextlib
import io
import torch as th

# ==============================================================================
# 1. SETUP
# ==============================================================================
PROJECT_ROOT = r'C:\AnKhoa\Project_Code'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from src.rl.environments.PPO_ALNS_Env_GP import PPO_ALNS_Env_GP
    from src.routing.cvrp.alns_cvrp import cvrp_helper_functions
except ImportError as e:
    print(f"❌ Lỗi Import: {e}"); sys.exit(1)

TRAIN_DATA_DIR = r'C:\AnKhoa\Project_Code\input_data_train'
SAVE_DIR = os.path.join(PROJECT_ROOT, "Save_model")
LOG_DIR = os.path.join(SAVE_DIR, "tensorboard_logs_final")
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints_final")
MODEL_PATH = os.path.join(SAVE_DIR, "ankhoa_model_final_clean")

# ==============================================================================
# 2. HELPER ĐƠN GIẢN
# ==============================================================================
def sanitize_input_data(problem_instance):
    """Đảm bảo data đầu vào là Numpy Array float32"""
    if not isinstance(problem_instance, dict): return problem_instance
    data = problem_instance.copy()
    
    keys = ['distance_matrix_farms', 'distance_depots_farms', 'distance_matrix_depots', 'demands', 'locations']
    for k in keys:
        if k in data and isinstance(data[k], list):
            data[k] = np.array(data[k], dtype=np.float32)
    return data

# ==============================================================================
# 3. WRAPPER CƠ BẢN
# ==============================================================================
class SimpleFloatWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.flat_shape = (9,) 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.flat_shape, dtype=np.float32
        )

    def _clean(self, obs):
        return np.array(obs, dtype=np.float32).flatten()

    def reset(self, seed=None, options=None):
        with contextlib.redirect_stdout(io.StringIO()):
            obs, info = self.env.reset(seed=seed, options=options)
        return self._clean(obs), info

    def step(self, action):
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, terminated, truncated, info = self.env.step(action)
        return self._clean(obs), float(reward), terminated, truncated, info

# ==============================================================================
# 4. MAIN
# ==============================================================================
class HeartbeatCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cnt = 0
    def _on_step(self) -> bool:
        self.cnt += 1
        if self.cnt % 50 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        return True

def train():
    print("🚀 TRAINING (CLEAN VERSION - MAX_ITER=200)")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.pkl"))
    if not files: print("❌ No Data"); return
    print(f"✅ Loaded {len(files)} maps.")

    # Cấu hình mạng to [128, 128]
    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # --- KHỞI TẠO ENV MẪU ---
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            (_, _, _, _, _, _, _, _, prob) = cvrp_helper_functions.read_input_cvrp(files[0])
            prob = sanitize_input_data(prob)
            # Init mẫu: 200
            raw_env = PPO_ALNS_Env_GP(prob, max_iterations=200, buffer_size=1, verbose=False)
        
        env = DummyVecEnv([lambda: Monitor(SimpleFloatWrapper(raw_env), LOG_DIR)])
        
    except Exception as e:
        print(f"❌ Init Error: {e}"); return

    # --- KHỞI TẠO AGENT ---
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
                learning_rate=3e-4, n_steps=1024, batch_size=64, ent_coef=0.05,
                policy_kwargs=policy_kwargs)
    
    ckpt = CheckpointCallback(save_freq=50000, save_path=CKPT_DIR, name_prefix="ppo_clean")
    heartbeat = HeartbeatCallback()

    TOTAL_STEPS = 500000
    STEPS_PER_LOOP = 2048 
    steps = 0
    idx = 0

    print(f"\n🏃 Running {TOTAL_STEPS} steps...")

    while steps < TOTAL_STEPS:
        f_path = random.choice(files)
        print(f"\n📂 [{idx+1}] Map: {os.path.basename(f_path)} | {steps}/{TOTAL_STEPS}")
        print("   Running: ", end="")

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                (_, _, _, _, _, _, _, _, prob) = cvrp_helper_functions.read_input_cvrp(f_path)
                prob = sanitize_input_data(prob)
                
                # --- ĐÃ CHỈNH LẠI THÀNH 200 ---
                new_raw = PPO_ALNS_Env_GP(prob, max_iterations=200, buffer_size=1, verbose=False)
            
            new_env = DummyVecEnv([lambda: Monitor(SimpleFloatWrapper(new_raw), LOG_DIR)])
            
            model.set_env(new_env)
            model._last_obs = None 
            
            model.learn(total_timesteps=STEPS_PER_LOOP, reset_num_timesteps=False, 
                        callback=[ckpt, heartbeat], progress_bar=False)
            
            steps += STEPS_PER_LOOP
            idx += 1
            new_env.close()
            
        except Exception as e:
            print(f"\n❌ Skipped: {e}")
            continue

    print(f"\n🎉 DONE!"); model.save(MODEL_PATH)

if __name__ == "__main__":
    train()