# scripts/train_rl.py
import os
import yaml
import numpy as np
from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing as mp
import math

def make_env(rank, base_seed=42):
    def _init():
        # 고정 목표(P2)
        p_goal = np.array(cfg["P2"]["pos"], dtype=float)
        q_goal_xyzw = wxyz_to_xyzw(cfg["P2"]["quat"])
        q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)

        task = RLTaskCfg(
            target_pos = p_goal,
            target_quat_wxyz = q_goal_wxyz,
            episode_time = cfg["demo"]["move_duration"] + cfg["demo"]["hold_duration"],
            ctrl_hz = cfg["control_hz"],
            action_scale = np.array(cfg["ctrl"]["torque_limit"], dtype=float) * 0.5,  # 안전 스케일
            pos_w=13.0, rot_w=12.0, torque_w=1e-4, smooth_w=5e-5,
            success_pos_tol=2e-3, success_rot_tol_deg=1.0
        )

        env = UR10eRLEnv(
            xml_path=cfg["mujoco_xml"],
            end_effector_site=cfg["end_effector_site"],
            q0=np.array(cfg["q0"], dtype=float),
            torque_limit=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
            task_cfg=task,
            render=False
        )
        
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open("config/sim.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    n_envs = 12  # 병렬 환경 개수
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        train_freq=(n_envs, "step"),
        gradient_steps=max(4, math.ceil(n_envs * 0.5)),
        verbose=1,
        tensorboard_log="tb/",
        tau=0.02,                  # 타깃 폴리시 업데이트
        policy_kwargs=dict(net_arch=[256, 256])
    )
    model.learn(total_timesteps=1_300_000)
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ur10e_sac")

    env.close()
