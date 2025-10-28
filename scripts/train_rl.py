# scripts/train_rl.py
import os
import yaml
import numpy as np
from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import multiprocessing as mp
import math
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks.progress_bar import ProgressBarCallback

def make_env(rank, cfg, base_seed=42):
    def _init():
        # 고정 목표(P2)
        p_goal = np.array(cfg["P2"]["pos"], dtype=float)
        q_goal_xyzw = wxyz_to_xyzw(cfg["P2"]["quat"])
        q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)

        task = RLTaskCfg(
            target_pos = p_goal,
            target_quat_wxyz = q_goal_wxyz,
            # episode_time : 도달 전에 episode가 끝나면 학습이 꼬인다
            episode_time = max(10.0, cfg["demo"]["move_duration"] + cfg["demo"]["hold_duration"]),
            ctrl_hz = cfg["control_hz"],
            # action_scale : 토크가 너무 작으면 이동 자체가 어렵다.
            action_scale = np.array(cfg["ctrl"]["torque_limit"], dtype=float) * 0.8,
            pos_w=3.0, rot_w=3.0, torque_w=1e-4, smooth_w=5e-5,
            success_pos_tol=0.001, success_rot_tol_deg=0.1,
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
    env = SubprocVecEnv([make_env(i, cfg) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        train_freq=(n_envs, "step"),
        gradient_steps=max(6, math.ceil(n_envs * 0.75)),
        # entropy coefficient : 엔트로피 목표를 약간 높게 잡아 초반 탐색을 늘린다. (기존 'auto'보다 강하게)
        ent_coef="auto",
        # learning_starts=20_000,            # 초기 랜덤 수집
        verbose=1,
        tensorboard_log="tb/",
        tau=0.02,                  # 타깃 폴리시 업데이트
        policy_kwargs=dict(net_arch=[256, 256])
    )

    # -------- 콜백들 연결 --------
    TOTAL_STEPS = 15_000_000

    # 평가용 환경(단일) — 훈련과 같은 방식으로 감싸되, 학습(통계 업데이트) 비활성화
    eval_env = DummyVecEnv([make_env(999, cfg, base_seed=1234)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/best",
        log_path="eval",
        eval_freq=10_000,          # 1만 스텝마다 1회 평가(기본 1 episode)
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="checkpoints",
        name_prefix="sac_step"
    )

    pbar_cb = ProgressBarCallback(total_timesteps=TOTAL_STEPS, desc="SAC train")

    callbacks = CallbackList([eval_cb, ckpt_cb, pbar_cb])

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ur10e_sac")
    # VecNormalize 통계 저장 (env가 VecNormalize로 감싸졌을 때만 가능)
    if isinstance(env, VecNormalize):
        env.save("checkpoints/vecnorm.pkl")

    env.close()
