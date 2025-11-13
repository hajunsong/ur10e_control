# scripts/train_rl.py
import os
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 반드시 torch import 전에
import yaml
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import random


from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz
from utils.seed_utils import set_global_seed

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import multiprocessing as mp
import math
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks.progress_bar import ProgressBarCallback
from callbacks.save_best_replay import EvalWithReplaySave
from callbacks.safety_callback import SafetyCallback

# ---------- 런 폴더 생성 ----------
def make_run_dir(root="runs", run_name=None, make_latest_symlink=True):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or ts
    run_dir = Path(root) / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    if make_latest_symlink:
        latest = Path(root) / "latest"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except Exception:
            pass
    return run_dir

def make_env(rank, cfg, task, render=False, base_seed=42):
    def _init():
        env = UR10eRLEnv(
            xml_path=cfg["mujoco_xml"],
            end_effector_site=cfg["end_effector_site"],
            q0=np.array(cfg["q0"], dtype=float),
            torque_limit=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
            task_cfg=task,
            render=render
        )
        
        # Gymnasium 스타일 시딩: action/obs space 포함
        env.action_space.seed(base_seed + rank)
        env.observation_space.seed(base_seed + rank)
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None, help="런 폴더 이름(기본: 타임스탬프)")
    parser.add_argument("--n-envs", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=6_000_000)
    parser.add_argument("--seed", type=int, default=123, help="전연 시드")

    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    # 전역 시드 고정
    set_global_seed(args.seed, deterministic_torch=False)

    with open("config/sim.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_dir = make_run_dir(run_name=args.run_name)  # runs/<name>/
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    best_dir = ckpt_dir / "best"
    eval_dir = run_dir / "eval"

    # config 스냅샷 저장
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # 목표(P2)
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
        action_scale = np.array(cfg["ctrl"]["torque_limit"], dtype=float),
        pos_w=4.0, rot_w=3.0, torque_w=1e-4, smooth_w=5e-5,
        success_pos_tol=0.001, success_rot_tol_deg=0.1,
        # VSD
        use_vsd=cfg["vsd"]["use_vsd"],
        vsd_alpha=cfg["vsd"]["vsd_alpha"],
        Kp_vsd_pos=cfg["vsd"]["Kp_vsd_pos"],
        Kd_vsd_pos=cfg["vsd"]["Kd_vsd_pos"],
        Kp_vsd_rot=cfg["vsd"]["Kp_vsd_rot"],
        Kd_vsd_rot=cfg["vsd"]["Kd_vsd_rot"],
    )

    # 병렬 환경
    n_envs = args.n_envs  # 병렬 환경 개수
    base_seed = args.seed*1000
    train_env = SubprocVecEnv([make_env(i, cfg, task, base_seed=base_seed) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)

    # 평가용 env
    eval_env_base = DummyVecEnv([make_env(999, cfg, task, render=False, base_seed=base_seed)])
    # 같은 통계 공유
    eval_env = VecNormalize(eval_env_base, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.clip_obs = train_env.clip_obs
    eval_env.clip_reward = train_env.clip_reward
    eval_env.norm_obs = train_env.norm_obs
    eval_env.norm_reward = False

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        # learning_rate=1e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        train_freq=(n_envs, "step"),
        gradient_steps=max(6, math.ceil(n_envs * 0.75)),
        # entropy coefficient : 엔트로피 목표를 약간 높게 잡아 초반 탐색을 늘린다. (기존 'auto'보다 강하게)
        ent_coef="auto_0.2",
        # ent_coef="auto",
        # learning_starts=20_000,            # 초기 랜덤 수집
        verbose=1,
        tensorboard_log=str(tb_dir),
        tau=0.02,                  # 타깃 폴리시 업데이트
        policy_kwargs=dict(net_arch=[256, 256])
    )
    model.set_random_seed(args.seed)

    # 콜백들
    eval_cb = EvalWithReplaySave(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        save_vecnorm=True,                    # ← VecNormalize 통계도 같이
        replay_name="best_replay_buffer.pkl", # 파일명 마음대로
        vecnorm_name="vecnorm.pkl"
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="sac_step",
        save_replay_buffer=True,
    )

    pbar_cb = ProgressBarCallback(total_timesteps=args.total_steps, desc=run_dir.name)

    safety_cb = SafetyCallback(save_path=f"{run_dir}/checkpoints/safe_ckpt_before_explode.zip", threshold=1e3)

    callbacks = CallbackList([eval_cb, ckpt_cb, pbar_cb, safety_cb])

    # 학습
    # ---- 이어서 학습 (스텝 카운트 연속) ----
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=False,  # ★ 중요: 이어서
        tb_log_name="SAC_continue"  # 텐서보드 런 이름(원하면 변경)
    )

    # 최종 저장 + VecNormalize 상태 저장
    model.save(str(ckpt_dir / "final_model"))
    train_env.save(str(ckpt_dir / "vecnorm.pkl"))

    train_env.close()
    eval_env.close()

    print(f"\n[Run Saved] {run_dir}\n"
          f"  - TB:          {tb_dir}\n"
          f"  - Checkpoints: {ckpt_dir}\n"
          f"  - Best:        {best_dir}\n"
          f"  - Eval logs:   {eval_dir}\n"
          f"  - Config:      {run_dir / 'config_snapshot.yaml'}\n")
