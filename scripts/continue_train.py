# scripts/continue_train.py
import os, math, yaml, argparse, multiprocessing as mp
from pathlib import Path
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz

# ---- 간단 런 폴더 유틸 (새 시간 폴더 생성) ----
from datetime import datetime
def make_run_dir(root="runs", tag="continue"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(root) / f"{ts}__{tag}"
    (run / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
    (run / "tb").mkdir(parents=True, exist_ok=True)
    (run / "eval").mkdir(parents=True, exist_ok=True)
    return run

def make_env(rank, cfg, task, seed_base=42, render=False):
    def _init():
        env = UR10eRLEnv(
            xml_path=cfg["mujoco_xml"],
            end_effector_site=cfg["end_effector_site"],
            q0=np.array(cfg["q0"], dtype=float),
            torque_limit=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
            task_cfg=task,
            render=render
        )
        env.reset(seed=seed_base + rank)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    # ap.add_argument("--prev-run", required=True, help="이전 런 폴더 (runs/2025xxxx_xxxxxx)")
    # ap.add_argument("--ckpt", default="best/best_model.zip", help="이전 런 내부의 모델 경로 (상대)")
    ap.add_argument("--n-envs", type=int, default=20)
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--tag", type=str, default="continue")
    ap.add_argument("--lr", type=float, default=1e-4, help="이어학습 시 새 학습률(선택)") # 3e-4 → 1e-4 혹은 5e-5.
    ap.add_argument("--load-rb", action="store_true", help="리플레이 버퍼도 불러오기")
    args = ap.parse_args()

    with open("config/sim.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    prev = Path("runs/20251110_123428__continue")
    ckpt_path   = prev / "checkpoints" / "sac_step_8000000_steps.zip"
    vecnorm_pkl = prev / "checkpoints" / "vecnorm.pkl"
    rb_path     = prev / "checkpoints" / "sac_step_replay_buffer_8000000_steps.pkl"

    # 목표(P2)
    p_goal = np.array(cfg["P2"]["pos"], dtype=float)
    q_goal_xyzw = wxyz_to_xyzw(cfg["P2"]["quat"])
    q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)

    task = RLTaskCfg(
        target_pos=p_goal,
        target_quat_wxyz=q_goal_wxyz,
        episode_time=max(10.0, cfg["demo"]["move_duration"] + cfg["demo"]["hold_duration"]),
        ctrl_hz=cfg["control_hz"],
        action_scale=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
        pos_w=3.0, rot_w=3.0, torque_w=1e-4, smooth_w=5e-5,
        success_pos_tol=0.001, success_rot_tol_deg=0.1,
        # (VSD를 학습 중에 썼다면 동일 설정 유지)
        use_vsd=True,
        vsd_alpha=1.0,  # 시작은 0.1~0.3 권장
        Kp_vsd_pos=1200.0,
        Kd_vsd_pos=50.0,
        Kp_vsd_rot=80.0,
        Kd_vsd_rot=6.0
    )

    # 새 런 폴더
    run_dir = make_run_dir(tag=args.tag)
    tb_dir  = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    best_dir = ckpt_dir / "best"
    eval_dir = run_dir / "eval"

    # ---- 환경 구성 (이전 VecNormalize 통계 로드) ----
    n_envs = args.n_envs
    train_env_base = SubprocVecEnv([make_env(i, cfg, task) for i in range(n_envs)])

    if vecnorm_pkl.exists():
        train_env = VecNormalize.load(str(vecnorm_pkl), train_env_base)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env_base, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)

    eval_env_base = DummyVecEnv([make_env(999, cfg, task, render=False)])
    eval_env = VecNormalize(eval_env_base, training=False)
    # 학습 env의 통계 공유
    eval_env.obs_rms  = train_env.obs_rms
    eval_env.ret_rms  = train_env.ret_rms
    eval_env.norm_obs = train_env.norm_obs
    eval_env.norm_reward = False
    eval_env.clip_obs = train_env.clip_obs
    eval_env.clip_reward = train_env.clip_reward

    # ---- 모델 로드 (env 지정 중요) ----
    print(f"Loading checkpoint: {ckpt_path}")
    model = SAC.load(str(ckpt_path), env=train_env, tensorboard_log=str(tb_dir), verbose=1)

    # (선택) 학습률 변경
    if args.lr is not None:
        # actor/critic/entropy 옵티마 모두 갱신
        for opt in [model.actor.optimizer, model.critic.optimizer, model.critic_target.optimizer] if hasattr(model.critic_target, "optimizer") else [model.actor.optimizer, model.critic.optimizer]:
            for pg in opt.param_groups:
                pg["lr"] = args.lr
        if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
            for pg in model.ent_coef_optimizer.param_groups:
                pg["lr"] = args.lr
        print(f"Set new learning rate = {args.lr}")

    # (선택) 리플레이 버퍼 로드
    if args.load_rb and rb_path.exists():
        try:
            model.load_replay_buffer(str(rb_path))
            print(f"Replay buffer loaded: {rb_path}")
        except Exception as e:
            print(f"Replay buffer load failed: {e}")

    # ---- 콜백: 체크포인트/평가 ----
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="sac_step",
        save_replay_buffer=True  # ← 다음 이어학습 위해 버퍼도 저장
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    # ---- 이어서 학습 (스텝 카운트 연속) ----
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=False,   # ★ 중요: 이어서
        tb_log_name="SAC_continue"   # 텐서보드 런 이름(원하면 변경)
    )

    # 최종 저장
    model.save(str(ckpt_dir / "final_model"))
    train_env.save(str(ckpt_dir / "vecnorm.pkl"))
    print(f"[Saved] {run_dir}")
