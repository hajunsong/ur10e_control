# scripts/eval_rl.py
import os, yaml, numpy as np, imageio.v2 as imageio
from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz
from stable_baselines3 import SAC

if __name__ == "__main__":
    with open("config/sim.yaml","r") as f:
        cfg = yaml.safe_load(f)

    p_goal = np.array(cfg["P2"]["pos"], dtype=float)
    q_goal_xyzw = wxyz_to_xyzw(cfg["P2"]["quat"])
    q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)

    task = RLTaskCfg(
        target_pos=p_goal,
        target_quat_wxyz=q_goal_wxyz,
        episode_time=cfg["demo"]["move_duration"]+cfg["demo"]["hold_duration"],
        ctrl_hz=cfg["control_hz"],
        action_scale=np.array(cfg["ctrl"]["torque_limit"], dtype=float)*0.5
    )

    # 렌더 켜고 비디오 저장
    os.environ.setdefault("MUJOCO_GL","osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM","osmesa")
    vw = imageio.get_writer("ur10e_rl_eval.mp4", fps=cfg["render_hz"])
    env = UR10eRLEnv(
        xml_path=cfg["mujoco_xml"], end_effector_site=cfg["end_effector_site"],
        q0=np.array(cfg["q0"], dtype=float),
        torque_limit=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
        task_cfg=task, render=True, width=480, height=640
    )

    model = SAC.load("checkpoints/ur10e_sac")

    obs, info = env.reset()
    done = False
    t = 0
    while True:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        frame = env.render()
        if frame is not None:
            vw.append_data(frame)
        t += 1
        if term or trunc:
            break
    vw.close()
    env.close()
