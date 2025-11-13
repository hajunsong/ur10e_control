# scripts/eval_rl.py (업데이트 버전)
import os
os.environ["MUJOCO_GL"] = "egl"  # WSL/헤드리스면 egl 권장
import yaml, numpy as np, imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R

from envs.ur10e_rl_env import UR10eRLEnv, RLTaskCfg
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz
from stable_baselines3 import SAC

import scripts.compare_results as result_plot

def site_pose(model, data, ee_id):
    pos = data.site_xpos[ee_id].copy()
    mat = data.site_xmat[ee_id].reshape(3, 3).copy()
    quat_xyzw = R.from_matrix(mat).as_quat()             # [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    rpy_deg = R.from_matrix(mat).as_euler('xyz', degrees=True)
    return pos, quat_wxyz, rpy_deg

if __name__ == "__main__":
    with open("config/sim.yaml","r") as f:
        cfg = yaml.safe_load(f)

    # 고정 목표(P2)
    p_goal = np.array(cfg["P2"]["pos"], dtype=float)
    q_goal_xyzw = wxyz_to_xyzw(cfg["P2"]["quat"])
    q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)  # [w,x,y,z]

    task = RLTaskCfg(
        target_pos = p_goal,
        target_quat_wxyz = q_goal_wxyz,
        episode_time = cfg["demo"]["move_duration"] + cfg["demo"]["hold_duration"],
        ctrl_hz = cfg["control_hz"],
        action_scale = np.array(cfg["ctrl"]["torque_limit"], dtype=float),
        success_pos_tol=0.001, success_rot_tol_deg=0.1,
        # VSD on
        use_vsd=cfg["vsd"]["use_vsd"],
        vsd_alpha=cfg["vsd"]["vsd_alpha"],
        Kp_vsd_pos=cfg["vsd"]["Kp_vsd_pos"],
        Kd_vsd_pos=cfg["vsd"]["Kd_vsd_pos"],
        Kp_vsd_rot=cfg["vsd"]["Kp_vsd_rot"],
        Kd_vsd_rot=cfg["vsd"]["Kd_vsd_rot"],
    )

    # 비디오 저장 (선택)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    vw = imageio.get_writer("ur10e_rl_eval_vsd.mp4", fps=cfg["render_hz"])

    # 환경/모델
    def make_eval_env():
        return UR10eRLEnv(
            xml_path=cfg["mujoco_xml"],
            end_effector_site=cfg["end_effector_site"],
            q0=np.array(cfg["q0"], dtype=float),
            torque_limit=np.array(cfg["ctrl"]["torque_limit"], dtype=float),
            task_cfg=task, render=True, width=480, height=640
        )
    base_vec = DummyVecEnv([make_eval_env])
    # 학습 시 저장한 VecNormalize 통계 사용(없다면 새로 감싸도 됨)

    RUN = "runs/20251113_044801_rl_fix"
    ckpt = f"{RUN}/checkpoints/best/best_model.zip"
    vecnorm = f"{RUN}/checkpoints/best/vecnorm.pkl"

    if os.path.exists(vecnorm):
        eval_env = VecNormalize.load(vecnorm, base_vec)
    else:
        eval_env = base_vec
    eval_env.training = False
    eval_env.norm_reward = False
    raw = eval_env.venv.envs[0]   # (Monitor가 껴있다면 raw = raw.env)

    model = SAC.load(ckpt)

    # 로그 버퍼 (plot_results.py와 동일 키)  
    log = dict(t=[], x=[], xquat=[], x_des=[], xquat_des=[], x_rpy_deg=[], x_rpy_deg_des=[])

    obs = eval_env.reset()

    import mujoco
    mujoco.mj_forward(raw.model, raw.data)

    # 목표 포즈(RPY)
    q_goal_xyzw_local = wxyz_to_xyzw(q_goal_wxyz)  # [x,y,z,w]
    rpy_goal_deg = R.from_quat(q_goal_xyzw_local).as_euler('xyz', degrees=True)

    t = 0.0
    dt = raw.dt
    step_idx = 0
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, infos = eval_env.step(act)
        rew = float(rews[0])
        done = bool(dones[0])

        if done:
            print(f"done step : {step_idx}")
            break

        # 현재 포즈
        pos, quat_wxyz, rpy_deg = site_pose(raw.model, raw.data, raw.ee_id)

        # 로그 누적 (타겟을 매 스텝 동일하게 기록)
        log['t'].append(t)
        log['x'].append(pos)
        log['xquat'].append(quat_wxyz)
        log['x_rpy_deg'].append(rpy_deg)
        log['x_des'].append(p_goal)
        log['xquat_des'].append(q_goal_wxyz)
        log['x_rpy_deg_des'].append(rpy_goal_deg)

        # 콘솔 출력(20스텝마다)
        if step_idx % 20 == 0:
            pos_err = np.linalg.norm(p_goal - pos)
            rot_err_deg = np.linalg.norm((R.from_quat(q_goal_xyzw_local) * R.from_quat(wxyz_to_xyzw(quat_wxyz)).inv()).as_rotvec()) * 180/np.pi
            print("="*86)
            print(f"t={t:6.3f}s  reward={rew:+.4f}  pos_err={pos_err*1000:6.2f} mm  rot_err={rot_err_deg:5.2f} deg")
            print(f"Cur Pos : {pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f} [m]")
            print(f"Tgt Pos : {p_goal[0]: .4f}, {p_goal[1]: .4f}, {p_goal[2]: .4f} [m]")
            print(f"Cur Ori (wxyz): {quat_wxyz[0]: .4f}, {quat_wxyz[1]: .4f}, {quat_wxyz[2]: .4f}, {quat_wxyz[3]: .4f}")
            print(f"Tgt Ori (wxyz): {q_goal_wxyz[0]: .4f}, {q_goal_wxyz[1]: .4f}, {q_goal_wxyz[2]: .4f}, {q_goal_wxyz[3]: .4f}")

        # 렌더 프레임 저장
        frame = raw.render()
        if frame is not None:
            vw.append_data(frame)

        t += dt
        step_idx += 1

    vw.close()
    eval_env.close()

    # numpy 변환 및 저장
    for k in log:
        log[k] = np.asarray(log[k])
    np.savez("logs/eval_rl_run1_vsd.npz", **log, p_goal=p_goal, q_goal_xyzw=q_goal_xyzw)

    print("Saved video to ur10e_rl_eval_vsd.mp4")
    print("Saved logs  to logs/eval_rl_run1_vsd.npz")

    result_plot.run()