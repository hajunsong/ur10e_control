import os
# os.environ["MUJOCO_GL"] = "egl"  # WSL/헤드리스면 egl 권장
# os.environ["PYOPENGL_PLATFORM"] = "egl" # PyOpenGL 쪽도 OSMesa로
os.environ.setdefault("MUJOCO_GL", "osmesa")         # or "egl" (GPU가 준비되었을 때만)
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa") # PyOpenGL 쪽도 OSMesa로

import numpy as np
import yaml
import imageio.v2 as imageio
from envs.ur10e_torque_env import UR10eTorqueEnv, CtrlCfg
from controllers.ik import DLSIK
from utils.math_utils import wxyz_to_xyzw, xyzw_to_wxyz

if __name__ == "__main__":
    with open("config/sim.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    USE_FIXED_P2 = bool(cfg['demo'].get('use_fixed_p2', True))

    # 비디오 라이터
    vw = imageio.get_writer(cfg['demo']['video_path'], fps=cfg['render_hz'])

    ctrl_cfg = CtrlCfg(
        kp=np.array(cfg['ctrl']['kp'], dtype=float),
        kd=np.array(cfg['ctrl']['kd'], dtype=float),
        torque_limit=np.array(cfg['ctrl']['torque_limit'], dtype=float),
        use_gravity_comp=bool(cfg['ctrl']['use_gravity_comp'])
    )

    env = UR10eTorqueEnv(
        xml_path=cfg['mujoco_xml'],
        end_effector_site=cfg['end_effector_site'],
        control_hz=cfg['control_hz'],
        render_hz=cfg['render_hz'],
        sim_substeps=cfg['sim_substeps'],
        q0=np.array(cfg['q0'], dtype=float),
        ctrl_cfg=ctrl_cfg,
        video_writer=vw,
    )

    p_goal = np.array(cfg['P2']['pos'], dtype=float)  # [-0.85, -0.20, 0.65]
    q_goal_xyzw = wxyz_to_xyzw(cfg['P2']['quat'])  # [x,y,z,w]
    q_goal_wxyz = xyzw_to_wxyz(q_goal_xyzw)

    # 역기구학 준비
    ik = DLSIK(env.model, cfg['end_effector_site'], cfg['ik'])

    # P1(q0) → P2 IK로 q_des 계산
    q_init = np.array(cfg['q0'], dtype=float)
    q_goal = ik.solve(q_init, p_goal, q_goal_xyzw)

    dt = env.model.opt.timestep
    move_T = float(cfg['demo']['move_duration'])
    hold_T = float(cfg['demo']['hold_duration'])
    T_total = move_T + hold_T
    N_total = int(np.ceil(T_total / dt))

    if USE_FIXED_P2:
        q_traj = [q_goal.copy() for _ in range(N_total)]
    else:
        # 최소 jerk 5차 다항 보간으로 관절 궤적 생성
        def minjerk(t, T):
            s = t / T
            s = np.clip(s, 0.0, 1.0)
            return 10*s**3 - 15*s**4 + 6*s**5

        N_move = int(np.ceil(move_T / dt))
        q_traj = []
        for i in range(N_move):
            t = (i+1) * dt
            s = minjerk(t, move_T)
            q_t = (1-s) * q_init + s * q_goal
            q_traj.append(q_t)
        for _ in range(int(np.ceil(hold_T/dt))):
            q_traj.append(q_goal.copy())

    # 실행
    # log = env.run_pd_tracking(q_traj, T_total=move_T+hold_T)
    fixed_ee_goal = {'pos': p_goal, 'quat_wxyz': q_goal_wxyz} if USE_FIXED_P2 else None
    log = env.run_pd_tracking(q_traj, T_total=T_total, fixed_ee_goal=fixed_ee_goal)
    vw.close()

    # 로그 저장
    os.makedirs(os.path.dirname(cfg['demo']['log_path']), exist_ok=True)
    np.savez(cfg['demo']['log_path'], **log, p_goal=p_goal, q_goal_xyzw=q_goal_xyzw)

    print(f"Saved video to {cfg['demo']['video_path']}")
    print(f"Saved logs to {cfg['demo']['log_path']}")