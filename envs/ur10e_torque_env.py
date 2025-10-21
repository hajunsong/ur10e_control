import os
import time
import numpy as np
import gymnasium as gym
import mujoco
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from utils.math_utils import xyzw_to_wxyz


@dataclass
class CtrlCfg:
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    use_gravity_comp: bool = True


class UR10eTorqueEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, xml_path, end_effector_site, control_hz=250, render_hz=25,
        sim_substeps=1, q0=None, ctrl_cfg: CtrlCfg=None, video_writer=None):
        super().__init__()
        assert os.path.exists(xml_path), f"XML not found: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.kin_data = mujoco.MjData(self.model)
        self.ee_site = end_effector_site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        assert self.ee_site_id >= 0, f"Site '{self.ee_site}' not found"
        self.ctrl_dt = 1.0 / control_hz
        self.render_dt = 1.0 / render_hz
        self.substeps = sim_substeps
        self.video_writer = video_writer

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        # 초기자세
        if q0 is None:
            q0 = np.zeros(self.nq)
        self.q0 = np.asarray(q0)

        # 제어기
        self.ctrl_cfg = ctrl_cfg
        assert self.nu == len(self.ctrl_cfg.kp), "kp length must match actuators"

        # 로그 버퍼
        self.log = dict(t=[], q=[], qd=[],
                        x=[], xquat=[], tau=[], x_des=[], xquat_des=[],
                        x_rpy_deg=[], x_rpy_deg_des=[])

        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            self._track_camera_name = "track" if cam_id >= 0 else None
        except Exception:
            self._track_camera_name = None

        self.reset_model()

    def reset_model(self):
        self.data.qpos[:self.nq] = self.q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_render_t = 0.0
        self._accum_time = 0.0

    # 중력보상 토크 계산: tau_bias = C(q,qd)+g(q)
    def gravity_bias(self):
        # 가속도 0
        self.data.qacc[:] = 0.0

        # 순수 중력만 보상
        qvel_bak = self.data.qvel.copy()
        self.data.qvel[:] = 0.0  # 코리올리 제외(=순수 중력)
        mujoco.mj_rne(self.model, self.data, 0, self.data.qfrc_inverse)
        tau = self.data.qfrc_inverse[:self.nu].copy()
        self.data.qvel[:] = qvel_bak

        return tau

    def pd_torque(self, q_des, qd_des=None):
        if qd_des is None:
            qd_des = np.zeros_like(q_des)
        q = self.data.qpos[:self.nu]
        qd = self.data.qvel[:self.nu]
        e = q_des - q
        ed = qd_des - qd
        tau = self.ctrl_cfg.kp * e + self.ctrl_cfg.kd * ed
        if self.ctrl_cfg.use_gravity_comp:
            tau += self.gravity_bias()
        return np.clip(tau, -self.ctrl_cfg.torque_limit, self.ctrl_cfg.torque_limit)

    def step_sim(self, tau):
        self.data.ctrl[:] = tau
        # 제어 주기 동안 시뮬 스텝
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def render_and_record(self, t):
        if self.video_writer is None:
            return
        if t - self._last_render_t + 1e-9 >= self.render_dt:
            # ← 수정: 존재하면 "track" 카메라 사용, 아니면 기본
            if self._track_camera_name is not None:
                self.renderer.update_scene(self.data, camera=self._track_camera_name)
            else:
                self.renderer.update_scene(self.data)
            img = self.renderer.render()
            self.video_writer.append_data(img)
            self._last_render_t = t

    def log_state(self, t):
        x = self.data.site_xpos[self.ee_site_id].copy()

        # MuJoCo 3.x: site_xmat 사용(길이 9의 row-major → 3x3)
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        quat_xyzw = R.from_matrix(mat).as_quat()      # [x,y,z,w]
        xq_wxyz = xyzw_to_wxyz(quat_xyzw)             # [w,x,y,z]로 변환(플롯 스크립트와 호환)

        rpy_deg = R.from_matrix(mat).as_euler('xyz', degrees=True)

        self.log['t'].append(t)
        self.log['q'].append(self.data.qpos[:self.nu].copy())
        self.log['qd'].append(self.data.qvel[:self.nu].copy())
        self.log['x'].append(x)
        self.log['xquat'].append(xq_wxyz)             # <- wxyz 저장
        self.log['tau'].append(self.data.ctrl[:self.nu].copy())
        self.log['x_rpy_deg'].append(rpy_deg)

    def run_pd_tracking(self, q_des_traj, T_total, goal):
        t = 0.0
        i = 0
        while t < T_total - 1e-9:
            q_des = q_des_traj[min(i, len(q_des_traj) - 1)]

            # 🔹 목표 pose 선택: 고정 P2가 주어졌으면 그걸 사용, 아니면 q_des의 FK
            x_des_log = goal['pos']
            xq_des_wxyz_log = goal['quat_wxyz']
            x_des, xq_des_wxyz = self.fk_pose_from_q(q_des)

            # 현재 pose
            x_cur = self.data.site_xpos[self.ee_site_id].copy()
            mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
            from scipy.spatial.transform import Rotation as R
            quat_xyzw = R.from_matrix(mat).as_quat()
            xq_cur_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # ---- 🔹 오차 계산 ----
            pos_err = np.linalg.norm(x_des_log - x_cur)
            # 쿼터니언 오차 (회전벡터 크기)
            q_cur_xyzw = np.array([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]])
            q_des_xyzw = np.array([xq_des_wxyz_log[1], xq_des_wxyz_log[2], xq_des_wxyz_log[3], xq_des_wxyz_log[0]])
            rot_err = R.from_quat(q_des_xyzw) * R.from_quat(q_cur_xyzw).inv()
            rot_angle = np.linalg.norm(rot_err.as_rotvec()) * 180 / np.pi  # deg 단위
            rpy_deg_des = R.from_quat(q_des_xyzw).as_euler('xyz', degrees=True)

            # ---- 🔹 콘솔 출력 ----
            if i % 20 == 0:
                print("=" * 90)
                print(f"t = {t:6.3f} s")
                print(f"Current Position : {x_cur[0]:8.4f}, {x_cur[1]:8.4f}, {x_cur[2]:8.4f} [m]")
                print(f"Target  Position : {x_des_log[0]:8.4f}, {x_des_log[1]:8.4f}, {x_des_log[2]:8.4f} [m]")
                print(f"Current Orientation (wxyz): "
                      f"{xq_cur_wxyz[0]:7.4f}, {xq_cur_wxyz[1]:7.4f}, {xq_cur_wxyz[2]:7.4f}, {xq_cur_wxyz[3]:7.4f}")
                print(f"Target  Orientation (wxyz): "
                      f"{xq_des_wxyz_log[0]:7.4f}, {xq_des_wxyz_log[1]:7.4f}, {xq_des_wxyz_log[2]:7.4f}, {xq_des_wxyz_log[3]:7.4f}")
                print(f"→ pos_err = {pos_err * 1000:7.3f} mm, rot_err = {rot_angle:6.2f} deg")
                print("=" * 90)

            tau = self.pd_torque(q_des)
            self.step_sim(tau)
            t += self.model.opt.timestep
            i += 1

            self.render_and_record(t)
            self.log_state(t)

            #  타겟 로깅
            self.log['x_des'].append(x_des_log)
            self.log['xquat_des'].append(xq_des_wxyz_log)
            self.log['x_rpy_deg_des'].append(rpy_deg_des)

        # numpy 변환
        for k in self.log:
            self.log[k] = np.asarray(self.log[k])
        return self.log

    def fk_pose_from_q(self, q_des):
        """별도의 MjData(kin_data)에서 q_des로 순전파 → (pos, quat_wxyz) 반환"""
        self.kin_data.qpos[:self.nq] = q_des
        self.kin_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.kin_data)

        pos = self.kin_data.site_xpos[self.ee_site_id].copy()
        mat = self.kin_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        quat_xyzw = R.from_matrix(mat).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz