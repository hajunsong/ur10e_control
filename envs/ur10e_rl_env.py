# envs/ur10e_rl_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

@dataclass
class RLTaskCfg:
    target_pos: np.ndarray      # (3,)  - P2 위치 [m]
    target_quat_wxyz: np.ndarray# (4,)  - P2 자세 [w,x,y,z]
    episode_time: float = 4.0   # [s]
    ctrl_hz: int = 250
    action_scale: np.ndarray = None  # (6,) 토크 스케일
    pos_w: float = 3.0          # 위치 오차 가중치
    rot_w: float = 2.0          # 회전각 오차 가중치
    torque_w: float = 1e-4      # 토크 패널티
    smooth_w: float = 5e-5      # 액션 변화율 패널티
    success_pos_tol: float = 2e-3   # [m]
    success_rot_tol_deg: float = 1.0# [deg]

class UR10eRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, xml_path, end_effector_site, q0, torque_limit, task_cfg: RLTaskCfg,
                 render=False, width=640, height=480):
        super().__init__()
        assert os.path.exists(xml_path), f"XML not found: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)  # FK용 버퍼
        self.ee_site = end_effector_site
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        assert self.ee_id >= 0
        self.q0 = np.asarray(q0, dtype=float)
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.dt = self.model.opt.timestep
        self.task = task_cfg
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        # self.max_steps = int(np.ceil(self.task.episode_time / self.dt))
        # self.ctrl_every = max(1, int(round(self.task.ctrl_hz * self.dt)))  # 보통 1
        self.max_steps = int(np.ceil(self.task.episode_time * self.task.ctrl_hz))  # 제어 스텝 수
        self.ctrl_every = max(1, int(round(1.0 / (self.task.ctrl_hz * self.dt))))  # 제어 1회당 물리 스텝 수

        # 렌더 옵션
        self.render_on = render
        self.renderer = mujoco.Renderer(self.model, width, height) if render else None

        # 액션: 토크 [-1,1] → scale → clip by torque_limit
        if self.task.action_scale is None:
            self.task.action_scale = 0.5 * self.torque_limit
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

        # 관측: [q(6), qd(6), pos_err(3), rot_err_axis(3), (선택) 목표까지의 거리/각]
        high = np.inf * np.ones(6 + 6 + 3 + 3, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._step = 0
        self._prev_u = np.zeros(self.nu, dtype=float)

        # 목표 쿼터니언 xyzw로 변환(계산편의)
        self.target_xyzw = np.array([self.task.target_quat_wxyz[1],
                                     self.task.target_quat_wxyz[2],
                                     self.task.target_quat_wxyz[3],
                                     self.task.target_quat_wxyz[0]], dtype=float)

    # --- 유틸 ---
    def _site_pose(self, data):
        p = data.site_xpos[self.ee_id].copy()
        mat = data.site_xmat[self.ee_id].reshape(3,3).copy()
        q_xyzw = R.from_matrix(mat).as_quat()
        return p, q_xyzw

    def _rot_err_vec(self, q_cur_xyzw, q_tar_xyzw):
        # 회전벡터(라디안)
        qc = R.from_quat(q_cur_xyzw)
        qt = R.from_quat(q_tar_xyzw)
        rv = (qt * qc.inv()).as_rotvec()
        return rv

    def _obs(self):
        q = self.data.qpos[:self.nu].copy()
        qd = self.data.qvel[:self.nu].copy()
        p_cur, q_cur_xyzw = self._site_pose(self.data)
        p_err = self.task.target_pos - p_cur
        r_err = self._rot_err_vec(q_cur_xyzw, self.target_xyzw)
        return np.concatenate([q, qd, p_err, r_err]).astype(np.float32)

    def _reward(self, u):
        p_cur, q_cur_xyzw = self._site_pose(self.data)
        
        p_err_vec = self.task.target_pos - p_cur
        pos_err = float(np.linalg.norm(p_err_vec))
        ang_deg = float(np.linalg.norm(self._rot_err_vec(q_cur_xyzw, self.target_xyzw)) * 180.0/np.pi)

        # 🔹 오차 "감소량" 보상 (감소하면 +)
        d_pos = (self._prev_pos_err - pos_err)
        d_ang = (self._prev_ang_deg - ang_deg)

        # 스케일: m → cm, deg는 그대로
        rew_improve = self.task.pos_w * (d_pos) + self.task.rot_w * d_ang

        # 완만한 제약 (너무 크면 '가만히 있기'가 유리해짐)
        torque_cost  = self.task.torque_w * float(np.sum(u**2))
        smooth_cost  = self.task.smooth_w * float(np.sum((u - self._prev_u)**2))
        time_penalty = 1e-3  # 빨리 도달 유도

        # 성공 보너스 (크게 줘서 확실히 이득 만들기)
        bonus = 0.0
        if (pos_err < self.task.success_pos_tol and ang_deg < self.task.success_rot_tol_deg):
            bonus = 15.0

        # 각도는 라디안으로 스케일하고, 위치/자세 오차는 적당한 스케일로 정규화
        ang_rad = np.linalg.norm(self._rot_err_vec(q_cur_xyzw, self.target_xyzw))
        pos_term = - (pos_err / 0.05)**2          # 5cm 스케일 (작업공간에 맞춰 조정)
        rot_term = - (ang_rad / np.deg2rad(10))**2 # 10° 스케일
        reward_dense = self.task.pos_w * pos_term + self.task.rot_w * rot_term
        reward = rew_improve + reward_dense - torque_cost - smooth_cost - time_penalty + bonus

        # 🔹 반드시 업데이트
        self._prev_pos_err = pos_err
        self._prev_ang_deg = ang_deg

        return reward, {
            "pos_err": pos_err,
            "ang_err_deg": ang_deg,
            "d_pos": d_pos,
            "d_ang": d_ang,
            "rew_improve": rew_improve,
            "torque_cost": torque_cost,
            "smooth_cost": smooth_cost,
            "bonus": bonus
        }

    def _terminated(self):
        p_cur, q_cur_xyzw = self._site_pose(self.data)
        p_err = self.task.target_pos - p_cur
        ang_deg = np.linalg.norm(self._rot_err_vec(q_cur_xyzw, self.target_xyzw)) * 180.0/np.pi
        return (np.linalg.norm(p_err) < self.task.success_pos_tol and
                (ang_deg) < self.task.success_rot_tol_deg)

    # --- Gym API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.data.qpos[:self.nq] = self.q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._prev_u[:] = 0.0
        
        self._prev_pos_err = None
        self._prev_ang_deg = None
        p_cur, q_cur_xyzw = self._site_pose(self.data)
        p_err = self.task.target_pos - p_cur
        ang_deg = np.linalg.norm(self._rot_err_vec(q_cur_xyzw, self.target_xyzw))*180.0/np.pi
        self._prev_pos_err = float(np.linalg.norm(p_err))
        self._prev_ang_deg = float(ang_deg)

        obs = self._obs()
        info = {"target_pos": self.task.target_pos.copy(),
                "target_quat_wxyz": self.task.target_quat_wxyz.copy()}
        
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        u = np.clip(action, -1.0, 1.0) * self.task.action_scale

        qvel_backup = self.data.qvel.copy()
        self.data.qvel[:] = 0.0  # 코리올리 제외(=순수 중력)
        mujoco.mj_rne(self.model, self.data, 0, self.data.qfrc_inverse)
        self.data.qvel[:] = qvel_backup
        tau_g = self.data.qfrc_inverse[:self.nu].copy()

        u = np.clip(u + tau_g, -self.torque_limit, self.torque_limit)

        # 한 컨트롤 스텝 동안 물리 스텝
        self.data.ctrl[:] = u
        for _ in range(self.ctrl_every):
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        obs = self._obs()
        rew, terms = self._reward(u)
        terminated = self._terminated()
        truncated = (self._step >= self.max_steps)
        info = dict(terms)
        if terminated or truncated:
            info["episode"] = {
                "r": float(rew),
                "l": self._step,
                "pos_err": terms.get('pos_err'),
                "ang_err_deg": terms.get('ang_err_deg')
            }

        self._prev_u = u.copy()
        return obs, float(rew), terminated, truncated, info

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass
