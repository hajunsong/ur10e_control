import numpy as np
import mujoco
from utils.math_utils import quat_error
from scipy.spatial.transform import Rotation as R


class DLSIK:
    def __init__(self, model, site_name, cfg):
        self.model = model
        self.data = mujoco.MjData(model)  # 🔹 IK 전용 데이터 버퍼
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert self.site_id >= 0, f"Site '{site_name}' not found"
        self.max_iters = cfg['max_iters']
        self.pos_tol = cfg['pos_tol']
        self.rot_tol = cfg['rot_tol']
        self.step_size = cfg['step_size']
        self.damping = cfg['damping']
        self.dt_limit = cfg['dt_limit']
        self._Jp = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))


    def get_site_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        # site_xmat은 길이 9의 row-major 벡터이므로 3x3으로 reshape
        mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        # scipy는 [x,y,z,w] 순서를 반환
        quat_xyzw = R.from_matrix(mat).as_quat().copy()
        return pos, quat_xyzw


    def step(self, q, target_pos, target_quat_xyzw):
        # q를 data.qpos에 반영하여 forward kinematics
        self.data.qpos[:self.model.nq] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        cur_pos, cur_quat_xyzw = self.get_site_pose()
        pos_err = target_pos - cur_pos
        rot_err = quat_error(cur_quat_xyzw, target_quat_xyzw) # 3x1

        # 종료 조건
        if np.linalg.norm(pos_err) < self.pos_tol and np.linalg.norm(rot_err) < self.rot_tol:
            return q, True

        # 자코비언 계산 (site wrt q)
        mujoco.mj_jacSite(self.model, self.data, self._Jp, self._Jr, self.site_id)
        J = np.vstack([self._Jp, self._Jr]) # 6 x nv

        err6 = np.hstack([pos_err, rot_err]) # 6

        # DLS: dq = J^T (J J^T + λ^2 I)^-1 * e
        JJt = J @ J.T
        lam2I = self.damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt + lam2I, err6)
        dq = np.clip(self.step_size * dq, -self.dt_limit, self.dt_limit)

        q_new = q + dq[:self.model.nq]
        return q_new, False


    def solve(self, q_init, target_pos, target_quat_xyzw):
        q = q_init.copy()
        for _ in range(self.max_iters):
            q, done = self.step(q, target_pos, target_quat_xyzw)
            if done:
                break
        return q