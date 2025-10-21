# scripts/print_pose.py
import os
import yaml
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

def wxyz_from_site(model, data, site_id):
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].reshape(3, 3).copy()
    quat_xyzw = R.from_matrix(mat).as_quat()            # [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])  # [w,x,y,z]
    rpy_rad = R.from_matrix(mat).as_euler('xyz', degrees=False)
    rpy_deg = np.degrees(rpy_rad)
    return pos, quat_wxyz, rpy_deg

if __name__ == "__main__":
    # config에서 xml과 site 이름을 읽어온다
    with open("config/sim.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    xml_path = cfg["mujoco_xml"]
    ee_site = cfg["end_effector_site"]

    # 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    assert site_id >= 0, f"Site '{ee_site}' not found in model"

    # 질문에서 주신 관절각 (rad)
    q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=float)
    data.qpos[:model.nq] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pos, quat_wxyz, rpy_deg = wxyz_from_site(model, data, site_id)

    print("="*72)
    print("UR10e FK @ q = [0, -1.57, 1.57, -1.57, -1.57, 0]  (base frame)")
    print(f"Position [m]        : x={pos[0]: .6f}, y={pos[1]: .6f}, z={pos[2]: .6f}")
    print(f"Orientation (wxyz)  : [{quat_wxyz[0]: .6f}, {quat_wxyz[1]: .6f}, {quat_wxyz[2]: .6f}, {quat_wxyz[3]: .6f}]")
    print(f"Orientation (RPY°)  : roll={rpy_deg[0]: .3f}, pitch={rpy_deg[1]: .3f}, yaw={rpy_deg[2]: .3f}")
    print("="*72)
