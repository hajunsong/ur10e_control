import numpy as np
from scipy.spatial.transform import Rotation as R

# [w,x,y,z] ↔ [x,y,z,w] 변환
def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

# 쿼터니언 오차(기저: 현재→목표), 소각벡터 표현
# 반환: 회전벡터(라디안), 크기가 각오차
def quat_error(current_xyzw, target_xyzw):
    # q_err = q_target * inv(q_current)
    qc = R.from_quat(current_xyzw)
    qt = R.from_quat(target_xyzw)
    q_err = qt * qc.inv()
    rotvec = q_err.as_rotvec()
    return rotvec