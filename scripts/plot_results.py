import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def unwrap_deg(y_deg):
    y_rad = np.deg2rad(y_deg)
    y_unwrap = np.unwrap(y_rad, axis=0)
    return np.rad2deg(y_unwrap)

def wrap_deg(delta_deg):
    """RPY 오차를 [-180, 180) 범위로 래핑."""
    return (delta_deg + 180.0) % 360.0 - 180.0

if __name__ == "__main__":
    data = np.load("logs/eval_rl_run1.npz")
    t = data['t']
    x = data['x'] # Nx3
    xq_wxyz = data['xquat'] # Nx4 [w,x,y,z]
    x_des = data['x_des']          # Nx3
    xq_des_wxyz = data['xquat_des']# Nx4 (wxyz)
    x_rpy_deg = data['x_rpy_deg']              # Nx3
    x_rpy_deg_des = data['x_rpy_deg_des']      # Nx3

    x_rpy_plot = unwrap_deg(x_rpy_deg)
    x_rpy_des_plot = unwrap_deg(x_rpy_deg_des)

    e_pos = x - x_des                  # Nx3
    e_pos_norm = np.linalg.norm(e_pos, axis=1)

    q_cur_xyzw = np.stack([xq_wxyz[:, 1], xq_wxyz[:, 2], xq_wxyz[:, 3], xq_wxyz[:, 0]], axis=1)
    q_tar_xyzw = np.stack([xq_des_wxyz[:, 1], xq_des_wxyz[:, 2], xq_des_wxyz[:, 3], xq_des_wxyz[:, 0]], axis=1)

    # 벡터화 계산
    rot_err_deg = np.empty(len(t))
    for k in range(len(t)):
        q_err = R.from_quat(q_tar_xyzw[k]) * R.from_quat(q_cur_xyzw[k]).inv()
        rot_err_deg[k] = np.linalg.norm(q_err.as_rotvec()) * 180.0 / np.pi

    e_rpy = wrap_deg(x_rpy_plot - x_rpy_des_plot)

    # 위치
    plt.figure()
    labels = ['x', 'y', 'z']
    for i, label in enumerate(labels):
        plt.plot(t, x[:, i], label=f"{label}(current)")
        plt.plot(t, x_des[:, i], linestyle="--", label='x(target)')
    plt.xlabel('time [s]')
    plt.ylabel('EE position [m]')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8))
    plt.title('End-effector position tracking')
    plt.tight_layout()
    plt.savefig("figures/position_tracking.png", dpi=200, bbox_inches="tight")

    # 자세(RPY)
    plt.figure()
    labels = ['roll (x)', 'pitch (y)', 'yaw (z)']
    for i, label in enumerate(labels):
        plt.plot(t, x_rpy_plot[:, i], label=f"{labels[i]}(current)")
        plt.plot(t, x_rpy_des_plot[:, i], linestyle="--", label=f"{labels[i]}(target)")
    plt.xlabel('time [s]')
    plt.ylabel('RPY [deg]')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8))
    plt.title('End-effector orientation tracking (RPY)')
    plt.tight_layout()
    plt.savefig("figures/orientation_rpy_tracking.png", dpi=200, bbox_inches="tight")

    # 위치 오차
    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t, e_pos[:, i], label=f"e_{label} (m)")
    plt.xlabel("time [s]");
    plt.ylabel("Position error [m]")
    plt.title("Position error (axis-wise)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("figures/position_error_axis.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(t, e_pos_norm, label="‖e_pos‖ (m)")
    plt.xlabel("time [s]");
    plt.ylabel("Position error norm [m]")
    plt.title("Position error norm")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figures/position_error_norm.png", dpi=200, bbox_inches="tight")

    # 자세 오차
    plt.figure()
    plt.plot(t, rot_err_deg, label="orientation error (angle) [deg]")
    plt.xlabel("time [s]");
    plt.ylabel("Angle error [deg]")
    plt.title("Orientation error (quaternion angle)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figures/orientation_error_angle.png", dpi=200, bbox_inches="tight")

    labels = ['roll (x)', 'pitch (y)', 'yaw (z)']
    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t, e_rpy[:, i], label=f"e_{label} [deg]")
    plt.xlabel("time [s]");
    plt.ylabel("RPY error [deg]")
    plt.title("Orientation error (RPY axis-wise)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("figures/orientation_error_rpy.png", dpi=200, bbox_inches="tight")

    plt.show()