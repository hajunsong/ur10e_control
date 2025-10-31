# scripts/compare_results.py
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def wxyz_to_xyzw(qwxyz):
    q = np.asarray(qwxyz)
    return np.array([q[...,1], q[...,2], q[...,3], q[...,0]]).T if q.ndim>1 else np.array([q[1],q[2],q[3],q[0]])

def quat_err_deg(q_cur_wxyz, q_des_wxyz):
    """
    q_cur_wxyz, q_des_wxyz: shape (N,4) or (4,)
    반환: 각도 오차 [deg], shape (N,)
    """
    qcur_xyzw = wxyz_to_xyzw(q_cur_wxyz)
    qdes_xyzw = wxyz_to_xyzw(q_des_wxyz)
    if qcur_xyzw.ndim == 1:
        qcur_xyzw = qcur_xyzw[None, :]
        qdes_xyzw = qdes_xyzw[None, :]
    # scipy는 [x,y,z,w]
    r_rel = R.from_quat(qdes_xyzw) * R.from_quat(qcur_xyzw).inv()
    ang = np.linalg.norm(r_rel.as_rotvec(), axis=1) * 180.0/np.pi
    return ang

def load_log(path):
    data = np.load(path, allow_pickle=True)
    # 필수 키 체크 및 호환 처리
    t           = data["t"]
    x           = data["x"]                 # (N,3) EE position current
    x_des       = data["x_des"]             # (N,3) EE position target
    # 자세는 wxyz 컨벤션 사용 (이미 로그가 wxyz로 저장되어 있다고 가정)
    xquat       = data["xquat"]             # (N,4) current wxyz
    xquat_des   = data["xquat_des"]         # (N,4) target wxyz
    # RPY가 있으면 활용(없어도 계산 가능)
    x_rpy_deg        = data["x_rpy_deg"]        if "x_rpy_deg" in data else None
    x_rpy_deg_des    = data["x_rpy_deg_des"]    if "x_rpy_deg_des" in data else None

    # 오차 계산
    pos_err = np.linalg.norm(x_des - x, axis=1)
    rot_err = quat_err_deg(xquat, xquat_des)

    return dict(
        t=t, x=x, x_des=x_des,
        xquat=xquat, xquat_des=xquat_des,
        x_rpy_deg=x_rpy_deg, x_rpy_deg_des=x_rpy_deg_des,
        pos_err=pos_err, rot_err=rot_err
    )

def summarize(name, log):
    t = log["t"]
    pos_err = log["pos_err"]
    rot_err = log["rot_err"]
    # 지표
    final_pos = pos_err[-1]
    final_rot = rot_err[-1]
    rmse_pos  = np.sqrt(np.mean(pos_err**2))
    rmse_rot  = np.sqrt(np.mean(rot_err**2))
    max_pos   = np.max(pos_err)
    max_rot   = np.max(rot_err)

    print(f"\n[{name}]")
    print(f"  Samples            : {len(t)}")
    print(f"  Final pos err      : {final_pos*1000:7.3f} mm")
    print(f"  Final rot err      : {final_rot:7.3f} deg")
    print(f"  RMSE pos err       : {rmse_pos*1000:7.3f} mm")
    print(f"  RMSE rot err       : {rmse_rot:7.3f} deg")
    print(f"  Max pos err        : {max_pos*1000:7.3f} mm")
    print(f"  Max rot err        : {max_rot:7.3f} deg")

def plot_positions(logA, logB, nameA, nameB, outdir, prefix):
    tA, tB = logA["t"], logB["t"]
    xA, xB = logA["x"], logB["x"]
    xdA, xdB = logA["x_des"], logB["x_des"]

    labels = ['x','y','z']
    plt.figure(figsize=(10,6))
    for i, lab in enumerate(labels):
        # current
        lA, = plt.plot(tA, xA[:,i], label=f"{lab} current ({nameA})")
        color = lA.get_color()
        # target (A) - dashed same color
        # plt.plot(tA, xdA[:,i], ls='--', color=color, label=f"{lab} target ({nameA})")

        # current B
        lB, = plt.plot(tB, xB[:,i], label=f"{lab} current ({nameB})")
        colorB = lB.get_color()
        # target (B)
        plt.plot(tB, xdB[:,i], ls='--', color=colorB, label=f"{lab} target")

    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("EE Position Tracking (IK vs RL)")
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.grid()
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{prefix}_position.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {path}")

def plot_rpy(logA, logB, nameA, nameB, outdir, prefix):
    # RPY가 로그에 없으면 쿼터니언에서 계산
    def quat_to_rpy_deg(q_wxyz):
        q_xyzw = wxyz_to_xyzw(q_wxyz)
        r = R.from_quat(q_xyzw)
        return r.as_euler('xyz', degrees=True)

    tA, tB = logA["t"], logB["t"]
    if logA["x_rpy_deg"] is None:
        rpyA = quat_to_rpy_deg(logA["xquat"])
        rpyA_des = quat_to_rpy_deg(logA["xquat_des"])
    else:
        rpyA = logA["x_rpy_deg"]
        rpyA_des = logA["x_rpy_deg_des"]

    if logB["x_rpy_deg"] is None:
        rpyB = quat_to_rpy_deg(logB["xquat"])
        rpyB_des = quat_to_rpy_deg(logB["xquat_des"])
    else:
        rpyB = logB["x_rpy_deg"]
        rpyB_des = logB["x_rpy_deg_des"]

    labels = ['roll','pitch','yaw']
    plt.figure(figsize=(10,6))
    for i, lab in enumerate(labels):
        lA, = plt.plot(tA, rpyA[:,i], label=f"{lab} current ({nameA})")
        color = lA.get_color()
        # plt.plot(tA, rpyA_des[:,i], ls='--', color=color, label=f"{lab} target ({nameA})")

        lB, = plt.plot(tB, rpyB[:,i], label=f"{lab} current ({nameB})")
        colorB = lB.get_color()
        plt.plot(tB, rpyB_des[:,i], ls='--', color=colorB, label=f"{lab} target")

    plt.xlabel("time [s]")
    plt.ylabel("RPY [deg]")
    plt.title("EE Orientation (RPY) Tracking (IK vs RL)")
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.grid()
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{prefix}_rpy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', bbox_extra_artists=[])
    print(f"Saved figure: {path}")

def plot_errors(logA, logB, nameA, nameB, outdir, prefix):
    tA, tB = logA["t"], logB["t"]
    posA, posB = logA["pos_err"], logB["pos_err"]
    rotA, rotB = logA["rot_err"], logB["rot_err"]

    plt.figure(figsize=(10,6))
    l1, = plt.plot(tA, posA*1000.0, label=f"pos err (mm) {nameA}")
    c1 = l1.get_color()
    plt.plot(tA, rotA, ls='--', color=c1, label=f"rot err (deg) {nameA}")

    l2, = plt.plot(tB, posB*1000.0, label=f"pos err (mm) {nameB}")
    c2 = l2.get_color()
    plt.plot(tB, rotB, ls='--', color=c2, label=f"rot err (deg) {nameB}")

    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title("Position / Orientation Error (IK vs RL)")
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.grid()
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{prefix}_errors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="logs/run1.npz", help="log A (e.g., run_demo)")
    parser.add_argument("--b", type=str, default="logs/eval_rl_run1.npz", help="log B (e.g., eval_rl)")
    parser.add_argument("--nameA", type=str, default="IK")
    parser.add_argument("--nameB", type=str, default="RL")
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--prefix", type=str, default="compare")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    logA = load_log(args.a)
    logB = load_log(args.b)

    # 터미널 요약
    summarize(args.nameA, logA)
    summarize(args.nameB, logB)

    # 그래프
    plot_positions(logA, logB, args.nameA, args.nameB, args.outdir, args.prefix)
    plot_rpy(logA, logB, args.nameA, args.nameB, args.outdir, args.prefix)
    plot_errors(logA, logB, args.nameA, args.nameB, args.outdir, args.prefix)

    plt.show()

if __name__ == "__main__":
    main()
