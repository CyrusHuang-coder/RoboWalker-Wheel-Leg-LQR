import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LQR计算代码'))
import calculate
import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_no_leg = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')

m = mujoco.MjModel.from_xml_string(xml_no_leg)
d = mujoco.MjData(m)

# Physical Q and R:
q = [2.0, 0.5, 2.0, 0.5, 10.0, 1.0, 10.0, 1.0, 50.0, 5.0]
r = [1e4, 1e5, 1e4, 1e5]

K = calculate.calculate(0.04045, 0.04045, q=q, r=r)
print("Calculated K matrix:\n", np.round(K, 4))

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.046236
d.qpos[3] = 1.0
mat = np.zeros(9)

print("\nRunning LQR closed-loop simulation with aligned signs...")
for step in range(5000):
    quat = d.sensor('body_quat').data
    mujoco.mju_quat2Mat(mat, quat)
    R_mat = mat.reshape(3, 3)
    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
    dpitch = d.sensor('body_angvel').data[1]
    yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    dyaw = d.sensor('body_angvel').data[2]
    
    x = d.qpos[0]
    dx = d.qvel[0]
    
    q_hip_l = d.sensor('left_hip_pos').data[0]
    dq_hip_l = d.sensor('left_hip_vel').data[0]
    q_hip_r = d.sensor('right_hip_pos').data[0]
    dq_hip_r = d.sensor('right_hip_vel').data[0]
    
    # State mapping from MuJoCo frame to calculate.py frame:
    # calculate.py has +X backward or +theta backward.
    # When tilted forward in MuJoCo (pitch > 0), in calculate.py theta < 0.
    f_lqr = - pitch
    df_lqr = - dpitch
    tl_lqr = q_hip_l - pitch
    dtl_lqr = dq_hip_l - dpitch
    tr_lqr = q_hip_r - pitch
    dtr_lqr = dq_hip_r - dpitch
    x_lqr = - x
    dx_lqr = - dx
    y_lqr = - yaw
    dy_lqr = - dyaw
    
    X = np.array([x_lqr, dx_lqr, y_lqr, dy_lqr, tl_lqr, dtl_lqr, tr_lqr, dtr_lqr, f_lqr, df_lqr])
    U = - K @ X
    
    # Output torques from calculate.py:
    # Positive Tlw drives forward in calculate.py (which is backward in MuJoCo).
    # So MuJoCo ctrl_wheel = - U_lqr[wheel]
    # MuJoCo ctrl_hip = - U_lqr[hip]
    d.ctrl[0] = np.clip(- U[1], -0.05, 0.05)
    d.ctrl[1] = np.clip(- U[0], -0.04, 0.04)
    d.ctrl[2] = np.clip(- U[3], -0.05, 0.05)
    d.ctrl[3] = np.clip(- U[2], -0.04, 0.04)
    
    if step < 20:
        print(f"step {step:2d} | pitch={pitch*180/np.pi:+6.2f} | ctrl0={d.ctrl[0]:+7.4f} ctrl1={d.ctrl[1]:+7.4f} | z={d.qpos[2]:.4f}")
    
    mujoco.mj_step(m, d)
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg")
        break
else:
    print("\n>>> SUCCESS! LQR CONTROLLER BALANCED FOR 5000 STEPS (5.0s)! <<<")
