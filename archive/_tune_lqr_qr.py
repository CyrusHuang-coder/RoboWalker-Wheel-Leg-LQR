import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LQR计算代码'))
import calculate
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('wheel_leg.xml')
d = mujoco.MjData(m)

# Find Q and R that stabilize the system:
# states: [x, dx, y, dy, tl, dtl, tr, dtr, f, df]
# We want gentle control on hip, solid control on wheels and pitch
best_steps = 0
best_qr = None

for r_wheel in [1e4, 5e4, 1e5, 5e5]:
    for r_hip in [1e5, 5e5, 1e6, 5e6]:
        for q_pitch in [50.0, 100.0, 200.0]:
            for q_x in [0.1, 0.5, 1.0]:
                q = [q_x, q_x*0.1, 1.0, 0.1, 10.0, 1.0, 10.0, 1.0, q_pitch, q_pitch*0.1]
                r = [r_wheel, r_hip, r_wheel, r_hip]
                try:
                    K = calculate.calculate(0.04045, 0.04045, q=q, r=r)
                except Exception:
                    continue
                
                mujoco.mj_resetData(m, d)
                d.qpos[2] = 0.046236
                d.qpos[3] = 1.0
                d.qpos[7] = -0.0042
                d.qpos[9] = -0.0042
                
                mat = np.zeros(9)
                survived = 0
                
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
                    
                    tl = pitch - (q_hip_l - (-0.0042))
                    dtl = dpitch - dq_hip_l
                    tr = pitch - (q_hip_r - (-0.0042))
                    dtr = dpitch - dq_hip_r
                    
                    X = np.array([x, dx, yaw, dyaw, tl, dtl, tr, dtr, pitch - 0.00157, dpitch])
                    
                    # ctrl = - K @ X
                    # actuators: [left_hip, left_wheel, right_hip, right_wheel]
                    # LQR inputs: [Tlw, Tll, Trw, Trl]
                    # left_wheel -> U[0], left_hip -> U[1], right_wheel -> U[2], right_hip -> U[3]
                    U = - K @ X
                    
                    d.ctrl[0] = np.clip(U[1], -0.008, 0.008)
                    d.ctrl[1] = np.clip(U[0], -0.04, 0.04)
                    d.ctrl[2] = np.clip(U[3], -0.008, 0.008)
                    d.ctrl[3] = np.clip(U[2], -0.04, 0.04)
                    
                    mujoco.mj_step(m, d)
                    if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
                        break
                    survived += 1
                
                if survived > best_steps:
                    best_steps = survived
                    best_qr = (r_wheel, r_hip, q_pitch, q_x)
                    print(f"New best: {survived} steps with rw={r_wheel}, rh={r_hip}, qp={q_pitch}, qx={q_x}")
                if survived >= 5000:
                    print(f">>> FULL 5000 STEPS SUCCESS! {best_qr} <<<")
                    break
            if best_steps >= 5000:
                break
        if best_steps >= 5000:
            break
    if best_steps >= 5000:
        break

print(f"\nGrid Search Finished. Best: {best_steps} steps, config: {best_qr}")
