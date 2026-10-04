import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LQR计算代码'))
import calculate
import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_smooth = xml.replace('integrator="RK4"', 'integrator="Euler"').replace(
    '<geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="left_wheel"/>',
    '<geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="left_wheel" contype="0" conaffinity="0"/>\n          <geom type="sphere" size="0.008" friction="1.5 0.01 0.0001"/>'
).replace(
    '<geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="right_wheel"/>',
    '<geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="right_wheel" contype="0" conaffinity="0"/>\n          <geom type="sphere" size="0.008" friction="1.5 0.01 0.0001"/>'
)

m = mujoco.MjModel.from_xml_string(xml_smooth)
d = mujoco.MjData(m)

l_nom = 0.04045
mat = np.zeros(9)
X_target = np.zeros(10)

best_survived = 0
best_config = None

for q_pitch in [20.0, 50.0, 100.0, 200.0]:
    for q_x in [0.5, 1.0, 2.0, 5.0]:
        for r_w in [20.0, 50.0, 100.0, 200.0]:
            for r_h in [50.0, 100.0, 500.0, 1000.0]:
                q = [q_x, q_x*0.1, 1.0, 0.1, 5.0, 0.5, 5.0, 0.5, q_pitch, q_pitch*0.1]
                r = [r_w, r_h, r_w, r_h]
                try:
                    K = calculate.calculate(l_nom, l_nom, q=q, r=r)
                except Exception:
                    continue
                
                mujoco.mj_resetData(m, d)
                d.qpos[2] = 0.049
                d.qpos[3] = 1.0
                
                survived = 0
                for step in range(3000):
                    quat = d.sensor('body_quat').data
                    mujoco.mju_quat2Mat(mat, quat)
                    R_mat = mat.reshape(3, 3)
                    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
                    pitch_vel = d.sensor('body_angvel').data[1]
                    yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
                    yaw_vel = d.sensor('body_angvel').data[2]
                    
                    x = d.qpos[0]
                    dx = d.qvel[0]
                    
                    q_hip_l = d.sensor('left_hip_pos').data[0]
                    dq_hip_l = d.sensor('left_hip_vel').data[0]
                    q_hip_r = d.sensor('right_hip_pos').data[0]
                    dq_hip_r = d.sensor('right_hip_vel').data[0]
                    
                    theta_l = pitch + q_hip_l
                    dtheta_l = pitch_vel + dq_hip_l
                    theta_r = pitch + q_hip_r
                    dtheta_r = pitch_vel + dq_hip_r
                    
                    X = np.array([x, dx, yaw, yaw_vel, theta_l, dtheta_l, theta_r, dtheta_r, pitch, pitch_vel])
                    U = -K @ (X - X_target)
                    
                    d.ctrl[0] = np.clip(U[1], -0.05, 0.05)
                    d.ctrl[1] = np.clip(U[0], -0.04, 0.04)
                    d.ctrl[2] = np.clip(U[3], -0.05, 0.05)
                    d.ctrl[3] = np.clip(U[2], -0.04, 0.04)
                    
                    mujoco.mj_step(m, d)
                    if abs(pitch) > 0.5 or d.qpos[2] < 0.02:
                        break
                    survived += 1
                    
                if survived > best_survived:
                    best_survived = survived
                    best_config = (q_pitch, q_x, r_w, r_h)
                    print(f"New Best: survived {survived} steps with q_pitch={q_pitch}, q_x={q_x}, r_w={r_w}, r_h={r_h}")
                if survived >= 3000:
                    print(f">>> PERFECT STABILITY! 3000 steps with {best_config} <<<")
                    break

print("Grid search complete. Best survived:", best_survived, "config:", best_config)
