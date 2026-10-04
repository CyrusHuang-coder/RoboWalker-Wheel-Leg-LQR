import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_no_leg = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')

m = mujoco.MjModel.from_xml_string(xml_no_leg)
d = mujoco.MjData(m)
mat = np.zeros(9)

best_steps = 0
best_params = None

for kp in [0.05, 0.07, 0.09, 0.12, 0.15]:
    for kd in [0.002, 0.003, 0.005, 0.008]:
        for kx in [0.005, 0.01, 0.02, 0.03]:
            for kdx in [0.003, 0.005, 0.01]:
                mujoco.mj_resetData(m, d)
                d.qpos[2] = 0.046236
                d.qpos[3] = 1.0
                pitch_int = 0.0
                survived = 0
                
                for step in range(10000):
                    quat = d.sensor('body_quat').data
                    mujoco.mju_quat2Mat(mat, quat)
                    R_mat = mat.reshape(3, 3)
                    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
                    dpitch = d.sensor('body_angvel').data[1]
                    
                    x = d.qpos[0]
                    dx = d.qvel[0]
                    
                    q_hip_l = d.sensor('left_hip_pos').data[0]
                    dq_hip_l = d.sensor('left_hip_vel').data[0]
                    q_hip_r = d.sensor('right_hip_pos').data[0]
                    dq_hip_r = d.sensor('right_hip_vel').data[0]
                    
                    d.ctrl[0] = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
                    d.ctrl[2] = np.clip(- 0.3 * q_hip_r - 0.01 * dq_hip_r, -0.05, 0.05)
                    
                    # Note: in inverted pendulum, to pull back from x > 0:
                    # u_w = kp * pitch + kd * dpitch + kx * x + kdx * dx
                    # or minus? Let's check!
                    u_w = (kp * pitch + kd * dpitch + kx * x + kdx * dx)
                    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
                    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
                    
                    mujoco.mj_step(m, d)
                    if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
                        break
                    survived += 1
                    
                if survived > best_steps:
                    best_steps = survived
                    best_params = (kp, kd, kx, kdx)
                    print(f"New best: {survived} steps with kp={kp}, kd={kd}, kx={kx}, kdx={kdx}")
                if survived >= 10000:
                    print(f">>> FULL 10000 STEPS SUCCESS! {best_params} <<<")
                    break
            if best_steps >= 10000:
                break
        if best_steps >= 10000:
            break
    if best_steps >= 10000:
        break

print(f"\nFinal Best: {best_steps} steps, params: {best_params}")
