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

for kp in [0.015, 0.02, 0.025, 0.03, 0.04]:
    for kd in [0.0005, 0.001, 0.0015, 0.002]:
        for kx in [0.0, 0.001, 0.003, 0.005]:
            for kdx in [0.0, 0.001, 0.002]:
                for ki in [0.0, 0.01, 0.03]:
                    mujoco.mj_resetData(m, d)
                    d.qpos[2] = 0.046236
                    d.qpos[3] = 1.0
                    pitch_int = 0.0
                    survived = 0
                    
                    for step in range(3000):
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
                        
                        pitch_int += pitch * 0.001
                        pitch_int = np.clip(pitch_int, -0.05, 0.05)
                        
                        u_w = (kp * pitch + kd * dpitch + ki * pitch_int - kx * x - kdx * dx)
                        d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
                        d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
                        
                        mujoco.mj_step(m, d)
                        if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
                            break
                        survived += 1
                        
                    if survived > best_steps:
                        best_steps = survived
                        best_params = (kp, kd, kx, kdx, ki)
                        print(f"New best: {survived} steps with kp={kp}, kd={kd}, kx={kx}, kdx={kdx}, ki={ki}")
                    if survived >= 3000:
                        print(f"!!! SUCCESS: 3000 steps reached with {best_params} !!!")
                        break
                if best_steps >= 3000:
                    break
            if best_steps >= 3000:
                break
        if best_steps >= 3000:
            break
    if best_steps >= 3000:
        break

print(f"\nFinal Best: {best_steps} steps, params: {best_params}")
