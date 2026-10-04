import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_no_leg = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')
m = mujoco.MjModel.from_xml_string(xml_no_leg)
d = mujoco.MjData(m)
mat = np.zeros(9)

best_survived = 0
best_gains = None

# Grid over physically realistic parameters:
for K_theta in [0.035, 0.04, 0.045, 0.05, 0.06]:
    for K_omega in [0.0015, 0.002, 0.0025, 0.003]:
        for K_x in [0.0005, 0.001, 0.002, 0.003]:
            for K_v in [0.0005, 0.001, 0.0015, 0.002]:
                mujoco.mj_resetData(m, d)
                d.qpos[2] = 0.046236
                d.qpos[3] = 1.0
                d.qpos[7] = -0.0042
                d.qpos[9] = -0.0042
                
                dpitch_filtered = 0.0
                survived = 0
                
                for step in range(5000):
                    quat = d.sensor('body_quat').data
                    mujoco.mju_quat2Mat(mat, quat)
                    R_mat = mat.reshape(3, 3)
                    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
                    dpitch_raw = d.sensor('body_angvel').data[1]
                    dpitch_filtered = 0.7 * dpitch_filtered + 0.3 * dpitch_raw
                    
                    x = d.qpos[0]
                    dx = d.qvel[0]
                    
                    q_hip_l = d.sensor('left_hip_pos').data[0]
                    dq_hip_l = d.sensor('left_hip_vel').data[0]
                    q_hip_r = d.sensor('right_hip_pos').data[0]
                    dq_hip_r = d.sensor('right_hip_vel').data[0]
                    
                    # Hip: hold straight gently
                    d.ctrl[0] = np.clip(- 0.08 * (q_hip_l - (-0.0042)) - 0.002 * dq_hip_l, -0.008, 0.008)
                    d.ctrl[2] = np.clip(- 0.08 * (q_hip_r - (-0.0042)) - 0.002 * dq_hip_r, -0.008, 0.008)
                    
                    # Wheel: LQR state feedback
                    u_w = K_theta * (pitch - 0.00157) + K_omega * dpitch_filtered - K_x * x - K_v * dx
                    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
                    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
                    
                    mujoco.mj_step(m, d)
                    if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
                        break
                    survived += 1
                
                if survived > best_survived:
                    best_survived = survived
                    best_gains = (K_theta, K_omega, K_x, K_v)
                    print(f"New best: {survived} steps with K_theta={K_theta}, K_omega={K_omega}, K_x={K_x}, K_v={K_v}")
                if survived >= 5000:
                    print(f">>> PERFECT STABILITY! 5000 steps reached with {best_gains}! <<<")
                    break
            if best_survived >= 5000:
                break
        if best_survived >= 5000:
            break
    if best_survived >= 5000:
        break

print(f"\nSearch complete: Best survived = {best_survived}, gains = {best_gains}")
