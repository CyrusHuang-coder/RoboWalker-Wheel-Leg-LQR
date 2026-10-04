import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('wheel_leg.xml')
d = mujoco.MjData(m)
mat = np.zeros(9)

for K_theta in [0.07, 0.08, 0.09, 0.10, 0.12, 0.15]:
    for K_omega in [0.002, 0.003, 0.004, 0.005]:
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.046236
        d.qpos[3] = 1.0
        d.qpos[7] = -0.0042
        d.qpos[9] = -0.0042
        
        dpitch_filtered = 0.0
        survived = 0
        
        for step in range(10000):
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
            
            d.ctrl[0] = np.clip(- 0.08 * (q_hip_l - (-0.0042)) - 0.002 * dq_hip_l, -0.008, 0.008)
            d.ctrl[2] = np.clip(- 0.08 * (q_hip_r - (-0.0042)) - 0.002 * dq_hip_r, -0.008, 0.008)
            
            u_w = K_theta * (pitch - 0.00157) + K_omega * dpitch_filtered - 0.0005 * x - 0.0005 * dx
            d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
            d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
            
            mujoco.mj_step(m, d)
            if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
                break
            survived += 1
        
        print(f"K_theta={K_theta:5.2f}, K_omega={K_omega:6.4f} -> survived {survived:5d} steps")
        if survived >= 10000:
            print(f">>> BINGO! 10000 STEPS (10.0 SECONDS) STABLE! <<<")
            break
    if survived >= 10000:
        break
