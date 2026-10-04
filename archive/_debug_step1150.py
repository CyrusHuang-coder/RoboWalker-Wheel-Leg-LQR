import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('wheel_leg.xml')
d = mujoco.MjData(m)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.046236
d.qpos[3] = 1.0
d.qpos[7] = -0.0042
d.qpos[9] = -0.0042

K_theta = 0.06
K_omega = 0.0015
K_x = 0.0005
K_v = 0.0005

dpitch_filtered = 0.0
mat = np.zeros(9)

for step in range(1200):
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
    
    u_w = K_theta * (pitch - 0.00157) + K_omega * dpitch_filtered - K_x * x - K_v * dx
    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
    
    if step % 200 == 0 or step > 1130:
        print(f"step {step:4d} | pitch={pitch*180/np.pi:+6.2f} dp={dpitch_filtered:+6.2f} | x={x:+6.3f} dx={dx:+6.3f} | u_w={u_w:+7.4f}")
    
    mujoco.mj_step(m, d)
    if abs(pitch) > 0.4 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step}")
        break
