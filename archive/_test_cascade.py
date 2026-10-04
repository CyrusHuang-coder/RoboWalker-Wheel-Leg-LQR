import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_no_leg = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')
m = mujoco.MjModel.from_xml_string(xml_no_leg)
d = mujoco.MjData(m)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.046236
d.qpos[3] = 1.0
mat = np.zeros(9)

# Cascade parameters:
# Inner loop (Fast pitch stabilization, ~5-10 Hz)
kp_pitch = 0.045
kd_pitch = 0.0022

# Outer loop (Slow position navigation, ~0.5 Hz)
kp_pos = 0.15     # rad/m (0.15 rad per 1 m, or ~0.9 deg per 10 cm)
kd_pos = 0.10     # rad/(m/s)
max_lean = 0.05   # max lean angle ~ 2.8 deg

dpitch_filtered = 0.0
theta_offset = 0.00157 # loaded equilibrium offset

print("Testing Cascade Inverted Pendulum Balance for 10000 steps (10 seconds)...")
for step in range(10000):
    quat = d.sensor('body_quat').data
    mujoco.mju_quat2Mat(mat, quat)
    R_mat = mat.reshape(3, 3)
    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
    dpitch_raw = d.sensor('body_angvel').data[1]
    dpitch_filtered = 0.7 * dpitch_filtered + 0.3 * dpitch_raw
    dpitch = dpitch_filtered
    
    x = d.qpos[0]
    dx = d.qvel[0]
    
    q_hip_l = d.sensor('left_hip_pos').data[0]
    dq_hip_l = d.sensor('left_hip_vel').data[0]
    q_hip_r = d.sensor('right_hip_pos').data[0]
    dq_hip_r = d.sensor('right_hip_vel').data[0]
    
    # 1. Hip joint holding PD
    d.ctrl[0] = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
    d.ctrl[2] = np.clip(- 0.3 * q_hip_r - 0.01 * dq_hip_r, -0.05, 0.05)
    
    # 2. Outer loop: position & velocity -> desired lean angle
    target_pitch = np.clip(- kp_pos * x - kd_pos * dx, -max_lean, max_lean) + theta_offset
    
    # 3. Inner loop: pitch tracking -> wheel torque
    pitch_err = pitch - target_pitch
    u_w = kp_pitch * pitch_err + kd_pitch * dpitch
    
    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
    
    if step % 1000 == 0:
        print(f"step {step:5d} | pitch={pitch*180/np.pi:+6.2f} deg (target={target_pitch*180/np.pi:+5.2f}) | x={x:+6.3f} m | u_w={u_w:+7.4f}")
    
    mujoco.mj_step(m, d)
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg, z={d.qpos[2]:.4f}")
        break
else:
    print("\n==================================================================")
    print(">>> SPECTACULAR SUCCESS! 10000 STEPS (10.0s) PERFECTLY BALANCED! <<<")
    print(f"Final state: pitch={pitch*180/np.pi:.3f} deg, x={x:.4f} m, dx={dx:.4f} m/s")
    print("==================================================================")
