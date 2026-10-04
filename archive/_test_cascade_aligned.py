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
d.qpos[7] = -0.0042
d.qpos[9] = -0.0042
mat = np.zeros(9)

# Cascade parameters:
kp_pitch = 0.045
kd_pitch = 0.0022
ki_pitch = 0.005

kp_pos = 0.12
kd_pos = 0.08
max_lean = 0.04

# Hip holding parameters:
q_hip_target = -0.0042
kp_hip = 0.5
kd_hip = 0.02
ki_hip = 0.5

dpitch_filtered = 0.0
pitch_int = 0.0
hip_l_int = 0.0
hip_r_int = 0.0

print("Testing Closed-Loop Cascade Balance with CoM Alignment for 10000 steps (10 seconds)...")
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
    
    # 1. Hip holding PID (holds CoM alignment)
    err_hl = q_hip_l - q_hip_target
    err_hr = q_hip_r - q_hip_target
    hip_l_int += err_hl * 0.001
    hip_r_int += err_hr * 0.001
    hip_l_int = np.clip(hip_l_int, -0.05, 0.05)
    hip_r_int = np.clip(hip_r_int, -0.05, 0.05)
    
    d.ctrl[0] = np.clip(- kp_hip * err_hl - kd_hip * dq_hip_l - ki_hip * hip_l_int, -0.05, 0.05)
    d.ctrl[2] = np.clip(- kp_hip * err_hr - kd_hip * dq_hip_r - ki_hip * hip_r_int, -0.05, 0.05)
    
    # 2. Outer loop: position & velocity -> desired lean angle
    target_pitch = np.clip(- kp_pos * x - kd_pos * dx, -max_lean, max_lean)
    
    # 3. Inner loop: pitch tracking -> wheel torque
    pitch_err = pitch - target_pitch
    pitch_int += pitch_err * 0.001
    pitch_int = np.clip(pitch_int, -0.05, 0.05)
    
    u_w = kp_pitch * pitch_err + kd_pitch * dpitch + ki_pitch * pitch_int
    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
    
    if step % 1000 == 0:
        print(f"step {step:5d} | pitch={pitch*180/np.pi:+6.2f} deg (tgt={target_pitch*180/np.pi:+5.2f}) | x={x:+6.3f} m | dx={dx:+6.3f} | u_w={u_w:+7.4f}")
    
    mujoco.mj_step(m, d)
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg, z={d.qpos[2]:.4f}")
        break
else:
    print("\n==================================================================")
    print(">>> SPECTACULAR SUCCESS! 10000 STEPS (10.0s) PERFECTLY BALANCED! <<<")
    print(f"Final state: pitch={pitch*180/np.pi:.3f} deg, x={x:.4f} m, dx={dx:.4f} m/s")
    print("==================================================================")
