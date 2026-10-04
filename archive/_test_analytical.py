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

# Theoretical gains derived from dynamics:
# J * theta_ddot = 0.0706 * theta - 12 * tau_w
kp = 0.018
kd = 0.0012
kx = 0.0015
kdx = 0.0012
ki = 0.01

mat = np.zeros(9)
pitch_int = 0.0

print("Testing theoretically derived gains for 10000 steps (10 seconds)...")
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
    
    # Hip joint PD (holds leg aligned with body)
    d.ctrl[0] = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
    d.ctrl[2] = np.clip(- 0.3 * q_hip_r - 0.01 * dq_hip_r, -0.05, 0.05)
    
    pitch_int += pitch * 0.001
    pitch_int = np.clip(pitch_int, -0.05, 0.05)
    
    u_w = (kp * pitch + kd * dpitch + ki * pitch_int - kx * x - kdx * dx)
    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
    
    mujoco.mj_step(m, d)
    
    if step % 1000 == 0:
        print(f"step {step:5d} | pitch={pitch*180/np.pi:+6.2f} deg | dpitch={dpitch:+6.2f} | u_w={u_w:+7.4f} | x={x:+6.3f} m | z={d.qpos[2]:.5f}")
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg, z={d.qpos[2]:.4f}")
        break
else:
    print(f"\n=======================================================")
    print(f">>> BINGO! PERFECT 10-SECOND STABLE SELF-BALANCING! <<<")
    print(f"=======================================================")
