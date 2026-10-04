import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_test = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')

m = mujoco.MjModel.from_xml_string(xml_test)
d = mujoco.MjData(m)

d.qpos[2] = 0.04904
d.qpos[3] = 1.0

mat = np.zeros(9)
kp = 0.07
kd = 0.003
kx = 0.03
kdx = 0.01

print("Step-by-step diagnostic:")
for step in range(100):
    quat = d.sensor('body_quat').data
    mujoco.mju_quat2Mat(mat, quat)
    R_mat = mat.reshape(3, 3)
    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
    dpitch = d.sensor('body_angvel').data[1]
    
    x = d.qpos[0]
    dx = d.qvel[0]
    z = d.qpos[2]
    dz = d.qvel[2]
    
    q_hip_l = d.sensor('left_hip_pos').data[0]
    dq_hip_l = d.sensor('left_hip_vel').data[0]
    
    u_h = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
    u_w = np.clip(kp * pitch + kd * dpitch + kx * x + kdx * dx, -0.03, 0.03)
    
    d.ctrl[0] = u_h
    d.ctrl[2] = u_h
    d.ctrl[1] = u_w
    d.ctrl[3] = u_w
    
    if step % 10 == 0 or step > 80:
        print(f"step {step:2d} | z={z:.5f} dz={dz:+.4f} | pitch={pitch*180/np.pi:+6.2f} dpitch={dpitch:+6.2f} | u_w={u_w:+.4f} | ncon={d.ncon}")
    
    mujoco.mj_step(m, d)
