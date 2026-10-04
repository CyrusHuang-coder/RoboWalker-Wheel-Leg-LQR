import mujoco
import numpy as np

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

# Make leg meshes visual only as well (contype=0 conaffinity=0)
xml_test = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')

m = mujoco.MjModel.from_xml_string(xml_test)
d = mujoco.MjData(m)

# Contact height
d.qpos[2] = 0.04904
d.qpos[3] = 1.0
mujoco.mj_step(m, d)
print('With leg contype=0: ncon =', d.ncon)
for i in range(d.ncon):
    con = d.contact[i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom1])
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom2])
    print(f'Contact {i}: {b1} vs {b2} dist={con.dist:.5f}')

# Now test the balance loop for 10000 steps!
mat = np.zeros(9)
kp = 0.07
kd = 0.003
kx = 0.03
kdx = 0.01

print("\nRunning 10000 steps balance simulation (10 seconds)...")
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
    
    # Hip: hold straight
    d.ctrl[0] = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
    d.ctrl[2] = np.clip(- 0.3 * q_hip_r - 0.01 * dq_hip_r, -0.05, 0.05)
    
    # Wheel: inverted pendulum
    u_w = (kp * pitch + kd * dpitch + kx * x + kdx * dx)
    d.ctrl[1] = np.clip(u_w, -0.03, 0.03)
    d.ctrl[3] = np.clip(u_w, -0.03, 0.03)
    
    mujoco.mj_step(m, d)
    if step % 2000 == 0:
        print(f"step {step:5d} | pitch={pitch*180/np.pi:+6.2f} deg | x={x:+6.3f} m | ncon={d.ncon}")
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg, z={d.qpos[2]:.4f}")
        break
else:
    print(">>> SUCCESS! SURVIVED ALL 10000 STEPS (10.0 FULL SECONDS)! <<<")
