import mujoco
import numpy as np

# Let's test with leg geoms non-colliding (contype=0 conaffinity=0)
with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_no_leg_collision = xml.replace('mesh="left_leg"', 'mesh="left_leg" contype="0" conaffinity="0"').replace('mesh="right_leg"', 'mesh="right_leg" contype="0" conaffinity="0"')

m = mujoco.MjModel.from_xml_string(xml_no_leg_collision)
d = mujoco.MjData(m)

# Find true contact height:
# Lower the robot slowly until contact force equals total weight (0.18 kg * 9.81 = 1.76 N)
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.049
d.qpos[3] = 1.0

# Settle down gently with high damping
for _ in range(300):
    d.qvel[:] *= 0.5
    d.ctrl[0] = - 0.3 * d.sensor('left_hip_pos').data[0]
    d.ctrl[2] = - 0.3 * d.sensor('right_hip_pos').data[0]
    mujoco.mj_step(m, d)

settled_z = d.qpos[2]
print(f"Settled equilibrium height: z = {settled_z:.6f}, pitch = {d.qpos[3]:.4f}")

# Now start balance test from settled state
mat = np.zeros(9)
kp = 0.08
kd = 0.004
kx = 0.04
kdx = 0.015

print("\nStarting 10,000 steps (10 seconds) test from settled equilibrium...")
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
    
    # Hip: hold straight
    d.ctrl[0] = np.clip(- 0.3 * q_hip_l - 0.01 * dq_hip_l, -0.05, 0.05)
    d.ctrl[2] = np.clip(- 0.3 * q_hip_r - 0.01 * dq_hip_r, -0.05, 0.05)
    
    # Wheel: inverted pendulum
    u_w = (kp * pitch + kd * dpitch + kx * x + kdx * dx)
    d.ctrl[1] = np.clip(u_w, -0.04, 0.04)
    d.ctrl[3] = np.clip(u_w, -0.04, 0.04)
    
    mujoco.mj_step(m, d)
    if step % 2000 == 0:
        print(f"step {step:5d} | pitch={pitch*180/np.pi:+6.2f} deg | x={x:+6.3f} m | z={d.qpos[2]:.5f} | ncon={d.ncon}")
    if abs(pitch) > 0.5 or d.qpos[2] < 0.02 or d.qpos[2] > 0.08:
        print(f"Fell at step {step} with pitch {pitch*180/np.pi:.2f} deg, z={d.qpos[2]:.4f}")
        break
    survived += 1
else:
    print(f"\n>>> AMAZING! PERFECTLY BALANCED FOR ALL 10000 STEPS (10.0 SECONDS)! <<<")
