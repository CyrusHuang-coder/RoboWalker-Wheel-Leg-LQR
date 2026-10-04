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

# Apply constant positive wheel torque
d.ctrl[1] = 0.005
d.ctrl[3] = 0.005
# Hold hip
d.ctrl[0] = 0.0
d.ctrl[2] = 0.0

mat = np.zeros(9)
print("Testing constant positive wheel torque (+0.005 Nm):")
for step in range(50):
    mujoco.mj_step(m, d)
    quat = d.sensor('body_quat').data
    mujoco.mju_quat2Mat(mat, quat)
    R_mat = mat.reshape(3, 3)
    pitch = np.arctan2(R_mat[0, 2], R_mat[2, 2])
    if step % 10 == 0:
        print(f"step {step:2d} | pitch={pitch*180/np.pi:+.2f} deg | x={d.qpos[0]:+.4f} m | vx={d.qvel[0]:+.4f}")
