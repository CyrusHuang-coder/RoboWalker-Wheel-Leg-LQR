import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('wheel_leg.xml')
d = mujoco.MjData(m)
d.qpos[2] = 0.0462
d.qpos[3] = 1.0

# Apply positive torque to both wheels
d.ctrl[1] = 0.01
d.ctrl[3] = 0.01

for step in range(50):
    mujoco.mj_step(m, d)

print(f"Wheel torque > 0 -> x = {d.qpos[0]:.6f}, vx = {d.qvel[0]:.6f}")
print(f"Left wheel pos = {d.sensor('left_wheel_pos').data[0]:.4f}, Right wheel pos = {d.sensor('right_wheel_pos').data[0]:.4f}")
