import mujoco
with open('rl/terrain/wheel_leg_terrain.xml', 'r', encoding='utf-8') as f:
    xml_str = f.read().replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')
m = mujoco.MjModel.from_xml_string(xml_str)
d = mujoco.MjData(m)
d.qpos[2] = 0.060
for _ in range(50):
    mujoco.mj_step(m, d)
d.actuator('left_wheel_motor').ctrl[0] = 0.02
d.actuator('right_wheel_motor').ctrl[0] = 0.02
for _ in range(50):
    mujoco.mj_step(m, d)
print('wheel vel:', d.sensor('left_wheel_vel').data[0], 'forward vel:', d.sensor('body_linvel').data[0])

