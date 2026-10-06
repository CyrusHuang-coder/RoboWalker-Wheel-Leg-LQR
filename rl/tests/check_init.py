import mujoco
from test_extended_run import extract_sensors

with open('rl/terrain/wheel_leg_extended_terrain.xml', 'r', encoding='utf-8') as f:
    xml_str = f.read().replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')
m = mujoco.MjModel.from_xml_string(xml_str)
d = mujoco.MjData(m)

d.qpos[0] = 0.08
d.qpos[1] = 0.0
d.qpos[2] = 0.060
for _ in range(80): mujoco.mj_step(m, d)

s = extract_sensors(m, d)
lw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_wheel')
rw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_wheel')
print('s[\"y_pos\"]:', s['y_pos'])
print('lw_y:', d.xpos[lw_id][1])
print('rw_y:', d.xpos[rw_id][1])
print('mid_wheels_y:', 0.5 * (d.xpos[lw_id][1] + d.xpos[rw_id][1]))
print('difference s[\"y_pos\"] - mid_wheels_y:', s['y_pos'] - 0.5 * (d.xpos[lw_id][1] + d.xpos[rw_id][1]))
