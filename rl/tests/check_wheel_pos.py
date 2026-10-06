import mujoco
import numpy as np

with open('rl/terrain/wheel_leg_extended_terrain.xml', 'r', encoding='utf-8') as f:
    xml_str = f.read().replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')
m = mujoco.MjModel.from_xml_string(xml_str)
d = mujoco.MjData(m)
d.qpos[0] = 0.08
d.qpos[1] = -0.0175
d.qpos[2] = 0.058
mujoco.mj_step(m, d)
lw_id = m.body('left_wheel').id
rw_id = m.body('right_wheel').id
base_id = m.body('base_link').id
print('Base pos:', d.xpos[base_id])
print('Left wheel pos:', d.xpos[lw_id])
print('Right wheel pos:', d.xpos[rw_id])
print('Center between wheels y:', (d.xpos[lw_id][1] + d.xpos[rw_id][1]) / 2)
print('dy left from base:', d.xpos[lw_id][1] - d.xpos[base_id][1])
print('dy right from base:', d.xpos[rw_id][1] - d.xpos[base_id][1])

