import os
import sys
import numpy as np
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from prior_controller import PriorController
from test_mega_terrain import build_mega_hfield, get_mega_track_reference
from test_extended_run import extract_sensors

# Load genuine terrain XML
xml_path = "rl/terrain/wheel_leg_terrain.xml"
with open(xml_path, 'r', encoding='utf-8') as f:
    xml_str = f.read().replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')

# Replace the hfield with 10.5m hfield
# size="5.25 0.5 0.045 0.01", pos="5.25 0.0 0.0"
xml_str = xml_str.replace('<hfield name="terrain_hfield" nrow="128" ncol="1024" size="2.0 0.5 0.04 0.01"/>',
                          '<hfield name="terrain_hfield" nrow="256" ncol="2560" size="5.25 0.5 0.045 0.01"/>')
xml_str = xml_str.replace('<geom name="terrain" type="hfield" hfield="terrain_hfield" pos="2.0 0.0 0.0" material="grid"/>',
                          '<geom name="terrain" type="hfield" hfield="terrain_hfield" pos="5.25 0.0 0.0" material="grid"/>')
xml_str = xml_str.replace('<geom name="ground_sub" type="plane" size="5 2 0.05" pos="2.0 0.0 -0.01" rgba="0.08 0.08 0.1 1"/>',
                          '<geom name="ground_sub" type="plane" size="11 3 0.05" pos="5.25 0.0 -0.002" rgba="0.08 0.08 0.1 1"/>')

model = mujoco.MjModel.from_xml_string(xml_str)
Z, X, Y = build_mega_hfield(x_len=10.5, max_elevation=0.045)
model.hfield_data[:] = (Z / 0.045).astype(np.float32).ravel()

data = mujoco.MjData(model)
ctrl = PriorController()

data.qpos[0] = 0.08
data.qpos[1] = -0.0175
data.qpos[2] = 0.058

for _ in range(80):
    mujoco.mj_step(model, data)

s = extract_sensors(model, data)
ctrl.reset(current_x=s['x_pos'], current_yaw=0.0, current_y=-0.0175)
v_target = 0.20
ctrl.set_target_velocity(v_target)

print(f"Initial state: x={s['x_pos']:.3f}, y={s['y_pos']:.3f}, z={s['z_pos']:.3f}, roll={np.degrees(s['roll']):.2f}°, pitch={np.degrees(s['pitch']):.2f}°")

dt = 0.001
for step in range(15000): # 15s
    t = step * dt
    s = extract_sensors(model, data)
    y_ref, yaw_ref = get_mega_track_reference(s['x_pos'], -0.0175)
    ctrl.target_y = y_ref
    ctrl.target_yaw = yaw_ref

    cmd = ctrl.compute(s, dt=dt, enable_compliance=False)
    data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
    data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
    data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
    data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
    mujoco.mj_step(model, data)

    if step % 500 == 0:
        print(f"t={t:4.2f}s | x={s['x_pos']:.3f}, y={s['y_pos']:.3f}, z={s['z_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")

    if abs(s['roll']) > 0.8 or abs(s['pitch']) > 0.8:
        print(f"FELL OVER at t={t:4.2f}s! x={s['x_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")
        break

