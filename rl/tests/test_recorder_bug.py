import os
import sys
import numpy as np
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from prior_controller import PriorController
from energy_tank import SlipCoupledEnergyTank
from sim_runner import build_single_obs, R_WHEEL, LINE_Y
from test_mega_terrain import build_mega_hfield, get_mega_track_reference
from test_extended_run import extract_sensors
from stable_baselines3 import PPO

xml_path = "rl/terrain/wheel_leg_extended_terrain.xml"
with open(xml_path, 'r', encoding='utf-8') as f:
    xml_str = f.read().replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')

model = mujoco.MjModel.from_xml_string(xml_str)
Z, X, Y = build_mega_hfield(x_len=10.5, max_elevation=0.045)
norm_hf = (Z / 0.045).astype(np.float32)
model.hfield_data[:] = norm_hf.ravel()

# Test PRCC and SCT with nominal trained action unscaling:
# Option A: act_un = act * [0.35, np.radians(2.0), 0.05, 0.05], k = 1.0 + act_un[0], dr = act_un[1]
# Option B: dr = act[2]*0.10, k = 1.0 + act[3]*0.40

for mode in ["prcc", "sct"]:
    data = mujoco.MjData(model)
    ctrl = PriorController()
    ctrl.kd_vel = 0.35
    ctrl.ki_vel = 0.08
    ctrl.max_steer_torque = 0.035

    tank = SlipCoupledEnergyTank(
        E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
        beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
    ) if mode == "sct" else None

    pol = PPO.load("rl/models/sct_master.zip" if mode == "sct" else "rl/models/curriculum_master_model.zip")

    data.qpos[0] = 0.08
    data.qpos[1] = LINE_Y
    data.qpos[2] = 0.058
    for _ in range(80):
        mujoco.mj_step(model, data)

    s = extract_sensors(model, data)
    ctrl.reset(current_x=s['x_pos'], current_yaw=0.0, current_y=LINE_Y)
    v_target = 0.20
    ctrl.set_target_velocity(v_target)

    dt = 0.001
    substep = 20
    obs_dim = 26 if mode == "sct" else 23
    obs_hist = np.zeros((3, obs_dim), dtype=np.float32)
    last_act = np.zeros(4, dtype=np.float32)

    dr_val, k_val = 0.0, 1.0
    fell = False

    print(f"\n--- Testing {mode.upper()} ---")
    for step in range(15000): # 15s
        t = step * dt
        s = extract_sensors(model, data)
        y_ref, yaw_ref = get_mega_track_reference(s['x_pos'], LINE_Y)
        ctrl.target_y = y_ref
        ctrl.target_yaw = yaw_ref

        if step % substep == 0:
            if mode == "sct":
                if step > 0:
                    tank.commit()
                tf = tank.features()
                o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=tf)
            else:
                o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=None)
            
            obs_hist = np.roll(obs_hist, -1, axis=0)
            obs_hist[-1] = o
            act, _ = pol.predict(obs_hist.flatten(), deterministic=True)
            last_act = act

            # In wheel_leg_env: delta_roll = action[2]*0.10, k_scale = 1.0 + action[3]*0.40
            dr_req = float(act[2] * 0.10)
            k_req = float(1.0 + act[3] * 0.40)

            if mode == "sct":
                k_val, dr_val, alpha, _ = tank.project(ctrl, s, k_req, dr_req)
            else:
                k_val, dr_val = k_req, dr_req

        cmd = ctrl.compute(s, dt=dt, delta_roll=dr_val, k_scale=k_val, enable_compliance=True)
        data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
        mujoco.mj_step(model, data)

        if mode == "sct":
            sl = abs(s['left_wheel_vel'] * R_WHEEL - s['forward_vel'])
            sr = abs(s['right_wheel_vel'] * R_WHEEL - s['forward_vel'])
            p_slip = (abs(cmd['torque_left_wheel']) * sl + abs(cmd['torque_right_wheel']) * sr) / R_WHEEL
            tank.accumulate(ctrl, s, p_slip, dt)

        if step % 1000 == 0:
            print(f"t={t:4.2f}s | x={s['x_pos']:.3f}, y={s['y_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")

        if abs(s['roll']) > 0.8 or abs(s['pitch']) > 0.8:
            print(f"[{mode.upper()} FELL] at t={t:4.2f}s, x={s['x_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")
            fell = True
            break
    if not fell:
        print(f"[{mode.upper()} SUCCESS] Finished 15s smoothly at x={s['x_pos']:.2f}m!")

