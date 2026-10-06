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

modes = ["lqr", "prcc", "sct"]
results = {}

for mode in modes:
    data = mujoco.MjData(model)
    ctrl = PriorController()
    ctrl.kd_vel = 0.35
    ctrl.ki_vel = 0.08
    ctrl.max_steer_torque = 0.035

    tank = SlipCoupledEnergyTank(
        E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
        beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
    ) if mode == "sct" else None

    pol = None
    if mode == "prcc":
        pol = PPO.load("rl/models/curriculum_master_model.zip")
    elif mode == "sct":
        pol = PPO.load("rl/models/sct_master.zip")

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
    finished = False
    stopped = False

    print(f"\n=======================================================")
    print(f"  Testing Full 10.5m Course: {mode.upper()}")
    print(f"=======================================================")

    max_steps = 55000  # 55.0s
    for step in range(max_steps):
        t = step * dt
        s = extract_sensors(model, data)
        y_ref, yaw_ref = get_mega_track_reference(s['x_pos'], LINE_Y)
        ctrl.target_y = y_ref
        ctrl.target_yaw = yaw_ref

        # Smooth deceleration at finish line
        if s['x_pos'] >= 9.55 and not finished:
            v_cmd = max(0.015, v_target * (10.00 - s['x_pos']) / 0.45)
            ctrl.set_target_velocity(v_cmd)
        if s['x_pos'] >= 10.00:
            finished = True
            ctrl.set_target_velocity(0.0)
            if not stopped and abs(s['forward_vel']) < 0.03:
                stopped = True
                ctrl.target_x = s['x_pos']

        if step % substep == 0 and pol is not None:
            if not finished:
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

                dr_req = float(act[2] * 0.10)
                k_req = float(1.0 + act[3] * 0.40)

                if mode == "sct":
                    k_val, dr_val, alpha, _ = tank.project(ctrl, s, k_req, dr_req)
                else:
                    k_val, dr_val = k_req, dr_req
            else:
                k_val = 1.0
                dr_val = 0.0

        comp = (mode != "lqr") and (not finished)
        cmd = ctrl.compute(s, dt=dt, delta_roll=dr_val, k_scale=k_val, enable_compliance=comp)
        data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
        mujoco.mj_step(model, data)

        if mode == "sct" and not finished:
            sl = abs(s['left_wheel_vel'] * R_WHEEL - s['forward_vel'])
            sr = abs(s['right_wheel_vel'] * R_WHEEL - s['forward_vel'])
            p_slip = (abs(cmd['torque_left_wheel']) * sl + abs(cmd['torque_right_wheel']) * sr) / R_WHEEL
            tank.accumulate(ctrl, s, p_slip, dt)

        if step % 2500 == 0 or (finished and step % 1000 == 0):
            print(f"t={t:5.1f}s | x={s['x_pos']:.3f}, y={s['y_pos']:.3f}, z={s['z_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")

        if abs(s['roll']) > 0.8 or abs(s['pitch']) > 0.8:
            print(f"[{mode.upper()} FELL] at t={t:5.1f}s, x={s['x_pos']:.3f}, roll={np.degrees(s['roll']):+5.1f}°, pitch={np.degrees(s['pitch']):+5.1f}°, v={s['forward_vel']:.2f}")
            fell = True
            break

        if finished and stopped and t > 52.0:
            print(f"[{mode.upper()} COMPLETE] Finished & smoothly parked at x={s['x_pos']:.3f}m, t={t:.1f}s!")
            break

    results[mode] = {"fell": fell, "finished": finished, "final_x": s['x_pos']}

print("\n" + "="*60)
print("FINAL BENCHMARK SUMMARY (10.5m Mega Course):")
for m, r in results.items():
    status = "SUCCESS (PARKED)" if (not r['fell'] and r['finished']) else "FAILED"
    print(f"  {m.upper():<6}: {status} | Final Pos: {r['final_x']:.2f}m")
print("="*60)

