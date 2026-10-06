"""
test_extended_run.py: 验证 7.0m 延长版多段复杂地形与三大算法动力学
"""
import os
import sys
import numpy as np
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from prior_controller import PriorController
from energy_tank import SlipCoupledEnergyTank
from sim_runner import build_single_obs, R_WHEEL, LINE_Y
from stable_baselines3 import PPO

def extract_sensors(model, data):
    s = {}
    s['x_pos'] = data.sensor('body_pos').data[0]
    s['y_pos'] = data.sensor('body_pos').data[1]
    s['z_pos'] = data.sensor('body_pos').data[2]
    q = data.sensor('body_quat').data
    w, x, y, z = q[0], q[1], q[2], q[3]
    s['roll'] = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    s['pitch'] = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    s['yaw'] = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    s['forward_vel'] = data.sensor('body_linvel').data[0]
    s['body_linvel'] = data.sensor('body_linvel').data
    s['roll_rate'] = data.sensor('body_angvel').data[0]
    s['pitch_rate'] = data.sensor('body_angvel').data[1]
    s['yaw_rate'] = data.sensor('body_angvel').data[2]
    s['left_hip_pos'] = data.sensor('left_hip_pos').data[0]
    s['left_hip_vel'] = data.sensor('left_hip_vel').data[0]
    s['right_hip_pos'] = data.sensor('right_hip_pos').data[0]
    s['right_hip_vel'] = data.sensor('right_hip_vel').data[0]
    s['left_wheel_vel'] = data.sensor('left_wheel_vel').data[0]
    s['right_wheel_vel'] = data.sensor('right_wheel_vel').data[0]
    return s

def test_run():
    xml_path = "rl/terrain/wheel_leg_extended_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    print(f"Model loaded successfully! ngeom={model.ngeom}, nbody={model.nbody}")

    model_prcc = PPO.load("rl/models/curriculum_master_model.zip")
    model_sct = PPO.load("rl/models/sct_master.zip")

    # Run each controller on the extended track
    for mode in ["lqr", "prcc", "sct"]:
        data = mujoco.MjData(model)
        ctrl = PriorController()
        tank = SlipCoupledEnergyTank(
            E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
            beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
        ) if mode == "sct" else None

        data.qpos[0] = 0.08
        data.qpos[2] = 0.060
        for _ in range(80):
            mujoco.mj_step(model, data)

        x0 = data.sensor('body_pos').data[0]
        ctrl.reset(current_x=x0, current_yaw=0.0, current_y=LINE_Y)
        v_target = 0.16
        ctrl.set_target_velocity(v_target)

        dt = model.opt.timestep
        substep = 20
        total_steps = int(45.0 / dt)  # 45 seconds

        obs_dim = 26 if mode == "sct" else 23
        obs_hist = np.zeros((3, obs_dim), dtype=np.float32)
        last_act = np.zeros(4, dtype=np.float32)

        k_val = 1.0
        dr_val = 0.0
        alpha_val = 1.0

        max_roll = 0.0
        total_slip = 0.0
        success = False
        t_finish = 45.0

        for step in range(total_steps):
            t = step * dt
            s = extract_sensors(model, data)

            if abs(s['roll']) > max_roll:
                max_roll = abs(s['roll'])

            if s['x_pos'] >= 7.00:
                success = True
                t_finish = t
                break

            # Fall check
            if abs(s['pitch']) > np.radians(45) or abs(s['roll']) > np.radians(45) or s['z_pos'] < 0.025:
                print(f"[{mode.upper()}] Failed/Fell at x={s['x_pos']:.2f}m, t={t:.2f}s! Roll={np.degrees(s['roll']):.1f}°, Pitch={np.degrees(s['pitch']):.1f}°")
                break

            # Policy update every 20ms
            if step % substep == 0:
                if mode == "prcc":
                    o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=None)
                    obs_hist = np.roll(obs_hist, -1, axis=0)
                    obs_hist[-1] = o
                    act, _ = model_prcc.predict(obs_hist.flatten(), deterministic=True)
                    last_act = act
                    act_un = act * [0.35, np.radians(2.0), 0.05, 0.05]
                    k_val = 1.0 + act_un[0]
                    dr_val = act_un[1]
                elif mode == "sct":
                    if step > 0:
                        tank.commit()
                    tf = tank.features()
                    o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=tf)
                    obs_hist = np.roll(obs_hist, -1, axis=0)
                    obs_hist[-1] = o
                    act, _ = model_sct.predict(obs_hist.flatten(), deterministic=True)
                    last_act = act
                    act_un = act * [0.35, np.radians(2.0), 0.05, 0.05]
                    k_req = 1.0 + act_un[0]
                    dr_req = act_un[1]
                    k_val, dr_val, alpha_val, _ = tank.project(ctrl, s, k_req, dr_req)

            # Controller step
            enable_c = (mode != "lqr")
            cmd = ctrl.compute(s, dt=dt, delta_roll=dr_val, k_scale=k_val, enable_compliance=enable_c)
            data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
            data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
            data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
            data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
            mujoco.mj_step(model, data)

            # Slip accounting
            sl = abs(s['left_wheel_vel'] * R_WHEEL - s['forward_vel'])
            sr = abs(s['right_wheel_vel'] * R_WHEEL - s['forward_vel'])
            slip_inst = 0.5 * (sl + sr)
            total_slip += slip_inst * dt

            if mode == "sct":
                p_slip = (abs(cmd['torque_left_wheel']) * sl + abs(cmd['torque_right_wheel']) * sr) / R_WHEEL
                tank.accumulate(ctrl, s, p_slip, dt)

        print(f"[{mode.upper():4s}] Finish: {success} | Time: {t_finish:5.2f}s | Max Roll: {np.degrees(max_roll):5.2f}° | Slip: {total_slip*1000:6.1f}mm | Final x: {s['x_pos']:.2f}m")

if __name__ == "__main__":
    test_run()

