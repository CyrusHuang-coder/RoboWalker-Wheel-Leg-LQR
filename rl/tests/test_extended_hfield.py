"""
test_extended_hfield.py: 包含 12mm 宏观爬坡、6mm 台阶下坎与 4.5mm 波浪起伏的 7.5m 实体地形
"""
import os
import sys
import numpy as np
import mujoco

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from test_extreme_terrain import build_extreme_hfield

build_extended_hfield = build_extreme_hfield

def main():
    from test_extended_run import extract_sensors
    from prior_controller import PriorController
    from energy_tank import SlipCoupledEnergyTank
    from sim_runner import build_single_obs, R_WHEEL, LINE_Y
    from stable_baselines3 import PPO

    xml_path = 'rl/terrain/wheel_leg_extended_terrain.xml'
    m = mujoco.MjModel.from_xml_path(xml_path)
    Z, X, Y = build_extended_hfield()
    norm_hf = (Z / 0.045).astype(np.float32)
    m.hfield_data[:] = norm_hf.ravel()

    model_prcc = PPO.load('rl/models/curriculum_master_model.zip')
    model_sct = PPO.load('rl/models/sct_master.zip')

    for mode in ['lqr', 'prcc', 'sct']:
        data = mujoco.MjData(m)
        ctrl = PriorController()
        tank = SlipCoupledEnergyTank(
            E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
            beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
        ) if mode == 'sct' else None

        data.qpos[0] = 0.08
        data.qpos[2] = 0.060
        for _ in range(80):
            mujoco.mj_step(m, data)

        x0 = data.sensor('body_pos').data[0]
        ctrl.reset(current_x=x0, current_yaw=0.0, current_y=LINE_Y)
        v_target = 0.16
        ctrl.set_target_velocity(v_target)

        obs_dim = 26 if mode == 'sct' else 23
        obs_hist = np.zeros((3, obs_dim), dtype=np.float32)
        last_act = np.zeros(4, dtype=np.float32)
        k_val, dr_val = 1.0, 0.0
        passed = False
        max_pitch, max_roll = 0.0, 0.0

        for step in range(35000):
            t = step * 0.001
            s = extract_sensors(m, data)
            if abs(s['pitch']) > max_pitch: max_pitch = abs(s['pitch'])
            if abs(s['roll']) > max_roll: max_roll = abs(s['roll'])
            if s['x_pos'] >= 7.00:
                passed = True
                break
            if abs(s['pitch']) > np.radians(45) or abs(s['roll']) > np.radians(45) or s['z_pos'] < 0.025:
                print(f"[{mode.upper()}] Failed at x={s['x_pos']:.2f}m, t={t:.2f}s")
                break
            if step % 20 == 0:
                if mode == 'prcc':
                    o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=None)
                    obs_hist = np.roll(obs_hist, -1, axis=0)
                    obs_hist[-1] = o
                    act, _ = model_prcc.predict(obs_hist.flatten(), deterministic=True)
                    last_act = act
                    act_un = act * [0.35, np.radians(2.0), 0.05, 0.05]
                    k_val = 1.0 + act_un[0]
                    dr_val = act_un[1]
                elif mode == 'sct':
                    if step > 0: tank.commit()
                    o = build_single_obs(s, v_target, ctrl, last_act, tank_feat=tank.features())
                    obs_hist = np.roll(obs_hist, -1, axis=0)
                    obs_hist[-1] = o
                    act, _ = model_sct.predict(obs_hist.flatten(), deterministic=True)
                    last_act = act
                    act_un = act * [0.35, np.radians(2.0), 0.05, 0.05]
                    k_val, dr_val, alpha_val, _ = tank.project(ctrl, s, 1.0 + act_un[0], act_un[1])

            cmd = ctrl.compute(s, dt=0.001, delta_roll=dr_val, k_scale=k_val, enable_compliance=(mode != 'lqr'))
            data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
            data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
            data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
            data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
            mujoco.mj_step(m, data)

            if mode == 'sct':
                sl = abs(s['left_wheel_vel'] * R_WHEEL - s['forward_vel'])
                sr = abs(s['right_wheel_vel'] * R_WHEEL - s['forward_vel'])
                p_slip = (abs(cmd['torque_left_wheel']) * sl + abs(cmd['torque_right_wheel']) * sr) / R_WHEEL
                tank.accumulate(ctrl, s, p_slip, 0.001)

        print(f"[{mode.upper():4s}] 7.0m Full Run with 12mm Ramp & Drop: Passed={passed} | t={t:.2f}s | Max Roll={np.degrees(max_roll):.1f}° | Max Pitch={np.degrees(max_pitch):.1f}° | Final x={s['x_pos']:.2f}m")

if __name__ == '__main__':
    main()
