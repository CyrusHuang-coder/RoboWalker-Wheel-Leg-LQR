"""
test_finish_standing.py: 验证冲线后优雅自平衡制动停驻 (不倾倒)
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
from test_extended_hfield import build_extended_hfield
from test_extended_run import extract_sensors
from stable_baselines3 import PPO

def test_standing():
    xml_path = "rl/terrain/wheel_leg_extended_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    Z, X, Y = build_extended_hfield()
    norm_hf = (Z / 0.0065).astype(np.float32)
    model.hfield_data[:] = norm_hf.ravel()

    model_sct = PPO.load("rl/models/sct_master.zip")

    data = mujoco.MjData(model)
    ctrl = PriorController()
    tank = SlipCoupledEnergyTank(
        E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
        beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
    )

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
    obs_hist = np.zeros((3, 26), dtype=np.float32)
    last_act = np.zeros(4, dtype=np.float32)

    finished = False
    stop_x = 7.05
    k_val, dr_val = 1.0, 0.0

    # 运行 30 秒 (经历冲线后保持 10 秒站立)
    for step in range(30000):
        t = step * dt
        s = extract_sensors(model, data)

        if s['x_pos'] >= 7.00 and not finished:
            finished = True
            print(f"Crossed finish line at t={t:.2f}s! Entering position-hold balance mode...")
            ctrl.set_target_velocity(0.0)
            ctrl.target_x = s['x_pos']

        if step % substep == 0 and not finished:
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
        elif finished:
            # 冲线后保持平稳纯平衡姿态
            k_val, dr_val = 1.0, 0.0

        cmd = ctrl.compute(s, dt=dt, delta_roll=dr_val, k_scale=k_val, enable_compliance=(not finished))
        data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
        mujoco.mj_step(model, data)

        if step % 2000 == 0:
            print(f"t={t:5.1f}s | x={s['x_pos']:.2f}m | pitch={np.degrees(s['pitch']):+5.1f}° | roll={np.degrees(s['roll']):+5.1f}°")

    print(f"Final state: x={s['x_pos']:.2f}m, upright={abs(s['pitch']) < 0.2}")

if __name__ == "__main__":
    test_standing()
