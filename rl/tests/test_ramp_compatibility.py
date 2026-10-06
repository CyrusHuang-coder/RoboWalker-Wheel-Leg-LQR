"""
test_ramp_compatibility.py: 严格实测现有预训练模型在 12mm 爬坡下坎复杂地形上的泛化能力
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
from test_extended_run import extract_sensors
from stable_baselines3 import PPO

def run_test():
    x_len = 5.0
    ncol = 1280
    x = np.linspace(0, x_len, ncol)
    y = np.linspace(-0.5, 0.5, 256)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 地形高程设计:
    # 0.0 ~ 0.7m: 平坦起点
    # 0.7 ~ 1.0m: 12mm 平缓长爬坡 (上升12mm, 距离0.30m, 平均坡度 2.3°，峰值 4.5°)
    # 1.0 ~ 1.4m: 12mm 顶峰大平台
    # 1.4 ~ 1.6m: 阶梯跌落下坎 (从12mm陡降至6mm, 落差6mm，带圆弧倒角)
    # 1.6 ~ 2.0m: 6mm 次级平台
    # 2.0 ~ 2.3m: 缓坡下冲至地面
    # 2.3 ~ 3.5m: 平地冲刺
    for i in range(len(x)):
        xi = x[i]
        if xi < 0.7:
            zi = 0.0
        elif xi < 1.0: # 12mm 爬坡
            zi = 0.012 * (0.5 - 0.5 * np.cos(np.pi * (xi - 0.7) / 0.30))
        elif xi <= 1.4: # 12mm 顶峰
            zi = 0.012
        elif xi <= 1.6: # 阶梯下坎跌落 6mm
            zi = 0.012 - 0.006 * (0.5 - 0.5 * np.cos(np.pi * (xi - 1.4) / 0.20))
        elif xi <= 2.0: # 6mm 次级平台
            zi = 0.006
        elif xi <= 2.3: # 缓坡下冲
            zi = 0.006 * (0.5 + 0.5 * np.cos(np.pi * (xi - 2.0) / 0.30))
        else:
            zi = 0.0
        Z[:, i] = zi

    # 赛道有效宽度
    y_mask = np.exp(-((Y + 0.0175)**2 / (2 * 0.065**2)))
    Z = Z * y_mask

    xml = f"""
<mujoco>
  <compiler angle='radian' meshdir='car_urdf/meshes' autolimits='true'/>
  <option gravity='0 0 -9.81' timestep='0.001' integrator='Euler'/>
  <default><geom friction='1.2 0.005 0.0001' condim='3'/></default>
  <asset>
    <mesh name='base_link' file='base_link.STL'/>
    <mesh name='left_leg' file='left_leg.STL'/>
    <mesh name='left_wheel' file='left_wheel.STL'/>
    <mesh name='right_leg' file='right_leg.STL'/>
    <mesh name='right_wheel' file='right_wheel.STL'/>
    <hfield name='hf' nrow='256' ncol='1280' size='2.5 0.5 0.015 0.01'/>
  </asset>
  <worldbody>
    <geom type='hfield' hfield='hf' pos='2.5 0 0'/>
    <body name='base_link' pos='0.08 0 0.058'>
      <freejoint name='root'/>
      <inertial pos='-0.011138 -0.0175 0.001698' mass='0.16794' diaginertia='1.6308e-05 2.8107e-05 3.0344e-05'/>
      <geom type='mesh' mesh='base_link'/>
      <body name='left_leg' pos='-0.0093975 0.0000 -0.0010809' quat='0.0159413 0.706926 0.706928 0.0159413'>
        <inertial pos='-0.00175 -0.00163 0.018' quat='0.998982 0.0451138 0 0' mass='0.0087651' diaginertia='1.2889e-06 1.24217e-06 6.45868e-08'/>
        <joint name='left_hip_joint' range='-3.14 3.14' actuatorfrcrange='-10 10' axis='1 0 0' damping='0.001'/>
        <geom type='mesh' mesh='left_leg' contype='0' conaffinity='0'/>
        <body name='left_wheel' pos='-0.0015 -0.0036054 0.039837' quat='0.498403 0.501591 -0.498403 -0.501594'>
          <inertial pos='0 0 -0.004057' quat='0.5 0.5 -0.5 0.5' mass='0.0022065' diaginertia='4.03259e-08 2.24677e-08 2.24677e-08'/>
          <joint name='left_wheel_joint' axis='0 0 -1' actuatorfrcrange='-10 10' damping='0.0001'/>
          <geom type='mesh' mesh='left_wheel' contype='0' conaffinity='0'/>
          <geom type='sphere' size='0.008' friction='1.5 0.01 0.0001' group='3'/>
        </body>
      </body>
      <body name='right_leg' pos='-0.0093975 -0.0315 -0.0010809' quat='0.0159413 0.706926 0.706928 0.0159413'>
        <inertial pos='-0.00175 -0.00163 0.018' quat='0.998982 0.0451138 0 0' mass='0.0087651' diaginertia='1.2889e-06 1.24217e-06 6.45868e-08'/>
        <joint name='right_hip_joint' range='-3.14 3.14' actuatorfrcrange='-10 10' axis='1 0 0' damping='0.001'/>
        <geom type='mesh' mesh='right_leg' contype='0' conaffinity='0'/>
        <body name='right_wheel' pos='-0.002 -0.0036054 0.039837' quat='0.501591 -0.498403 0.501594 -0.498403'>
          <inertial pos='0 0 -0.004057' quat='0.5 0.5 -0.5 0.5' mass='0.0022065' diaginertia='4.03259e-08 2.24677e-08 2.24677e-08'/>
          <joint name='right_wheel_joint' axis='0 0 1' actuatorfrcrange='-10 10' damping='0.0001'/>
          <geom type='mesh' mesh='right_wheel' contype='0' conaffinity='0'/>
          <geom type='sphere' size='0.008' friction='1.5 0.01 0.0001' group='3'/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name='left_hip_motor' joint='left_hip_joint' ctrlrange='-10 10'/>
    <motor name='left_wheel_motor' joint='left_wheel_joint' ctrlrange='-10 10'/>
    <motor name='right_hip_motor' joint='right_hip_joint' ctrlrange='-10 10'/>
    <motor name='right_wheel_motor' joint='right_wheel_joint' ctrlrange='-10 10'/>
  </actuator>
  <sensor>
    <framepos name='body_pos' objtype='body' objname='base_link'/>
    <framequat name='body_quat' objtype='body' objname='base_link'/>
    <framelinvel name='body_linvel' objtype='body' objname='base_link'/>
    <frameangvel name='body_angvel' objtype='body' objname='base_link'/>
    <jointpos name='left_hip_pos' joint='left_hip_joint'/>
    <jointvel name='left_hip_vel' joint='left_hip_joint'/>
    <jointpos name='right_hip_pos' joint='right_hip_joint'/>
    <jointvel name='right_hip_vel' joint='right_hip_joint'/>
    <jointpos name='left_wheel_pos' joint='left_wheel_joint'/>
    <jointvel name='left_wheel_vel' joint='left_wheel_joint'/>
    <jointpos name='right_wheel_pos' joint='right_wheel_joint'/>
    <jointvel name='right_wheel_vel' joint='right_wheel_joint'/>
  </sensor>
</mujoco>
"""
    m = mujoco.MjModel.from_xml_string(xml)
    norm_hf = (Z / 0.015).astype(np.float32)
    m.hfield_data[:] = norm_hf.ravel()

    model_prcc = PPO.load("rl/models/curriculum_master_model.zip")
    model_sct = PPO.load("rl/models/sct_master.zip")

    for mode in ["lqr", "prcc", "sct"]:
        data = mujoco.MjData(m)
        ctrl = PriorController()
        tank = SlipCoupledEnergyTank(
            E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
            beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
        ) if mode == "sct" else None

        data.qpos[0] = 0.08
        data.qpos[2] = 0.060
        for _ in range(80):
            mujoco.mj_step(m, data)

        x0 = data.sensor('body_pos').data[0]
        ctrl.reset(current_x=x0, current_yaw=0.0, current_y=-0.0175)
        ctrl.set_target_velocity(0.16)

        obs_dim = 26 if mode == "sct" else 23
        obs_hist = np.zeros((3, obs_dim), dtype=np.float32)
        last_act = np.zeros(4, dtype=np.float32)

        k_val = 1.0
        dr_val = 0.0
        alpha_val = 1.0
        passed = False
        max_pitch = 0.0
        max_roll = 0.0

        for step in range(25000):
            t = step * 0.001
            s = extract_sensors(m, data)

            if abs(s['pitch']) > max_pitch:
                max_pitch = abs(s['pitch'])
            if abs(s['roll']) > max_roll:
                max_roll = abs(s['roll'])

            if s['x_pos'] >= 2.5:
                passed = True
                break

            if abs(s['pitch']) > np.radians(45) or abs(s['roll']) > np.radians(45) or s['z_pos'] < 0.025:
                print(f"[{mode.upper()}] Failed at x={s['x_pos']:.2f}m, t={t:.2f}s | pitch={np.degrees(s['pitch']):.1f}°, roll={np.degrees(s['roll']):.1f}°")
                break

            if step % 20 == 0:
                if mode == "prcc":
                    o = build_single_obs(s, 0.16, ctrl, last_act, tank_feat=None)
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
                    o = build_single_obs(s, 0.16, ctrl, last_act, tank_feat=tf)
                    obs_hist = np.roll(obs_hist, -1, axis=0)
                    obs_hist[-1] = o
                    act, _ = model_sct.predict(obs_hist.flatten(), deterministic=True)
                    last_act = act
                    act_un = act * [0.35, np.radians(2.0), 0.05, 0.05]
                    k_val, dr_val, alpha_val, _ = tank.project(ctrl, s, 1.0 + act_un[0], act_un[1])

            cmd = ctrl.compute(s, dt=0.001, delta_roll=dr_val, k_scale=k_val, enable_compliance=(mode != "lqr"))
            data.actuator('left_hip_motor').ctrl[0] = cmd['torque_left_hip']
            data.actuator('right_hip_motor').ctrl[0] = cmd['torque_right_hip']
            data.actuator('left_wheel_motor').ctrl[0] = cmd['torque_left_wheel']
            data.actuator('right_wheel_motor').ctrl[0] = cmd['torque_right_wheel']
            mujoco.mj_step(m, data)

            if mode == "sct":
                sl = abs(s['left_wheel_vel'] * 0.008 - s['forward_vel'])
                sr = abs(s['right_wheel_vel'] * 0.008 - s['forward_vel'])
                p_slip = (abs(cmd['torque_left_wheel']) * sl + abs(cmd['torque_right_wheel']) * sr) / 0.008
                tank.accumulate(ctrl, s, p_slip, 0.001)

        print(f"[{mode.upper():4s}] 12mm Ramp & Drop Passed: {passed} | Max Pitch: {np.degrees(max_pitch):.1f}° | Max Roll: {np.degrees(max_roll):.1f}° | Final x={s['x_pos']:.2f}m")

if __name__ == "__main__":
    run_test()
