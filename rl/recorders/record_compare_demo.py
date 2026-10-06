"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/recorders/record_compare_demo.py`: 4.0m 赛道经典 Pure LQR 与 PRCC-RL 双联对比录制器
====================================================================================================
1. 核心定位与对比目标：
   - 录制 Pure LQR Baseline 与 PRCC 残差强化学习策略在 4.0m 基础测试赛道上的 Side-by-Side 双联动图。
   - 直观展示在单侧 4.8mm 实体减速垄冲击下，刚性车身与主动顺应车身的姿态对比：
     * 左侧 (Pure LQR): 刚性底盘无法吸收单侧冲击，车身剧烈倾斜颠簸 (>2.5° Severe Tilt)；
     * 右侧 (PRCC-RL): 主动虚拟顺应与差动防侧倾协调介入，机身保持平稳水准。
====================================================================================================
"""

import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

# --------------------------------------------------------------------------------------------------
# 鲁棒路径引导
# --------------------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
          os.path.join(rl_dir, "controllers"),
          os.path.join(rl_dir, "envs"),
          os.path.join(rl_dir, "terrain"),
          os.path.join(rl_dir, "evaluation"),
          os.path.join(rl_dir, "training")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prior_controller import PriorController, quat2rpy


def extract_sensors(model, data):
    """从 MuJoCo 传感器中提取标准观测量字典"""
    body_quat = data.sensor('body_quat').data
    roll, pitch, yaw = quat2rpy(body_quat)
    body_angvel = data.sensor('body_angvel').data
    body_linvel = data.sensor('body_linvel').data
    body_pos = data.sensor('body_pos').data

    lh_pos = float(data.sensor('left_hip_pos').data[0])
    lh_vel = float(data.sensor('left_hip_vel').data[0])
    rh_pos = float(data.sensor('right_hip_pos').data[0])
    rh_vel = float(data.sensor('right_hip_vel').data[0])
    lw_vel = float(data.sensor('left_wheel_vel').data[0])
    rw_vel = float(data.sensor('right_wheel_vel').data[0])

    return {
        'x_pos': float(body_pos[0]),
        'y_pos': float(body_pos[1]),
        'z_pos': float(body_pos[2]),
        'roll': float(roll),
        'pitch': float(pitch),
        'yaw': float(yaw),
        'pitch_rate': float(body_angvel[1]),
        'yaw_rate': float(body_angvel[2]),
        'roll_rate': float(body_angvel[0]),
        'forward_vel': float(body_linvel[0]),
        'body_linvel': body_linvel.copy(),
        'left_hip_pos': lh_pos,
        'left_hip_vel': lh_vel,
        'right_hip_pos': rh_pos,
        'right_hip_vel': rh_vel,
        'left_wheel_vel': lw_vel,
        'right_wheel_vel': rw_vel,
    }


def record_comparison(out_gif="demo/wheel_leg_rl_comparison.gif"):
    print("=" * 65)
    print("  RoboWalker 2026 - 正在录制 LQR vs PRCC-RL 高清并排对比 GIF...")
    print("=" * 65)

    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 两个独立环境状态
    data_lqr = mujoco.MjData(model)
    data_rl  = mujoco.MjData(model)

    ctrl_lqr = PriorController()
    ctrl_rl  = PriorController()

    data_lqr.qpos[0] = 0.08
    data_lqr.qpos[2] = 0.060
    data_rl.qpos[0]  = 0.08
    data_rl.qpos[2]  = 0.060

    # 预热沉降
    for _ in range(50):
        mujoco.mj_step(model, data_lqr)
        mujoco.mj_step(model, data_rl)

    init_x_lqr = data_lqr.sensor('body_pos').data[0]
    init_x_rl  = data_rl.sensor('body_pos').data[0]

    ctrl_lqr.reset(current_x=init_x_lqr, current_yaw=0.0, current_y=-0.0175)
    ctrl_rl.reset(current_x=init_x_rl, current_yaw=0.0, current_y=-0.0175)

    ctrl_lqr.set_target_velocity(0.16)
    ctrl_rl.set_target_velocity(0.16)

    model_path = os.path.join(rl_dir, "models", "best_model.zip")
    rl_policy = PPO.load(model_path)

    renderer_lqr = mujoco.Renderer(model, 360, 480)
    renderer_rl  = mujoco.Renderer(model, 360, 480)

    cam_lqr = mujoco.MjvCamera()
    cam_rl  = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_lqr)
    mujoco.mjv_defaultCamera(cam_rl)

    for cam in [cam_lqr, cam_rl]:
        cam.distance = 0.32
        cam.elevation = -14
        cam.azimuth = 145

    fps = 16
    dt = model.opt.timestep
    duration = 24.0
    render_interval = int(1.0 / fps / dt)
    total_steps = int(duration / dt)

    frames = []
    obs_history = np.zeros((3, 23), dtype=np.float32)
    last_action = np.zeros(4, dtype=np.float32)
    action_rl = np.zeros(4, dtype=np.float32)

    init_s_rl = extract_sensors(model, data_rl)
    init_obs_rl = np.array([
        np.sin(init_s_rl['pitch']), np.cos(init_s_rl['pitch']),
        np.sin(init_s_rl['roll']),  np.cos(init_s_rl['roll']),
        init_s_rl['roll_rate'] * 0.2,
        init_s_rl['pitch_rate'] * 0.2,
        init_s_rl['yaw_rate'] * 0.2,
        init_s_rl['forward_vel'] * 5.0,
        init_s_rl['body_linvel'][1] * 5.0,
        init_s_rl['body_linvel'][2] * 5.0,
        init_s_rl['left_hip_pos'],
        init_s_rl['left_hip_vel'] * 0.1,
        init_s_rl['right_hip_pos'],
        init_s_rl['right_hip_vel'] * 0.1,
        init_s_rl['left_wheel_vel'] * 0.05,
        init_s_rl['right_wheel_vel'] * 0.05,
        0.16 * 5.0,
        ctrl_rl.last_target_pitch * 5.0,
        ctrl_rl.last_u_balance * 20.0,
        0.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    for i in range(3):
        obs_history[i] = init_obs_rl

    font = ImageFont.load_default()
    font_bold = font

    print(f"Recording bump traversal: {total_steps} steps at {fps} fps...")

    for step in range(total_steps):
        t = step * dt

        # 1. LQR 步进
        s_lqr = extract_sensors(model, data_lqr)
        act_lqr = ctrl_lqr.compute(s_lqr, dt=dt, enable_compliance=False)
        data_lqr.actuator('left_hip_motor').ctrl[0] = act_lqr['torque_left_hip']
        data_lqr.actuator('right_hip_motor').ctrl[0] = act_lqr['torque_right_hip']
        data_lqr.actuator('left_wheel_motor').ctrl[0] = act_lqr['torque_left_wheel']
        data_lqr.actuator('right_wheel_motor').ctrl[0] = act_lqr['torque_right_wheel']
        mujoco.mj_step(model, data_lqr)

        # 2. RL 步进
        s_rl = extract_sensors(model, data_rl)
        if step % 20 == 0:
            cur_obs = np.array([
                np.sin(s_rl['pitch']), np.cos(s_rl['pitch']),
                np.sin(s_rl['roll']),  np.cos(s_rl['roll']),
                s_rl['roll_rate'] * 0.2,
                s_rl['pitch_rate'] * 0.2,
                s_rl['yaw_rate'] * 0.2,
                s_rl['forward_vel'] * 5.0,
                s_rl['body_linvel'][1] * 5.0,
                s_rl['body_linvel'][2] * 5.0,
                s_rl['left_hip_pos'],
                s_rl['left_hip_vel'] * 0.1,
                s_rl['right_hip_pos'],
                s_rl['right_hip_vel'] * 0.1,
                s_rl['left_wheel_vel'] * 0.05,
                s_rl['right_wheel_vel'] * 0.05,
                0.16 * 5.0,
                ctrl_rl.last_target_pitch * 5.0,
                ctrl_rl.last_u_balance * 20.0,
                last_action[0], last_action[1], last_action[2], last_action[3]
            ], dtype=np.float32)
            obs_history = np.roll(obs_history, shift=-1, axis=0)
            obs_history[-1] = cur_obs
            action_rl, _ = rl_policy.predict(obs_history.flatten(), deterministic=True)
            last_action = action_rl.copy()

        delta_pitch = 0.0
        delta_hip   = 0.0
        delta_roll  = float(action_rl[2] * 0.10)
        k_scale     = float(1.0 + action_rl[3] * 0.40)

        act_rl = ctrl_rl.compute(
            s_rl, dt=dt,
            delta_pitch=delta_pitch, delta_hip=delta_hip,
            delta_roll=delta_roll, k_scale=k_scale,
            enable_compliance=True
        )
        data_rl.actuator('left_hip_motor').ctrl[0] = act_rl['torque_left_hip']
        data_rl.actuator('right_hip_motor').ctrl[0] = act_rl['torque_right_hip']
        data_rl.actuator('left_wheel_motor').ctrl[0] = act_rl['torque_left_wheel']
        data_rl.actuator('right_wheel_motor').ctrl[0] = act_rl['torque_right_wheel']
        mujoco.mj_step(model, data_rl)

        if data_rl.sensor('body_pos').data[0] >= 3.45 and data_lqr.sensor('body_pos').data[0] >= 3.45:
            print("Both robots cleared obstacle course cleanly! Stopping recording.")
            break

        # 3. 画面渲染
        if step % render_interval == 0:
            cam_lqr.lookat = [data_lqr.sensor('body_pos').data[0], 0.0, 0.045]
            cam_rl.lookat  = [data_rl.sensor('body_pos').data[0], 0.0, 0.045]

            renderer_lqr.update_scene(data_lqr, camera=cam_lqr)
            img_lqr = Image.fromarray(renderer_lqr.render())

            renderer_rl.update_scene(data_rl, camera=cam_rl)
            img_rl  = Image.fromarray(renderer_rl.render())

            combined = Image.new("RGB", (960, 360))
            combined.paste(img_lqr, (0, 0))
            combined.paste(img_rl, (480, 0))

            draw = ImageDraw.Draw(combined)
            # 左侧：LQR 遥测面板
            draw.rectangle([(10, 10), (270, 38)], fill=(180, 30, 30))
            draw.text((15, 14), "BASELINE: PURE LQR (RIGID)", fill=(255, 255, 255), font=font_bold)
            roll_lqr_deg = np.degrees(s_lqr['roll'])
            roll_lqr_col = (255, 60, 60) if abs(roll_lqr_deg) > 2.5 else (255, 180, 180)
            status_lqr = " [SEVERE TILT!]" if abs(roll_lqr_deg) > 2.5 else " [Stiff PD]"
            draw.text((15, 45), f"Body Roll: {roll_lqr_deg:+5.1f} deg{status_lqr}", fill=roll_lqr_col, font=font)
            draw.text((15, 68), f"Pos x: {s_lqr['x_pos']:.2f}m | Vel: {s_lqr['forward_vel']:.2f} m/s", fill=(210, 210, 210), font=font)

            # 右侧：PRCC-RL 遥测面板
            draw.rectangle([(490, 10), (790, 38)], fill=(30, 140, 40))
            draw.text((495, 14), "OURS: PRCC RESIDUAL RL", fill=(255, 255, 255), font=font_bold)
            roll_rl_deg = np.degrees(s_rl['roll'])
            status_rl = " [Active Suspension]" if abs(delta_roll) > 0.005 else " [Level]"
            draw.text((495, 45), f"Body Roll: {roll_rl_deg:+5.1f} deg{status_rl}", fill=(120, 255, 120), font=font)
            draw.text((495, 68), f"Pos x: {s_rl['x_pos']:.2f}m | Vel: {s_rl['forward_vel']:.2f} m/s | Leg Diff: {np.degrees(delta_roll):+4.1f}\u00b0", fill=(210, 255, 210), font=font)

            # 底部说明
            draw.line([(480, 0), (480, 360)], fill=(255, 255, 255), width=2)
            draw.rectangle([(250, 328), (710, 354)], fill=(20, 20, 20))
            draw.text((260, 332), f"t = {t:.1f}s | Speed Bumps: Yellow(4.8mm) Red(4.5mm) | v_cmd=0.16m/s", fill=(255, 215, 0), font=font)

            frames.append(combined)

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    print(f"Encoding {len(frames)} frames to {out_gif} ...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    print(f"[Done] Comparison animation saved: {out_gif}")


if __name__ == "__main__":
    record_comparison()
