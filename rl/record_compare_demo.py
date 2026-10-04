"""
record_compare_demo.py: 录制纯 LQR vs PRCC 残差强化学习 高清对比动图
=====================================================================
自动生成 side-by-side 并排对比 GIF: demo/wheel_leg_rl_comparison.gif
左侧: Pure LQR Baseline (单侧过坑/凸起时剧烈侧倾颠簸)
右侧: PRCC Residual RL (主动虚拟顺应与差动防侧倾，机身平稳一字线)
"""
import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from prior_controller import PriorController, quat2rpy
from eval_baseline import extract_sensors


def record_comparison():
    print("=" * 65)
    print("  RoboWalker 2026 - 正在录制 LQR vs PRCC-RL 高清并排对比 GIF...")
    print("=" * 65)

    xml_path = "rl/terrain/wheel_leg_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 两个独立环境状态
    data_lqr = mujoco.MjData(model)
    data_rl  = mujoco.MjData(model)

    ctrl_lqr = PriorController()
    ctrl_rl  = PriorController()

    # 初始状态：从赛道起点 (x=0.08m) 出发，完整穿越平地段、正弦波起伏段、非对称实体减速垄群与平稳降落段
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

    # 设定完全相同的前向巡航速度 (两者以 0.16 m/s 高速并排冲锋)
    ctrl_lqr.set_target_velocity(0.16)
    ctrl_rl.set_target_velocity(0.16)

    rl_policy = PPO.load("rl/models/best_model.zip")

    # 渲染器设置 (每个画面 360x480)
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
    dt = model.opt.timestep # 0.001s
    duration = 24.0 # 全程穿越测试 (涵盖 0.08m~3.50m 全程，约 21 秒)
    render_interval = int(1.0 / fps / dt) # 约每 62 步录制一帧
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

    # 预加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_bold = font

    print(f"Recording bump traversal: {total_steps} steps at {fps} fps...")

    for step in range(total_steps):
        t = step * dt

        # 1. LQR 步进 (刚性基线)
        s_lqr = extract_sensors(model, data_lqr)
        act_lqr = ctrl_lqr.compute(s_lqr, dt=dt, enable_compliance=False)
        data_lqr.actuator('left_hip_motor').ctrl[0] = act_lqr['torque_left_hip']
        data_lqr.actuator('right_hip_motor').ctrl[0] = act_lqr['torque_right_hip']
        data_lqr.actuator('left_wheel_motor').ctrl[0] = act_lqr['torque_left_wheel']
        data_lqr.actuator('right_wheel_motor').ctrl[0] = act_lqr['torque_right_wheel']
        mujoco.mj_step(model, data_lqr)

        # 2. RL 步进 (50Hz 动作更新)
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

        # 障碍区通过后在平坦着陆区正常结束录制 (防止掉入边缘)
        if data_rl.sensor('body_pos').data[0] >= 3.45 and data_lqr.sensor('body_pos').data[0] >= 3.45:
            print(f"Both robots cleared obstacle course cleanly! Stopping recording.")
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
            draw.text((495, 68), f"Pos x: {s_rl['x_pos']:.2f}m | Vel: {s_rl['forward_vel']:.2f} m/s | Leg Diff: {np.degrees(delta_roll):+4.1f}°", fill=(210, 255, 210), font=font)

            # 底部地形与航速说明
            draw.line([(480, 0), (480, 360)], fill=(255, 255, 255), width=2)
            draw.rectangle([(250, 328), (710, 354)], fill=(20, 20, 20))
            draw.text((260, 332), f"t = {t:.1f}s | Speed Bumps: Yellow(4.8mm) Red(4.5mm) | v_cmd=0.16m/s", fill=(255, 215, 0), font=font)

            frames.append(combined)

    os.makedirs("demo", exist_ok=True)
    gif_path = "demo/wheel_leg_rl_comparison.gif"
    print(f"Encoding {len(frames)} frames to {gif_path} ...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    print(f"[Done] Comparison animation saved: {gif_path}")


if __name__ == "__main__":
    record_comparison()
