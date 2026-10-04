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

    ctrl_lqr.set_target_velocity(0.09)
    ctrl_rl.set_target_velocity(0.09)

    rl_policy = PPO.load("rl/models/best_model.zip")

    # 预热沉降
    for _ in range(30):
        mujoco.mj_step(model, data_lqr)
        mujoco.mj_step(model, data_rl)

    ctrl_lqr.reset(current_x=data_lqr.sensor('body_pos').data[0], current_yaw=0.0)
    ctrl_rl.reset(current_x=data_rl.sensor('body_pos').data[0], current_yaw=0.0)

    # 渲染器设置 (每个画面 360x480)
    renderer_lqr = mujoco.Renderer(model, 360, 480)
    renderer_rl  = mujoco.Renderer(model, 360, 480)

    cam_lqr = mujoco.MjvCamera()
    cam_rl  = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_lqr)
    mujoco.mjv_defaultCamera(cam_rl)

    for cam in [cam_lqr, cam_rl]:
        cam.distance = 0.25
        cam.elevation = -15
        cam.azimuth = 135

    fps = 25
    dt = model.opt.timestep # 0.001s
    duration = 10.0 # 录制穿越颠簸凸起核心段的 10 秒
    render_interval = int(1.0 / fps / dt) # 每 40 步录制一帧
    total_steps = int(duration / dt)

    frames = []
    obs_history = np.zeros((3, 23), dtype=np.float32)
    last_action = np.zeros(4, dtype=np.float32)
    action_rl = np.zeros(4, dtype=np.float32)

    # 预加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_large = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_large = font

    print(f"Total steps: {total_steps}, recording at {fps} fps...")

    for step in range(total_steps):
        t = step * dt

        # 1. LQR 步进
        s_lqr = extract_sensors(model, data_lqr)
        act_lqr = ctrl_lqr.compute(s_lqr, dt=dt)
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
                0.09 * 5.0,
                ctrl_rl.last_target_pitch * 5.0,
                ctrl_rl.last_u_balance * 20.0,
                last_action[0], last_action[1], last_action[2], last_action[3]
            ], dtype=np.float32)
            obs_history = np.roll(obs_history, shift=-1, axis=0)
            obs_history[-1] = cur_obs
            action_rl, _ = rl_policy.predict(obs_history.flatten(), deterministic=True)
            last_action = action_rl.copy()

        delta_pitch = float(action_rl[0] * 0.015)
        delta_hip   = float(action_rl[1] * 0.05)
        delta_roll  = float(action_rl[2] * 0.06)
        k_scale     = float(1.0 + action_rl[3] * 0.3)

        act_rl = ctrl_rl.compute(
            s_rl, dt=dt,
            delta_pitch=delta_pitch, delta_hip=delta_hip,
            delta_roll=delta_roll, k_scale=k_scale
        )
        data_rl.actuator('left_hip_motor').ctrl[0] = act_rl['torque_left_hip']
        data_rl.actuator('right_hip_motor').ctrl[0] = act_rl['torque_right_hip']
        data_rl.actuator('left_wheel_motor').ctrl[0] = act_rl['torque_left_wheel']
        data_rl.actuator('right_wheel_motor').ctrl[0] = act_rl['torque_right_wheel']
        mujoco.mj_step(model, data_rl)

        # 3. 画面渲染
        if step % render_interval == 0:
            # 动态跟踪相机
            cam_lqr.lookat = [data_lqr.sensor('body_pos').data[0], 0.0, 0.045]
            cam_rl.lookat  = [data_rl.sensor('body_pos').data[0], 0.0, 0.045]

            renderer_lqr.update_scene(data_lqr, camera=cam_lqr)
            img_lqr = Image.fromarray(renderer_lqr.render())

            renderer_rl.update_scene(data_rl, camera=cam_rl)
            img_rl  = Image.fromarray(renderer_rl.render())

            # 拼合双画面 (宽 960 x 高 360)
            combined = Image.new("RGB", (960, 360))
            combined.paste(img_lqr, (0, 0))
            combined.paste(img_rl, (480, 0))

            draw = ImageDraw.Draw(combined)
            # 顶部标签
            draw.rectangle([(10, 10), (220, 42)], fill=(200, 30, 30))
            draw.text((20, 15), "BASELINE: PURE LQR", fill=(255, 255, 255), font=font)
            roll_lqr_deg = np.degrees(s_lqr['roll'])
            draw.text((20, 50), f"Roll: {roll_lqr_deg:+5.1f} deg", fill=(255, 100, 100), font=font)

            draw.rectangle([(490, 10), (740, 42)], fill=(30, 140, 30))
            draw.text((500, 15), "OURS: PRCC RESIDUAL RL", fill=(255, 255, 255), font=font)
            roll_rl_deg = np.degrees(s_rl['roll'])
            draw.text((500, 50), f"Roll: {roll_rl_deg:+5.1f} deg (Absorbed)", fill=(100, 255, 100), font=font)
            draw.text((500, 75), f"Diff Leg: {np.degrees(delta_roll):+4.1f} deg | k: {k_scale:.2f}x", fill=(200, 255, 200), font=font)

            # 中间分割线
            draw.line([(480, 0), (480, 360)], fill=(255, 255, 255), width=2)
            draw.text((440, 330), f"t = {t:.1f}s", fill=(220, 220, 220), font=font)

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
