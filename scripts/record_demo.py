"""
RoboWalker 2026 轮腿自平衡机器人 - 官方演示动画录制脚本 (Demo Recorder)
=================================================================
自动执行标准化机动序列，并通过 MuJoCo 离线跟踪相机渲染高清动画:
  1. [0.0s ~ 2.5s] 原地稳态自平衡 (静止于原点，误差 < 0.1mm)
  2. [2.5s ~ 5.5s] 前向遥控巡航加速 (v = +0.08 m/s)
  3. [5.5s ~ 8.0s] 差速转向转弯机动 (Yaw 转向)
  4. [8.0s ~ 8.1s] 突发 0.25 N 外部脉冲冲击扰动
  5. [8.1s ~ 11.0s] 自适应前冲追赶平衡并稳健回位驻车

输出: ./demo/wheel_leg_demo.gif
"""

import os
import mujoco
import numpy as np
from PIL import Image
from test_balance import WheelLegBalanceController

def record_demo():
    print("=" * 65)
    print("  RoboWalker 2026 轮腿机器人 - 正在录制官方演示 GIF...")
    print("=" * 65)

    def find_xml_path(name="wheel_leg.xml"):
        for p in [name, os.path.join("..", name), os.path.join(os.path.dirname(__file__), "..", name), os.path.join(os.path.dirname(__file__), name)]:
            if os.path.exists(p):
                return p
        return name

    demo_dir = "demo" if os.path.exists("wheel_leg.xml") else os.path.join(os.path.dirname(__file__), "..", "demo")
    os.makedirs(demo_dir, exist_ok=True)
    gif_path = os.path.join(demo_dir, "wheel_leg_demo.gif")

    model = mujoco.MjModel.from_xml_path(find_xml_path("wheel_leg.xml"))
    data = mujoco.MjData(model)
    controller = WheelLegBalanceController()

    # 初始化姿态
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.04905
    data.qpos[3] = 1.0
    data.qpos[7] = -0.0042
    data.qpos[9] = -0.0042
    mujoco.mj_forward(model, data)
    controller.reset(current_x=0.0, current_yaw=0.0)

    # 离线渲染器设置 (480x640, 30fps)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.22
    cam.elevation = -18
    cam.azimuth = 140

    total_duration = 11.0  # 秒
    sim_dt = model.opt.timestep # 0.001s (1000Hz)
    total_steps = int(total_duration / sim_dt)
    render_interval = int(1.0 / 30.0 / sim_dt) # 每 33 步渲染一帧 (30 fps)

    frames = []
    print(f">> 开始执行 11.0 秒机动序列 (总步数: {total_steps}, 目标帧数: {total_steps // render_interval} 帧)...")

    for step in range(total_steps):
        t = step * sim_dt

        # 动作编排:
        if 2.5 <= t < 5.5:
            # 阶段 2: 前向巡航
            controller.target_v = 0.08
        elif 5.5 <= t < 8.0:
            # 阶段 3: 边走边转弯
            controller.target_v = 0.08
            controller.target_yaw += 0.3 * sim_dt
        elif 8.0 <= t < 8.06:
            # 阶段 4: 0.25N 猛烈推力脉冲 (持续 60ms)
            controller.target_v = 0.0
            data.xfrc_applied[1, 0] = 0.25
        else:
            # 阶段 1 & 5: 驻车制动与自恢复
            data.xfrc_applied[1, 0] = 0.0
            controller.target_v = 0.0

        # 控制器计算
        u_lh, u_lw, u_rh, u_rw, roll, pitch, yaw, x = controller.compute(data)
        data.ctrl[0] = u_lh
        data.ctrl[1] = u_lw
        data.ctrl[2] = u_rh
        data.ctrl[3] = u_rw

        mujoco.mj_step(model, data)

        # 抽帧渲染 (30 fps)
        if step % render_interval == 0:
            # 摄像机平滑跟踪小车机身中心
            cam.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2]]
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            frames.append(Image.fromarray(img))

            if len(frames) % 30 == 0:
                print(f"   [录制进度] 已渲染 {len(frames)} 帧 (仿真时间: {t:4.1f}s / {total_duration:.1f}s)...")

    print(f"\n>> 正在编码保存为 GIF 动图: {gif_path} ...")
    # 保存优化调色板的轻量级高质量 GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=33,  # 30 fps -> ~33ms 每帧
        loop=0,
        optimize=True
    )
    file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f">> [DONE] Record finished! Demo saved to: {gif_path} (File size: {file_size_mb:.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    record_demo()
