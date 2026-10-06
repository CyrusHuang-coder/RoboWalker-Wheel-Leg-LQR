"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/recorders/record_tri_comparison.py`: 4.0m 程序化起伏赛道三大控制算法同台竞技对比录制器
====================================================================================================
1. 核心定位与评估目的：
   - 录制 Pure LQR Baseline、Standard PRCC-RL 与 SCT-RRL 在 4.0m 程序化复杂随机地形
     (Difficulty = 0.80) 上的三联 (Side-by-Side-by-Side) 高清对比 GIF。
   - 验证能量储罐有源-无源边界在单侧突发冲击减速垄下的自适应介入与保形效果：
     * Pure LQR (刚性底盘): 遇到单侧凸起触发车身猛烈侧倾 (>3.0° TILT ALERT)；
     * Standard PRCC-RL: 主动摆动腿吸收高程差，但伴随高频微颤；
     * SCT-RRL (Ours): 能量储罐动态投影调节，稳定且平顺。
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
from procedural_terrain import ProceduralTerrainGenerator
from energy_tank import SlipCoupledEnergyTank
from sim_runner import build_single_obs, R_WHEEL, LINE_Y


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


def record_tri_comparison(difficulty=0.80, seed=42000, duration=22.0, fps=15,
                          out_gif="demo/wheel_leg_tri_comparison_4m.gif"):
    print("=" * 76)
    print("  RoboWalker 2026 - 生成三大算法全景对比 GIF (4.0m 随机地形)")
    print(f"  地形难度: {difficulty:.2f} | 随机种子: {seed} | 目标航速: 0.16 m/s")
    print("=" * 76)

    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 1. 在共享模型中注入完全相同的程序化随机地形
    gen = ProceduralTerrainGenerator()
    rng = np.random.default_rng(seed)
    gen.update_model_hfield(model, difficulty=difficulty, rng=rng)

    # 2. 实例化三个独立仿真状态
    data_lqr = mujoco.MjData(model)
    data_prcc = mujoco.MjData(model)
    data_sct = mujoco.MjData(model)

    ctrl_lqr = PriorController()
    ctrl_prcc = PriorController()
    ctrl_sct = PriorController()

    tank_sct = SlipCoupledEnergyTank(
        E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
        beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
    )

    # 载入预训练策略
    prcc_path = os.path.join(rl_dir, "models", "curriculum_master_model.zip")
    sct_path = os.path.join(rl_dir, "models", "sct_master.zip")
    model_prcc = PPO.load(prcc_path)
    model_sct = PPO.load(sct_path)

    # 初始位置统一放置在平稳起点
    for d in [data_lqr, data_prcc, data_sct]:
        d.qpos[0] = 0.08
        d.qpos[2] = 0.060

    # 预热沉降
    for _ in range(80):
        mujoco.mj_step(model, data_lqr)
        mujoco.mj_step(model, data_prcc)
        mujoco.mj_step(model, data_sct)

    x0_lqr = data_lqr.sensor('body_pos').data[0]
    x0_prcc = data_prcc.sensor('body_pos').data[0]
    x0_sct = data_sct.sensor('body_pos').data[0]

    ctrl_lqr.reset(current_x=x0_lqr, current_yaw=0.0, current_y=LINE_Y)
    ctrl_prcc.reset(current_x=x0_prcc, current_yaw=0.0, current_y=LINE_Y)
    ctrl_sct.reset(current_x=x0_sct, current_yaw=0.0, current_y=LINE_Y)

    v_target = 0.16
    ctrl_lqr.set_target_velocity(v_target)
    ctrl_prcc.set_target_velocity(v_target)
    ctrl_sct.set_target_velocity(v_target)

    # 渲染器设置
    w_sub, h_sub = 400, 300
    renderer_lqr = mujoco.Renderer(model, h_sub, w_sub)
    renderer_prcc = mujoco.Renderer(model, h_sub, w_sub)
    renderer_sct = mujoco.Renderer(model, h_sub, w_sub)

    cam_lqr = mujoco.MjvCamera()
    cam_prcc = mujoco.MjvCamera()
    cam_sct = mujoco.MjvCamera()
    for cam in [cam_lqr, cam_prcc, cam_sct]:
        mujoco.mjv_defaultCamera(cam)
        cam.distance = 0.31
        cam.elevation = -12.0
        cam.azimuth = 145.0

    dt = model.opt.timestep
    substep = 20
    render_interval = int(1.0 / fps / dt)
    total_steps = int(duration / dt)

    obs_hist_prcc = np.zeros((3, 23), dtype=np.float32)
    obs_hist_sct = np.zeros((3, 26), dtype=np.float32)
    last_act_prcc = np.zeros(4, dtype=np.float32)
    last_act_sct = np.zeros(4, dtype=np.float32)

    font_main = ImageFont.load_default()
    font_bold = font_main
    font_hud = font_main
    font_small = font_main

    frames = []
    print(f"开始物理仿真与多视角同步渲染 (预计渲染 {total_steps // render_interval} 帧)...")

    act_prcc_val = np.zeros(4)
    k_sct, dr_sct, alpha_sct = 1.0, 0.0, 1.0

    for step in range(total_steps):
        t = step * dt

        # 1. 策略推理 (每 20ms 一步)
        if step % substep == 0:
            s_prcc = extract_sensors(model, data_prcc)
            o_p = build_single_obs(s_prcc, v_target, ctrl_prcc, last_act_prcc, tank_feat=None)
            obs_hist_prcc = np.roll(obs_hist_prcc, -1, axis=0)
            obs_hist_prcc[-1] = o_p
            act_p, _ = model_prcc.predict(obs_hist_prcc.flatten(), deterministic=True)
            last_act_prcc = act_p
            act_prcc_un = act_p * [0.35, np.radians(2.0), 0.05, 0.05]
            k_prcc = 1.0 + act_prcc_un[0]
            dr_prcc = act_prcc_un[1]

            if step > 0:
                tank_sct.commit()
            s_sct = extract_sensors(model, data_sct)
            tf_sct = tank_sct.features()
            o_s = build_single_obs(s_sct, v_target, ctrl_sct, last_act_sct, tank_feat=tf_sct)
            obs_hist_sct = np.roll(obs_hist_sct, -1, axis=0)
            obs_hist_sct[-1] = o_s
            act_s, _ = model_sct.predict(obs_hist_sct.flatten(), deterministic=True)
            last_act_sct = act_s
            act_sct_un = act_s * [0.35, np.radians(2.0), 0.05, 0.05]
            k_req = 1.0 + act_sct_un[0]
            dr_req = act_sct_un[1]
            k_sct, dr_sct, alpha_sct, dW_sct = tank_sct.project(ctrl_sct, s_sct, k_req, dr_req)

        # 2. 动力学控制步解算
        # (1) LQR (刚性)
        s_l = extract_sensors(model, data_lqr)
        cmd_l = ctrl_lqr.compute(s_l, dt=dt, enable_compliance=False)
        data_lqr.actuator('left_hip_motor').ctrl[0] = cmd_l['torque_left_hip']
        data_lqr.actuator('right_hip_motor').ctrl[0] = cmd_l['torque_right_hip']
        data_lqr.actuator('left_wheel_motor').ctrl[0] = cmd_l['torque_left_wheel']
        data_lqr.actuator('right_wheel_motor').ctrl[0] = cmd_l['torque_right_wheel']
        mujoco.mj_step(model, data_lqr)

        # (2) Standard PRCC
        s_p = extract_sensors(model, data_prcc)
        cmd_p = ctrl_prcc.compute(s_p, dt=dt, delta_roll=dr_prcc, k_scale=k_prcc, enable_compliance=True)
        data_prcc.actuator('left_hip_motor').ctrl[0] = cmd_p['torque_left_hip']
        data_prcc.actuator('right_hip_motor').ctrl[0] = cmd_p['torque_right_hip']
        data_prcc.actuator('left_wheel_motor').ctrl[0] = cmd_p['torque_left_wheel']
        data_prcc.actuator('right_wheel_motor').ctrl[0] = cmd_p['torque_right_wheel']
        mujoco.mj_step(model, data_prcc)

        # (3) SCT-RRL (Ours)
        s_s = extract_sensors(model, data_sct)
        cmd_s = ctrl_sct.compute(s_s, dt=dt, delta_roll=dr_sct, k_scale=k_sct, enable_compliance=True)
        data_sct.actuator('left_hip_motor').ctrl[0] = cmd_s['torque_left_hip']
        data_sct.actuator('right_hip_motor').ctrl[0] = cmd_s['torque_right_hip']
        data_sct.actuator('left_wheel_motor').ctrl[0] = cmd_s['torque_left_wheel']
        data_sct.actuator('right_wheel_motor').ctrl[0] = cmd_s['torque_right_wheel']
        mujoco.mj_step(model, data_sct)

        sl_s = abs(s_s['left_wheel_vel'] * R_WHEEL - s_s['forward_vel'])
        sr_s = abs(s_s['right_wheel_vel'] * R_WHEEL - s_s['forward_vel'])
        p_slip = (abs(cmd_s['torque_left_wheel']) * sl_s + abs(cmd_s['torque_right_wheel']) * sr_s) / R_WHEEL
        tank_sct.accumulate(ctrl_sct, s_s, p_slip, dt)

        # 3. 画面渲染与 HUD 仪表盘合成
        if step % render_interval == 0:
            cam_lqr.lookat = [data_lqr.sensor('body_pos').data[0], 0.0, 0.045]
            cam_prcc.lookat = [data_prcc.sensor('body_pos').data[0], 0.0, 0.045]
            cam_sct.lookat = [data_sct.sensor('body_pos').data[0], 0.0, 0.045]

            renderer_lqr.update_scene(data_lqr, camera=cam_lqr)
            img_lqr = Image.fromarray(renderer_lqr.render())

            renderer_prcc.update_scene(data_prcc, camera=cam_prcc)
            img_prcc = Image.fromarray(renderer_prcc.render())

            renderer_sct.update_scene(data_sct, camera=cam_sct)
            img_sct = Image.fromarray(renderer_sct.render())

            canvas = Image.new("RGB", (w_sub * 3, h_sub + 40), color=(18, 18, 22))
            canvas.paste(img_lqr, (0, 0))
            canvas.paste(img_prcc, (w_sub, 0))
            canvas.paste(img_sct, (w_sub * 2, 0))

            draw = ImageDraw.Draw(canvas)
            draw.line([(w_sub, 0), (w_sub, h_sub)], fill=(70, 70, 80), width=2)
            draw.line([(w_sub * 2, 0), (w_sub * 2, h_sub)], fill=(70, 70, 80), width=2)
            draw.line([(0, h_sub), (w_sub * 3, h_sub)], fill=(100, 100, 110), width=2)

            # Panel 1: LQR
            draw.rectangle([(8, 8), (w_sub - 8, 34)], fill=(160, 35, 35))
            draw.text((14, 12), "1. Pure LQR Baseline (Rigid)", fill=(255, 255, 255), font=font_bold)
            roll_l = np.degrees(s_l['roll'])
            r_col_l = (255, 80, 80) if abs(roll_l) > 3.0 else (255, 200, 200)
            tag_l = " [SEVERE TILT!]" if abs(roll_l) > 3.0 else " [Locked PD]"
            draw.text((14, 40), f"Body Roll: {roll_l:+5.1f}\u00b0{tag_l}", fill=r_col_l, font=font_hud)
            draw.text((14, 58), f"Position x: {s_l['x_pos']:.2f} m | Vel: {s_l['forward_vel']:.2f} m/s", fill=(200, 200, 200), font=font_small)
            draw.text((14, 74), "Suspension: Disabled (Stiff)", fill=(180, 180, 180), font=font_small)

            # Panel 2: PRCC-RL
            draw.rectangle([(w_sub + 8, 8), (w_sub * 2 - 8, 34)], fill=(190, 130, 20))
            draw.text((w_sub + 14, 12), "2. Standard PRCC-RL (Unconstrained)", fill=(255, 255, 255), font=font_bold)
            roll_p = np.degrees(s_p['roll'])
            draw.text((w_sub + 14, 40), f"Body Roll: {roll_p:+5.1f}\u00b0 [Active Compensated]", fill=(255, 230, 100), font=font_hud)
            draw.text((w_sub + 14, 58), f"Position x: {s_p['x_pos']:.2f} m | Vel: {s_p['forward_vel']:.2f} m/s", fill=(200, 200, 200), font=font_small)
            draw.text((w_sub + 14, 74), f"Active dRoll: {np.degrees(dr_prcc):+4.1f}\u00b0 | Stiff: {k_prcc:.2f}x", fill=(240, 210, 120), font=font_small)

            # Panel 3: SCT-RRL
            draw.rectangle([(w_sub * 2 + 8, 8), (w_sub * 3 - 8, 34)], fill=(30, 135, 60))
            draw.text((w_sub * 2 + 14, 12), "3. SCT-RRL (Ours: Passivity-Bounded)", fill=(255, 255, 255), font=font_bold)
            roll_s = np.degrees(s_s['roll'])
            draw.text((w_sub * 2 + 14, 40), f"Body Roll: {roll_s:+5.1f}\u00b0 [Tank-Protected]", fill=(120, 255, 140), font=font_hud)
            draw.text((w_sub * 2 + 14, 58), f"Position x: {s_s['x_pos']:.2f} m | Vel: {s_s['forward_vel']:.2f} m/s", fill=(200, 200, 200), font=font_small)
            e_str = f"Tank E_T: {tank_sct.E*1e4:.2f} | Alpha*: {alpha_sct:.2f}"
            draw.text((w_sub * 2 + 14, 74), f"{e_str} | Active dRoll: {np.degrees(dr_sct):+4.1f}\u00b0", fill=(160, 255, 180), font=font_small)

            # Progress Bar
            draw.rectangle([(10, h_sub + 6), (w_sub * 3 - 10, h_sub + 34)], fill=(28, 28, 34))
            progress_ratio = np.clip(s_s['x_pos'] / 3.50, 0.0, 1.0)
            prog_w = int((w_sub * 3 - 24) * progress_ratio)
            draw.rectangle([(12, h_sub + 8), (12 + prog_w, h_sub + 14)], fill=(60, 160, 240))
            info_txt = (f"Time: {t:4.1f}s | Traversal: {progress_ratio*100:4.1f}% | "
                        f"Course: Procedural Rough Terrain (Diff={difficulty:.2f}) | Controlled Cmd: {v_target:.2f} m/s")
            draw.text((14, h_sub + 18), info_txt, fill=(220, 220, 220), font=font_small)

            frames.append(canvas)

        if s_l['x_pos'] >= 3.50 and s_p['x_pos'] >= 3.50 and s_s['x_pos'] >= 3.50:
            print(f"三大算法均顺利穿越全部障碍冲线！(t={t:.2f}s)")
            break

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    print(f"正在编码合成高清动图 ({len(frames)} 帧, {fps} fps) 至 {out_gif} ...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    print(f"  [SUCCESS] 动图生成完毕: {out_gif} (总帧数: {len(frames)})")


if __name__ == "__main__":
    record_tri_comparison(difficulty=0.80, seed=42000, duration=22.0, fps=15)
