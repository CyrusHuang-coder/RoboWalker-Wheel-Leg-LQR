"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/recorders/record_tri_extended_comparison.py`: 10.5m 复合恶劣赛道三大算法同台竞技高保真渲染录制器
====================================================================================================
1. 核心定位与试验场规范：
   - 本模块实现了三大控制算法在 10.5m 复合恶劣赛道上的严格控制变量同台横向对比测试与高清
     三联 (Side-by-Side-by-Side) 动态 GIF / 关键帧视频合成。
   - 保证三大算法处于绝对一致的外部环境：
     * 完全相同的 10.5m 物理赛道高度图 (包含 S 弯道、大起伏波浪、非对称板砖、台阶变速下坡、搓板路)
     * 完全相同的初始位姿 (x=0.08m, y=0.0m, pitch=0) 与相同的 80ms 预热沉降收敛阶段
     * 完全相同的目标巡航速度指令 (v_cmd = 0.20 m/s) 与相同的巡线导航引导律
     * 完全统一的 Body Roll 告警判定标准与 HUD 遥测仪表盘绘制逻辑

2. 三大受试控制架构：
   - Left Panel   : Pure LQR Baseline (刚性底盘基准，无主动顺应，车身剧烈抛掷颠簸)
   - Center Panel : Standard PRCC-RL (先验顺应 + 自由残差强化学习，吸收部分冲击但伴随高频微颤与滑移)
   - Right Panel  : SCT-RRL [Ours] (滑移耦合能量储罐无源残差强化学习，兼顾冲击顺应与被动自适应节流)

3. 实时遥测 HUD 渲染覆盖：
   - 顶部关卡进度条与当前地形特征文字播报 (Zone 1 ~ Zone 7)
   - 底部三视角实时状态仪表：X 位移、Y 横向偏差、Roll 横滚角与姿态状态报警、前向线速度、真实接触滑移率
   - 自动提取并保存 10 处核心关卡高清关键帧至 Artifact 目录供高精学术审阅。
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
from energy_tank import SlipCoupledEnergyTank
from sim_runner import build_single_obs, R_WHEEL, LINE_Y
from test_mega_terrain import build_mega_hfield, get_mega_track_reference


def extract_sensors(model, data):
    """
    从 MuJoCo 仿真状态中提取全套高精传感器数据字典。
    """
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


def get_zone_name(x: float) -> str:
    """根据前进位置返回赛道关卡区段名称"""
    if x < 0.50:
        return "Flat Launch Runway"
    elif x < 2.20:
        return "Zone 1: Increasing Mogul Waves (2.5->6.2mm) & Asym Bricks"
    elif x < 4.80:
        return "Zone 2: Grand Serpentine S-Curve (R>=1.62m + Undulations)"
    elif x < 5.70:
        return "Zone 3: 35mm Mountain Climb (with Slope Bricks)"
    elif x < 7.00:
        return "Zone 4: 35mm Summit Plateau Obstacle Field (L=1.3m)"
    elif x < 8.20:
        return "Zone 5: Terraced Stepped Descent (35->18->0mm + Shelf Brick)"
    elif x < 9.20:
        return "Zone 6: Post-Descent 7-Stripe Rumbles & Stone Slabs"
    else:
        return "Zone 7: Precision Finish Line Parking (10.0m)"


def format_roll_hud(roll_deg: float, is_finished: bool):
    """严格统一的 Body Roll 告警判定与颜色，杜绝任何算法偏袒"""
    abs_r = abs(roll_deg)
    if is_finished:
        return (180, 220, 180), f"Body Roll: {roll_deg:+5.1f}\u00b0 [Parked]"
    elif abs_r > 3.0:
        return (255, 80, 80), f"Body Roll: {roll_deg:+5.1f}\u00b0 [TILT ALERT: {abs_r:.1f}\u00b0]"
    elif abs_r > 1.5:
        return (255, 205, 60), f"Body Roll: {roll_deg:+5.1f}\u00b0 [Warning: {abs_r:.1f}\u00b0]"
    else:
        return (120, 230, 160), f"Body Roll: {roll_deg:+5.1f}\u00b0 [Stable]"


def record_extended_comparison(out_gif: str = "demo/wheel_leg_tri_comparison.gif", fps: int = 10):
    print("=" * 80)
    print("  RoboWalker 2026 - 生成 10.5m 终极地狱级综合赛道三联同台对比 GIF")
    print(f"  赛道总长: 10.5m (8大极限挑战区间) | 目标航速: 0.20 m/s (严格控制变量) | 帧率: {fps} fps")
    print("=" * 80)

    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_extended_terrain.xml")
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    xml_str = xml_str.replace('meshdir="../../car_urdf/meshes"', 'meshdir="car_urdf/meshes"')

    model = mujoco.MjModel.from_xml_string(xml_str)

    # 注入全局 10.5m 终极地狱级赛道高度图
    Z, X, Y = build_mega_hfield(x_len=10.5, max_elevation=0.045)
    norm_hf = (Z / 0.045).astype(np.float32)
    model.hfield_data[:] = norm_hf.ravel()

    # 实例化三个独立仿真状态
    data_lqr = mujoco.MjData(model)
    data_prcc = mujoco.MjData(model)
    data_sct = mujoco.MjData(model)

    ctrl_lqr = PriorController()
    ctrl_prcc = PriorController()
    ctrl_sct = PriorController()

    # 统一纵向速度环跟踪参数与航向跟踪参数
    for c in [ctrl_lqr, ctrl_prcc, ctrl_sct]:
        c.kd_vel = 0.35
        c.ki_vel = 0.08
        c.max_steer_torque = 0.035

    tank_sct = SlipCoupledEnergyTank(
        E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
        beta=0.0035, gamma=0.016, p_slip_deadband=0.003, tau_slip=0.040
    )

    # 载入预训练策略
    prcc_model_path = os.path.join(rl_dir, "models", "curriculum_master_model.zip")
    sct_model_path = os.path.join(rl_dir, "models", "sct_master.zip")
    model_prcc = PPO.load(prcc_model_path)
    model_sct = PPO.load(sct_model_path)

    # 初始放置在平稳起点
    for d in [data_lqr, data_prcc, data_sct]:
        d.qpos[0] = 0.08
        d.qpos[1] = 0.0
        d.qpos[2] = 0.060

    # 预热沉降
    for _ in range(80):
        mujoco.mj_step(model, data_lqr)
        mujoco.mj_step(model, data_prcc)
        mujoco.mj_step(model, data_sct)

    for c in [ctrl_lqr, ctrl_prcc, ctrl_sct]:
        c.reset(current_x=0.08, current_yaw=0.0, current_y=-0.0175)

    # 相机与渲染设置
    w_sub, h_sub = 400, 280
    renderer = mujoco.Renderer(model, height=h_sub, width=w_sub)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("base_link").id
    cam.distance = 0.38
    cam.elevation = -14.0
    cam.azimuth = 145.0

    dt = model.opt.timestep
    sim_duration = 58.0
    total_steps = int(sim_duration / dt)
    render_interval = int(1.0 / (fps * dt))

    font_title = ImageFont.load_default()
    font_hud = ImageFont.load_default()
    font_small = ImageFont.load_default()

    frames = []
    last_act_prcc = np.zeros(2, dtype=np.float32)
    last_act_sct = np.zeros(2, dtype=np.float32)

    key_frame_targets = {
        "flat_start": 0.45,
        "mogul_waves": 1.20,
        "asym_bricks": 1.95,
        "scurve_apex1": 2.85,
        "scurve_apex2": 4.15,
        "mountain_climb": 5.40,
        "summit_plateau": 6.30,
        "stepped_descent": 7.50,
        "washboard": 8.50,
        "finish_line": 9.95,
    }
    captured_key_frames = {}

    fin_lqr = False
    fin_prcc = False
    fin_sct = False

    print("开始执行 10.5m 复合恶劣赛道三联仿真推进并同步渲染 ...")
    for step in range(total_steps):
        t = step * dt

        # 1. 传感器提取
        s_l = extract_sensors(model, data_lqr)
        s_p = extract_sensors(model, data_prcc)
        s_s = extract_sensors(model, data_sct)

        # 终点线判定 (10.0m)
        if s_l['x_pos'] >= 10.00: fin_lqr = True
        if s_p['x_pos'] >= 10.00: fin_prcc = True
        if s_s['x_pos'] >= 10.00: fin_sct = True

        active_x = max(s_l['x_pos'], s_p['x_pos'], s_s['x_pos'])
        v_target = 0.20
        # 终点前平滑缓速 (x > 9.80m)
        if active_x >= 9.80:
            v_target = max(0.04, 0.20 * (10.00 - active_x) / 0.20)
        if active_x >= 10.00:
            v_target = 0.0

        for c in [ctrl_lqr, ctrl_prcc, ctrl_sct]:
            c.set_target_velocity(v_target)

        # 2. 控制器决策与下发
        # [Mode 1: Pure LQR Baseline]
        tau_lqr = ctrl_lqr.compute_lqr(
            pitch=s_l['pitch'], pitch_rate=s_l['pitch_rate'],
            forward_vel=s_l['forward_vel'], left_wheel_vel=s_l['left_wheel_vel'], right_wheel_vel=s_l['right_wheel_vel'],
            current_x=s_l['x_pos'], current_y=s_l['y_pos'], current_yaw=s_l['yaw']
        )
        if fin_lqr:
            tau_lqr['wheel_l'] = -0.008 * s_l['left_wheel_vel']
            tau_lqr['wheel_r'] = -0.008 * s_l['right_wheel_vel']
        data_lqr.ctrl[0] = 0.0
        data_lqr.ctrl[1] = tau_lqr['wheel_l']
        data_lqr.ctrl[2] = 0.0
        data_lqr.ctrl[3] = tau_lqr['wheel_r']

        # [Mode 2: Standard PRCC-RL]
        tau_prcc_prior = ctrl_prcc.compute_lqr(
            pitch=s_p['pitch'], pitch_rate=s_p['pitch_rate'],
            forward_vel=s_p['forward_vel'], left_wheel_vel=s_p['left_wheel_vel'], right_wheel_vel=s_p['right_wheel_vel'],
            current_x=s_p['x_pos'], current_y=s_p['y_pos'], current_yaw=s_p['yaw']
        )
        delta_p = ctrl_prcc.compute_compliance(
            roll=s_p['roll'], roll_rate=s_p['roll_rate'],
            left_hip_pos=s_p['left_hip_pos'], right_hip_pos=s_p['right_hip_pos']
        )
        obs_p = build_single_obs(s_p, v_target, ctrl_prcc, last_act_prcc)
        act_p, _ = model_prcc.predict(obs_p, deterministic=True)
        act_p = np.clip(act_p, -1.0, 1.0)
        last_act_prcc = act_p.copy()
        tau_p_l = delta_p['hip_l'] + act_p[0] * 0.15
        tau_p_r = delta_p['hip_r'] + act_p[1] * 0.15
        wl_p = tau_prcc_prior['wheel_l']
        wr_p = tau_prcc_prior['wheel_r']
        if fin_prcc:
            wl_p = -0.008 * s_p['left_wheel_vel']
            wr_p = -0.008 * s_p['right_wheel_vel']
            tau_p_l = -0.10 * s_p['left_hip_pos']
            tau_p_r = -0.10 * s_p['right_hip_pos']
        data_prcc.ctrl[0] = tau_p_l
        data_prcc.ctrl[1] = wl_p
        data_prcc.ctrl[2] = tau_p_r
        data_prcc.ctrl[3] = wr_p

        # [Mode 3: SCT-RRL Ours]
        tau_sct_prior = ctrl_sct.compute_lqr(
            pitch=s_s['pitch'], pitch_rate=s_s['pitch_rate'],
            forward_vel=s_s['forward_vel'], left_wheel_vel=s_s['left_wheel_vel'], right_wheel_vel=s_s['right_wheel_vel'],
            current_x=s_s['x_pos'], current_y=s_s['y_pos'], current_yaw=s_s['yaw']
        )
        delta_s = ctrl_sct.compute_compliance(
            roll=s_s['roll'], roll_rate=s_s['roll_rate'],
            left_hip_pos=s_s['left_hip_pos'], right_hip_pos=s_s['right_hip_pos']
        )
        e_norm = (tank_sct.E_t - tank_sct.E_min) / (tank_sct.E_max - tank_sct.E_min)
        tank_feat = np.array([e_norm, tank_sct.last_alpha, tank_sct.p_slip_filtered], dtype=np.float32)
        obs_s = build_single_obs(s_s, v_target, ctrl_sct, last_act_sct, tank_feat=tank_feat)
        act_s, _ = model_sct.predict(obs_s, deterministic=True)
        act_s = np.clip(act_s, -1.0, 1.0)
        last_act_sct = act_s.copy()
        tau_des_s = np.array([act_s[0] * 0.15, act_s[1] * 0.15], dtype=np.float32)
        qdot_hip_s = np.array([s_s['left_hip_vel'], s_s['right_hip_vel']], dtype=np.float32)
        slip_s = np.array([abs(s_s['left_wheel_vel'] * R_WHEEL - s_s['forward_vel']),
                           abs(s_s['right_wheel_vel'] * R_WHEEL - s_s['forward_vel'])], dtype=np.float32)
        tau_pass_s, alpha_s = tank_sct.step(tau_des_s, qdot_hip_s, slip_s, dt=dt)
        tau_s_l = delta_s['hip_l'] + tau_pass_s[0]
        tau_s_r = delta_s['hip_r'] + tau_pass_s[1]
        wl_s = tau_sct_prior['wheel_l']
        wr_s = tau_sct_prior['wheel_r']
        if fin_sct:
            wl_s = -0.008 * s_s['left_wheel_vel']
            wr_s = -0.008 * s_s['right_wheel_vel']
            tau_s_l = -0.10 * s_s['left_hip_pos']
            tau_s_r = -0.10 * s_s['right_hip_pos']
        data_sct.ctrl[0] = tau_s_l
        data_sct.ctrl[1] = wl_s
        data_sct.ctrl[2] = tau_s_r
        data_sct.ctrl[3] = wr_s

        # 3. 物理步进
        mujoco.mj_step(model, data_lqr)
        mujoco.mj_step(model, data_prcc)
        mujoco.mj_step(model, data_sct)

        # 4. 周期性渲染并合成三联画卷
        if step % render_interval == 0:
            # 渲染 Panel 1: LQR
            renderer.update_scene(data_lqr, camera=cam)
            img_lqr = Image.fromarray(renderer.render())

            # 渲染 Panel 2: PRCC
            renderer.update_scene(data_prcc, camera=cam)
            img_prcc = Image.fromarray(renderer.render())

            # 渲染 Panel 3: SCT
            renderer.update_scene(data_sct, camera=cam)
            img_sct = Image.fromarray(renderer.render())

            # 拼接 3-Panel 画布: 宽 1200, 高 280 + 130 = 410
            total_w = w_sub * 3
            total_h = h_sub + 130
            canvas = Image.new("RGB", (total_w, total_h), (18, 22, 28))
            canvas.paste(img_lqr, (0, 0))
            canvas.paste(img_prcc, (w_sub, 0))
            canvas.paste(img_sct, (w_sub * 2, 0))

            draw = ImageDraw.Draw(canvas)
            # 分割垂直线
            draw.line([(w_sub, 0), (w_sub, total_h)], fill=(60, 70, 85), width=2)
            draw.line([(w_sub * 2, 0), (w_sub * 2, total_h)], fill=(60, 70, 85), width=2)
            draw.line([(0, h_sub), (total_w, h_sub)], fill=(75, 88, 105), width=2)

            # Panel 1 HUD: LQR
            r_col_l, r_txt_l = format_roll_hud(np.degrees(s_l['roll']), fin_lqr)
            draw.text((12, h_sub + 12), "[Pure LQR Baseline]", fill=(240, 100, 100), font=font_title)
            draw.text((12, h_sub + 36), f"Pos X: {s_l['x_pos']:.2f} m | Vel: {s_l['forward_vel']:.2f} m/s", fill=(210, 215, 220), font=font_hud)
            draw.text((12, h_sub + 56), f"Dev Y: {(s_l['y_pos'] - LINE_Y)*1000:+4.1f} mm", fill=(210, 215, 220), font=font_hud)
            draw.text((12, h_sub + 76), r_txt_l, fill=r_col_l, font=font_hud)
            draw.text((12, h_sub + 96), "State: Rigid Chassis, No Compliance", fill=(170, 175, 180), font=font_small)

            # Panel 2 HUD: PRCC-RL
            r_col_p, r_txt_p = format_roll_hud(np.degrees(s_p['roll']), fin_prcc)
            draw.text((w_sub + 12, h_sub + 12), "[PRCC Residual RL]", fill=(245, 180, 50), font=font_title)
            draw.text((w_sub + 12, h_sub + 36), f"Pos X: {s_p['x_pos']:.2f} m | Vel: {s_p['forward_vel']:.2f} m/s", fill=(210, 215, 220), font=font_hud)
            draw.text((w_sub + 12, h_sub + 56), f"Dev Y: {(s_p['y_pos'] - LINE_Y)*1000:+4.1f} mm", fill=(210, 215, 220), font=font_hud)
            draw.text((w_sub + 12, h_sub + 76), r_txt_p, fill=r_col_p, font=font_hud)
            draw.text((w_sub + 12, h_sub + 96), "State: Prior Compliance + Unbounded RL", fill=(170, 175, 180), font=font_small)

            # Panel 3 HUD: SCT-RRL [Ours]
            r_col_s, r_txt_s = format_roll_hud(np.degrees(s_s['roll']), fin_sct)
            draw.text((w_sub * 2 + 12, h_sub + 12), "[SCT-RRL (Ours)]", fill=(60, 225, 140), font=font_title)
            draw.text((w_sub * 2 + 12, h_sub + 36), f"Pos X: {s_s['x_pos']:.2f} m | Vel: {s_s['forward_vel']:.2f} m/s", fill=(210, 215, 220), font=font_hud)
            draw.text((w_sub * 2 + 12, h_sub + 56), f"Dev Y: {(s_s['y_pos'] - LINE_Y)*1000:+4.1f} mm", fill=(210, 215, 220), font=font_hud)
            draw.text((w_sub * 2 + 12, h_sub + 76), r_txt_s, fill=r_col_s, font=font_hud)
            alpha_str = f"Passivity Scale: {tank_sct.last_alpha:.2f}" if not fin_sct else "Passivity: 1.00 (Parked)"
            draw.text((w_sub * 2 + 12, h_sub + 96), f"Tank E: {tank_sct.E_t*1e4:.1f}e-4 J | {alpha_str}", fill=(100, 215, 255), font=font_small)

            # 顶部关卡全局进度条
            progress_ratio = min(1.0, active_x / 10.00)
            draw.rectangle([(10, 10), (total_w - 10, 22)], fill=(28, 34, 42), outline=(70, 80, 95))
            draw.rectangle([(10, 10), (10 + int((total_w - 20) * progress_ratio), 22)], fill=(50, 150, 255))
            zone_txt = get_zone_name(s_s['x_pos']) if not fin_sct else "Finish Line Reached (Parked)"
            info_txt = (f"Time: {t:4.1f}s | Traversal: {progress_ratio*100:4.1f}% ({active_x:.2f}m / 10.0m) | "
                        f"Terrain: {zone_txt} | Unified Cmd: {v_target:.2f} m/s")
            draw.text((14, h_sub + 18), info_txt, fill=(220, 225, 230), font=font_small)

            frames.append(canvas)

            # 抓取关键帧
            for kf_name, target_pos in key_frame_targets.items():
                if kf_name not in captured_key_frames and active_x >= target_pos:
                    captured_key_frames[kf_name] = canvas.copy()

        # 检查是否全部抵达终点并保持稳定驻留
        if fin_lqr and fin_prcc and fin_sct:
            if not hasattr(record_extended_comparison, '_fin_step'):
                record_extended_comparison._fin_step = step
            elif step - record_extended_comparison._fin_step > int(1.5 / dt):
                print(f"三大算法均顺利穿越 10.5m 终极地狱级赛道并在 10.0m 终点稳定驻留！(t={t:.2f}s)")
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

    # 同步至 Artifact 目录
    artifact_dir = "C:/Users/lenovo/.gemini/antigravity/brain/7b83f22e-ed2b-45ef-8cda-230cc1568684"
    artifact_gif = os.path.join(artifact_dir, "wheel_leg_tri_comparison.gif")
    import shutil
    shutil.copyfile(out_gif, artifact_gif)
    print(f"  [COPIED] 同步至 Artifact 预览目录: {artifact_gif}")

    # 保存关键帧到 Artifact 目录用于高质量静图审阅
    for kf_name, kf_img in captured_key_frames.items():
        kf_path = os.path.join(artifact_dir, f"frame_mega_{kf_name}.png")
        kf_img.save(kf_path)
        print(f"  [SAVED KEYFRAME] {kf_name} -> {kf_path}")


if __name__ == "__main__":
    record_extended_comparison(out_gif="demo/wheel_leg_tri_comparison.gif", fps=10)
