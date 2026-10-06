"""
sim_runner.py: 统一单回合仿真步进与遥测采集引擎 (Simulation Runner & Telemetry Logger)
========================================================================================
【模块功能介绍】
本模块作为算法评估与策略测试的核心步进引擎，统一封装了轮腿机器人多种控制模式下的
单回合仿真流程与全量状态遥测记录功能：

1. 跨算法统一受试协议 (Standardized Benchmarking Protocol):
   支持三大基准模式在控制变量下公平对比：
   - "lqr"      : 纯刚性 LQR 基线（关闭主动顺应律，固化名义弹簧阻尼）
   - "lqr_comp" : 仅手工先验 PD 顺应律（用于严格消融强化学习残差贡献）
   - "rl"       : 完整残差强化学习策略（支持可选搭载 SCT 无源能量储罐）

2. 多源高精遥测记录 (Full-State Telemetry Logging):
   步步采集并导出机体三维轨迹 (x, y, z)、欧拉角姿态 (roll, pitch)、前向速度 vx、
   运动学滑移率 (kinematic_slip)、接触点真实摩擦滑移 (slip_true)、关节力矩与机械功耗、
   能量储罐荷电状态 (E_T) 与做功增量 (dW)。

3. 时序观测构建 (Rolling Observation History):
   提供 build_single_obs 函数，将单帧本体感知特征 (23维) 与可选储罐特征 (3维)
   打包输出，支持策略网络捕捉角加速度、冲击微分与趋势。
========================================================================================
"""
import os
import sys
import numpy as np
import mujoco

# 路径自适应引导：无论从何处运行，均能正确载入控制器与环境模块
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
         os.path.join(rl_dir, "controllers"),
         os.path.join(rl_dir, "terrain"),
         os.path.join(rl_dir, "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prior_controller import PriorController
from procedural_terrain import ProceduralTerrainGenerator
from eval_baseline import extract_sensors
from metrics_utils import ContactSlipMeter

# 机器人与赛道物理常数标定
R_WHEEL = 0.008   # 车轮有效接触滚动半径 (m)
L_LEG   = 0.040   # 摆动腿等效几何长度 (m)
LINE_Y  = -0.0175 # 双轮几何对称巡线基准 Y 坐标 (m)


def build_single_obs(s, v_cmd, controller, last_action, tank_feat=None):
    """
    单帧观测向量构建 (Proprioceptive Observation Builder)
    输入:
      - s: extract_sensors 提取的传感器字典
      - v_cmd: 目标巡航航速指令 (m/s)
      - controller: PriorController 实例 (提取前倾角与平衡力矩缓存)
      - last_action: 上一步动作向量 a_{t-1}
      - tank_feat: 可选储罐特征 [E_norm, alpha*, P_slip]
    输出:
      - np.ndarray: 23维 (无储罐) 或 26维 (带储罐) 归一化浮点特征数组
    """
    obs = [
        # 1. 机体姿态三角函数特征 (保证 360° 无跳变，权重 1.0)
        np.sin(s['pitch']), np.cos(s['pitch']),
        np.sin(s['roll']), np.cos(s['roll']),
        # 2. 机体空间角速度 (缩放 0.2)
        s['roll_rate'] * 0.2, s['pitch_rate'] * 0.2, s['yaw_rate'] * 0.2,
        # 3. 机体线速度 (前向速度与其他轴线速度，缩放 5.0)
        s['forward_vel'] * 5.0, s['body_linvel'][1] * 5.0, s['body_linvel'][2] * 5.0,
        # 4. 髋关节位置与角速度
        s['left_hip_pos'], s['left_hip_vel'] * 0.1,
        s['right_hip_pos'], s['right_hip_vel'] * 0.1,
        # 5. 轮毂电机旋转速度
        s['left_wheel_vel'] * 0.05, s['right_wheel_vel'] * 0.05,
        # 6. 任务指令与先验控制内部状态
        v_cmd * 5.0,
        controller.last_target_pitch * 5.0,
        controller.last_u_balance * 20.0,
        # 7. 上一步执行动作 (动作平滑性约束)
        last_action[0], last_action[1], last_action[2], last_action[3],
    ]
    if tank_feat is not None:
        obs.extend(list(tank_feat))
    return np.array(obs, dtype=np.float32)


def kinematic_slip(s):
    """
    运动学近似滑移率计算 (Kinematic Slip Approximation)
    通过轮缘线速度与腿部足端前向几何速度之差，估计车轮的相对滑移剧烈程度。
    """
    v_lw = abs(s['left_wheel_vel'] * R_WHEEL)
    v_rw = abs(s['right_wheel_vel'] * R_WHEEL)
    v_hl = abs(s['forward_vel'] + s['left_hip_vel'] * L_LEG * np.cos(s['left_hip_pos']))
    v_hr = abs(s['forward_vel'] + s['right_hip_vel'] * L_LEG * np.cos(s['right_hip_pos']))
    return 0.5 * (abs(v_lw - v_hl) + abs(v_rw - v_hr))


def run_episode(mode="lqr", policy=None, difficulty=0.0, seed=None, duration=28.0,
                v_cmd=0.16, x_finish=3.50, tank=None, obs_history_len=3,
                tank_obs=False, random_policy=False, rng_policy=None):
    """
    单回合仿真评估主循环
    输入:
      - mode: 控制模式 ("lqr", "lqr_comp", "rl")
      - policy: 策略模型 (Stable-Baselines3 导出的策略)
      - difficulty: 随机地形难度等级 (0.0 ~ 1.0)
      - duration: 最大仿真时长 (s)
      - v_cmd: 目标巡航航速 (m/s)
      - x_finish: 通关目标终点距离 (m)
      - tank: SlipCoupledEnergyTank 实例 (若为 None 则无储罐约束)
    输出:
      - h: 包含全量时域信号与最终成功标志 (success, fell) 的字典
    """
    # 动态定位地形 XML 资源
    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    if not os.path.exists(xml_path):
        xml_path = "rl/terrain/wheel_leg_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    gen = ProceduralTerrainGenerator()
    rng = np.random.default_rng(seed) if seed is not None else None
    gen.update_model_hfield(model, difficulty=difficulty, rng=rng)

    controller = PriorController()
    controller.reset(current_x=data.sensor('body_pos').data[0], current_yaw=0.0, current_y=LINE_Y)
    controller.set_target_velocity(v_cmd)
    meter = ContactSlipMeter(model)

    dt = model.opt.timestep
    substep = 20  # 50Hz 策略步 (1kHz 物理步 / 20)

    # 初始预热与静置沉降
    for _ in range(120):
        mujoco.mj_step(model, data)

    if tank is not None:
        tank.reset()

    last_action = np.zeros(4, dtype=np.float32)
    action = np.zeros(4, dtype=np.float32)
    s0 = extract_sensors(model, data)
    tf0 = tank.features() if (tank is not None and tank_obs) else None
    o0 = build_single_obs(s0, v_cmd, controller, last_action, tf0)
    obs_hist = np.tile(o0, (obs_history_len, 1))

    keys = ['time', 'x', 'y', 'z', 'roll_deg', 'pitch_deg', 'vx', 'slip_kin', 'slip_true',
            'tau_l_hip', 'tau_r_hip', 'p_hip_mech', 'k_scale', 'delta_roll',
            'E_T', 'alpha', 'P_slip', 'dW']
    h = {k: [] for k in keys}
    h['success'] = False
    h['fell'] = False

    k_exec, dr_exec = 1.0, 0.0
    alpha = 1.0
    dW = 0.0
    total = int(duration / dt)

    for step in range(total):
        s = extract_sensors(model, data)

        # 跌倒与倾覆保护判定 (超 32° 即判负)
        if abs(s['pitch']) > np.radians(32) or abs(s['roll']) > np.radians(32):
            h['fell'] = True
            break
        # 成功抵达目标终点
        if s['x_pos'] >= x_finish:
            h['success'] = True
            break

        # 策略推理步 (50Hz / 20ms 一步)
        if step % substep == 0:
            if mode == "rl":
                if tank is not None and step > 0:
                    tank.commit()
                tf = tank.features() if (tank is not None and tank_obs) else None
                o = build_single_obs(s, v_cmd, controller, last_action, tf)
                obs_hist = np.roll(obs_hist, -1, axis=0)
                obs_hist[-1] = o
                if random_policy:
                    action = rng_policy.uniform(-1, 1, size=4).astype(np.float32)
                else:
                    action, _ = policy.predict(obs_hist.flatten(), deterministic=True)
                last_action = np.array(action, dtype=np.float32).copy()
                k_req = float(1.0 + action[3] * 0.40)
                dr_req = float(action[2] * 0.10)
                if tank is not None:
                    k_exec, dr_exec, alpha, dW = tank.project(controller, s, k_req, dr_req)
                else:
                    k_exec, dr_exec, alpha, dW = k_req, dr_req, 1.0, 0.0

        if mode == "rl":
            comp, k_use, dr_use = True, k_exec, dr_exec
        elif mode == "lqr_comp":
            comp, k_use, dr_use = True, 1.0, 0.0
        else:
            comp, k_use, dr_use = False, 1.0, 0.0

        # 底层动力学解算与力矩注入 (1kHz)
        act = controller.compute(s, dt=dt, delta_pitch=0.0, delta_hip=0.0,
                                 delta_roll=dr_use, k_scale=k_use, enable_compliance=comp)
        data.actuator('left_hip_motor').ctrl[0] = act['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = act['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = act['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = act['torque_right_wheel']
        mujoco.mj_step(model, data)

        # 接触滑移功率度量与能量储罐积分
        sl, sr, _, _ = meter.measure(data)
        p_slip = (abs(act['torque_left_wheel']) * sl + abs(act['torque_right_wheel']) * sr) / R_WHEEL
        if tank is not None and mode == "rl":
            tank.accumulate(controller, s, p_slip, dt)

        # 遥测记录
        h['time'].append(step * dt)
        h['x'].append(s['x_pos']); h['y'].append(s['y_pos']); h['z'].append(s['z_pos'])
        h['roll_deg'].append(np.degrees(s['roll'])); h['pitch_deg'].append(np.degrees(s['pitch']))
        h['vx'].append(s['forward_vel'])
        h['slip_kin'].append(kinematic_slip(s))
        h['slip_true'].append(0.5 * (sl + sr))
        h['tau_l_hip'].append(act['torque_left_hip']); h['tau_r_hip'].append(act['torque_right_hip'])
        h['p_hip_mech'].append(abs(act['torque_left_hip'] * s['left_hip_vel']) +
                               abs(act['torque_right_hip'] * s['right_hip_vel']))
        h['k_scale'].append(k_use); h['delta_roll'].append(np.degrees(dr_use))
        h['E_T'].append(tank.E if tank is not None else np.nan)
        p_slip_log = tank.p_slip_filtered if tank is not None else p_slip
        h['alpha'].append(alpha); h['P_slip'].append(p_slip_log); h['dW'].append(dW)

    for k in keys:
        h[k] = np.asarray(h[k])
    return h
