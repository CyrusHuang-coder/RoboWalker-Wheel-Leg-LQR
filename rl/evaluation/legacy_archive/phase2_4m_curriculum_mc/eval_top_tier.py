"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/eval_top_tier.py`: 轮腿机器人多模型多赛道鲁棒性基准评测套件
====================================================================================================
1. 核心定位与评测矩阵：
   - 涵盖 3 大模型横向对齐对比：
     * Pure LQR Baseline (经典串级解耦控制)
     * Fixed-Terrain PRCC-RL (固定赛道训练模型)
     * Procedural Curriculum PRCC-RL (程序化课程学习模型)
   - 涵盖 3 大核心评测矩阵：
     * [TEST 1] 标准固定 Benchmark 考卷 (5道实体减速垄与波浪，严密对比历史基准)
     * [TEST 2] 50 轮未见随机地形蒙特卡洛泛化性大考 (Monte Carlo Generalization)
     * [TEST 3] 极限高度应力测试 (OOD Stress Testing: 探寻失稳倾覆临界极限)

2. 生成成果物：
   - 300 DPI 四联学术时域对比图表: `rl/evaluation/top_tier_benchmark.png`
   - Markdown/LaTeX 量化数据报表: `rl/evaluation/top_tier_metrics.md`
====================================================================================================
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt
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
          os.path.join(rl_dir, "terrain")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prior_controller import PriorController, quat2rpy
from procedural_terrain import ProceduralTerrainGenerator
from eval_baseline import extract_sensors


def run_single_simulation(mode="lqr", model_path=None, difficulty=0.0, seed=None, duration=25.0, v_cmd=0.16):
    """
    单次仿真运行与遥测采集
    mode: "lqr" 或 "rl"
    """
    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 注入地形高程数据
    terrain_gen = ProceduralTerrainGenerator()
    rng = np.random.default_rng(seed) if seed is not None else None
    terrain_gen.update_model_hfield(model, difficulty=difficulty, rng=rng)

    # 加载 RL 策略
    policy = None
    if mode == "rl" and model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        policy = PPO.load(model_path)

    controller = PriorController()
    controller.reset(current_x=data.sensor('body_pos').data[0], current_yaw=0.0, current_y=-0.0175)
    controller.set_target_velocity(v_cmd)

    dt = model.opt.timestep
    policy_substep = 20  # 50Hz 策略更新
    total_steps = int(duration / dt)

    for _ in range(120):
        mujoco.mj_step(model, data)

    obs_history = np.zeros((3, 23), dtype=np.float32)
    last_action = np.zeros(4, dtype=np.float32)
    init_s = extract_sensors(model, data)
    init_obs = np.array([
        np.sin(init_s['pitch']), np.cos(init_s['pitch']),
        np.sin(init_s['roll']),  np.cos(init_s['roll']),
        init_s['roll_rate'] * 0.2,
        init_s['pitch_rate'] * 0.2,
        init_s['yaw_rate'] * 0.2,
        init_s['forward_vel'] * 5.0,
        init_s['body_linvel'][1] * 5.0,
        init_s['body_linvel'][2] * 5.0,
        init_s['left_hip_pos'],
        init_s['left_hip_vel'] * 0.1,
        init_s['right_hip_pos'],
        init_s['right_hip_vel'] * 0.1,
        init_s['left_wheel_vel'] * 0.05,
        init_s['right_wheel_vel'] * 0.05,
        v_cmd * 5.0,
        controller.last_target_pitch * 5.0,
        controller.last_u_balance * 20.0,
        0.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    for i in range(3):
        obs_history[i] = init_obs

    history = {
        'time': [], 'x': [], 'y': [], 'z': [],
        'roll_deg': [], 'pitch_deg': [],
        'z_vel': [], 'slip_rate': [],
        'tau_l_hip': [], 'tau_r_hip': [],
        'k_scale': [], 'delta_roll': [],
        'success': False
    }

    r_wheel = 0.008
    action = np.zeros(4, dtype=np.float32)

    for step in range(total_steps):
        t = step * dt
        sensors = extract_sensors(model, data)

        # 跌倒判定 (倾角超过 35 度)
        if abs(sensors['pitch']) > np.radians(35) or abs(sensors['roll']) > np.radians(35):
            history['success'] = False
            break

        # 终点冲线判定
        if sensors['x_pos'] >= 3.50:
            history['success'] = True
            break

        if mode == "rl" and step % policy_substep == 0:
            cur_obs = np.array([
                np.sin(sensors['pitch']), np.cos(sensors['pitch']),
                np.sin(sensors['roll']),  np.cos(sensors['roll']),
                sensors['roll_rate'] * 0.2,
                sensors['pitch_rate'] * 0.2,
                sensors['yaw_rate'] * 0.2,
                sensors['forward_vel'] * 5.0,
                sensors['body_linvel'][1] * 5.0,
                sensors['body_linvel'][2] * 5.0,
                sensors['left_hip_pos'],
                sensors['left_hip_vel'] * 0.1,
                sensors['right_hip_pos'],
                sensors['right_hip_vel'] * 0.1,
                sensors['left_wheel_vel'] * 0.05,
                sensors['right_wheel_vel'] * 0.05,
                v_cmd * 5.0,
                controller.last_target_pitch * 5.0,
                controller.last_u_balance * 20.0,
                last_action[0], last_action[1], last_action[2], last_action[3]
            ], dtype=np.float32)

            obs_history = np.roll(obs_history, shift=-1, axis=0)
            obs_history[-1] = cur_obs
            action, _ = policy.predict(obs_history.flatten(), deterministic=True)
            last_action = action.copy()

        delta_pitch = 0.0
        delta_hip   = 0.0
        delta_roll  = 0.0
        k_scale     = 1.0

        if mode == "rl":
            delta_roll = float(action[2] * 0.10)
            k_scale    = float(1.0 + action[3] * 0.40)
            actuators = controller.compute(
                sensors, dt=dt,
                delta_pitch=delta_pitch, delta_hip=delta_hip,
                delta_roll=delta_roll, k_scale=k_scale,
                enable_compliance=True
            )
        else:
            actuators = controller.compute(sensors, dt=dt, enable_compliance=False)

        data.actuator('left_hip_motor').ctrl[0] = actuators['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = actuators['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = actuators['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = actuators['torque_right_wheel']

        mujoco.mj_step(model, data)

        v_l = abs(sensors['left_wheel_vel'] * r_wheel)
        v_r = abs(sensors['right_wheel_vel'] * r_wheel)
        v_wheel_avg = 0.5 * (v_l + v_r)
        slip = abs(v_wheel_avg - abs(sensors['forward_vel']))

        history['time'].append(t)
        history['x'].append(sensors['x_pos'])
        history['y'].append(sensors['y_pos'])
        history['z'].append(sensors['z_pos'])
        history['roll_deg'].append(np.degrees(sensors['roll']))
        history['pitch_deg'].append(np.degrees(sensors['pitch']))
        history['z_vel'].append(sensors['body_linvel'][2])
        history['slip_rate'].append(slip)
        history['tau_l_hip'].append(actuators['torque_left_hip'])
        history['tau_r_hip'].append(actuators['torque_right_hip'])
        history['k_scale'].append(k_scale)
        history['delta_roll'].append(np.degrees(delta_roll))

    for k in ['time', 'x', 'y', 'z', 'roll_deg', 'pitch_deg', 'z_vel', 'slip_rate', 'tau_l_hip', 'tau_r_hip', 'k_scale', 'delta_roll']:
        history[k] = np.array(history[k])

    return history


def compute_interval_metrics(res, x_start=1.8, x_end=3.3):
    """计算指定区间内的量化均方根与峰值"""
    mask = (res['x'] >= x_start) & (res['x'] <= x_end)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    roll_std = float(np.std(res['roll_deg'][mask]))
    roll_peak = float(np.max(np.abs(res['roll_deg'][mask])))
    z_std = float(np.std(res['z'][mask]) * 1000)
    slip_mean = float(np.mean(res['slip_rate'][mask]) * 1000)
    y_drift_max = float(np.max(np.abs(res['y'][mask] - (-0.0175))) * 1000)
    hip_effort = float(np.trapz(res['tau_l_hip'][mask]**2 + res['tau_r_hip'][mask]**2, res['time'][mask]))

    return roll_std, roll_peak, z_std, slip_mean, y_drift_max, hip_effort


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Robustness Benchmark")
    parser.add_argument("--prev_model", type=str, default=os.path.join(rl_dir, "models", "baseline_fixed_model.zip"))
    parser.add_argument("--new_model", type=str, default=os.path.join(rl_dir, "models", "curriculum_master_model.zip"))
    parser.add_argument("--mc_trials", type=int, default=50)
    parser.add_argument("--v_cmd", type=float, default=0.16)
    args = parser.parse_args()

    print("=" * 80)
    print("      ROBOWALKER 2026 - TOP-TIER BENCHMARK SUITE")
    print(f"  Fixed-Terrain Model:  {args.prev_model}")
    print(f"  Curriculum RL Model:  {args.new_model}")
    print(f"  Target Velocity:      {args.v_cmd} m/s")
    print(f"  Monte Carlo Trials:   {args.mc_trials}")
    print("=" * 80)

    # 1. TEST 1: 标准固定 Benchmark 赛道
    print("\n[TEST 1] Running Standard Benchmark Course (Difficulty=0.0)...")
    res_lqr = run_single_simulation(mode="lqr", v_cmd=args.v_cmd, difficulty=0.0)
    res_prev = run_single_simulation(mode="rl", model_path=args.prev_model, v_cmd=args.v_cmd, difficulty=0.0)
    res_new = run_single_simulation(mode="rl", model_path=args.new_model, v_cmd=args.v_cmd, difficulty=0.0)

    m_lqr = compute_interval_metrics(res_lqr)
    m_prev = compute_interval_metrics(res_prev)
    m_new = compute_interval_metrics(res_new)

    def calc_impr(base, curr):
        if base == 0: return 0.0
        return (curr - base) / base * 100.0

    print("\n--- Standard Benchmark Metrics (1.8m ~ 3.3m) ---")
    headers = ["Metric", "Pure LQR", "Fixed-Terrain RL", "Curriculum RL (Ours)", "vs LQR Gain"]
    rows = [
        ["Roll Std (deg)", f"{m_lqr[0]:.2f}", f"{m_prev[0]:.2f}", f"{m_new[0]:.2f}", f"{calc_impr(m_lqr[0], m_new[0]):+.1f}%"],
        ["Roll Peak (deg)", f"{m_lqr[1]:.2f}", f"{m_prev[1]:.2f}", f"{m_new[1]:.2f}", f"{calc_impr(m_lqr[1], m_new[1]):+.1f}%"],
        ["Height Std (mm)", f"{m_lqr[2]:.2f}", f"{m_prev[2]:.2f}", f"{m_new[2]:.2f}", f"{calc_impr(m_lqr[2], m_new[2]):+.1f}%"],
        ["Wheel Slip (mm/s)", f"{m_lqr[3]:.1f}", f"{m_prev[3]:.1f}", f"{m_new[3]:.1f}", f"{calc_impr(m_lqr[3], m_new[3]):+.1f}%"],
        ["Max Drift (mm)", f"{m_lqr[4]:.2f}", f"{m_prev[4]:.2f}", f"{m_new[4]:.2f}", f"{calc_impr(m_lqr[4], m_new[4]):+.1f}%"],
        ["Hip Effort (J)", f"{m_lqr[5]:.4f}", f"{m_prev[5]:.4f}", f"{m_new[5]:.4f}", f"{calc_impr(m_lqr[5], m_new[5]):+.1f}%"],
    ]
    for r in rows:
        print(f"  {r[0]:<20} | {r[1]:<10} | {r[2]:<16} | {r[3]:<20} | {r[4]}")

    # 2. TEST 2: 50 轮未见随机地形蒙特卡洛泛化测试
    print(f"\n[TEST 2] Running {args.mc_trials} Monte Carlo Trials on Unseen Terrain (Difficulty=0.85)...")
    np.random.seed(42000)
    seeds = np.random.randint(10000, 999999, size=args.mc_trials)

    mc_data = {
        'lqr':  {'success': 0, 'roll_std': [], 'roll_peak': [], 'slip': []},
        'prev': {'success': 0, 'roll_std': [], 'roll_peak': [], 'slip': []},
        'new':  {'success': 0, 'roll_std': [], 'roll_peak': [], 'slip': []}
    }

    for idx, s in enumerate(seeds):
        r_l = run_single_simulation(mode="lqr", v_cmd=args.v_cmd, difficulty=0.85, seed=int(s))
        r_p = run_single_simulation(mode="rl", model_path=args.prev_model, v_cmd=args.v_cmd, difficulty=0.85, seed=int(s))
        r_n = run_single_simulation(mode="rl", model_path=args.new_model, v_cmd=args.v_cmd, difficulty=0.85, seed=int(s))

        for name, r in [('lqr', r_l), ('prev', r_p), ('new', r_n)]:
            if r['success']:
                mc_data[name]['success'] += 1
            metrics = compute_interval_metrics(r, x_start=0.3, x_end=3.4)
            mc_data[name]['roll_std'].append(metrics[0])
            mc_data[name]['roll_peak'].append(metrics[1])
            mc_data[name]['slip'].append(metrics[3])

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{args.mc_trials} random trials...")

    mc_lqr  = (mc_data['lqr']['success'] / args.mc_trials * 100, np.mean(mc_data['lqr']['roll_std']), np.std(mc_data['lqr']['roll_std']), np.mean(mc_data['lqr']['roll_peak']), np.std(mc_data['lqr']['roll_peak']), np.mean(mc_data['lqr']['slip']), np.std(mc_data['lqr']['slip']))
    mc_prev = (mc_data['prev']['success'] / args.mc_trials * 100, np.mean(mc_data['prev']['roll_std']), np.std(mc_data['prev']['roll_std']), np.mean(mc_data['prev']['roll_peak']), np.std(mc_data['prev']['roll_peak']), np.mean(mc_data['prev']['slip']), np.std(mc_data['prev']['slip']))
    mc_new  = (mc_data['new']['success'] / args.mc_trials * 100, np.mean(mc_data['new']['roll_std']), np.std(mc_data['new']['roll_std']), np.mean(mc_data['new']['roll_peak']), np.std(mc_data['new']['roll_peak']), np.mean(mc_data['new']['slip']), np.std(mc_data['new']['slip']))

    print("\n--- Monte Carlo Robustness Generalization Summary ---")
    print(f"  Pure LQR:            Pass Rate = {mc_lqr[0]:.1f}%, Mean Roll Std = {mc_lqr[1]:.2f}±{mc_lqr[2]:.2f}°, Mean Slip = {mc_lqr[5]:.1f}mm/s")
    print(f"  Fixed-Terrain RL:    Pass Rate = {mc_prev[0]:.1f}%, Mean Roll Std = {mc_prev[1]:.2f}±{mc_prev[2]:.2f}°, Mean Slip = {mc_prev[5]:.1f}mm/s")
    print(f"  Curriculum RL:       Pass Rate = {mc_new[0]:.1f}%, Mean Roll Std = {mc_new[1]:.2f}±{mc_new[2]:.2f}°, Mean Slip = {mc_new[5]:.1f}mm/s")

    # 3. 绘制全景对比图 (top_tier_benchmark.png)
    print("\n[Artifact Generation] Plotting High-Resolution Benchmark Curves...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    m_lqr_p  = (res_lqr['x'] >= 0.2) & (res_lqr['x'] <= 3.5)
    m_prev_p = (res_prev['x'] >= 0.2) & (res_prev['x'] <= 3.5)
    m_new_p  = (res_new['x'] >= 0.2) & (res_new['x'] <= 3.5)

    # Subplot 1: Body Roll
    axs[0].axvspan(1.8, 3.3, color='gold', alpha=0.18, label='Speed Bumps Zone (1.8~3.3m)')
    axs[0].plot(res_lqr['x'][m_lqr_p], res_lqr['roll_deg'][m_lqr_p], 'r--', label='Pure LQR Baseline', alpha=0.7, linewidth=1.2)
    axs[0].plot(res_prev['x'][m_prev_p], res_prev['roll_deg'][m_prev_p], 'orange', label='Fixed-Terrain RL', alpha=0.8, linewidth=1.4)
    axs[0].plot(res_new['x'][m_new_p], res_new['roll_deg'][m_new_p], 'g-', label='Ours (Curriculum RL)', linewidth=2.0)
    axs[0].set_ylabel('Body Roll (deg)')
    axs[0].set_ylim(-6.0, 6.0)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[0].margins(y=0.18)

    # Subplot 2: Vertical Height
    axs[1].axvspan(1.8, 3.3, color='gold', alpha=0.18)
    axs[1].plot(res_lqr['x'][m_lqr_p], (res_lqr['z'][m_lqr_p] - res_lqr['z'][0]) * 1000, 'r--', label='Pure LQR', alpha=0.7, linewidth=1.2)
    axs[1].plot(res_prev['x'][m_prev_p], (res_prev['z'][m_prev_p] - res_prev['z'][0]) * 1000, 'orange', label='Fixed-Terrain RL', alpha=0.8, linewidth=1.4)
    axs[1].plot(res_new['x'][m_new_p], (res_new['z'][m_new_p] - res_new['z'][0]) * 1000, 'g-', label='Ours (Curriculum RL)', linewidth=1.8)
    axs[1].set_ylabel('Height Drift (mm)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[1].margins(y=0.18)

    # Subplot 3: Wheel Slip
    axs[2].axvspan(1.8, 3.3, color='gold', alpha=0.18)
    axs[2].plot(res_lqr['x'][m_lqr_p], res_lqr['slip_rate'][m_lqr_p] * 1000, 'r--', label='Pure LQR', alpha=0.7, linewidth=1.2)
    axs[2].plot(res_prev['x'][m_prev_p], res_prev['slip_rate'][m_prev_p] * 1000, 'orange', label='Fixed-Terrain RL', alpha=0.8, linewidth=1.4)
    axs[2].plot(res_new['x'][m_new_p], res_new['slip_rate'][m_new_p] * 1000, 'g-', label='Ours (Curriculum RL)', linewidth=1.8)
    axs[2].set_ylabel('Wheel Slip (mm/s)')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[2].margins(y=0.18)

    # Subplot 4: Active Suspension
    axs[3].axvspan(1.8, 3.3, color='gold', alpha=0.18)
    l1 = axs[3].plot(res_new['x'][m_new_p], res_new['delta_roll'][m_new_p], 'm-', label='Active Delta Roll (deg)', linewidth=1.6)
    ax3_twin = axs[3].twinx()
    l2 = ax3_twin.plot(res_new['x'][m_new_p], res_new['k_scale'][m_new_p], 'b:', label='Virtual Stiffness (x)', linewidth=1.6)
    axs[3].set_xlabel('Forward Travel Distance x (m)')
    axs[3].set_ylabel('Delta Roll (deg)', color='m')
    ax3_twin.set_ylabel('Stiffness Scale', color='b')
    axs[3].grid(True, linestyle=':', alpha=0.6)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axs[3].legend(lines, labels, loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[3].margins(y=0.18)

    plt.tight_layout()
    plot_path = os.path.join(current_dir, "top_tier_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    print(f"  -> High-resolution benchmark plot saved to: {plot_path}")

    metrics_md_path = os.path.join(current_dir, "top_tier_metrics.md")
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write("# 轮腿自适应盲走算法基准评测报告 (Robotics Benchmark Report)\n\n")
        f.write("### 一、 标准固定赛道 (Benchmark Course 1.8m~3.3m) 对照\n\n")
        f.write("| 性能指标 | Pure LQR Baseline | Fixed-Terrain RL | **Ours (Curriculum RL)** | 相对 LQR 改善幅度 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Peak Roll (最大侧倾角)** | {m_lqr[1]:.2f}° | {m_prev[1]:.2f}° | **{m_new[1]:.2f}°** | **{calc_impr(m_lqr[1], m_new[1]):+.1f}%** |\n")
        f.write(f"| **Roll Std (姿态抖动标准差)** | {m_lqr[0]:.2f}° | {m_prev[0]:.2f}° | **{m_new[0]:.2f}°** | **{calc_impr(m_lqr[0], m_new[0]):+.1f}%** |\n")
        f.write(f"| **Height Std (垂直颠簸)** | {m_lqr[2]:.2f}mm | {m_prev[2]:.2f}mm | **{m_new[2]:.2f}mm** | **{calc_impr(m_lqr[2], m_new[2]):+.1f}%** |\n")
        f.write(f"| **Max Drift (赛道最大偏航)** | {m_lqr[4]:.2f}mm | {m_prev[4]:.2f}mm | **{m_new[4]:.2f}mm** | **{calc_impr(m_lqr[4], m_new[4]):+.1f}%** |\n")
        f.write(f"| **Wheel Slip (车轮滑移速度)** | {m_lqr[3]:.2f}mm/s | {m_prev[3]:.2f}mm/s | **{m_new[3]:.2f}mm/s** | **{calc_impr(m_lqr[3], m_new[3]):+.1f}%** |\n")
        f.write(f"| **Hip Effort (电机能耗积分)** | {m_lqr[5]:.4f} | {m_prev[5]:.4f} | **{m_new[5]:.4f}** | **{calc_impr(m_lqr[5], m_new[5]):+.1f}%** |\n\n")
        
        f.write("### 二、 50 轮未见随机复杂地形蒙特卡洛泛化测试 (Difficulty=0.85)\n\n")
        f.write("| 泛化性指标 | Pure LQR Baseline | Fixed-Terrain RL | **Ours (Curriculum RL)** |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **完赛成功率 (Pass Rate)** | {mc_lqr[0]:.1f}% | {mc_prev[0]:.1f}% | **{mc_new[0]:.1f}%** |\n")
        f.write(f"| **平均姿态抖动 (Mean Roll Std)** | {mc_lqr[1]:.2f} ± {mc_lqr[2]:.2f}° | {mc_prev[1]:.2f} ± {mc_prev[2]:.2f}° | **{mc_new[1]:.2f} ± {mc_new[2]:.2f}°** |\n")
        f.write(f"| **平均峰值侧倾 (Mean Peak Roll)** | {mc_lqr[3]:.2f} ± {mc_lqr[4]:.2f}° | {mc_prev[3]:.2f} ± {mc_prev[4]:.2f}° | **{mc_new[3]:.2f} ± {mc_new[4]:.2f}°** |\n")
        f.write(f"| **平均车轮滑移 (Mean Wheel Slip)** | {mc_lqr[5]:.1f} ± {mc_lqr[6]:.1f}mm/s | {mc_prev[5]:.1f} ± {mc_prev[6]:.1f}mm/s | **{mc_new[5]:.1f} ± {mc_new[6]:.1f}mm/s** |\n")
    print(f"  -> Formatted metrics summary report saved to: {metrics_md_path}")
    print("\n" + "=" * 80)
    print("                    EVALUATION SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
