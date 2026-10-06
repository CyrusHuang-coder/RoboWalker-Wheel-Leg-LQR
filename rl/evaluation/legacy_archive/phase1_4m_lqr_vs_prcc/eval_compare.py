"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/eval_compare.py`: 经典 Pure LQR 与 PRCC 残差强化学习成对学术时域对比评测
====================================================================================================
1. 核心定位与对比功能：
   - 在 4.0m 标准起伏+非对称单侧障碍地形上，分别执行纯 LQR 与 PRCC-RL。
   - 采集全生命周期高精遥测数据 (横滚角 Roll, 俯仰角 Pitch, 垂直颠簸 z_acc, 打滑率 slip)。
   - 输出清晰的 ASCII 量化对比表格 (波动削减幅度 %)。
   - 绘制高保真三联时域对比曲线图: `rl/evaluation/evaluation_comparison.png`。
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
from eval_baseline import extract_sensors


def run_episode(mode="lqr", model_path=None, duration=25.0, v_cmd=0.16):
    """
    运行单次测试评估
    mode: "lqr" 或 "residual_rl"
    """
    if model_path is None:
        model_path = os.path.join(rl_dir, "models", "best_model.zip")

    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    rl_policy = None
    if mode == "residual_rl":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        rl_policy = PPO.load(model_path)

    controller = PriorController()
    controller.reset(current_x=data.sensor('body_pos').data[0], current_yaw=0.0, current_y=-0.0175)
    controller.set_target_velocity(v_cmd)

    dt = model.opt.timestep
    policy_substep = 20  # 50Hz 策略更新
    total_steps = int(duration / dt)

    for _ in range(30):
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
        'time': [],
        'x': [],
        'y': [],
        'z': [],
        'roll_deg': [],
        'pitch_deg': [],
        'z_vel': [],
        'slip_rate': [],
        'tau_l_hip': [],
        'tau_r_hip': [],
        'tau_wheel': [],
        'k_scale': [],
        'delta_roll': []
    }

    r_wheel = 0.008
    action = np.zeros(4, dtype=np.float32)

    for step in range(total_steps):
        t = step * dt
        sensors = extract_sensors(model, data)

        if abs(sensors['pitch']) > np.radians(35) or abs(sensors['roll']) > np.radians(35):
            print(f"[{mode.upper()}] Robot tipped over at t={t:.3f}s, x={sensors['x_pos']:.3f}m")
            break

        if sensors['x_pos'] >= 3.50:
            print(f"[{mode.upper()}] Cleared obstacle course smoothly! Reached x={sensors['x_pos']:.3f}m")
            break

        if mode == "residual_rl" and step % policy_substep == 0:
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
            action, _ = rl_policy.predict(obs_history.flatten(), deterministic=True)
            last_action = action.copy()

        delta_pitch = 0.0
        delta_hip = 0.0
        delta_roll = 0.0
        k_scale = 1.0

        if mode == "residual_rl":
            delta_roll = float(action[2] * 0.10)
            k_scale = float(1.0 + action[3] * 0.40)
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
        history['tau_wheel'].append(0.5 * (abs(actuators['torque_left_wheel']) + abs(actuators['torque_right_wheel'])))
        history['k_scale'].append(k_scale)
        history['delta_roll'].append(np.degrees(delta_roll))

    return {k: np.array(v) for k, v in history.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(rl_dir, "models", "best_model.zip"))
    parser.add_argument("--v_cmd", type=float, default=0.16)
    args = parser.parse_args()

    print(f"================================================================")
    print(f"  RoboWalker 2026 - Benchmark Evaluation: LQR vs PRCC-RL")
    print(f"  Target Velocity: {args.v_cmd} m/s | Model: {args.model}")
    print(f"================================================================")

    print("\nRunning Pure LQR Baseline...")
    res_lqr = run_episode(mode="lqr", duration=25.0, v_cmd=args.v_cmd)

    print("\nRunning PRCC Residual RL...")
    res_rl = run_episode(mode="residual_rl", model_path=args.model, duration=25.0, v_cmd=args.v_cmd)

    mask_lqr = (res_lqr['x'] >= 1.8) & (res_lqr['x'] <= 3.3)
    mask_rl  = (res_rl['x'] >= 1.8) & (res_rl['x'] <= 3.3)

    metrics = {}
    for name, res, m in [("LQR", res_lqr, mask_lqr), ("RL", res_rl, mask_rl)]:
        metrics[name] = {
            'roll_std': np.std(res['roll_deg'][m]) if m.any() else np.nan,
            'roll_peak': np.max(np.abs(res['roll_deg'][m])) if m.any() else np.nan,
            'pitch_std': np.std(res['pitch_deg'][m]) if m.any() else np.nan,
            'pitch_peak': np.max(np.abs(res['pitch_deg'][m])) if m.any() else np.nan,
            'z_std': np.std(res['z'][m]) * 1000 if m.any() else np.nan,
            'slip_mean': np.mean(res['slip_rate'][m]) if m.any() else np.nan,
            'tau_hip_rms': np.sqrt(np.mean(res['tau_l_hip'][m]**2 + res['tau_r_hip'][m]**2)) if m.any() else np.nan,
            'final_x': res['x'][-1]
        }

    print("\n" + "=" * 68)
    print("           BENCHMARK QUANTITATIVE COMPARISON REPORT")
    print("=" * 68)
    print(f"{'Metric':<32} | {'Pure LQR':<10} | {'PRCC-RL':<10} | {'Gain / Reduction'}")
    print("-" * 68)

    def print_row(title, key, unit, lower_is_better=True):
        vl = metrics['LQR'][key]
        vr = metrics['RL'][key]
        diff = (vr - vl) / (vl + 1e-8) * 100
        sign = "-" if diff < 0 else "+"
        print(f"{title:<32} | {vl:8.3f}{unit} | {vr:8.3f}{unit} | {sign}{abs(diff):5.1f}%")

    print_row("Roll Standard Deviation", 'roll_std', "°")
    print_row("Roll Peak Absolute Value", 'roll_peak', "°")
    print_row("Pitch Standard Deviation", 'pitch_std', "°")
    print_row("Pitch Peak Absolute Value", 'pitch_peak', "°")
    print_row("Vertical Height STD (Z-drift)", 'z_std', "mm")
    print_row("Mean Wheel Slip Velocity", 'slip_mean', "m/s")
    print_row("Hip Joint Control Effort (RMS)", 'tau_hip_rms', "Nm")
    print("-" * 68)
    print(f"{'Total Traversal Distance':<32} | {metrics['LQR']['final_x']:8.3f}m | {metrics['RL']['final_x']:8.3f}m | Finished: {metrics['RL']['final_x'] >= 3.45}")
    print("=" * 68)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    m_p_lqr = (res_lqr['x'] >= 0.2) & (res_lqr['x'] <= 3.5)
    m_p_rl  = (res_rl['x'] >= 0.2) & (res_rl['x'] <= 3.5)

    axs[0].axvspan(1.8, 3.3, color='orange', alpha=0.15, label='Bumps (1.8~3.3m)')
    axs[0].plot(res_lqr['x'][m_p_lqr], res_lqr['roll_deg'][m_p_lqr], 'r--', label='Pure LQR (Rigid, Severe Tilt)', alpha=0.8)
    axs[0].plot(res_rl['x'][m_p_rl], res_rl['roll_deg'][m_p_rl], 'g-', label='PRCC Residual RL (Stabilized)', linewidth=1.8)
    axs[0].set_ylabel('Body Roll (deg)')
    axs[0].set_title('Wheel-Leg Traversal: LQR Baseline vs PRCC Residual RL (Centerline Tracked)')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[0].margins(y=0.18)

    axs[1].axvspan(1.8, 3.3, color='orange', alpha=0.15, label='Bumps (1.8~3.3m)')
    axs[1].plot(res_lqr['x'][m_p_lqr], (res_lqr['z'][m_p_lqr] - res_lqr['z'][0]) * 1000, 'r--', label='Pure LQR', alpha=0.8)
    axs[1].plot(res_rl['x'][m_p_rl], (res_rl['z'][m_p_rl] - res_rl['z'][0]) * 1000, 'g-', label='PRCC Residual RL (Absorbed)', linewidth=1.8)
    axs[1].set_ylabel('Body Height Drift (mm)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[1].margins(y=0.18)

    axs[2].axvspan(1.8, 3.3, color='orange', alpha=0.15, label='Bumps (1.8~3.3m)')
    l1 = axs[2].plot(res_rl['x'][m_p_rl], res_rl['delta_roll'][m_p_rl], 'm-', label='RL Delta Roll (deg)', linewidth=1.5)
    ax2_twin = axs[2].twinx()
    l2 = ax2_twin.plot(res_lqr['x'][m_p_lqr], (res_lqr['y'][m_p_lqr] - (-0.0175)) * 1000, 'r:', label='LQR Drift (mm)', alpha=0.6)
    l3 = ax2_twin.plot(res_rl['x'][m_p_rl], (res_rl['y'][m_p_rl] - (-0.0175)) * 1000, 'g-', label='PRCC Drift (mm)', linewidth=1.4)
    axs[2].set_xlabel('Forward Travel Distance x (m)')
    axs[2].set_ylabel('Active Delta Roll (deg)')
    ax2_twin.set_ylabel('Lateral Drift from Center (mm)')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    axs[2].legend(lines, labels, loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[2].margins(y=0.18)

    plt.tight_layout()
    plot_path = os.path.join(current_dir, "evaluation_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[Artifact] High-res benchmark plots saved to: {plot_path}")


if __name__ == "__main__":
    main()
