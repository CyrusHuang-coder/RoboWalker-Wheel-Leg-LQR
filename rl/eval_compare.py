"""
eval_compare.py: 纯 Baseline (LQR) vs 残差强化学习 (PRCC-RL) 终极对比评测
======================================================================
功能：
  1. 在完全相同的一组轻起伏+非对称单侧障碍地形上，分别运行纯 LQR 与 PRCC-RL
  2. 采集全生命周期高精遥测数据 (横滚角 Roll, 俯仰角 Pitch, 垂直颠簸 z_acc, 打滑率 slip)
  3. 输出清晰的 ASCII 量化对比表格 (波动削减幅度 %)
  4. 绘制学术级高保真对比曲线图: rl/evaluation_comparison.png
"""
import os
import sys
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入先验与环境工具
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from prior_controller import PriorController, quat2rpy
from eval_baseline import extract_sensors


def run_episode(mode="lqr", model_path="rl/models/best_model.zip", duration=25.0, v_cmd=0.16):
    """
    运行单次测试评估
    mode: "lqr" 或 "residual_rl"
    """
    xml_path = "rl/terrain/wheel_leg_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 加载 RL 策略 (如果是 RL 模式)
    rl_policy = None
    if mode == "residual_rl":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        rl_policy = PPO.load(model_path)

    controller = PriorController()
    controller.reset(current_x=data.sensor('body_pos').data[0], current_yaw=0.0)
    controller.set_target_velocity(v_cmd)

    dt = model.opt.timestep
    policy_substep = 20  # 50Hz 策略更新 (每 20 步 1ms 物理步)
    total_steps = int(duration / dt)

    # 状态与历史缓存
    obs_history = np.zeros((3, 23), dtype=np.float32)
    last_action = np.zeros(4, dtype=np.float32)

    # 预热沉降
    for _ in range(30):
        mujoco.mj_step(model, data)

    history = {
        'time': [],
        'x': [],
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

        # 跌倒判定 (倾角超过 35 度)
        if abs(sensors['pitch']) > np.radians(35) or abs(sensors['roll']) > np.radians(35):
            print(f"[{mode.upper()}] Robot tipped over at t={t:.3f}s, x={sensors['x_pos']:.3f}m")
            break

        # 每 20ms (50Hz) 更新一次 RL 动作
        if step % policy_substep == 0:
            if mode == "residual_rl":
                # 构造当前帧观测 (23维)
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

                # 策略推理
                action, _ = rl_policy.predict(obs_history.flatten(), deterministic=True)
                last_action = action.copy()
            else:
                action.fill(0.0)

        # 解码残差指令 (与训练环境严格一致)
        if mode == "residual_rl":
            delta_pitch = float(action[0] * 0.005)
            delta_hip   = float(action[1] * 0.04)
            delta_roll  = float(action[2] * 0.08)
            k_scale     = float(1.0 + action[3] * 0.3)
        else:
            delta_pitch = 0.0
            delta_hip   = 0.0
            delta_roll  = 0.0
            k_scale     = 1.0

        # 先验控制器综合解算 (1000Hz)
        actuators = controller.compute(
            sensors, dt=dt,
            delta_pitch=delta_pitch, delta_hip=delta_hip,
            delta_roll=delta_roll, k_scale=k_scale
        )

        data.actuator('left_hip_motor').ctrl[0] = actuators['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = actuators['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = actuators['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = actuators['torque_right_wheel']

        mujoco.mj_step(model, data)

        # 记录遥测数据 (计算轮毂中心真实前向速度，剔除髋关节摆动的几何耦合误差)
        v_lw = abs(sensors['left_wheel_vel'] * r_wheel)
        v_rw = abs(sensors['right_wheel_vel'] * r_wheel)
        L_leg = 0.040
        v_hub_l = abs(sensors['forward_vel'] + sensors['left_hip_vel'] * L_leg * np.cos(sensors['left_hip_pos']))
        v_hub_r = abs(sensors['forward_vel'] + sensors['right_hip_vel'] * L_leg * np.cos(sensors['right_hip_pos']))
        slip = 0.5 * (abs(v_lw - v_hub_l) + abs(v_rw - v_hub_r))

        history['time'].append(t)
        history['x'].append(sensors['x_pos'])
        history['z'].append(sensors['z_pos'])
        history['roll_deg'].append(np.degrees(sensors['roll']))
        history['pitch_deg'].append(np.degrees(sensors['pitch']))
        history['z_vel'].append(sensors['body_linvel'][2])
        history['slip_rate'].append(slip)
        history['tau_l_hip'].append(actuators['torque_left_hip'])
        history['tau_r_hip'].append(actuators['torque_right_hip'])
        history['tau_wheel'].append(actuators['torque_left_wheel'])
        history['k_scale'].append(k_scale)
        history['delta_roll'].append(np.degrees(delta_roll))

    for k in history:
        history[k] = np.array(history[k])

    return history


def main():
    parser = argparse.ArgumentParser(description="Compare LQR Baseline vs Residual RL")
    parser.add_argument("--model", type=str, default="rl/models/best_model.zip", help="Path to RL model")
    parser.add_argument("--duration", type=float, default=45.0, help="Test duration (s)")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  [1/2] Running Benchmark for Pure LQR Baseline...")
    print("=======================================================")
    res_lqr = run_episode(mode="lqr", duration=args.duration)

    print("\n=======================================================")
    print("  [2/2] Running Benchmark for PRCC-Residual RL Policy...")
    print("=======================================================")
    res_rl = run_episode(mode="residual_rl", model_path=args.model, duration=args.duration)

    # 确定共同有效前进区间
    max_common_x = min(res_lqr['x'][-1], res_rl['x'][-1])

    # 1. 核心非对称障碍颠簸区 [1.8m ~ min(3.2m, max_common_x)]
    bump_end = min(3.2, max_common_x)
    m_bump_lqr = (res_lqr['x'] >= 1.8) & (res_lqr['x'] <= bump_end)
    m_bump_rl  = (res_rl['x'] >= 1.8) & (res_rl['x'] <= bump_end)

    # 2. 全程综合测试区 [0.3m ~ max_common_x]
    m_all_lqr = (res_lqr['x'] >= 0.3) & (res_lqr['x'] <= max_common_x)
    m_all_rl  = (res_rl['x'] >= 0.3) & (res_rl['x'] <= max_common_x)

    def compute_metrics(res, mask):
        r_std = np.std(res['roll_deg'][mask])
        r_max = np.max(np.abs(res['roll_deg'][mask]))
        z_std = np.std(res['z'][mask]) * 1000
        slip  = np.mean(res['slip_rate'][mask]) * 1000
        return r_std, r_max, z_std, slip

    b_rstd_lqr, b_rmax_lqr, b_zstd_lqr, b_slip_lqr = compute_metrics(res_lqr, m_bump_lqr)
    b_rstd_rl,  b_rmax_rl,  b_zstd_rl,  b_slip_rl  = compute_metrics(res_rl, m_bump_rl)

    a_rstd_lqr, a_rmax_lqr, a_zstd_lqr, a_slip_lqr = compute_metrics(res_lqr, m_all_lqr)
    a_rstd_rl,  a_rmax_rl,  a_zstd_rl,  a_slip_rl  = compute_metrics(res_rl, m_all_rl)

    def calc_impr(v_base, v_ours):
        return (v_base - v_ours) / (v_base + 1e-6) * 100

    print("\n" + "="*76)
    print("       ROBOTIC PERFORMANCE BENCHMARK: BUMP ZONE & FULL TRAVERSAL")
    print("="*76)
    print(" [SECTION A: 核心障碍颠簸区测试 (1.8m ~ 3.2m 单侧凸起群)]")
    print(f" {'Metric Indicator':<30} | {'Pure LQR':<12} | {'PRCC-RL':<12} | {'Improvement':<12}")
    print("-" * 76)
    print(f" {'Peak Roll (最大侧倾冲击角)':<28} | {b_rmax_lqr:>9.2f}°  | {b_rmax_rl:>9.2f}°  | {calc_impr(b_rmax_lqr, b_rmax_rl):>+9.1f}%")
    print(f" {'Roll Std (横滚姿态抖动标准差)':<27} | {b_rstd_lqr:>9.2f}°  | {b_rstd_rl:>9.2f}°  | {calc_impr(b_rstd_lqr, b_rstd_rl):>+9.1f}%")
    print(f" {'Height Std (机身垂直颠簸标准差)':<25} | {b_zstd_lqr:>8.2f}mm  | {b_zstd_rl:>8.2f}mm  | {calc_impr(b_zstd_lqr, b_zstd_rl):>+9.1f}%")
    print(f" {'Wheel Slip (轮毂真实滑移速度)':<26} | {b_slip_lqr:>7.2f}mm/s | {b_slip_rl:>7.2f}mm/s | {calc_impr(b_slip_lqr, b_slip_rl):>+9.1f}%")
    
    print("\n [SECTION B: 全程综合测试 (0.3m ~ 终点 包含平地/正弦波/凸起)]")
    print(f" {'Metric Indicator':<30} | {'Pure LQR':<12} | {'PRCC-RL':<12} | {'Improvement':<12}")
    print("-" * 76)
    print(f" {'Peak Roll (全程最大侧倾)':<28} | {a_rmax_lqr:>9.2f}°  | {a_rmax_rl:>9.2f}°  | {calc_impr(a_rmax_lqr, a_rmax_rl):>+9.1f}%")
    print(f" {'Roll Std (全程姿态抖动标准差)':<27} | {a_rstd_lqr:>9.2f}°  | {a_rstd_rl:>9.2f}°  | {calc_impr(a_rstd_lqr, a_rstd_rl):>+9.1f}%")
    print(f" {'Height Std (全程机身颠簸)':<26} | {a_zstd_lqr:>8.2f}mm  | {a_zstd_rl:>8.2f}mm  | {calc_impr(a_zstd_lqr, a_zstd_rl):>+9.1f}%")
    print(f" {'Wheel Slip (全程平均滑移)':<27} | {a_slip_lqr:>7.2f}mm/s | {a_slip_rl:>7.2f}mm/s | {calc_impr(a_slip_lqr, a_slip_rl):>+9.1f}%")
    print("="*76)

    # 绘制高精学术对比图
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. 横滚角对比 (标出障碍区间)
    axs[0].axvspan(1.8, 3.2, color='orange', alpha=0.15, label='Asymmetric Bumps (1.8m~3.2m)')
    axs[0].plot(res_lqr['x'], res_lqr['roll_deg'], 'r--', label='Pure LQR Baseline (Stiff)', alpha=0.8, linewidth=1.5)
    axs[0].plot(res_rl['x'], res_rl['roll_deg'], 'b-', label='PRCC Residual RL (Compliant)', linewidth=1.8)
    axs[0].set_ylabel('Body Roll (deg)')
    axs[0].set_title('Wheel-Leg Traversal: LQR Baseline vs PRCC Residual RL')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')

    # 2. 机身垂直颠簸 (z 轴绝对高度变化)
    axs[1].axvspan(1.8, 3.2, color='orange', alpha=0.15)
    axs[1].plot(res_lqr['x'], (res_lqr['z'] - res_lqr['z'][0]) * 1000, 'r--', label='Pure LQR', alpha=0.8)
    axs[1].plot(res_rl['x'], (res_rl['z'] - res_rl['z'][0]) * 1000, 'g-', label='PRCC Residual RL (Absorbed)', linewidth=1.8)
    axs[1].set_ylabel('Body Height Drift (mm)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')

    # 3. 策略网络的主动调节动作 (差动横滚补偿角度与等效刚度)
    axs[2].axvspan(1.8, 3.2, color='orange', alpha=0.15)
    axs[2].plot(res_rl['x'], res_rl['delta_roll'], 'm-', label='RL Delta Roll (Active Suspension, deg)', linewidth=1.5)
    ax2_twin = axs[2].twinx()
    ax2_twin.plot(res_rl['x'], res_rl['k_scale'], 'c:', label='RL Adaptive Stiffness (k_scale)', linewidth=1.5)
    axs[2].set_xlabel('Forward Travel Distance x (m)')
    axs[2].set_ylabel('Active Delta Roll (deg)')
    ax2_twin.set_ylabel('Stiffness Scale')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plot_path = "rl/evaluation_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[Artifact] High-res benchmark plots saved to: {plot_path}")


if __name__ == "__main__":
    main()
