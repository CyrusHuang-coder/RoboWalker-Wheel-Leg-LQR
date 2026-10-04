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


def run_episode(mode="lqr", model_path="rl/models/best_model.zip", duration=15.0, v_cmd=0.10):
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
            delta_pitch = float(action[0] * 0.015)
            delta_hip   = float(action[1] * 0.05)
            delta_roll  = float(action[2] * 0.06)
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

        # 记录遥测数据
        v_lw = abs(sensors['left_wheel_vel'] * r_wheel)
        v_rw = abs(sensors['right_wheel_vel'] * r_wheel)
        slip = 0.5 * (abs(v_lw - abs(sensors['forward_vel'])) + abs(v_rw - abs(sensors['forward_vel'])))

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
    parser.add_argument("--duration", type=float, default=55.0, help="Test duration (s)")
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
    mask_lqr = (res_lqr['x'] >= 0.3) & (res_lqr['x'] <= max_common_x)
    mask_rl  = (res_rl['x'] >= 0.3) & (res_rl['x'] <= max_common_x)

    roll_std_lqr = np.std(res_lqr['roll_deg'][mask_lqr])
    roll_max_lqr = np.max(np.abs(res_lqr['roll_deg'][mask_lqr]))
    z_std_lqr    = np.std(res_lqr['z'][mask_lqr]) * 1000 # mm
    slip_lqr     = np.mean(res_lqr['slip_rate'][mask_lqr])

    roll_std_rl  = np.std(res_rl['roll_deg'][mask_rl])
    roll_max_rl  = np.max(np.abs(res_rl['roll_deg'][mask_rl]))
    z_std_rl     = np.std(res_rl['z'][mask_rl]) * 1000 # mm
    slip_rl      = np.mean(res_rl['slip_rate'][mask_rl])

    # 性能改善百分比
    impr_roll_std = (roll_std_lqr - roll_std_rl) / roll_std_lqr * 100
    impr_roll_max = (roll_max_lqr - roll_max_rl) / roll_max_lqr * 100
    impr_z_std    = (z_std_lqr - z_std_rl) / z_std_lqr * 100
    impr_slip     = (slip_lqr - slip_rl) / (slip_lqr + 1e-6) * 100

    print("\n" + "="*72)
    print("         PERFORMANCE QUANTITATIVE BENCHMARK REPORT")
    print("="*72)
    print(f" {'Metric Indicator':<30} | {'Pure LQR':<12} | {'PRCC-RL':<12} | {'Improvement':<12}")
    print("-" * 72)
    print(f" {'Roll Angle Std (横滚抖动标准差)':<26} | {roll_std_lqr:>9.2f}°  | {roll_std_rl:>9.2f}°  | {impr_roll_std:>+9.1f}%")
    print(f" {'Roll Angle Peak (最大侧倾冲击)':<26} | {roll_max_lqr:>9.2f}°  | {roll_max_rl:>9.2f}°  | {impr_roll_max:>+9.1f}%")
    print(f" {'Vertical Height Std (机身沉浮)':<24} | {z_std_lqr:>8.2f}mm  | {z_std_rl:>8.2f}mm  | {impr_z_std:>+9.1f}%")
    print(f" {'Wheel Slip Rate (轮地平均滑移)':<25} | {slip_lqr*1000:>7.2f}mm/s | {slip_rl*1000:>7.2f}mm/s | {impr_slip:>+9.1f}%")
    print("="*72)

    # 绘制高精学术对比图
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. 横滚角对比 (核心亮点：RL 主动吸震，Roll 显著平缓)
    axs[0].plot(res_lqr['x'], res_lqr['roll_deg'], 'r--', label='Pure LQR Baseline', alpha=0.8, linewidth=1.5)
    axs[0].plot(res_rl['x'], res_rl['roll_deg'], 'b-', label='PRCC Residual RL (Ours)', linewidth=1.8)
    axs[0].set_ylabel('Body Roll (deg)')
    axs[0].set_title('Wheel-Leg Rough Terrain Traversal: LQR Baseline vs Residual RL')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')

    # 2. 机身垂直颠簸 (z 轴绝对高度变化)
    axs[1].plot(res_lqr['x'], (res_lqr['z'] - res_lqr['z'][0]) * 1000, 'r--', label='Pure LQR', alpha=0.8)
    axs[1].plot(res_rl['x'], (res_rl['z'] - res_rl['z'][0]) * 1000, 'g-', label='PRCC Residual RL (Absorbed)', linewidth=1.8)
    axs[1].set_ylabel('Body Height Drift (mm)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')

    # 3. 策略网络的主动调节动作 (差动横滚补偿角度与等效刚度)
    axs[2].plot(res_rl['x'], res_rl['delta_roll'], 'm-', label='RL Delta Roll (Diff Compliance, deg)', linewidth=1.5)
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
