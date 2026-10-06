"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/eval_baseline.py`: 经典倒立摆基线 (LQR/Cascade) 起伏地形性能评测与数据固化脚本
====================================================================================================
1. 核心定位与测试目标：
   - 本模块独立测试并评估未经任何强化学习残差补偿的纯经典先验控制器 (Baseline LQR) 在 4.0m
     标准起伏与非对称减速垄赛道上的通过能力与姿态震荡。
   - 提取并保存全时域遥测数据至 `baseline_data.npz` (时间戳、X 位移、Z 高度、Roll、Pitch、打滑率)，
     作为后续全部强化学习算法消融与提升对比的基准锚点。

2. 关键退化表征：
   - 刚性底盘由于无主动悬架与阻抗调节律，在压过单侧 4.8mm 凸起时机身会发生剧烈侧倾 (Roll 峰值高达 4°~6°)，
     并诱发车轮悬空飞转与着地时的强力机械冲击。
====================================================================================================
"""

import os
import sys
import time
import numpy as np
import mujoco

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


def extract_sensors(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """
    从 MuJoCo 传感器中提取标准化物理观测量字典。
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

    forward_vel = float(body_linvel[0])

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
        'forward_vel': forward_vel,
        'body_linvel': body_linvel.copy(),
        'left_hip_pos': lh_pos,
        'left_hip_vel': lh_vel,
        'right_hip_pos': rh_pos,
        'right_hip_vel': rh_vel,
        'left_wheel_vel': lw_vel,
        'right_wheel_vel': rw_vel,
    }


def run_baseline(render: bool = False, v_cmd: float = 0.10, duration: float = 8.0,
                 out_npz: str = None):
    """
    运行经典基准控制器评测并导出指标。
    """
    if out_npz is None:
        out_npz = os.path.join(current_dir, "baseline_data.npz")

    xml_path = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    controller = PriorController()
    controller.reset(current_x=data.sensor('body_pos').data[0], current_yaw=0.0)
    controller.set_target_velocity(v_cmd)

    dt = model.opt.timestep
    steps = int(duration / dt)

    history = {
        'time': [],
        'x': [],
        'z': [],
        'roll': [],
        'pitch': [],
        'slip_rate': [],
        'fall': False
    }

    r_wheel = 0.008

    for step in range(steps):
        t = step * dt
        sensors = extract_sensors(model, data)

        # 跌倒判定 (倾角超过 35 度)
        if abs(sensors['pitch']) > np.radians(35) or abs(sensors['roll']) > np.radians(35):
            print(f"[Baseline] Robot fell at t={t:.3f}s, x={sensors['x_pos']:.3f}m!")
            history['fall'] = True
            break

        # 纯 Baseline: 不加任何残差 (delta_pitch=0, delta_hip=0, delta_roll=0, k_scale=1.0)
        actuators = controller.compute(sensors, dt=dt, delta_pitch=0.0, delta_hip=0.0, delta_roll=0.0, k_scale=1.0)

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
        history['z'].append(sensors['z_pos'])
        history['roll'].append(np.degrees(sensors['roll']))
        history['pitch'].append(np.degrees(sensors['pitch']))
        history['slip_rate'].append(slip)

    roll_arr = np.array(history['roll'])
    pitch_arr = np.array(history['pitch'])
    slip_arr = np.array(history['slip_rate'])
    final_x = history['x'][-1] if len(history['x']) > 0 else 0.0

    print(f"\n--- Baseline Benchmark Results ---")
    print(f"Final distance traveled: {final_x:.3f} m / 4.0 m")
    print(f"Roll std (横滚波动标准差): {np.std(roll_arr):.2f} deg, max abs: {np.max(np.abs(roll_arr)):.2f} deg")
    print(f"Pitch std (俯仰波动标准差): {np.std(pitch_arr):.2f} deg, max abs: {np.max(np.abs(pitch_arr)):.2f} deg")
    print(f"Mean slip (平均滑移量): {np.mean(slip_arr):.4f} m/s")

    np.savez(out_npz,
             time=history['time'], x=history['x'], z=history['z'],
             roll=history['roll'], pitch=history['pitch'], slip=history['slip_rate'])
    print(f"[Baseline] Saved metrics to {out_npz}\n")


if __name__ == "__main__":
    run_baseline(duration=10.0, v_cmd=0.10)
