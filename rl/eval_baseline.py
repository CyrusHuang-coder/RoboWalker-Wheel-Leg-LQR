"""
测试与评估经典先验控制器 (Baseline LQR/Cascade) 在波浪与非对称起伏地形上的表现
目的：
  1. 验证在轻起伏地形下，纯 Baseline 能走多远
  2. 捕获纯 Baseline 压过单侧凸起时的机身严重颠簸、Roll 角剧烈偏转与轮子空转打滑现象
  3. 保存 Baseline 的遥测数据 (x, roll, pitch, vertical_acc, slip)，为后续 RL 对比提供量化基准线
"""
import os
import time
import numpy as np
import mujoco
from prior_controller import PriorController, quat2rpy


def extract_sensors(model, data):
    """从 MuJoCo 提取传感器观测量"""
    body_quat = data.sensor('body_quat').data
    roll, pitch, yaw = quat2rpy(body_quat)
    body_angvel = data.sensor('body_angvel').data
    body_linvel = data.sensor('body_linvel').data
    body_pos = data.sensor('body_pos').data

    lh_pos = data.sensor('left_hip_pos').data[0]
    lh_vel = data.sensor('left_hip_vel').data[0]
    rh_pos = data.sensor('right_hip_pos').data[0]
    rh_vel = data.sensor('right_hip_vel').data[0]

    lw_vel = data.sensor('left_wheel_vel').data[0]
    rw_vel = data.sensor('right_wheel_vel').data[0]

    # 车体正向线速度 (世界坐标系投影至机身纵轴或直接取 vx)
    # 对于小角度，前向线速度约等于 body_linvel[0]
    forward_vel = body_linvel[0]

    return {
        'x_pos': body_pos[0],
        'y_pos': body_pos[1],
        'z_pos': body_pos[2],
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'pitch_rate': body_angvel[1],
        'yaw_rate': body_angvel[2],
        'roll_rate': body_angvel[0],
        'forward_vel': forward_vel,
        'body_linvel': body_linvel,
        'left_hip_pos': lh_pos,
        'left_hip_vel': lh_vel,
        'right_hip_pos': rh_pos,
        'right_hip_vel': rh_vel,
        'left_wheel_vel': lw_vel,
        'right_wheel_vel': rw_vel,
    }


def run_baseline(render=False, v_cmd=0.10, duration=8.0):
    xml_path = "rl/terrain/wheel_leg_terrain.xml"
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

        # 将力矩赋予执行器
        data.actuator('left_hip_motor').ctrl[0] = actuators['torque_left_hip']
        data.actuator('right_hip_motor').ctrl[0] = actuators['torque_right_hip']
        data.actuator('left_wheel_motor').ctrl[0] = actuators['torque_left_wheel']
        data.actuator('right_wheel_motor').ctrl[0] = actuators['torque_right_wheel']

        mujoco.mj_step(model, data)

        # 计算滑移率 (简单度量: |w*R - vx|)
        # 左轮与右轮理论线速度
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

    # 统计数据
    roll_arr = np.array(history['roll'])
    pitch_arr = np.array(history['pitch'])
    slip_arr = np.array(history['slip_rate'])
    final_x = history['x'][-1] if len(history['x']) > 0 else 0.0

    print(f"\n--- Baseline Benchmark Results ---")
    print(f"Final distance traveled: {final_x:.3f} m / 4.0 m")
    print(f"Roll std (横滚波动标准差): {np.std(roll_arr):.2f} deg, max abs: {np.max(np.abs(roll_arr)):.2f} deg")
    print(f"Pitch std (俯仰波动标准差): {np.std(pitch_arr):.2f} deg, max abs: {np.max(np.abs(pitch_arr)):.2f} deg")
    print(f"Mean slip (平均滑移量): {np.mean(slip_arr):.4f} m/s")

    # 保存 Baseline 结果以便后续比对
    np.savez("rl/baseline_data.npz", 
             time=history['time'], x=history['x'], z=history['z'],
             roll=history['roll'], pitch=history['pitch'], slip=history['slip_rate'])
    print("[Baseline] Saved metrics to rl/baseline_data.npz\n")


if __name__ == "__main__":
    run_baseline(duration=10.0, v_cmd=0.10)
