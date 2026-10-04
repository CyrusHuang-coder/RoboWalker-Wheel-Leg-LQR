"""
RoboWalker 2026 轮腿机器人 - 自平衡与键盘遥控仿真 (Python 原型)
======================================================
控制架构：串级状态反馈与解耦差速遥控 (Cascade State Feedback & Decoupled Teleoperation)
- 闭环 1 (髋关节姿态锁定): 保持双腿与机身标称平衡夹角，提供刚度与阻尼
- 闭环 2 (双轮倒立摆平衡): 状态反馈 (Pitch - Pitch_target) + 角速度一阶滤波阻尼
- 闭环 3 (纵向位移/速度外环): 
    * 停车静止模式 (Hold): 自动锁定目标位置坐标，消除静差
    * 遥控巡航模式 (Cruise): 跟踪目标前进速度，前后倾斜车身自然加速
- 闭环 4 (航向角保持与差速转向): 左右轮差模力矩解耦控制 Yaw 角

交互控制按键清单:
  - [W] : 前进加速 (增加巡航速度 +0.05 m/s)
  - [S] : 后退减速 / 倒车 (-0.05 m/s)
  - [A] : 向左旋转航向 (+0.15 rad)
  - [D] : 向右旋转航向 (-0.15 rad)
  - [Space] (空格) : 刹车制动，锁定当前坐标原地直立
  - [P] : 施加向前 0.25 N 冲击推力 (测试抗推恢复能力)
  - [B] : 施加向后 0.25 N 冲击推力
  - [R] : 一键重置复位立正 (摔倒后瞬间站立复原)
"""

import os
import sys
import time
import argparse
import mujoco
import numpy as np

try:
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False


def quat2rpy(q):
    """四元数转欧拉角 (Roll, Pitch, Yaw)"""
    w, x, y, z = q
    # Roll (横滚角)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (俯仰角)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # Yaw (偏航角)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class WheelLegBalanceController:
    """轮腿自平衡与运动控制器"""
    def __init__(self):
        # 1. 髋关节位置锁定参数 (标称静平衡角度 q_0 = -0.0042 rad)
        self.q_hip_target = -0.0042
        self.kp_hip = 0.08
        self.kd_hip = 0.002
        self.max_hip_torque = 0.008  # N*m

        # 2. 倒立摆平衡内环参数
        self.kp_pitch = 0.10
        self.kd_pitch = 0.0005
        self.max_wheel_torque = 0.04  # N*m

        # 3. 纵向位置与速度控制外环参数
        self.kp_pos = 0.25
        self.kd_pos = 0.35
        self.ki_pos = 0.05
        self.kd_vel = 0.15
        self.ki_vel = 0.03
        self.max_pitch_target = 0.10  # 最大允许倾角约 5.7 度，防止打滑

        # 4. 航向偏航与转向参数
        self.kp_yaw = 0.002
        self.kd_yaw = 0.0001
        self.target_yaw = 0.0
        self.target_x = 0.0
        self.target_v = 0.0

        # 控制器内部状态
        self.dpitch_filtered = 0.0
        self.x_integral = 0.0
        self.v_integral = 0.0
        self.filter_alpha = 0.85  # 角速度低通滤波因子

    def reset(self, current_x=0.0, current_yaw=0.0):
        self.dpitch_filtered = 0.0
        self.x_integral = 0.0
        self.v_integral = 0.0
        self.target_x = current_x
        self.target_v = 0.0
        self.target_yaw = current_yaw

    def compute(self, data):
        """
        根据 MuJoCo 传感器数据计算 4 路电机控制力矩:
        返回: (ctrl_l_hip, ctrl_l_wheel, ctrl_r_hip, ctrl_r_wheel, roll, pitch, yaw, x)
        """
        # 读取姿态与角速度传感器
        quat = data.sensor('body_quat').data
        roll, pitch, yaw = quat2rpy(quat)
        dpitch_raw = data.sensor('body_angvel').data[1]
        dyaw = data.sensor('body_angvel').data[2]

        # 角速度一阶低通滤波
        self.dpitch_filtered = self.filter_alpha * self.dpitch_filtered + (1.0 - self.filter_alpha) * dpitch_raw

        # 读取纵向位置与车身前进速度
        x = data.qpos[0]
        vx_world = data.qvel[0]
        vy_world = data.qvel[1]
        v_forward = vx_world * np.cos(yaw) + vy_world * np.sin(yaw)

        # 读取髋关节状态
        q_l = data.sensor('left_hip_pos').data[0]
        dq_l = data.sensor('left_hip_vel').data[0]
        q_r = data.sensor('right_hip_pos').data[0]
        dq_r = data.sensor('right_hip_vel').data[0]

        # 1. 髋关节 PD 控制 (保持腿直立刚度)
        tau_l_hip = np.clip(-self.kp_hip * (q_l - self.q_hip_target) - self.kd_hip * dq_l,
                            -self.max_hip_torque, self.max_hip_torque)
        tau_r_hip = np.clip(-self.kp_hip * (q_r - self.q_hip_target) - self.kd_hip * dq_r,
                            -self.max_hip_torque, self.max_hip_torque)

        # 2. 纵向外环：根据是“驻车静止”还是“遥控巡航”智能切换
        if abs(self.target_v) < 1e-3:
            # 驻车模式: 锁定目标位置 target_x
            x_err = x - self.target_x
            self.x_integral = np.clip(self.x_integral + x_err * 0.001, -0.5, 0.5)
            pitch_target = - self.kp_pos * x_err - self.kd_pos * data.qvel[0] - self.ki_pos * self.x_integral
        else:
            # 巡航模式: 跟踪速度 target_v (车身前倾/后仰自然加速)
            v_err = v_forward - self.target_v
            self.v_integral = np.clip(self.v_integral + v_err * 0.001, -0.5, 0.5)
            pitch_target = -0.0171 - self.kd_vel * v_err - self.ki_vel * self.v_integral

        pitch_target = np.clip(pitch_target, -self.max_pitch_target, self.max_pitch_target)

        # 3. 倒立摆平衡内环 -> 基础共模平衡力矩
        pitch_err = pitch - pitch_target
        u_balance = self.kp_pitch * pitch_err + self.kd_pitch * self.dpitch_filtered

        # 4. 航向保持与转向外环 -> 差模转向力矩
        yaw_err = yaw - self.target_yaw
        u_yaw = - self.kp_yaw * yaw_err - self.kd_yaw * dyaw

        # 左右轮力矩解耦叠加
        tau_l_wheel = np.clip(u_balance - u_yaw, -self.max_wheel_torque, self.max_wheel_torque)
        tau_r_wheel = np.clip(u_balance + u_yaw, -self.max_wheel_torque, self.max_wheel_torque)

        return tau_l_hip, tau_l_wheel, tau_r_hip, tau_r_wheel, roll, pitch, yaw, x


def main():
    parser = argparse.ArgumentParser(description="RoboWalker 2026 轮腿自平衡仿真验证与键盘遥控")
    parser.add_argument("--headless", action="store_true", help="无界面快速验证模式")
    parser.add_argument("--steps", type=int, default=10000, help="无界面模式下的仿真步数 (默认 10000 步 = 10 秒)")
    args = parser.parse_args()

    print("=" * 70)
    print("  RoboWalker 2026 轮腿机器人 - 自平衡与键盘遥控 (Python Prototype)")
    print("=" * 70)

    def find_xml_path(name="wheel_leg.xml"):
        for p in [name, os.path.join("..", name), os.path.join(os.path.dirname(__file__), "..", name), os.path.join(os.path.dirname(__file__), name)]:
            if os.path.exists(p):
                return p
        return name

    model = mujoco.MjModel.from_xml_path(find_xml_path("wheel_leg.xml"))
    data = mujoco.MjData(model)
    controller = WheelLegBalanceController()

    def reset_robot():
        """重置机器人到立正姿态"""
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.04905   # 地面精确接触高度 (轮底严格切合地面)
        data.qpos[3] = 1.0       # 初始四元数
        data.qpos[7] = -0.0042   # 左髋关节初始角
        data.qpos[9] = -0.0042   # 右髋关节初始角
        mujoco.mj_forward(model, data)
        controller.reset(current_x=data.qpos[0], current_yaw=0.0)

    reset_robot()

    if args.headless or not HAS_VIEWER:
        print(f">> 启动无头测试模式 (总计 {args.steps} 步，模拟时长 {args.steps * 0.001:.1f} 秒)...")
        for step in range(args.steps):
            u_lh, u_lw, u_rh, u_rw, roll, pitch, yaw, x = controller.compute(data)
            data.ctrl[0] = u_lh
            data.ctrl[1] = u_lw
            data.ctrl[2] = u_rh
            data.ctrl[3] = u_rw

            mujoco.mj_step(model, data)

            if abs(pitch) > 0.45 or data.qpos[2] < 0.02:
                print(f"[FAIL] 小车在第 {step} 步倾覆摔倒！(Pitch: {np.degrees(pitch):.1f} deg)")
                sys.exit(1)

            if step % 2000 == 0:
                print(f"[Step {step:5d}] 时长: {step*0.001:.1f}s | 俯仰角: {np.degrees(pitch):+5.2f}° | "
                      f"位置: {x*1000:+6.1f}mm | 轮扭矩: {u_lw*1000:+5.2f}mN*m")

        print("=" * 70)
        print(f"[PASS] 仿真圆满成功！连续平衡 {args.steps} 步，最终稳态误差: {x*1000:.2f} mm")
        print("=" * 70)

    else:
        # 外部推力计数器 (剩余推力施加步数)
        push_force_steps = 0
        push_force_value = 0.0
        last_status = None

        def on_key(keycode):
            """处理键盘输入回调 (支持方向键与 WASD，优先推荐方向键避免触发 MuJoCo 内部线框/骨骼快捷键)"""
            nonlocal push_force_steps, push_force_value, last_status
            # 前进加速: [↑ 方向键] 或 [W]
            if keycode == 265 or keycode == ord('W') or keycode == ord('w'):
                controller.target_v = min(controller.target_v + 0.05, 0.30)
                print(f"\n[{data.time:6.2f}s] [按键指令] 🏎️ 前进加速 -> 目标巡航速度: {controller.target_v*100:+.0f} cm/s")
            # 后退减速: [↓ 方向键] 或 [S]
            elif keycode == 264 or keycode == ord('S') or keycode == ord('s'):
                controller.target_v = max(controller.target_v - 0.05, -0.30)
                print(f"\n[{data.time:6.2f}s] [按键指令] 🏎️ 减速倒车 -> 目标巡航速度: {controller.target_v*100:+.0f} cm/s")
            # 左转弯: [← 方向键] 或 [A]
            elif keycode == 263 or keycode == ord('A') or keycode == ord('a'):
                controller.target_yaw += 0.15
                print(f"\n[{data.time:6.2f}s] [按键指令] ↩️ 向左转弯 -> 目标偏航角: {np.degrees(controller.target_yaw):+.1f}°")
            # 右转弯: [→ 方向键] 或 [D]
            elif keycode == 262 or keycode == ord('D') or keycode == ord('d'):
                controller.target_yaw -= 0.15
                print(f"\n[{data.time:6.2f}s] [按键指令] ↪️ 向右转弯 -> 目标偏航角: {np.degrees(controller.target_yaw):+.1f}°")
            # 空格: 刹车制动并原地定点
            elif keycode == 32:
                controller.target_v = 0.0
                controller.target_x = data.qpos[0]
                controller.x_integral = 0.0
                print(f"\n[{data.time:6.2f}s] [按键指令] 🛑 刹车制动！锁定在当前坐标 x={data.qpos[0]*1000:+.1f}mm")
            # P: 施加向前冲击推力
            elif keycode == ord('P') or keycode == ord('p'):
                push_force_steps = 60  # 施加 60ms
                push_force_value = 0.25 # 0.25 N (相当于自重的 14%)
                print(f"\n[{data.time:6.2f}s] [外力扰动] 💥 施加了 +0.25 N 前向推力扰动！观察小车冲刺追平衡...")
            # B: 施加向后推力
            elif keycode == ord('B') or keycode == ord('b'):
                push_force_steps = 60
                push_force_value = -0.25
                print(f"\n[{data.time:6.2f}s] [外力扰动] 💥 施加了 -0.25 N 后向推力扰动！观察小车倒车回稳...")
            # R: 一键重置站立
            elif keycode == ord('R') or keycode == ord('r'):
                reset_robot()
                last_status = None
                print(f"\n[{data.time:6.2f}s] [系统复位] 🔄 机器人已一键复位立正！重回平衡状态。")

        print(">> 启动 MuJoCo 官方 3D 实时交互窗口...")
        print("   【推荐键盘遥控 (方向键，不与引擎快捷键冲突)】:")
        print("   - [↑] / [↓] 方向键 : 前进加速 / 倒车减速 (当前巡航速度以 cm/s 为单位)")
        print("   - [←] / [→] 方向键 : 左右差速转向 (画圆转弯)")
        print("   - (也兼容 W/S/A/D，但注意 W 会触发 MuJoCo 线框透视，M 会触发骨骼透视)")
        print("   - [Space] 空格键   : 刹车并原地定点站立")
        print("   - [P] / [B] 键     : 施加前向/后向温和推力 (测试抗推恢复)")
        print("   - [R] 键           : 一键复位立正 (摔倒按 R 立即站起来)")
        print("   【鼠标交互说明】:")
        print("   - 旋转视角: 鼠标左键拖动 | 平移视角: 鼠标右键拖动")
        print("   - 鼠标加力: 双击橙色机身，按住 Ctrl + 右键拖拽拉动红色受力弹簧！")
        print("-" * 70)

        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:

            while viewer.is_running():
                step_start = time.time()

                # 处理外力脉冲注入
                if push_force_steps > 0:
                    data.xfrc_applied[1, 0] = push_force_value
                    push_force_steps -= 1
                else:
                    data.xfrc_applied[1, 0] = 0.0

                # 计算真实物理状态
                u_lh, u_lw, u_rh, u_rw, roll, pitch, yaw, x = controller.compute(data)

                # 动态摔倒检测: 质心高度低于 3.5cm 或 倾角过大
                is_fallen = (data.qpos[2] < 0.035 or abs(pitch) > 0.50 or abs(roll) > 0.50)

                if is_fallen:
                    # 摔倒保护：切断轮子狂转，保护电机
                    data.ctrl[0] = 0.0
                    data.ctrl[1] = 0.0
                    data.ctrl[2] = 0.0
                    data.ctrl[3] = 0.0
                    status_text = "❌ 倾覆摔倒 (按 [R] 键一键复位立正)"
                elif abs(controller.target_v) > 1e-3:
                    data.ctrl[0] = u_lh
                    data.ctrl[1] = u_lw
                    data.ctrl[2] = u_rh
                    data.ctrl[3] = u_rw
                    status_text = f"🚗 巡航中 (目标速度: {controller.target_v*100:+.0f} cm/s)"
                elif push_force_steps > 0 or abs(pitch - (-0.0171)) > 0.05 or abs(controller.dpitch_filtered) > 0.2:
                    data.ctrl[0] = u_lh
                    data.ctrl[1] = u_lw
                    data.ctrl[2] = u_rh
                    data.ctrl[3] = u_rw
                    status_text = "⚡ 正在动态抵抗扰动并恢复平衡..."
                else:
                    data.ctrl[0] = u_lh
                    data.ctrl[1] = u_lw
                    data.ctrl[2] = u_rh
                    data.ctrl[3] = u_rw
                    status_text = "✅ 稳定自平衡直立"

                mujoco.mj_step(model, data)

                # 同步 3D 渲染画面
                viewer.sync()

                # 仅在状态改变时才打印通知 (Event-Driven State Logging)
                if status_text != last_status:
                    print(f"[{data.time:6.2f}s] [状态变更] {status_text} "
                          f"(俯仰: {np.degrees(pitch):+5.2f}°, 位置: {x*1000:+6.1f}mm, 偏航: {np.degrees(yaw):+5.2f}°)")
                    last_status = status_text

                # 保持 1000Hz 物理真实时序
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        print("\n\n交互窗口已正常关闭。")


if __name__ == "__main__":
    main()
