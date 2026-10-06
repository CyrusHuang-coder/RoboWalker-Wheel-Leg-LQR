"""
prior_controller.py: 经典串级解耦先验控制器与物理先验自平衡基底
================================================================================
【模块功能介绍】
本模块实现了轮腿式机器人的核心先验自平衡控制律 (Prior Controller)，作为整个控制框架
的物理基础与稳定基石（对应 C++ 控制层中的底层状态机与反馈闭环）。具备以下关键特性：

1. 层次化解耦控制架构：
   - 髋关节虚拟悬架顺应环 (Hip Virtual Suspension):
     构建等效虚拟弹簧-阻尼系统，结合车体倾侧角速度前馈 (PD Anticipation) 与非对称回弹阻尼 (Rebound Damping)，
     在车轮刚触碰障碍瞬间主动屈腿吸震，回弹时加大阻尼抑制机体跳跃。
   - 倒立摆内环姿态自平衡 (Inner Inverted Pendulum Balancing):
     实时跟踪目标俯仰角，计算轮毂电机基础平衡驱动力矩 u_balance。
   - 纵向串级速度/位置外环 (Outer Longitudinal Cascade Loop):
     通过实际航速/位置与设定点偏差，动态解算机体所需前倾角 (Base Target Pitch)，形成前倾自然加速。
   - 偏航航向与车道线横向纠偏环 (Yaw & Lateral Lane Keeping):
     基于双轮几何中线误差动态计算航向偏航指令，通过两轮差动力矩实现精确巡线与 S 弯跟踪。

2. 强化学习残差接入规范 (Residual RL Interface):
   允许外部上层策略（如 PRCC-RL、SCT-RRL）注入以下 4 类物理残差信号：
   - delta_pitch: 目标俯仰角残差 (rad)，用于坡道动态前倾/后仰补偿
   - delta_hip:   双腿对称俯仰角残差 (rad)，用于质心高度自适应调节
   - delta_roll:  双腿差动防侧倾髋角残差 (rad)，用于非对称台阶或路面倾斜吸收
   - k_scale:     髋关节刚度连续缩放因子 (0.3 ~ 2.0x)，用于刚柔动态切换
================================================================================
"""
import os
import sys
import numpy as np

# 路径自适应引导：确保从任何子目录或主工程运行均能定位依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def quat2rpy(q):
    """
    四元数转航向欧拉角 (Roll, Pitch, Yaw)
    输入: q = [w, x, y, z] (MuJoCo 标量在首的标准顺序)
    输出: (roll, pitch, yaw) 弧度制
    """
    w, x, y, z = q
    # 1. Roll (横滚角 - 绕 X 轴旋转)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # 2. Pitch (俯仰角 - 绕 Y 轴旋转，限制 [-pi/2, pi/2])
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # 3. Yaw (偏航角 - 绕 Z 轴旋转)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class PriorController:
    """
    经典串级与解耦先验控制器 (对应 C++ RobotController)
    具备独立的平衡能力，同时向强化学习策略提供确定性动作基底。
    """
    def __init__(self):
        # ---------------- 1. 髋关节虚拟腿保持参数 ----------------
        self.q_hip_nominal = -0.0042  # 标称站立髋关节角 (轻微后摆保持直立质心)
        self.kp_hip_base = 0.08       # 虚拟弹簧基础刚度系数 (N·m/rad)
        self.kd_hip_base = 0.002      # 虚拟阻尼基础阻尼系数 (N·m·s/rad)
        self.max_hip_torque = 0.02    # 单髋关节最大控制力矩保护限幅 (N·m)

        # ---------------- 2. 倒立摆平衡内环参数 ----------------
        self.kp_pitch = 0.10          # 俯仰角度刚度增益 (P 环)
        self.kd_pitch = 0.0005        # 俯仰角速度阻尼增益 (D 环)
        self.filter_alpha = 0.85      # 角速度一阶低通滤波因子
        self.max_wheel_torque = 0.08  # 轮毂电机最大力矩限幅 (N·m)

        # ---------------- 3. 纵向串级外环参数 ----------------
        self.kp_pos = 0.25            # 位置环比例增益
        self.kd_pos = 0.35            # 位置环速度微分增益
        self.ki_pos = 0.05            # 位置环积分增益
        self.kd_vel = 0.35            # 速度外环比例增益
        self.ki_vel = 0.08            # 速度外环积分增益
        self.max_pitch_target = 0.12  # 最大前倾角限幅 (~6.88°，保证不脱离平衡锥)

        # ---------------- 4. 偏航转向外环与车道线保持 ----------------
        self.kp_yaw = 0.030           # 偏航角跟踪刚度增益
        self.kd_yaw = 0.0018          # 偏航角速度阻尼增益
        self.target_y = -0.0175       # 巡线默认参考横向位置 (对应双轮几何中线)
        self.kp_y = 5.5               # 横向偏差转航向角指令反馈增益

        # ---------------- 5. 内部状态寄存器 ----------------
        self.target_v = 0.0           # 目标航速指令 (m/s)
        self.target_x = 0.0           # 目标驻车 X 坐标 (m)
        self.target_yaw = 0.0         # 目标偏航角 (rad)
        self.dpitch_filtered = 0.0    # 滤波后的俯仰角速度
        self.x_integral = 0.0         # 纵向位置误差积分
        self.v_integral = 0.0         # 纵向速度误差积分

        # ---------------- 6. 上次计算缓存 (供 RL 构建观测向量) ----------------
        self.last_target_pitch = 0.0  # 最近一次指令前倾角
        self.last_u_balance = 0.0     # 最近一次平衡基础力矩

    def reset(self, current_x=0.0, current_yaw=0.0, current_y=-0.0175):
        """重置控制器内部积分器与设定目标点"""
        self.dpitch_filtered = 0.0
        self.x_integral = 0.0
        self.v_integral = 0.0
        self.target_x = current_x
        self.target_yaw = current_yaw
        self.target_y = current_y
        self.target_v = 0.0
        self.last_target_pitch = 0.0
        self.last_u_balance = 0.0

    def set_target_velocity(self, v_cmd):
        """设定纵向目标巡航速度 (m/s)"""
        self.target_v = np.clip(v_cmd, -0.3, 0.3)

    def set_target_yaw(self, yaw_cmd):
        """设定目标偏航航向角 (rad)"""
        self.target_yaw = yaw_cmd

    def hip_spring_params(self, sensors, delta_roll=0.0, k_scale=1.0):
        """
        纯函数计算接口：返回虚拟弹簧阻尼当前步的等效物理参数
        返回: (kp, kd_l, kd_r, q_target_l, q_target_r)
        供能量储罐 (Energy Tank) 精确计算弹性势能变化量 dW，不产生离散化误差。
        """
        k = float(np.clip(k_scale, 0.3, 2.0))
        kp = self.kp_hip_base * k
        kd = self.kd_hip_base * np.sqrt(k)

        # 引入姿态角速度前馈超前顺应
        roll_eff = sensors['roll'] + 0.035 * sensors.get('roll_rate', 0.0)
        prior_lh = float(np.clip(roll_eff * 1.6, 0.0, 0.25))
        prior_rh = float(np.clip(-roll_eff * 1.6, 0.0, 0.25))

        q_l = self.q_hip_nominal + np.clip(prior_lh + max(0.0, delta_roll), 0.0, 0.28)
        q_r = self.q_hip_nominal + np.clip(prior_rh + max(0.0, -delta_roll), 0.0, 0.28)

        # 非对称回弹阻尼
        kd_l = kd * (2.2 if sensors['left_hip_vel'] < 0.0 else 1.0)
        kd_r = kd * (2.2 if sensors['right_hip_vel'] < 0.0 else 1.0)
        return kp, kd_l, kd_r, q_l, q_r

    def compute(self, sensors, dt=0.001, delta_pitch=0.0, delta_hip=0.0, delta_roll=0.0, k_scale=1.0, enable_compliance=True):
        """
        核心控制动力学解算主函数
        输入:
          - sensors: 字典形式的机体传感数据 (roll, pitch, yaw, 角速度, 关节位置与车轮转速)
          - dt: 控制仿真步长 (默认 0.001s / 1kHz)
          - delta_pitch / delta_hip / delta_roll / k_scale: 外部注入残差
          - enable_compliance: 是否启用主动顺应 (LQR 基线置 False，RL 置 True)
        输出:
          - actuators: 四路执行器力矩指令字典 (left/right hip & wheel)
        """
        # 1. 俯仰角速度一阶低通滤波 (削弱传感器高频噪声)
        self.dpitch_filtered = (self.filter_alpha * self.dpitch_filtered + 
                                (1.0 - self.filter_alpha) * sensors['pitch_rate'])

        # 2. 髋关节虚拟弹簧阻尼主动吸震
        kp_hip = self.kp_hip_base * np.clip(k_scale, 0.3, 2.0)
        kd_hip = self.kd_hip_base * np.sqrt(np.clip(k_scale, 0.3, 2.0))

        if enable_compliance:
            roll_rate = sensors.get('roll_rate', 0.0)
            roll_eff = sensors['roll'] + 0.035 * roll_rate
            prior_lh = float(np.clip(roll_eff * 1.6, 0.0, 0.25))
            prior_rh = float(np.clip(-roll_eff * 1.6, 0.0, 0.25))
            delta_lh = np.clip(prior_lh + max(0.0, delta_roll), 0.0, 0.28)
            delta_rh = np.clip(prior_rh + max(0.0, -delta_roll), 0.0, 0.28)
        else:
            delta_lh = 0.0
            delta_rh = 0.0

        q_target_lh = self.q_hip_nominal + delta_lh
        q_target_rh = self.q_hip_nominal + delta_rh

        # 撞击抬起低阻尼吸震，越障复位高阻尼抑制弹跳
        kd_lh = kd_hip * (2.2 if sensors['left_hip_vel'] < 0.0 else 1.0)
        kd_rh = kd_hip * (2.2 if sensors['right_hip_vel'] < 0.0 else 1.0)

        tau_l_hip = -kp_hip * (sensors['left_hip_pos'] - q_target_lh) - kd_lh * sensors['left_hip_vel']
        tau_r_hip = -kp_hip * (sensors['right_hip_pos'] - q_target_rh) - kd_rh * sensors['right_hip_vel']
        tau_l_hip = np.clip(tau_l_hip, -self.max_hip_torque, self.max_hip_torque)
        tau_r_hip = np.clip(tau_r_hip, -self.max_hip_torque, self.max_hip_torque)

        # 3. 纵向串级外环计算基础目标前倾角
        if abs(self.target_v) < 1e-4:
            x_err = sensors['x_pos'] - self.target_x
            self.x_integral = np.clip(self.x_integral + x_err * dt, -0.2, 0.2)
            base_target_pitch = -(self.kp_pos * x_err + self.kd_pos * sensors['forward_vel'] + self.ki_pos * self.x_integral)
            self.v_integral = 0.0
        else:
            v_err = sensors['forward_vel'] - self.target_v
            self.v_integral = np.clip(self.v_integral + v_err * dt, -0.3, 0.3)
            pitch_ff = -0.0171 * (self.target_v / 0.1)  # 自然前倾前馈项
            base_target_pitch = pitch_ff - (self.kd_vel * v_err + self.ki_vel * self.v_integral)
            self.target_x = sensors['x_pos']
            self.x_integral = 0.0

        base_target_pitch = np.clip(base_target_pitch, -self.max_pitch_target, self.max_pitch_target)

        # 叠加 RL 俯仰残差
        final_target_pitch = np.clip(base_target_pitch + delta_pitch, -self.max_pitch_target * 1.5, self.max_pitch_target * 1.5)
        self.last_target_pitch = final_target_pitch

        # 4. 倒立摆内环姿态自平衡控制
        pitch_err = sensors['pitch'] - final_target_pitch
        u_balance = self.kp_pitch * pitch_err + self.kd_pitch * self.dpitch_filtered
        u_balance = np.clip(u_balance, -self.max_wheel_torque, self.max_wheel_torque)
        self.last_u_balance = u_balance

        # 5. 偏航转向与横向直行纠偏控制 (针对双轮几何中线)
        y_pos = sensors.get('y_pos', self.target_y)
        y_err = y_pos - self.target_y
        yaw_cmd = self.target_yaw - np.clip(self.kp_y * y_err, -0.25, 0.25)
        yaw_err = sensors['yaw'] - yaw_cmd
        while yaw_err > np.pi: yaw_err -= 2.0 * np.pi
        while yaw_err < -np.pi: yaw_err += 2.0 * np.pi
        u_yaw = -(self.kp_yaw * yaw_err + self.kd_yaw * sensors['yaw_rate'])
        u_yaw = np.clip(u_yaw, -0.045, 0.045)

        # 6. 四路力矩合成与输出
        actuators = {
            'torque_left_hip': tau_l_hip,
            'torque_right_hip': tau_r_hip,
            'torque_left_wheel': u_balance - u_yaw,
            'torque_right_wheel': u_balance + u_yaw
        }
        return actuators
