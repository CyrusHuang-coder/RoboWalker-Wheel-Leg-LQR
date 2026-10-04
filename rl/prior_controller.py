"""
PriorController: 封装经过实车/C++验证的经典串级自平衡控制器
可独立作为 Baseline 运行，也可接收 RL 策略输出的残差修正 (Residual Correction)
"""
import numpy as np


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


class PriorController:
    """
    经典串级与解耦先验控制器 (对应 C++ RobotController)
    支持外部注入残差：
      - delta_pitch: 目标俯仰角残差 (rad)
      - delta_hip:   双腿对称高度/髋角残差 (rad)
      - delta_roll:  双腿差动防侧倾髋角残差 (rad)
      - k_scale:     髋关节刚度缩放因子 (1.0 为默认刚度)
    """
    def __init__(self):
        # 1. 髋关节虚拟腿保持参数
        self.q_hip_nominal = -0.0042
        self.kp_hip_base = 0.08
        self.kd_hip_base = 0.002
        self.max_hip_torque = 0.02  # 留出起伏适度裕度

        # 2. 倒立摆平衡内环参数
        self.kp_pitch = 0.10
        self.kd_pitch = 0.0005
        self.filter_alpha = 0.85
        self.max_wheel_torque = 0.08  # 从 0.04 适度放宽至 0.08 N·m，保证起伏爬坡动力

        # 3. 纵向串级外环参数
        self.kp_pos = 0.25
        self.kd_pos = 0.35
        self.ki_pos = 0.05
        self.kd_vel = 0.15
        self.ki_vel = 0.03
        self.max_pitch_target = 0.12  # 约 6.8 度

        # 4. 偏航转向外环与赛道直线保持 (强化航向与中线保持，彻底抑制单侧缩腿造成的扭摆)
        self.kp_yaw = 0.030
        self.kd_yaw = 0.0018
        self.target_y = -0.0175
        self.kp_y = 3.5

        # 控制器内部状态与设定点
        self.target_v = 0.0
        self.target_x = 0.0
        self.target_yaw = 0.0
        self.dpitch_filtered = 0.0
        self.x_integral = 0.0
        self.v_integral = 0.0

        # 上次计算结果缓存 (用于 RL 观测)
        self.last_target_pitch = 0.0
        self.last_u_balance = 0.0

    def reset(self, current_x=0.0, current_yaw=0.0, current_y=-0.0175):
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
        self.target_v = np.clip(v_cmd, -0.3, 0.3)

    def set_target_yaw(self, yaw_cmd):
        self.target_yaw = yaw_cmd

    def compute(self, sensors, dt=0.001, delta_pitch=0.0, delta_hip=0.0, delta_roll=0.0, k_scale=1.0, enable_compliance=True):
        """
        核心控制力矩解算
        sensors dict 包含:
          - roll, pitch, yaw, pitch_rate, yaw_rate, roll_rate
          - x_pos, forward_vel
          - left_hip_pos, left_hip_vel, right_hip_pos, right_hip_vel
          - left_wheel_vel, right_wheel_vel
        """
        # 1. 俯仰角速度一阶低通滤波
        self.dpitch_filtered = (self.filter_alpha * self.dpitch_filtered + 
                                (1.0 - self.filter_alpha) * sensors['pitch_rate'])

        # 2. 髋关节虚拟弹簧阻尼 (加入残差与自适应刚度)
        kp_hip = self.kp_hip_base * np.clip(k_scale, 0.3, 2.0)
        kd_hip = self.kd_hip_base * np.sqrt(np.clip(k_scale, 0.3, 2.0))

        if enable_compliance:
            # 先验单侧主动顺应律: 引入 PD 超前顺应 (姿态角 + 角速度微分超前)
            # 在车轮刚撞击障碍瞬间利用角速度提前起腿，削平第一波冲击尖峰
            roll_rate = sensors.get('roll_rate', 0.0)
            roll_eff = sensors['roll'] + 0.035 * roll_rate
            
            # 放宽主动行程: 最大后屈 0.25 rad (约 14.3°)，吸震垂直行程由 1.2mm 提升至 2.6mm
            prior_lh = float(np.clip(roll_eff * 1.6, 0.0, 0.25))
            prior_rh = float(np.clip(-roll_eff * 1.6, 0.0, 0.25))

            # 叠加残差调谐 (RL 微调与环境自适应)
            delta_lh = np.clip(prior_lh + max(0.0, delta_roll), 0.0, 0.28)
            delta_rh = np.clip(prior_rh + max(0.0, -delta_roll), 0.0, 0.28)
        else:
            delta_lh = 0.0
            delta_rh = 0.0

        q_target_lh = self.q_hip_nominal + delta_lh
        q_target_rh = self.q_hip_nominal + delta_rh

        # 非对称回弹阻尼 (Rebound Damping):
        # 撞击抬起时顺应吸震 (低阻尼)，越障后复位伸展时加大阻尼 (2.2倍)，彻底抑制末端机身弹跳
        kd_lh = kd_hip * (2.2 if sensors['left_hip_vel'] < 0.0 else 1.0)
        kd_rh = kd_hip * (2.2 if sensors['right_hip_vel'] < 0.0 else 1.0)

        tau_l_hip = -kp_hip * (sensors['left_hip_pos'] - q_target_lh) - kd_lh * sensors['left_hip_vel']
        tau_r_hip = -kp_hip * (sensors['right_hip_pos'] - q_target_rh) - kd_rh * sensors['right_hip_vel']
        tau_l_hip = np.clip(tau_l_hip, -self.max_hip_torque, self.max_hip_torque)
        tau_r_hip = np.clip(tau_r_hip, -self.max_hip_torque, self.max_hip_torque)

        # 3. 纵向串级外环解算 base target_pitch
        if abs(self.target_v) < 1e-4:
            x_err = sensors['x_pos'] - self.target_x
            self.x_integral = np.clip(self.x_integral + x_err * dt, -0.2, 0.2)
            base_target_pitch = -(self.kp_pos * x_err + self.kd_pos * sensors['forward_vel'] + self.ki_pos * self.x_integral)
            self.v_integral = 0.0
        else:
            v_err = sensors['forward_vel'] - self.target_v
            self.v_integral = np.clip(self.v_integral + v_err * dt, -0.3, 0.3)
            # 前倾自然加速 (负号为前倾追赶速度)
            pitch_ff = -0.0171 * (self.target_v / 0.1)
            base_target_pitch = pitch_ff - (self.kd_vel * v_err + self.ki_vel * self.v_integral)
            self.target_x = sensors['x_pos']
            self.x_integral = 0.0

        base_target_pitch = np.clip(base_target_pitch, -self.max_pitch_target, self.max_pitch_target)

        # 叠加 RL 俯仰残差
        final_target_pitch = np.clip(base_target_pitch + delta_pitch, -self.max_pitch_target * 1.5, self.max_pitch_target * 1.5)
        self.last_target_pitch = final_target_pitch

        # 4. 倒立摆内环平衡控制
        pitch_err = sensors['pitch'] - final_target_pitch
        u_balance = self.kp_pitch * pitch_err + self.kd_pitch * self.dpitch_filtered
        u_balance = np.clip(u_balance, -self.max_wheel_torque, self.max_wheel_torque)
        self.last_u_balance = u_balance

        # 5. 偏航转向与横向直行纠偏控制 (巡线保持: target_y = -0.0175)
        y_pos = sensors.get('y_pos', self.target_y)
        y_err = y_pos - self.target_y
        yaw_cmd = self.target_yaw - np.clip(self.kp_y * y_err, -0.15, 0.15)
        yaw_err = sensors['yaw'] - yaw_cmd
        while yaw_err > np.pi: yaw_err -= 2.0 * np.pi
        while yaw_err < -np.pi: yaw_err += 2.0 * np.pi
        u_yaw = -(self.kp_yaw * yaw_err + self.kd_yaw * sensors['yaw_rate'])
        u_yaw = np.clip(u_yaw, -0.03, 0.03)

        # 6. 力矩合成
        actuators = {
            'torque_left_hip': tau_l_hip,
            'torque_right_hip': tau_r_hip,
            'torque_left_wheel': u_balance - u_yaw,
            'torque_right_wheel': u_balance + u_yaw
        }
        return actuators
