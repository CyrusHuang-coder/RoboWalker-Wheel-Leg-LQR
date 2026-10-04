#include "controller.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
    inline double clamp_val(double val, double limit) {
        return std::max(-limit, std::min(limit, val));
    }
}

RobotController::RobotController()
{
    reset(0.0, 0.0);
}

void RobotController::reset(double current_x, double current_yaw)
{
    m_target_x   = current_x;
    m_target_v   = 0.0;
    m_target_yaw = current_yaw;
    m_target_pitch = 0.0;

    m_dpitch_filtered = 0.0;
    m_x_integral      = 0.0;
    m_v_integral      = 0.0;

    m_mode = ControlMode::HOLD_POSITION;
}

void RobotController::set_target_velocity(double v_cmd)
{
    // 限制在安全巡航速度区间内，防止倒立摆倾角过大失稳
    m_target_v = clamp_val(v_cmd, m_max_cruise_vel);
    if (std::abs(m_target_v) < 1e-4) {
        m_target_v = 0.0;
        m_mode = ControlMode::HOLD_POSITION;
    } else {
        m_mode = ControlMode::CRUISE_SPEED;
    }
}

void RobotController::change_target_velocity(double delta_v)
{
    set_target_velocity(m_target_v + delta_v);
}

void RobotController::set_target_yaw(double yaw_cmd)
{
    m_target_yaw = yaw_cmd;
}

void RobotController::change_target_yaw(double delta_yaw)
{
    m_target_yaw += delta_yaw;
}

void RobotController::trigger_brake(double current_x)
{
    m_target_v   = 0.0;
    m_target_x   = current_x;
    m_x_integral = 0.0;
    m_v_integral = 0.0;
    m_mode = ControlMode::HOLD_POSITION;
}

void RobotController::compute(const RobotSensors& sensors, RobotActuators& actuators, double dt)
{
    // -------------------------------------------------------------
    // 1. 传感器角速度一阶低通滤波 (滤除地面高频碰撞冲击噪声)
    // -------------------------------------------------------------
    m_dpitch_filtered = m_filter_alpha * m_dpitch_filtered + (1.0 - m_filter_alpha) * sensors.pitch_rate;

    // -------------------------------------------------------------
    // 2. 髋关节 PD 控制器 (维持机械腿直立虚拟刚度)
    // -------------------------------------------------------------
    double tau_l_hip = -m_kp_hip * (sensors.left_hip_pos - m_q_hip_nominal) - m_kd_hip * sensors.left_hip_vel;
    double tau_r_hip = -m_kp_hip * (sensors.right_hip_pos - m_q_hip_nominal) - m_kd_hip * sensors.right_hip_vel;
    tau_l_hip = clamp_val(tau_l_hip, m_max_hip_torque);
    tau_r_hip = clamp_val(tau_r_hip, m_max_hip_torque);

    // -------------------------------------------------------------
    // 3. 纵向外环串级控制器 (计算期望俯仰角 target_pitch)
    //    核心物理原理：倒立摆机器人必须身体向前倾斜才能获得前向加速度！
    // -------------------------------------------------------------
    if (std::abs(m_target_v) < 1e-4) {
        // [模式 A: 原地驻车位置锁定]
        m_mode = ControlMode::HOLD_POSITION;
        double x_err = sensors.x_pos - m_target_x;
        m_x_integral += x_err * dt;
        m_x_integral = clamp_val(m_x_integral, 0.2); // 积分限幅抗抗饱和

        // 倾角反馈：位置偏离原点越大，或者车速越快，就往回倾斜身体拉住车身
        m_target_pitch = -(m_kp_pos * x_err + m_kd_pos * sensors.forward_vel + m_ki_pos * m_x_integral);
        m_v_integral = 0.0;
    } else {
        // [模式 B: 遥控巡航速度跟踪]
        m_mode = ControlMode::CRUISE_SPEED;
        double v_err = sensors.forward_vel - m_target_v;
        m_v_integral += v_err * dt;
        m_v_integral = clamp_val(m_v_integral, 0.3); // 积分限幅

        // 速度前馈基准角 (推力补偿)，加上速度误差 PID 调谐
        double pitch_ff = 0.012 * (m_target_v / 0.1);
        m_target_pitch = pitch_ff - (m_kd_vel * v_err + m_ki_vel * m_v_integral);

        // 巡航时同步将当前实际位置设为驻车参考点，保证一松手瞬间无缝刹车
        m_target_x = sensors.x_pos;
        m_x_integral = 0.0;
    }
    // 目标倾角物理限幅 (防止身体倾斜过大导致打滑失稳)
    m_target_pitch = clamp_val(m_target_pitch, m_max_pitch_target);

    // -------------------------------------------------------------
    // 4. 倒立摆内环平衡控制器 (由俯仰倾角误差驱动车轮力矩)
    // -------------------------------------------------------------
    double pitch_err = sensors.pitch - m_target_pitch;
    double u_balance = m_kp_pitch * pitch_err + m_kd_pitch * m_dpitch_filtered;
    u_balance = clamp_val(u_balance, m_max_wheel_torque);

    // -------------------------------------------------------------
    // 5. 偏航角航向与差速转向控制器 (Yaw PD)
    // -------------------------------------------------------------
    double yaw_err = sensors.yaw - m_target_yaw;
    // 航向角误差规范化到 [-pi, pi]
    while (yaw_err > M_PI)  yaw_err -= 2.0 * M_PI;
    while (yaw_err < -M_PI) yaw_err += 2.0 * M_PI;

    double u_yaw = -(m_kp_yaw * yaw_err + m_kd_yaw * sensors.yaw_rate);
    u_yaw = clamp_val(u_yaw, 0.015); // 转向力矩限幅，避免强力打滑

    // -------------------------------------------------------------
    // 6. 控制力矩合成并分配给 4 路执行器
    //    左轮 = 平衡推力 - 偏航差速
    //    右轮 = 平衡推力 + 偏航差速
    // -------------------------------------------------------------
    actuators.torque_left_hip    = tau_l_hip;
    actuators.torque_left_wheel  = u_balance - u_yaw;
    actuators.torque_right_hip   = tau_r_hip;
    actuators.torque_right_wheel = u_balance + u_yaw;
}
