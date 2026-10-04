#pragma once

#include "interface.h"

/**
 * @brief 控制器运行模式枚举 (Finite State Machine Mode)
 */
enum class ControlMode {
    HOLD_POSITION, // 原地驻车位置锁定模式 (目标速度为 0)
    CRUISE_SPEED,  // 遥控巡航速度跟踪模式 (目标速度不为 0)
    EMERGENCY_STOP // 跌倒急停/保护切断模式
};

/**
 * @brief 轮腿自平衡核心控制器类 (Robot Controller Layer)
 * 职责：
 *  1. 纯数学控制逻辑，完全解耦底层的 MuJoCo 或硬件细节
 *  2. 串级控制体系：外环（位置/速度 -> 目标倾角）+ 内环（倾角误差 -> 驱动力矩）
 *  3. 偏航差速转向控制（左右轮反向力矩）
 *  4. 髋关节虚拟腿刚度维持（PD 锁死标称角度）
 */
class RobotController {
public:
    RobotController();

    /**
     * @brief 核心控制拍拍计算函数 (每 1ms 调用一次)
     * @param sensors 当前从接口层读取的传感器观测量 (只读输入)
     * @param actuators 计算并填充的目标执行器力矩 (只写输出)
     * @param dt 仿真积分步长 (默认 0.001s, 1000Hz)
     */
    void compute(const RobotSensors& sensors, RobotActuators& actuators, double dt = 0.001);

    /**
     * @brief 重置控制器内部积分器与历史状态
     * @param current_x 当前小车所在位置 (将作为新的驻车目标)
     * @param current_yaw 当前机头朝向 (将作为新的航向目标)
     */
    void reset(double current_x = 0.0, double current_yaw = 0.0);

    // ================= 遥控指令接口 (由键盘或调度层调用) =================
    void set_target_velocity(double v_cmd);      // 设置目标巡航速度 (m/s)
    void change_target_velocity(double delta_v);  // 增量微调目标速度
    void set_target_yaw(double yaw_cmd);         // 设置目标航向角 (rad)
    void change_target_yaw(double delta_yaw);    // 增量微调目标航向
    void trigger_brake(double current_x);        // 一键刹车并锁死当前位置

    // ================= 状态查询接口 (用于调试打印与状态机观测) =================
    ControlMode get_mode() const { return m_mode; }
    double get_target_pitch() const { return m_target_pitch; }
    double get_target_velocity() const { return m_target_v; }

private:
    // 当前控制模式
    ControlMode m_mode = ControlMode::HOLD_POSITION;

    // 1. 目标设定值 (Setpoint)
    double m_target_x     = 0.0; // 目标前向位置 (m)
    double m_target_v     = 0.0; // 目标前向线速度 (m/s)
    double m_target_yaw   = 0.0; // 目标偏航航向角 (rad)
    double m_target_pitch = 0.0; // 外环计算出的期望俯仰角 (rad)

    // 2. 内部积分与滤波历史状态 (Memory States)
    double m_dpitch_filtered = 0.0; // 一阶低通滤波后的俯仰角速度
    double m_x_integral      = 0.0; // 驻车位置误差积分项
    double m_v_integral      = 0.0; // 巡航速度误差积分项

    // 3. 算法超参数配置 (经过 Python 闭环 100% 验证锁定的最优参数)
    // 髋关节虚拟腿保持参数
    const double m_q_hip_nominal  = -0.0042; // 标称质心平衡偏角 (rad)
    const double m_kp_hip         = 0.08;
    const double m_kd_hip         = 0.002;
    const double m_max_hip_torque = 0.008;   // 髋关节微弱维持力矩 (N·m)

    // 直立平衡内环参数
    const double m_kp_pitch          = 0.10;
    const double m_kd_pitch          = 0.0005;
    const double m_filter_alpha      = 0.85; // 角速度低通滤波平滑因子
    const double m_max_wheel_torque  = 0.04; // 平衡最大驱动力矩 (N·m)

    // 纵向串级外环参数 (产生 target_pitch)
    const double m_kp_pos            = 0.25;
    const double m_kd_pos            = 0.35;
    const double m_ki_pos            = 0.05;
    const double m_kd_vel            = 0.15;
    const double m_ki_vel            = 0.03;
    const double m_max_pitch_target  = 0.10; // 最大允许倾角 ~5.7 度，防止轮地打滑
    const double m_max_cruise_vel    = 0.30; // 最大安全巡航线速度 (m/s)，防超速失稳

    // 偏航差速转向参数
    const double m_kp_yaw            = 0.002;
    const double m_kd_yaw            = 0.0001;
};
