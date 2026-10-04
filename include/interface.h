#pragma once

#include <mujoco/mujoco.h>
#include <cmath>
#include <algorithm>

/**
 * @brief 机器人传感器抽象观测结构体 (Robot Sensors Observation)
 * 纯物理单位：rad, rad/s, m, m/s
 * 算法层（Controller）只读取此结构体，完全不接触底层硬件或仿真指针
 */
struct RobotSensors {
    // 1. 机体欧拉角姿态 (IMU姿态)
    double roll;         // 横滚角 (rad, 向右倾为正)
    double pitch;        // 俯仰角 (rad, 低头向前为正)
    double yaw;          // 偏航角 (rad, 逆时针为正)

    // 2. 机体角速度 (陀螺仪)
    double roll_rate;    // 横滚角速度 (rad/s)
    double pitch_rate;   // 俯仰角速度 (rad/s)
    double yaw_rate;     // 偏航角速度 (rad/s)

    // 3. 纵向位移与前进线速度
    double x_pos;        // 全局前向位移 (m)
    double forward_vel;  // 机体航向线速度 (m/s)

    // 4. 关节状态 (编码器反馈)
    double left_hip_pos;    // 左髋关节角度 (rad)
    double left_hip_vel;    // 左髋关节角速度 (rad/s)
    double right_hip_pos;   // 右髋关节角度 (rad)
    double right_hip_vel;   // 右髋关节角速度 (rad/s)
    double left_wheel_vel;  // 左轮转速 (rad/s)
    double right_wheel_vel; // 右轮转速 (rad/s)

    // 5. 健康状态与基础物理量
    double base_z;       // 机身离地高度 (m)
};

/**
 * @brief 机器人执行器输出指令结构体 (Robot Actuators Command)
 * 4 路力矩输出，单位: N·m
 */
struct RobotActuators {
    double torque_left_hip;    // 左髋关节力矩 (N·m)
    double torque_left_wheel;  // 左驱动轮力矩 (N·m)
    double torque_right_hip;   // 右髋关节力矩 (N·m)
    double torque_right_wheel; // 右驱动轮力矩 (N·m)

    // 默认构造函数：全部力矩清零
    RobotActuators()
        : torque_left_hip(0.0), torque_left_wheel(0.0),
          torque_right_hip(0.0), torque_right_wheel(0.0) {}
};

/**
 * @brief 硬件与仿真抽象接口类 (Robot Interface Layer)
 * 职责：
 *  1. 从 MuJoCo mjData 提取传感器数据并解析为 RobotSensors
 *  2. 保护性限幅并写入 mjData->ctrl
 *  3. 提供跌倒检测与底层安全急停机制
 */
class RobotInterface {
public:
    RobotInterface(const mjModel* model, mjData* data);

    // 刷新并读取全部传感器观测量
    void update_sensors(RobotSensors& sensors);

    // 向执行器写入控制力矩 (带硬限幅与安全急停检查)
    void apply_actuators(const RobotActuators& cmd, bool enable_safety = true);

    // 跌倒判断检测 (倾角 > 0.5 rad 或高度 < 0.035 m)
    bool is_fallen(const RobotSensors& sensors) const;

    // 清零电机输出 (急停)
    void stop_all_motors();

private:
    const mjModel* m_model;
    mjData* m_data;

    // 传感器在 mjData->sensordata 中的内存地址缓存 (提高每次读取性能)
    int m_adr_body_quat;
    int m_adr_body_angvel;
    int m_adr_left_hip_pos;
    int m_adr_left_hip_vel;
    int m_adr_right_hip_pos;
    int m_adr_right_hip_vel;
    int m_adr_left_wheel_vel;
    int m_adr_right_wheel_vel;

    // 硬件物理安全极值保护 (189g 微型轮腿机器人标定阈值)
    const double m_max_wheel_torque = 0.05; // 轮子最大安全扭矩: 0.05 N·m
    const double m_max_hip_torque   = 0.02; // 髋关节最大安全扭矩: 0.02 N·m

    // 四元数转欧拉角工具函数
    static void quat_to_euler(const double q[4], double& roll, double& pitch, double& yaw);
};
