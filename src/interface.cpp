#include "interface.h"
#include <iostream>

RobotInterface::RobotInterface(const mjModel* model, mjData* data)
    : m_model(model), m_data(data)
{
    // 辅助 lambda：根据传感器名称查询其在 sensordata 中的内存偏置地址
    auto get_sensor_adr = [model](const char* name) -> int {
        int id = mj_name2id(model, mjOBJ_SENSOR, name);
        if (id >= 0 && id < model->nsensor) {
            return model->sensor_adr[id];
        }
        std::cerr << "[Interface Warning] 未找到传感器: " << name << std::endl;
        return -1;
    };

    // 缓存所有传感器的底层偏置地址，避免每次读取循环调用字符串哈希查找
    m_adr_body_quat      = get_sensor_adr("body_quat");
    m_adr_body_angvel    = get_sensor_adr("body_angvel");
    m_adr_left_hip_pos   = get_sensor_adr("left_hip_pos");
    m_adr_left_hip_vel   = get_sensor_adr("left_hip_vel");
    m_adr_right_hip_pos  = get_sensor_adr("right_hip_pos");
    m_adr_right_hip_vel  = get_sensor_adr("right_hip_vel");
    m_adr_left_wheel_vel = get_sensor_adr("left_wheel_vel");
    m_adr_right_wheel_vel= get_sensor_adr("right_wheel_vel");
}

void RobotInterface::quat_to_euler(const double q[4], double& roll, double& pitch, double& yaw)
{
    // MuJoCo 四元数格式: [w, x, y, z]
    double w = q[0], x = q[1], y = q[2], z = q[3];

    // Roll (横滚角，绕X轴)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (俯仰角，绕Y轴，低头向前为正)
    double sinp = 2.0 * (w * y - z * x);
    if (sinp > 1.0) sinp = 1.0;
    if (sinp < -1.0) sinp = -1.0;
    pitch = std::asin(sinp);

    // Yaw (偏航角，绕Z轴)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

void RobotInterface::update_sensors(RobotSensors& sensors)
{
    if (!m_data) return;

    // 1. 读取 IMU 四元数并转为欧拉角
    if (m_adr_body_quat >= 0) {
        quat_to_euler(m_data->sensordata + m_adr_body_quat, sensors.roll, sensors.pitch, sensors.yaw);
    } else {
        // 回退机制：直接从自由关节 qpos[3..6] 读取
        quat_to_euler(m_data->qpos + 3, sensors.roll, sensors.pitch, sensors.yaw);
    }

    // 2. 读取 IMU 陀螺仪角速度 (wx, wy, wz)
    if (m_adr_body_angvel >= 0) {
        sensors.roll_rate  = m_data->sensordata[m_adr_body_angvel + 0];
        sensors.pitch_rate = m_data->sensordata[m_adr_body_angvel + 1];
        sensors.yaw_rate   = m_data->sensordata[m_adr_body_angvel + 2];
    } else {
        sensors.roll_rate  = m_data->qvel[3];
        sensors.pitch_rate = m_data->qvel[4];
        sensors.yaw_rate   = m_data->qvel[5];
    }

    // 3. 纵向位移与前进线速度
    sensors.x_pos = m_data->qpos[0];
    double vx_world = m_data->qvel[0];
    double vy_world = m_data->qvel[1];
    // 沿机身当前朝向的速度投影: v_fwd = vx * cos(yaw) + vy * sin(yaw)
    sensors.forward_vel = vx_world * std::cos(sensors.yaw) + vy_world * std::sin(sensors.yaw);

    // 4. 读取关节角度与速度
    sensors.left_hip_pos    = (m_adr_left_hip_pos >= 0)   ? m_data->sensordata[m_adr_left_hip_pos]   : m_data->qpos[7];
    sensors.left_hip_vel    = (m_adr_left_hip_vel >= 0)   ? m_data->sensordata[m_adr_left_hip_vel]   : m_data->qvel[6];
    sensors.right_hip_pos   = (m_adr_right_hip_pos >= 0)  ? m_data->sensordata[m_adr_right_hip_pos]  : m_data->qpos[9];
    sensors.right_hip_vel   = (m_adr_right_hip_vel >= 0)  ? m_data->sensordata[m_adr_right_hip_vel]  : m_data->qvel[8];
    sensors.left_wheel_vel  = (m_adr_left_wheel_vel >= 0) ? m_data->sensordata[m_adr_left_wheel_vel] : m_data->qvel[7];
    sensors.right_wheel_vel = (m_adr_right_wheel_vel >= 0)? m_data->sensordata[m_adr_right_wheel_vel]: m_data->qvel[9];

    // 5. 机体离地绝对高度
    sensors.base_z = m_data->qpos[2];
}

void RobotInterface::apply_actuators(const RobotActuators& cmd, bool enable_safety)
{
    if (!m_data) return;

    // 限幅辅助函数
    auto clamp_val = [](double val, double limit) -> double {
        return std::max(-limit, std::min(limit, val));
    };

    // 执行器映射 (严格对齐 wheel_leg.xml 的 actuator 声明顺序):
    // 0: left_hip_motor
    // 1: left_wheel_motor
    // 2: right_hip_motor
    // 3: right_wheel_motor
    if (enable_safety) {
        m_data->ctrl[0] = clamp_val(cmd.torque_left_hip,    m_max_hip_torque);
        m_data->ctrl[1] = clamp_val(cmd.torque_left_wheel,  m_max_wheel_torque);
        m_data->ctrl[2] = clamp_val(cmd.torque_right_hip,   m_max_hip_torque);
        m_data->ctrl[3] = clamp_val(cmd.torque_right_wheel, m_max_wheel_torque);
    } else {
        m_data->ctrl[0] = cmd.torque_left_hip;
        m_data->ctrl[1] = cmd.torque_left_wheel;
        m_data->ctrl[2] = cmd.torque_right_hip;
        m_data->ctrl[3] = cmd.torque_right_wheel;
    }
}

bool RobotInterface::is_fallen(const RobotSensors& sensors) const
{
    // 安全跌倒保护判断标准:
    // 1. 俯仰倾角 > 0.5 rad (~28.6度)
    // 2. 横滚翻转 > 0.5 rad
    // 3. 机身离地高度 < 0.035 m (轮半径 8mm，正常高度 ~0.049m)
    return (std::abs(sensors.pitch) > 0.5 ||
            std::abs(sensors.roll)  > 0.5 ||
            sensors.base_z < 0.035);
}

void RobotInterface::stop_all_motors()
{
    if (!m_data) return;
    for (int i = 0; i < 4; ++i) {
        m_data->ctrl[i] = 0.0;
    }
}
