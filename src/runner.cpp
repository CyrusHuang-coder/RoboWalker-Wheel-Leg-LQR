#include "runner.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <GLFW/glfw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 全局静态单例指针，用于 GLFW 回调路由
static SimulationRunner* g_runner_instance = nullptr;

SimulationRunner::SimulationRunner()
{
    g_runner_instance = this;
}

SimulationRunner::~SimulationRunner()
{
    if (m_window) {
        mjr_freeContext(&m_con);
        mjv_freeScene(&m_scn);
        glfwDestroyWindow(m_window);
        glfwTerminate();
        m_window = nullptr;
    }

    delete m_controller;
    delete m_interface;

    if (m_data)  mj_deleteData(m_data);
    if (m_model) mj_deleteModel(m_model);

    g_runner_instance = nullptr;
}

bool SimulationRunner::init(const std::string& xml_path)
{
    char error[1000] = {0};
    m_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
    if (!m_model) {
        std::cerr << "[Runner Error] 无法加载 XML 模型: " << xml_path << "\n"
                  << "MuJoCo 错误信息: " << error << std::endl;
        return false;
    }

    m_data = mj_makeData(m_model);
    if (!m_data) {
        std::cerr << "[Runner Error] 无法为模型分配 mjData 内存！" << std::endl;
        return false;
    }

    // 实例化接口层与控制算法层
    m_interface  = new RobotInterface(m_model, m_data);
    m_controller = new RobotController();

    // 物理初始姿态标定就位
    reset();

    std::cout << "=========================================================\n"
              << "  RoboWalker 2026 轮腿机器人 - C++ 仿真引擎初始化就绪\n"
              << "  - 物理时间步长 dt: " << m_model->opt.timestep * 1000.0 << " ms (1000Hz)\n"
              << "  - 模型广义自由度 nq: " << m_model->nq << ", nv: " << m_model->nv << "\n"
              << "  - 3 层解耦架构: Interface -> Controller -> Runner 绑定成功\n"
              << "=========================================================" << std::endl;
    return true;
}

void SimulationRunner::reset()
{
    if (!m_model || !m_data) return;

    mj_resetData(m_model, m_data);

    // 黄金初始位姿标定 (高刚度无沉降 12μm 零位)
    m_data->qpos[0] = 0.0;
    m_data->qpos[1] = 0.0;
    m_data->qpos[2] = 0.04905; // 标称质心离地高度
    m_data->qpos[3] = 1.0;     // 四元数 w
    m_data->qpos[4] = 0.0;
    m_data->qpos[5] = 0.0;
    m_data->qpos[6] = 0.0;
    m_data->qpos[7] = -0.0042; // 左髋质心偏置角
    m_data->qpos[8] = 0.0;
    m_data->qpos[9] = -0.0042; // 右髋质心偏置角
    m_data->qpos[10] = 0.0;

    // 前向运动学刷新
    mj_forward(m_model, m_data);

    // 控制器与接口状态清零
    if (m_controller) m_controller->reset(0.0, 0.0);
    if (m_interface)  m_interface->stop_all_motors();

    m_push_remaining_time = 0.0;
    m_push_force = 0.0;
    m_sim_time = 0.0;
    m_step_count = 0;
    m_last_mode = ControlMode::HOLD_POSITION;
    m_last_fallen = false;
    m_last_v_cmd = 0.0;
    m_last_yaw_cmd = 0.0;

    std::cout << "\n>> [EVENT] 仿真状态已重置，机器人就位直立静止 (Hold Mode)！\n" << std::endl;
}

void SimulationRunner::apply_external_push(double force_x, double duration_sec)
{
    m_push_force = force_x;
    m_push_remaining_time = duration_sec;
    std::cout << "\n>> [DISTURBANCE] 施加外部脉冲冲击推力: "
              << std::fixed << std::setprecision(2) << force_x << " N (持续 "
              << (int)(duration_sec * 1000.0) << " ms)..." << std::endl;
}

void SimulationRunner::step_simulation()
{
    if (m_paused || !m_model || !m_data) return;

    // 0. 外部推力扰动注入 (注入至 base_link body 1)
    if (m_push_remaining_time > 0.0) {
        m_data->xfrc_applied[1 * 6 + 0] = m_push_force; // 沿世界X轴推力
        m_push_remaining_time -= m_model->opt.timestep;
        if (m_push_remaining_time <= 0.0) {
            m_data->xfrc_applied[1 * 6 + 0] = 0.0;
            std::cout << ">> [DISTURBANCE] 外部推力结束，自主平衡恢复中..." << std::endl;
        }
    }

    // 1. Interface 层读取全车传感器状态
    RobotSensors sensors;
    m_interface->update_sensors(sensors);

    // 2. 跌倒安全判定
    bool fallen = m_interface->is_fallen(sensors);

    // 3. Controller 层运算并输出给 Interface 层写入执行器
    if (fallen) {
        m_interface->stop_all_motors();
    } else {
        RobotActuators actuators;
        m_controller->compute(sensors, actuators, m_model->opt.timestep);
        m_interface->apply_actuators(actuators, true);
    }

    // 4. MuJoCo 物理积分推进 1 步 (1ms)
    mj_step(m_model, m_data);
    m_sim_time += m_model->opt.timestep;
    m_step_count++;

    // 5. 事件驱动防刷屏日志
    handle_event_logging(sensors);
}

void SimulationRunner::handle_event_logging(const RobotSensors& sensors)
{
    bool fallen = m_interface->is_fallen(sensors);
    ControlMode mode = m_controller->get_mode();
    double v_cmd = m_controller->get_target_velocity();

    // 仅在状态改变、跌倒或遥控设定值改变时打印
    bool state_changed = (fallen != m_last_fallen) ||
                         (mode != m_last_mode) ||
                         (std::abs(v_cmd - m_last_v_cmd) > 1e-4);

    if (state_changed) {
        std::cout << "[" << std::fixed << std::setprecision(3) << m_sim_time << "s] ";
        if (fallen) {
            std::cout << "[FALL ALERT] 机器人严重倾斜跌倒！电机力矩已自动切断急停。按 'R' 键扶起重置。" << std::endl;
        } else if (mode == ControlMode::HOLD_POSITION) {
            std::cout << "[STATE] 进入驻车稳态自平衡 | 坐标 x: "
                      << std::setprecision(4) << sensors.x_pos
                      << " m | 俯仰角: " << sensors.pitch * 180.0 / M_PI << "°" << std::endl;
        } else if (mode == ControlMode::CRUISE_SPEED) {
            std::cout << "[STATE] 巡航速度跟踪模式 | 目标速度: "
                      << std::setprecision(2) << v_cmd
                      << " m/s | 实际航速: " << sensors.forward_vel << " m/s" << std::endl;
        }
        m_last_fallen = fallen;
        m_last_mode   = mode;
        m_last_v_cmd  = v_cmd;
    }
}

void SimulationRunner::init_visual_resources()
{
    if (!glfwInit()) {
        std::cerr << "[GLFW Error] 无法初始化 GLFW 渲染子系统！" << std::endl;
        return;
    }

    // 创建 1200x800 高清窗口
    m_window = glfwCreateWindow(1200, 800, "RoboWalker 2026 - Wheel-Leg LQR Simulation (C++ Native)", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "[GLFW Error] 窗口创建失败！" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // 开启垂直同步

    // 初始化 MuJoCo 内部渲染构件
    mjv_defaultCamera(&m_cam);
    mjv_defaultOption(&m_opt);
    mjv_defaultScene(&m_scn);
    mjr_defaultContext(&m_con);

    mjv_makeScene(m_model, &m_scn, 2000);
    mjr_makeContext(m_model, &m_con, mjFONTSCALE_150);

    // 黄金跟车视角配置
    m_cam.distance  = 0.25;  // 黄金跟车视距 25cm
    m_cam.elevation = -18.0; // 俯仰角 -18度
    m_cam.azimuth   = 140.0; // 观察偏航角 140度

    // 绑定 GLFW 键鼠回调
    glfwSetKeyCallback(m_window, glfw_key_callback);
    glfwSetMouseButtonCallback(m_window, glfw_mouse_button_callback);
    glfwSetCursorPosCallback(m_window, glfw_cursor_pos_callback);
    glfwSetScrollCallback(m_window, glfw_scroll_callback);

    std::cout << "\n================== [操作交互快捷键指南] ==================\n"
              << "  [方向键 ↑ / ↓] : 前进加速 (+0.05 m/s) / 后退减速 (-0.05 m/s)\n"
              << "  [方向键 ← / →] : 左转微调 (+0.15 rad) / 右转微调 (-0.15 rad)\n"
              << "  [Space 空格]   : 紧急刹车并原地驻车锁死\n"
              << "  [P 键]         : 施加前向 +0.25 N 脉冲推力扰动\n"
              << "  [B 键]         : 施加后向 -0.25 N 脉冲推力扰动\n"
              << "  [R 键]         : 扶起机器人重置初始稳态\n"
              << "  [鼠标左键拖拽] : 旋转 3D 观察视角\n"
              << "  [鼠标右键拖拽] : 平移观察视野\n"
              << "  [鼠标滚轮]     : 缩放视野距离\n"
              << "=========================================================\n" << std::endl;
}

void SimulationRunner::render_frame()
{
    if (!m_window) return;

    // 摄像机平滑跟踪机身质心
    m_cam.lookat[0] = m_data->qpos[0];
    m_cam.lookat[1] = m_data->qpos[1];
    m_cam.lookat[2] = m_data->qpos[2];

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    // 更新场景与绘制
    mjv_updateScene(m_model, m_data, &m_opt, nullptr, &m_cam, mjCAT_ALL, &m_scn);
    mjr_render(viewport, &m_scn, &m_con);

    glfwSwapBuffers(m_window);
}

void SimulationRunner::run_gui()
{
    init_visual_resources();
    if (!m_window) {
        std::cerr << "[Runner Warning] 无法创建 GUI 窗口，将自动回退至命令行无头仿真模式。" << std::endl;
        run_headless(10.0);
        return;
    }

    // 60fps 渲染步长调度器 (每帧推进 ~16.6ms 物理步)
    while (!glfwWindowShouldClose(m_window)) {
        double sim_start = m_sim_time;
        // 维持 60fps 与物理 1000Hz 时间同步
        while (m_sim_time - sim_start < 1.0 / 60.0) {
            step_simulation();
        }

        render_frame();
        glfwPollEvents();
    }
}

void SimulationRunner::run_headless(double duration_sec)
{
    std::cout << ">> [RUNNER] 启动命令行无头仿真评测 (总时长: " << duration_sec << " 秒)..." << std::endl;
    long long total_steps = (long long)(duration_sec / m_model->opt.timestep);

    for (long long s = 0; s < total_steps; ++s) {
        step_simulation();
    }
    std::cout << ">> [RUNNER] 无头仿真测试圆满完成，共执行 " << total_steps << " 步！" << std::endl;
}

// ---------------- GLFW 静态键鼠事件中转 ----------------

void SimulationRunner::glfw_key_callback(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
    if (!g_runner_instance || !g_runner_instance->m_controller) return;

    auto* ctrl = g_runner_instance->m_controller;

    switch (key) {
        case GLFW_KEY_UP:    // 前进加速
            ctrl->change_target_velocity(+0.05);
            break;
        case GLFW_KEY_DOWN:  // 后退减速
            ctrl->change_target_velocity(-0.05);
            break;
        case GLFW_KEY_LEFT:  // 左转
            ctrl->change_target_yaw(+0.15);
            break;
        case GLFW_KEY_RIGHT: // 右转
            ctrl->change_target_yaw(-0.15);
            break;
        case GLFW_KEY_SPACE: // 刹车驻停
            ctrl->trigger_brake(g_runner_instance->m_data->qpos[0]);
            std::cout << "\n>> [MANUAL] 触发紧急刹车，驻车位置锁死在当前点！" << std::endl;
            break;
        case GLFW_KEY_P:     // 前推扰动
            g_runner_instance->apply_external_push(+0.25, 0.06);
            break;
        case GLFW_KEY_B:     // 后推扰动
            g_runner_instance->apply_external_push(-0.25, 0.06);
            break;
        case GLFW_KEY_R:     // 重置扶起
            g_runner_instance->reset();
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
    }
}

void SimulationRunner::glfw_mouse_button_callback(GLFWwindow* window, int button, int act, int mods)
{
    if (!g_runner_instance) return;
    g_runner_instance->m_button_left   = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    g_runner_instance->m_button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    g_runner_instance->m_button_right  = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &g_runner_instance->m_last_x, &g_runner_instance->m_last_y);
}

void SimulationRunner::glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (!g_runner_instance) return;
    auto* r = g_runner_instance;
    if (!r->m_button_left && !r->m_button_middle && !r->m_button_right) return;

    double dx = xpos - r->m_last_x;
    double dy = ypos - r->m_last_y;
    r->m_last_x = xpos;
    r->m_last_y = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (r->m_button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (r->m_button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(r->m_model, action, dx / height, dy / height, &r->m_cam);
}

void SimulationRunner::glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (!g_runner_instance) return;
    mjv_moveCamera(g_runner_instance->m_model, mjMOUSE_ZOOM, 0, -0.05 * yoffset,
                   &g_runner_instance->m_cam);
}
