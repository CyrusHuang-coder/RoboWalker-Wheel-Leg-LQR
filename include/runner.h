#pragma once

#include <string>
#include <mujoco/mujoco.h>
#include "interface.h"
#include "controller.h"

// 前向声明 GLFW 窗口结构体
struct GLFWwindow;

/**
 * @brief 仿真运行调度与交互类 (Runner Layer)
 * 职责：
 *  1. 加载 MuJoCo XML 模型文件并初始化世界状态
 *  2. 调度推进 1000Hz 物理主循环 (mj_step)
 *  3. 连接 Interface 与 Controller，执行闭环数据流
 *  4. 维护 GLFW 3D 渲染窗口、跟踪摄像机与键鼠交互
 *  5. 状态机事件监测与终端防刷屏状态打印
 */
class SimulationRunner {
public:
    SimulationRunner();
    ~SimulationRunner();

    // 禁用拷贝与赋值
    SimulationRunner(const SimulationRunner&) = delete;
    SimulationRunner& operator=(const SimulationRunner&) = delete;

    /**
     * @brief 初始化 MuJoCo 仿真模型与接口控制器
     * @param xml_path XML 物理描述文件路径
     * @return true 初始化成功，false 失败
     */
    bool init(const std::string& xml_path);

    /**
     * @brief 启动 GUI 交互仿真窗口 (GLFW 渲染与键盘控制)
     */
    void run_gui();

    /**
     * @brief 启动无头模式仿真 (命令行批量测试与自动化评测)
     * @param duration_sec 仿真总时长 (秒)
     */
    void run_headless(double duration_sec = 10.0);

    /**
     * @brief 执行单个 1000Hz 物理控制步进
     */
    void step_simulation();

    /**
     * @brief 重置仿真状态 (扶起小车、清空力矩与积分器)
     */
    void reset();

    /**
     * @brief 向车身施加瞬态脉冲推力扰动 (用于抗扰鲁棒性测试)
     * @param force_x 前向推力 (N)
     * @param duration_sec 持续时间 (秒)
     */
    void apply_external_push(double force_x, double duration_sec = 0.06);

private:
    // MuJoCo 核心数据结构指针
    mjModel* m_model = nullptr;
    mjData*  m_data  = nullptr;

    // 解耦层实例指针
    RobotInterface*  m_interface  = nullptr;
    RobotController* m_controller = nullptr;

    // 仿真步进统计
    double m_sim_time = 0.0;
    long long m_step_count = 0;
    bool m_paused = false;

    // 扰动状态
    double m_push_remaining_time = 0.0;
    double m_push_force = 0.0;

    // 事件驱动打印历史缓存
    ControlMode m_last_mode = ControlMode::HOLD_POSITION;
    bool m_last_fallen = false;
    double m_last_v_cmd = 0.0;
    double m_last_yaw_cmd = 0.0;

    // GLFW 可视化渲染内部资源
    mjvCamera  m_cam;
    mjvOption  m_opt;
    mjvScene   m_scn;
    mjrContext m_con;
    GLFWwindow* m_window = nullptr;

    // 鼠标拖拽状态
    bool m_button_left   = false;
    bool m_button_middle = false;
    bool m_button_right  = false;
    double m_last_x = 0;
    double m_last_y = 0;

    // 内部帮助函数
    void init_visual_resources();
    void render_frame();
    void handle_event_logging(const RobotSensors& sensors);

    // GLFW 回调静态中转函数
    static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int act, int mods);
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int act, int mods);
    static void glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};
