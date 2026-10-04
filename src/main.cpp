#include "runner.h"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv)
{
#ifdef _WIN32
    // 强制将 Windows 控制台输出代码页切换为 UTF-8 (CP 65001)，彻底消除中文乱码
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string xml_path = "wheel_leg.xml";
    bool headless = false;
    bool user_specified_xml = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else {
            xml_path = arg;
            user_specified_xml = true;
        }
    }

    // 智能定位模型文件：若当前目录未找到，自动在上一级目录查找
    if (!user_specified_xml) {
        FILE* f = fopen(xml_path.c_str(), "r");
        if (f) {
            fclose(f);
        } else {
            FILE* f2 = fopen("../wheel_leg.xml", "r");
            if (f2) {
                xml_path = "../wheel_leg.xml";
                fclose(f2);
            }
        }
    }

    std::cout << ">>> 正在启动 RoboWalker 2026 轮腿自平衡机器人仿真系统..." << std::endl;
    std::cout << ">>> 加载模型: " << xml_path << std::endl;

    SimulationRunner runner;
    if (!runner.init(xml_path)) {
        std::cerr << "[FATAL] 仿真模型初始化失败，程序终止。" << std::endl;
        return 1;
    }

    if (headless) {
        runner.run_headless(2.0);
    } else {
        // 启动 3D 可视化渲染与键鼠交互窗口
        runner.run_gui();
    }

    std::cout << ">>> 仿真已正常退出，感谢使用！" << std::endl;
    return 0;
}
