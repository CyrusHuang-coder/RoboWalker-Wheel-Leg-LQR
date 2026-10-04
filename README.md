# RoboWalker 2026 轮腿自平衡机器人仿真与控制算法工程
> **战队项目**：RoboWalker 2026 赛季第一期考核大作业 · 轮腿机器人 LQR 仿真控制  
> **核心目标**：基于 **LQR 现代控制理论** 与 **MuJoCo 物理引擎**，实现双轮腿机器人的自平衡直立、纵向巡航机动、差速转向与外部抗扰自恢复。  
> **工程规范**：严格遵循《概要.pdf》第 7 页的 **Interface - Controller - Runner 三层硬件抽象与解耦架构**，支持 C++ 原生运行与 Python 交互验证。

---

## 🎬 官方动态效果展示 (Official Demo)

![RoboWalker 2026 轮腿机器人演示](./demo/wheel_leg_demo.gif)

> **演示机动全流程 (11.0 秒, 30 fps 高清跟踪视角)**：
> 1. `0.0s ~ 2.5s`：**原地稳态自平衡**（静止于坐标原点，质心稳态误差 < 0.1 mm）；
> 2. `2.5s ~ 5.5s`：**前向巡航加速跟踪**（身体平滑前倾，遥控巡航速度 $v = +0.08\text{ m/s}$）；
> 3. `5.5s ~ 8.0s`：**行进间差速转弯机动**（偏航角 $\psi$ 跟踪转向，内力偶动力学正交解耦）；
> 4. `8.0s ~ 8.06s`：**突发 $0.25\text{ N}$ 强力脉冲扰动**（相当于机器人自重 13.5% 的瞬态冲击）；
> 5. `8.1s ~ 11.0s`：**自适应俯冲追赶平衡**并平滑回位刹车，重新锁死驻车稳态。

---

## 🏗️ 核心三层解耦工程架构 (Three-Tier Architecture)

控制算法层（Controller）**完全不依赖任何 MuJoCo 头文件或底层硬件代码**，真正实现“算法与仿真/硬件平台彻底解耦”，可 100% 无缝移植至真实机器人的嵌入式主控（如 STM32 或 ROS2 节点）。

```
 真实机器人 (STM32 / CAN总线)             MuJoCo 物理仿真环境
┌───────────────────────────┐         ┌───────────────────────────┐
│ CAN总线 / BMI088 IMU / 编码器│         │   mjData (qpos, qvel)     │
└─────────────┬─────────────┘         └─────────────┬─────────────┘
              │                                     │
              ▼                                     ▼
      ┌─────────────────────────────────────────────────────┐
      │            1. Interface 层 (接口抽象层)              │  <--- 抹平物理与仿真的鸿沟
      │  - 头文件: include/interface.h                      │
      │  - 源文件: src/interface.cpp                        │
      │  - 职责: 提取欧拉角/线速度; 电机力矩安全硬限幅 (0.05N·m)│
      └─────────────────────────┬───────────────────────────┘
                                │ 标准状态观测 (RobotSensors)
                                ▼
      ┌─────────────────────────────────────────────────────┐
      │            2. Controller 层 (算法控制层)            │  <--- 纯净的控制算法大脑
      │  - 头文件: include/controller.h                     │
      │  - 源文件: src/controller.cpp                       │
      │  - 职责: 串级纵向外环 + LQR平衡内环 + 偏航差速 + 腿PD│
      └─────────────────────────┬───────────────────────────┘
                                │ 期望控制指令 (RobotActuators)
                                ▲
      ┌─────────────────────────┴───────────────────────────┐
      │            3. Runner 层 (运行调度与交互层)          │  <--- 运行骨架与外界交互
      │  - 头文件: include/runner.h                         │
      │  - 源文件: src/runner.cpp, src/main.cpp             │
      │  - 职责: 1000Hz 物理主循环 (mj_step) + GLFW 3D 渲染 │
      └─────────────────────────────────────────────────────┘
```

---

## 🚀 快速启动指南 (Quick Start)

本项目提供 **C++ 原生高性能可执行程序** 与 **Python 交互式调参原型** 双轨支持。

### 1. 运行 C++ 原生仿真工程 (推荐 · 评审首选)
本项目已实现**自包含便携部署（Self-Contained Deployment）**，内置全部所需动态库，无需额外配置环境变量：

```powershell
# 运行 C++ 原生仿真程序 (即开即玩)
.\bin\wheel_leg_sim.exe
```

若需重新编译或二次开发：
```powershell
# 方式 A: 基于 CMake + Ninja 构建
cmake -B build -G "Ninja"
cmake --build build

# 方式 B: 基于 g++ 直接单行编译
g++ -std=c++17 -O2 -Iinclude -I"C:\Users\lenovo\miniforge3\envs\wheel_leg\Lib\site-packages\mujoco\include" -I"C:\Users\lenovo\miniforge3\envs\wheel_leg\Lib\site-packages\mujoco\include\mujoco" -I"C:\msys64\ucrt64\include" src/interface.cpp src/controller.cpp src/runner.cpp src/main.cpp -L"C:\Users\lenovo\miniforge3\envs\wheel_leg\Lib\site-packages\mujoco" -l:mujoco.dll -L"C:\msys64\ucrt64\lib" -lglfw3 -lopengl32 -lgdi32 -o bin/wheel_leg_sim.exe
```

---

### 2. 运行 Python 闭环交互原型与调参工具
进入项目的专属 Conda 环境 `wheel_leg`：

```powershell
# 1. 启动 Python 实时交互仿真 (支持键盘控制与事件驱动打印)
python scripts/test_balance.py

# 2. 求解连续代数黎卡提方程 (CARE) 计算 LQR 增益矩阵 K
python LQR计算代码/calculate.py

# 3. 重新录制官方演示 GIF 动图
python scripts/record_demo.py

# 4. 纯静态模型查看器 (支持鼠标施加外力拖拽)
python scripts/view_model.py
```

---

## 🎮 键盘与鼠标交互指令指南 (Teleoperation)

在仿真运行窗口中，键盘键位设计经过优化，**彻底规避了 MuJoCo 内置单键冲突**（如 `W` 键网格化、`M` 键连杆胶囊化）：

| 操作按键 | 功能说明 | 物理响应与控制逻辑 |
| :--- | :--- | :--- |
| **`↑` (方向键上)** | **前向巡航加速** | 目标速度 $+0.05\text{ m/s}$，机身前倾产生加速度 |
| **`↓` (方向键下)** | **后退/减速** | 目标速度 $-0.05\text{ m/s}$，机身后倾产生减速度 |
| **`←` (方向键左)** | **偏航逆时针左转** | 航向目标 $+0.15\text{ rad}$，左轮减速右轮加速 |
| **`→` (方向键右)** | **偏航顺时针右转** | 航向目标 $-0.15\text{ rad}$，左轮加速右轮减速 |
| **`Space` (空格键)** | **紧急驻车刹车** | 目标速度瞬间清零，**死锁当前位置坐标原点** |
| **`P` 键** | **前向脉冲推力扰动** | 向机身注入 $+0.25\text{ N}$ 瞬时推力（持续 60ms） |
| **`B` 键** | **后向脉冲推力扰动** | 向机身注入 $-0.25\text{ N}$ 瞬时推力（持续 60ms） |
| **`R` 键** | **系统复位扶起** | 重置积分器，小车在原点重新恢复直立稳态 |
| **鼠标左键拖拽** | **3D 视角旋转** | 绕机器人中心球形旋转观察视角 |
| **鼠标右键拖拽** | **视角平移** | 平移跟踪观察视线 |
| **鼠标滚轮** | **视角缩放** | 调整与机器人的跟车视距 |

---

## 🔬 核心攻坚亮点与技术指标

1. **高刚度接触模型（12 微米极限沉陷）**：
   - 彻底解决默认柔性接触参数下 $189\text{ g}$ 微型小车下陷 $6.14\text{ mm}$（占轮半径 $77\%$）的严重穿透缺陷；
   - 标定刚性阻抗参数（`solref="0.002 1"` 与 `solimp="0.99 0.999 0.0001 0.5 2"`），实测沉陷量降至 **$0.012\text{ mm}$（12 微米）**，车轮完整圆润着地。
2. **反作用力矩佯谬与串级控制（Cascade Control）**：
   - 从牛顿第三定律严格证明两轮倒立摆无法在垂直状态直立加速的本质机理；
   - 构建“纵向外环解算目标前倾角 $\theta_{\text{target}}$ + 内环状态反馈快速追角”的串级拓扑，兼顾加速敏捷性与抗扰稳健性。
3. **高频冲击滤波与正交差速解耦**：
   - 针对接触碰撞噪声引入截止频率 $23.8\text{ Hz}$ 的离散一阶低通滤波器（LPF）；
   - 偏航差速控制与纵向平衡力矩线性解耦，转向过程零轴向力扰动。
4. **自包含绿色便携交付**：
   - 解决 Windows 控制台默认 GBK 代码页导致的多字节中文乱码（调用 `SetConsoleOutputCP(CP_UTF8)` 动态切换）；
   - 打包全部 GCC/GLFW/MuJoCo 运行时 DLL 至 `bin/` 目录，脱离开发环境免安装秒开。

---

## 📂 项目完整精简文件目录导航

```text
wheel-leg/
├── README.md              # [工程首页] 架构说明、动态展示与全套使用指南
├── CMakeLists.txt         # [构建工程] 标准 C++17 跨平台构建配置文件
├── wheel_leg.xml          # [MuJoCo核心] 12微米高刚度接触、对称轮系与传感器执行器描述
│
├── bin/                   # [可执行与运行时库] 独立自包含便携运行目录
│   ├── wheel_leg_sim.exe  # C++ 原生仿真可执行程序
│   ├── mujoco.dll         # MuJoCo 物理引擎运行时动态库
│   ├── glfw3.dll          # GLFW 窗口与渲染交互动态库
│   ├── libstdc++-6.dll    # GCC C++ 标准库运行时
│   ├── libgcc_s_seh-1.dll # GCC C 核心运行时
│   └── libwinpthread-1.dll# POSIX 线程支持库
│
├── include/               # [C++头文件] 硬件抽象与控制算法解耦接口定义
│   ├── interface.h        # 硬件与仿真抽象接口类定义 (RobotSensors, RobotActuators)
│   ├── controller.h       # 核心控制算法大脑类定义 (纯数学与逻辑)
│   └── runner.h           # 仿真调度主循环与 GLFW 渲染交互类定义
│
├── src/                   # [C++源文件] 核心三层架构实现
│   ├── interface.cpp      # 传感器地址快速缓存、四元数转欧拉角、力矩硬限幅
│   ├── controller.cpp     # 串级前倾外环、LQR内环、偏航差速、虚拟腿PD
│   ├── runner.cpp         # 1000Hz 物理调度、60fps渲染同步、GLFW键鼠回调
│   └── main.cpp           # 程序启动入口与 Windows UTF-8 代码页初始化
│
├── rl/                    # [强化学习进阶模块 (PRCC Residual RL)]
│   ├── prior_controller.py# 经典串级先验控制器 (无缝对接 RL 残差与刚度缩放接口)
│   ├── wheel_leg_env.py   # Gymnasium 标准盲走环境 (69维本体感知 + 4维动作解映射 + 正则奖励)
│   ├── train_rl.py        # PPO 强化学习高吞吐训练调度器 (DummyVecEnv 并行加速)
│   ├── eval_compare.py    # Baseline LQR vs PRCC-RL 自动化量化评测与学术图表导出
│   ├── record_compare_demo.py # 高清并排对比 GIF 离线渲染脚本
│   ├── evaluation_comparison.png # 4.0米全地形对比评测高清图表
│   ├── models/            # [模型权重]
│   │   └── best_model.zip # 经过 120k 步收敛锁定的最佳残差策略网络权重
│   └── terrain/           # [轻起伏地形资产]
│       ├── generate_terrain.py # 高保真波浪 + 左右交错高斯隆起地形生成器
│       ├── rough_terrain.png   # 16-bit 高精度地形高度图 (4.0m x 1.0m)
│       └── wheel_leg_terrain.xml# 嵌入 Heightfield 地形的 MuJoCo 仿真物理描述
│
├── scripts/               # [Python算法与工具包]
│   ├── test_balance.py    # 闭环平衡与遥控交互原型脚本 (带事件驱动防刷屏日志)
│   ├── record_demo.py     # 11.0秒全流程机动动作离线跟踪渲染与 GIF 编码脚本
│   └── view_model.py      # 交互式 3D 模型轻量级查看器
│
├── docs/                  # [完整技术文档与图表资产]
│   ├── 任务报告.md        # [总考核报告] 全流程成果汇报、LQR与RL两阶段全汇总、指标看板与技术沉淀
│   ├── 操作手册.md        # [实操SOP] 3天冲刺排期、实测参数表、构建指令与变更日志
│   ├── 学习手册.md        # [通识宝典] 动力学推导、反作用力矩佯谬、四大版本调优全纪实与答辩题库
│   ├── 概要.pdf           # 官方大作业任务书
│   └── images/            # 仿真装配与接触校验高清图像 (final_verified_*.png)
│
├── demo/                  # [交付物成果归档]
│   ├── wheel_leg_demo.gif # 官方 11.0s 平地基础自平衡与抗扰演示 (30 fps, 7.15 MB)
│   └── wheel_leg_rl_comparison.gif # LQR vs PRCC-RL 全赛道起伏越障并排对比动画 (16 fps, 13.2s, 9.7 MB)
│
├── car_urdf/              # [URDF功能包] 轻量化几何网格 (STL) 与运动学拓扑
├── LQR计算代码/           # [参数与求解] parameter.py (19项物理参数) 与 calculate.py (Riccati求解器)
├── archive/               # [历史探索档案] 24份真实探索/调试全过程脚本与原始模型转储
└── 仿真大作业_模型/       # 原始 SolidWorks CAD 装配体与零件包
```

---

## 📚 配套手册与报告全集阅读导航

- **想全面通读整期大作业的成果汇总、理论创新与量化对比评测？**  
  👉 请查阅：**[《docs/任务报告.md》](./docs/任务报告.md)**
- **想了解完整复现流水线、构建命令与操作步骤？**  
  👉 请查阅：**[《docs/操作手册.md》](./docs/操作手册.md)**
- **想搞懂深入数学推导、排错避坑经历、四大版本调优档案与战队答辩核心题库？**  
  👉 请查阅：**[《docs/学习手册.md》](./docs/学习手册.md)**
- **想回顾开发过程中的 24 份调试演进脚本？**  
  👉 请查阅：**[《archive/README.md》](./archive/README.md)**

---

**战队声明**：本项目由 RoboWalker 控制算法组 2026 赛季大作业规范要求开发，代码遵循模块化解耦标准，版权与知识产权归 RoboWalker 战队及作者所有。