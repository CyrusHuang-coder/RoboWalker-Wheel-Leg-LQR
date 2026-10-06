"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/__init__.py`: 轮腿机器人强化学习与控制系统根包 (Root Package Initialization & Facade)
====================================================================================================
1. 模块化子系统导航 (Subsystems Architecture):
   - `rl.controllers` : 核心控制算法库 (PriorController 先验平衡控制器, SlipCoupledEnergyTank 能量储罐)
   - `rl.envs`        : 仿真环境与步进引擎 (WheelLegRoughTerrainEnv 强化学习环境, sim_runner 遥测运行器)
   - `rl.training`    : 策略训练流水线 (CurriculumManager 课程调度器, train_curriculum, train_rl)
   - `rl.evaluation`  : 算法评估与基准测试 (ContactSlipMeter 接触滑移计, 蒙特卡洛评测脚本)
   - `rl.recorders`   : 高精仿真渲染与视频录制 (10.5m 三联对比动图, 关键帧提取)
   - `rl.terrain`     : 赛道建模与程序化高程生成 (10.5m 复合赛道, 4.0m 随机地形生成器)

2. 全局符号导出 (Top-Level Facade Exports):
   向顶层统一导出常用核心类与函数，保障旧版导入与新版模块化导入的双向无缝兼容。
====================================================================================================
"""

import os
import sys

# 注入所有子包目录至搜索路径，实现绝对与相对导入的全局鲁棒性
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_dirs = ["controllers", "envs", "training", "evaluation", "recorders", "terrain", "tests"]
for d in sub_dirs:
    dp = os.path.join(current_dir, d)
    if dp not in sys.path:
        sys.path.insert(0, dp)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 统一导出常用核心对象
from controllers.prior_controller import PriorController, quat2rpy
from controllers.energy_tank import SlipCoupledEnergyTank
from envs.wheel_leg_env import WheelLegRoughTerrainEnv
from envs.sim_runner import run_episode, build_single_obs
from terrain.procedural_terrain import ProceduralTerrainGenerator
from terrain.test_mega_terrain import build_mega_hfield, get_mega_track_reference

__all__ = [
    "PriorController",
    "quat2rpy",
    "SlipCoupledEnergyTank",
    "WheelLegRoughTerrainEnv",
    "run_episode",
    "build_single_obs",
    "ProceduralTerrainGenerator",
    "build_mega_hfield",
    "get_mega_track_reference"
]
