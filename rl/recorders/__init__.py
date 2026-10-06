"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/recorders/__init__.py`: 物理仿真渲染与多算法对比录制组件包初始化
====================================================================================================
导出组件：
  - record_extended_comparison: 10.5m 复合恶劣赛道三大算法同台横向对比 GIF/关键帧录制器
  - record_tri_comparison: 4.0m 程序化起伏赛道三大算法同台竞技对比录制器
  - record_comparison: 4.0m 基础赛道 Pure LQR vs PRCC-RL 双联对比录制器
====================================================================================================
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from record_tri_extended_comparison import record_extended_comparison
from record_tri_comparison import record_tri_comparison
from record_compare_demo import record_comparison

__all__ = [
    "record_extended_comparison",
    "record_tri_comparison",
    "record_comparison"
]
