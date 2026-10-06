"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/terrain/__init__.py`: 赛道建模与程序化地形生成核心组件包初始化
====================================================================================================
导出组件：
  - ProceduralTerrainGenerator: 纯内存高效动态地形生成器 (支持课程难度无缝自适应映射)
  - generate_terrain: 4.0m 基础测试赛道离线 PNG 高程生成函数
  - build_mega_hfield: 10.5m 复合恶劣赛道高精连续实体高度场矩阵生成函数
  - get_mega_track_reference: 10.5m 赛道理想横向偏置与期望航向角轨迹参考模型
  - OFF_LEFT_WHEEL, OFF_RIGHT_WHEEL, OFF_TRACK_MID: 实车轮轴接触点物理偏置常量
====================================================================================================
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from procedural_terrain import ProceduralTerrainGenerator
from generate_terrain import generate_terrain
from test_mega_terrain import (
    build_mega_hfield,
    get_mega_track_reference,
    OFF_LEFT_WHEEL,
    OFF_RIGHT_WHEEL,
    OFF_TRACK_MID
)

__all__ = [
    "ProceduralTerrainGenerator",
    "generate_terrain",
    "build_mega_hfield",
    "get_mega_track_reference",
    "OFF_LEFT_WHEEL",
    "OFF_RIGHT_WHEEL",
    "OFF_TRACK_MID"
]
