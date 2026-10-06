"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/__init__.py`: 算法评估、基准测试与指标分析组件包初始化
====================================================================================================
导出组件：
  - ContactSlipMeter: 接触面微观滑移测量计
  - high_freq_energy: 力矩高频能谱分析函数
  - extract_sensors: 统一标准化传感器观测量字典提取函数
====================================================================================================
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from metrics_utils import ContactSlipMeter, high_freq_energy
from eval_baseline import extract_sensors

__all__ = [
    "ContactSlipMeter",
    "high_freq_energy",
    "extract_sensors"
]
