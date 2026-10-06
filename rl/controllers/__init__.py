"""
rl.controllers: 轮腿机器人底层控制律与无源能量约束模块
======================================================
包含模块:
  1. PriorController: 先验复合倒立摆平衡与巡线控制器
  2. SlipCoupledEnergyTank: 滑移耦合精确离散能量储罐 (SCT-RRL)
"""
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .prior_controller import PriorController, quat2rpy
from .energy_tank import SlipCoupledEnergyTank

__all__ = ["PriorController", "quat2rpy", "SlipCoupledEnergyTank"]
