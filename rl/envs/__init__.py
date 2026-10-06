"""
rl.envs: 轮腿机器人强化学习仿真环境与步进引擎模块
==================================================
包含模块:
  1. WheelLegRoughTerrainEnv: Gymnasium 轮腿机器人复杂地形强化学习标准环境
  2. sim_runner: 统一单回合仿真步进与遥测采集引擎 (build_single_obs, run_episode)
"""
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
         os.path.join(rl_dir, "controllers"),
         os.path.join(rl_dir, "terrain"),
         os.path.join(rl_dir, "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .wheel_leg_env import WheelLegRoughTerrainEnv
from .sim_runner import build_single_obs, run_episode, kinematic_slip, R_WHEEL, L_LEG, LINE_Y

__all__ = [
    "WheelLegRoughTerrainEnv",
    "build_single_obs",
    "run_episode",
    "kinematic_slip",
    "R_WHEEL",
    "L_LEG",
    "LINE_Y",
]
