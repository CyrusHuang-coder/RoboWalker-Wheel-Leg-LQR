"""
rl.training: 轮腿机器人课程强化学习与多阶段训练模块
======================================================
包含模块:
  1. CurriculumManager: 动态课程学习调度器 (步数退火 + 胜率门控)
  2. train_curriculum: PRCC-RL 多阶段渐进课程学习主入口
  3. train_rl: 基础单阶段 PPO 强化学习训练入口
"""
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
         os.path.join(rl_dir, "controllers"),
         os.path.join(rl_dir, "envs"),
         os.path.join(rl_dir, "terrain"),
         os.path.join(rl_dir, "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .curriculum_manager import CurriculumManager

__all__ = ["CurriculumManager"]
