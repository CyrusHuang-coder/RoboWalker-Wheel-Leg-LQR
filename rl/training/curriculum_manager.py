"""
curriculum_manager.py: 课程学习动态调度器 (Curriculum Learning Scheduler)
========================================================================================
【模块功能介绍】
本模块实现了自适应课程学习 (Curriculum Learning) 调度器，用于管理轮腿机器人从平坦路面
逐步过渡到极限复杂路面的渐进式训练全过程：

1. 阶梯式难度退火 (Difficulty Annealing):
   - Novice 阶段 (前 25% 步数): 难度 0.10 ~ 0.35，平地与小波浪，建立平稳巡航与初级吸震动作；
   - Adept 阶段 (25% ~ 65% 步数): 难度 0.35 ~ 0.75，引入非对称小板砖与交错搓板，学习腿部差动顺应；
   - Master 阶段 (65% ~ 100% 步数): 难度 0.75 ~ 1.00，全要素随机起伏与极限台阶，固化高鲁棒性。

2. 胜率门控保护机制 (Success Rate Gating):
   动态维护最近 N 个 Episode 的滚动通关率。当机器人遭遇瓶颈、通关率低于 60% 时，
   自动冻结或临时回调地形难度系数，防止奖励剧烈震荡导致策略网络梯度崩溃。
========================================================================================
"""
import os
import sys
import numpy as np

# 路径自适应引导
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


class CurriculumManager:
    """
    课程学习动态调度器
    支持多并行环境共享实例与无锁状态同步。
    """
    def __init__(self, total_timesteps=350000, 
                 start_difficulty=0.10, 
                 max_difficulty=1.0, 
                 win_rate_window=40,
                 min_pass_rate=0.75):
        """
        参数:
          - total_timesteps: 预计总训练步数
          - start_difficulty: 初始难度 (0.10)
          - max_difficulty: 最大难度 (1.00)
          - win_rate_window: 滚动统计最近 N 局的窗口尺寸 (默认 40)
          - min_pass_rate: 允许提升难度的最低完赛率阈值 (默认 75%)
        """
        self.total_timesteps = total_timesteps
        self.start_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.win_rate_window = win_rate_window
        self.min_pass_rate = min_pass_rate

        self.current_step = 0
        self.difficulty = start_difficulty
        self.episode_results = []  # 记录最近完成的 episode 是否成功越过全部障碍 (True/False)

    def record_episode(self, success: bool):
        """记录单次 Episode 越障结果"""
        self.episode_results.append(1.0 if success else 0.0)
        if len(self.episode_results) > self.win_rate_window:
            self.episode_results.pop(0)

    @property
    def rolling_success_rate(self) -> float:
        """计算最近 N 局的滚动通过率 (0.0 ~ 1.0)"""
        if not self.episode_results:
            return 1.0
        return float(np.mean(self.episode_results))

    def update_steps(self, total_env_steps: int):
        """
        根据训练总步数与成功率自适应更新难度
        采用平滑 S 曲线 (Cosine Annealing) 递增，辅以胜率门控
        """
        self.current_step = total_env_steps
        progress = float(np.clip(total_env_steps / float(self.total_timesteps), 0.0, 1.0))

        # 基础步数退火难度 (0.10 -> 1.0)
        base_diff = self.start_difficulty + (self.max_difficulty - self.start_difficulty) * (progress ** 1.2)

        # 胜率保护机制：如果近期跌倒严重 (通过率低于 60%)，适当下调或冻结难度，避免梯度崩溃
        s_rate = self.rolling_success_rate
        if len(self.episode_results) >= 15:
            if s_rate < 0.60:
                difficulty_scale = 0.80
            elif s_rate < self.min_pass_rate:
                difficulty_scale = 0.90
            else:
                difficulty_scale = 1.0
        else:
            difficulty_scale = 1.0

        self.difficulty = float(np.clip(base_diff * difficulty_scale, self.start_difficulty, self.max_difficulty))
        return self.difficulty

    def get_difficulty(self) -> float:
        """获取当前难度系数"""
        return self.difficulty

    def set_difficulty(self, diff: float):
        """手动强制覆盖难度系数 (用于测试)"""
        self.difficulty = float(np.clip(diff, 0.0, 1.0))
