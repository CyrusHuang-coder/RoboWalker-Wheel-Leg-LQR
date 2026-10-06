"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/training/train_rl.py`: 基于 Stable-Baselines3 PPO 的单阶段基准残差强化学习训练流水线
====================================================================================================
1. 核心定位与训练范式：
   - 本模块实现了基于 PPO (Proximal Policy Optimization) 的经典非课程单阶段残差学习。
   - 策略网络 (Actor-Critic MLP) 输出叠加在经典先验控制器 (LQR 倒立摆基准) 基础上的残差动作：
     \tau_act = \tau_{prior} + \Delta \tau_{RL}
   - 通过极轻量级网络架构 ([64, 64] 隐藏层 + Tanh 激活)，单步前向推理延迟 < 0.1ms，保证实机与高频
     仿真控制回路的极速实时性要求。

2. 关键训练特性与稳定性设计：
   - 矢量化环境并行：默认在 Windows 环境下采用 DummyVecEnv 进行多环境串行批处理采样，
     免除多进程跨进程通信 (IPC) 死锁风险，在满足稳定性的同时大幅提高样本收集吞吐量。
   - 周期性 Checkpoint：定期保存模型权重到 `rl/models/`，记录 TensorBoard 训练曲线。
   - 残差引导快速收敛：相较于从零训练的纯端到端 RL，先验引导机制大幅降低探索空间，在
     10~30 万步以内即可使轮腿机器人学会基础的越障吸震与速度调整策略。
====================================================================================================
"""

import os
import sys
import time
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# --------------------------------------------------------------------------------------------------
# 鲁棒模块路径检索与自适应加载 (Bootstrap Paths)
# --------------------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
          os.path.join(rl_dir, "controllers"),
          os.path.join(rl_dir, "envs"),
          os.path.join(rl_dir, "terrain"),
          os.path.join(rl_dir, "evaluation"),
          os.path.join(rl_dir, "training"),
          os.path.join(rl_dir, "recorders")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from wheel_leg_env import WheelLegRoughTerrainEnv


def make_env(rank: int, seed: int = 0):
    """
    创建单个带独立随机种子的轮腿环境闭包。

    Args:
        rank (int): 环境线程序号。
        seed (int): 基础随机种子。
    """
    def _init():
        env = WheelLegRoughTerrainEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO Residual RL Training on Rough Terrain")
    parser.add_argument("--timesteps", type=int, default=300000, help="总训练步数 (默认 30万步，约15~20分钟)")
    parser.add_argument("--num_envs", type=int, default=8, help="并行环境数量 (DummyVecEnv)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--save_dir", type=str, default=os.path.join(rl_dir, "models"), help="模型存储目录")
    parser.add_argument("--log_dir", type=str, default=os.path.join(rl_dir, "logs", "tb"), help="TensorBoard 日志目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("  Starting Wheel-Leg Rough Terrain Baseline Residual PPO")
    print(f"  Parallel Envs   : {args.num_envs}")
    print(f"  Total Timesteps : {args.timesteps}")
    print(f"  Learning Rate   : {args.lr}")
    print(f"  PyTorch Threads : {torch.get_num_threads()}")
    print(f"  Model Save Dir  : {args.save_dir}")
    print(f"  Log Dir         : {args.log_dir}")
    print("=" * 60)

    # 1. 创建矢量化环境
    print(f"Creating {args.num_envs} vectorized environments...")
    env = DummyVecEnv([make_env(i) for i in range(args.num_envs)])

    # 2. 定期模型保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 50000 // args.num_envs),
        save_path=args.save_dir,
        name_prefix="ppo_wheel_leg"
    )

    # 3. PPO 策略网络架构 (极轻量双层 MLP)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=512,            # 每次更新每个 env 采集 512 步 (512 * 8 = 4096 步/迭代)
        batch_size=args.batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,         # 轻微探索熵
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        verbose=1
    )

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    elapsed = time.time() - start_time

    # 保存最终基准模型
    final_model_path = os.path.join(args.save_dir, "baseline_fixed_model.zip")
    model.save(final_model_path)
    print(f"\n[Training Finished] Elapsed: {elapsed:.2f}s, Saved to {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
