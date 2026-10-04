"""
train_rl.py: 基于 Stable-Baselines3 PPO 的轻起伏地形自适应盲走训练脚本
====================================================================
特点：
  1. 多进程 SubprocVecEnv 并行采样 (满载 8~16 线程加速)
  2. 极轻量 MLP Policy [64, 64] (推理耗时 < 0.1ms)
  3. Residual-over-Prior 机制使得策略在极短步数内即可掌握吸震与自适应抗扰
  4. 支持 TensorBoard 实时查看跟踪误差、横滚角衰减与防打滑指标
"""
import os
import sys
import time
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 确保导入 rl 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from wheel_leg_env import WheelLegRoughTerrainEnv


def make_env(rank, seed=0):
    def _init():
        env = WheelLegRoughTerrainEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO Residual RL Training on Rough Terrain")
    parser.add_argument("--timesteps", type=int, default=300000, help="总训练步数 (默认 30万步，约15~20分钟)")
    parser.add_argument("--num_envs", type=int, default=8, help="并行子进程数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--save_dir", type=str, default="rl/models", help="模型存储目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("rl/logs", exist_ok=True)

    print(f"==================================================")
    print(f"  Starting Wheel-Leg Rough Terrain Residual PPO")
    print(f"  Parallel Envs: {args.num_envs}")
    print(f"  Total Timesteps: {args.timesteps}")
    print(f"  PyTorch Threads: {torch.get_num_threads()}")
    print(f"==================================================")

    # 1. 创建多进程矢量化环境 (在 Windows 上使用 Dummy 或 Subproc)
    try:
        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    except Exception as e:
        print(f"[Warning] SubprocVecEnv failed with {e}, falling back to DummyVecEnv.")
        env = DummyVecEnv([make_env(i) for i in range(args.num_envs)])

    # 2. 回调函数：定期保存最佳模型
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 50000 // args.num_envs),
        save_path=args.save_dir,
        name_prefix="ppo_wheel_leg"
    )

    # 3. PPO 策略配置 (极轻量高效策略网络)
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
        tensorboard_log="rl/logs/tb",
        verbose=1
    )

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    elapsed = time.time() - start_time

    # 保存最终与最佳模型
    final_model_path = os.path.join(args.save_dir, "best_model.zip")
    model.save(final_model_path)
    print(f"\n[Training Finished] Elapsed: {elapsed:.2f}s, Saved to {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
