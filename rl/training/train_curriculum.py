"""
train_curriculum.py: 轮腿机器人程序化课程强化学习训练系统 (Curriculum PRCC-RL / SCT-RRL)
========================================================================================
【模块功能介绍】
本模块是轮腿自适应顺应策略网络的主训练流水线，集成了课程学习调度、物理域随机化与
预训练模型热启动（Warm-start）功能：

1. 内存级程序化连续随机地形流 (Procedural Heightfield Streaming):
   每个 Episode 在内存中动态生成随机高度场，彻底打破固定静态赛道的时序轨迹过拟合，
   强迫策略网络学会依据实时本体感知信号进行自适应顺应。

2. 3 级渐进退火课程机制 (Curriculum Annealing):
   Novice (0.10~0.35) -> Adept (0.35~0.75) -> Master (0.75~1.00)，结合胜率门控自适应增阻。

3. 探索噪声平滑退火 (Exploration Std Annealing):
   策略输出的动作高斯噪声标准差随步数由 0.65 渐变至 0.35，后期消除高难地形下的多动抖颤。

4. 跨维度零填充热启动 (Zero-Padded Warm-Start):
   支持将 69 维传统策略（无储罐）的权重零填充迁移至 78 维 SCT 封闭能量槽策略网络中，
   无缝继承基础平衡直立能力，节省数十万步预热开销。
========================================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# 路径自适应引导
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
         os.path.join(rl_dir, "controllers"),
         os.path.join(rl_dir, "envs"),
         os.path.join(rl_dir, "terrain"),
         os.path.join(rl_dir, "training")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from wheel_leg_env import WheelLegRoughTerrainEnv
from curriculum_manager import CurriculumManager


class CurriculumCallback(BaseCallback):
    """
    监控并同步课程学习难度的 SB3 训练回调函数
    负责在每个环境步触发难度更新、探索噪声衰减与 TensorBoard 指标上报。
    """
    def __init__(self, curriculum_manager: CurriculumManager, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.cm = curriculum_manager
        self.log_freq = log_freq
        self.last_print_step = 0

    def _on_step(self) -> bool:
        # 1. 更新总步数并获取自适应调整后的难度
        diff = self.cm.update_steps(self.num_timesteps)

        # 2. 动作探索噪声自适应平滑退火: 0.65 -> 0.35 (消除后期高难度地形下的盲目手抖)
        progress = min(1.0, max(0.0, self.num_timesteps / float(self.cm.total_timesteps)))
        target_std = float(0.65 - (0.65 - 0.35) * progress)
        target_log_std = float(np.log(target_std))
        with torch.no_grad():
            self.model.policy.log_std.clamp_(max=target_log_std)

        # 3. 定期记录指标到 TensorBoard
        if self.num_timesteps % self.log_freq == 0:
            s_rate = self.cm.rolling_success_rate
            self.logger.record("curriculum/difficulty", diff)
            self.logger.record("curriculum/rolling_success_rate", s_rate)

            # 每 10000 步打印一次阶段性状态看板
            if self.num_timesteps - self.last_print_step >= 10000:
                cur_std = float(torch.exp(self.model.policy.log_std).mean().item())
                print(f"[Curriculum Progress] Step: {self.num_timesteps:>7d} | "
                      f"Difficulty: {diff:.3f} | "
                      f"Rolling Clear Rate: {s_rate * 100:>5.1f}% | "
                      f"Exploration Std: {cur_std:.3f}")
                self.last_print_step = self.num_timesteps

        return True


# 训练变体架构配置对照表
VARIANTS = {
    # name: (use_tank, slip_coupling, tank_obs, obs_history_len)
    'baseline':   (False, False, False, 3), # 69维无约束残差基线
    'tank':       (True,  False, False, 3), # 69维能量储罐单向做功约束
    'tank_slip':  (True,  True,  False, 3), # 69维滑移耦合双向能量储罐
    'full':       (True,  True,  True,  6), # 156维长时序储罐网络
    'sct_master': (True,  True,  True,  3), # 78维闭环 SCT-RRL 核心策略网络
}


def warm_start_weights(target_model: PPO, source_path: str):
    """
    跨特征维度零填充权重迁移函数 (Zero-Padded Transfer)
    将 69 维预训练策略权重平滑迁移至 78 维封闭能量槽网络，实现热启动。
    """
    print(f"\n[Warm Start] Loading base balance weights from: {source_path}")
    src = PPO.load(source_path, device=target_model.device)
    src_sd = src.policy.state_dict()
    tgt_sd = target_model.policy.state_dict()

    src_in = src.policy.observation_space.shape[0]
    tgt_in = target_model.policy.observation_space.shape[0]

    if src_in == tgt_in:
        target_model.policy.load_state_dict(src_sd)
        print(f"[Warm Start] Direct weight transfer succeeded ({src_in} -> {tgt_in} dims).")
        return

    if src_in == 69 and tgt_in == 78:
        print(f"[Warm Start] Performing Zero-Padded Observation Weight Transfer (69 -> 78 dims)...")
        for net_name in ['mlp_extractor.policy_net.0', 'mlp_extractor.value_net.0']:
            w_src = src_sd[f'{net_name}.weight']  # (64, 69)
            b_src = src_sd[f'{net_name}.bias']    # (64,)
            w_tgt = torch.zeros((64, 78), dtype=w_src.dtype, device=w_src.device)
            for t in range(3):
                # 拷贝 23 个本体感知物理特征
                w_tgt[:, t*26 : t*26 + 23] = w_src[:, t*23 : (t+1)*23]
                # 储罐特征 [t*26+23 : (t+1)*26] 初始保持 0.0 权重
            tgt_sd[f'{net_name}.weight'] = w_tgt
            tgt_sd[f'{net_name}.bias'] = b_src

        # 其余隐藏层与输出头直接拷贝
        for k in tgt_sd.keys():
            if not k.startswith('mlp_extractor.policy_net.0') and not k.startswith('mlp_extractor.value_net.0'):
                if k in src_sd and tgt_sd[k].shape == src_sd[k].shape:
                    tgt_sd[k] = src_sd[k]

        target_model.policy.load_state_dict(tgt_sd)
        with torch.no_grad():
            target_model.policy.log_std.fill_(float(np.log(0.65)))
        print("[Warm Start] Zero-Padded transfer complete! Base balance and posture fully preserved.")
    else:
        print(f"[Warm Start Warning] Incompatible obs shapes ({src_in} vs {tgt_in}), training from scratch.")


def make_env(rank, curriculum_manager, seed=0, variant='baseline'):
    """多进程/多实例向量化环境生成工厂函数"""
    use_tank, slip_c, tank_obs, hist = VARIANTS[variant]

    def _init():
        env = WheelLegRoughTerrainEnv(
            enable_random_terrain=True,
            curriculum_manager=curriculum_manager,
            domain_randomization=True,
            use_tank=use_tank, slip_coupling=slip_c,
            tank_obs=tank_obs, obs_history_len=hist,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Curriculum Procedural Residual RL Training")
    parser.add_argument("--timesteps", type=int, default=150000, help="总训练步数 (默认 15万步)")
    parser.add_argument("--num_envs", type=int, default=8, help="并行环境实例数 (DummyVecEnv)")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=128, help="PPO mini-batch size")
    default_save_dir = os.path.join(rl_dir, "models")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="模型存储目录")
    default_pre = os.path.join(rl_dir, "models", "curriculum_master_model.zip")
    if not os.path.exists(default_pre):
        default_pre = os.path.join(rl_dir, "models", "best_model.zip")
    parser.add_argument("--pretrained", type=str, default=default_pre, help="预训练模型路径 (Warm-start)")
    parser.add_argument("--no_warm_start", action="store_true", help="是否从头从零开始训练")
    parser.add_argument("--variant", type=str, default="sct_master", choices=list(VARIANTS.keys()),
                        help="SCT 变体；默认 sct_master (78维封闭能量槽回路)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.variant is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(rl_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(rl_dir, "checkpoints_history"), exist_ok=True)

    print("=" * 76)
    print("  CURRICULUM RESIDUAL RL: PROCEDURAL RANDOM TERRAIN")
    print(f"  Parallel Envs: {args.num_envs} | Total Timesteps: {args.timesteps}")
    print(f"  Variant: {args.variant} | Save Dir: {args.save_dir}")
    print("=" * 76)

    # 1. 实例化核心课程学习管理器 (支持动态退火与胜率门控)
    curriculum_mgr = CurriculumManager(
        total_timesteps=args.timesteps,
        start_difficulty=0.10,
        max_difficulty=1.0,
        win_rate_window=40,
        min_pass_rate=0.75
    )

    # 2. 创建共享课程状态的向量化并行环境
    print(f"Initializing {args.num_envs} vectorized environments with procedural terrain...")
    variant_name = args.variant if args.variant else 'sct_master'
    env = DummyVecEnv([make_env(i, curriculum_mgr, seed=args.seed, variant=variant_name) for i in range(args.num_envs)])

    # 3. 回调函数：课程调度与定期 Checkpoint
    curriculum_cb = CurriculumCallback(curriculum_mgr, log_freq=1000)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, 30000 // args.num_envs),
        save_path=os.path.join(rl_dir, "checkpoints_history"),
        name_prefix=f"ppo_{variant_name}"
    )

    # 4. 构建 PPO 策略网络
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=args.batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(rl_dir, "logs", "tb"),
        verbose=1
    )

    if not args.no_warm_start and os.path.exists(args.pretrained):
        warm_start_weights(model, args.pretrained)

    # 5. 执行课程训练
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[curriculum_cb, checkpoint_cb]
        )
    except KeyboardInterrupt:
        print("\n[Training Interrupted by User]")

    elapsed = time.time() - start_time

    # 保存训练权重
    if args.variant is not None:
        saved_model_path = os.path.join(args.save_dir, f"sct_{args.variant}_s{args.seed}.zip")
        model.save(saved_model_path)
        model.save(os.path.join(args.save_dir, "sct_master.zip"))
        model.save(os.path.join(args.save_dir, "sct_full_master.zip"))
    else:
        saved_model_path = os.path.join(args.save_dir, "curriculum_master_model.zip")
        model.save(saved_model_path)
        model.save(os.path.join(args.save_dir, "best_model.zip"))

    print("\n" + "=" * 76)
    print(f"  TRAINING COMPLETED SUCCESSFULLY in {elapsed:.2f}s ({elapsed / 60.0:.2f} min)")
    print(f"  Model Saved to: {saved_model_path}")
    print(f"  Final Curriculum Difficulty: {curriculum_mgr.get_difficulty():.3f}")
    print(f"  Final Rolling Clear Rate: {curriculum_mgr.rolling_success_rate * 100:.1f}%")
    print("=" * 76)

    env.close()


if __name__ == "__main__":
    main()
