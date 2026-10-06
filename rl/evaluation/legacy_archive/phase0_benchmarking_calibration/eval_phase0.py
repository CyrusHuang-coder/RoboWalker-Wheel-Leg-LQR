"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/eval_phase0.py`: 修正统计口径后的四算法公平对比基准评测
====================================================================================================
1. 核心定位与修正原则：
   - 解决传统测试中“算法速度不一致导致冲量不同”、“运动学近似滑移受腿摆动污染”等口径偏差问题。
   - 在严格统一的工况下横向评测 4 种控制器：
     * LQR (纯经典刚性倒立摆)
     * LQR + hand compliance (手工工程参数顺应律，用于消融实验)
     * Fixed-Terrain RL (固定单一地形过拟合残差模型)
     * Curriculum RL (多阶段课程学习泛化模型)

2. 评测输出：
   - 输出量化 Markdown 评测报告至 `rl/evaluation/phase0_metrics.md`。
   - 包含标准基准赛道指标与 20 轮未见随机地形蒙特卡洛测试 (难度 0.85)。
====================================================================================================
"""

import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO

# --------------------------------------------------------------------------------------------------
# 鲁棒路径引导
# --------------------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
          os.path.join(rl_dir, "controllers"),
          os.path.join(rl_dir, "envs"),
          os.path.join(rl_dir, "terrain")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim_runner import run_episode
from metrics_utils import high_freq_energy


def zone_metrics(h: dict, x0: float = 1.8, x1: float = 3.3, dt: float = 0.001) -> dict:
    """计算指定 X 区间内的核心稳态与冲击指标"""
    m = (h['x'] >= x0) & (h['x'] <= x1)
    if m.sum() < 10:
        return None
    vx = np.mean(np.abs(h['vx'][m]))
    slip_t = np.mean(h['slip_true'][m])
    return {
        'peak_roll': float(np.max(np.abs(h['roll_deg'][m]))),
        'roll_std': float(np.std(h['roll_deg'][m])),
        'z_std_mm': float(np.std(h['z'][m]) * 1000),
        'speed_mm_s': float(vx * 1000),
        'slip_kin_mm_s': float(np.mean(h['slip_kin'][m]) * 1000),
        'slip_true_mm_s': float(slip_t * 1000),
        'slip_ratio_pct': float(slip_t / max(vx, 1e-6) * 100),
        'hip_work_mJ': float(np.sum(h['p_hip_mech'][m]) * dt * 1000),
        'hip_hf_uJ': float((high_freq_energy(h['tau_l_hip'][m], dt) +
                            high_freq_energy(h['tau_r_hip'][m], dt)) * 1e6),
    }


COLS = [
    ('peak_roll', 'Peak Roll (deg)'),
    ('roll_std', 'Roll Std (deg)'),
    ('z_std_mm', 'Height Std (mm)'),
    ('speed_mm_s', 'Mean Speed (mm/s)'),
    ('slip_kin_mm_s', 'Slip, kinematic approx (mm/s)'),
    ('slip_true_mm_s', 'Slip, contact-point (mm/s)'),
    ('slip_ratio_pct', 'Slip Ratio (%)'),
    ('hip_work_mJ', 'Hip Mech. Work (mJ)'),
    ('hip_hf_uJ', 'Hip Torque >10Hz Energy (uJ)')
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prev_model', default=os.path.join(rl_dir, 'models', 'baseline_fixed_model.zip'))
    ap.add_argument('--new_model', default=os.path.join(rl_dir, 'models', 'curriculum_master_model.zip'))
    ap.add_argument('--mc_trials', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(current_dir, 'phase0_metrics.md'))
    args = ap.parse_args()

    ctrls = [('LQR', 'lqr', None), ('LQR + hand compliance', 'lqr_comp', None)]
    if os.path.exists(args.prev_model):
        ctrls.append(('Fixed-Terrain RL', 'rl', PPO.load(args.prev_model)))
    ctrls.append(('Curriculum RL', 'rl', PPO.load(args.new_model)))

    lines = ['# Phase 0: 修正口径后的公平基线\n',
             '同一目标速度 0.16 m/s；指标统计区间 x ∈ [1.8, 3.3] m。\n',
             '## A. 标准赛道\n']
    std = {}
    for name, mode, pol in ctrls:
        h = run_episode(mode=mode, policy=pol, difficulty=0.0)
        std[name] = (zone_metrics(h), h['success'], h['time'][-1] if len(h['time']) else 0)
        print(f'[std] {name}: success={h["success"]} t={std[name][2]:.2f}s')

    names = [c[0] for c in ctrls]
    lines.append('| 指标 | ' + ' | '.join(names) + ' |')
    lines.append('| :--- |' + ' :---: |' * len(names))
    lines.append('| 完赛 / 用时 (s) | ' + ' | '.join(f'{"✓" if std[n][1] else "✗"} / {std[n][2]:.2f}' for n in names) + ' |')
    for k, label in COLS:
        lines.append(f'| {label} | ' + ' | '.join(f'{std[n][0][k]:.3f}' if std[n][0] else '-' for n in names) + ' |')

    lines.append(f'\n## B. {args.mc_trials} 条未见随机地形 (difficulty=0.85, 均值 ± 标准差)\n')
    mc = {n: [] for n in names}
    succ = {n: 0 for n in names}
    for t in range(args.mc_trials):
        seed = 10000 + t
        for name, mode, pol in ctrls:
            h = run_episode(mode=mode, policy=pol, difficulty=0.85, seed=seed)
            succ[name] += int(h['success'])
            zm = zone_metrics(h, 0.3, 3.4)
            if zm:
                mc[name].append(zm)
        print(f'[mc] trial {t + 1}/{args.mc_trials}')
    lines.append('| 指标 | ' + ' | '.join(names) + ' |')
    lines.append('| :--- |' + ' :---: |' * len(names))
    lines.append('| 完赛率 | ' + ' | '.join(f'{succ[n] / args.mc_trials * 100:.0f}%' for n in names) + ' |')
    for k, label in COLS:
        cells = []
        for n in names:
            v = np.array([d[k] for d in mc[n]])
            cells.append(f'{v.mean():.3f} ± {v.std():.3f}' if len(v) else '-')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
