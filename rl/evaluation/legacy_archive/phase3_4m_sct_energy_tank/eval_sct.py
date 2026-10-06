"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/eval_sct.py`: 滑移耦合能量储罐 (SCT-RRL) 核心物理机制与学术基准全景评测套件
====================================================================================================
1. 核心定位与评估矩阵：
   - 评测搭载无源能量储罐与滑移耗散耦合的 SCT-RRL 控制器相对经典 LQR 与无约束 PRCC-RL 的
     系统级性能提升。
   - 评估矩阵包括：
     * [Part 1] 标准固定地形 (1.8m ~ 3.3m 减速垄核心区) 零误差对比。
     * [Part 2] 50 轮未见高难度随机起伏地形 (Difficulty = 0.85) 蒙特卡洛统计显著性泛化大考。
     * [Part 3] 能量储罐物理微观动力学演化过程观测 (E_T 充放电, \alpha^* 阻抗缩放系数, P_{slip} 耗散)。

2. 输出成果物：
   - 四联学术时域对比图: `rl/evaluation/sct_benchmark.png`
   - 储罐微观动态图: `rl/evaluation/sct_tank_dynamics.png`
   - 量化评测报告: `rl/evaluation/sct_metrics.md`
====================================================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
from energy_tank import SlipCoupledEnergyTank
from metrics_utils import high_freq_energy


def compute_metrics(h: dict, x0: float = 1.8, x1: float = 3.3, dt: float = 0.001) -> dict:
    """计算指定区间的遥测指标"""
    m = (h['x'] >= x0) & (h['x'] <= x1)
    if m.sum() < 10:
        m = (h['x'] >= 0.3)
    if m.sum() < 5:
        m = np.ones_like(h['x'], dtype=bool)
    if len(m) == 0 or m.sum() == 0:
        return {
            'peak_roll': 0.0, 'roll_std': 0.0, 'z_std_mm': 0.0,
            'speed_mm_s': 0.0, 'slip_kin_mm_s': 0.0, 'slip_true_mm_s': 0.0,
            'slip_ratio_pct': 0.0, 'hip_work_mJ': 0.0, 'hip_hf_uJ': 0.0,
            'throttle_rate_pct': 0.0
        }
    vx = np.mean(np.abs(h['vx'][m]))
    slip_t = np.mean(h['slip_true'][m])
    slip_k = np.mean(h['slip_kin'][m])

    clip_rate = 0.0
    if len(h['alpha']) > 0 and not np.all(np.isnan(h['alpha'])):
        clip_rate = float(np.mean(h['alpha'][m] < 0.99) * 100)

    return {
        'peak_roll': float(np.max(np.abs(h['roll_deg'][m]))),
        'roll_std': float(np.std(h['roll_deg'][m])),
        'z_std_mm': float(np.std(h['z'][m]) * 1000),
        'speed_mm_s': float(vx * 1000),
        'slip_kin_mm_s': float(slip_k * 1000),
        'slip_true_mm_s': float(slip_t * 1000),
        'slip_ratio_pct': float(slip_t / max(vx, 1e-6) * 100),
        'hip_work_mJ': float(np.sum(h['p_hip_mech'][m]) * dt * 1000),
        'hip_hf_uJ': float((high_freq_energy(h['tau_l_hip'][m], dt) +
                            high_freq_energy(h['tau_r_hip'][m], dt)) * 1e6),
        'throttle_rate_pct': clip_rate
    }


def main():
    parser = argparse.ArgumentParser(description="SCT-RRL Benchmark Evaluation")
    parser.add_argument("--sct_model", type=str, default=os.path.join(rl_dir, "models", "sct_master.zip"))
    parser.add_argument("--curric_model", type=str, default=os.path.join(rl_dir, "models", "curriculum_master_model.zip"))
    parser.add_argument("--mc_trials", type=int, default=50)
    args = parser.parse_args()

    # 候选查找
    if not os.path.exists(args.sct_model):
        for candidate in [
            os.path.join(rl_dir, "models", "sct_master_s0.zip"),
            os.path.join(rl_dir, "models", "sct_full_master.zip"),
            os.path.join(rl_dir, "models", "sct_full_s0.zip")
        ]:
            if os.path.exists(candidate):
                args.sct_model = candidate
                break

    print("=" * 76)
    print("       SCT-RRL COMPREHENSIVE BENCHMARK EVALUATION")
    print(f"  SCT Model: {args.sct_model}")
    print(f"  Curriculum Baseline: {args.curric_model}")
    print("=" * 76)

    pol_curric = PPO.load(args.curric_model)
    pol_sct = PPO.load(args.sct_model)

    obs_dim = pol_sct.observation_space.shape[0]
    if obs_dim % 26 == 0:
        hist_len_sct = obs_dim // 26
        tank_obs_sct = True
    elif obs_dim % 23 == 0:
        hist_len_sct = obs_dim // 23
        tank_obs_sct = False
    else:
        hist_len_sct = 3
        tank_obs_sct = False
    print(f"  -> Auto-detected SCT policy observation dimension: {obs_dim} (hist_len={hist_len_sct}, tank_obs={tank_obs_sct})")

    tank_eval = SlipCoupledEnergyTank(slip_coupling=True)

    # 1. 标准固定赛道评测
    print("\n[1/3] Running Fixed Standard Benchmark Course...")
    h_lqr = run_episode(mode="lqr", difficulty=0.0)
    h_curric = run_episode(mode="rl", policy=pol_curric, difficulty=0.0, obs_history_len=3, tank_obs=False)
    h_sct = run_episode(mode="rl", policy=pol_sct, difficulty=0.0, tank=tank_eval, obs_history_len=hist_len_sct, tank_obs=tank_obs_sct)

    m_lqr = compute_metrics(h_lqr)
    m_curric = compute_metrics(h_curric)
    m_sct = compute_metrics(h_sct)

    print("\n--- Standard Course Metrics (1.8m ~ 3.3m) ---")
    print(f"LQR:        Roll Peak = {m_lqr['peak_roll']:.2f}°, Roll Std = {m_lqr['roll_std']:.2f}°, Slip = {m_lqr['slip_true_mm_s']:.2f}mm/s, HF = {m_lqr['hip_hf_uJ']:.2f}uJ")
    print(f"PRCC-RL:    Roll Peak = {m_curric['peak_roll']:.2f}°, Roll Std = {m_curric['roll_std']:.2f}°, Slip = {m_curric['slip_true_mm_s']:.2f}mm/s, HF = {m_curric['hip_hf_uJ']:.2f}uJ")
    print(f"SCT-RRL:    Roll Peak = {m_sct['peak_roll']:.2f}°, Roll Std = {m_sct['roll_std']:.2f}°, Slip = {m_sct['slip_true_mm_s']:.2f}mm/s, HF = {m_sct['hip_hf_uJ']:.2f}uJ, Throttle = {m_sct['throttle_rate_pct']:.1f}%")

    # 2. 蒙特卡洛泛化性评测
    print(f"\n[2/3] Running {args.mc_trials} Monte Carlo Trials on Random Terrain (Diff=0.85)...")
    mc_results = {'lqr': [], 'curric': [], 'sct': []}
    mc_succ = {'lqr': 0, 'curric': 0, 'sct': 0}

    for trial in range(args.mc_trials):
        seed = 42000 + trial * 17
        hl = run_episode(mode="lqr", difficulty=0.85, seed=seed)
        hc = run_episode(mode="rl", policy=pol_curric, difficulty=0.85, seed=seed, obs_history_len=3, tank_obs=False)
        hs = run_episode(mode="rl", policy=pol_sct, difficulty=0.85, seed=seed, tank=tank_eval, obs_history_len=hist_len_sct, tank_obs=tank_obs_sct)

        mc_succ['lqr'] += int(hl['success'])
        mc_succ['curric'] += int(hc['success'])
        mc_succ['sct'] += int(hs['success'])

        mc_results['lqr'].append(compute_metrics(hl, x0=0.3, x1=3.4))
        mc_results['curric'].append(compute_metrics(hc, x0=0.3, x1=3.4))
        mc_results['sct'].append(compute_metrics(hs, x0=0.3, x1=3.4))

        if (trial + 1) % 10 == 0:
            print(f"  Progress: {trial+1}/{args.mc_trials} trials complete.")

    print("\n--- Monte Carlo Pass Rates ---")
    print(f"LQR:      {mc_succ['lqr']}/{args.mc_trials} ({mc_succ['lqr']/args.mc_trials*100:.1f}%)")
    print(f"PRCC-RL:  {mc_succ['curric']}/{args.mc_trials} ({mc_succ['curric']/args.mc_trials*100:.1f}%)")
    print(f"SCT-RRL:  {mc_succ['sct']}/{args.mc_trials} ({mc_succ['sct']/args.mc_trials*100:.1f}%)")

    # 3. 绘制全景对比时域图 (sct_benchmark.png)
    print("\n[3/3] Generating Benchmark Figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    m_l = (h_lqr['x'] >= 0.2) & (h_lqr['x'] <= 3.5)
    m_c = (h_curric['x'] >= 0.2) & (h_curric['x'] <= 3.5)
    m_s = (h_sct['x'] >= 0.2) & (h_sct['x'] <= 3.5)

    # (1) Roll
    axs[0].axvspan(1.8, 3.3, color='gold', alpha=0.15, label='Speed Bumps Zone')
    axs[0].plot(h_lqr['x'][m_l], h_lqr['roll_deg'][m_l], 'r--', label='Pure LQR', alpha=0.7, linewidth=1.2)
    axs[0].plot(h_curric['x'][m_c], h_curric['roll_deg'][m_c], 'orange', label='PRCC-RL (Unconstrained)', alpha=0.8, linewidth=1.5)
    axs[0].plot(h_sct['x'][m_s], h_sct['roll_deg'][m_s], 'g-', label='SCT-RRL (Ours)', linewidth=2.0)
    axs[0].set_ylabel('Body Roll Angle (deg)')
    axs[0].set_ylim(-5.5, 5.5)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right', framealpha=0.85)

    # (2) Slip
    axs[1].axvspan(1.8, 3.3, color='gold', alpha=0.15)
    axs[1].plot(h_lqr['x'][m_l], h_lqr['slip_true'][m_l] * 1000, 'r--', label='Pure LQR', alpha=0.7, linewidth=1.2)
    axs[1].plot(h_curric['x'][m_c], h_curric['slip_true'][m_c] * 1000, 'orange', label='PRCC-RL', alpha=0.8, linewidth=1.5)
    axs[1].plot(h_sct['x'][m_s], h_sct['slip_true'][m_s] * 1000, 'g-', label='SCT-RRL (Ours)', linewidth=2.0)
    axs[1].set_ylabel('Contact Slip (mm/s)')
    axs[1].set_ylim(0.0, 75.0)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right', framealpha=0.85)

    # (3) Hip Torque
    axs[2].axvspan(1.8, 3.3, color='gold', alpha=0.15)
    axs[2].plot(h_lqr['x'][m_l], h_lqr['tau_l_hip'][m_l], 'r--', label='Pure LQR', alpha=0.7, linewidth=1.2)
    axs[2].plot(h_curric['x'][m_c], h_curric['tau_l_hip'][m_c], 'orange', label='PRCC-RL', alpha=0.8, linewidth=1.5)
    axs[2].plot(h_sct['x'][m_s], h_sct['tau_l_hip'][m_s], 'g-', label='SCT-RRL (Ours)', linewidth=1.8)
    axs[2].set_ylabel('Left Hip Torque (N*m)')
    axs[2].set_ylim(-0.25, 0.25)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper right', framealpha=0.85)

    # (4) Passivity / Energy Tank
    axs[3].axvspan(1.8, 3.3, color='gold', alpha=0.15)
    l1 = axs[3].plot(h_sct['x'][m_s], h_sct['alpha'][m_s], 'b-', label='Passivity Ratio alpha*', linewidth=2.0)
    axs[3].set_ylabel('Passivity Ratio alpha*', color='b')
    axs[3].tick_params(axis='y', labelcolor='b')
    axs[3].set_ylim(0.50, 1.05)
    axs[3].set_xlabel('Forward Travel Distance x (m)')

    ax3_twin = axs[3].twinx()
    l2 = ax3_twin.plot(h_sct['x'][m_s], h_sct['E_T'][m_s] * 1e4, 'k--', label='Tank Energy E_T (x1e-4 J)', linewidth=1.6, alpha=0.8)
    ax3_twin.set_ylabel('Tank Energy E_T (10^-4 J)', color='black')
    ax3_twin.tick_params(axis='y', labelcolor='black')
    ax3_twin.set_ylim(0.0, 9.5)

    axs[3].grid(True, linestyle=':', alpha=0.6)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axs[3].legend(lines, labels, loc='upper left', framealpha=0.85, ncol=2, fontsize=9)
    axs[3].margins(y=0.15)

    plt.tight_layout()
    plot_path = os.path.join(current_dir, "sct_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    print(f"  -> Saved {plot_path}")

    # 4. 生成储罐物理微观动力学图 (sct_tank_dynamics.png)
    fig_dyn, (ax_d1, ax_d2, ax_d3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax_d1.axvspan(1.8, 3.3, color='gold', alpha=0.15, label='Bumps Zone (1.8~3.3m)')
    ax_d1.plot(h_sct['x'][m_s], h_sct['E_T'][m_s] * 1e4, 'b-', label='Tank Energy E_T (x1e-4 J)', linewidth=2.0)
    ax_d1.set_ylabel('Tank Energy (10^-4 J)')
    ax_d1.set_ylim(0.0, 9.5)
    ax_d1.grid(True, linestyle=':', alpha=0.6)
    ax_d1.legend(loc='lower left', framealpha=0.85)

    ax_d2.axvspan(1.8, 3.3, color='gold', alpha=0.15)
    ax_d2.plot(h_sct['x'][m_s], h_sct['alpha'][m_s], 'r-', label='Passivity Pass Ratio alpha*', linewidth=2.0)
    ax_d2.set_ylabel('Alpha* (1.0=Free, <1.0=Throttled)')
    ax_d2.set_ylim(0.50, 1.05)
    ax_d2.grid(True, linestyle=':', alpha=0.6)
    ax_d2.legend(loc='lower left', framealpha=0.85)

    ax_d3.axvspan(1.8, 3.3, color='gold', alpha=0.15)
    ax_d3.plot(h_sct['x'][m_s], h_sct['P_slip'][m_s] * 1000, 'darkorange', label='Filtered Slip Dissipation (mW)', linewidth=2.0)
    ax_d3.set_xlabel('Forward Travel Distance x (m)')
    ax_d3.set_ylabel('Slip Power (mW)')
    ax_d3.set_ylim(0.0, 35.0)
    ax_d3.grid(True, linestyle=':', alpha=0.6)
    ax_d3.legend(loc='upper left', framealpha=0.85)

    plt.tight_layout()
    dyn_path = os.path.join(current_dir, "sct_tank_dynamics.png")
    plt.savefig(dyn_path, dpi=300)
    print(f"  -> Saved {dyn_path}")

    # 5. 生成量化指标报告 (rl/evaluation/sct_metrics.md)
    md_path = os.path.join(current_dir, "sct_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 滑移耦合能量储罐残差强化学习 (SCT-RRL) 基准评测报告\n\n")
        f.write("## 一、 标准固定赛道 (1.8m ~ 3.3m 减速垄核心区) 对照\n\n")
        f.write("| 性能指标 | Pure LQR Baseline | Standard PRCC-RL | **SCT-RRL (Ours)** | 相对 PRCC 变化 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Peak Roll (峰值侧倾)** | {m_lqr['peak_roll']:.2f}° | {m_curric['peak_roll']:.2f}° | **{m_sct['peak_roll']:.2f}°** | **{(m_sct['peak_roll'] - m_curric['peak_roll'])/m_curric['peak_roll']*100:+.1f}%** |\n")
        f.write(f"| **Roll Std (姿态晃动抖动)** | {m_lqr['roll_std']:.2f}° | {m_curric['roll_std']:.2f}° | **{m_sct['roll_std']:.2f}°** | **{(m_sct['roll_std'] - m_curric['roll_std'])/m_curric['roll_std']*100:+.1f}%** |\n")
        f.write(f"| **Height Std (垂直颠簸)** | {m_lqr['z_std_mm']:.2f}mm | {m_curric['z_std_mm']:.2f}mm | **{m_sct['z_std_mm']:.2f}mm** | **{(m_sct['z_std_mm'] - m_curric['z_std_mm'])/m_curric['z_std_mm']*100:+.1f}%** |\n")
        f.write(f"| **True Contact Slip (真实接触滑移)** | {m_lqr['slip_true_mm_s']:.2f}mm/s | {m_curric['slip_true_mm_s']:.2f}mm/s | **{m_sct['slip_true_mm_s']:.2f}mm/s** | **{(m_sct['slip_true_mm_s'] - m_curric['slip_true_mm_s'])/m_curric['slip_true_mm_s']*100:+.1f}%** |\n")
        f.write(f"| **Slip Ratio (滑移率)** | {m_lqr['slip_ratio_pct']:.2f}% | {m_curric['slip_ratio_pct']:.2f}% | **{m_sct['slip_ratio_pct']:.2f}%** | **{(m_sct['slip_ratio_pct'] - m_curric['slip_ratio_pct'])/m_curric['slip_ratio_pct']*100:+.1f}%** |\n")
        f.write(f"| **Hip Torque >10Hz Energy (高频力矩抖动)** | {m_lqr['hip_hf_uJ']:.2f}μJ | {m_curric['hip_hf_uJ']:.2f}μJ | **{m_sct['hip_hf_uJ']:.2f}μJ** | **{(m_sct['hip_hf_uJ'] - m_curric['hip_hf_uJ'])/m_curric['hip_hf_uJ']*100:+.1f}%** |\n")
        f.write(f"| **Passivity Modulation Active Rate (被动阻抗调节介入率)** | - | - | **{m_sct['throttle_rate_pct']:.2f}%** | - |\n\n")

        f.write(f"## 二、 {args.mc_trials} 轮未见复杂随机地形蒙特卡洛泛化性大考 (Difficulty=0.85)\n\n")
        f.write("| 泛化性指标 | Pure LQR | Standard PRCC-RL | **SCT-RRL (Ours)** |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **完赛成功率 (Pass Rate)** | {mc_succ['lqr']/args.mc_trials*100:.1f}% | {mc_succ['curric']/args.mc_trials*100:.1f}% | **{mc_succ['sct']/args.mc_trials*100:.1f}%** |\n")
        
        for k, label in [('peak_roll', '平均峰值侧倾 (Peak Roll, °)'),
                         ('roll_std', '平均姿态抖动 (Roll Std, °)'),
                         ('z_std_mm', '平均垂直颠簸 (Height Std, mm)'),
                         ('slip_true_mm_s', '平均真实接触滑移 (Slip, mm/s)'),
                         ('slip_ratio_pct', '平均滑移率 (Slip Ratio, %)'),
                         ('hip_hf_uJ', '高频力矩抖动能量 (Torque HF, μJ)')]:
            v_l = np.array([x[k] for x in mc_results['lqr']])
            v_c = np.array([x[k] for x in mc_results['curric']])
            v_s = np.array([x[k] for x in mc_results['sct']])
            f.write(f"| **{label}** | {v_l.mean():.2f} ± {v_l.std():.2f} | {v_c.mean():.2f} ± {v_c.std():.2f} | **{v_s.mean():.2f} ± {v_s.std():.2f}** |\n")

    print(f"  -> Saved {md_path}")
    print("\n" + "=" * 76)
    print("       SCT-RRL BENCHMARK EVALUATION COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
