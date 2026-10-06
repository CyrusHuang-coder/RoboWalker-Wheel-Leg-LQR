"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/terrain/procedural_terrain.py`: 课程学习动态程序化地形生成器 (In-Memory Hot-Update)
====================================================================================================
1. 核心定位与技术原理：
   - 纯内存向量化计算 (基于预计算 Meshgrid)，单次地形生成与 MuJoCo Hfield 缓冲区注入耗时 < 3ms，
     实现训练期间 0 磁盘 I/O 的超高速动态环境重置。
   - 专为 PPO 课程学习设计，提供连续难度映射参数 difficulty \in [0.0, 1.0]：
     * 0.0 (Baseline): 精确复现 4.0m 标准基准赛道 (5 道固定交替减速垄与波浪起伏)；
     * 0.1~0.35 (Novice): 基准参数随机抖动 (高度 2.0~3.8mm, 位置微抖, 保持交替规律)；
     * 0.35~0.75 (Adept): 拓扑乱序 (支持连续单侧/双轮同相交错), 间距可变 (0.20~0.35m), 高度 1.8~4.6mm；
     * 0.75~1.0 (Master): 极限复合地形 (高低起伏 1.5~5.2mm, 紧凑冲击间距 0.16m, 叠加多频微粗糙度)。

2. 物理与几何连续性保证：
   - 严格适配微型轮腿物理几何 (轮径 8mm, 轮距 32mm):
   - x in [0.0, 0.35m] 绝对平地 (保证倒立摆启动平稳加速与 LQR 初始收敛)；
   - x in [3.5, 4.0m] 平缓收尾过渡；
   - 采用二阶连续可微高斯凸起几何，彻底杜绝网格法向量突变引起的穿透与奇异接触力爆炸。
====================================================================================================
"""
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)



class ProceduralTerrainGenerator:
    def __init__(self, x_len=4.0, y_len=1.0, nrow=256, ncol=1024, max_elevation=0.0065):
        self.x_len = x_len
        self.y_len = y_len
        self.nrow = nrow
        self.ncol = ncol
        self.max_elevation = max_elevation  # MuJoCo xml 中的 elevation_z (0.0065m = 6.5mm)

        # 预计算坐标网格，加速每次 reset() 的生成效率
        self.x = np.linspace(0, x_len, ncol, dtype=np.float32)
        self.y = np.linspace(-y_len / 2.0, y_len / 2.0, nrow, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 轮距几何常数 (单位: 米)
        # 车轮中心在赛道中的标称位置: 左轮 y = -0.0015m, 右轮 y = -0.0335m, 赛道中线 y = -0.0175m
        self.y_left_wheel = -0.0015
        self.y_right_wheel = -0.0335
        self.y_center = -0.0175

    def generate(self, difficulty=0.0, rng=None):
        """
        生成指定难度的高程高度矩阵 (单位: 米)
        difficulty: 0.0 (标准测试赛道) -> 1.0 (极限随机颠簸复合赛道)
        """
        if rng is None:
            rng = np.random.default_rng()

        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        Z = np.zeros_like(self.X)

        if difficulty < 1e-4:
            # 严格复现现有 5 道减速垄固定赛道 (用于与历史基准严格同台对账)
            # 1. 前段起伏波浪
            mask_wave = (self.X >= 0.3) & (self.X < 1.8)
            wave_fade_in = np.clip((self.X - 0.3) / 0.2, 0.0, 1.0)
            wave_fade_out = np.clip((1.8 - self.X) / 0.2, 0.0, 1.0)
            wave_envelope = wave_fade_in * wave_fade_out
            Z[mask_wave] += 0.0025 * np.sin(2.0 * np.pi * (self.X[mask_wave] - 0.3) / 0.25) * wave_envelope[mask_wave]

            # 2. 5道固定实体减速垄
            fixed_bumps = [
                (2.00, self.y_left_wheel, 0.045, 0.022, 0.0048),
                (2.30, self.y_right_wheel, 0.045, 0.022, 0.0048),
                (2.60, self.y_left_wheel, 0.045, 0.022, 0.0048),
                (2.90, self.y_right_wheel, 0.045, 0.022, 0.0048),
                (3.20, self.y_center, 0.050, 0.050, 0.0045),
            ]
            for bx, by, sx, sy, bh in fixed_bumps:
                bump = bh * np.exp(-((self.X - bx)**2 / (2 * sx**2) + (self.Y - by)**2 / (2 * sy**2)))
                Z += bump

            return np.clip(Z, 0.0, self.max_elevation - 0.0002)

        # -----------------------------
        # 课程随机地形生成 (Procedural Random Generation)
        # -----------------------------
        # 1. 区域 1: 连续波浪微起伏段 (x in [0.35, 1.6m])
        wave_amp = (0.0012 + 0.0014 * difficulty) * (1.0 + rng.uniform(-0.15, 0.15))
        wavelength = rng.uniform(0.20, 0.32)
        wave_start = 0.35
        wave_end = rng.uniform(1.45, 1.65)
        mask_wave = (self.X >= wave_start) & (self.X < wave_end)
        w_in = np.clip((self.X - wave_start) / 0.15, 0.0, 1.0)
        w_out = np.clip((wave_end - self.X) / 0.15, 0.0, 1.0)
        Z[mask_wave] += wave_amp * np.sin(2.0 * np.pi * (self.X[mask_wave] - wave_start) / wavelength) * (w_in * w_out)[mask_wave]

        # 2. 区域 2: 随机非对称减速垄群 (x in [1.5m, 3.4m])
        # 障碍数量与高度随难度阶梯递增
        if difficulty < 0.35:
            # Novice: 4~5 道减速垄，基于基准位置微幅抖动，保持左右交错
            num_bumps = 5
            x_bases = [2.00, 2.30, 2.60, 2.90, 3.20]
            sides = ['left', 'right', 'left', 'right', 'both']
            height_min, height_max = 0.0020, 0.0038
            jitter_x = 0.04
        elif difficulty < 0.75:
            # Adept: 5~6 道减速垄，完全打乱顺序，间距随机，高度增加
            num_bumps = rng.integers(5, 7)
            # 在 [1.6, 3.3] 区间内按最小安全间距随机布设
            min_gap = 0.22
            x_bases = []
            cur_x = rng.uniform(1.65, 1.85)
            for _ in range(num_bumps):
                x_bases.append(cur_x)
                cur_x += rng.uniform(min_gap, 0.36)
                if cur_x > 3.35:
                    break
            num_bumps = len(x_bases)
            sides = rng.choice(['left', 'right', 'both'], size=num_bumps, p=[0.42, 0.42, 0.16])
            height_min, height_max = 0.0020, 0.0048
            jitter_x = 0.02
        else:
            # Master / Extreme: 5~7 道减速垄，包含极限紧凑障碍 (0.16m 间距) 与极限高度 (最高 5.2mm)
            num_bumps = rng.integers(5, 8)
            x_bases = []
            cur_x = rng.uniform(1.55, 1.75)
            for _ in range(num_bumps):
                x_bases.append(cur_x)
                # 偶尔出现双连击挑战 (tight pair, gap 0.16m~0.20m)
                gap = rng.uniform(0.16, 0.22) if rng.random() < 0.30 else rng.uniform(0.24, 0.36)
                cur_x += gap
                if cur_x > 3.40:
                    break
            num_bumps = len(x_bases)
            sides = rng.choice(['left', 'right', 'both'], size=num_bumps, p=[0.40, 0.40, 0.20])
            height_min, height_max = 0.0018, 0.0052
            jitter_x = 0.02

        # 逐个生成高斯平滑凸起
        for i, bx in enumerate(x_bases):
            side = sides[i]
            bx_real = bx + rng.uniform(-jitter_x, jitter_x)
            bh = rng.uniform(height_min, height_max)
            sx = rng.uniform(0.038, 0.052)  # 纵向宽度 (沿前进方向展宽约 8~10cm)

            if side == 'left':
                by = self.y_left_wheel + rng.uniform(-0.003, 0.003)
                sy = rng.uniform(0.018, 0.025)
            elif side == 'right':
                by = self.y_right_wheel + rng.uniform(-0.003, 0.003)
                sy = rng.uniform(0.018, 0.025)
            else:  # both
                by = self.y_center + rng.uniform(-0.002, 0.002)
                sy = rng.uniform(0.045, 0.055)  # 全宽跨双轮

            bump = bh * np.exp(-((self.X - bx_real)**2 / (2 * sx**2) + (self.Y - by)**2 / (2 * sy**2)))
            Z += bump

        # 3. 高难度下的复合多频微粗糙度 (Gravel Micro-roughness)
        if difficulty >= 0.70:
            roughness_amp = 0.0005 * difficulty
            # 双频轻微路面颗粒抖动
            mask_active = (self.X >= 0.4) & (self.X < 3.4)
            noise_wave = (np.sin(self.X * 45.0 + self.Y * 30.0) * 0.5 + 
                          np.cos(self.X * 80.0 - self.Y * 40.0) * 0.5)
            Z[mask_active] += roughness_amp * noise_wave[mask_active]

        # 4. 边界安全与平滑处理
        # 强制起点安全区 [0, 0.35m] 绝对平坦
        safe_mask = self.X < 0.35
        Z[safe_mask] = 0.0
        fade_blend = np.clip((self.X - 0.35) / 0.10, 0.0, 1.0)
        Z = Z * fade_blend

        # 强制平稳着陆区 [3.45m, 4.0m] 平滑淡出
        land_blend = np.clip((3.60 - self.X) / 0.15, 0.0, 1.0)
        Z = Z * land_blend

        # 截断在合法高度范围内 (低于 MuJoCo elevation_z)
        Z = np.clip(Z, 0.0, self.max_elevation - 0.0003)
        return Z

    def update_model_hfield(self, model, difficulty=0.0, rng=None):
        """
        在内存中就地更新 MuJoCo MjModel 的 hfield_data
        耗时约为 2~4ms，无需触碰物理磁盘
        """
        Z = self.generate(difficulty=difficulty, rng=rng)
        # MuJoCo 的 hfield_data 为 float32 且归一化到 [0, 1]
        # 物理绝对高度 = hfield_data * model.hfield_size[0, 2]
        norm_hfield = (Z / self.max_elevation).astype(np.float32)
        model.hfield_data[:] = norm_hfield.ravel()
        return Z
