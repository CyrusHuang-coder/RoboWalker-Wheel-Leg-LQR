"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/terrain/generate_terrain.py`: 4.0m 离线基础轻起伏高度图 PNG 静态生成器
====================================================================================================
1. 核心定位：
   - 离线生成 4.0m x 1.0m 的基础测试高度图，保存为 16-bit 灰度 PNG (`rough_terrain.png`)。
   - 供早期静态 MuJoCo XML (`wheel_leg_terrain.xml`) 中的 `<hfield file="..."/>` 直接引用加载。

2. 地形特征与物理尺寸：
   - 启动平坦区 (x in [0, 0.3m]): 供倒立摆在启动瞬间平稳收敛直立平衡。
   - 正弦连续微波 (x in [0.3m, 1.8m]): 波长 0.25m, 峰值 2.5mm，测试平顺路面吸震。
   - 左右非对称交错减速垄 (x in [1.8m, 3.5m]): 峰值 4.8mm 单侧隆起，激发双腿差动自适应横滚。
   - 平缓收尾缓冲带 (x in [3.5m, 4.0m]): 平稳制动停车。
====================================================================================================
"""
import os
import sys
import numpy as np
import imageio.v2 as imageio

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def generate_terrain(output_path=None,
                     nrow=256, ncol=1024,
                     x_len=4.0, y_len=1.0,
                     max_height=0.005):
    """
    x_len: 地形沿前进方向全长 (米)
    y_len: 地形横向全长 (米)
    max_height: 最大高差 (米, 0.005m = 5mm)
    """
    if output_path is None:
        output_path = os.path.join(current_dir, "rough_terrain.png")

    x = np.linspace(0, x_len, ncol)
    y = np.linspace(-y_len / 2, y_len / 2, nrow)
    X, Y = np.meshgrid(x, y)

    # 基础高度矩阵 (单位: 米)
    Z = np.zeros_like(X)

    # 1. 区域划分:
    #   x in [0, 0.3]: 完全平地
    #   x in [0.3, 1.8]: 纵向连续正弦微波 (波长 0.25m, 峰值 2.5mm)
    #   x in [1.8, 3.5]: 左右非对称交错颠簸凸起 (凸起峰值 3.5mm)
    #   x in [3.5, 4.0]: 平缓缓冲带

    # 段落 1: 正弦微波 (起伏波浪)
    mask_wave = (X >= 0.3) & (X < 1.8)
    wave_fade_in = np.clip((X - 0.3) / 0.2, 0.0, 1.0)
    wave_fade_out = np.clip((1.8 - X) / 0.2, 0.0, 1.0)
    wave_envelope = wave_fade_in * wave_fade_out
    Z[mask_wave] += 0.0025 * np.sin(2.0 * np.pi * (X[mask_wave] - 0.3) / 0.25) * wave_envelope[mask_wave]

    # 段落 2: 左右非对称交错实体减速垄 (单侧 bump, 左轮 y=-0.0015m, 右轮 y=-0.0335m)
    bumps = [
        # (x_center, y_center, x_sigma, y_sigma, height)
        (2.00, -0.0015, 0.045, 0.022, 0.0048), # 左侧实体减速垄 4.8mm
        (2.30, -0.0335, 0.045, 0.022, 0.0048), # 右侧实体减速垄 4.8mm
        (2.60, -0.0015, 0.045, 0.022, 0.0048), # 左侧实体减速垄 4.8mm
        (2.90, -0.0335, 0.045, 0.022, 0.0048), # 右侧实体减速垄 4.8mm
        (3.20, -0.0175, 0.050, 0.050, 0.0045), # 双轮全幅减速条 4.5mm
    ]

    for bx, by, sx, sy, bh in bumps:
        bump = bh * np.exp(-((X - bx)**2 / (2 * sx**2) + (Y - by)**2 / (2 * sy**2)))
        Z += bump

    # 归一化为 0~1 的 float，再转为 16-bit PNG 图像存储
    # 保留 0 高度基准
    Z_min = np.min(Z)
    if Z_min < 0:
        Z = Z - Z_min  # 抬升使得最低点为 0
    Z_max = np.max(Z)
    if Z_max == 0:
        Z_max = 1e-4

    # 归一化到 [0, 65535] 16位无损灰度 (MuJoCo 加载 PNG 图像时存在行翻转，必须 flipud 以保证 Y 轴与物理坐标系严格一致)
    Z_norm = (np.flipud(Z) / Z_max * 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, Z_norm)

    print(f"[Terrain] Generated {output_path}")
    print(f"  Grid size: {nrow}x{ncol}, physical: {x_len}m x {y_len}m")
    print(f"  Physical Peak Height: {Z_max * 1000:.2f} mm")
    return x_len, y_len, Z_max


if __name__ == "__main__":
    generate_terrain()
