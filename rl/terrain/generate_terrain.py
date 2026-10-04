"""
生成高保真轻起伏地形高度图 (Heightfield .png)
地形特征：
1. 启动缓冲区 (x < 0.2m): 平坦路面，供初始直立收敛
2. 纵向轻微正弦波段 (幅值 2.0mm ~ 3.5mm, 对应微型车 8mm 轮径)
3. 左右非对称随机凸起段 (单侧隆起 3.0mm, 用于激发并测试双腿自适应横滚平衡)
4. 输出: rl/terrain/rough_terrain.png (16-bit 灰度或 8-bit 高精度灰度图)
"""
import os
import numpy as np
import imageio.v2 as imageio


def generate_terrain(output_path="rl/terrain/rough_terrain.png",
                     nrow=256, ncol=1024,
                     x_len=4.0, y_len=1.0,
                     max_height=0.005):
    """
    x_len: 地形沿前进方向全长 (米)
    y_len: 地形横向全长 (米)
    max_height: 最大高差 (米, 0.005m = 5mm)
    """
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

    # 归一化到 [0, 65535] 16位无损灰度
    Z_norm = (Z / Z_max * 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, Z_norm)

    print(f"[Terrain] Generated {output_path}")
    print(f"  Grid size: {nrow}x{ncol}, physical: {x_len}m x {y_len}m")
    print(f"  Physical Peak Height: {Z_max * 1000:.2f} mm")
    return x_len, y_len, Z_max


if __name__ == "__main__":
    generate_terrain()
