"""
test_extreme_terrain.py: 构建用户要求的极复杂分段地形
===============================================
地形构成:
  1. [0.0 ~ 0.5m] 平路起步
  2. [0.5 ~ 2.0m] 逐渐加剧的波浪形颠簸起伏 (波高从 2.5mm 递增至 6.0mm)
  3. [2.0 ~ 2.35m] 非对称减速垄
  4. [2.4 ~ 3.3m] 爬上宏观 35mm 大坡，坡面中后期带有突起障碍坎
  5. [3.3 ~ 4.2m] 35mm 峰顶平台上设置交错障碍物群 (左垄、横条、右坎)
  6. [4.2 ~ 5.3m] 非均匀下坡：陡降1 (35->18mm) -> 缓坡/中继平台 (18mm) -> 陡降2 (18->0mm)
  7. [5.4 ~ 6.3m] 坡后凹凸不平高频搓板与错位石阶
  8. [6.3 ~ 7.2m] 平顺减速与终点线稳定驻留 (无后溜)
"""
import os
import sys
import numpy as np

def build_extreme_hfield(x_len=7.5, y_len=1.0, nrow=256, ncol=1920, max_elevation=0.045):
    x = np.linspace(0, x_len, ncol, dtype=np.float32)
    y = np.linspace(-y_len / 2.0, y_len / 2.0, nrow, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    y_left = -0.0015
    y_right = -0.0335
    y_center = -0.0175

    # 1. [0.5m ~ 2.0m] 起伏程度逐渐增加的波浪路面
    mask_waves = (X >= 0.50) & (X < 2.00)
    wave_progress = np.clip((X[mask_waves] - 0.50) / 1.50, 0.0, 1.0)
    amp_progressive = 0.0025 + 0.0035 * wave_progress  # 2.5mm -> 6.0mm
    w1 = np.sin(2.0 * np.pi * (X[mask_waves] - 0.50) / 0.28)
    w2 = 0.3 * np.sin(2.0 * np.pi * (X[mask_waves] - 0.50) / 0.14)
    roll_phase = 0.25 * ((Y[mask_waves] - y_center) / 0.04)
    Z[mask_waves] += amp_progressive * (w1 + w2 + roll_phase * w1)

    # 2. [2.0m ~ 2.35m] 非对称大颠簸减速垄
    bumps_pre = [
        (2.08, y_left,  0.040, 0.018, 0.0045),
        (2.28, y_right, 0.040, 0.018, 0.0045),
    ]
    for bx, by, sx, sy, bh in bumps_pre:
        Z += bh * np.exp(-((X - bx)**2 / (2 * sx**2) + (Y - by)**2 / (2 * sy**2)))

    # 3. [2.40m ~ 5.30m] 宏观山丘主干骨架 (35mm 高，含非均匀台阶式下坡)
    hill_spine = np.zeros_like(x)
    H_SUMMIT = 0.035
    H_SHELF = 0.018
    for i in range(len(x)):
        xi = x[i]
        if xi < 2.40:
            zi = 0.0
        elif xi < 3.10: # 大爬坡
            zi = H_SUMMIT * (0.5 - 0.5 * np.cos(np.pi * (xi - 2.40) / 0.70))
        elif xi <= 4.20: # 峰顶长平台
            zi = H_SUMMIT
        elif xi <= 4.55: # 陡降第 1 阶
            zi = H_SUMMIT - (H_SUMMIT - H_SHELF) * (0.5 - 0.5 * np.cos(np.pi * (xi - 4.20) / 0.35))
        elif xi <= 4.85: # 中继缓坡平台
            zi = H_SHELF
        elif xi <= 5.30: # 陡降第 2 阶回归地面
            zi = H_SHELF * (0.5 + 0.5 * np.cos(np.pi * (xi - 4.85) / 0.45))
        else:
            zi = 0.0
        hill_spine[i] = zi

    # 桥面横向 14cm 平顶，两侧斜坡
    hill_2d = np.tile(hill_spine, (nrow, 1))
    dy = np.abs(Y - y_center)
    y_mask = np.where(dy <= 0.07, 1.0, np.clip(1.0 - (dy - 0.07) / 0.05, 0.0, 1.0))
    Z += hill_2d * y_mask

    # 4. 爬坡后期坡面上的突起障碍物 (x in [2.90, 3.15])
    slope_bumps = [
        (2.95, y_right, 0.035, 0.016, 0.0035),
        (3.08, y_left,  0.035, 0.016, 0.0038),
    ]
    for bx, by, sx, sy, bh in slope_bumps:
        Z += bh * np.exp(-((X - bx)**2 / (2 * sx**2) + (Y - by)**2 / (2 * sy**2)))

    # 5. 35mm 峰顶平台上的密集障碍物群 (x in [3.35, 4.10])
    summit_obstacles = [
        (3.45, y_left,   0.040, 0.018, 0.0038),
        (3.68, y_center, 0.025, 0.055, 0.0030),
        (3.92, y_right,  0.040, 0.018, 0.0038),
        (4.12, y_left,   0.035, 0.018, 0.0032),
    ]
    for bx, by, sx, sy, bh in summit_obstacles:
        Z += bh * np.exp(-((X - bx)**2 / (2 * sx**2) + (Y - by)**2 / (2 * sy**2)))

    # 6. 下坡中继半山平台上的小障碍 (x = 4.70m)
    Z += 0.0028 * np.exp(-((X - 4.70)**2 / (2 * 0.030**2) + (Y - y_right)**2 / (2 * 0.018**2)))

    # 7. 坡后 [5.45m ~ 6.25m] 高频搓板路与错位石阶
    rumbles = [5.48, 5.58, 5.68, 5.78, 5.88, 5.98]
    for rx in rumbles:
        Z += 0.0022 * np.exp(-((X - rx)**2 / (2 * 0.015**2) + (Y - y_center)**2 / (2 * 0.060**2)))

    slabs = [
        (6.10, y_left,  0.045, 0.020, 0.0038),
        (6.25, y_right, 0.045, 0.020, 0.0038),
    ]
    for sx_pos, sy_pos, sx_w, sy_w, sh in slabs:
        Z += sh * np.exp(-((X - sx_pos)**2 / (2 * sx_w**2) + (Y - sy_pos)**2 / (2 * sy_w**2)))

    # 安全淡入淡出
    fade_in = np.clip((X - 0.40) / 0.10, 0.0, 1.0)
    fade_out = np.clip((7.20 - X) / 0.15, 0.0, 1.0)
    Z = Z * fade_in * fade_out
    Z = np.clip(Z, 0.0, max_elevation - 0.0002)

    return Z, X, Y
