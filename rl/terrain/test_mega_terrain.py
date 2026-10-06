"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/terrain/test_mega_terrain.py`: 10.5m 终极地狱级赛道高度图、几何航向参考与物理轮距标定
====================================================================================================
1. 核心定位与赛道架构：
   - 本模块定义了 10.5m 复合恶劣赛道的高精实体高度场数学解析模型与中心引导线轨迹。
   - 赛道全长 10.5m，横向宽 1.0m，离散网格 2560 x 256，最大标称物理高差 45mm。
   - 赛道分为六大高难度关卡：
     * Zone 1 [0.55m ~ 1.70m]: 巨幅左右异相波浪起伏 (2.0mm -> 3.8mm, 左右轮相位差 0.60*pi, 强烈摇晃)
     * Zone 2 [1.80m & 2.02m]: 弯前单侧垫高小板砖 (+4.2mm, 精准对齐左右车轮滚动轨迹)
     * Zone 3 [2.20m ~ 4.80m]: 连续大摆幅 S 弯道 (摆幅 +-90mm, 叠加 2.8mm 连续凹凸波浪路面)
     * Zone 4 [4.80m ~ 5.70m]: 35mm 陡峭仰攻大坡 (坡度 ~6.5°, 坡面叠加 2.5mm 波浪与交错单侧板砖)
     * Zone 5 [5.70m ~ 7.00m]: 35mm 峰顶超长开阔平台 (1.30m 平台，包含 4 处交错小板砖与横断障碍)
     * Zone 6 [7.00m ~ 7.98m]: 非均匀变速双阶下坡 (陡降阶梯一 35->18mm, 0.40m 中继平台带右轮砖, 陡降阶梯二 18->0mm)
     * Zone 7 [8.15m ~ 9.20m]: 坡后 7 条高频密集搓板路 (幅值 2.8mm) + 错位重型石阶 (+4.5mm)
     * Zone 8 [9.20m ~ 10.0m]: 终点平稳缓冲冲刺区 (10.0m 停在终点黄线处)

2. 实车轮轴与左右轮接触面标定几何常数：
   - 机器人在 MuJoCo XML 中的 base_link 位于 y_ref
   - 左右两腿在 Y 轴上的装配偏置：左腿 y_offset = 0.000m, 右腿 y_offset = -0.0315m
   - 经过腿部与轮毂的装配变换后：
     * 左轮中心接触轨迹：y_left  = y_ref - 0.0015 m (相对双轮中线偏移 +16.0 mm)
     * 右轮中心接触轨迹：y_right = y_ref - 0.0335 m (相对双轮中线偏移 -16.0 mm)
     * 机器人物理轮距：32.0 mm
     * 双轮正中几何中心线：y_mid = y_ref - 0.0175 m (以此为原点，左右轮各对称分布 +-16mm)
====================================================================================================
"""
import os
import sys
import numpy as np

# --------------------------------------------------------------------------------------------------
# 鲁棒路径引导
# --------------------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==================================================================================================
# 实车轮轴接触点相对双轮中线的物理偏置 (标称轮距 32mm):
# ==================================================================================================
OFF_LEFT_WHEEL  = +0.0160   # 左轮接触轨迹中心 (+16mm)
OFF_RIGHT_WHEEL = -0.0160   # 右轮接触轨迹中心 (-16mm)
OFF_TRACK_MID   =  0.0000   # 赛道几何正中线 (中心虚线所在位置)


def get_mega_track_reference(x: float, y0: float = -0.0175):
    """
    计算给定前进距离 x 处的理想路径横向位置参考 y_ref 和期望航向角 yaw_ref。

    - 直线段 (x < 2.20m 或 x > 4.80m): y_ref = y0, yaw_ref = 0
    - S 弯道段 (2.20m <= x <= 4.80m): 正弦平滑摆动 (摆幅 +-90mm, 长度 2.60m, 最小转弯半径 R_min >= 1.62m)

    Args:
        x (float): 机器人当前沿赛道前进的 X 坐标 (米)
        y0 (float): 赛道中线在全局坐标系中的初始偏置 (默认 -0.0175m)

    Returns:
        tuple[float, float]: (y_ref, yaw_ref)
    """
    if 2.20 <= x <= 4.80:
        rel_x = x - 2.20
        # 航向参考线平滑连续
        y_ref = y0 + 0.09 * np.sin(2.0 * np.pi * rel_x / 2.60)
        dy_dx = 0.09 * (2.0 * np.pi / 2.60) * np.cos(2.0 * np.pi * rel_x / 2.60)
        yaw_ref = np.arctan(dy_dx)
    else:
        y_ref = y0
        yaw_ref = 0.0
    return float(y_ref), float(yaw_ref)


def build_mega_hfield(x_len: float = 10.5, y_len: float = 1.0, nrow: int = 256, ncol: int = 2560, max_elevation: float = 0.045):
    """
    构建 10.5m 复合恶劣赛道的高精高程矩阵 Z(x, y)。

    Args:
        x_len (float): 赛道前进方向总长 (米, 默认 10.5)
        y_len (float): 赛道横向宽度 (米, 默认 1.0)
        nrow (int): Y 轴离散格点数 (默认 256)
        ncol (int): X 轴离散格点数 (默认 2560)
        max_elevation (float): MuJoCo XML 中的 elevation_z 上限 (默认 0.045m = 45mm)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (Z, X, Y)
    """
    x = np.linspace(0, x_len, ncol, dtype=np.float32)
    y = np.linspace(-y_len / 2.0, y_len / 2.0, nrow, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    y0 = -0.0175

    # 计算沿 X 轴分布的 base_link 参考线 Y_base_1d(x)
    y_base_1d = np.zeros_like(x)
    for i in range(len(x)):
        y_base_1d[i], _ = get_mega_track_reference(x[i], y0)
    Y_base_2d = np.tile(y_base_1d, (nrow, 1))

    # 局部横向相对坐标与中线距离
    rel_y = Y - Y_base_2d
    dist_to_mid = np.abs(rel_y - OFF_TRACK_MID)

    # =========================================================================
    # 1. [0.55m ~ 1.70m] 巨幅波浪起伏 (2.0mm -> 3.8mm, 波长 0.20m, 左右轮异相扭摆)
    # =========================================================================
    mask_wave1 = (X >= 0.55) & (X < 1.70)
    prog1 = np.clip((X[mask_wave1] - 0.55) / 1.15, 0.0, 1.0)
    amp1 = 0.0020 + 0.0018 * prog1  # 2.0mm -> 3.8mm (诱发剧烈可见摇晃，绝对不翻车)

    # 左右轮在波浪上有 0.60*pi 的相位差，左起右伏，诱发剧烈横滚摇晃
    k_w1 = 2.0 * np.pi / 0.20
    phase_l = k_w1 * (X[mask_wave1] - 0.55)
    phase_r = phase_l + 0.60 * np.pi
    w_weight = np.clip((rel_y[mask_wave1] - OFF_RIGHT_WHEEL) / (OFF_LEFT_WHEEL - OFF_RIGHT_WHEEL), 0.0, 1.0)
    wave_val = (1.0 - w_weight) * np.sin(phase_r) + w_weight * np.sin(phase_l)
    Z[mask_wave1] += amp1 * wave_val

    # 弯前单侧垫高小板砖 (精准置于左右轮轨迹中心，车轮 100% 正中压上去)
    # 1.80m: 左轮板砖 (y = OFF_LEFT_WHEEL, 高 4.2mm)
    Z += 0.0042 * np.exp(-((X - 1.80)**2 / (2 * 0.032**2) + (rel_y - OFF_LEFT_WHEEL)**2 / (2 * 0.012**2)))
    # 2.02m: 右轮板砖 (y = OFF_RIGHT_WHEEL, 高 4.2mm)
    Z += 0.0042 * np.exp(-((X - 2.02)**2 / (2 * 0.032**2) + (rel_y - OFF_RIGHT_WHEEL)**2 / (2 * 0.012**2)))

    # =========================================================================
    # 2. [2.20m ~ 4.80m] 连续 S 弯道区域：叠加连续 2.8mm 凹凸波浪路面 (波长 0.28m)
    # =========================================================================
    mask_scurve = (X >= 2.20) & (X < 4.80)
    w_scurve = 0.0028 * np.sin(2.0 * np.pi * (X[mask_scurve] - 2.20) / 0.28)
    env_s = np.clip((X[mask_scurve] - 2.20) / 0.15, 0.0, 1.0) * np.clip((4.80 - X[mask_scurve]) / 0.15, 0.0, 1.0)
    Z[mask_scurve] += w_scurve * env_s

    # =========================================================================
    # 3. [4.80m ~ 8.20m] 宏观 35mm 大山丘 + 峰顶超长平台 + 非均匀变速台阶下坡
    # =========================================================================
    hill_spine = np.zeros_like(x)
    H_SUMMIT = 0.035
    H_SHELF  = 0.018
    for i in range(len(x)):
        xi = x[i]
        if xi < 4.80:
            zi = 0.0
        elif xi < 5.70:  # 仰攻爬坡 (0 -> 35mm, 坡长 0.90m)
            zi = H_SUMMIT * (0.5 - 0.5 * np.cos(np.pi * (xi - 4.80) / 0.90))
        elif xi <= 7.00:  # 峰顶长平台 (长 1.30m)
            zi = H_SUMMIT
        elif xi <= 7.28:  # 陡降第 1 阶 (35 -> 18mm, 坡长仅 0.28m)
            zi = H_SUMMIT - (H_SUMMIT - H_SHELF) * (0.5 - 0.5 * np.cos(np.pi * (xi - 7.00) / 0.28))
        elif xi <= 7.68:  # 半山腰 18mm 水平中继平台 (长 0.40m)
            zi = H_SHELF
        elif xi <= 7.98:  # 陡降第 2 阶 (18 -> 0mm, 坡长仅 0.30m)
            zi = H_SHELF * (0.5 + 0.5 * np.cos(np.pi * (xi - 7.68) / 0.30))
        else:
            zi = 0.0
        hill_spine[i] = zi

    hill_2d = np.tile(hill_spine, (nrow, 1))
    Z += hill_2d

    # 4. 爬坡路段：坡面波浪起伏 + 坡面中后期交替单侧小板砖
    mask_climb = (X >= 4.90) & (X < 5.65)
    Z[mask_climb] += 0.0025 * np.sin(2.0 * np.pi * (X[mask_climb] - 4.90) / 0.20)

    # 5.35m 右轮垫高板砖 (+4.2mm)
    Z += 0.0042 * np.exp(-((X - 5.35)**2 / (2 * 0.032**2) + (rel_y - OFF_RIGHT_WHEEL)**2 / (2 * 0.012**2)))
    # 5.55m 左轮垫高板砖 (+4.2mm)
    Z += 0.0042 * np.exp(-((X - 5.55)**2 / (2 * 0.032**2) + (rel_y - OFF_LEFT_WHEEL)**2 / (2 * 0.012**2)))

    # 5. 35mm 峰顶超长平台上的交错小板砖与横断障碍群 (x in [5.80, 6.90])
    summit_bricks = [
        (5.95, OFF_LEFT_WHEEL,  0.032, 0.012, 0.0045),  # 左轮单侧小板砖 (+4.5mm)
        (6.20, OFF_TRACK_MID,   0.022, 0.055, 0.0032),  # 赛道中央横断带 (双轮齐碾)
        (6.45, OFF_RIGHT_WHEEL, 0.032, 0.012, 0.0045),  # 右轮单侧小板砖 (+4.5mm)
        (6.75, OFF_LEFT_WHEEL,  0.032, 0.012, 0.0042),  # 左轮单侧小板砖 (+4.2mm)
    ]
    for bx, by_off, sx, sy, bh in summit_bricks:
        Z += bh * np.exp(-((X - bx)**2 / (2 * sx**2) + (rel_y - by_off)**2 / (2 * sy**2)))

    # 6. 非均匀下坡区域：下坡路面凹凸起伏 + 半山腰平台小板砖
    mask_desc1 = (X >= 7.02) & (X < 7.26)
    Z[mask_desc1] += 0.0028 * np.sin(2.0 * np.pi * (X[mask_desc1] - 7.02) / 0.12)

    # 半山腰中继平台 (7.28 ~ 7.68m) 上的右轮小板砖 (7.48m, +4.0mm)
    Z += 0.0040 * np.exp(-((X - 7.48)**2 / (2 * 0.032**2) + (rel_y - OFF_RIGHT_WHEEL)**2 / (2 * 0.012**2)))

    # 陡降第 2 阶坡面上的微起伏 (7.70 ~ 7.96m)
    mask_desc2 = (X >= 7.70) & (X < 7.96)
    Z[mask_desc2] += 0.0025 * np.sin(2.0 * np.pi * (X[mask_desc2] - 7.70) / 0.14)

    # =========================================================================
    # 7. [8.15m ~ 9.20m] 坡后 7 条密集高频搓板路 + 错位重型石阶
    # =========================================================================
    rumbles = [8.20, 8.29, 8.38, 8.47, 8.56, 8.65, 8.74]
    for rx in rumbles:
        Z += 0.0028 * np.exp(-((X - rx)**2 / (2 * 0.012**2) + (dist_to_mid)**2 / (2 * 0.055**2)))

    # 坡后错位重型石阶小板砖
    Z += 0.0045 * np.exp(-((X - 8.95)**2 / (2 * 0.035**2) + (rel_y - OFF_LEFT_WHEEL)**2 / (2 * 0.014**2)))
    Z += 0.0045 * np.exp(-((X - 9.15)**2 / (2 * 0.035**2) + (rel_y - OFF_RIGHT_WHEEL)**2 / (2 * 0.014**2)))

    # 全赛道横向平顶掩码：赛道双轮范围外侧平滑收敛，防止边缘畸变
    track_mask = np.where(dist_to_mid <= 0.075, 1.0, np.clip(1.0 - (dist_to_mid - 0.075) / 0.04, 0.0, 1.0))
    Z = Z * track_mask

    # 安全淡入淡出 (起步 0.40m 平顺，终点 10.1m 后平顺)
    fade_in = np.clip((X - 0.45) / 0.10, 0.0, 1.0)
    fade_out = np.clip((10.35 - X) / 0.15, 0.0, 1.0)
    Z = Z * fade_in * fade_out
    Z = np.clip(Z, 0.0, max_elevation - 0.0002)

    return Z, X, Y
