"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/evaluation/metrics_utils.py`: 物理评测核心指标量化分析工具箱 (Contact Slip & Spectral Energy)
====================================================================================================
1. 真实接触面切向滑移计 (ContactSlipMeter):
   - 基于 MuJoCo 真实接触几何体 (mjContact) 提取车轮与地面的微观接触点 p。
   - 利用空间刚体运动学公式精确计算机体在接触点处的瞬时绝对速度：
     v_p = v_{com} + \omega \times (p - p_{com})
   - 将法向速度分量剔除，提取纯切向滑移速度模长：
     v_t = v_p - (v_p \cdot n) n, \quad \text{slip} = \|v_t\|
   - 彻底摆脱传统轮腿粗糙运动学近似 (|\omega r| - |v_{hub}|) 受髋关节高频摆动污染的缺陷，
     精准度量真实轮地抓地力与滑转损失。

2. 控制力矩高频抖动能谱分析器 (high_freq_energy):
   - 基于一维离散傅里叶变换 (RFFT) 与帕塞瓦尔能量守恒定理 (Parseval's Theorem)。
   - 提取执行机构关节力矩信号在截止频率 f_{cut} (默认 10Hz) 以上的高频颤振能谱能量，
     用于衡量强化学习策略是否存在破坏减速箱寿命与引起高频抖动的非平滑动作。
====================================================================================================
"""

import os
import sys
import numpy as np
import mujoco

# --------------------------------------------------------------------------------------------------
# 鲁棒路径引导
# --------------------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


class ContactSlipMeter:
    """
    MuJoCo 接触点真实轮地切向滑移测量器。
    """
    def __init__(self, model: mujoco.MjModel, wheel_body_names=("left_wheel", "right_wheel")):
        self.model = model
        self.wheel_ids = [model.body(n).id for n in wheel_body_names]
        self._vel6 = np.zeros(6, dtype=np.float64)

    def _point_velocity(self, data: mujoco.MjData, body_id: int, p: np.ndarray) -> np.ndarray:
        # mj_objectVelocity: flg_local=0 -> 世界坐标系, 速度参考点为 body 的 com (xipos)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, self._vel6, 0)
        w = self._vel6[:3]
        v = self._vel6[3:]
        r = p - data.xipos[body_id]
        return v + np.cross(w, r)

    def measure(self, data: mujoco.MjData):
        """
        测量当前物理步的车轮接触滑移与法向接触力。

        Returns:
            tuple[float, float, float, float]: (左轮滑移速度 m/s, 右轮滑移速度 m/s, 左轮法向力 N, 右轮法向力 N)
        """
        slip = [0.0, 0.0]
        fn = [0.0, 0.0]
        cnt = [0, 0]
        f6 = np.zeros(6, dtype=np.float64)
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            for k, wid in enumerate(self.wheel_ids):
                if b1 == wid or b2 == wid:
                    n = c.frame[:3]
                    vp = self._point_velocity(data, wid, c.pos)
                    vt = vp - np.dot(vp, n) * n
                    slip[k] += float(np.linalg.norm(vt))
                    mujoco.mj_contactForce(self.model, data, i, f6)
                    fn[k] += float(abs(f6[0]))
                    cnt[k] += 1
        for k in range(2):
            if cnt[k] > 0:
                slip[k] /= cnt[k]
        return slip[0], slip[1], fn[0], fn[1]


def high_freq_energy(signal: np.ndarray, dt: float, f_cut: float = 10.0) -> float:
    """
    计算输入时域信号在截断频率 f_cut 以上的高频抖动谱能量 (Parseval 归一化)。

    Args:
        signal (np.ndarray): 一维时域信号 (如关节力矩序列)。
        dt (float): 采样时间步长 (秒)。
        f_cut (float): 截止高频阈值 (Hz，默认 10.0)。

    Returns:
        float: 高频带谱能量密度积分值。
    """
    x = np.asarray(signal, dtype=float)
    if len(x) < 8:
        return 0.0
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), dt)
    return float(np.sum(np.abs(X[f >= f_cut]) ** 2) / len(x) * dt)
