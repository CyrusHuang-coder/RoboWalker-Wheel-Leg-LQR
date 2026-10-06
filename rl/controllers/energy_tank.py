"""
energy_tank.py: 滑移耦合离散精确能量储罐 (Slip-Coupled Exact Discrete Energy Tank, SCT-RRL)
=================================================================================================
【模块功能介绍】
本模块实现了无源性理论指导下的安全残差约束系统 (Passivity-Bounded Energy Tank)。
针对强化学习残差网络在越障工况下容易输出激进动作、引发轮地打滑失稳和高频抖颤的物理痛点，
本模块构建了一套可证明无源性的能量记账与动作几何投影机制：

1. 弹性势能精确记账 (Exact Potential Energy Accounting):
   将髋关节虚拟悬架建模为刚度受控弹簧系统：tau_i = -K * (q_i - q_target_i) - D * dq_i。
   系统势能 H(q) = 0.5 * K * ||q - q_target||^2。
   残差动作仅在策略步边界 (20ms/50Hz) 产生跳变。在当前物理构型 q 下，参数跳变注入系统的能量 dW
   通过解析几何严格计算，杜绝离散积分累积误差。

2. 二分无源投影与自适应衰减 (Bisection Passivity Projection & Authority Scaling):
   - 预算硬约束: 严格限制单步注能 dW <= budget = E_T - E_min。若策略超出预算，二分求解最优缩放因子 alpha*。
   - 复归动作豁免 (Restoring Exemption): 当残差动作朝向标称稳态 (k=1.0, dr=0.0) 复原时无条件放行，彻底防止自锁。
   - 弹性阻抗预缩放: 当储能低于 70% 时，平滑柔化残差权限，自适应切断过度做功。

3. 滑移与耗散双向动态耦合 (Slip & Dissipation Dynamic Coupling):
   - 1kHz 内循环中，阻尼吸收的冲击动能以系数 beta 持续充入储罐；
   - 轮胎超额打滑机械能损耗以系数 gamma 实时泄放储罐能量，强迫系统进入“被动节流”模式。
=================================================================================================
"""
import os
import sys
import numpy as np

# 路径自适应引导
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


class SlipCoupledEnergyTank:
    """
    滑移耦合离散精确能量储罐控制器
    维持虚拟储罐能量标量 E_T，通过几何投影将动作限制在无源性安全凸集内。
    """
    def __init__(self, E_init=6.5e-4, E_min=0.5e-4, E_max=8.0e-4,
                 beta=0.0035, gamma=0.016,
                 slip_coupling=True, n_bisect=8,
                 p_slip_deadband=0.003, tau_slip=0.040, lambda_relax=0.92):
        """
        参数:
          - E_init / E_min / E_max: 储罐初始、最低保护与最高储能阈值 (J)
          - beta: 悬架阻尼耗散回充效率系数
          - gamma: 轮地超额打滑功率泄放系数 (slip_coupling=False 时关闭)
          - n_bisect: 二分投影搜索迭代次数 (8次已达 <0.4% 数值精度)
          - p_slip_deadband: 平地正常微滑死区基线 (W)
          - tau_slip: 打滑功率低通滤波时间常数 (s)
          - lambda_relax: 残差权限受限时的名义姿态松弛阻尼因子
        """
        self.E_init = E_init
        self.E_min = E_min
        self.E_max = E_max
        self.beta = beta
        self.gamma = gamma if slip_coupling else 0.0
        self.n_bisect = n_bisect
        self.p_slip_deadband = p_slip_deadband
        self.tau_slip = tau_slip
        self.lambda_relax = lambda_relax
        self.reset()

    def reset(self):
        """重置能量状态与历史统计"""
        self.E = float(self.E_init)
        self.k_prev = 1.0
        self.dr_prev = 0.0
        self.pending_dW = 0.0
        self.last_alpha = 1.0
        self.P_slip_avg = 0.0
        self.p_slip_filtered = 0.0
        self._acc_n = 0
        self._acc_slip = 0.0

    @staticmethod
    def spring_energy(controller, s, k_scale, delta_roll):
        """计算给定刚度与偏置下，双侧虚拟弹簧当前的瞬时弹性势能 (J)"""
        kp, _, _, ql, qr = controller.hip_spring_params(s, delta_roll, k_scale)
        return 0.5 * kp * ((s['left_hip_pos'] - ql) ** 2 + (s['right_hip_pos'] - qr) ** 2)

    def project(self, controller, s, k_req, dr_req):
        """
        核心无源投影算子：将策略请求的 (k_req, dr_req) 投影为安全执行的 (k_exec, dr_exec)
        返回: (k_exec, dr_exec, eff_alpha, dW)
        """
        H0 = self.spring_energy(controller, s, self.k_prev, self.dr_prev)
        budget = max(0.0, self.E - self.E_min)

        # 1. 判定是否为向标称位姿复归的动作 (Restoring Action: 动作减小，系统自然回落，放行)
        is_restoring = (abs(dr_req) <= abs(self.dr_prev) + 1e-5) and \
                       (abs(k_req - 1.0) <= abs(self.k_prev - 1.0) + 1e-5)

        # 2. 弹性阻抗预缩放: 当储能低于 70% 时平滑连续收拢残差权限
        energy_ratio = np.clip(budget / max(self.E_max - self.E_min, 1e-12), 0.0, 1.0)
        if is_restoring:
            alpha_elastic = 1.0
        else:
            alpha_elastic = float(min(1.0, np.sqrt(energy_ratio / 0.70)))

        k_target = 1.0 + alpha_elastic * (k_req - 1.0)
        dr_target = alpha_elastic * dr_req

        def dW_at(a):
            k = self.k_prev + a * (k_target - self.k_prev)
            dr = self.dr_prev + a * (dr_target - self.dr_prev)
            return self.spring_energy(controller, s, k, dr) - H0

        d1 = dW_at(1.0)
        if d1 <= budget:
            alpha = 1.0
            dW = d1
        elif is_restoring:
            alpha = 1.0
            dW = max(0.0, min(d1, budget))
        else:
            # 3. 超过注能预算：二分法求解最大允许注能比例 alpha*
            lo, hi = 0.0, 1.0
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                if dW_at(mid) <= budget:
                    lo = mid
                else:
                    hi = mid
            alpha = lo
            dW = dW_at(alpha)

        k_exec = self.k_prev + alpha * (k_target - self.k_prev)
        dr_exec = self.dr_prev + alpha * (dr_target - self.dr_prev)

        # 4. 残留偏置向标称值平滑松弛 (Nominal Relaxation)
        eff_alpha = alpha * alpha_elastic
        if eff_alpha < 0.999:
            k_exec = 1.0 + (k_exec - 1.0) * self.lambda_relax
            dr_exec = dr_exec * self.lambda_relax

        self.k_prev, self.dr_prev = k_exec, dr_exec
        self.E = float(np.clip(self.E - dW, self.E_min, self.E_max))
        self.last_alpha = eff_alpha
        return k_exec, dr_exec, eff_alpha, dW

    def accumulate(self, controller, s, p_slip, dt):
        """
        1 kHz 底层内循环物理更新：
        进行打滑功率低通滤波、阻尼能量回收与打滑过度泄放。
        """
        # 1. 一阶低通滤波 (时间常数 tau_slip = 40ms)
        alpha_lp = dt / (self.tau_slip + dt)
        self.p_slip_filtered += alpha_lp * (p_slip - self.p_slip_filtered)

        # 2. 悬架阻尼吸收耗散功率
        _, kdl, kdr, _, _ = controller.hip_spring_params(s, self.dr_prev, self.k_prev)
        p_damp = kdl * s['left_hip_vel'] ** 2 + kdr * s['right_hip_vel'] ** 2

        # 3. 超额打滑功率 (扣除平地正常基线)
        p_slip_excess = max(0.0, self.p_slip_filtered - self.p_slip_deadband)

        # 4. 连续储能动态积分 (打滑剧烈时抑制回充，优先平息打滑)
        recharge_gate = max(0.0, 1.0 - p_slip_excess / 0.005)
        dE = (self.beta * p_damp * recharge_gate - self.gamma * p_slip_excess) * dt
        self.E = float(np.clip(self.E + dE, self.E_min, self.E_max))

        self._acc_slip += self.p_slip_filtered
        self._acc_n += 1

    def commit(self):
        """策略步边界 (50Hz) 调用：刷新平均打滑功率供上层观测"""
        if self._acc_n > 0:
            self.P_slip_avg = self._acc_slip / self._acc_n
        self._acc_slip = 0.0
        self._acc_n = 0

    def features(self):
        """构建储罐特征观测向量: [归一化储能 (0~1), 最近缩放因子 alpha*, 归一化打滑功率]"""
        e = (self.E - self.E_min) / max(self.E_max - self.E_min, 1e-12)
        return np.array([e, self.last_alpha, np.tanh(self.P_slip_avg * 20.0)], dtype=np.float32)
