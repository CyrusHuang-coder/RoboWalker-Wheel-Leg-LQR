# 轮腿机器人仿真实战 - 24 个原始实验探针完整技术档案 (Archive)

> **归档说明**：  
> 本目录收录了在突破轮腿自平衡控制算法过程中**真实编写、未经任何删改或跳步**的全部 **24 个原始实验脚本**。  
> 每一个脚本都记录了一次真实的工程试错、物理假设与机理突破，完整重现从“开局翻倒弹飞”到“破千步、破万步稳健自立”的全过程研发路径。

---

## 五大研发阶段与 24 个原始脚本全景索引

### 阶段一：初探 LQR 与接触“弹跳飞天”疑难排查（Phase 1: Initial Exploration & Bounce Diagnosis）
* **`find_lqr_signs.py`**：最初穷举 16 种控制力矩与状态变量正负号组合的脚本；
* **`_test_smooth.py`**：初次尝试给轮电机加入平滑滤波，探查力矩突变对稳定性的影响；
* **`_test_lqr_grid.py`**：第一次对原始 Riccati 方程 $Q/R$ 矩阵进行大范围网格扫描；
* **`_test_lqr_1e7.py`**：将 $R$ 控制能量惩罚扩大到 $10^7$，验证极小力矩下小车响应；
* **`_debug_bounce.py`**：实时跟踪接触点垂直冲量，排查小车为何在没有任何大外力下离地腾空；
* **`_test_leg_contype.py`**：**【重大物理发现】** 实测腿部 STL 顶点比车轮底低 $4.8\text{ mm}$，锁定“腿部刮地擦飞”真凶，确立 `contype="0"`。

### 阶段二：真实地表接触标定与动力学杠杆解密（Phase 2: Contact Physics & Torque Leverage）
* **`_test_settled.py`**：在地面施加超大阻尼让整车沉降，精确测量受载静平衡高度（$z=0.0448\text{ m}$）与实际静平衡倾角；
* **`_test_wheel_dir.py`**：单点标定测试：给轮子施加 $+0.01\text{ N}\cdot\text{m}$，确认在 MuJoCo 局部轴下正扭矩对应 $+X$ 前进；
* **`_test_analytical.py`**：基于解析动力学方程对比仿真角加速度与理论加速度，验证转动惯量量纲；
* **`_test_torque_effect.py`**：**【重大力学突破】** 推导轮地接触力矩杠杆放大比 $1 + L/r = 6$（双轮达 **12倍**），揭示为什么微型小车必须使用毫牛米（$\text{mN}\cdot\text{m}$）级微弱力矩。

### 阶段三：坐标系轴心对齐与纯俯仰平衡收敛（Phase 3: Coordinate Alignment & Pure Pitch Balance）
* **`_test_lqr_clean.py`**：去除腿部碰撞干涉后，首次在干净物理环境下运行原生 LQR；
* **`_test_lqr_aligned.py`**：引入 SolidWorks CAD 装配偏置补偿（质心与髋轴偏差 $1.74\text{ mm}$）；
* **`_test_pure_pitch.py`**：临时切断位置闭环，仅保留俯仰角闭环，验证倒立摆自身是否具备纯姿态恢复能力；
* **`_grid_fast.py`** 与 **`_grid_fast2.py`**：纯 Python 极速扫描器，3 秒内完成 2000 组姿态反馈增益剪枝。

### 阶段四：串级位置外环与 Step 1150 奇异点攻坚（Phase 4: Cascade Architecture & Step 1150 Bug）
* **`_test_cascade.py`**：首次构建“位置误差 $\to$ 期望俯仰倾角 $\to$ 轮电机平衡力矩”的串级反馈；
* **`_test_cascade_aligned.py`**：将髋关节标称锁定角精确校准至 $q_{\text{hip}} = -0.0042\text{ rad}$，平衡步数首破千步；
* **`_debug_step1150.py`**：诊断运行到 1.15 秒（1150 步）时位置漂移积累导致轮电机力矩打顶饱和的机理；
* **`_find_stable_gains.py`**：针对长时漂移，自动重构位置恢复增益与倾角限幅。

### 阶段五：高频阻尼滤波、速度巡航与终极收敛（Phase 5: Damping Filter & Long-Horizon Stability）
* **`_test_higher_gains.py`**：测试增大比例刚度对抗外部轻微偏载的可行性；
* **`_test_smaller_omega.py`**：调低角速度微分项，减小高频噪声对执行器的离散冲击；
* **`_test_heavy_filter.py`**：对比不同低通滤波常数 $\alpha \in [0.70, 0.95]$ 对传感器毛刺的抑制能力；
* **`_test_velocity_damping.py`**：**【关键转折点】** 成功突破 **2144 步（超 2.1 秒）平稳直立**，验证一阶滤波阻尼核心有效性；
* **`_tune_lqr_qr.py`**：结合滤波与物理约束，对 Riccati 代数方程的 $Q$ 与 $R$ 矩阵进行终极收敛调谐。

---

## 与交付物 `test_balance.py` 的传承关系

经过上述 24 个单点脚本的逐步推导与验证，我们最终将**消除腿刮地、12倍力矩折算、一阶角速度滤波、-0.0042rad髋对齐、串级倒立摆非最小相位位置恢复**等全部物理法则，凝练到了根目录的 `test_balance.py` 中，实现了 10000 步（10.0 秒）且稳态误差仅 $8.68\text{ mm}$ 的自平衡。
