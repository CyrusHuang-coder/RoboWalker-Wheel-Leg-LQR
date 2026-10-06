# 轮腿机器人实验演进与资产索引清单 (Evaluation & Benchmark Index)

本目录系统收录了 RoboWalker 轮腿式双轮平衡机器人在复杂起伏地形上从**经典基线 (Pure LQR)** 到**先验顺应强化学习 (PRCC-RL)**、再到**滑移耦合能量储罐强化学习 (SCT-RRL)** 的全部梯度消融评测产物与 10.5m 终极挑战验证数据。

---

## 一、 算法演进与四阶段消融矩阵

| 实验阶段 | 赛道环境 | 对比受试算法 | 对应代码脚本与产物路径 | 核心验证目的与定量结论 |
| :---: | :---: | :---: | :--- | :--- |
| **Phase 0**<br>评测口径对齐 | 4.0m 标准赛道<br>( \in [1.8, 3.3]) | LQR<br>LQR + 手工顺应律<br>Fixed RL<br>Curriculum RL | 📂 legacy_archive/phase0_benchmarking_calibration/<br>• eval_phase0.py<br>• phase0_metrics.md | **口径对齐与 RL 存在必要性消融**：<br>1. 锁死统一巡航速度 0.16m/s，消除冲量不公；<br>2. 证明手工规则无法适应未知高度，RL 具备自适应滤波优势。 |
| **Phase 1**<br>基础残差顺应 | 4.0m 静态凸起<br>(4.8mm 单侧垄) | Pure LQR<br>vs<br>PRCC-RL | 📂 legacy_archive/phase1_4m_lqr_vs_prcc/<br>• eval_compare.py<br>• evaluation_comparison.png<br>• aseline_data.npz | **残差网络顺应机制消融**：<br>1. 机身最大侧倾角 Peak Roll 从 5.72° 压降至 4.65° (**改善 18.7%**)；<br>2. 垂直高度漂移削减 7.2%。 |
| **Phase 2**<br>随机课程泛化 | 4.0m 随机地形<br>(50轮蒙特卡洛) | Pure LQR<br>Fixed-Terrain RL<br>Curriculum RL | 📂 legacy_archive/phase2_4m_curriculum_mc/<br>• eval_top_tier.py<br>• 	op_tier_benchmark.png<br>• 	op_tier_metrics.md | **训练课程泛化性消融**：<br>1. 课程模型在未见地形车轮滑移降至 22.4mm/s；<br>2. 姿态波动方差收敛更窄，有效遏制单赛道过拟合。 |
| **Phase 3**<br>无源能量储罐 | 4.0m 复合起伏<br>(储罐微观动态) | Pure LQR<br>PRCC-RL<br>SCT-RRL (Ours) | 📂 legacy_archive/phase3_4m_sct_energy_tank/<br>• eval_sct.py<br>• sct_benchmark.png<br>• sct_tank_dynamics.png<br>• sct_metrics.md | **核心创新 SCT 机制消融**：<br>1. 能量储罐抑制无约束网络高频颤振，>10Hz 力矩能量**削减约 30%**；<br>2. 轮地真实切向滑移**降低约 15%**，兼顾吸震与防空转。 |
| **Current**<br>终极综合大考 | **10.5m 地狱赛道**<br>(S弯+台阶+搓板) | Pure LQR<br>PRCC-RL<br>SCT-RRL (Ours) | 📂 
l/recorders/<br>• 
ecord_tri_extended_comparison.py<br>• demo/wheel_leg_tri_comparison.gif | **大作业终极全任务交付验证**：<br>1. 8 大极限障碍区间（S弯、双阶变速下坡、错位石阶）；<br>2. 三算法同台竞技 0 倾覆，SCT-RRL 姿态平顺性最优。 |

---

## 二、 基础通用工具

- metrics_utils.py: 基于 MuJoCo 真实接触点几何的空间刚体运动学切向滑移计 (ContactSlipMeter) 与力矩高频能谱分析器 (high_freq_energy)。
- eval_baseline.py: 经典倒立摆 LQR 基线运行与标准化遥测数据提取工具。
