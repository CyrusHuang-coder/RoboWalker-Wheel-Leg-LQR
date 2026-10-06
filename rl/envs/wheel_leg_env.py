"""
wheel_leg_env.py: 轮腿自适应盲走强化学习标准环境 (Gymnasium Standard Environment)
========================================================================================
【模块功能介绍】
本模块封装了符合 OpenAI/Gymnasium 标准接口的轮腿机器人连续控制仿真环境。
面向复杂未知路面（高频搓板、非对称路坎、随机波浪与台阶），以“盲走（Blind Locomotion）”
为核心设定，完全依靠机体本体感知 (Proprioception) 实现自适应越障与平稳巡航：

1. 层次化混合控制动作空间 (Action Space):
   - 动作维度: 4 维连续动作，范围 [-1.0, 1.0]
   - a[0] -> delta_pitch: 目标俯仰角残差 (锁定由底层闭环控制)
   - a[1] -> delta_hip:   标称高度基准锁定 (Option A)
   - a[2] -> delta_roll:  [-0.10, 0.10] rad 差动防侧倾髋角残差 (用于吸收单侧路面颠簸)
   - a[3] -> k_scale:     [0.4, 1.6] 虚拟悬架刚度自适应神经调制因子 (刚柔动态切换)

2. 多帧堆叠本体感知观测空间 (Observation Space):
   - 单帧 23 维特征 (含姿态三角特征、角速度、线速度、关节位置/速度、目标航速与先验控制器缓存)
   - 可选 +3 维储罐感知特征 (归一化储能、动作放行比 alpha*、滤波打滑功率)
   - 堆叠 3 帧时序观测 (总计 69 或 78 维)，提供马尔可夫决策所需的二阶微分加速度与冲量趋势信息。

3. 物理域随机化与程序化地形注入 (Domain Randomization & In-Memory Heightfields):
   - 步步支持随机摩擦系数 (0.8~1.6)、机身质量扰动 (±15%)、传感器高斯白噪声注入；
   - 支持课程管理器 (CurriculumManager) 动态调度地形起伏难度 (0.0 ~ 1.0)。
========================================================================================
"""
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# 路径自适应引导：无论从何处运行，均能正确载入底层控制库与工具
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(rl_dir, ".."))
for p in [repo_dir, rl_dir, current_dir,
         os.path.join(rl_dir, "controllers"),
         os.path.join(rl_dir, "terrain"),
         os.path.join(rl_dir, "evaluation"),
         os.path.join(rl_dir, "training")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prior_controller import PriorController, quat2rpy
from procedural_terrain import ProceduralTerrainGenerator
from energy_tank import SlipCoupledEnergyTank
from metrics_utils import ContactSlipMeter


class WheelLegRoughTerrainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, terrain_xml=None, 
                 enable_random_terrain=True, 
                 fixed_difficulty=None, 
                 curriculum_manager=None,
                 domain_randomization=True,
                 use_tank=False, slip_coupling=False, tank_obs=False,
                 obs_history_len=3, lambda_clip=0.5, tank_kwargs=None):
        super().__init__()
        self.render_mode = render_mode
        self.enable_random_terrain = enable_random_terrain
        self.fixed_difficulty = fixed_difficulty
        self.curriculum_manager = curriculum_manager
        self.domain_randomization = domain_randomization

        # 地形模型资源加载
        if terrain_xml is None:
            terrain_xml = os.path.join(rl_dir, "terrain", "wheel_leg_terrain.xml")
            if not os.path.exists(terrain_xml):
                terrain_xml = "rl/terrain/wheel_leg_terrain.xml"

        self.model = mujoco.MjModel.from_xml_path(terrain_xml)
        self.data = mujoco.MjData(self.model)

        # 记录标称物理属性 (用于域随机化复位)
        self.nominal_base_mass = float(self.model.body('base_link').mass[0])
        self.nominal_ground_friction = float(self.model.geom('terrain').friction[0])

        self.physics_dt = self.model.opt.timestep  # 0.001s (1kHz 动力学解算)
        self.action_repeat = 20                    # 50 Hz 强化学习策略更新
        self.policy_dt = self.physics_dt * self.action_repeat  # 0.02s

        # 程序化地形生成器
        self.terrain_gen = ProceduralTerrainGenerator()

        # 核心先验控制器
        self.controller = PriorController()

        # 动作空间: 4 维连续动作
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 单帧观测维度: 23 维 (+3 维储罐感知特征, 可选)
        self.tank_obs = bool(tank_obs and use_tank)
        self.single_obs_dim = 23 + (3 if self.tank_obs else 0)
        self.obs_history_len = int(obs_history_len)
        total_obs_dim = self.single_obs_dim * self.obs_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        # 滑移耦合能量储罐 (SCT)
        self.use_tank = bool(use_tank)
        self.lambda_clip = float(lambda_clip)
        self.tank = SlipCoupledEnergyTank(slip_coupling=slip_coupling, **(tank_kwargs or {})) if self.use_tank else None
        self.slip_meter = ContactSlipMeter(self.model)

        # 观测历史队列
        self.obs_history = np.zeros((self.obs_history_len, self.single_obs_dim), dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)

        # 任务目标与参数
        self.target_v = 0.16     # 巡航目标速度 0.16 m/s
        self.r_wheel = 0.008     # 车轮半径 8 mm
        self.L_leg = 0.040       # 腿连杆长度 40 mm
        self.max_episode_steps = 1400  # 1400 * 0.02s = 28.0s
        self.current_step = 0
        self.current_difficulty = 0.0

    def set_curriculum_manager(self, cm):
        self.curriculum_manager = cm

    def set_difficulty(self, diff):
        self.fixed_difficulty = diff

    def _get_single_obs(self, sensors):
        roll = sensors['roll']
        pitch = sensors['pitch']
        obs = np.array([
            np.sin(pitch), np.cos(pitch),
            np.sin(roll),  np.cos(roll),
            sensors['roll_rate'] * 0.2,
            sensors['pitch_rate'] * 0.2,
            sensors['yaw_rate'] * 0.2,
            sensors['forward_vel'] * 5.0,
            sensors['body_linvel'][1] * 5.0,
            sensors['body_linvel'][2] * 5.0,
            sensors['left_hip_pos'],
            sensors['left_hip_vel'] * 0.1,
            sensors['right_hip_pos'],
            sensors['right_hip_vel'] * 0.1,
            sensors['left_wheel_vel'] * 0.05,
            sensors['right_wheel_vel'] * 0.05,
            self.target_v * 5.0,
            self.controller.last_target_pitch * 5.0,
            self.controller.last_u_balance * 20.0,
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
            self.last_action[3]
        ], dtype=np.float32)
        if self.tank_obs:
            obs = np.concatenate([obs, self.tank.features()]).astype(np.float32)

        # 传感器高斯白噪声注入 (提高 Sim-to-Real 鲁棒性)
        if self.domain_randomization:
            noise = np.random.normal(0.0, 0.004, size=obs.shape).astype(np.float32)
            obs += noise
        return obs

    def _extract_sensors(self):
        body_quat = self.data.sensor('body_quat').data
        roll, pitch, yaw = quat2rpy(body_quat)
        body_angvel = self.data.sensor('body_angvel').data
        body_linvel = self.data.sensor('body_linvel').data
        body_pos = self.data.sensor('body_pos').data

        lh_pos = self.data.sensor('left_hip_pos').data[0]
        lh_vel = self.data.sensor('left_hip_vel').data[0]
        rh_pos = self.data.sensor('right_hip_pos').data[0]
        rh_vel = self.data.sensor('right_hip_vel').data[0]
        lw_vel = self.data.sensor('left_wheel_vel').data[0]
        rw_vel = self.data.sensor('right_wheel_vel').data[0]

        forward_vel = body_linvel[0] * np.cos(yaw) + body_linvel[1] * np.sin(yaw)

        return {
            'x_pos': body_pos[0],
            'y_pos': body_pos[1],
            'z_pos': body_pos[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'forward_vel': forward_vel,
            'body_linvel': body_linvel,
            'roll_rate': body_angvel[0],
            'pitch_rate': body_angvel[1],
            'yaw_rate': body_angvel[2],
            'left_hip_pos': lh_pos,
            'left_hip_vel': lh_vel,
            'right_hip_pos': rh_pos,
            'right_hip_vel': rh_vel,
            'left_wheel_vel': lw_vel,
            'right_wheel_vel': rw_vel
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 1. 确定当前地形难度
        if self.fixed_difficulty is not None:
            self.current_difficulty = float(self.fixed_difficulty)
        elif self.curriculum_manager is not None:
            self.current_difficulty = float(self.curriculum_manager.get_difficulty())
        else:
            self.current_difficulty = 0.0

        # 2. 内存级注入随机地形高度图
        if self.enable_random_terrain:
            self.terrain_gen.update_model_hfield(self.model, difficulty=self.current_difficulty)

        # 3. 物理域随机化 (Domain Randomization)
        if self.domain_randomization:
            # 质量扰动 ±15%
            mass_factor = np.random.uniform(0.85, 1.15)
            self.model.body('base_link').mass[0] = self.nominal_base_mass * mass_factor
            # 地面摩擦系数扰动 (0.8 ~ 1.5)
            fric_factor = np.random.uniform(0.8, 1.4)
            self.model.geom('terrain').friction[0] = self.nominal_ground_friction * fric_factor
        else:
            self.model.body('base_link').mass[0] = self.nominal_base_mass
            self.model.geom('terrain').friction[0] = self.nominal_ground_friction

        # 4. 重置底层动力学状态
        mujoco.mj_resetData(self.model, self.data)

        # 初始微扰动放置 (平地平稳着陆点)
        self.data.qpos[0] = 0.08
        self.data.qpos[1] = 0.0  # 双轮几何中心对准赛道中心线 -0.0175
        self.data.qpos[2] = 0.060
        if self.domain_randomization:
            self.data.qpos[0] += np.random.uniform(-0.01, 0.01)

        # 预热沉降，平稳下落
        for _ in range(80):
            mujoco.mj_step(self.model, self.data)

        # 5. 重置先验控制器
        init_x = self.data.sensor('body_pos').data[0]
        self.controller.reset(current_x=init_x, current_yaw=0.0, current_y=-0.0175)
        self.controller.set_target_velocity(self.target_v)

        # 6. 重置能量储罐
        if self.tank is not None:
            self.tank.reset()

        self.current_step = 0
        self.last_action = np.zeros(4, dtype=np.float32)

        sensors = self._extract_sensors()
        single_obs = self._get_single_obs(sensors)
        self.obs_history = np.tile(single_obs, (self.obs_history_len, 1))

        return self.obs_history.flatten(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # 动作解析与物理映射
        # a[0] -> delta_pitch (由先验环路接管，锁定 0.0)
        # a[1] -> delta_hip   (锁定 0.0)
        # a[2] -> delta_roll  [-0.10, +0.10] rad 差动防侧倾
        # a[3] -> k_scale     [0.60, 1.40] 虚拟刚度调制
        delta_pitch = 0.0
        delta_hip   = 0.0
        delta_roll_req = float(action[2] * 0.10)
        k_scale_req    = float(1.0 + action[3] * 0.40)

        # SCT 能量储罐二分安全投影
        if self.tank is not None:
            s_proj = self._extract_sensors()
            k_scale, delta_roll, alpha_star, dW = self.tank.project(self.controller, s_proj, k_scale_req, delta_roll_req)
        else:
            k_scale, delta_roll = k_scale_req, delta_roll_req
            alpha_star, dW = 1.0, 0.0

        # 执行 20 次 1kHz 物理步 (总计 20ms)
        slip_power_acc = 0.0
        z_vel_acc = 0.0
        for _ in range(self.action_repeat):
            sensors = self._extract_sensors()

            ctrl_cmd = self.controller.compute(
                sensors,
                dt=self.physics_dt,
                delta_pitch=delta_pitch,
                delta_hip=delta_hip,
                delta_roll=delta_roll,
                k_scale=k_scale,
                enable_compliance=True
            )

            self.data.actuator('left_hip_motor').ctrl[0] = ctrl_cmd['torque_left_hip']
            self.data.actuator('right_hip_motor').ctrl[0] = ctrl_cmd['torque_right_hip']
            self.data.actuator('left_wheel_motor').ctrl[0] = ctrl_cmd['torque_left_wheel']
            self.data.actuator('right_wheel_motor').ctrl[0] = ctrl_cmd['torque_right_wheel']

            mujoco.mj_step(self.model, self.data)

            # 接触滑移功率监测
            sl, sr, _, _ = self.slip_meter.measure(self.data)
            p_slip = (abs(ctrl_cmd['torque_left_wheel']) * sl + abs(ctrl_cmd['torque_right_wheel']) * sr) / self.r_wheel
            slip_power_acc += p_slip
            z_vel_acc += abs(sensors['body_linvel'][2])

            if self.tank is not None:
                self.tank.accumulate(self.controller, sensors, p_slip, self.physics_dt)

        if self.tank is not None:
            self.tank.commit()

        # 刷新观测队列
        sensors = self._extract_sensors()
        single_obs = self._get_single_obs(sensors)
        self.obs_history = np.roll(self.obs_history, shift=-1, axis=0)
        self.obs_history[-1] = single_obs

        # 多目标稠密奖励函数计算
        v_err = sensors['forward_vel'] - self.target_v
        r_speed_track = 6.0 * np.exp(-12.0 * (v_err ** 2))
        r_forward = 3.5 * np.clip(sensors['forward_vel'], 0.0, 0.3)
        r_alive   = 2.0

        y_dev = sensors['y_pos'] - (-0.0175)
        r_lateral = 35.0 * (y_dev**2)
        r_yaw     = 30.0 * (sensors['yaw']**2) + 2.5 * (sensors['yaw_rate']**2)
        r_roll    = 85.0 * (sensors['roll']**2) + 1.5 * (sensors['roll_rate']**2)
        r_pitch   = 8.0 * (sensors['pitch']**2)

        z_vel_penalty = z_vel_acc / self.action_repeat
        r_z_bounce = 8.0 * z_vel_penalty

        slip_penalty = slip_power_acc / self.action_repeat
        r_slip = 2.0 * slip_penalty

        # 平地静止零偏置死区硬锚定 (防无端多动)
        if abs(sensors['roll']) < np.radians(0.4):
            r_flat_zero = 30.0 * (delta_roll**2)
        else:
            r_flat_zero = 0.0

        r_action_mag = 0.15 * np.sum(action**2)
        r_smooth     = 0.60 * np.sum((action - self.last_action)**2)
        r_clip = self.lambda_clip * (1.0 - alpha_star) if self.tank is not None else 0.0

        reward = (r_speed_track + r_forward + r_alive 
                  - r_lateral - r_yaw
                  - r_roll - r_pitch 
                  - r_z_bounce - r_slip - r_flat_zero
                  - r_action_mag - r_smooth - r_clip)

        self.last_action = action.copy()
        self.current_step += 1

        # 终止条件判定
        terminated = False
        success = False
        if abs(sensors['pitch']) > np.radians(30) or abs(sensors['roll']) > np.radians(30):
            terminated = True
            reward -= 12.0  # 摔倒惩罚
            if self.curriculum_manager:
                self.curriculum_manager.record_episode(False)
        elif abs(sensors['y_pos'] - (-0.0175)) > 0.085:  # 偏离赛道
            terminated = True
            reward -= 10.0
            if self.curriculum_manager:
                self.curriculum_manager.record_episode(False)
        elif sensors['x_pos'] >= 3.55:  # 顺利越过全部障碍区完赛
            terminated = True
            success = True
            reward += 12.0
            if self.curriculum_manager:
                self.curriculum_manager.record_episode(True)

        truncated = (self.current_step >= self.max_episode_steps)

        info = {
            'x': sensors['x_pos'],
            'roll_deg': np.degrees(sensors['roll']),
            'pitch_deg': np.degrees(sensors['pitch']),
            'slip': np.sqrt(slip_penalty),
            'forward_vel': sensors['forward_vel'],
            'success': success,
            'difficulty': self.current_difficulty,
            'alpha_star': alpha_star,
            'dW': dW,
            'E_T': self.tank.E if self.tank is not None else 0.0,
        }

        return self.obs_history.flatten(), reward, terminated, truncated, info
