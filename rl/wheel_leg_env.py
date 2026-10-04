"""
WheelLegRoughTerrainEnv: 基于 Gymnasium 规范的轮腿起伏盲走强化学习环境
===================================================================
1. 观察空间 (Observation Space): 
   - 盲走设定：无前方雷达扫描，全靠本体感知 (Proprioception)
   - 堆叠 3 帧观测 (单帧 22 维 -> 总计 66 维)，支持网络捕捉角加速度、接触冲击与接触趋势
   - 单帧特征:
     [0..3]   body orientation: sin(pitch), cos(pitch), sin(roll), cos(roll)
     [4..6]   body angular velocity: wx, wy, wz
     [7..9]   body linear velocity: vx, vy, vz
     [10..13] hip joint state: q_lh, dq_lh, q_rh, dq_rh
     [14..15] wheel velocity: dq_lw, dq_rw
     [16]     target velocity: v_cmd
     [17..18] prior controller cache: base_pitch_target, base_u_balance
     [19..22] last action: a_{t-1} (4-dim)

2. 动作空间 (Action Space):
   - 4 维连续动作，范围 [-1.0, 1.0]
   - a[0] -> delta_pitch: [-0.05, 0.05] rad
   - a[1] -> delta_hip:   [-0.12, 0.12] rad (对称高度升降)
   - a[2] -> delta_roll:  [-0.10, 0.10] rad (主动差动侧倾吸收)
   - a[3] -> k_scale:     [0.4, 1.6] (虚拟悬架刚度自适应因子)

3. 频率设计:
   - 物理仿真: 1000 Hz (dt = 0.001s)
   - 策略控制: 50 Hz (action_repeat = 20 steps)
"""
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# 保证能直接导入当前目录下的 prior_controller
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from prior_controller import PriorController, quat2rpy


class WheelLegRoughTerrainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, terrain_xml=None):
        super().__init__()
        self.render_mode = render_mode

        if terrain_xml is None:
            terrain_xml = "rl/terrain/wheel_leg_terrain.xml"
            if not os.path.exists(terrain_xml):
                terrain_xml = os.path.join(current_dir, "terrain", "wheel_leg_terrain.xml")
        
        self.model = mujoco.MjModel.from_xml_path(terrain_xml)
        self.data = mujoco.MjData(self.model)

        self.physics_dt = self.model.opt.timestep # 0.001s
        self.action_repeat = 20                   # 50 Hz 策略更新
        self.policy_dt = self.physics_dt * self.action_repeat # 0.02s

        # 核心先验控制器
        self.controller = PriorController()

        # 动作空间: 4 维
        # a[0]: delta_pitch ([-1, 1] -> [-0.05, 0.05] rad)
        # a[1]: delta_hip   ([-1, 1] -> [-0.12, 0.12] rad)
        # a[2]: delta_roll  ([-1, 1] -> [-0.10, 0.10] rad)
        # a[3]: k_scale     ([-1, 1] -> [0.4, 1.6])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 单帧观测维度: 23 维
        self.single_obs_dim = 23
        self.obs_history_len = 3
        total_obs_dim = self.single_obs_dim * self.obs_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        # 观测历史队列
        self.obs_history = np.zeros((self.obs_history_len, self.single_obs_dim), dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)

        # 任务目标与参数
        self.target_v = 0.16  # 巡航目标速度提升至 0.16 m/s (与 LQR 速度完全对齐)
        self.r_wheel = 0.008  # 车轮半径 8 mm
        self.max_episode_steps = 400  # 400 * 0.02s = 8.0s
        self.current_step = 0

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

        # 沿机身航向的前向速度
        forward_vel = body_linvel[0] * np.cos(yaw) + body_linvel[1] * np.sin(yaw)

        return {
            'x_pos': body_pos[0],
            'y_pos': body_pos[1],
            'z_pos': body_pos[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'roll_rate': body_angvel[0],
            'pitch_rate': body_angvel[1],
            'yaw_rate': body_angvel[2],
            'body_linvel': body_linvel,
            'forward_vel': forward_vel,
            'left_hip_pos': lh_pos,
            'left_hip_vel': lh_vel,
            'right_hip_pos': rh_pos,
            'right_hip_vel': rh_vel,
            'left_wheel_vel': lw_vel,
            'right_wheel_vel': rw_vel,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 初始微小扰动 (Domain Randomization 提升鲁棒性)
        if seed is not None:
            np.random.seed(seed)
        
        # 初始位置微扰
        self.data.qpos[0] = 0.08 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[1] = np.random.uniform(-0.005, 0.005)
        self.data.qpos[2] = 0.058 # 标称高度

        # 控制器重置 (设定赛道中线 y=-0.0175m)
        self.controller.reset(current_x=self.data.qpos[0], current_yaw=0.0, current_y=-0.0175)
        self.controller.set_target_velocity(self.target_v)

        # 沉降几个微步达到稳定接触
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        self.last_action.fill(0.0)

        sensors = self._extract_sensors()
        init_obs = self._get_single_obs(sensors)
        for i in range(self.obs_history_len):
            self.obs_history[i] = init_obs

        return self.obs_history.flatten(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # 动作解映射 (PRCC 专注单侧独立屈曲顺应与避震，锁定俯仰由先验闭环全权负责，彻底杜绝超速失控)
        delta_pitch = 0.0                       # 锁定俯仰残差，速度与纵向平衡完全由先验控制器闭环掌控
        delta_hip   = 0.0                       # 锁定标称直立基准
        delta_roll  = float(action[2] * 0.08)   # 单侧独立屈曲残差调谐 [-0.08, 0.08] rad = [-4.6°, 4.6°]
        k_scale     = float(1.0 + action[3] * 0.25) # 虚拟悬架刚度比 [0.75, 1.25]

        z_vel_penalty = 0.0

        # 以 1000Hz 执行底层闭环控制 (action_repeat 次)
        for _ in range(self.action_repeat):
            sensors = self._extract_sensors()

            # 底层先验控制器闭环解算
            actuators = self.controller.compute(
                sensors, dt=self.physics_dt,
                delta_pitch=delta_pitch, delta_hip=delta_hip,
                delta_roll=delta_roll, k_scale=k_scale,
                enable_compliance=True
            )

            # 驱动电机
            self.data.actuator('left_hip_motor').ctrl[0] = actuators['torque_left_hip']
            self.data.actuator('right_hip_motor').ctrl[0] = actuators['torque_right_hip']
            self.data.actuator('left_wheel_motor').ctrl[0] = actuators['torque_left_wheel']
            self.data.actuator('right_wheel_motor').ctrl[0] = actuators['torque_right_wheel']

            mujoco.mj_step(self.model, self.data)

            z_vel_penalty += sensors['body_linvel'][2]**2

        z_vel_penalty /= self.action_repeat

        # 获取结束时的传感器读数
        sensors = self._extract_sensors()
        current_obs = self._get_single_obs(sensors)

        # 更新历史帧队列
        self.obs_history = np.roll(self.obs_history, shift=-1, axis=0)
        self.obs_history[-1] = current_obs

        # 计算奖励 (Reward Shaping)
        # 1. 速度跟踪与平稳巡航奖励 (在 target_v 附近获得最高奖励)
        v_fwd = sensors['forward_vel']
        r_speed_track = 10.0 * np.exp(-30.0 * (v_fwd - self.target_v)**2)
        r_forward = 5.0 * np.clip(v_fwd / self.target_v, 0.0, 1.2)
        # 2. 存活奖励
        r_alive   = 2.0
        # 3. 赛道对中与直线巡线约束 (严厉惩罚偏离中线 y=-0.0175 与航向跑偏，杜绝撞击边缘护栏)
        y_dev = sensors['y_pos'] - (-0.0175)
        r_lateral = 60.0 * (y_dev**2)
        r_yaw     = 15.0 * (sensors['yaw']**2)
        # 4. 强力横滚与角速度抑制 (核心避震目标)
        r_roll    = 45.0 * (sensors['roll']**2)
        r_pitch   = 8.0 * (sensors['pitch']**2)
        r_angvel  = 0.5 * (sensors['roll_rate']**2 + sensors['pitch_rate']**2)
        # 5. 垂直颠簸惩罚
        r_z_bounce = 3.0 * z_vel_penalty
        # 6. 动作平滑惩罚 (杜绝高频锯齿折线)
        r_action_mag = 0.05 * np.sum(action**2)
        r_smooth     = 0.25 * np.sum((action - self.last_action)**2)

        reward = (r_speed_track + r_forward + r_alive 
                  - r_lateral - r_yaw
                  - r_roll - r_pitch - r_angvel 
                  - r_z_bounce - r_action_mag - r_smooth)

        self.last_action = action.copy()
        self.current_step += 1

        # 终止条件判定 (倒立摆摔倒 / 偏离赛道 / 越障完成)
        terminated = False
        if abs(sensors['pitch']) > np.radians(30) or abs(sensors['roll']) > np.radians(30):
            terminated = True
            reward -= 10.0  # 严重摔倒惩罚
        elif abs(sensors['y_pos'] - (-0.0175)) > 0.06: # 偏离赛道护栏边缘
            terminated = True
            reward -= 10.0
        elif sensors['x_pos'] >= 3.60: # 顺利越过全部障碍区完赛
            terminated = True
            reward += 10.0

        # 超时截断
        truncated = (self.current_step >= self.max_episode_steps)

        info = {
            'x': sensors['x_pos'],
            'roll_deg': np.degrees(sensors['roll']),
            'pitch_deg': np.degrees(sensors['pitch']),
            'slip': 0.0,
            'forward_vel': sensors['forward_vel']
        }

        return self.obs_history.flatten(), reward, terminated, truncated, info
