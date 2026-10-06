"""
====================================================================================================
模块功能介绍 (Module Overview):
`rl/terrain/generate_terrain_xml.py`: 10.5m 复合恶劣赛道 MuJoCo 物理与视觉高保真 XML 场景构建器
====================================================================================================
1. 核心定位与自动化渲染：
   - 本模块基于解析高度场函数与几何航向参考，自动构建并输出 `wheel_leg_extended_terrain.xml`。
   - 保证物理仿真器 (MuJoCo) 中既有严谨的碰撞几何体 (hfield)，又有清晰鲜明的视觉贴地标线与障碍物指示：
     * 全赛道连续虚线中心线 (穿过全图与 S 弯道，清晰标明过弯中心航迹，双轮完美夹持中心线)
     * 每一个垫高单侧小板砖的高亮实体对比材质 (肉眼清晰可见车轮踏上颠簸)
     * 台阶式变速下坡的断崖边缘斑马纹警示排与侧向台阶护栏 (立体落差鲜明)
     * 沿赛道蜿蜒布置的高程跟随导向反光立柱
     * 10.0m 醒目终点线与终点立柱

2. 物理机构零篡改一致性保障：
   - 严格继承机器人经过官方动力学验证的原生机构模型 (mass, diaginertia, quat, 接触球摩擦系数与接触维数)，
     确保控制算法在真实动力学环境下得到严苛考验。
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
for p in [repo_dir, rl_dir, current_dir,
          os.path.join(rl_dir, "terrain")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from test_mega_terrain import (
    build_mega_hfield,
    get_mega_track_reference,
    OFF_LEFT_WHEEL,
    OFF_RIGHT_WHEEL,
    OFF_TRACK_MID
)


def generate():
    """
    自动计算并生成 10.5m 终极地狱级赛道 XML 场景文件。
    """
    x_len = 10.5
    y_len = 1.0
    ncol = 2560
    nrow = 256
    Z, X, Y = build_mega_hfield(x_len=x_len, y_len=y_len, nrow=nrow, ncol=ncol, max_elevation=0.045)

    def get_z_at(x_val, y_val=-0.0175):
        idx = int(np.clip(x_val / x_len * ncol, 0, ncol - 1))
        row = int(np.clip((y_val + 0.5) / y_len * nrow, 0, nrow - 1))
        return float(Z[row, idx])

    # 1. 全赛道连续虚线中心线 (x in [0.40, 9.80m], 贯穿全程，尤其是 S 弯道动态弯折曲线)
    centerline_geoms = []
    c_xs = np.arange(0.40, 9.80, 0.08)
    for cx in c_xs:
        y_ref, _ = get_mega_track_reference(cx, y0=-0.0175)
        y_mid = y_ref + OFF_TRACK_MID  # 左右轮正中线
        zc = get_z_at(cx, y_mid)
        # S 弯区域用琥珀黄/白色交替高亮，直道用纯白
        is_scurve = (2.20 <= cx <= 4.80)
        mat = "stripe_amber" if (is_scurve and int(cx / 0.16) % 2 == 1) else "stripe_white"
        centerline_geoms.append(
            f'    <geom type="box" size="0.025 0.0025 0.0003" pos="{cx:.3f} {y_mid:.4f} {zc + 0.0004:.4f}" material="{mat}" contype="0" conaffinity="0"/>'
        )

    # 2. 导向反光立柱 (沿蛇形 S 弯与直线段两侧动态边缘布置，间距 0.25m)
    post_geoms = []
    post_xs = np.arange(0.5, 10.2, 0.25)
    for px in post_xs:
        y_ref, _ = get_mega_track_reference(px, y0=-0.0175)
        yl = y_ref + 0.075
        yr = y_ref - 0.075
        pzl = get_z_at(px, yl)
        pzr = get_z_at(px, yr)
        # 左侧立柱 (深灰座身 + 亮黄色顶帽反光体)
        post_geoms.append(f'    <geom type="box" size="0.003 0.003 0.008" pos="{px:.2f} {yl:.4f} {pzl + 0.008:.4f}" rgba="0.30 0.33 0.38 1" contype="0" conaffinity="0"/>')
        post_geoms.append(f'    <geom type="box" size="0.0032 0.0032 0.002" pos="{px:.2f} {yl:.4f} {pzl + 0.015:.4f}" rgba="0.98 0.72 0.10 1" contype="0" conaffinity="0"/>')
        # 右侧立柱
        post_geoms.append(f'    <geom type="box" size="0.003 0.003 0.008" pos="{px:.2f} {yr:.4f} {pzr + 0.008:.4f}" rgba="0.30 0.33 0.38 1" contype="0" conaffinity="0"/>')
        post_geoms.append(f'    <geom type="box" size="0.0032 0.0032 0.002" pos="{px:.2f} {yr:.4f} {pzr + 0.015:.4f}" rgba="0.98 0.72 0.10 1" contype="0" conaffinity="0"/>')

    # 3. 垫高单侧小板砖的高亮视觉贴片 (精准覆盖左右车轮碾压轨迹，使车轮压砖肉眼极其明显)
    brick_caps = [
        (1.80, OFF_LEFT_WHEEL,  "Left Wheel Mogul-Exit Brick (+4.2mm)"),
        (2.02, OFF_RIGHT_WHEEL, "Right Wheel Mogul-Exit Brick (+4.2mm)"),
        (5.35, OFF_RIGHT_WHEEL, "Right Wheel Climb Brick (+4.2mm)"),
        (5.55, OFF_LEFT_WHEEL,  "Left Wheel Climb Brick (+4.2mm)"),
        (5.95, OFF_LEFT_WHEEL,  "Left Wheel Summit Brick (+4.5mm)"),
        (6.20, OFF_TRACK_MID,   "Summit Center Hurdle (+3.2mm)"),
        (6.45, OFF_RIGHT_WHEEL, "Right Wheel Summit Brick (+4.5mm)"),
        (6.75, OFF_LEFT_WHEEL,  "Left Wheel Summit Brick (+4.2mm)"),
        (7.48, OFF_RIGHT_WHEEL, "Right Wheel Shelf-Mid Brick (+4.0mm)"),
        (8.95, OFF_LEFT_WHEEL,  "Left Wheel Heavy Slab (+4.5mm)"),
        (9.15, OFF_RIGHT_WHEEL, "Right Wheel Heavy Slab (+4.5mm)"),
    ]
    brick_geoms = []
    for bx, by_off, bname in brick_caps:
        y_ref, _ = get_mega_track_reference(bx, y0=-0.0175)
        yb = y_ref + by_off
        zb = get_z_at(bx, yb)
        w_size = 0.035 if by_off == OFF_TRACK_MID else 0.010
        brick_geoms.append(
            f'    <!-- {bname} -->' + "\n" +
            f'    <geom type="box" size="0.022 {w_size:.4f} 0.0006" pos="{bx:.3f} {yb:.4f} {zb + 0.0005:.4f}" rgba="0.98 0.65 0.12 0.95" contype="0" conaffinity="0"/>'
        )

    # 4. 台阶式变速下坡强化：断崖边缘彩色斑马警示带与阶梯侧缘
    stepped_descent_markers = [
        (7.00, "stripe_amber", 0.035, "Step 1 Drop Edge (35mm)"),
        (7.28, "stripe_white", 0.018, "Halfway Shelf Landing (18mm)"),
        (7.68, "stripe_amber", 0.018, "Step 2 Drop Edge (18mm)"),
        (7.98, "stripe_white", 0.000, "Ground Flat Landing (0mm)"),
    ]
    step_geoms = []
    for sx, smat, sh, sdesc in stepped_descent_markers:
        sy, _ = get_mega_track_reference(sx, y0=-0.0175)
        ymid = sy + OFF_TRACK_MID
        sz = get_z_at(sx, ymid)
        step_geoms.append(
            f'    <!-- {sdesc} -->' + "\n" +
            f'    <geom type="box" size="0.012 0.080 0.0008" pos="{sx:.3f} {ymid:.4f} {sz + 0.0008:.4f}" material="{smat}" contype="0" conaffinity="0"/>'
        )

    # 5. 关键转折点的高亮防滑斑马警示带
    key_markers = [
        (0.55, "stripe_amber"),  # 递增巨幅波浪入口
        (2.20, "stripe_amber"),  # S 弯入口
        (2.85, "stripe_white"),  # S 弯第 1 弯心
        (3.50, "stripe_amber"),  # S 弯反向过渡点
        (4.15, "stripe_white"),  # S 弯第 2 弯心
        (4.80, "stripe_amber"),  # S 弯出口 / 35mm 爬坡坡脚
        (5.70, "stripe_amber"),  # 35mm 峰顶长平台入口
        (8.15, "stripe_white"),  # 坡底回归平地 / 7条搓板路入口
        (8.85, "stripe_white"),  # 搓板结束 / 冲刺终点直道
    ]
    marker_geoms = []
    for mx, mat in key_markers:
        my, _ = get_mega_track_reference(mx, y0=-0.0175)
        ymid = my + OFF_TRACK_MID
        mz = get_z_at(mx, ymid)
        marker_geoms.append(f'    <geom type="box" size="0.008 0.075 0.0005" pos="{mx:.2f} {ymid:.4f} {mz + 0.0006:.4f}" material="{mat}" contype="0" conaffinity="0"/>')

    # 6. 10.0m 终点线
    y_fin, _ = get_mega_track_reference(10.00, y0=-0.0175)
    y_fin_mid = y_fin + OFF_TRACK_MID
    z_fin = get_z_at(10.00, y_fin_mid)
    fin_geom = f'    <geom type="box" size="0.015 0.090 0.0006" pos="10.00 {y_fin_mid:.4f} {z_fin + 0.0007:.4f}" material="stripe_amber" contype="0" conaffinity="0"/>'

    # 7. 组合最终 XML
    xml_content = f"""<mujoco model="wheel_leg_extended_terrain">
  <compiler angle="radian" meshdir="../../car_urdf/meshes" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="Euler"/>

  <visual>
    <headlight ambient="0.32 0.34 0.38" diffuse="0.65 0.68 0.72" specular="0.25 0.25 0.25"/>
    <quality shadowsize="4096"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <geom friction="1.2 0.005 0.0001" condim="3" solref="0.002 1" solimp="0.99 0.999 0.0001 0.5 2"/>
  </default>

  <asset>
    <mesh name="base_link" file="base_link.STL"/>
    <mesh name="left_leg" file="left_leg.STL"/>
    <mesh name="left_wheel" file="left_wheel.STL"/>
    <mesh name="right_leg" file="right_leg.STL"/>
    <mesh name="right_wheel" file="right_wheel.STL"/>
    
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.13 0.18" rgb2="0.03 0.04 0.06" width="512" height="512"/>
    <texture name="asphalt" type="2d" builtin="checker" width="512" height="512" rgb1=".15 .17 .20" rgb2=".22 .25 .30"/>
    <material name="asphalt" texture="asphalt" texrepeat="170 16" reflectance="0.08"/>
    
    <material name="curb_mat" specular="0.3" shininess="0.4" rgba="0.50 0.53 0.58 1"/>
    <material name="stripe_white" specular="0.6" shininess="0.8" rgba="0.94 0.96 0.98 1"/>
    <material name="stripe_amber" specular="0.6" shininess="0.8" rgba="0.98 0.72 0.10 1"/>
    <material name="stone_mat" specular="0.3" shininess="0.3" rgba="0.44 0.46 0.48 1"/>

    <hfield name="terrain_hfield" nrow="256" ncol="2560" size="5.25 0.5 0.045 0.01"/>
  </asset>

  <worldbody>
    <!-- 定向柔和侧光投射阴影，强烈凸显坡道与弯道起伏 -->
    <light directional="true" pos="5.25 -2.0 3.2" dir="0.3 0.6 -1.2" castshadow="true" diffuse="0.98 0.98 1.0" specular="0.4 0.4 0.4"/>
    <light directional="true" pos="5.25 2.0 3.2" dir="-0.3 -0.6 -1.2" castshadow="false" diffuse="0.25 0.28 0.35" specular="0.1 0.1 0.1"/>

    <!-- 10.5m 连续实体起伏地形 -->
    <geom name="terrain" type="hfield" hfield="terrain_hfield" pos="5.25 0.0 0.0" material="asphalt"/>
    
    <!-- 边界延伸深色吸光地面 -->
    <geom name="ground_outer" type="plane" size="11 3 0.05" pos="5.25 0.0 -0.002" rgba="0.08 0.09 0.11 1"/>

    <!-- 全赛道中心虚线引导标线 (S弯与直道全覆盖) -->
{chr(10).join(centerline_geoms)}

    <!-- 赛道两侧高程跟随导向反光立柱 (x in [0.5, 10.2m]) -->
{chr(10).join(post_geoms)}

    <!-- 垫高单侧小板砖的高亮视觉贴片 -->
{chr(10).join(brick_geoms)}

    <!-- 台阶式变速下坡断崖边缘警示条 -->
{chr(10).join(step_geoms)}

    <!-- 关键关卡高亮警示带 -->
{chr(10).join(marker_geoms)}

    <!-- 10.0m 终点线停车带 -->
{fin_geom}
    <!-- 终点线两侧醒目警示柱 -->
    <geom type="box" size="0.005 0.005 0.022" pos="10.00 {y_fin_mid + 0.085:.4f} {z_fin + 0.022:.4f}" rgba="0.98 0.72 0.10 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.005 0.005 0.022" pos="10.00 {y_fin_mid - 0.085:.4f} {z_fin + 0.022:.4f}" rgba="0.98 0.72 0.10 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.004 0.004 0.006" pos="10.00 {y_fin_mid + 0.085:.4f} {z_fin + 0.006:.4f}" rgba="0.30 0.33 0.38 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.004 0.004 0.006" pos="10.00 {y_fin_mid - 0.085:.4f} {z_fin + 0.006:.4f}" rgba="0.30 0.33 0.38 1" contype="0" conaffinity="0"/>

    <!-- 官方标定高精机器人物理装配 (保持原汁原味，杜绝篡改) -->
    <body name="base_link" pos="0.08 0 0.058">
      <freejoint name="root"/>
      <inertial pos="-0.011138 -0.0175 0.001698" mass="0.16794" diaginertia="1.6308e-05 2.8107e-05 3.0344e-05"/>
      <geom type="mesh" rgba="0.85 0.45 0.15 1" mesh="base_link"/>
      
      <!-- Left Leg -->
      <body name="left_leg" pos="-0.0093975 0.0000 -0.0010809" quat="0.0159413 0.706926 0.706928 0.0159413">
        <inertial pos="-0.00175 -0.00163 0.018" quat="0.998982 0.0451138 0 0" mass="0.0087651" diaginertia="1.2889e-06 1.24217e-06 6.45868e-08"/>
        <joint name="left_hip_joint" range="-3.14 3.14" actuatorfrcrange="-10 10" axis="1 0 0" damping="0.001"/>
        <geom type="mesh" rgba="0.2 0.6 0.85 1" mesh="left_leg" contype="0" conaffinity="0"/>
        
        <!-- Left Wheel -->
        <body name="left_wheel" pos="-0.0015 -0.0036054 0.039837" quat="0.498403 0.501591 -0.498403 -0.501594">
          <inertial pos="0 0 -0.004057" quat="0.5 0.5 -0.5 0.5" mass="0.0022065" diaginertia="4.03259e-08 2.24677e-08 2.24677e-08"/>
          <joint name="left_wheel_joint" axis="0 0 -1" actuatorfrcrange="-10 10" damping="0.0001"/>
          <geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="left_wheel" contype="0" conaffinity="0"/>
          <geom type="sphere" size="0.008" friction="1.5 0.01 0.0001" rgba="0 0 0 0" group="3"/>
        </body>
      </body>

      <!-- Right Leg -->
      <body name="right_leg" pos="-0.0093975 -0.0315 -0.0010809" quat="0.0159413 0.706926 0.706928 0.0159413">
        <inertial pos="-0.00175 -0.00163 0.018" quat="0.998982 0.0451138 0 0" mass="0.0087651" diaginertia="1.2889e-06 1.24217e-06 6.45868e-08"/>
        <joint name="right_hip_joint" range="-3.14 3.14" actuatorfrcrange="-10 10" axis="1 0 0" damping="0.001"/>
        <geom type="mesh" rgba="0.2 0.85 0.45 1" mesh="right_leg" contype="0" conaffinity="0"/>
        
        <!-- Right Wheel -->
        <body name="right_wheel" pos="-0.002 -0.0036054 0.039837" quat="0.501591 -0.498403 0.501594 -0.498403">
          <inertial pos="0 0 -0.004057" quat="0.5 0.5 -0.5 0.5" mass="0.0022065" diaginertia="4.03259e-08 2.24677e-08 2.24677e-08"/>
          <joint name="right_wheel_joint" axis="0 0 1" actuatorfrcrange="-10 10" damping="0.0001"/>
          <geom type="mesh" rgba="0.15 0.15 0.15 1" mesh="right_wheel" contype="0" conaffinity="0"/>
          <geom type="sphere" size="0.008" friction="1.5 0.01 0.0001" rgba="0 0 0 0" group="3"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_hip_motor" joint="left_hip_joint" ctrlrange="-10 10"/>
    <motor name="left_wheel_motor" joint="left_wheel_joint" ctrlrange="-10 10"/>
    <motor name="right_hip_motor" joint="right_hip_joint" ctrlrange="-10 10"/>
    <motor name="right_wheel_motor" joint="right_wheel_joint" ctrlrange="-10 10"/>
  </actuator>

  <sensor>
    <framepos name="body_pos" objtype="body" objname="base_link"/>
    <framequat name="body_quat" objtype="body" objname="base_link"/>
    <framelinvel name="body_linvel" objtype="body" objname="base_link"/>
    <frameangvel name="body_angvel" objtype="body" objname="base_link"/>
    <jointpos name="left_hip_pos" joint="left_hip_joint"/>
    <jointvel name="left_hip_vel" joint="left_hip_joint"/>
    <jointpos name="right_hip_pos" joint="right_hip_joint"/>
    <jointvel name="right_hip_vel" joint="right_hip_joint"/>
    <jointpos name="left_wheel_pos" joint="left_wheel_joint"/>
    <jointvel name="left_wheel_vel" joint="left_wheel_joint"/>
    <jointpos name="right_wheel_pos" joint="right_wheel_joint"/>
    <jointvel name="right_wheel_vel" joint="right_wheel_joint"/>
  </sensor>
</mujoco>
"""
    out_path = os.path.join(rl_dir, "terrain", "wheel_leg_extended_terrain.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())
    print(f"[SUCCESS] 已生成 10.5m 终极地狱级赛道 XML: {out_path} (中心线虚线数: {len(centerline_geoms)}, 板砖贴片数: {len(brick_geoms)}, 台阶带数: {len(step_geoms)})")


if __name__ == "__main__":
    generate()
