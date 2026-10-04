import sys
sys.path.append("LQR计算代码")
import calculate
import parameter as p
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("wheel_leg.xml")
data = mujoco.MjData(model)

K = calculate.calculate(0.04045, 0.04045)

def quat2rpy(q):
    w, x, y, z = q
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return pitch, yaw

# Test sign combinations for actuators
# K columns: [x, dx, y, dy, tl, dtl, tr, dtr, f, df]
# K rows: [Tlw, Tll, Trw, Trl]

def simulate_with_signs(sign_wheel, sign_hip, sign_state_pitch):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.0485  # exact ground contact height
    # Small pitch perturbation: 0.03 rad (1.7 deg)
    pitch_init = 0.03
    data.qpos[3] = np.cos(pitch_init / 2.0)
    data.qpos[5] = np.sin(pitch_init / 2.0)
    mujoco.mj_forward(model, data)

    r = p.r
    pitches = []
    
    for step in range(1000): # 1.0 second
        pitch, yaw = quat2rpy(data.qpos[3:7])
        d_pitch = data.qvel[4]
        d_yaw = data.qvel[5]

        q_lhip = data.qpos[7]
        q_lwheel = data.qpos[8]
        q_rhip = data.qpos[9]
        q_rwheel = data.qpos[10]

        dq_lhip = data.qvel[6]
        dq_lwheel = data.qvel[7]
        dq_rhip = data.qvel[8]
        dq_rwheel = data.qvel[9]

        # Invert wheel rotation sign if needed
        x = r * (q_lwheel + q_rwheel) / 2.0 * sign_wheel
        dx = r * (dq_lwheel + dq_rwheel) / 2.0 * sign_wheel

        # Pitch signed
        f = pitch * sign_state_pitch
        df = d_pitch * sign_state_pitch

        tl = f + q_lhip * sign_hip
        dtl = df + dq_lhip * sign_hip
        tr = f + q_rhip * sign_hip
        dtr = df + dq_rhip * sign_hip

        state = np.array([x, dx, yaw, d_yaw, tl, dtl, tr, dtr, f, df])
        u = -K @ state

        # Apply to motors
        # Tlw -> ctrl[1], Tll -> ctrl[0], Trw -> ctrl[3], Trl -> ctrl[2]
        data.ctrl[0] = u[1] * sign_hip
        data.ctrl[1] = u[0] * sign_wheel
        data.ctrl[2] = u[3] * sign_hip
        data.ctrl[3] = u[2] * sign_wheel

        mujoco.mj_step(model, data)
        pitches.append(abs(pitch))

        if abs(pitch) > 0.5: # fell over
            return False, step, max(pitches)

    return True, 1000, pitches[-1]

for sw in [1, -1]:
    for sh in [1, -1]:
        for sp in [1, -1]:
            ok, steps, final_p = simulate_with_signs(sw, sh, sp)
            if ok:
                print(f"SUCCESS! sw={sw}, sh={sh}, sp={sp}, steps={steps}, final pitch={final_p:.4f} rad")
            else:
                pass
