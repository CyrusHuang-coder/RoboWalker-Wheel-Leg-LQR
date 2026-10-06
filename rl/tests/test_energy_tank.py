import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prior_controller import PriorController
from energy_tank import SlipCoupledEnergyTank


def make_s(rng=None, roll=0.0, ql=None, qr=None):
    c = PriorController()
    q0 = c.q_hip_nominal
    if rng is not None:
        return {'roll': rng.uniform(-0.1, 0.1), 'roll_rate': rng.uniform(-2, 2),
                'left_hip_pos': q0 + rng.uniform(-0.3, 0.3), 'right_hip_pos': q0 + rng.uniform(-0.3, 0.3),
                'left_hip_vel': rng.uniform(-3, 3), 'right_hip_vel': rng.uniform(-3, 3)}
    return {'roll': roll, 'roll_rate': 0.0,
            'left_hip_pos': q0 + (ql or 0.0), 'right_hip_pos': q0 + (qr or 0.0),
            'left_hip_vel': 0.0, 'right_hip_vel': 0.0}


def test_full_tank_passes_action():
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=1.0, E_max=1.0)
    k, dr, a, dW = t.project(c, make_s(ql=-0.1, qr=-0.1), 1.4, 0.05)
    assert a == 1.0 and abs(k - 1.4) < 1e-12 and abs(dr - 0.05) < 1e-12


def test_empty_tank_blocks_stiffening():
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=0.0)
    k, dr, a, dW = t.project(c, make_s(ql=-0.1, qr=-0.1), 1.4, 0.0)
    assert a == 0.0 and k == 1.0 and dW <= 1e-15


def test_softening_always_allowed_and_refills():
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=0.0, E_max=1.0)
    k, dr, a, dW = t.project(c, make_s(ql=-0.1, qr=-0.1), 0.6, 0.0)
    assert a == 1.0 and dW < 0 and t.E > 0


def test_random_invariant():
    rng = np.random.default_rng(0)
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=1e-4, E_max=4e-4)
    injected, recharged = 0.0, 0.0
    E0 = t.E
    for _ in range(10000):
        s = make_s(rng)
        E_before = t.E
        _, _, _, dW = t.project(c, s, rng.uniform(0.6, 1.4), rng.uniform(-0.1, 0.1))
        assert dW <= E_before - t.E_min + 1e-12
        assert t.E >= t.E_min - 1e-15
        injected += dW
        for _ in range(20):
            e_pre = t.E
            t.accumulate(c, s, rng.uniform(0, 0.05), 0.001)
            recharged += t.E - e_pre
        t.commit()
    # 累计净注能不超过初始储能 + 净回充 (clip 只会让上限更紧)
    assert injected <= E0 - t.E_min + recharged + 1e-9


def test_restoring_exemption_when_empty():
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=0.0, E_min=0.0, E_max=4e-4)
    # 模拟先前的非对称偏移偏置
    t.k_prev = 1.3
    t.dr_prev = 0.05
    s = make_s(ql=-0.05, qr=0.05)
    # 策略请求复归零位 (k=1.0, dr=0.0)
    k, dr, a, dW = t.project(c, s, 1.0, 0.0)
    assert a == 1.0, f"Restoring should be unconditionally allowed, got alpha={a}"
    assert abs(k - 1.0) < 1e-6 and abs(dr - 0.0) < 1e-6, "Must reach neutral posture"
    assert t.E >= t.E_min - 1e-15


def test_slip_deadband():
    c = PriorController()
    t = SlipCoupledEnergyTank(E_init=2e-4, E_min=0.0, E_max=4e-4, p_slip_deadband=0.015)
    s = make_s(ql=0.0, qr=0.0)  # 速度为 0，无阻尼耗散
    E_start = t.E
    # 正常平地滚动微滑移 (10 mW < 15 mW 死区)
    for _ in range(100):
        t.accumulate(c, s, 0.010, 0.001)
    assert abs(t.E - E_start) < 1e-12, "Micro-slip below deadband must NOT drain the tank!"

    # 障碍剧烈打滑 (30 mW > 15 mW 死区)
    for _ in range(100):
        t.accumulate(c, s, 0.030, 0.001)
    assert t.E < E_start, "Excess slip above deadband MUST drain the tank!"
