import mujoco

with open('wheel_leg.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

xml_stiff = xml.replace('condim="3"', 'condim="3" solref="0.002 1" solimp="0.99 0.999 0.0001 0.5 2"')

m = mujoco.MjModel.from_xml_string(xml_stiff)
d = mujoco.MjData(m)
d.qpos[2] = 0.049
d.qpos[3] = 1.0
d.qpos[7] = -0.0042
d.qpos[9] = -0.0042

for _ in range(500):
    mujoco.mj_step(m, d)

print(f"Stiff contact: wheel_z = {d.xpos[3, 2]:.6f} m, bottom_z = {d.xpos[3, 2] - 0.008:.6f} m")
print(f"Penetration: {-(d.xpos[3, 2] - 0.008)*1000:.3f} mm")
