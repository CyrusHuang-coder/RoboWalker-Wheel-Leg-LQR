"""
render_preview_frame.py: 渲染测试帧并检查视觉质感与车轮着地情况
"""
import os
import sys
import numpy as np
import mujoco
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from test_extended_hfield import build_extended_hfield

def main():
    xml_path = "rl/terrain/wheel_leg_extended_terrain.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    Z, X, Y = build_extended_hfield()
    norm_hf = (Z / 0.0065).astype(np.float32)
    model.hfield_data[:] = norm_hf.ravel()

    data = mujoco.MjData(model)

    # 将机器人放置在大平台上 (x=6.05m) 检验贴地与视觉
    data.qpos[0] = 6.05
    data.qpos[2] = 0.065
    for _ in range(100):
        mujoco.mj_step(model, data)

    w, h = 640, 480
    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.32
    cam.elevation = -13.0
    cam.azimuth = 142.0
    cam.lookat = [data.sensor('body_pos').data[0] + 0.04, -0.0175, 0.042]

    renderer.update_scene(data, camera=cam)
    img = Image.fromarray(renderer.render())
    
    out_path = "C:/Users/lenovo/.gemini/antigravity/brain/7b83f22e-ed2b-45ef-8cda-230cc1568684/frame_preview_plateau.png"
    img.save(out_path)
    print(f"Saved plateau preview frame to {out_path}")

    # 搓板路区域 (x=3.70m)
    data.qpos[0] = 3.70
    data.qpos[2] = 0.062
    for _ in range(100):
        mujoco.mj_step(model, data)
    cam.lookat = [data.sensor('body_pos').data[0] + 0.04, -0.0175, 0.042]
    renderer.update_scene(data, camera=cam)
    img_rumble = Image.fromarray(renderer.render())
    out_rumble = "C:/Users/lenovo/.gemini/antigravity/brain/7b83f22e-ed2b-45ef-8cda-230cc1568684/frame_preview_rumble.png"
    img_rumble.save(out_rumble)
    print(f"Saved rumble preview frame to {out_rumble}")

    # 错位石阶区域 (x=4.70m)
    data.qpos[0] = 4.70
    data.qpos[2] = 0.065
    for _ in range(100):
        mujoco.mj_step(model, data)
    cam.lookat = [data.sensor('body_pos').data[0] + 0.04, -0.0175, 0.042]
    renderer.update_scene(data, camera=cam)
    img_slab = Image.fromarray(renderer.render())
    out_slab = "C:/Users/lenovo/.gemini/antigravity/brain/7b83f22e-ed2b-45ef-8cda-230cc1568684/frame_preview_slab.png"
    img_slab.save(out_slab)
    print(f"Saved slab preview frame to {out_slab}")

if __name__ == "__main__":
    main()
