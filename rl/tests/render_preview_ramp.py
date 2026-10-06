"""
render_preview_ramp.py: 渲染 12mm 宏观爬坡与实验室背景墙预览图
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
    norm_hf = (Z / 0.016).astype(np.float32)
    model.hfield_data[:] = norm_hf.ravel()

    data = mujoco.MjData(model)

    # 放置在 12mm 爬坡坡面上 (x=2.90m)
    data.qpos[0] = 2.90
    data.qpos[2] = 0.075
    for _ in range(100):
        mujoco.mj_step(model, data)

    w, h = 640, 480
    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.35
    cam.elevation = -12.0
    cam.azimuth = 142.0
    cam.lookat = [data.sensor('body_pos').data[0] + 0.04, -0.0175, 0.050]

    renderer.update_scene(data, camera=cam)
    img = Image.fromarray(renderer.render())
    
    out_path = "C:/Users/lenovo/.gemini/antigravity/brain/7b83f22e-ed2b-45ef-8cda-230cc1568684/frame_preview_ramp12mm.png"
    img.save(out_path)
    print(f"Saved ramp preview frame to {out_path}")

if __name__ == "__main__":
    main()
