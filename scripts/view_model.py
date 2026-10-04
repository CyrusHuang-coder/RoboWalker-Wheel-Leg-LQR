import os

def find_xml_path(name="wheel_leg.xml"):
    for p in [name, os.path.join("..", name), os.path.join(os.path.dirname(__file__), "..", name), os.path.join(os.path.dirname(__file__), name)]:
        if os.path.exists(p):
            return p
    return name

def main():
    xml_path = find_xml_path("wheel_leg.xml")
    print(f"Loading {xml_path} into MuJoCo Viewer...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Opening interactive viewer window...")
    print("Tip: Double click on any body and right-click drag to apply forces!")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Maintain real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
