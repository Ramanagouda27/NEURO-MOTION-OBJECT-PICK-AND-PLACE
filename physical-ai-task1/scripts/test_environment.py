#!/usr/bin/env python3
"""
Quick test to verify MuJoCo + SO101 scene loads correctly.
Run this first to check your setup before running the full pipeline.
"""

import os
import sys
import numpy as np

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import mujoco
        print(f"  mujoco:    OK (v{mujoco.__version__})")
    except ImportError:
        print("  mujoco:    MISSING - pip install mujoco")
        return False
    
    try:
        import numpy as np
        print(f"  numpy:     OK (v{np.__version__})")
    except ImportError:
        print("  numpy:     MISSING - pip install numpy")
        return False
    
    try:
        import cv2
        print(f"  opencv:    OK (v{cv2.__version__})")
    except ImportError:
        print("  opencv:    MISSING (optional) - pip install opencv-python-headless")
    
    return True


def test_scene_loading():
    """Test that the MuJoCo scene loads correctly."""
    print("\nTesting scene loading...")
    
    import mujoco
    
    scene_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "assets", "so101_scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"  FAILED: Scene file not found at {scene_path}")
        return False
    
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        print(f"  Scene loaded successfully!")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  Actuators: {model.nu}")
        print(f"  Sensors: {model.nsensor}")
        print(f"  Timestep: {model.opt.timestep}s")
    except Exception as e:
        print(f"  FAILED to load scene: {e}")
        return False
    
    return True


def test_simulation():
    """Run a quick simulation test."""
    print("\nTesting simulation (1000 steps)...")
    
    import mujoco
    
    scene_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "assets", "so101_scene.xml")
    
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    
    for i in range(1000):
        mujoco.mj_step(model, data)
    
    print(f"  Simulation time: {data.time:.3f}s")
    print(f"  Gripper position: {data.sensordata[7:10]}")
    print(f"  Object position:  {data.sensordata[10:13]}")
    print(f"  Target position:  {data.sensordata[13:16]}")
    
    # Check object hasn't fallen through the floor
    obj_z = data.sensordata[12]
    if obj_z > 0.3:
        print(f"  Object stable on table (z={obj_z:.3f}) - OK")
    else:
        print(f"  WARNING: Object may have fallen (z={obj_z:.3f})")
    
    return True


def test_rendering():
    """Test headless rendering."""
    print("\nTesting headless rendering...")
    
    import mujoco
    
    scene_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "assets", "so101_scene.xml")
    
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    try:
        renderer = mujoco.Renderer(model, 480, 640)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="front_cam")
        frame = renderer.render()
        print(f"  Rendered frame: {frame.shape} dtype={frame.dtype}")
        
        if frame.sum() > 0:
            print(f"  Frame has content - OK")
        else:
            print(f"  WARNING: Frame appears blank")
        
        return True
    except Exception as e:
        print(f"  Rendering failed: {e}")
        print(f"  Try: export MUJOCO_GL=egl  (for headless)")
        print(f"  Or:  export MUJOCO_GL=glx  (for X11)")
        return False


def main():
    print("="*50)
    print("  SO101 Environment Test Suite")
    print("="*50)
    
    results = {}
    results['imports'] = test_imports()
    
    if results['imports']:
        results['scene'] = test_scene_loading()
        results['simulation'] = test_simulation()
        results['rendering'] = test_rendering()
    
    print("\n" + "="*50)
    print("  TEST RESULTS")
    print("="*50)
    
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:15s} : {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n  All tests passed! Ready to run Task 1.")
        print("  Next: python scripts/task1_pick_and_place.py --render")
    else:
        print("\n  Some tests failed. Fix the issues above first.")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
