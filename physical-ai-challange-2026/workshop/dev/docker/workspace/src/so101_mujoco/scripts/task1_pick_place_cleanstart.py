#!/usr/bin/env python3
import json
import math
import socket
import time

BRIDGE_IP = '127.0.0.1'
BRIDGE_PORT = 9876

JOINT_LIMITS = [
    (-1.91986,  1.91986),
    (-1.74533,  1.74533),
    (-1.69000,  1.69000),
    (-1.65806,  1.65806),
    (-2.74385,  2.84121),
    (-0.17453,  1.74533),
]

GRIPPER_OPEN =  1.74533
GRIPPER_CLOSED = -0.12453

def offset_above(pos, dz=0.10):
    return [pos[0], pos[1], pos[2] + dz]


CUBE_POS = [0.20, 0.00, 0.02]
TARGET_POS = [0.28, -0.08, 0.02]

PRE_GRASP_POS = offset_above(CUBE_POS, 0.10)
GRASP_POS = offset_above(CUBE_POS, 0.02)
LIFT_POS = offset_above(CUBE_POS, 0.15)
PRE_PLACE_POS = offset_above(TARGET_POS, 0.10)
PLACE_POS = offset_above(TARGET_POS, 0.02)
RETREAT_POS = offset_above(TARGET_POS, 0.15)

# REAL values recorded from the actual scene
POSES = {
    # Safe neutral home
    'home':      [ 0.00, -0.50,  0.50, -0.20,  0.00,  1.74533],

    # Above the red cube - same pan/roll as grasp but higher
    'pre_grasp': [ 0.1265, -0.32,  0.62,  0.129,  1.553,  1.74533],

    # Exact position recorded from the scene when gripper is around cube
    'grasp':     [ 0.1265, -0.10,  0.66,  0.1289,  1.5527,  1.74533],

    # Same as grasp but lifted up (shoulder_lift back to -0.50)
    'lift':      [ 0.1265, -0.42,  0.78,  0.129,  1.553, -0.12453],

    # Above the blue cylinder target - panned right
    'pre_place': [-0.3750, -0.2000,  0.6431,  0.5421,  0.2750, GRIPPER_CLOSED],

    # Lowered to place level at target
    'place':     [-0.3749683816676039, -0.10228149027285953,  0.6430991834294894,  0.5420980004187164,  0.27496522107931565, GRIPPER_CLOSED],

    # Retreat back up
    'retreat':   [-0.3750, -0.5000,  0.8000,  0.1290,  0.2750, GRIPPER_OPEN],
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def send_joints(sock, joints):
    clamped = [clamp(v, lo, hi) for v, (lo, hi) in zip(joints, JOINT_LIMITS)]
    if all(math.isfinite(v) for v in clamped):
        sock.sendto(json.dumps(clamped).encode('utf-8'), (BRIDGE_IP, BRIDGE_PORT))

def move_to_pose(sock, start, end, duration=2.0, hz=50):
    steps = int(duration * hz)
    for i in range(steps + 1):
        t = i / steps
        t_smooth = t * t * (3 - 2 * t)
        joints = [s + (e - s) * t_smooth for s, e in zip(start, end)]
        send_joints(sock, joints)
        time.sleep(1.0 / hz)
    return end

def run_task1():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("TASK 1 - Autonomous Object Pick and Place")
    print("Starting in 3 seconds...")
    time.sleep(3)
    current = list(POSES['home'])
    print("[1/9] HOME position...")
    current = move_to_pose(sock, current, POSES['home'], duration=2.0)
    time.sleep(0.5)
    print("[2/9] PRE-GRASP - moving above object...")
    current = move_to_pose(sock, current, POSES['pre_grasp'], duration=2.5)
    time.sleep(0.5)
    print("[3/9] GRASP - descending to object...")
    current = move_to_pose(sock, current, POSES['grasp'], duration=1.5)
    time.sleep(0.5)
    print("[4/9] Closing gripper...")
    grasp_closed = list(POSES['grasp'])
    grasp_closed[5] = -0.12453
    current = move_to_pose(sock, current, grasp_closed, duration=1.0)
    time.sleep(1.5)
    print("[5/9] LIFTING object...")
    current = move_to_pose(sock, current, POSES['lift'], duration=2.0)
    time.sleep(0.5)
    print("[6/9] Moving to above target location...")
    current = move_to_pose(sock, current, POSES['pre_place'], duration=2.5)
    time.sleep(0.5)
    print("[7/9] Lowering to place...")
    current = move_to_pose(sock, current, POSES['place'], duration=1.5)
    time.sleep(0.5)
    print("[8/9] Opening gripper - releasing object...")
    place_open = list(POSES['place'])
    place_open[5] = 1.74533
    current = move_to_pose(sock, current, place_open, duration=1.0)
    time.sleep(0.8)
    print("[9/9] Retreating...")
    current = move_to_pose(sock, current, POSES['retreat'], duration=2.0)
    time.sleep(0.5)
    print("TASK 1 COMPLETE")
    sock.close()

if __name__ == '__main__':
    run_task1()
