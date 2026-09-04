# -*- coding: utf-8 -*-
"""
test_digital_ir.py — Quick test script for Digital IR I/O sensors on RoboMaster EP.
Reads get_io(id=..., port=...) from boards 1-4, ports 1-2.
"""

import time
from robomaster import robot

import config


def test_ir():
    print("=" * 55)
    print("  RoboMaster EP — Digital IR I/O Sensor Tester")
    print("=" * 55)
    print(f"Connecting via {config.CONN_TYPE.upper()} mode...")

    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type=config.CONN_TYPE)
        adaptor = ep_robot.sensor_adaptor

        print("\nConfigured Sensors:")
        print(f"  - Left IR  : Board {config.LEFT_IO_BOARD}, Port {config.LEFT_IO_PORT}")
        print(f"  - Right IR : Board {config.RIGHT_IO_BOARD}, Port {config.RIGHT_IO_PORT}")
        print(f"  - Wall Detected Value: {config.IO_WALL_VALUE}\n")

        print("Testing continuous readings (Press Ctrl+C to stop)...")
        while True:
            left_val = adaptor.get_io(id=config.LEFT_IO_BOARD, port=config.LEFT_IO_PORT)
            right_val = adaptor.get_io(id=config.RIGHT_IO_BOARD, port=config.RIGHT_IO_PORT)

            left_wall = (left_val == config.IO_WALL_VALUE)
            right_wall = (right_val == config.IO_WALL_VALUE)

            print(f"\r[IR I/O] Left: {left_val} ({'WALL' if left_wall else 'FREE'}) | "
                  f"Right: {right_val} ({'WALL' if right_wall else 'FREE'})", end="", flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ep_robot.close()
        print("Robot connection closed.")


if __name__ == "__main__":
    test_ir()
