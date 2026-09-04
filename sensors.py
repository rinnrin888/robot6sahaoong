# -*- coding: utf-8 -*-
"""
sensors.py -- Hardware Abstraction Layer for Lab 6.

Handles:
  1. Front ToF Distance Sensor on Gimbal (locked pitch=0, yaw=0).
  2. Left/Right Digital IR Obstacle Sensors via sensor_adaptor.get_io().
  3. Mock Sensor simulation with a HIDDEN randomly-generated maze.
     The robot has NO prior knowledge of the maze layout.
     Walls are discovered ONLY through sensor readings during exploration.
"""

import sys
import os
import time
import random
import threading
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config


class SensorManager:
    """
    Sensor manager for Lab 6:
      - Lock Gimbal at pitch=0, yaw=0 (stationary, horizontal) for Front ToF.
      - Read Digital IR I/O sensors (Left & Right) via sensor_adaptor.get_io().
      - In MOCK mode: simulate a HIDDEN random maze that the robot must discover.
    """

    def __init__(self, ep_robot=None, mock=False):
        self.ep_robot = ep_robot
        self.mock = mock

        # Front ToF state
        self._front_dist_cm = None
        self._last_tof_time = 0.0
        self._tof_raw_samples = []
        self._tof_lock = threading.Lock()
        self.tof_ready = threading.Event()

        # ตำแหน่งล่าสุดของหุ่น (ใช้ใน mock mode ให้ read_front_tof / sample_front_tof รู้ทิศ)
        self._mock_rx = 0
        self._mock_ry = 0
        self._mock_heading = 0

        # ----------------------------------------------------------------
        # HIDDEN MAZE for Mock simulation.
        # Stored as edge-based walls (same format as MazeWalls):
        #   h_walls[row][col] = 1  -> horizontal wall between row and row+1
        #                             (i.e. NORTH wall of cell (col, row))
        #   v_walls[row][col] = 1  -> vertical wall between col and col+1
        #                             (i.e. EAST wall of cell (col, row))
        # The ROBOT does NOT have access to these arrays.
        # It only learns about walls through sensor readings.
        # ----------------------------------------------------------------
        self._h_walls, self._v_walls = self._generate_maze()
        self._print_hidden_maze()

    # ----------------------------------------------------------------
    # Maze generation (DFS / recursive backtracking)
    # ----------------------------------------------------------------
    def _generate_maze(self):
        """
        Generate a solvable random maze using DFS recursive backtracking.
        Returns (h_walls, v_walls) as 2-D lists.
          h_walls[y][x] = 1  => wall on NORTH edge of cell (x, y)
                                (between row y and y+1)
          v_walls[y][x] = 1  => wall on EAST edge of cell (x, y)
                                (between col x and x+1)
        All inner walls start closed; DFS carves passages.
        """
        W = config.GRID_WIDTH
        H = config.GRID_HEIGHT

        # Start with all inner walls closed (1 = wall present)
        # h_walls has (H-1) rows  (no wall above row H-1 or below row 0)
        # v_walls has (W-1) cols  (no wall left of col 0 or right of col W-1)
        h_walls = [[1] * W for _ in range(H - 1)]   # between rows
        v_walls = [[1] * (W - 1) for _ in range(H)] # between cols

        visited = [[False] * W for _ in range(H)]
        stack = [(0, 0)]
        visited[0][0] = True

        while stack:
            cx, cy = stack[-1]
            # Find unvisited neighbours
            neighbours = []
            for nx, ny in [(cx, cy+1), (cx+1, cy), (cx, cy-1), (cx-1, cy)]:
                if 0 <= nx < W and 0 <= ny < H and not visited[ny][nx]:
                    neighbours.append((nx, ny))

            if neighbours:
                nx, ny = random.choice(neighbours)
                # Carve the wall between (cx,cy) and (nx,ny)
                if ny == cy + 1:           # moving North
                    h_walls[cy][cx] = 0   # remove NORTH wall of (cx,cy)
                elif ny == cy - 1:         # moving South
                    h_walls[ny][cx] = 0   # remove NORTH wall of (nx,ny)
                elif nx == cx + 1:         # moving East
                    v_walls[cy][cx] = 0   # remove EAST wall of (cx,cy)
                elif nx == cx - 1:         # moving West
                    v_walls[cy][nx] = 0   # remove EAST wall of (nx,ny)

                visited[ny][nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        return h_walls, v_walls

    def _print_hidden_maze(self):
        """Print the hidden maze layout to terminal (for human reference only).
        The robot never reads this — it learns through sensors."""
        W = config.GRID_WIDTH
        H = config.GRID_HEIGHT
        print("")
        print("=" * 50)
        print("  [MOCK] HIDDEN MAZE LAYOUT (robot does NOT know this)")
        print("  Legend:  +--+  = wall    +  +  = open passage")
        print("=" * 50)

        for y in range(H - 1, -1, -1):
            # Top edge of row y
            top_row = "  +"
            for x in range(W):
                if y == H - 1:
                    top_row += "---+"   # outer north border
                else:
                    top_row += "---+" if self._h_walls[y][x] else "   +"
            print(top_row)
            # Cell row
            cell_row = "  |"
            for x in range(W):
                label = "(%d,%d)" % (x, y)
                if x < W - 1:
                    cell_row += " %s %s" % (label, "|" if self._v_walls[y][x] else " ")
                else:
                    cell_row += " %s |" % label
            print(cell_row)

        # Bottom border
        print("  +" + "---+" * W)
        print("  START=(0,0) bottom-left   EXIT=(%d,%d) top-right" % (W-1, H-1))
        print("=" * 50)
        print("")

    # ----------------------------------------------------------------
    # Mock sensor: check whether an edge wall exists
    # ----------------------------------------------------------------
    def _mock_edge_has_wall(self, from_x, from_y, direction):
        """
        Check if there is a wall on the specified edge of cell (from_x, from_y).
        Uses the hidden maze edge arrays.
        Returns True if wall, False if open.
        """
        W = config.GRID_WIDTH
        H = config.GRID_HEIGHT

        d = direction.upper()
        nx = from_x + (1 if d == "EAST" else -1 if d == "WEST" else 0)
        ny = from_y + (1 if d == "NORTH" else -1 if d == "SOUTH" else 0)

        # Arena border = always wall
        if nx < 0 or nx >= W or ny < 0 or ny >= H:
            return True

        # Inner edge wall lookup
        if d == "NORTH":
            # h_walls[from_y][from_x]: wall between row from_y and from_y+1
            if from_y < H - 1:
                return bool(self._h_walls[from_y][from_x])
            return True   # top border

        if d == "SOUTH":
            # wall between row from_y-1 and from_y
            if from_y > 0:
                return bool(self._h_walls[from_y - 1][from_x])
            return True   # bottom border

        if d == "EAST":
            # v_walls[from_y][from_x]: wall between col from_x and from_x+1
            if from_x < W - 1:
                return bool(self._v_walls[from_y][from_x])
            return True   # right border

        if d == "WEST":
            # wall between col from_x-1 and from_x
            if from_x > 0:
                return bool(self._v_walls[from_y][from_x - 1])
            return True   # left border

        return False

    # ----------------------------------------------------------------
    # Hardware setup
    # ----------------------------------------------------------------
    def setup_hardware(self):
        """Lock Gimbal at pitch=0/yaw=0 and start ToF subscription."""
        if self.mock:
            print("[Sensors] [MOCK] Running in MOCK simulation mode.")
            return True

        print("[Sensors] Setting up Gimbal and ToF sensor...")
        try:
            gimbal = self.ep_robot.gimbal
            sensor  = self.ep_robot.sensor

            print("[Gimbal] Locking at Pitch=0, Yaw=0 (stationary, horizontal)...")
            gimbal.recenter(
                pitch_speed=config.GIMBAL_SPEED,
                yaw_speed=config.GIMBAL_SPEED
            ).wait_for_completed()
            gimbal.moveto(
                pitch=config.GIMBAL_PITCH,
                yaw=config.GIMBAL_YAW
            ).wait_for_completed()
            print("[Gimbal] [OK] Gimbal locked.")

            sensor.sub_distance(freq=config.CONTROL_FREQ_HZ, callback=self._tof_callback)
            print("[ToF] Waiting for first reading...")
            if self.tof_ready.wait(timeout=3.0):
                with self._tof_lock:
                    print("[ToF] [OK] Ready: %.1f cm" % self._front_dist_cm)
            else:
                print("[ToF] [WARN] No reading within 3s — check wiring.")

            l_io, _ = self.read_left_io()
            r_io, _ = self.read_right_io()
            print("[IR] [OK] Left IO=%d  Right IO=%d" % (l_io, r_io))
            return True

        except Exception as e:
            print("[Sensors] [ERROR] Hardware setup failed: %s" % e)
            return False

    # ----------------------------------------------------------------
    # ToF callback
    # ----------------------------------------------------------------
    def _tof_callback(self, sub_info):
        if not sub_info or len(sub_info) == 0:
            return
        val_mm = sub_info[0]
        now = time.monotonic()
        with self._tof_lock:
            self._last_tof_time = now
            # ในเซนเซอร์ RoboMaster ToF:
            # val_mm == 0 หมายถึงไม่มีแสงสะท้อนกลับมา (พื้นที่ว่าง / ระยะไกลเกิน 2 เมตร)
            # 20 < val_mm < 5000 คือระยะสิ่งกีดขวางที่ตรวจพบจริง
            if val_mm == 0 or val_mm >= 5000:
                self._front_dist_cm = 999.0
            elif val_mm > 20:
                self._front_dist_cm = float(val_mm) / 10.0
            else:
                self._front_dist_cm = 999.0

            self._tof_raw_samples.append((now, self._front_dist_cm))
            if len(self._tof_raw_samples) > 30:
                self._tof_raw_samples.pop(0)
            self.tof_ready.set()

    # ----------------------------------------------------------------
    # Real hardware reads
    # ----------------------------------------------------------------
    def read_front_tof(self):
        if self.mock:
            # เช็คกำแพงจาก hidden maze ตามตำแหน่งล่าสุดที่รู้
            d = int(round(self._mock_heading / 90.0) * 90) % 360
            front_dir = {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}[d]
            has_wall = self._mock_edge_has_wall(self._mock_rx, self._mock_ry, front_dir)
            return (35.0 + np.random.normal(0, 2.0)) if has_wall else (95.0 + np.random.normal(0, 5.0))
        with self._tof_lock:
            return self._front_dist_cm

    def sample_front_tof(self, num_samples=6, sample_interval=0.035, max_wait_s=0.35):
        """
        เก็บตัวอย่างค่า ToF ด้านหน้าหลายๆ ครั้ง (Multi-sampling) ขณะที่หุ่นหยุดนิ่ง
        แล้วคำนวณค่า Median เพื่อกรอง Noise และป้องกันการอ่านค่าค้างจากการหมุน
        """
        if self.mock:
            # Mock: เช็คกำแพงจาก hidden maze โดยตรง (ไม่ใช้ค่า 999.0)
            d = int(round(self._mock_heading / 90.0) * 90) % 360
            front_dir = {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}[d]
            has_wall = self._mock_edge_has_wall(self._mock_rx, self._mock_ry, front_dir)
            return (35.0 + np.random.normal(0, 2.0)) if has_wall else (95.0 + np.random.normal(0, 5.0))

        time.sleep(0.04)  # รอรับรอบใหม่ 1 รอบหลังหยุดหมุน (20Hz = 50ms)
        samples = []
        start_t = time.monotonic()

        while len(samples) < num_samples and (time.monotonic() - start_t) < max_wait_s:
            with self._tof_lock:
                if self._front_dist_cm is not None and (time.monotonic() - self._last_tof_time) < 0.25:
                    samples.append(self._front_dist_cm)
            time.sleep(sample_interval)

        if not samples:
            with self._tof_lock:
                return self._front_dist_cm if self._front_dist_cm is not None else 999.0

        sorted_s = sorted(samples)
        median_val = sorted_s[len(sorted_s) // 2]
        return median_val

    def read_left_io(self):
        if self.mock or not self.ep_robot:
            return 0, False
        try:
            val = self.ep_robot.sensor_adaptor.get_io(
                id=config.LEFT_IO_BOARD, port=config.LEFT_IO_PORT)
            return val, (val == config.IO_WALL_VALUE)
        except Exception:
            return 0, False

    def read_right_io(self):
        if self.mock or not self.ep_robot:
            return 0, False
        try:
            val = self.ep_robot.sensor_adaptor.get_io(
                id=config.RIGHT_IO_BOARD, port=config.RIGHT_IO_PORT)
            return val, (val == config.IO_WALL_VALUE)
        except Exception:
            return 0, False

    # ----------------------------------------------------------------
    # Unified sensor read
    # ----------------------------------------------------------------
    def read_all_sensors(self, robot_x=0, robot_y=0, heading_deg=0, multi_sample=True):
        """Read all 3 sensors (front ToF with multi-sample median filter, left IR, right IR)."""
        # อัปเดตตำแหน่งล่าสุดใน mock (ให้ read_front_tof/sample_front_tof ใช้ได้)
        self._mock_rx = robot_x
        self._mock_ry = robot_y
        self._mock_heading = heading_deg

        if self.mock:
            return self._get_mock_sensor_data(robot_x, robot_y, heading_deg)

        if multi_sample:
            front_dist = self.sample_front_tof(num_samples=5, sample_interval=0.035)
        else:
            front_cm = self.read_front_tof()
            front_dist = front_cm if front_cm is not None else 999.0

        front_is_wall = (front_dist < config.WALL_DETECT_CM)

        left_val, left_is_wall   = self.read_left_io()
        right_val, right_is_wall = self.read_right_io()

        return {
            "front_dist_cm": front_dist,
            "front_is_wall": front_is_wall,
            "left_io":       left_val,
            "left_is_wall":  left_is_wall,
            "right_io":      right_val,
            "right_is_wall": right_is_wall,
        }

    # ----------------------------------------------------------------
    # Mock sensor simulation — uses hidden edge-based maze
    # ----------------------------------------------------------------
    def _get_mock_sensor_data(self, rx, ry, heading_deg):
        """
        Simulate sensor readings from the HIDDEN maze.
        The robot only sees True/False for each direction.
        It does NOT have direct access to _h_walls or _v_walls.
        """
        def heading_to_dir(deg):
            d = int(round(deg / 90.0) * 90) % 360
            return {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}[d]

        front_dir = heading_to_dir(heading_deg)
        left_dir  = heading_to_dir(heading_deg - 90)
        right_dir = heading_to_dir(heading_deg + 90)

        front_wall = self._mock_edge_has_wall(rx, ry, front_dir)
        left_wall  = self._mock_edge_has_wall(rx, ry, left_dir)
        right_wall = self._mock_edge_has_wall(rx, ry, right_dir)

        # Simulate ToF distance with noise
        # กำแพงอยู่ที่ ~30-35cm (กึ่งกลางช่อง 60cm), โล่งอยู่ที่ ~95cm (ช่องถัดไป)
        if front_wall:
            front_dist = 35.0 + np.random.normal(0, 2.0)
        else:
            front_dist = 95.0 + np.random.normal(0, 5.0)

        # Digital IO (1 = wall detected, 0 = clear)
        left_io  = config.IO_WALL_VALUE if left_wall  else (1 - config.IO_WALL_VALUE)
        right_io = config.IO_WALL_VALUE if right_wall else (1 - config.IO_WALL_VALUE)

        return {
            "front_dist_cm": max(5.0, front_dist),
            "front_is_wall": front_wall,
            "left_io":       left_io,
            "left_is_wall":  left_wall,
            "right_io":      right_io,
            "right_is_wall": right_wall,
        }
