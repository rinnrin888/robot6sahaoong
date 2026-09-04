# -*- coding: utf-8 -*-
"""
lab6_main.py — Main Experiment Runner for Lab 6 (Occupancy Grid Mapping)
RoboMaster EP 4x4 Grid OGM using Bayesian Log-Odds Updates.

Usage:
  - Mock/Simulation mode:
      python lab6_main.py --mock
  - Real RoboMaster EP mode (Wi-Fi AP):
      cd C:\robotproject\robot_sahapong\lab6
python lab6_main.py --real --explore

"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

# Reconfigure stdout for utf-8 if on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure proper paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from ogm import OccupancyGridMap
from sensors import SensorManager
from explorer import AutonomousExplorer
from dashboard import InteractiveDashboard
from maze_walls import MazeWalls


class Lab6Experiment:
    """
    ตัวควบคุมการทดลอง Lab 6:
      - นำทางหุ่นยนต์บน Grid 4x4 (ทีละก้าว 60 cm)
      - อ่านค่าเซนเซอร์ Front ToF (ล็อก Gimbal ตรง), Left/Right Digital IR I/O
      - อัปเดตตารางกำแพงรอบด้าน (MazeWalls: North, East, South, West) และ OGM
      - ระบบสำรวจเขาวงกตหาทางออกอัตโนมัติ (Autonomous Maze Exploration)
      - หน้าจอ Interactive Dashboard กำหนดทางออกและความเร็วได้สดๆ (OpenCV)
    """

    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.ep_robot = None
        self.ogm = OccupancyGridMap()
        self.maze_walls = MazeWalls()
        self.dashboard = InteractiveDashboard()
        self.explorer = AutonomousExplorer(exit_cell=self.dashboard.exit_cell)

        # สถานะตำแหน่งของหุ่นยนต์ (เริ่มต้นที่จุด (0, 0) หันหน้าไปทางทิศเหนือ Heading = 0°)
        # 0° = North (+Y), 90° = East (+X), 180° = South (-Y), 270° = West (-X)
        self.robot_x = 0
        self.robot_y = 0
        self.heading_deg = 0

        # บันทึกเส้นทางที่เดิน (Trajectory)
        self.path_history = [(self.robot_x, self.robot_y)]
        self.step_counter = 0
        self.scanned_cells = set()

        # เตรียมเซนเซอร์
        self.sensor_mgr = None

    def apply_setup_choices(self):
        """Apply start_cell/exit_cell chosen in Dashboard Setup phase."""
        sx, sy = self.dashboard.start_cell
        self.robot_x = sx
        self.robot_y = sy
        self.path_history = [(sx, sy)]
        self.scanned_cells.clear()
        self.explorer.set_exit_cell(self.dashboard.exit_cell)
        config.EXIT_CELL = self.dashboard.exit_cell
        print("[Setup] Start=(%d,%d)  Exit=(%d,%d)" % (sx, sy,
              self.dashboard.exit_cell[0], self.dashboard.exit_cell[1]))

    def initialize_system(self):
        """เชื่อมต่อหุ่นยนต์และตั้งค่าฮาร์ดแวร์"""
        if not self.use_mock:
            print(f"[Main] [CONNECT] กำลังเชื่อมต่อ RoboMaster EP ผ่าน Wi-Fi ({config.CONN_TYPE.upper()})...")
            try:
                from robomaster import robot
                self.ep_robot = robot.Robot()
                self.ep_robot.initialize(conn_type=config.CONN_TYPE)
                print("[Main] [OK] เชื่อมต่อหุ่นยนต์สำเร็จ!")
            except Exception as e:
                print(f"[Main] [ERROR] ไม่สามารถเชื่อมต่อกับหุ่นยนต์ได้: {e}")
                print("[Main] [ERROR] ตรวจสอบ:")
                print("         1. PC เชื่อม Wi-Fi ชื่อ RoboMaster_XXXX แล้วหรือยัง?")
                print("         2. หุ่นยนต์เปิดอยู่และไฟ LED กระพริบหรือไม่?")
                print("         3. ถ้าต้องการ Simulation ให้ใช้ --mock แทน --real")
                print("[Main] [EXIT] ยกเลิกการทดลอง (ไม่มี fallback ไป simulation โดยอัตโนมัติ)")
                return False

        self.sensor_mgr = SensorManager(ep_robot=self.ep_robot, mock=self.use_mock)
        return self.sensor_mgr.setup_hardware()

    @staticmethod
    def _deg_to_cardinal(deg):
        """แปลงมุมองศาเป็นทิศหลัก (NORTH, EAST, SOUTH, WEST)"""
        d = int(round(deg / 90.0) * 90) % 360
        if d == 0: return "NORTH"
        elif d == 90: return "EAST"
        elif d == 180: return "SOUTH"
        elif d == 270: return "WEST"
        return "NORTH"

    def perform_scan_and_update(self, status_reason="SCANNING", use_side_ir=True):
        """
        อ่านค่าเซนเซอร์ทั้งหมดและอัปเดตค่าความน่าจะเป็นของกำแพงรอบด้าน (MazeWalls)
        และ OGM พร้อมเรนเดอร์หน้าจอ Interactive Dashboard
        - use_side_ir: หากเป็น False (เช่น ขณะหมุน 360 สแกน) จะใช้เฉพาะ Front ToF ในการระบุสิ่งกีดขวาง
          เพื่อป้องกันไม่ให้เซนเซอร์ IR ด้านข้างที่อาจมี noise มารบกวนหรือทับค่าพื้นที่ว่าง
        """
        self.step_counter += 1
        print(f"\n[Step #{self.step_counter}] กำลังสแกนเซนเซอร์ ณ ตำแหน่ง ({self.robot_x}, {self.robot_y}) Heading: {self.heading_deg}°...")

        # 1. อ่านค่าเซนเซอร์ 3 ด้าน
        data = self.sensor_mgr.read_all_sensors(self.robot_x, self.robot_y, self.heading_deg)
        front_cm = data["front_dist_cm"]
        front_wall = data["front_is_wall"]
        left_io = data["left_io"]
        left_wall = data["left_is_wall"]
        right_io = data["right_io"]
        right_wall = data["right_is_wall"]

        print(f"       -> ด้านหน้า (ToF บน Gimbal): {front_cm:.1f} cm [{'WALL' if front_wall else 'CLEAR'}]")
        print(f"       -> ด้านซ้าย  (Digital IR IO): Value={left_io} [{'WALL' if left_wall else 'CLEAR'}]")
        print(f"       -> ด้านขวา   (Digital IR IO): Value={right_io} [{'WALL' if right_wall else 'CLEAR'}]")

        # 2. คำนวณทิศทางจริงตามเข็มทิศ
        front_dir = self._deg_to_cardinal(self.heading_deg)
        left_dir = self._deg_to_cardinal(self.heading_deg - 90)
        right_dir = self._deg_to_cardinal(self.heading_deg + 90)

        # 2.1 อัปเดตขอบกำแพงรอบด้าน (MazeWalls)
        # ToF ด้านหน้าเป็นตัวหลักในการสร้างแมพ
        self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, front_dir, is_wall=front_wall)
        
        # ปิดการใช้ IR ด้านข้างสร้างกำแพงถาวร ป้องกัน false positive เวลารถเอียง
        # if use_side_ir:
        #     self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, left_dir, is_wall=left_wall)
        #     self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, right_dir, is_wall=right_wall)

        # 2.2 อัปเดต Bayesian Log-Odds OGM (เซลล์)
        self.ogm.update_direction_sensor(self.robot_x, self.robot_y, self.heading_deg, 0, front_cm)
        if use_side_ir:
            self.ogm.update_direction_binary(self.robot_x, self.robot_y, self.heading_deg, -90, left_wall)
            self.ogm.update_direction_binary(self.robot_x, self.robot_y, self.heading_deg, 90, right_wall)

        # บันทึกประวัติ
        log_entry = (f"Step {self.step_counter}: Pos=({self.robot_x},{self.robot_y}) "
                     f"Heading={self.heading_deg} deg | Front={front_cm:.1f}cm ({'WALL' if front_wall else 'CLEAR'}), "
                     f"Left_IO={left_io} ({'WALL' if left_wall else 'CLEAR'}), "
                     f"Right_IO={right_io} ({'WALL' if right_wall else 'CLEAR'})")
        self.ogm.history.append(log_entry)

        # 3. แสดงผลตาราง Real-time ASCII บน Terminal
        self.ogm.print_ascii_map(self.robot_x, self.robot_y, self.heading_deg)

        # 4. ซิงค์เป้าหมายทางออกถ้าผู้ใช้คลิกเลือกบน Dashboard
        if self.explorer.exit_cell != self.dashboard.exit_cell:
            self.explorer.set_exit_cell(self.dashboard.exit_cell)

        # 5. อัปเดตหน้าต่าง Interactive GUI Dashboard (OpenCV)
        self.dashboard.render(
            maze_walls=self.maze_walls,
            ogm=self.ogm,
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            heading_deg=self.heading_deg,
            path_history=self.path_history,
            sensor_data=data,
            explorer_status=status_reason,
            mode_str="MOCK SIMULATION" if self.use_mock else "REAL ROBOT",
            step_count=self.step_counter
        )
        return data

    # ================================================================
    # IR WALL AVOIDANCE SYSTEM
    # ================================================================

    def read_ir_quick(self, num_samples=3):
        """
        อ่านค่า IR ซ้าย/ขวาแบบ Real-time หลาย samples แล้ว Majority-Vote
        เพื่อป้องกัน Noise จากมอเตอร์หรือสัญญาณรบกวนชั่วคราว
        คืนค่า: (left_wall: bool, right_wall: bool, left_io: int, right_io: int)
        """
        if self.use_mock or not self.ep_robot:
            # Mock: อ่านจาก sensor_mgr ตามปกติ
            data = self.sensor_mgr.read_all_sensors(self.robot_x, self.robot_y, self.heading_deg)
            return data["left_is_wall"], data["right_is_wall"], data["left_io"], data["right_io"]

        left_readings  = []
        right_readings = []
        for _ in range(num_samples):
            l_val, l_wall = self.sensor_mgr.read_left_io()
            r_val, r_wall = self.sensor_mgr.read_right_io()
            left_readings.append((l_val, l_wall))
            right_readings.append((r_val, r_wall))
            time.sleep(0.025)  # 25ms ต่อ sample → รวม ~75ms

        # Majority vote: เจอกำแพง >= 2 ใน 3 ครั้ง ถือว่าเจอกำแพง
        left_wall_votes  = sum(1 for _, w in left_readings  if w)
        right_wall_votes = sum(1 for _, w in right_readings if w)
        left_wall  = left_wall_votes  >= (num_samples // 2 + 1)
        right_wall = right_wall_votes >= (num_samples // 2 + 1)
        left_io  = left_readings[-1][0]
        right_io = right_readings[-1][0]
        return left_wall, right_wall, left_io, right_io

    def read_front_tof_quick(self):
        """
        อ่าน ToF ด้านหน้าแบบเร็ว (multi-sample median) สำหรับ avoidance check
        คืนค่า: (dist_cm: float, is_wall: bool)
        """
        if self.use_mock or not self.ep_robot:
            data = self.sensor_mgr.read_all_sensors(self.robot_x, self.robot_y, self.heading_deg)
            return data["front_dist_cm"], data["front_is_wall"]
        dist_cm = self.sensor_mgr.sample_front_tof(num_samples=4, sample_interval=0.03)
        is_wall = dist_cm < config.WALL_DETECT_CM
        return dist_cm, is_wall

    def ir_safety_check_before_move(self):
        """
        ตรวจสอบความปลอดภัยก่อนเดินหน้า 1 ช่อง โดยใช้ IR Real-time:
          1. ToF ด้านหน้า: ถ้าระยะ < WALL_DETECT_CM → ห้ามเดิน (BLOCKED)
          2. IR ซ้าย/ขวา: บันทึกค่าไว้อัปเดต MazeWalls แต่ไม่บล็อกการเดิน
             (เนื่องจาก IR ซ้าย/ขวา = กำแพงข้างๆ ช่องปัจจุบัน ไม่ใช่ทางเดินข้างหน้า)
        คืนค่า: (safe_to_move: bool, front_cm: float, left_wall: bool, right_wall: bool)
        """
        print("  [IR-Check] กำลังตรวจสอบ IR ก่อนเดิน...")
        # อ่าน ToF หน้า
        front_cm, front_wall = self.read_front_tof_quick()
        # อ่าน IR ซ้าย/ขวา
        left_wall, right_wall, left_io, right_io = self.read_ir_quick()

        left_dir  = self._deg_to_cardinal(self.heading_deg - 90)
        right_dir = self._deg_to_cardinal(self.heading_deg + 90)

        # อัปเดต MazeWalls จากข้อมูล IR ล่าสุด
        # ปิดการอัปเดตแมพหลักด้วย IR เพื่อป้องกันกำแพงปลอมเวลารถวิ่งเอียง
        # self.maze_walls.update_wall_sensor(
        #     self.robot_x, self.robot_y, left_dir,  is_wall=left_wall,  weight=1.5)
        # self.maze_walls.update_wall_sensor(
        #     self.robot_x, self.robot_y, right_dir, is_wall=right_wall, weight=1.5)

        lbl_l = f"IO={left_io} [{'WALL' if left_wall else 'CLEAR'}]"
        lbl_r = f"IO={right_io} [{'WALL' if right_wall else 'CLEAR'}]"
        lbl_f = f"{front_cm:.1f} cm [{'WALL!' if front_wall else 'CLEAR'}]"
        print(f"  [IR-Check] Front ToF: {lbl_f}  | Left IR: {lbl_l}  | Right IR: {lbl_r}")

        if front_wall:
            print("  [IR-Check] *** กำแพงด้านหน้า! ยกเลิกการเดิน ***")
            # อัปเดต MazeWalls ฝั่งหน้าด้วย weight สูง
            front_dir = self._deg_to_cardinal(self.heading_deg)
            self.maze_walls.update_wall_sensor(
                self.robot_x, self.robot_y, front_dir, is_wall=True, weight=2.0)
            return False, front_cm, left_wall, right_wall

        return True, front_cm, left_wall, right_wall

    def ir_wall_avoidance(self, reason="AVOIDANCE"):
        """
        ระบบหลีกเลี่ยงกำแพงฉุกเฉินด้วย IR:
        ตรวจสอบ IR ซ้าย/ขวา และ ToF หน้า แล้วสั่งเลี้ยวหาทางว่าง
        Logic Priority:
          1. ถ้า Front CLEAR: ไม่ทำอะไร (ปลอดภัย)
          2. ถ้า Front WALL + Left CLEAR:  หมุนซ้าย
          3. ถ้า Front WALL + Right CLEAR: หมุนขวา
          4. ถ้า Front+Left WALL, Right CLEAR: หมุนขวา
          5. ถ้า Front+Right WALL, Left CLEAR: หมุนซ้าย
          6. ถ้า Wall รอบด้าน: หมุน 180° (กลับหลัง)
        คืนค่า: (action_taken: str)  เช่น "NONE", "TURN_LEFT", "TURN_RIGHT", "REVERSE"
        """
        print("\n" + "=" * 55)
        print("  [AVOIDANCE] ตรวจสอบ IR เพื่อหลีกเลี่ยงกำแพง...")
        print("=" * 55)

        front_cm, front_wall = self.read_front_tof_quick()
        left_wall, right_wall, left_io, right_io = self.read_ir_quick()

        front_dir = self._deg_to_cardinal(self.heading_deg)
        left_dir  = self._deg_to_cardinal(self.heading_deg - 90)
        right_dir = self._deg_to_cardinal(self.heading_deg + 90)

        print(f"  Front ToF : {front_cm:.1f} cm -> [{'WALL' if front_wall else 'CLEAR'}]")
        print(f"  Left  IR  : IO={left_io}  -> [{'WALL' if left_wall else 'CLEAR'}]")
        print(f"  Right IR  : IO={right_io} -> [{'WALL' if right_wall else 'CLEAR'}]")

        # อัปเดต MazeWalls
        self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, front_dir, is_wall=front_wall, weight=2.0)
        self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, left_dir,  is_wall=left_wall,  weight=2.0)
        self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, right_dir, is_wall=right_wall, weight=2.0)

        # ทางหน้าโล่ง → ไม่ต้องหลีก
        if not front_wall:
            print("  [AVOIDANCE] ทางหน้าโล่ง — ไม่จำเป็นต้องหลีก")
            return "NONE"

        # ทางหน้าโดนบล็อก → หาทางเลี้ยว
        # ตรวจสอบ Left / Right ว่าข้างไหนโล่ง
        can_left  = not self.maze_walls.is_wall_blocked(self.robot_x, self.robot_y, left_dir)
        can_right = not self.maze_walls.is_wall_blocked(self.robot_x, self.robot_y, right_dir)

        # ถ้า IR บอกโล่งแม้ maze_walls ยังไม่รู้ ก็ใช้ IR โดยตรงเป็น override
        if not left_wall:
            can_left = True
        if not right_wall:
            can_right = True

        print(f"  [AVOIDANCE] Front=WALL | Left={'CLEAR' if can_left else 'WALL'} | Right={'CLEAR' if can_right else 'WALL'}")

        if can_right and not can_left:
            # ขวาโล่งอย่างเดียว → เลี้ยวขวา
            print("  [AVOIDANCE] ➜ เลี้ยวขวา (Right CLEAR)")
            self.turn_right(reason=f"{reason}_AVOID_RIGHT", use_side_ir=True)
            return "TURN_RIGHT"

        elif can_left and not can_right:
            # ซ้ายโล่งอย่างเดียว → เลี้ยวซ้าย
            print("  [AVOIDANCE] ➜ เลี้ยวซ้าย (Left CLEAR)")
            self.turn_left(reason=f"{reason}_AVOID_LEFT", use_side_ir=True)
            return "TURN_LEFT"

        elif can_left and can_right:
            # ทั้งสองข้างโล่ง → เลือกข้างที่ IR บอกว่าโล่งกว่า
            # ถ้าทั้งคู่โล่งเท่ากัน → เลี้ยวขวาก่อน (กฎมือขวา)
            if not right_wall:
                print("  [AVOIDANCE] ➜ เลี้ยวขวา (ทั้งสองข้างโล่ง → ใช้กฎมือขวา)")
                self.turn_right(reason=f"{reason}_AVOID_RIGHT_PREF", use_side_ir=True)
                return "TURN_RIGHT"
            else:
                print("  [AVOIDANCE] ➜ เลี้ยวซ้าย (IR ซ้ายโล่งกว่า)")
                self.turn_left(reason=f"{reason}_AVOID_LEFT_PREF", use_side_ir=True)
                return "TURN_LEFT"

        else:
            # ติดกำแพงทุกด้าน → ระงับการเดินและรอให้ Explorer ถอยหลัง
            print("  [AVOIDANCE] *** ติดกำแพงทุกด้าน! ยกเลิกการเคลื่อนที่เพื่อเตรียมถอยหลัง ***")
            return "REVERSE"

    def move_forward_one_cell(self, reason="MOVING", use_ir_avoidance=True):
        """
        เดินหน้า 1 ช่อง (60 cm) พร้อม Real-time IR Steering:
        - ใช้ chassis.drive_speed() เพื่อให้ IR ซ้าย/ขวาปรับทิศได้ขณะเดินหน้า
        - ตรวจสอบ ToF ทุก 200ms ขณะเดิน — หยุดฉุกเฉินถ้าเจอกำแพงหน้า
        - อัปเดต dashboard ขณะเดินให้แมพตอบสนองทันที
        """
        # คำนวณตำแหน่งถัดไปตาม Heading
        dx, dy = 0, 0
        norm_h = int(round(self.heading_deg / 90.0) * 90) % 360
        if norm_h == 0:
            dy = 1   # North
        elif norm_h == 90:
            dx = 1   # East
        elif norm_h == 180:
            dy = -1  # South
        elif norm_h == 270:
            dx = -1  # West

        next_x = self.robot_x + dx
        next_y = self.robot_y + dy

        # ── ตรวจสอบขอบเขตสนาม ─────────────────────────────────────────────
        if not self.ogm.is_valid_cell(next_x, next_y):
            print(f"[Move] [WARN] ไม่สามารถเดินหน้าได้: ชนขอบเขตสนาม 4x4! (เป้าหมาย: {next_x}, {next_y})")
            return False

        # ── ตรวจสอบกำแพงจาก MazeWalls ────────────────────────────────────────
        card_dir = self._deg_to_cardinal(self.heading_deg)
        if self.maze_walls.is_wall_blocked(self.robot_x, self.robot_y, card_dir):
            print(f"[Move] [WARN] ไม่สามารถเดินหน้าได้: MazeWalls บอกกำแพงกั้นในทิศ {card_dir}!")
            return False

        # ── [IR AVOIDANCE] ตรวจสอบ IR Real-time ก่อนเดิน ──────────────────────
        if use_ir_avoidance:
            safe, front_cm, left_wall, right_wall = self.ir_safety_check_before_move()
            if not safe:
                print(f"[Move] [AVOIDANCE] IR ตรวจพบกำแพงด้านหน้า (ToF={front_cm:.1f} cm) → ยกเลิกการเดินไปที่ ({next_x}, {next_y})")
                # เรียกใช้ระบบหลีกเลี่ยงกำแพงฉุกเฉิน (จะเช็คซ้าย/ขวาและเลี้ยวหนีทันที)
                self.ir_wall_avoidance(reason="AVOID_BEFORE_MOVE")
                return False

        # ── สั่งเดินหน้าจริง พร้อม Real-time IR Steering ─────────────────────
        fwd_speed    = self.dashboard.move_speed     # m/s
        target_dist  = config.STEP_DISTANCE_M        # 0.60 m
        est_time     = target_dist / max(fwd_speed, 0.05)  # วินาที
        poll_ms      = 0.06       # ตรวจ IR ทุก 60ms
        tof_interval = 0.20       # ตรวจ ToF ทุก 200ms
        lateral_corr = 0.07       # ความเร็วซ้าย/ขวา สำหรับ steer (m/s)

        print(f"[Move] [FWD] เดินหน้า {target_dist*100:.0f} cm @ {fwd_speed:.2f} m/s พร้อม IR Steering → ({next_x},{next_y})")

        if not self.use_mock and self.ep_robot:
            start_t       = time.monotonic()
            last_tof_chk  = 0.0
            emergency_stop = False

            try:
                while True:
                    elapsed = time.monotonic() - start_t

                    # ─ หยุดเมื่อครบระยะ ─
                    if elapsed >= est_time:
                        break

                    # ─ ตรวจ ToF ทุก 200ms (emergency front-wall check) ─
                    if elapsed - last_tof_chk >= tof_interval:
                        last_tof_chk = elapsed
                        f_cm, f_wall = self.read_front_tof_quick()
                        if f_wall:
                            print(f"  [Steer] ⚠ EMERGENCY STOP — กำแพงหน้า! (ToF={f_cm:.1f}cm)")
                            emergency_stop = True
                            break

                    # ─ อ่าน IR ซ้าย/ขวา แบบ single-sample (เร็ว ไม่ delay) ─
                    lw, rw, l_io, r_io = self.read_ir_quick(num_samples=1)

                    # ─ คำนวณ lateral correction ─
                    vy = 0.0
                    if lw and not rw:
                        vy = -lateral_corr   # ซ้ายเจอกำแพง → ดันขวา (y ลบ)
                        print(f"  [Steer] L=WALL → Steer Right (vy={vy:.2f})", end="\r")
                    elif rw and not lw:
                        vy = +lateral_corr   # ขวาเจอกำแพง → ดันซ้าย (y บวก)
                        print(f"  [Steer] R=WALL → Steer Left  (vy={vy:.2f})", end="\r")

                    # ─ สั่ง drive_speed ต่อเนื่อง ─
                    self.ep_robot.chassis.drive_speed(x=fwd_speed, y=vy, z=0)
                    time.sleep(poll_ms)

            except Exception as e:
                print(f"\n[Move] [ERROR] drive_speed error: {e}")
                emergency_stop = True
            finally:
                # หยุดหุ่น
                try:
                    self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                    time.sleep(0.12)  # รอให้หยุดนิ่ง
                except Exception:
                    pass

            if emergency_stop:
                # อัปเดต MazeWalls ว่าด้านหน้ามีกำแพง
                self.maze_walls.update_wall_sensor(
                    self.robot_x, self.robot_y, card_dir, is_wall=True, weight=2.5)
                # เรียกใช้ระบบหลีกเลี่ยงกำแพงฉุกเฉิน
                if use_ir_avoidance:
                    self.ir_wall_avoidance(reason="EMERGENCY_STOP")
                return False
        else:
            # Mock mode: simulate time
            time.sleep(est_time * 0.5)

        # ─── อัปเดตตำแหน่งบน Map ────────────────────────────────────────────
        self.robot_x = next_x
        self.robot_y = next_y
        self.path_history.append((self.robot_x, self.robot_y))

        # Mark ทางที่เดินมา (Back wall ของ cell ใหม่) เป็น CLEAR เสมอ เพราะเพิ่งผ่านมา
        back_dir = self._deg_to_cardinal((self.heading_deg + 180) % 360)
        self.maze_walls.update_wall_sensor(
            self.robot_x, self.robot_y, back_dir, is_wall=False, weight=2.0)

        # สแกนและอัปเดตอัตโนมัติเมื่อถึงช่องใหม่
        self.perform_scan_and_update(status_reason=reason)
        return True

    def move_backward_one_cell(self, reason="REVERSE"):
        """ถอยหลัง 1 ช่อง (60 cm) โดยไม่หันกลับ เพื่อออกจากทางตันอย่างรวดเร็ว"""
        dx, dy = 0, 0
        norm_h = int(round(self.heading_deg / 90.0) * 90) % 360
        # ถอยหลัง = เดินไปในทิศตรงข้ามของ Heading
        if norm_h == 0:
            dy = -1
        elif norm_h == 90:
            dx = -1
        elif norm_h == 180:
            dy = 1
        elif norm_h == 270:
            dx = 1

        next_x = self.robot_x + dx
        next_y = self.robot_y + dy

        if not self.ogm.is_valid_cell(next_x, next_y):
            print(f"[Move] [WARN] ไม่สามารถถอยหลังได้: ชนขอบเขตสนาม 4x4! (เป้าหมาย: {next_x}, {next_y})")
            return False

        current_speed = self.dashboard.move_speed
        print(f"[Move] [BACKWARD] ถอยหลัง 1 ช่อง ({config.STEP_DISTANCE_M*100:.0f} cm) @ {current_speed:.2f} m/s ไปที่ ({next_x}, {next_y})...")

        if not self.use_mock and self.ep_robot:
            try:
                # ถอยหลัง ให้ความเร็ว x เป็นลบ
                self.ep_robot.chassis.move(
                    x=-config.STEP_DISTANCE_M, y=0, z=0,
                    xy_speed=current_speed
                ).wait_for_completed()
            except Exception as e:
                print(f"[Move] [ERROR] ความผิดพลาดในการถอยหลัง: {e}")
                return False
        else:
            time.sleep(config.STEP_DISTANCE_M / max(current_speed, 0.05) * 0.5)

        self.robot_x = next_x
        self.robot_y = next_y
        self.path_history.append((self.robot_x, self.robot_y))

        # หลังจากถอยหลังเสร็จแล้ว อัปเดตข้อมูลเซนเซอร์ในช่องใหม่
        self.perform_scan_and_update(status_reason=reason)
        return True

    def _scan_unknown_sides_only(self, cell):
        """
        สแกนเฉพาะด้านที่ยังไม่รู้ค่ากำแพง (P ใกล้ 0.50):
        หมุนไปยังทิศที่ Unknown เท่านั้น → ลดจำนวนการหมุนลง = เร็วขึ้น
        """
        orig_heading = self.heading_deg
        cx, cy = cell

        # หาทิศที่ยังไม่รู้
        unknown_dirs = []
        for d in ["NORTH", "EAST", "SOUTH", "WEST"]:
            p = self.maze_walls.get_wall_prob(cx, cy, d)
            if config.THRESHOLD_FREE < p < config.THRESHOLD_OCC:
                unknown_dirs.append(d)

        dir_to_deg = {"NORTH": 0, "EAST": 90, "SOUTH": 180, "WEST": 270}
        print(f"  [SmartScan] \u0e2a\u0e41\u0e01\u0e19\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e17\u0e34\u0e28\u0e17\u0e35\u0e48\u0e44\u0e21\u0e48\u0e23\u0e39\u0e49: {unknown_dirs}")

        for target_dir in unknown_dirs:
            target_hdg = dir_to_deg[target_dir]
            # หมุนไปยังทิศที่ต้องการ
            turns = 0
            while self.heading_deg != target_hdg and turns < 4:
                diff = (target_hdg - self.heading_deg) % 360
                if diff == 90 or diff == 180:
                    self.turn_right(reason=f"SMART_SCAN_{target_dir}", use_side_ir=False)
                else:
                    self.turn_left(reason=f"SMART_SCAN_{target_dir}", use_side_ir=False)
                turns += 1
            if not self.use_mock:
                time.sleep(0.10)
            self.perform_scan_and_update(status_reason=f"SMART_SCAN_{target_dir}", use_side_ir=False)

        # หมุนกลับสู่ทิศเดิม
        while self.heading_deg != orig_heading:
            diff = (orig_heading - self.heading_deg) % 360
            if diff == 90 or diff == 180:
                self.turn_right(reason="SMART_SCAN_RETURN", use_side_ir=False)
            else:
                self.turn_left(reason="SMART_SCAN_RETURN", use_side_ir=False)

        self.scanned_cells.add(cell)

    def turn_left(self, reason="TURNED_LEFT", use_side_ir=True, do_scan=True):
        """เลี้ยวซ้าย 90° ตามความเร็วหมุนใน Dashboard"""
        current_turn_speed = self.dashboard.turn_speed
        print(f"[Move] [TURN-L] กำลังเลี้ยวซ้าย 90° (ความเร็ว: {current_turn_speed:.1f} °/s)...")
        if not self.use_mock and self.ep_robot:
            try:
                self.ep_robot.chassis.move(x=0, y=0, z=90, z_speed=current_turn_speed).wait_for_completed()
                time.sleep(0.1)
            except Exception as e:
                print(f"[Move] [ERROR] ความผิดพลาดในการเลี้ยว: {e}")
        else:
            time.sleep(0.20)

        self.heading_deg = (self.heading_deg - 90) % 360
        print(f"[Move] ปัจจุบันหันหน้าทิศ Heading: {self.heading_deg}°")
        if do_scan:
            return self.perform_scan_and_update(status_reason=reason, use_side_ir=use_side_ir)
        return None

    def turn_right(self, reason="TURNED_RIGHT", use_side_ir=True, do_scan=True):
        """เลี้ยวขวา 90° ตามความเร็วหมุนใน Dashboard"""
        current_turn_speed = self.dashboard.turn_speed
        print(f"[Move] [TURN-R] กำลังเลี้ยวขวา 90° (ความเร็ว: {current_turn_speed:.1f} °/s)...")
        if not self.use_mock and self.ep_robot:
            try:
                self.ep_robot.chassis.move(x=0, y=0, z=-90, z_speed=current_turn_speed).wait_for_completed()
                time.sleep(0.1)
            except Exception as e:
                print(f"[Move] [ERROR] ความผิดพลาดในการเลี้ยว: {e}")
        else:
            time.sleep(0.20)

        self.heading_deg = (self.heading_deg + 90) % 360
        print(f"[Move] ปัจจุบันหันหน้าทิศ Heading: {self.heading_deg}°")
        if do_scan:
            return self.perform_scan_and_update(status_reason=reason, use_side_ir=use_side_ir)
        return None

    def scan_cell_360(self, passes=None):
        """
        หมุนหุ่นยนต์สแกนตรวจสอบกำแพงรอบด้าน 360° แบบหลายรอบ (Multi-Pass Confirmation)
        ตามที่ผู้ใช้ร้องขอ:
        - แต่ละทิศทาง: ใช้เฉพาะ Front ToF ที่หันไปวัดระยะโดยตรง (ไม่ใช้ Side IR ที่อาจรบกวนค่า)
        - ToF หยุดนิ่งแล้วอ่านค่าหลายครั้งพร้อม Median Filter
        - หมุนตรวจสอบ 2 รอบ (Pass 1 และ Pass 2): เพื่อ Double-Check ป้องกันรายงานเจอกำแพงทั้งที่เป็นพื้นที่ว่าง
        - เมื่อทั้ง 2 รอบยืนยันตรงกัน จึงสรุปผลกำแพงและอัปเดตลงทั้ง MazeWalls และ OGM
        - หมุนกลับสู่ทิศทางเดิมเสมอ
        """
        if passes is None:
            passes = getattr(self.dashboard, "scan_passes", 1)

        cell = (self.robot_x, self.robot_y)
        orig_heading = self.heading_deg
        print("\n" + "=" * 65)
        if passes == 1:
            print(f"  [SCAN 360°] เริ่มหมุนกวาดเช็คกำแพง 1 รอบ (4 ทิศ) ณ ช่อง {cell} (Heading: {orig_heading}°)")
        else:
            print(f"  [SCAN 360°] เริ่มหมุนกวาดเช็คกำแพง {passes} รอบ ณ ช่อง {cell} (Heading: {orig_heading}°)")
        print("=" * 65)

        pass_results = []  # pass_results[pass_i][card_dir] = (dist_cm, is_wall)

        for p_idx in range(1, passes + 1):
            if passes > 1:
                print(f"\n>>> [PASS {p_idx}/{passes}] กำลังสแกนรอบที่ {p_idx}...")
            dir_data = {}

            # ทิศที่ 1 (ทิศเริ่มต้น) - ใช้เฉพาะ Front ToF ตรงหน้า ไม่ใช้ Side IR รบกวน
            if not self.use_mock:
                time.sleep(0.15)
            step_lbl = f"SCAN P{p_idx} [1/4]" if passes > 1 else "SCAN [1/4]"
            d1 = self.perform_scan_and_update(status_reason=step_lbl, use_side_ir=False)
            card1 = self._deg_to_cardinal(self.heading_deg)
            f_cm1 = d1["front_dist_cm"] if d1 else 999.0
            f_w1 = d1["front_is_wall"] if d1 else False
            dir_data[card1] = (f_cm1, f_w1)
            print(f"      [{card1:5s}] ToF: {f_cm1:5.1f} cm -> [{'WALL' if f_w1 else 'CLEAR'}]")

            # ทิศที่ 2, 3, 4 (หมุนขวา 90° ทีละทิศ)
            for step_i in range(2, 5):
                turn_lbl = f"SCAN P{p_idx} [{step_i}/4]" if passes > 1 else f"SCAN [{step_i}/4]"
                # 1. หมุนไปยังทิศถัดไปก่อน (ไม่ scan ทันทีซ้ำซ้อน)
                self.turn_right(reason=turn_lbl, use_side_ir=False, do_scan=False)
                if not self.use_mock:
                    time.sleep(0.15)
                # 2. อ่านค่า ToF หลังหมุนเสร็จแล้ว (heading อัปเดตแล้วแน่นอน)
                card_i = self._deg_to_cardinal(self.heading_deg)
                di = self.perform_scan_and_update(status_reason=f"{turn_lbl}_READ", use_side_ir=False)
                f_cmi = di["front_dist_cm"] if di else 999.0
                f_wi = di["front_is_wall"] if di else False
                dir_data[card_i] = (f_cmi, f_wi)
                print(f"      [{card_i:5s}] ToF: {f_cmi:5.1f} cm -> [{'WALL' if f_wi else 'CLEAR'}]")

            # หมุนอีก 90° เพื่อกลับสู่ทิศเดิมของ Pass นี้
            reset_lbl = f"SCAN P{p_idx} [RESET]" if passes > 1 else "SCAN [RESET]"
            self.turn_right(reason=reset_lbl, use_side_ir=False, do_scan=False)
            pass_results.append(dir_data)

        # -----------------------------------------------------------------
        # รวมผลและยืนยันค่ากำแพง (Verification / Summary)
        # -----------------------------------------------------------------
        print("\n" + "=" * 65)
        if passes == 1:
            print(f"  [SUMMARY] สรุปผลการตรวจกำแพง 4 ทิศ ณ ช่อง {cell}:")
        else:
            print(f"  [VERIFICATION SUMMARY] สรุปผลการตรวจกำแพง {passes} รอบ ณ ช่อง {cell}:")
        print("=" * 65)

        for d in ["NORTH", "EAST", "SOUTH", "WEST"]:
            p1_dist, p1_wall = pass_results[0].get(d, (999.0, False))
            if passes >= 2:
                p2_dist, p2_wall = pass_results[1].get(d, (999.0, False))
                confirmed_wall = (p1_wall and p2_wall)
                status_str = "CONFIRMED WALL" if confirmed_wall else "CONFIRMED CLEAR"
                print(f"  - {d:5s} : Pass1={p1_dist:5.1f}cm ({'WALL' if p1_wall else 'CLEAR'}) | Pass2={p2_dist:5.1f}cm ({'WALL' if p2_wall else 'CLEAR'})  ==> [{status_str}]")
            else:
                confirmed_wall = p1_wall
                status_str = "WALL" if confirmed_wall else "CLEAR"
                print(f"  - {d:5s} : ToF={p1_dist:5.1f} cm ==> [{status_str}]")

            # คำนวณพิกัดช่องข้างเคียง
            dx = 1 if d == "EAST" else -1 if d == "WEST" else 0
            dy = 1 if d == "NORTH" else -1 if d == "SOUTH" else 0
            nx = self.robot_x + dx
            ny = self.robot_y + dy

            # อัปเดตตารางกำแพง MazeWalls ด้วยค่าที่ผ่านการยืนยัน Double-Check
            self.maze_walls.update_wall_sensor(self.robot_x, self.robot_y, d, is_wall=confirmed_wall, weight=3.0)

            # อัปเดตแผนที่ตาราง OGM สำหรับช่องข้างเคียงโดยตรงด้วยค่าที่ยืนยันแล้ว
            if self.ogm.is_valid_cell(nx, ny):
                self.ogm.update_cell_status(nx, ny, is_occupied=confirmed_wall, weight=3.0)

        # เซลล์ปัจจุบันที่หุ่นยนต์ยืนอยู่ต้องเป็น Free เสมอ
        self.ogm.update_cell_status(self.robot_x, self.robot_y, is_occupied=False, weight=3.0)

        self.scanned_cells.add(cell)
        print("=" * 65 + "\n")
        self.perform_scan_and_update(status_reason="SCAN_CONFIRMED", use_side_ir=False)

    def run_auto_demonstration(self):
        """
        รันเส้นทางสำรวจตัวอย่างอัตโนมัติ:
        (0,0) -> (0,1) -> (0,2) -> หมุนเลี้ยวขวา -> (1,2) -> (2,2) -> ...
        """
        print("\n" + "=" * 55)
        print("  [AUTO] เริ่มต้นการทดลองสำรวจอัตโนมัติ (AUTO EXPLORATION)")
        print("=" * 55)

        commands = ["w", "w", "d", "w", "w", "d", "w", "a", "w"]
        for cmd in commands:
            print(f"\n[Auto Plan] กำลังรันคำสั่ง: {cmd.upper()}")
            if cmd == "w":
                self.move_forward_one_cell()
            elif cmd == "a":
                self.turn_left()
            elif cmd == "d":
                self.turn_right()
            time.sleep(0.8)

    def navigate_to_cell(self, target_x, target_y, reason="EXPLORING"):
        """
        หมุนหุ่นยนต์และก้าวเดินไปยังช่องเป้าหมาย (target_x, target_y)
        ระบบ IR Avoidance ทำงานอัตโนมัติใน move_forward_one_cell:
          - ถ้า IR ตรวจพบกำแพงด้านหน้าก่อนเดิน → ยกเลิก และรายงาน False
          - Caller (run_autonomous_exploration) จะ trigger การ re-scan แล้วเลือกทางใหม่
        """
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y

        target_heading = 0
        if dx == 0 and dy == 1:
            target_heading = 0     # North
        elif dx == 1 and dy == 0:
            target_heading = 90    # East
        elif dx == 0 and dy == -1:
            target_heading = 180   # South
        elif dx == -1 and dy == 0:
            target_heading = 270   # West

        turn_diff = (target_heading - self.heading_deg) % 360
        if turn_diff == 90:
            self.turn_right(reason=f"{reason}_TURN_R", use_side_ir=True)
        elif turn_diff == 270:
            self.turn_left(reason=f"{reason}_TURN_L", use_side_ir=True)
        elif turn_diff == 180:
            # เป้าหมายอยู่ด้านหลัง → ถอยหลัง 1 ช่องโดยไม่ต้องหมุนตัว
            return self.move_backward_one_cell(reason=reason)

        # เดินไปข้างหน้า พร้อม IR avoidance real-time
        success = self.move_forward_one_cell(reason=reason, use_ir_avoidance=True)

        if not success:
            # IR ตรวจพบกำแพง → ลบช่องนี้ออกจาก scanned_cells เพื่อบังคับ re-scan
            current = (self.robot_x, self.robot_y)
            if current in self.scanned_cells:
                self.scanned_cells.discard(current)
                print(f"[IR-Avoid] ล้าง scan cache ที่ {current} เพื่อบังคับ re-scan ใหม่")

        return success

    def run_autonomous_exploration(self, max_steps=50):
        """
        สำรวจเขาวงกตที่ไม่รู้แผนที่มาก่อนโดยอัตโนมัติ (Autonomous Exploration & Exit Finding)
        เพื่อค้นหาทางออกเป้าหมาย (Exit Cell) พร้อมอัปเดต OGM และขอบกำแพงแบบ Real-time
        """
        print("\n" + "=" * 60)
        print("  [AUTONOMOUS EXPLORATION] เริ่มภารกิจสำรวจหาทางออกอัตโนมัติ")
        print(f"  จุดเริ่มต้น: ({self.robot_x}, {self.robot_y}) ---> เป้าหมายทางออก (Exit): {self.explorer.exit_cell}")
        print("=" * 60)

        step = 0
        while step < max_steps:
            step += 1
            current = (self.robot_x, self.robot_y)

            # ตรวจจับปุ่มกดคีย์บอร์ดหรือหน้าต่าง OpenCV เพื่อสั่งหยุดชั่วคราว
            k = cv2.waitKey(20) & 0xFF
            if k in [ord('q'), ord('Q'), 27]:
                print("\n[AI Explorer] ผู้ใช้สั่งหยุดการสำรวจ (Paused/Stopped by user)")
                return False

            # ซิงค์ Exit Cell หากผู้ใช้คลิกเปลี่ยนบน Dashboard
            if self.explorer.exit_cell != self.dashboard.exit_cell:
                print(f"[AI Explorer] ผู้ใช้เปลี่ยน Exit Cell เป็น {self.dashboard.exit_cell}")
                self.explorer.set_exit_cell(self.dashboard.exit_cell)

            # ตรวจสอบว่าถึงทางออกหรือยัง
            if self.explorer.is_at_exit(self.robot_x, self.robot_y):
                print(f"\n[SUCCESS] ถึงเป้าหมายทางออกที่ {current} เรียบร้อยแล้ว! (ใช้ไป {step-1} ก้าว)")
                self.perform_scan_and_update(status_reason=f"GOAL_REACHED at {current}")
                return True

            # ─────────────────────────────────────────────────────────────────
            # สแกน 360° เฉพาะเมื่อจำเป็น:
            # ถ้าเพิ่งเดินเข้ามาจากทิศที่รู้แล้ว ทิศนั้น+ตรงข้ามรู้แล้ว
            # → สแกนเฉพาะ 2 ด้านที่ยังไม่รู้ (หมุนน้อยลง = เร็วขึ้น)
            # ─────────────────────────────────────────────────────────────────
            if current not in self.scanned_cells:
                # นับจำนวนด้านที่รู้ค่ากำแพงแน่ชัดแล้ว (P > 0.60 หรือ P < 0.40)
                known_sides = 0
                for chk_dir in ["NORTH", "EAST", "SOUTH", "WEST"]:
                    p = self.maze_walls.get_wall_prob(current[0], current[1], chk_dir)
                    if p >= config.THRESHOLD_OCC or p <= config.THRESHOLD_FREE:
                        known_sides += 1

                if known_sides >= 3:
                    # รู้ 3 ด้านแล้ว → สแกนด้านที่ 4 เพียงด้านเดียว ไม่ต้องหมุน 360°
                    print(f"[Explorer] ช่อง {current}: รู้ {known_sides}/4 ด้านแล้ว → อ่านเซนเซอร์ตรงหน้าเท่านั้น")
                    self.perform_scan_and_update(status_reason="QUICK_SCAN", use_side_ir=True)
                    self.scanned_cells.add(current)
                elif known_sides >= 2:
                    # รู้ 2 ด้าน → สแกนเฉพาะด้านที่ยังไม่รู้ ลดจาก 4 หมุน → 2 หมุน
                    print(f"[Explorer] ช่อง {current}: รู้ {known_sides}/4 ด้านแล้ว → สแกนครึ่งรอบ (2 ทิศ)")
                    self._scan_unknown_sides_only(current)
                else:
                    # ยังไม่รู้เลย → สแกนเต็ม 360° 1 รอบ
                    print(f"[Explorer] ช่อง {current}: ยังไม่มีข้อมูล → สแกน 360° เต็มรอบ")
                    self.scan_cell_360(passes=1)

            next_cell, reason = self.explorer.decide_next_move(current, self.maze_walls)
            print(f"\n[AI Explorer Step #{step}] ณ {current} -> ตัดสินใจ: {reason}")

            if next_cell is None:
                if reason == "GOAL_REACHED":
                    print(f"\n[SUCCESS] บรรลุเป้าหมายทางออกเรียบร้อย!")
                    self.perform_scan_and_update(status_reason=f"GOAL_REACHED at {current}")
                    return True
                else:
                    print(f"\n[AI Explorer] [WARN] สำรวจครบทุกช่องแล้ว ไม่สามารถเดินต่อได้: {reason}")
                    return False

            # สั่งหุ่นยนต์เดินไปยังช่องถัดไป
            success = self.navigate_to_cell(next_cell[0], next_cell[1], reason=reason)
            if not success:
                print(f"[AI Explorer] [WARN] ไม่สามารถเดินไป {next_cell} ได้ (พบสิ่งกีดขวาง)")
                # ให้ AI ตัดสินใจใหม่ในรอบถัดไป
                continue

            time.sleep(0.3)

        return self.explorer.is_at_exit(self.robot_x, self.robot_y)

    def save_all_results(self):
        """บันทึกผลลัพธ์ทั้งหมดลงไฟล์ในโฟลเดอร์ lab6"""
        csv_file = os.path.join(current_dir, "ogm_result.csv")
        txt_file = os.path.join(current_dir, "ogm_log.txt")
        png_file = os.path.join(current_dir, "ogm_result.png")

        print("\n" + "=" * 55)
        print("  [SAVE] กำลังบันทึกผลการทดลอง Lab 6...")
        print("=" * 55)

        self.ogm.save_to_csv(csv_file)
        self.ogm.save_log_file(txt_file)
        self.ogm.save_plot_image(png_file, robot_path=self.path_history, maze_walls=self.maze_walls)

        print("\n[DONE] ผลลัพธ์ทั้งหมดถูกบันทึกเรียบร้อย:")
        print(f"  1. ตารางความน่าจะเป็น CSV : {csv_file}")
        print(f"  2. บันทึกผลการทดลอง TXT    : {txt_file}")
        print(f"  3. แผนที่ความร้อน PNG      : {png_file}")

    def cleanup(self):
        """ปิดการเชื่อมต่อหุ่นยนต์และปิดหน้าต่าง Dashboard อย่างปลอดภัย"""
        self.dashboard.close()
        if self.ep_robot:
            try:
                print("\n[Main] กำลังปิดการเชื่อมต่อกับ RoboMaster...")
                # Unsubscribe sensor before close to avoid SDK __del__ error
                try:
                    self.ep_robot.sensor.unsub_distance()
                except Exception:
                    pass
                self.ep_robot.close()
                print("[Main] ปิดการเชื่อมต่อเรียบร้อย")
            except Exception:
                pass
            finally:
                self.ep_robot = None


def main():
    parser = argparse.ArgumentParser(description="Lab 6: Occupancy Grid Mapping & Autonomous Exploration")
    parser.add_argument("--real", action="store_true", help="Run on real robot via Wi-Fi AP")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation mode")
    parser.add_argument("--auto", action="store_true", help="Run fixed auto demo sequence immediately")
    parser.add_argument("--explore", action="store_true", help="Run autonomous maze exploration & exit finding")
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Open Dashboard SETUP screen first.
    # User picks: REAL ROBOT or SIMULATION, start cell, exit cell, speed.
    # Robot mode (real/mock) is decided by button click — not by CLI flag.
    # CLI flags --real / --mock are still accepted as shortcuts:
    #   --real  -> pre-select REAL ROBOT mode
    #   --mock  -> pre-select SIMULATION mode
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("      LAB 6: OCCUPANCY GRID MAPPING (OGM) 4x4")
    print("      RoboMaster EP -- Bayes' Rule & Autonomous Maze Exploration")
    print("=" * 60)

    # Create a temporary dashboard just for setup (use_mock unknown yet)
    setup_dash = __import__('dashboard').InteractiveDashboard()
    # Pre-select mode if CLI flag given
    if args.real:
        setup_dash.robot_mode = "REAL"
    elif args.mock:
        setup_dash.robot_mode = "MOCK"
    else:
        setup_dash.robot_mode = None   # user must choose on screen

    print("[Main] Dashboard SETUP — choose mode, start & exit, then press START")
    while True:
        setup_dash.render_setup()
        act = setup_dash.poll_action()
        if act == "BTN_MODE_REAL":
            setup_dash.robot_mode = "REAL"
            print("[Setup] Mode -> REAL ROBOT")
        elif act == "BTN_MODE_MOCK":
            setup_dash.robot_mode = "MOCK"
            print("[Setup] Mode -> SIMULATION")
        elif act == "BTN_START_RUN":
            if setup_dash.robot_mode is None:
                print("[Setup] Please select REAL ROBOT or SIMULATION first!")
                continue
            setup_dash.phase = "RUNNING"
            print("[Main] START pressed — Mode=%s" % setup_dash.robot_mode)
            break
        k = cv2.waitKey(25) & 0xFF
        if k in [27, ord('q'), ord('Q')]:
            print("[Main] Setup cancelled.")
            setup_dash.close()
            return

    use_mock = (setup_dash.robot_mode == "MOCK")
    print("[Main] Mode: %s" % ('SIMULATION' if use_mock else 'REAL ROBOT (Wi-Fi AP)'))

    # Create main experiment with correct mode, copy setup choices
    app = Lab6Experiment(use_mock=use_mock)
    app.dashboard = setup_dash        # reuse the same window
    app.explorer  = __import__('explorer').AutonomousExplorer(
        exit_cell=setup_dash.exit_cell)

    # Apply chosen start & exit cells
    app.apply_setup_choices()

    # ── PHASE 2: Connect robot ────────────────────────────────────────────────
    if not app.initialize_system():
        print("[Main] [ERROR] ไม่สามารถเริ่มต้นระบบได้ ยกเลิกการทดลอง")
        app.dashboard.close()
        return

    # ── PHASE 3: Initial Scan & Autonomous Exploration ───────────────────────
    # หมุนสแกน 360° ตรวจสอบกำแพงรอบด้าน 4 ทิศ 1 รอบ ณ จุดเริ่มต้น
    app.scan_cell_360(passes=1)

    if args.auto:
        app.run_auto_demonstration()
        app.save_all_results()
        app.cleanup()
        return

    # เริ่มต้นเดินทางสำรวจเขาวงกตเพื่อหาทางออกโดยอัตโนมัติต่อเนื่องทันที
    print("\n" + "=" * 65)
    print("  [Main] เริ่มภารกิจเดินทางสำรวจหาทางออกอัตโนมัติ (Autonomous Exploration)...")
    print("=" * 65)
    app.run_autonomous_exploration()
    app.save_all_results()

    # Interactive Command Loop (สามารถควบคุมต่อด้วยปุ่ม หรือกด Q เพื่อจบโปรแกรม)
    print("\n" + "-" * 60)
    print("  [Main] ภารกิจสำรวจเสร็จสิ้น! คุณสามารถสั่งการต่อ หรือกด [Q / ESC] เพื่อปิดโปรแกรม:")
    print("    - [W] เดินหน้า 1 ช่อง | [A] เลี้ยวซ้าย | [D] เลี้ยวขวา | [S] สแกน 360°")
    print("    - [E] สั่ง AI สำรวจต่อ | [P] บันทึกผลลัพธ์ | [Q / ESC] จบการทดลอง")
    print("-" * 60 + "\n")
    print("        [W]       : เดินหน้า 1 ช่อง (60 cm)")
    print("        [A]       : หมุนเลี้ยวซ้าย 90°")
    print("        [D]       : หมุนเลี้ยวขวา 90°")
    print("        [S]       : หมุนสแกนกำแพงรอบช่อง 360° (Full 360° Wall Scan)")
    print("        [E]       : ให้หุ่นสำรวจเขาวงกตเพื่อหาทางออกเองอัตโนมัติ")
    print("        [AUTO]    : รันเส้นทางสำรวจตัวอย่างอัตโนมัติ")
    print("        [P]       : พล็อตและบันทึกผลแผนที่ทันที")
    print("        [Q / ESC] : จบการทดลองและบันทึกผลลัพธ์ทั้งหมด")
    print("-" * 60 + "\n")

    import msvcrt

    try:
        while True:
            # 1. ตรวจสอบ Action ที่คลิกผ่านปุ่มบนหน้าจอ Dashboard
            btn_act = app.dashboard.poll_action()
            cmd = None

            if btn_act:
                if btn_act == "BTN_EXPLORE":
                    cmd = "e"
                elif btn_act == "BTN_STEP_FWD":
                    cmd = "w"
                elif btn_act == "BTN_TURN_L":
                    cmd = "a"
                elif btn_act == "BTN_TURN_R":
                    cmd = "d"
                elif btn_act == "BTN_SCAN":
                    cmd = "s"
                elif btn_act == "BTN_SAVE":
                    cmd = "p"
                elif btn_act.startswith("SET_EXIT_"):
                    # Refresh dashboard to highlight new Exit cell
                    app.perform_scan_and_update(status_reason=f"TARGET_CHANGED -> {app.dashboard.exit_cell}")

            # 2. ตรวจสอบปุ่มกดจากหน้าต่าง OpenCV Dashboard
            k = cv2.waitKey(25) & 0xFF
            if k in [ord('w'), ord('W')]:
                cmd = "w"
            elif k in [ord('a'), ord('A')]:
                cmd = "a"
            elif k in [ord('d'), ord('D')]:
                cmd = "d"
            elif k in [ord('s'), ord('S')]:
                cmd = "s"
            elif k in [ord('e'), ord('E')]:
                cmd = "e"
            elif k in [ord('p'), ord('P')]:
                cmd = "p"
            elif k in [ord('q'), ord('Q'), 27]: # 27 = ESC
                cmd = "q"

            # 3. ตรวจสอบปุ่มกดจาก Terminal (Non-blocking)
            if cmd is None and msvcrt.kbhit():
                try:
                    ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if ch in ["w", "a", "d", "s", "e", "p", "q"]:
                        cmd = ch
                except Exception:
                    pass

            # 4. ประมวลผลคำสั่ง
            if cmd == "w":
                app.move_forward_one_cell()
            elif cmd == "a":
                app.turn_left()
            elif cmd == "d":
                app.turn_right()
            elif cmd == "s":
                app.scan_cell_360()
            elif cmd == "e":
                app.run_autonomous_exploration()
            elif cmd == "p":
                app.save_all_results()
            elif cmd == "q":
                print("\n[Main] กำลังจบการทดลอง...")
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[Main] ตรวจพบการกด Ctrl+C ยุติการทดลอง...")

    finally:
        app.save_all_results()
        app.cleanup()
        print("\n[DONE] การทดลอง Lab 6 เสร็จสมบูรณ์แล้ว!")


if __name__ == "__main__":
    main()
