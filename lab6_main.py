# -*- coding: utf-8 -*-
"""
lab6_main.py — Main Experiment Runner for Lab 6 (Occupancy Grid Mapping)
RoboMaster EP 4x4 Grid OGM using Bayesian Log-Odds Updates.

Navigation logic matches occupancy_grid_mapping.py exactly:
  1. mark_visited(x, y) -> A* plan -> pick next_node
  2. Turn to face next_node
  3. Front ToF check:
     - < WALL_DETECT_CM: set_edge = -1 (wall), update_cell OCC, replan
     - >= WALL_DETECT_CM: set_edge = 1 (open), move forward 60cm with Yaw PID
  4. After exploring all reachable cells -> navigate to Goal (G)

Usage:
  - Simulation:   python lab6_main.py --mock
  - Real robot:   python lab6_main.py --real
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
from ogm import OccupancyGridMap, DIRS
from sensors import SensorManager
from explorer import a_star_planner, heading_deg_to_idx, heading_idx_to_deg
from dashboard import InteractiveDashboard


class Lab6Experiment:
    """
    Main experiment controller for Lab 6.
    Navigation logic matches occupancy_grid_mapping.py exactly.
    """

    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.ep_robot = None
        self.ogm = OccupancyGridMap()
        self.dashboard = InteractiveDashboard()

        # Robot state
        self.robot_x = 0
        self.robot_y = 0
        self.heading_deg = 0
        self.initial_yaw_offset = 0.0

        # Trajectory
        self.path_history = [(self.robot_x, self.robot_y)]
        self.step_counter = 0

        # Sensors
        self.sensor_mgr = None

    def apply_setup_choices(self):
        """Apply start_cell/goal_cell chosen in Dashboard Setup phase."""
        sx, sy = self.dashboard.start_cell
        self.robot_x = sx
        self.robot_y = sy
        self.path_history = [(sx, sy)]
        config.EXIT_CELL = self.dashboard.goal_cell
        print("[Setup] Start=(%d,%d)  Goal=(%d,%d)" % (sx, sy,
              self.dashboard.goal_cell[0], self.dashboard.goal_cell[1]))

    def initialize_system(self):
        """Connect to robot and setup hardware."""
        if not self.use_mock:
            print(f"[Main] [CONNECT] Connecting to RoboMaster EP via Wi-Fi ({config.CONN_TYPE.upper()})...")
            try:
                from robomaster import robot
                self.ep_robot = robot.Robot()
                self.ep_robot.initialize(conn_type=config.CONN_TYPE)
                print("[Main] [OK] Robot connected!")
            except Exception as e:
                print(f"[Main] [ERROR] Cannot connect: {e}")
                print("[Main] Use --mock for simulation mode")
                return False

        self.sensor_mgr = SensorManager(ep_robot=self.ep_robot, mock=self.use_mock)
        ok = self.sensor_mgr.setup_hardware()
        if ok and not self.use_mock:
            time.sleep(0.5)
            self.initial_yaw_offset = self.sensor_mgr.get_current_yaw()
            print(f"[Main] [OK] Initial IMU Yaw Offset = {self.initial_yaw_offset:.2f} deg")
        return ok

    # ══════════════════════════════════════════════════════════════════════
    #  Movement Helpers (matching occupancy_grid_mapping.py)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_angle(angle):
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _deg_to_cardinal(deg):
        d = int(round(deg / 90.0) * 90) % 360
        return {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}.get(d, "NORTH")

    def _heading_to_target_yaw(self, heading_deg):
        norm_h = int(round(heading_deg / 90.0) * 90) % 360
        return {0: 0.0, 90: 90.0, 180: 180.0, 270: -90.0}.get(norm_h, 0.0)

    def turn_to_heading(self, target_heading_deg):
        """Turn robot to face a specific heading (0/90/180/270).
        Matching RobotController.turn_to() from occupancy_grid_mapping.py."""
        target_heading_deg = int(round(target_heading_deg / 90.0) * 90) % 360
        if self.heading_deg == target_heading_deg:
            return

        diff = (target_heading_deg - self.heading_deg) % 360
        current_turn_speed = self.dashboard.turn_speed

        if not self.use_mock and self.ep_robot:
            try:
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.1)
                if diff == 90:
                    self.ep_robot.chassis.move(x=0, y=0, z=-90, z_speed=current_turn_speed).wait_for_completed()
                elif diff == 180:
                    self.ep_robot.chassis.move(x=0, y=0, z=180, z_speed=current_turn_speed).wait_for_completed()
                elif diff == 270:
                    self.ep_robot.chassis.move(x=0, y=0, z=90, z_speed=current_turn_speed).wait_for_completed()
                time.sleep(0.3)
            except Exception as e:
                print(f"[Move] [ERROR] Turn error: {e}")
        else:
            time.sleep(0.2)

        self.heading_deg = target_heading_deg
        print(f"[Move] Heading -> {self.heading_deg} deg ({self._deg_to_cardinal(self.heading_deg)})")

    def move_forward_pid(self):
        """Move forward 60cm with IMU Yaw PID + IR lateral steering.
        Matching RobotController.move_forward_pid() from occupancy_grid_mapping.py."""
        heading_idx = heading_deg_to_idx(self.heading_deg)
        dx, dy = DIRS[heading_idx]
        next_x = self.robot_x + dx
        next_y = self.robot_y + dy

        if not self.ogm.is_valid_cell(next_x, next_y):
            print(f"[Move] [WARN] Out of bounds: ({next_x}, {next_y})")
            return False

        fwd_speed = self.dashboard.move_speed
        target_dist = config.STEP_DISTANCE_M
        duration = target_dist / max(fwd_speed, 0.05)

        print(f"[Move] [FWD] Moving {target_dist*100:.0f}cm @ {fwd_speed:.2f}m/s -> ({next_x},{next_y})")

        if not self.use_mock and self.ep_robot:
            t_start = time.monotonic()
            last_yaw_err = 0.0
            target_yaw = self.initial_yaw_offset + self._heading_to_target_yaw(self.heading_deg)

            try:
                while time.monotonic() - t_start < duration:
                    # Emergency stop if very close wall
                    tof_now = self.sensor_mgr.read_front_tof()
                    if tof_now is not None and 0 < tof_now < 15.0:
                        print(f"  [Steer] EMERGENCY STOP (ToF={tof_now:.1f}cm)")
                        break

                    # Read IR for lateral steering (0 = Wall, 1 = Clear)
                    l_val, l_wall = self.sensor_mgr.read_left_io()
                    r_val, r_wall = self.sensor_mgr.read_right_io()

                    # IMU Yaw PID
                    current_yaw = self.sensor_mgr.get_current_yaw()
                    yaw_error = self._normalize_angle(target_yaw - current_yaw)
                    d_yaw = yaw_error - last_yaw_err
                    z_speed = (config.KP_YAW * yaw_error) + (config.KD_YAW * d_yaw)
                    last_yaw_err = yaw_error

                    # Digital IR steering nudge (NOT for wall mapping!)
                    vy = 0.0
                    if l_wall and not r_wall:
                        vy = -0.05  # Wall on left -> nudge right
                    elif r_wall and not l_wall:
                        vy = 0.05   # Wall on right -> nudge left

                    vy = max(min(vy, config.MAX_CORRECTION_Y), -config.MAX_CORRECTION_Y)
                    self.ep_robot.chassis.drive_speed(x=fwd_speed, y=vy, z=z_speed)
                    time.sleep(0.02)
            except Exception as e:
                print(f"[Move] [ERROR] drive_speed error: {e}")
            finally:
                try:
                    self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                    time.sleep(0.12)
                except Exception:
                    pass
        else:
            # Mock mode
            time.sleep(duration * 0.3)

        # Update position
        self.robot_x = next_x
        self.robot_y = next_y
        self.path_history.append((self.robot_x, self.robot_y))
        return True

    # ══════════════════════════════════════════════════════════════════════
    #  Dashboard / Scan
    # ══════════════════════════════════════════════════════════════════════

    def read_ir_quick(self):
        """Quick IR read for display/steering info."""
        if self.use_mock or not self.ep_robot:
            data = self.sensor_mgr.read_all_sensors(self.robot_x, self.robot_y, self.heading_deg)
            return data["left_io"], data["right_io"], data["left_is_wall"], data["right_is_wall"]
        l_val, l_wall = self.sensor_mgr.read_left_io()
        r_val, r_wall = self.sensor_mgr.read_right_io()
        return l_val, r_val, l_wall, r_wall

    def update_dashboard(self, mode_label="Exploring..."):
        """Render dashboard with current state."""
        ir_l, ir_r, _, _ = self.read_ir_quick()
        tof_cm = self.sensor_mgr.sample_front_tof(num_samples=3, sample_interval=0.03) if self.sensor_mgr else 999.0

        sensor_data = {
            "front_dist_cm": tof_cm,
            "front_is_wall": tof_cm < config.WALL_DETECT_CM,
            "left_io": ir_l,
            "left_is_wall": (ir_l == config.IO_WALL_VALUE),
            "right_io": ir_r,
            "right_is_wall": (ir_r == config.IO_WALL_VALUE),
        }

        self.dashboard.render(
            ogm=self.ogm,
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            heading_deg=self.heading_deg,
            path_history=self.path_history,
            sensor_data=sensor_data,
            explorer_status=mode_label,
            mode_str="MOCK SIMULATION" if self.use_mock else "REAL ROBOT",
            step_count=self.step_counter
        )

    # ══════════════════════════════════════════════════════════════════════
    #  A* Exploration Loop (matching occupancy_grid_mapping.py main loop)
    # ══════════════════════════════════════════════════════════════════════

    def run_autonomous_exploration(self, max_steps=80):
        """
        Autonomous exploration loop matching occupancy_grid_mapping.py exactly:
        1. mark_visited(x, y)
        2. A* plan path to nearest unvisited cell
        3. If no unvisited -> switch to "goal" mode toward Goal cell
        4. Take next_node = path[1], turn to face it
        5. ToF check: wall -> set_edge -1, replan; clear -> set_edge 1, move
        """
        goal_cell = self.dashboard.goal_cell
        is_going_to_goal = False

        print("\n" + "=" * 60)
        print("  [AUTONOMOUS EXPLORATION] A* Matching occupancy_grid_mapping.py")
        print(f"  Start: ({self.robot_x}, {self.robot_y}) -> Goal (G): {goal_cell}")
        print("=" * 60)

        step = 0
        while step < max_steps:
            step += 1
            self.step_counter += 1
            x, y = self.robot_x, self.robot_y
            heading_idx = heading_deg_to_idx(self.heading_deg)

            # Mark current cell as visited (sets to FREE)
            self.ogm.mark_visited(x, y)

            # Read sensors for logging/display (IR for steering only)
            ir_l, ir_r, _, _ = self.read_ir_quick()
            tof_f = self.sensor_mgr.sample_front_tof(num_samples=3, sample_interval=0.03) if self.sensor_mgr else 999.0
            self.ogm.history.append(
                f"Step {self.step_counter}: Pos=({x},{y}) Heading={self.heading_deg} | "
                f"ToF={tof_f:.1f}cm | IR L={ir_l} R={ir_r} (Steering Only)"
            )

            # Print ASCII map
            self.ogm.print_ascii_map(self.robot_x, self.robot_y, self.heading_deg)

            # Check keyboard for stop
            k = cv2.waitKey(20) & 0xFF
            if k in [ord('q'), ord('Q'), 27]:
                print("\n[Explorer] User stopped exploration")
                return False

            # Sync goal cell if user clicked
            if self.dashboard.goal_cell != goal_cell:
                goal_cell = self.dashboard.goal_cell
                print(f"[Explorer] Goal changed to {goal_cell}")

            # ─── Phase 1: Explore ─────────────────────────────────────────
            if not is_going_to_goal:
                mode_lbl = "Exploration (A*)"
                self.update_dashboard(mode_lbl)

                path = a_star_planner(self.ogm, x, y, heading_idx, target_mode="explore")

                if not path:
                    print("\n[SUCCESS] All reachable cells explored! Navigating to Goal...")
                    is_going_to_goal = True

            # ─── Phase 2: Navigate to Goal ────────────────────────────────
            if is_going_to_goal:
                mode_lbl = "Navigate to Goal (A*)"
                self.update_dashboard(mode_lbl)

                if (x, y) == goal_cell:
                    print(f"\n[MISSION COMPLETE] Arrived at Goal {goal_cell}!")
                    self.update_dashboard("GOAL REACHED!")
                    return True

                path = a_star_planner(self.ogm, x, y, heading_idx,
                                      target_mode="goal", target_pos=goal_cell)
                if not path:
                    print(f"\n[ERROR] Cannot find path to Goal {goal_cell}!")
                    return False

            # ─── Execute next step ─────────────────────────────────────────
            next_node = path[1]
            target_dir_idx = DIRS.index((next_node[0] - x, next_node[1] - y))
            target_heading_deg = heading_idx_to_deg(target_dir_idx)

            # 1. Turn to face next_node
            self.turn_to_heading(target_heading_deg)
            self.update_dashboard(mode_lbl)

            # 2. Front ToF multi-round verify (ตัด noise ด้วย majority vote)
            #    - รอ stabilize หลังหมุน 0.3 วินาที
            #    - อ่าน 3 รอบ (แต่ละรอบ median 5 samples)
            #    - Majority vote: wall >= 2/3 รอบ → wall, otherwise → clear
            #    - ถ้า borderline (ค่าใกล้ threshold ±8cm) → อ่านเพิ่มอีก 2 รอบ ยืนยัน
            if not self.use_mock:
                time.sleep(0.3)  # Stabilization delay หลังหมุนเสร็จ

            NUM_ROUNDS = 3
            tof_readings = []
            for rd in range(NUM_ROUNDS):
                t = self.sensor_mgr.sample_front_tof(num_samples=5, sample_interval=0.03)
                tof_readings.append(t)
                if not self.use_mock:
                    time.sleep(0.05)  # gap ระหว่าง round

            wall_votes = sum(1 for t in tof_readings if t < config.WALL_DETECT_CM)
            tof_median = sorted(tof_readings)[len(tof_readings) // 2]

            # Borderline check: ถ้าค่าใกล้ threshold (±8cm) อ่านเพิ่มอีก 2 รอบ
            borderline_low = config.WALL_DETECT_CM - 8.0
            borderline_high = config.WALL_DETECT_CM + 8.0
            if borderline_low < tof_median < borderline_high:
                print(f"  [ToF] Borderline ({tof_median:.1f}cm ~{config.WALL_DETECT_CM}cm) -> extra 2 rounds...")
                if not self.use_mock:
                    time.sleep(0.15)
                for rd in range(2):
                    t = self.sensor_mgr.sample_front_tof(num_samples=5, sample_interval=0.03)
                    tof_readings.append(t)
                    if t < config.WALL_DETECT_CM:
                        wall_votes += 1
                    if not self.use_mock:
                        time.sleep(0.05)
                tof_median = sorted(tof_readings)[len(tof_readings) // 2]

            total_rounds = len(tof_readings)
            is_wall = wall_votes >= (total_rounds // 2 + 1)  # majority
            tof_verify = tof_median

            # 🛡️ Safety Override: ถ้ามีค่าใดอ่านได้ชิดมาก (<= 15.0cm เช่น 5.0cm)
            # แสดงว่าเซนเซอร์อยู่ประชิดกำแพงจริง ห้ามหลงเชื่อค่า 999.0 (timeout/blind zone) เด็ดขาด!
            if any(t <= 15.0 for t in tof_readings):
                is_wall = True
                tof_verify = min(tof_readings)
                print(f"  [ToF Safety] ตรวจพบระยะชิดกำแพง ({tof_verify:.1f}cm <= 15cm) -> บังคับตัดสินเป็น WALL!")

            print(f"  [ToF] {total_rounds} rounds: {[f'{t:.1f}' for t in tof_readings]} "
                  f"-> median={tof_verify:.1f}cm, wall_votes={wall_votes}/{total_rounds} "
                  f"-> {'WALL' if is_wall else 'CLEAR'}")

            if is_wall:
                # ─── WALL DETECTED ─────────────────────────────────────────
                print(f"[WALL] ToF={tof_verify:.1f}cm -> Wall at {next_node}. "
                      f"Recording edge & replanning...")
                self.ogm.set_edge((x, y), next_node, -1)   # Wall!
                self.ogm.update_cell(next_node[0], next_node[1], True)   # OCC
                self.ogm.history.append(
                    f"Step {self.step_counter}: BLOCKED ({x},{y})->{next_node} "
                    f"ToF={tof_verify:.1f}cm votes={wall_votes}/{total_rounds}"
                )
                self.update_dashboard(f"WALL at {next_node}")
                continue   # Replan immediately, don't move
            else:
                # ─── PATH CLEAR ────────────────────────────────────────────
                print(f"[CLEAR] ToF={tof_verify:.1f}cm -> Open to {next_node}. Moving...")
                self.ogm.set_edge((x, y), next_node, 1)   # Open!
                self.move_forward_pid()
                self.update_dashboard(mode_lbl)

            time.sleep(0.2)

        return (self.robot_x, self.robot_y) == goal_cell

    # ══════════════════════════════════════════════════════════════════════
    #  Results & Cleanup
    # ══════════════════════════════════════════════════════════════════════

    def save_all_results(self):
        """Save all results to lab6 directory."""
        csv_file = os.path.join(current_dir, "ogm_result.csv")
        txt_file = os.path.join(current_dir, "ogm_log.txt")
        png_file = os.path.join(current_dir, "ogm_result.png")

        print("\n" + "=" * 55)
        print("  [SAVE] Saving Lab 6 results...")
        print("=" * 55)

        self.ogm.save_to_csv(csv_file)
        self.ogm.save_log_file(txt_file)
        self.ogm.save_plot_image(png_file, robot_path=self.path_history)

        print(f"\n[DONE] Results saved:")
        print(f"  1. CSV : {csv_file}")
        print(f"  2. LOG : {txt_file}")
        print(f"  3. PNG : {png_file}")

    def cleanup(self):
        """Close robot connection and dashboard."""
        self.dashboard.close()
        if self.ep_robot:
            try:
                print("\n[Main] Closing robot connection...")
                try:
                    self.ep_robot.sensor.unsub_distance()
                except Exception:
                    pass
                try:
                    self.ep_robot.chassis.unsub_attitude()
                except Exception:
                    pass
                self.ep_robot.close()
                print("[Main] Connection closed")
            except Exception:
                pass
            finally:
                self.ep_robot = None


def main():
    parser = argparse.ArgumentParser(description="Lab 6: OGM & A* Autonomous Exploration")
    parser.add_argument("--real", action="store_true", help="Run on real robot via Wi-Fi AP")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation mode")
    args = parser.parse_args()

    # ── PHASE 1: Setup Dashboard — pick mode, start, goal ─────────────────
    print("=" * 60)
    print("      LAB 6: OCCUPANCY GRID MAPPING (OGM) 4x4")
    print("      A* Navigation matching occupancy_grid_mapping.py")
    print("=" * 60)

    setup_dash = InteractiveDashboard()
    if args.real:
        setup_dash.robot_mode = "REAL"
    elif args.mock:
        setup_dash.robot_mode = "MOCK"
    else:
        setup_dash.robot_mode = None

    print("[Main] Dashboard SETUP — choose mode, Start (S) & Goal (G), then press START")
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
    print("[Main] Mode: %s" % ('SIMULATION' if use_mock else 'REAL ROBOT'))

    # ── Create experiment ─────────────────────────────────────────────────
    app = Lab6Experiment(use_mock=use_mock)
    app.dashboard = setup_dash
    app.apply_setup_choices()

    # ── PHASE 2: Connect robot ────────────────────────────────────────────
    if not app.initialize_system():
        print("[Main] [ERROR] System init failed")
        app.dashboard.close()
        return

    import signal
    def safe_exit(sig, frame):
        print("\n[WARN] Ctrl+C detected -> Saving...")
        app.save_all_results()
        app.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, safe_exit)

    # ── PHASE 3: A* Autonomous Exploration ────────────────────────────────
    app.update_dashboard("STARTING...")
    app.run_autonomous_exploration()
    app.save_all_results()

    # ── Interactive Command Loop ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  [Main] Exploration complete! Manual control available:")
    print("    [W] Forward | [A] Turn L | [D] Turn R | [E] Explore again")
    print("    [P] Save | [Q / ESC] Quit")
    print("-" * 60 + "\n")

    import msvcrt

    try:
        while True:
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
                elif btn_act == "BTN_SAVE":
                    cmd = "p"
                elif btn_act.startswith("SET_GOAL_"):
                    app.update_dashboard(f"Goal changed -> {app.dashboard.goal_cell}")

            k = cv2.waitKey(25) & 0xFF
            if k in [ord('w'), ord('W')]:
                cmd = "w"
            elif k in [ord('a'), ord('A')]:
                cmd = "a"
            elif k in [ord('d'), ord('D')]:
                cmd = "d"
            elif k in [ord('e'), ord('E')]:
                cmd = "e"
            elif k in [ord('p'), ord('P')]:
                cmd = "p"
            elif k in [ord('q'), ord('Q'), 27]:
                cmd = "q"

            if cmd is None and msvcrt.kbhit():
                try:
                    ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if ch in ["w", "a", "d", "e", "p", "q"]:
                        cmd = ch
                except Exception:
                    pass

            if cmd == "w":
                # Manual forward: check ToF first
                heading_idx = heading_deg_to_idx(app.heading_deg)
                nx = app.robot_x + DIRS[heading_idx][0]
                ny = app.robot_y + DIRS[heading_idx][1]
                tof = app.sensor_mgr.sample_front_tof(num_samples=3, sample_interval=0.03)
                if tof < config.WALL_DETECT_CM:
                    print(f"[Manual] Wall ahead (ToF={tof:.1f}cm)")
                    app.ogm.set_edge((app.robot_x, app.robot_y), (nx, ny), -1)
                else:
                    app.ogm.set_edge((app.robot_x, app.robot_y), (nx, ny), 1)
                    app.move_forward_pid()
                    app.ogm.mark_visited(app.robot_x, app.robot_y)
                app.update_dashboard("Manual Control")
            elif cmd == "a":
                app.heading_deg = (app.heading_deg - 90) % 360
                if not app.use_mock and app.ep_robot:
                    try:
                        app.ep_robot.chassis.move(x=0, y=0, z=90, z_speed=app.dashboard.turn_speed).wait_for_completed()
                    except Exception:
                        pass
                app.update_dashboard("Manual Control")
            elif cmd == "d":
                app.heading_deg = (app.heading_deg + 90) % 360
                if not app.use_mock and app.ep_robot:
                    try:
                        app.ep_robot.chassis.move(x=0, y=0, z=-90, z_speed=app.dashboard.turn_speed).wait_for_completed()
                    except Exception:
                        pass
                app.update_dashboard("Manual Control")
            elif cmd == "e":
                app.run_autonomous_exploration()
                app.save_all_results()
            elif cmd == "p":
                app.save_all_results()
            elif cmd == "q":
                print("\n[Main] Ending experiment...")
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected")

    finally:
        app.save_all_results()
        app.cleanup()
        print("\n[DONE] Lab 6 complete!")


if __name__ == "__main__":
    main()
