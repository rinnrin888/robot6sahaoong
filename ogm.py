# -*- coding: utf-8 -*-
"""
ogm.py — Occupancy Grid Mapping (OGM) Engine
Implemented using Bayesian updates with Log-Odds representation.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt

import config


class OccupancyGridMap:
    """
    ตารางความน่าจะเป็นแบบ Occupancy Grid Map (ขนาด 4 x 4)
    ใช้ตัวแทน Log-Odds ในการอัปเดตแบบเบย์ (Bayesian Update)
    """

    def __init__(self, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, cell_size_cm=config.CELL_SIZE_CM):
        self.width = width
        self.height = height
        self.cell_size_cm = cell_size_cm

        # Log-odds array: l(m) = ln( P / (1 - P) )
        # ค่าเริ่มต้น Prior P = 0.50 -> l = ln(1.0) = 0.0
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float64)

        # ค่าคงที่ Log-odds สำหรับ Inverse Sensor Model
        self.l_free = math.log(config.P_FREE / (1.0 - config.P_FREE))          # ~ -0.619
        self.l_occ  = math.log(config.P_OCCUPIED / (1.0 - config.P_OCCUPIED))  # ~ +1.735
        self.l_prior = math.log(config.P_PRIOR / (1.0 - config.P_PRIOR))       # 0.0

        # บันทึกประวัติการสำรวจ
        self.history = []

    def log_odds_to_probability(self, l_val):
        """แปลง Log-Odds กลับมาเป็นความน่าจะเป็น P in [0, 1]"""
        return 1.0 / (1.0 + np.exp(-l_val))

    def probability_to_log_odds(self, p_val):
        """แปลงค่าความน่าจะเป็น P มาเป็น Log-Odds"""
        p_val = np.clip(p_val, 1e-4, 1.0 - 1e-4)
        return np.log(p_val / (1.0 - p_val))

    def get_probabilities(self):
        """คืนค่า Matrix ความน่าจะเป็น P(m) ขนาด 4 x 4"""
        return self.log_odds_to_probability(self.log_odds)

    def is_valid_cell(self, x, y):
        """ตรวจสอบว่าพิกัด (x, y) อยู่ในขอบเขต 4 x 4 หรือไม่"""
        return 0 <= x < self.width and 0 <= y < self.height

    def update_cell_status(self, x, y, is_occupied: bool, weight: float = 1.0):
        """
        อัปเดต Log-Odds ของเซลล์ (x, y) ด้วย Bayes Rule
        l_t = l_{t-1} + weight * (l_sensor - l_prior)
        """
        if not self.is_valid_cell(x, y):
            return

        l_sensor = self.l_occ if is_occupied else self.l_free
        delta = weight * (l_sensor - self.l_prior)
        
        # Bayesian log-odds update
        new_l = self.log_odds[y, x] + delta
        # Clamp เพื่อความเสถียรของตัวเลข
        self.log_odds[y, x] = np.clip(new_l, config.L_CLAMP_MIN, config.L_CLAMP_MAX)

    def update_direction_sensor(self, robot_x, robot_y, heading_deg, sensor_rel_angle, distance_cm):
        """
        อัปเดต OGM จากการอ่านค่าเซนเซอร์ในทิศทางสัมพัทธ์กับตัวหุ่นยนต์:
          - sensor_rel_angle: 0° (หน้า), -90° (ซ้าย), +90° (ขวา)
          - distance_cm: ระยะที่อ่านได้ (None หรือ float)
        """
        if distance_cm is None:
            return

        # คำนวณทิศทางจริงใน Grid (Global Heading)
        # Heading 0° = +Y (North), 90° = +X (East), 180° = -Y (South), 270° = -X (West)
        global_angle = (heading_deg + sensor_rel_angle) % 360

        # แปลงเป็นเวกเตอร์ช่อง (dx, dy)
        if 315 <= global_angle or global_angle < 45:
            dx, dy = 0, 1   # North (+Y)
        elif 45 <= global_angle < 135:
            dx, dy = 1, 0   # East (+X)
        elif 135 <= global_angle < 225:
            dx, dy = 0, -1  # South (-Y)
        else:
            dx, dy = -1, 0  # West (-X)

        target_x = robot_x + dx
        target_y = robot_y + dy

        # พิจารณาเกณฑ์ระยะสิ่งกีดขวาง (Wall Detect Threshold)
        # ถ้าเจอกำแพงในระยะ 1 ช่อง (< 55 ซม.) แสดงว่าช่องข้างๆ นั้นเป็น Occupied
        # หากระยะมากกว่า 55 ซม. หรือไม่มีสิ่งกีดขวาง แสดงว่าช่องข้างๆ เป็น Free
        is_occupied = (distance_cm < config.WALL_DETECT_CM)

        if self.is_valid_cell(target_x, target_y):
            self.update_cell_status(target_x, target_y, is_occupied=is_occupied)

        # เซลล์ที่หุ่นยนต์ยืนอยู่ ย่อมเป็น Free แน่นอน
        self.update_cell_status(robot_x, robot_y, is_occupied=False, weight=1.5)

    def update_direction_binary(self, robot_x, robot_y, heading_deg, sensor_rel_angle, is_occupied: bool):
        """
        อัปเดต OGM โดยตรงจาก Digital IR I/O Sensor (True = มีกำแพง, False = โล่ง)
        """
        global_angle = (heading_deg + sensor_rel_angle) % 360

        if 315 <= global_angle or global_angle < 45:
            dx, dy = 0, 1   # North (+Y)
        elif 45 <= global_angle < 135:
            dx, dy = 1, 0   # East (+X)
        elif 135 <= global_angle < 225:
            dx, dy = 0, -1  # South (-Y)
        else:
            dx, dy = -1, 0  # West (-X)

        target_x = robot_x + dx
        target_y = robot_y + dy

        if self.is_valid_cell(target_x, target_y):
            self.update_cell_status(target_x, target_y, is_occupied=is_occupied)

        # เซลล์ปัจจุบันที่หุ่นยนต์ยืนอยู่ต้องเป็น Free
        self.update_cell_status(robot_x, robot_y, is_occupied=False, weight=1.5)

    def print_ascii_map(self, robot_x=None, robot_y=None, heading_deg=0):
        """
        แสดงแผนที่ OGM 4x4 เป็นตาราง ASCII บน Terminal
        แสดงค่าความน่าจะเป็น P และตำแหน่งของหุ่นยนต์
        """
        probs = self.get_probabilities()
        heading_icons = {0: "^", 90: ">", 180: "v", 270: "<"}
        # หาทิศที่ใกล้เคียงที่สุด
        norm_head = int(round(heading_deg / 90.0) * 90) % 360
        r_icon = heading_icons.get(norm_head, "R")

        print("\n" + "=" * 55)
        print("         4x4 OCCUPANCY GRID MAP (OGM) DISPLAY")
        print("=" * 55)
        print("  Y ^  Legend: [ . ] Free  [ ? ] Unknown  [###] Occupied")
        print("    |")

        for y in range(self.height - 1, -1, -1):
            row_str = f" {y}  | "
            for x in range(self.width):
                p = probs[y, x]
                is_robot = (robot_x == x and robot_y == y)

                if is_robot:
                    cell_str = f" [R{r_icon}] "
                elif p >= config.THRESHOLD_OCC:
                    cell_str = f" [#{p:.2f}]"
                elif p <= config.THRESHOLD_FREE:
                    cell_str = f" [ {p:.2f}]"
                else:
                    cell_str = f" [?{p:.2f}]"

                row_str += cell_str
            print(row_str)

        print("    +-----------------------------------------------")
        print("        (0,y)   (1,y)   (2,y)   (3,y)   --> X\n")

    def save_to_csv(self, filepath="ogm_result.csv"):
        """บันทึก Matrix ความน่าจะเป็น P ลงไฟล์ CSV"""
        probs = self.get_probabilities()
        np.savetxt(filepath, probs, delimiter=",", fmt="%.4f",
                   header="Occupancy Probabilities 4x4 (Row: Y=0..3, Col: X=0..3)", comments="")
        print(f"[OGM] [OK] บันทึกข้อมูล CSV เรียบร้อย: {filepath}")

    def save_log_file(self, filepath="ogm_log.txt"):
        """บันทึกประวัติการเดินและการอัปเดตลงใน Text File"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== LAB 6: OCCUPANCY GRID MAPPING LOG ===\n")
            f.write(f"Grid Size: {self.width}x{self.height} cells (Cell: {self.cell_size_cm}cm)\n\n")
            
            probs = self.get_probabilities()
            f.write("--- FINAL PROBABILITY MATRIX ---\n")
            for y in range(self.height - 1, -1, -1):
                row = " ".join([f"{probs[y, x]:.3f}" for x in range(self.width)])
                f.write(f"Y={y} | {row}\n")
            
            f.write("\n--- CLASSIFIED CELL STATUS ---\n")
            for y in range(self.height - 1, -1, -1):
                row_status = []
                for x in range(self.width):
                    p = probs[y, x]
                    status = "OCCUPIED" if p >= config.THRESHOLD_OCC else ("FREE" if p <= config.THRESHOLD_FREE else "UNKNOWN")
                    row_status.append(f"({x},{y}):{status}")
                f.write(" | ".join(row_status) + "\n")

            f.write("\n--- STEP HISTORY ---\n")
            for entry in self.history:
                f.write(f"{entry}\n")
        print(f"[OGM] [OK] บันทึกบันทึกการทดลอง TXT เรียบร้อย: {filepath}")

    def save_plot_image(self, filepath="ogm_result.png", robot_path=None, maze_walls=None):
        """สร้างภาพ Heatmap แผนที่ 4x4 พร้อมขอบกำแพงรอบด้าน (Wall Edges) และเส้นทางการเดิน"""
        probs = self.get_probabilities()

        fig, ax = plt.subplots(figsize=(8, 7))
        
        # สร้าง heatmap โดยใช้ colormap RdYlBu_r (แดง = Occupied, น้ำเงิน = Free, เหลือง = 0.5 Unknown)
        cax = ax.imshow(probs, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, origin="lower")
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Occupancy Probability P(m)", fontsize=11)

        # ตั้งค่าแกน Grid
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels([f"X={x}" for x in range(self.width)])
        ax.set_yticklabels([f"Y={y}" for y in range(self.height)])
        ax.set_title("RoboMaster EP -- 4x4 Grid OGM & Wall Edges", fontsize=13, pad=12, fontweight="bold")
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

        # เขียนข้อความค่าความน่าจะเป็นในแต่ละช่อง
        for y in range(self.height):
            for x in range(self.width):
                p = probs[y, x]
                label_text = f"P={p:.2f}\n"
                if p >= config.THRESHOLD_OCC:
                    label_text += "WALL"
                    color = "white"
                elif p <= config.THRESHOLD_FREE:
                    label_text += "FREE"
                    color = "white"
                else:
                    label_text += "UNK"
                    color = "black"

                ax.text(x, y, label_text, ha="center", va="center", color=color, fontweight="bold", fontsize=9)

        # วาดขอบกำแพงรอบช่อง (Wall Edges: North, East, South, West)
        if maze_walls is not None:
            for y in range(self.height):
                for x in range(self.width):
                    # North
                    pn = maze_walls.get_wall_prob(x, y, "NORTH")
                    if pn >= 0.50:
                        ax.plot([x - 0.48, x + 0.48], [y + 0.5, y + 0.5], color="red", linewidth=4, zorder=4)
                    elif pn <= config.THRESHOLD_FREE:
                        ax.plot([x - 0.40, x + 0.40], [y + 0.5, y + 0.5], color="limegreen", linewidth=1.5, linestyle=":", zorder=4)

                    # South
                    ps = maze_walls.get_wall_prob(x, y, "SOUTH")
                    if ps >= 0.50:
                        ax.plot([x - 0.48, x + 0.48], [y - 0.5, y - 0.5], color="red", linewidth=4, zorder=4)
                    elif ps <= config.THRESHOLD_FREE:
                        ax.plot([x - 0.40, x + 0.40], [y - 0.5, y - 0.5], color="limegreen", linewidth=1.5, linestyle=":", zorder=4)

                    # East
                    pe = maze_walls.get_wall_prob(x, y, "EAST")
                    if pe >= 0.50:
                        ax.plot([x + 0.5, x + 0.5], [y - 0.48, y + 0.48], color="red", linewidth=4, zorder=4)
                    elif pe <= config.THRESHOLD_FREE:
                        ax.plot([x + 0.5, x + 0.5], [y - 0.40, y + 0.40], color="limegreen", linewidth=1.5, linestyle=":", zorder=4)

                    # West
                    pw = maze_walls.get_wall_prob(x, y, "WEST")
                    if pw >= 0.50:
                        ax.plot([x - 0.5, x - 0.5], [y - 0.48, y + 0.48], color="red", linewidth=4, zorder=4)
                    elif pw <= config.THRESHOLD_FREE:
                        ax.plot([x - 0.5, x - 0.5], [y - 0.40, y + 0.40], color="limegreen", linewidth=1.5, linestyle=":", zorder=4)

            # วาดกรอบนอกสุดรอบสนาม
            ax.plot([-0.5, 3.5, 3.5, -0.5, -0.5], [-0.5, -0.5, 3.5, 3.5, -0.5], color="darkred", linewidth=3, zorder=4)

        # วาดเส้นทางการเดินของหุ่นยนต์ (Robot Path) ถ้ามี
        if robot_path and len(robot_path) > 1:
            px = [p[0] for p in robot_path]
            py = [p[1] for p in robot_path]
            ax.plot(px, py, color="darkorange", linewidth=2.8, linestyle="-", marker="o", markersize=7, label="Robot Trajectory", zorder=6)
            # เน้นจุดเริ่มต้นและจุดสิ้นสุด
            ax.scatter([px[0]], [py[0]], color="lime", s=130, zorder=7, edgecolors="black", label="Start (0,0)")
            ax.scatter([px[-1]], [py[-1]], color="gold", s=130, zorder=7, edgecolors="black", label=f"Current ({px[-1]},{py[-1]})")
            ax.legend(loc="upper right", framealpha=0.9)

        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(-0.6, 3.6)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close(fig)
        print(f"[OGM] [OK] บันทึกภาพแผนที่ PNG เรียบร้อย: {filepath}")
