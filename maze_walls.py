# -*- coding: utf-8 -*-
"""
maze_walls.py — Wall-Edge Representation for 4x4 Maze Grid.
Tracks walls on each edge (North, East, South, West) of every grid cell
using Bayesian Log-Odds updates.
"""

import math
import numpy as np
import config


class MazeWalls:
    """
    ระบบเก็บข้อมูลกำแพงรอบแต่ละช่องตาราง (Edge-based Maze Model):
      - ช่องตารางขนาด 4x4 (16 ช่อง)
      - แต่ละช่องมี 4 ขอบ: NORTH, EAST, SOUTH, WEST
      - ขอบร่วมกันจะซิงค์ข้อมูลถึงกัน (เช่น NORTH ของ (x, y) คือ SOUTH ของ (x, y+1))
      - ขอบนอกสุดของสนาม (Perimeter) ถูกกำหนดให้เป็นกำแพงเสมอ
    """

    def __init__(self, width=config.GRID_WIDTH, height=config.GRID_HEIGHT):
        self.width = width
        self.height = height

        # กำแพงแนวนอน (Horizontal Edges): ขนาด (height + 1, width)
        # h_walls[y, x] คือ ขอบใต้ของแถว y (หรือขอบเหนือของแถว y-1)
        # ค่าคือ Log-odds (0.0 = Unknown P=0.5, >0 = Wall, <0 = Open)
        self.h_log_odds = np.zeros((self.height + 1, self.width), dtype=np.float64)

        # กำแพงแนวตั้ง (Vertical Edges): ขนาด (height, width + 1)
        # v_log_odds[y, x] คือ ขอบซ้ายของคอลัมน์ x (หรือขอบขวาของคอลัมน์ x-1)
        self.v_log_odds = np.zeros((self.height, self.width + 1), dtype=np.float64)

        # ตั้งค่าขอบนอกสนามให้เป็นกำแพงแน่นอน (+5.0 log-odds)
        self.h_log_odds[0, :] = 5.0              # ขอบใต้สุด (South outer border)
        self.h_log_odds[self.height, :] = 5.0    # ขอบเหนือสุด (North outer border)
        self.v_log_odds[:, 0] = 5.0              # ขอบซ้ายสุด (West outer border)
        self.v_log_odds[:, self.width] = 5.0     # ขอบขวาสุด (East outer border)

        self.l_occ = math.log(config.P_OCCUPIED / (1.0 - config.P_OCCUPIED))   # ~ +1.735
        self.l_free = math.log(config.P_FREE / (1.0 - config.P_FREE))          # ~ -0.619

    def _get_edge_ref(self, x, y, direction):
        """
        แปลง (x, y, direction) ไปเป็นตัวชี้ตำแหน่งในตาราง h_log_odds หรือ v_log_odds
        คืนค่า: (is_horizontal, row, col)
        """
        dir_upper = direction.upper()
        if dir_upper == "NORTH":
            return True, y + 1, x
        elif dir_upper == "SOUTH":
            return True, y, x
        elif dir_upper == "EAST":
            return False, y, x + 1
        elif dir_upper == "WEST":
            return False, y, x
        return None

    def update_wall_sensor(self, x, y, direction, is_wall: bool, weight: float = 1.0):
        """
        อัปเดตสถานะกำแพงที่ขอบของช่อง (x, y) ในทิศทางที่กำหนดด้วย Bayesian Log-Odds
        """
        ref = self._get_edge_ref(x, y, direction)
        if not ref:
            return

        is_h, r, c = ref
        # ขอบนอกสนามห้ามแก้ (เป็นกำแพงถาวร)
        if is_h and (r == 0 or r == self.height):
            return
        if not is_h and (c == 0 or c == self.width):
            return

        delta = weight * (self.l_occ if is_wall else self.l_free)

        if is_h:
            self.h_log_odds[r, c] = np.clip(self.h_log_odds[r, c] + delta, config.L_CLAMP_MIN, config.L_CLAMP_MAX)
        else:
            self.v_log_odds[r, c] = np.clip(self.v_log_odds[r, c] + delta, config.L_CLAMP_MIN, config.L_CLAMP_MAX)

    def get_wall_prob(self, x, y, direction):
        """คืนค่าความน่าจะเป็น P(wall) ของขอบในทิศทางที่กำหนด (0.0 ถึง 1.0)"""
        ref = self._get_edge_ref(x, y, direction)
        if not ref:
            return 1.0
        is_h, r, c = ref
        l_val = self.h_log_odds[r, c] if is_h else self.v_log_odds[r, c]
        return 1.0 / (1.0 + np.exp(-l_val))

    def is_wall_blocked(self, x, y, direction):
        """มีกำแพงกั้นหรือไม่ (P > 0.55) — Unknown (P=0.50) ไม่นับเป็นกำแพง"""
        return self.get_wall_prob(x, y, direction) > 0.55

    def is_wall_open(self, x, y, direction):
        """เป็นช่องเปิดโล่งเดินผ่านได้หรือไม่ (P < 0.40)"""
        return self.get_wall_prob(x, y, direction) <= config.THRESHOLD_FREE

    def can_move(self, x, y, direction):
        """
        ตรวจสอบว่าสามารถก้าวข้ามจาก (x, y) ไปยังทิศทางนั้นได้หรือไม่:
          - ต้องไม่ชนกำแพงที่ตรวจพบแล้ว (P < 0.55)
          - ช่องปลายทางต้องอยู่ในขอบเขตสนาม 4x4
        """
        dx, dy = 0, 0
        d = direction.upper()
        if d == "NORTH": dy = 1
        elif d == "EAST": dx = 1
        elif d == "SOUTH": dy = -1
        elif d == "WEST": dx = -1

        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return False

        return not self.is_wall_blocked(x, y, direction)

    def reset_inner_walls(self):
        """รีเซ็ตกำแพงด้านในทั้งหมดให้เป็น Unknown (0.0 log-odds)"""
        self.h_log_odds[1:self.height, :] = 0.0
        self.v_log_odds[:, 1:self.width] = 0.0
