# -*- coding: utf-8 -*-
"""
dashboard.py -- Interactive Real-Time Dashboard for Lab 6.

Phase 1 (SETUP):
  - User clicks START cell (green) and GOAL cell (cyan) on the 4x4 grid.
  - Buttons: [SET START] [SET GOAL] [START EXPLORATION]
  - Robot does NOT move yet.

Phase 2 (RUNNING):
  - Full telemetry: wall edges from ogm.get_edge(), sensor bars, AI status.
  - Buttons: EXPLORE, STEP FWD, TURN L/R, SAVE.
  - Live speed/threshold config via +/- buttons.
  - Click grid cell -> change goal target.
"""

import math
import numpy as np
import cv2
import config

# ── Colours (BGR) ────────────────────────────────────────────────────────────
C_BG_DARK    = (24,  20,  18)
C_BG_PANEL   = (38,  30,  26)
C_BG_CELL    = (34,  28,  26)
C_WALL_CONF  = (40,  40, 220)
C_OPEN       = (50, 180,  50)
C_UNKNOWN    = (70,  60,  55)
C_BORDER     = (200,  60,  30)
C_ROBOT      = (0,  140, 255)
C_HEADER     = (240, 240, 240)
C_ACCENT     = (0,  215, 255)
C_BTN_GREEN  = (40, 120,  50)
C_BTN_BLUE   = (50,  80, 130)
C_BTN_GRAY   = (55,  48,  42)
C_BTN_WARN   = (30,  80, 160)
C_BTN_ACT    = (0,  200, 100)
C_START      = (0,  200, 100)
C_GOAL_C     = (0,  215, 255)

# Golden Yellow #FDCB6E in BGR (matching occupancy_grid_mapping.py)
C_GOLDEN_WALL = (110, 203, 253)


class InteractiveDashboard:
    """Interactive real-time dashboard.  Two phases: SETUP and RUNNING."""

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self, window_name="RoboMaster Lab6 -- A* OGM Explorer"):
        self.window_name = window_name
        self.W, self.H = 1060, 700
        self.enabled = True

        # Grid pixel layout
        self.gox = 60     # left edge of cell (0,0)
        self.goy = 610    # bottom edge of row 0
        self.cpx = 120    # cell side in pixels

        # ── Phase: "SETUP" or "RUNNING" ──────────────────────────────────────
        self.phase = "SETUP"
        self.set_target = "GOAL"           # which cell is being placed: "START" or "GOAL"
        self.start_cell = (0, 0)           # chosen by user in SETUP
        self.goal_cell  = config.EXIT_CELL # Goal (formerly "exit")
        self.robot_mode = None             # "REAL" or "MOCK"

        # ── Live config (RUNNING phase) ────────────────────────────────────────
        self.move_speed     = config.MOVE_SPEED_MPS
        self.turn_speed     = config.TURN_SPEED_DPS
        self.wall_detect_cm = config.WALL_DETECT_CM
        self.io_wall_value  = config.IO_WALL_VALUE

        self.requested_action = None
        self.buttons = []

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self._on_mouse)
        except Exception:
            self.enabled = False

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def _on_mouse(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Grid cell click
        for gx in range(4):
            for gy in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    if self.phase == "SETUP":
                        if self.set_target == "START":
                            self.start_cell = (gx, gy)
                            print("[Setup] Start cell -> (%d,%d)" % (gx, gy))
                        else:
                            self.goal_cell = (gx, gy)
                            config.EXIT_CELL = (gx, gy)
                            print("[Setup] Goal cell -> (%d,%d)" % (gx, gy))
                    else:
                        # RUNNING: change goal target
                        self.goal_cell = (gx, gy)
                        config.EXIT_CELL = (gx, gy)
                        self.requested_action = "SET_GOAL_%d_%d" % (gx, gy)
                        print("[Dashboard] Goal target -> (%d,%d)" % (gx, gy))
                    return

        # Button click
        for (bx1, by1, bx2, by2, act) in self.buttons:
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                self._handle(act)
                return

    def _handle(self, action):
        if action == "SET_START_MODE":
            self.set_target = "START"
            print("[Setup] Click a cell to set START position")
            return
        if action == "SET_GOAL_MODE":
            self.set_target = "GOAL"
            print("[Setup] Click a cell to set GOAL position")
            return
        if action == "BTN_START_RUN":
            self.requested_action = "BTN_START_RUN"
            return

        # Running phase config buttons
        if action == "SPEED_DOWN":
            self.move_speed = max(0.05, round(self.move_speed - 0.05, 2))
            config.MOVE_SPEED_MPS = self.move_speed
        elif action == "SPEED_UP":
            self.move_speed = min(0.60, round(self.move_speed + 0.05, 2))
            config.MOVE_SPEED_MPS = self.move_speed
        elif action == "TURN_DOWN":
            self.turn_speed = max(10.0, round(self.turn_speed - 5.0, 1))
            config.TURN_SPEED_DPS = self.turn_speed
        elif action == "TURN_UP":
            self.turn_speed = min(120.0, round(self.turn_speed + 5.0, 1))
            config.TURN_SPEED_DPS = self.turn_speed
        elif action == "WDET_DOWN":
            self.wall_detect_cm = max(20.0, round(self.wall_detect_cm - 5.0, 0))
            config.WALL_DETECT_CM = self.wall_detect_cm
        elif action == "WDET_UP":
            self.wall_detect_cm = min(100.0, round(self.wall_detect_cm + 5.0, 0))
            config.WALL_DETECT_CM = self.wall_detect_cm
        elif action == "IO_TOGGLE":
            self.io_wall_value = 1 if self.io_wall_value == 0 else 0
            config.IO_WALL_VALUE = self.io_wall_value
        else:
            self.requested_action = action

    def poll_action(self):
        a = self.requested_action
        self.requested_action = None
        return a

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def _crect(self, gx, gy):
        x1 = self.gox + gx * self.cpx
        y1 = self.goy - (gy + 1) * self.cpx
        return x1, y1, x1 + self.cpx, y1 + self.cpx

    def _cctr(self, gx, gy):
        x1, y1, x2, y2 = self._crect(gx, gy)
        return (x1 + x2) // 2, (y1 + y2) // 2

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _title(self, c, px, py, txt):
        cv2.putText(c, txt, (px+14, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_ACCENT, 2, cv2.LINE_AA)

    def _hline(self, c, px, pw, py):
        cv2.line(c, (px+10, py), (px+pw-10, py), (60,50,45), 1)

    def _pbar(self, c, x, y, w, h, frac, col, bg=(50,42,38)):
        frac = max(0.0, min(1.0, frac))
        cv2.rectangle(c, (x, y), (x+w, y+h), bg, -1)
        if frac > 0:
            cv2.rectangle(c, (x, y), (x+int(w*frac), y+h), col, -1)

    def _btn(self, c, x, y, w, h, txt, tag, bg, text_col=(240,240,240)):
        cv2.rectangle(c, (x,y), (x+w, y+h), bg, -1)
        cv2.rectangle(c, (x,y), (x+w, y+h), (110,100,90), 1)
        fs = 0.43
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = x + max(2, (w-tw)//2)
        ty = y + (h+th)//2
        cv2.putText(c, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, text_col, 1, cv2.LINE_AA)
        self.buttons.append((x, y, x+w, y+h, tag))

    # ── SETUP PHASE render ────────────────────────────────────────────────────
    def render_setup(self):
        """Draw the setup screen: pick start & goal, then press RUN."""
        if not self.enabled:
            return
        canvas = np.full((self.H, self.W, 3), C_BG_DARK, dtype=np.uint8)
        self.buttons.clear()

        # Header
        cv2.rectangle(canvas, (0,0), (self.W, 46), (38,30,26), -1)
        cv2.putText(canvas, "ROBOMASTER EP | LAB 6: SETUP -- Choose Start (S) & Goal (G)",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_HEADER, 2, cv2.LINE_AA)
        cv2.putText(canvas, "[ SETUP ]", (self.W-140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,200,60), 2, cv2.LINE_AA)

        # ── Draw empty 4x4 grid ───────────────────────────────────────────────
        for gy in range(4):
            for gx in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                if (gx, gy) == self.start_cell:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), (20,60,20), -1)
                    cv2.rectangle(canvas, (x1+4,y1+4), (x2-4,y2-4), C_START, 3)
                    cv2.putText(canvas, "START (S)", (x1+8, y1+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_START, 2, cv2.LINE_AA)
                elif (gx, gy) == self.goal_cell:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), (20,50,60), -1)
                    cv2.rectangle(canvas, (x1+4,y1+4), (x2-4,y2-4), C_GOAL_C, 3)
                    cv2.putText(canvas, "GOAL (G)", (x1+8, y1+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GOAL_C, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), C_BG_CELL, -1)

                if self.set_target == "START" and (gx,gy) != self.start_cell and (gx,gy) != self.goal_cell:
                    cv2.rectangle(canvas, (x1+2,y1+2), (x2-2,y2-2), (0,80,30), 1)
                elif self.set_target == "GOAL" and (gx,gy) != self.start_cell and (gx,gy) != self.goal_cell:
                    cv2.rectangle(canvas, (x1+2,y1+2), (x2-2,y2-2), (0,80,100), 1)

                cv2.putText(canvas, "(%d,%d)" % (gx,gy), (x1+6, y2-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,100,95), 1, cv2.LINE_AA)

        # Arena border
        bx1=self.gox; by1=self.goy-4*self.cpx
        bx2=bx1+4*self.cpx; by2=self.goy
        cv2.rectangle(canvas, (bx1,by1), (bx2,by2), C_BORDER, 4)

        # Axis labels
        for gx in range(4):
            tx = self.gox + int((gx+0.35)*self.cpx)
            cv2.putText(canvas, "X=%d"%gx, (tx, self.goy+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (155,150,145), 1, cv2.LINE_AA)
        for gy in range(4):
            ty = self.goy - int((gy+0.55)*self.cpx)
            cv2.putText(canvas, "Y=%d"%gy, (self.gox-50, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (155,150,145), 1, cv2.LINE_AA)

        # ── Right panel ───────────────────────────────────────────────────────
        px = 565; pw = self.W - px - 10
        cv2.rectangle(canvas, (px,52), (px+pw, self.H-10), C_BG_PANEL, -1)
        cv2.rectangle(canvas, (px,52), (px+pw, self.H-10), (70,60,55), 1)

        # Mode selection
        self._title(canvas, px, 78, "STEP 1: SELECT MODE")
        r_bg = (20,140,20)  if self.robot_mode == "REAL" else C_BTN_GRAY
        m_bg = (30, 80,160) if self.robot_mode == "MOCK" else C_BTN_GRAY
        r_tc = (255,255,255) if self.robot_mode == "REAL" else (160,160,160)
        m_tc = (255,255,255) if self.robot_mode == "MOCK" else (160,160,160)
        self._btn(canvas, px+14,  88, 192, 42, "REAL ROBOT (Wi-Fi)", "BTN_MODE_REAL", r_bg, text_col=r_tc)
        self._btn(canvas, px+214, 88, 180, 42, "SIMULATION",         "BTN_MODE_MOCK", m_bg, text_col=m_tc)
        if self.robot_mode == "REAL":
            mode_lbl = "Mode: REAL ROBOT selected"
            mode_col = (80, 255, 80)
        elif self.robot_mode == "MOCK":
            mode_lbl = "Mode: SIMULATION selected"
            mode_col = (80, 180, 255)
        else:
            mode_lbl = "<-- Please select a mode"
            mode_col = (80, 80, 200)
        cv2.putText(canvas, mode_lbl, (px+14, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_col, 2, cv2.LINE_AA)
        self._hline(canvas, px, pw, 162)

        # Start & Goal selection
        self._title(canvas, px, 182, "STEP 2: CHOOSE START (S) & GOAL (G)")
        cv2.putText(canvas, "a. Click [SET START] then click a grid cell",
                    (px+14,208), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "b. Click [SET GOAL]  then click a grid cell",
                    (px+14,228), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,220), 1, cv2.LINE_AA)
        self._hline(canvas, px, pw, 242)

        cv2.putText(canvas, "Start (S): (%d, %d)" % self.start_cell,
                    (px+14,266), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_START, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Goal  (G): (%d, %d)" % self.goal_cell,
                    (px+14,294), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GOAL_C, 2, cv2.LINE_AA)
        self._hline(canvas, px, pw, 308)

        s_bg = C_BTN_ACT if self.set_target == "START" else C_BTN_GRAY
        g_bg = C_BTN_ACT if self.set_target == "GOAL"  else C_BTN_GRAY
        self._btn(canvas, px+14,  322, 185, 38, "SET START (S)", "SET_START_MODE", s_bg,
                  text_col=(255,255,255) if self.set_target=="START" else (180,180,180))
        self._btn(canvas, px+208, 322, 185, 38, "SET GOAL  (G)", "SET_GOAL_MODE",  g_bg,
                  text_col=(255,255,255) if self.set_target=="GOAL"  else (180,180,180))

        hint = "Click a grid cell to place START (S)" if self.set_target=="START" else "Click a grid cell to place GOAL (G)"
        hint_col = C_START if self.set_target=="START" else C_GOAL_C
        cv2.putText(canvas, hint, (px+14, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.46, hint_col, 1, cv2.LINE_AA)
        self._hline(canvas, px, pw, 392)

        # Speed pre-config
        self._title(canvas, px, 410, "STEP 3: PRE-CONFIGURE SPEED")
        cv2.putText(canvas, "Forward: %.2f m/s" % self.move_speed,
                    (px+14,438), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,230,100), 2, cv2.LINE_AA)
        self._btn(canvas, px+190,422,36,24," - ","SPEED_DOWN",C_BTN_GRAY)
        self._btn(canvas, px+232,422,36,24," + ","SPEED_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,446,pw-28,8,self.move_speed/0.60,(80,200,80))
        cv2.putText(canvas, "Turn:    %.0f deg/s" % self.turn_speed,
                    (px+14,472), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,230,100), 2, cv2.LINE_AA)
        self._btn(canvas, px+190,456,36,24," - ","TURN_DOWN",C_BTN_GRAY)
        self._btn(canvas, px+232,456,36,24," + ","TURN_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,480,pw-28,8,self.turn_speed/120.0,(80,180,220))
        self._hline(canvas, px, pw, 496)

        # BIG START BUTTON
        cv2.rectangle(canvas, (px+14,508), (px+pw-14,608), (20,120,20), -1)
        cv2.rectangle(canvas, (px+14,508), (px+pw-14,608), (50,220,50), 3)
        cv2.putText(canvas, "START EXPLORATION",
                    (px+50,556), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Explore all, then navigate to GOAL",
                    (px+48,584), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,255,180), 1, cv2.LINE_AA)
        self.buttons.append((px+14, 508, px+pw-14, 608, "BTN_START_RUN"))

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    # ── RUNNING PHASE render ──────────────────────────────────────────────────
    def render(self, ogm, robot_x, robot_y, heading_deg, path_history,
               sensor_data=None, explorer_status="READY", mode_str="REAL ROBOT", step_count=0):
        if not self.enabled:
            return
        canvas = np.full((self.H, self.W, 3), C_BG_DARK, dtype=np.uint8)
        self.buttons.clear()

        # Header
        cv2.rectangle(canvas, (0,0), (self.W, 46), (38,30,26), -1)
        cv2.putText(canvas, "ROBOMASTER EP | LAB 6: A* OGM AUTONOMOUS EXPLORER",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_HEADER, 2, cv2.LINE_AA)
        mc = (60,200,70) if "REAL" in mode_str.upper() else (80,160,240)
        cv2.putText(canvas, "[ %s ]" % mode_str.upper(),
                    (self.W-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mc, 2, cv2.LINE_AA)

        # Grid cells (styled matching occupancy_grid_mapping.py)
        probs = ogm.get_probabilities() if ogm is not None else np.full((4, 4), 0.50)
        for gy in range(4):
            for gx in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                p_val = probs[gy, gx]

                # Colors from occupancy_grid_mapping.py:
                # OCC (#D63031) -> BGR (49, 48, 214)
                # FREE (#00B894) -> BGR (148, 184, 0)
                # UNK (#636E72) -> BGR (114, 110, 99)
                if p_val >= config.THRESHOLD_OCC:
                    bg = (49, 48, 214)
                    t_col = (255, 255, 255)
                    sub_lbl = "OCC"
                elif p_val <= config.THRESHOLD_FREE:
                    bg = (148, 184, 0)
                    t_col = (45, 52, 54)
                    sub_lbl = "FREE"
                else:
                    bg = (114, 110, 99)
                    t_col = (255, 255, 255)
                    sub_lbl = "UNK"

                cv2.rectangle(canvas, (x1+1, y1+1), (x2-1, y2-1), bg, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 52, 54), 1)

                cv2.putText(canvas, f"{p_val:.2f}", (x1 + 28, y1 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, t_col, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"({gx},{gy}) {sub_lbl}", (x1 + 8, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, t_col, 1, cv2.LINE_AA)

                if (gx,gy) == self.start_cell:
                    cv2.rectangle(canvas, (x1+3,y1+3),(x2-3,y2-3), (0, 255, 120), 2)
                    cv2.putText(canvas, "S",(x1+8,y1+20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.50,(0, 255, 120),2,cv2.LINE_AA)
                if (gx,gy) == self.goal_cell:
                    cv2.rectangle(canvas, (x1+3,y1+3),(x2-3,y2-3), (0, 230, 255), 2)
                    pos = (x1+8,y1+20) if (gx,gy)!=self.start_cell else (x1+8,y1+34)
                    cv2.putText(canvas, "G", pos,
                                cv2.FONT_HERSHEY_SIMPLEX,0.50,(0, 230, 255),2,cv2.LINE_AA)

        # Wall edges from ogm.get_edge() (Golden Yellow #FDCB6E)
        if ogm is not None:
            for gy in range(4):
                for gx in range(4):
                    x1, y1, x2, y2 = self._crect(gx, gy)
                    # North: edge between (gx,gy) and (gx,gy+1)
                    if ogm.get_edge((gx, gy), (gx, gy+1)) == -1:
                        cv2.line(canvas, (x1, y1), (x2, y1), C_GOLDEN_WALL, 6, cv2.LINE_AA)
                    # South: edge between (gx,gy) and (gx,gy-1)
                    if ogm.get_edge((gx, gy), (gx, gy-1)) == -1:
                        cv2.line(canvas, (x1, y2), (x2, y2), C_GOLDEN_WALL, 6, cv2.LINE_AA)
                    # East: edge between (gx,gy) and (gx+1,gy)
                    if ogm.get_edge((gx, gy), (gx+1, gy)) == -1:
                        cv2.line(canvas, (x2, y1), (x2, y2), C_GOLDEN_WALL, 6, cv2.LINE_AA)
                    # West: edge between (gx,gy) and (gx-1,gy)
                    if ogm.get_edge((gx, gy), (gx-1, gy)) == -1:
                        cv2.line(canvas, (x1, y1), (x1, y2), C_GOLDEN_WALL, 6, cv2.LINE_AA)

        # Outer perimeter border (Golden Yellow)
        bx1 = self.gox; by1 = self.goy - 4 * self.cpx; bx2 = bx1 + 4 * self.cpx; by2 = self.goy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), C_GOLDEN_WALL, 6)

        for gx in range(4):
            tx=self.gox+int((gx+0.35)*self.cpx)
            cv2.putText(canvas,"X=%d"%gx,(tx,self.goy+22),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,175,170),1,cv2.LINE_AA)
        for gy in range(4):
            ty=self.goy-int((gy+0.55)*self.cpx)
            cv2.putText(canvas,"Y=%d"%gy,(self.gox-50,ty),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,175,170),1,cv2.LINE_AA)

        # Legend
        lx,ly=self.gox,self.goy+42
        items=[(C_GOLDEN_WALL,"Wall (#FDCB6E)"), ((49, 48, 214),"OCC (#D63031)"),
               ((148, 184, 0),"FREE (#00B894)"), ((114, 110, 99),"UNK (#636E72)")]
        cv2.putText(canvas,"LEGEND:",(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,200,200),1,cv2.LINE_AA)
        for i,(col,lbl) in enumerate(items):
            lxi=lx+i*115
            cv2.line(canvas,(lxi+65,ly+2),(lxi+85,ly+2),col,5,cv2.LINE_AA)
            cv2.putText(canvas,lbl,(lxi+88,ly+6),cv2.FONT_HERSHEY_SIMPLEX,0.32,(180,180,180),1,cv2.LINE_AA)

        # Trajectory
        if len(path_history) > 1:
            for i in range(len(path_history)-1):
                p1=self._cctr(path_history[i][0],path_history[i][1])
                p2=self._cctr(path_history[i+1][0],path_history[i+1][1])
                cv2.line(canvas,p1,p2,(0, 200, 255),3,cv2.LINE_AA)
                cv2.circle(canvas,p1,5,(0, 220, 255),-1)

        # Robot Triangle (from occupancy_grid_mapping.py)
        rx, ry = self._cctr(robot_x, robot_y)
        rs = 22
        norm_h = int(round(heading_deg / 90.0) * 90) % 360
        if norm_h == 0:
            pts = np.array([[rx, ry - rs], [rx - rs, ry + rs], [rx + rs, ry + rs]], np.int32)
        elif norm_h == 90:
            pts = np.array([[rx + rs, ry], [rx - rs, ry - rs], [rx - rs, ry + rs]], np.int32)
        elif norm_h == 180:
            pts = np.array([[rx, ry + rs], [rx - rs, ry - rs], [rx + rs, ry - rs]], np.int32)
        else:
            pts = np.array([[rx - rs, ry], [rx + rs, ry - rs], [rx + rs, ry + rs]], np.int32)

        C_ROBOT_PINK = (147, 67, 232)  # #E84393 in BGR
        cv2.fillPoly(canvas, [pts], C_ROBOT_PINK, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

        # Right panel
        px=565; pw=self.W-px-10
        cv2.rectangle(canvas,(px,52),(px+pw,self.H-10),C_BG_PANEL,-1)
        cv2.rectangle(canvas,(px,52),(px+pw,self.H-10),(70,60,55),1)

        self._title(canvas,px,78,"CONFIG & TELEMETRY")
        cv2.putText(canvas,"Step #%d  Pos:(%d,%d)  Hdg:%ddeg"%(step_count,robot_x,robot_y,heading_deg),
                    (px+14,110),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"Goal (G):(%d,%d) <- click map to change"%self.goal_cell,
                    (px+14,132),cv2.FONT_HERSHEY_SIMPLEX,0.44,C_ACCENT,1,cv2.LINE_AA)
        self._hline(canvas,px,pw,148)

        self._title(canvas,px,168,"LIVE SPEED CONFIG")
        cv2.putText(canvas,"Forward Speed:",(px+14,198),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"%.2f m/s"%self.move_speed,(px+148,198),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,230,100),2,cv2.LINE_AA)
        self._btn(canvas,px+236,182,36,24," - ","SPEED_DOWN",C_BTN_GRAY)
        self._btn(canvas,px+278,182,36,24," + ","SPEED_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,206,pw-28,8,self.move_speed/0.60,(80,200,80))

        cv2.putText(canvas,"Turn Speed:",(px+14,234),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"%.0f deg/s"%self.turn_speed,(px+148,234),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,230,100),2,cv2.LINE_AA)
        self._btn(canvas,px+236,218,36,24," - ","TURN_DOWN",C_BTN_GRAY)
        self._btn(canvas,px+278,218,36,24," + ","TURN_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,242,pw-28,8,self.turn_speed/120.0,(80,180,220))
        self._hline(canvas,px,pw,258)

        self._title(canvas,px,278,"SENSOR THRESHOLD")
        cv2.putText(canvas,"Wall Detect:",(px+14,308),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"%.0f cm"%self.wall_detect_cm,(px+130,308),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,200,80),2,cv2.LINE_AA)
        self._btn(canvas,px+190,292,36,24," - ","WDET_DOWN",C_BTN_GRAY)
        self._btn(canvas,px+232,292,36,24," + ","WDET_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,316,pw-28,8,self.wall_detect_cm/100.0,(220,180,40))
        cv2.putText(canvas,"IO_WALL=%d"%self.io_wall_value,
                    (px+14,344),cv2.FONT_HERSHEY_SIMPLEX,0.43,(220,220,220),1,cv2.LINE_AA)
        self._btn(canvas,px+120,328,100,24,"TOGGLE 0/1","IO_TOGGLE",C_BTN_WARN)
        self._hline(canvas,px,pw,368)

        self._title(canvas,px,388,"LIVE SENSOR READINGS")
        f_cm=sensor_data.get("front_dist_cm",999.0) if sensor_data else 999.0
        l_io=sensor_data.get("left_io",0)            if sensor_data else 0
        r_io=sensor_data.get("right_io",0)           if sensor_data else 0
        fw=(f_cm<config.WALL_DETECT_CM)
        fc=C_WALL_CONF if fw else C_OPEN
        cv2.putText(canvas,"Front ToF: %.1fcm [%s]"%(f_cm,"WALL" if fw else "CLEAR"),
                    (px+14,414),cv2.FONT_HERSHEY_SIMPLEX,0.47,fc,2,cv2.LINE_AA)
        self._pbar(canvas,px+14,424,pw-28,10,min(f_cm/150.0,1.0),fc,bg=(55,45,40))
        lw=(l_io==self.io_wall_value); lc=C_WALL_CONF if lw else C_OPEN
        cv2.circle(canvas,(px+24,456),8,lc,-1)
        cv2.putText(canvas,"Left  IR: IO=%d [%s] (steer)"%(l_io,"WALL" if lw else "CLEAR"),
                    (px+40,461),cv2.FONT_HERSHEY_SIMPLEX,0.44,(220,220,220),1,cv2.LINE_AA)
        rw=(r_io==self.io_wall_value); rc=C_WALL_CONF if rw else C_OPEN
        cv2.circle(canvas,(px+24,482),8,rc,-1)
        cv2.putText(canvas,"Right IR: IO=%d [%s] (steer)"%(r_io,"WALL" if rw else "CLEAR"),
                    (px+40,487),cv2.FONT_HERSHEY_SIMPLEX,0.44,(220,220,220),1,cv2.LINE_AA)
        self._hline(canvas,px,pw,502)

        self._title(canvas,px,520,"A* EXPLORER STATUS")
        parts=explorer_status.split("->")
        for i,part in enumerate(parts):
            cv2.putText(canvas,"%s%s"%("Action: " if i==0 else "  -> ",part.strip()),
                        (px+14,542+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,120),1,cv2.LINE_AA)

        # Edge stats
        total_edges = sum(1 for v in ogm.edges.values() if v != 0) if ogm else 0
        wall_count = sum(1 for v in ogm.edges.values() if v == -1) if ogm else 0
        open_count = sum(1 for v in ogm.edges.values() if v == 1) if ogm else 0
        visited_count = len(ogm.visited) if ogm else 0
        cv2.putText(canvas, f"Visited: {visited_count}/16 | Walls: {wall_count} | Open: {open_count}",
                    (px+14, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,220,180), 1, cv2.LINE_AA)
        self._hline(canvas,px,pw,596)

        self._title(canvas,px,612,"ACTION BUTTONS  [W A D E P Q]")
        y1b=620; y2b=y1b+34
        self._btn(canvas,px+14, y1b,162,30,"EXPLORE (E)","BTN_EXPLORE", C_BTN_GREEN)
        self._btn(canvas,px+184,y1b,156,30,"STEP FWD (W)","BTN_STEP_FWD",C_BTN_BLUE)
        self._btn(canvas,px+14, y2b,100,28,"TURN L (A)","BTN_TURN_L",C_BTN_GRAY)
        self._btn(canvas,px+120,y2b,102,28,"TURN R (D)","BTN_TURN_R",C_BTN_GRAY)
        self._btn(canvas,px+228,y2b, 98,28,"SAVE  (P)","BTN_SAVE",  C_BTN_GRAY)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
