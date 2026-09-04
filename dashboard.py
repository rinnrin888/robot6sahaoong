# -*- coding: utf-8 -*-
"""
dashboard.py -- Interactive Real-Time Dashboard for Lab 6.

Phase 1 (SETUP):
  - User clicks START cell (green) and EXIT cell (cyan) on the 4x4 grid.
  - Buttons: [SET START] [SET EXIT] [START EXPLORATION]
  - Robot does NOT move yet.

Phase 2 (RUNNING):
  - Full telemetry: wall edges, sensor bars, AI status, speed config.
  - Buttons: EXPLORE, STEP FWD, TURN L/R, SCAN, SAVE.
  - Live speed/threshold config via +/- buttons.
  - Click grid cell -> change exit target.
"""

import math
import numpy as np
import cv2
import config

# ── Colours (BGR) ────────────────────────────────────────────────────────────
C_BG_DARK    = (24,  20,  18)
C_BG_PANEL   = (38,  30,  26)
C_BG_CELL    = (34,  28,  26)
C_BG_VISITED = (46,  40,  35)
C_WALL_CONF  = (40,  40, 220)
C_WALL_LIKE  = (40, 120, 230)
C_OPEN       = (50, 180,  50)
C_UNKNOWN    = (70,  60,  55)
C_BORDER     = (200,  60,  30)
C_ROBOT      = (0,  140, 255)
C_ARROW      = (0,  255, 255)
C_TRAJ       = (0,  200, 255)
C_START      = (0,  200, 100)
C_EXIT_C     = (0,  215, 255)
C_HEADER     = (240, 240, 240)
C_ACCENT     = (0,  215, 255)
C_BTN_GREEN  = (40, 120,  50)
C_BTN_BLUE   = (50,  80, 130)
C_BTN_GRAY   = (55,  48,  42)
C_BTN_WARN   = (30,  80, 160)
C_BTN_START  = (20, 160,  20)
C_BTN_ACT    = (0,  200, 100)


class InteractiveDashboard:
    """Interactive real-time dashboard.  Two phases: SETUP and RUNNING."""

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self, window_name="RoboMaster Lab6 -- Wall Map & Explorer"):
        self.window_name = window_name
        self.W, self.H = 1060, 700
        self.enabled = True

        # Grid pixel layout
        self.gox = 60     # left edge of cell (0,0)
        self.goy = 610    # bottom edge of row 0
        self.cpx = 120    # cell side in pixels

        # ── Phase: "SETUP" or "RUNNING" ──────────────────────────────────────
        self.phase = "SETUP"
        self.set_target = "EXIT"          # which cell is being placed: "START" or "EXIT"
        self.start_cell = (0, 0)          # chosen by user in SETUP
        self.exit_cell  = config.EXIT_CELL
        self.robot_mode = None            # "REAL" or "MOCK" — chosen on setup screen

        # ── Live config (RUNNING phase) ────────────────────────────────────────
        self.move_speed     = config.MOVE_SPEED_MPS
        self.turn_speed     = config.TURN_SPEED_DPS
        self.wall_detect_cm = config.WALL_DETECT_CM
        self.io_wall_value  = config.IO_WALL_VALUE
        self.scan_passes    = 1           # 1-pass 360 scan (1 round)

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
                            self.exit_cell = (gx, gy)
                            config.EXIT_CELL = (gx, gy)
                            print("[Setup] Exit cell -> (%d,%d)" % (gx, gy))
                    else:
                        # RUNNING: change exit target
                        self.exit_cell = (gx, gy)
                        config.EXIT_CELL = (gx, gy)
                        self.requested_action = "SET_EXIT_%d_%d" % (gx, gy)
                        print("[Dashboard] Exit target -> (%d,%d)" % (gx, gy))
                    return

        # Button click
        for (bx1, by1, bx2, by2, act) in self.buttons:
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                self._handle(act)
                return

    def _handle(self, action):
        # Setup phase buttons
        if action == "SET_START_MODE":
            self.set_target = "START"
            print("[Setup] Click a cell to set START position")
            return
        if action == "SET_EXIT_MODE":
            self.set_target = "EXIT"
            print("[Setup] Click a cell to set EXIT position")
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
        elif action == "TOGGLE_PASSES":
            self.scan_passes = 1 if self.scan_passes == 2 else 2
            print("[Dashboard] 360 Scan Passes -> %d" % self.scan_passes)
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

    # ── Wall edge drawing ─────────────────────────────────────────────────────
    def _wall_style(self, p):
        if p >= 0.60:  return C_WALL_CONF, 7, False
        if p >= 0.50:  return C_WALL_LIKE, 5, False
        if p <= 0.40:  return C_OPEN, 2, True
        return C_UNKNOWN, 1, False

    def _draw_edge(self, canvas, pt1, pt2, p, outer=False):
        if outer:
            cv2.line(canvas, pt1, pt2, C_BORDER, 5, cv2.LINE_AA)
            return
        color, thick, dotted = self._wall_style(p)
        x1, y1 = pt1; x2, y2 = pt2
        if dotted:
            dx, dy = x2 - x1, y2 - y1
            L = max(abs(dx), abs(dy), 1)
            step = 10
            for i in range(0, L, step * 2):
                t0 = i / L; t1 = min((i + step) / L, 1.0)
                pa = (int(x1 + t0*dx), int(y1 + t0*dy))
                pb = (int(x1 + t1*dx), int(y1 + t1*dy))
                cv2.line(canvas, pa, pb, color, thick, cv2.LINE_AA)
        else:
            if x1 == x2:
                cv2.line(canvas, (x1, y1+4), (x2, y2-4), color, thick, cv2.LINE_AA)
            else:
                cv2.line(canvas, (x1+4, y1), (x2-4, y2), color, thick, cv2.LINE_AA)

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
        """Draw the setup screen: pick start & exit, then press RUN."""
        if not self.enabled:
            return
        canvas = np.full((self.H, self.W, 3), C_BG_DARK, dtype=np.uint8)
        self.buttons.clear()

        # Header
        cv2.rectangle(canvas, (0,0), (self.W, 46), (38,30,26), -1)
        cv2.putText(canvas, "ROBOMASTER EP | LAB 6: SETUP -- Choose Start & Exit, then press RUN",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_HEADER, 2, cv2.LINE_AA)
        cv2.putText(canvas, "[ SETUP ]", (self.W-140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,200,60), 2, cv2.LINE_AA)

        # ── Draw empty 4x4 grid ───────────────────────────────────────────────
        for gy in range(4):
            for gx in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                # Highlight selected cells
                if (gx, gy) == self.start_cell:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), (20,60,20), -1)
                    cv2.rectangle(canvas, (x1+4,y1+4), (x2-4,y2-4), C_START, 3)
                    cv2.putText(canvas, "START", (x1+8, y1+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_START, 2, cv2.LINE_AA)
                elif (gx, gy) == self.exit_cell:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), (20,50,60), -1)
                    cv2.rectangle(canvas, (x1+4,y1+4), (x2-4,y2-4), C_EXIT_C, 3)
                    cv2.putText(canvas, "EXIT", (x1+8, y1+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_EXIT_C, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), C_BG_CELL, -1)

                # Hover-hint overlay
                if self.set_target == "START" and (gx,gy) != self.start_cell and (gx,gy) != self.exit_cell:
                    cv2.rectangle(canvas, (x1+2,y1+2), (x2-2,y2-2), (0,80,30), 1)
                elif self.set_target == "EXIT" and (gx,gy) != self.start_cell and (gx,gy) != self.exit_cell:
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

        # ── Step 0: Choose REAL or MOCK ────────────────────────────────────────
        self._title(canvas, px, 78, "STEP 1: SELECT MODE")
        r_bg = (20,140,20)  if self.robot_mode == "REAL" else C_BTN_GRAY
        m_bg = (30, 80,160) if self.robot_mode == "MOCK" else C_BTN_GRAY
        r_tc = (255,255,255) if self.robot_mode == "REAL" else (160,160,160)
        m_tc = (255,255,255) if self.robot_mode == "MOCK" else (160,160,160)
        self._btn(canvas, px+14,  88, 192, 42, "REAL ROBOT (Wi-Fi)", "BTN_MODE_REAL", r_bg, text_col=r_tc)
        self._btn(canvas, px+214, 88, 180, 42, "SIMULATION",         "BTN_MODE_MOCK", m_bg, text_col=m_tc)
        # Mode status label
        if self.robot_mode == "REAL":
            mode_lbl = "Mode: REAL ROBOT selected"
            mode_col = (80, 255, 80)
        elif self.robot_mode == "MOCK":
            mode_lbl = "Mode: SIMULATION selected"
            mode_col = (80, 180, 255)
        else:
            mode_lbl = "<-- Please select a mode before starting"
            mode_col = (80, 80, 200)
        cv2.putText(canvas, mode_lbl, (px+14, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_col, 2, cv2.LINE_AA)
        self._hline(canvas, px, pw, 162)

        # ── Step 1: Set start & exit ────────────────────────────────────────────
        self._title(canvas, px, 182, "STEP 2: CHOOSE START & EXIT CELLS")
        cv2.putText(canvas, "a. Click [SET START] then click a grid cell",
                    (px+14,208), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "b. Click [SET EXIT]  then click a grid cell",
                    (px+14,228), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,220), 1, cv2.LINE_AA)
        self._hline(canvas, px, pw, 242)

        # Current selection
        cv2.putText(canvas, "Start Cell : (%d, %d)" % self.start_cell,
                    (px+14,266), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_START, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Exit  Cell : (%d, %d)" % self.exit_cell,
                    (px+14,294), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_EXIT_C, 2, cv2.LINE_AA)
        self._hline(canvas, px, pw, 308)

        # Toggle buttons (SET START / SET EXIT)
        self._title(canvas, px, 328, "STEP 2 CONTINUED: CLICK TO SET CELL TYPE")
        s_bg = C_BTN_ACT if self.set_target == "START" else C_BTN_GRAY
        e_bg = C_BTN_ACT if self.set_target == "EXIT"  else C_BTN_GRAY
        self._btn(canvas, px+14,  342, 185, 38, "SET START POSITION", "SET_START_MODE", s_bg,
                  text_col=(255,255,255) if self.set_target=="START" else (180,180,180))
        self._btn(canvas, px+208, 342, 185, 38, "SET EXIT POSITION",  "SET_EXIT_MODE",  e_bg,
                  text_col=(255,255,255) if self.set_target=="EXIT"  else (180,180,180))

        # Active mode hint
        hint = "Click a grid cell to place START" if self.set_target=="START" else "Click a grid cell to place EXIT"
        hint_col = C_START if self.set_target=="START" else C_EXIT_C
        cv2.putText(canvas, hint, (px+14, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.46, hint_col, 1, cv2.LINE_AA)
        self._hline(canvas, px, pw, 412)

        # Speed pre-config
        self._title(canvas, px, 430, "STEP 3: PRE-CONFIGURE SPEED")
        cv2.putText(canvas, "Forward: %.2f m/s" % self.move_speed,
                    (px+14,458), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,230,100), 2, cv2.LINE_AA)
        self._btn(canvas, px+190,442,36,24," - ","SPEED_DOWN",C_BTN_GRAY)
        self._btn(canvas, px+232,442,36,24," + ","SPEED_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,466,pw-28,8,self.move_speed/0.60,(80,200,80))
        cv2.putText(canvas, "Turn:    %.0f deg/s" % self.turn_speed,
                    (px+14,492), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,230,100), 2, cv2.LINE_AA)
        self._btn(canvas, px+190,476,36,24," - ","TURN_DOWN",C_BTN_GRAY)
        self._btn(canvas, px+232,476,36,24," + ","TURN_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,500,pw-28,8,self.turn_speed/120.0,(80,180,220))
        self._hline(canvas, px, pw, 516)

        # BIG START BUTTON
        cv2.rectangle(canvas, (px+14,518), (px+pw-14,618), (20,120,20), -1)
        cv2.rectangle(canvas, (px+14,518), (px+pw-14,618), (50,220,50), 3)
        cv2.putText(canvas, "START EXPLORATION",
                    (px+50,568), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Robot will begin moving after this",
                    (px+60,596), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,255,180), 1, cv2.LINE_AA)
        self.buttons.append((px+14, 518, px+pw-14, 618, "BTN_START_RUN"))

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    # ── RUNNING PHASE render ──────────────────────────────────────────────────
    def render(self, maze_walls, ogm, robot_x, robot_y, heading_deg, path_history,
               sensor_data=None, explorer_status="READY", mode_str="REAL ROBOT", step_count=0):
        if not self.enabled:
            return
        canvas = np.full((self.H, self.W, 3), C_BG_DARK, dtype=np.uint8)
        self.buttons.clear()

        # Header
        cv2.rectangle(canvas, (0,0), (self.W, 46), (38,30,26), -1)
        cv2.putText(canvas, "ROBOMASTER EP | LAB 6: EDGE-WALL MAP & AUTONOMOUS MAZE EXPLORER",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_HEADER, 2, cv2.LINE_AA)
        mc = (60,200,70) if "REAL" in mode_str.upper() else (80,160,240)
        cv2.putText(canvas, "[ %s ]" % mode_str.upper(),
                    (self.W-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mc, 2, cv2.LINE_AA)

        # Grid cells
        vis = set((p[0],p[1]) for p in path_history)
        for gy in range(4):
            for gx in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                bg = C_BG_VISITED if (gx,gy) in vis else C_BG_CELL
                cv2.rectangle(canvas, (x1+1,y1+1), (x2-1,y2-1), bg, -1)
                cv2.putText(canvas, "(%d,%d)"%(gx,gy), (x1+6,y2-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,100,95), 1, cv2.LINE_AA)
                if (gx,gy) == self.start_cell:
                    cv2.rectangle(canvas, (x1+4,y1+4),(x2-4,y2-4), C_START, 2)
                    cv2.putText(canvas, "START",(x1+8,y1+20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.40,C_START,1,cv2.LINE_AA)
                if (gx,gy) == self.exit_cell:
                    cv2.rectangle(canvas, (x1+4,y1+4),(x2-4,y2-4), C_EXIT_C, 2)
                    cv2.putText(canvas, "EXIT",(x1+8,y1+20) if (gx,gy)!=self.start_cell else (x1+8,y1+36),
                                cv2.FONT_HERSHEY_SIMPLEX,0.40,C_EXIT_C,1,cv2.LINE_AA)

        # Wall edges
        for gy in range(4):
            for gx in range(4):
                x1, y1, x2, y2 = self._crect(gx, gy)
                oN=(gy==3); oS=(gy==0); oE=(gx==3); oW=(gx==0)
                self._draw_edge(canvas,(x1,y1),(x2,y1),maze_walls.get_wall_prob(gx,gy,"NORTH"),outer=oN)
                self._draw_edge(canvas,(x1,y2),(x2,y2),maze_walls.get_wall_prob(gx,gy,"SOUTH"),outer=oS)
                self._draw_edge(canvas,(x2,y1),(x2,y2),maze_walls.get_wall_prob(gx,gy,"EAST"), outer=oE)
                self._draw_edge(canvas,(x1,y1),(x1,y2),maze_walls.get_wall_prob(gx,gy,"WEST"), outer=oW)

        bx1=self.gox; by1=self.goy-4*self.cpx; bx2=bx1+4*self.cpx; by2=self.goy
        cv2.rectangle(canvas,(bx1,by1),(bx2,by2),C_BORDER,4)

        for gx in range(4):
            tx=self.gox+int((gx+0.35)*self.cpx)
            cv2.putText(canvas,"X=%d"%gx,(tx,self.goy+22),cv2.FONT_HERSHEY_SIMPLEX,0.42,(155,150,145),1,cv2.LINE_AA)
        for gy in range(4):
            ty=self.goy-int((gy+0.55)*self.cpx)
            cv2.putText(canvas,"Y=%d"%gy,(self.gox-50,ty),cv2.FONT_HERSHEY_SIMPLEX,0.42,(155,150,145),1,cv2.LINE_AA)

        # Legend
        lx,ly=self.gox,self.goy+42
        items=[(C_BORDER,"Arena"),(C_WALL_CONF,"Wall(P>=0.60)"),(C_WALL_LIKE,"Likely(P>=0.50)"),
               (C_OPEN,"Clear(P<=0.40)"),(C_UNKNOWN,"Unknown")]
        cv2.putText(canvas,"LEGEND:",(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.40,(175,170,165),1,cv2.LINE_AA)
        for i,(col,lbl) in enumerate(items):
            lxi=lx+i*95
            cv2.line(canvas,(lxi+50,ly+2),(lxi+70,ly+2),col,4,cv2.LINE_AA)
            cv2.putText(canvas,lbl,(lxi+72,ly+6),cv2.FONT_HERSHEY_SIMPLEX,0.32,(160,155,150),1,cv2.LINE_AA)

        # Trajectory & robot
        if len(path_history) > 1:
            for i in range(len(path_history)-1):
                p1=self._cctr(path_history[i][0],path_history[i][1])
                p2=self._cctr(path_history[i+1][0],path_history[i+1][1])
                cv2.line(canvas,p1,p2,C_TRAJ,2,cv2.LINE_AA)
                cv2.circle(canvas,p1,4,(0,220,255),-1)
        rx,ry=self._cctr(robot_x,robot_y)
        cv2.circle(canvas,(rx,ry),20,C_ROBOT,-1,cv2.LINE_AA)
        cv2.circle(canvas,(rx,ry),20,(255,255,255),2,cv2.LINE_AA)
        rad=math.radians(heading_deg)
        ax=int(rx+24*math.sin(rad)); ay=int(ry-24*math.cos(rad))
        cv2.arrowedLine(canvas,(rx,ry),(ax,ay),C_ARROW,3,cv2.LINE_AA,tipLength=0.35)

        # Right panel
        px=565; pw=self.W-px-10
        cv2.rectangle(canvas,(px,52),(px+pw,self.H-10),C_BG_PANEL,-1)
        cv2.rectangle(canvas,(px,52),(px+pw,self.H-10),(70,60,55),1)

        self._title(canvas,px,78,"CONFIG & TELEMETRY")
        cv2.putText(canvas,"Step #%d  Pos:(%d,%d)  Hdg:%ddeg"%(step_count,robot_x,robot_y,heading_deg),
                    (px+14,110),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"Exit:(%d,%d) <- click map to change"%self.exit_cell,
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

        self._title(canvas,px,278,"SENSOR THRESHOLD CONFIG")
        cv2.putText(canvas,"Wall Detect Threshold:",(px+14,308),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(canvas,"%.0f cm"%self.wall_detect_cm,(px+216,308),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,200,80),2,cv2.LINE_AA)
        self._btn(canvas,px+262,292,36,24," - ","WDET_DOWN",C_BTN_GRAY)
        self._btn(canvas,px+304,292,36,24," + ","WDET_UP",  C_BTN_GRAY)
        self._pbar(canvas,px+14,316,pw-28,8,self.wall_detect_cm/100.0,(220,180,40))
        cv2.putText(canvas,"IO_WALL_VALUE=%d (wall when IO==%d)"%(self.io_wall_value,self.io_wall_value),
                    (px+14,344),cv2.FONT_HERSHEY_SIMPLEX,0.43,(220,220,220),1,cv2.LINE_AA)
        self._btn(canvas,px+14,354,116,24,"TOGGLE 0/1","IO_TOGGLE",C_BTN_WARN)
        p_lbl = "PASSES: 1 (1-ROUND)" if self.scan_passes == 1 else "PASSES: 2 (DBL-CHK)"
        self._btn(canvas,px+138,354,166,24,p_lbl,"TOGGLE_PASSES",C_BTN_BLUE)
        self._hline(canvas,px,pw,388)

        self._title(canvas,px,408,"LIVE SENSOR READINGS")
        f_cm=sensor_data.get("front_dist_cm",999.0) if sensor_data else 999.0
        l_io=sensor_data.get("left_io",0)            if sensor_data else 0
        r_io=sensor_data.get("right_io",0)           if sensor_data else 0
        fw=(f_cm<config.WALL_DETECT_CM)
        fc=C_WALL_CONF if fw else C_OPEN
        cv2.putText(canvas,"Front ToF: %.1fcm [%s]"%(f_cm,"WALL" if fw else "CLEAR"),
                    (px+14,434),cv2.FONT_HERSHEY_SIMPLEX,0.47,fc,2,cv2.LINE_AA)
        self._pbar(canvas,px+14,444,pw-28,10,min(f_cm/150.0,1.0),fc,bg=(55,45,40))
        lw=(l_io==self.io_wall_value); lc=C_WALL_CONF if lw else C_OPEN
        cv2.circle(canvas,(px+24,476),8,lc,-1)
        cv2.putText(canvas,"Left  IR: IO=%d [%s]"%(l_io,"WALL" if lw else "CLEAR"),
                    (px+40,481),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        rw=(r_io==self.io_wall_value); rc=C_WALL_CONF if rw else C_OPEN
        cv2.circle(canvas,(px+24,502),8,rc,-1)
        cv2.putText(canvas,"Right IR: IO=%d [%s]"%(r_io,"WALL" if rw else "CLEAR"),
                    (px+40,507),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        self._hline(canvas,px,pw,522)

        self._title(canvas,px,540,"AI EXPLORER STATUS")
        parts=explorer_status.split("->")
        for i,part in enumerate(parts):
            cv2.putText(canvas,"%s%s"%("Action: " if i==0 else "  -> ",part.strip()),
                        (px+14,562+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,120),1,cv2.LINE_AA)
        self._hline(canvas,px,pw,606)

        self._title(canvas,px,622,"ACTION BUTTONS  [W A D S E P Q]")
        y1b=630; y2b=y1b+34
        self._btn(canvas,px+14, y1b,162,30,"EXPLORE MAZE (E)","BTN_EXPLORE", C_BTN_GREEN)
        self._btn(canvas,px+184,y1b,156,30,"STEP FWD  (W)",   "BTN_STEP_FWD",C_BTN_BLUE)
        self._btn(canvas,px+14, y2b,100,28,"TURN L (A)","BTN_TURN_L",C_BTN_GRAY)
        self._btn(canvas,px+120,y2b,102,28,"SCAN 360(S)","BTN_SCAN",  C_BTN_GRAY)
        self._btn(canvas,px+228,y2b, 98,28,"TURN R (D)","BTN_TURN_R",C_BTN_GRAY)
        self._btn(canvas,px+332,y2b, 88,28,"SAVE  (P)", "BTN_SAVE",  C_BTN_GRAY)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
