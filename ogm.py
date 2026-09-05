# -*- coding: utf-8 -*-
"""
ogm.py — Occupancy Grid Mapping (OGM) Engine
Integrated edge-based wall system matching occupancy_grid_mapping.py MapSystem.

Edge system:
  - edges dict: {tuple(sorted([c1,c2])): status}
    -1 = wall, 0 = unknown, 1 = open
  - visited set: cells the robot has physically entered
  - Outer perimeter edges pre-filled as -1 (wall)

Cell system (Bayesian Log-Odds):
  - occ_grid[x][y] log-odds value
  - L_MAX / L_MIN clamp
  - mark_visited sets cell to L_MIN (FREE)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt

import config


# Log-odds constants matching occupancy_grid_mapping.py
L_MAX = 2.0
L_MIN = -2.0
P_OCC_GIVEN_DETECTION = 0.7
P_FREE_GIVEN_DETECTION = 0.3
L_OCC = math.log(P_OCC_GIVEN_DETECTION / (1 - P_OCC_GIVEN_DETECTION))
L_FREE = math.log(P_FREE_GIVEN_DETECTION / (1 - P_FREE_GIVEN_DETECTION))

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIR_NAMES = ['NORTH (^)', 'EAST (>)', 'SOUTH (v)', 'WEST (<)']


class OccupancyGridMap:
    """
    Unified map system matching MapSystem from occupancy_grid_mapping.py:
      - Edge-based walls: edges dict {(c1,c2): -1/0/1}
      - Cell log-odds: occ_grid[x][y]
      - Visited tracking: visited set
    Plus backward-compatible methods from the old ogm.py.
    """

    def __init__(self, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, cell_size_cm=config.CELL_SIZE_CM):
        self.width = width
        self.height = height
        self.cell_size_cm = cell_size_cm

        # ── Edge system (from MapSystem) ──────────────────────────────────
        self.occ_grid = [[0.0 for _ in range(self.height)] for _ in range(self.width)]
        self.edges = {}       # {tuple(sorted([c1,c2])): -1/0/1}
        self.visited = set()  # set of (x, y) visited cells

        # Pre-fill outer perimeter walls
        self._init_perimeter_walls()

        # ── Backward-compatible log-odds array ────────────────────────────
        # This numpy array mirrors occ_grid for methods that use np indexing
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float64)

        # Legacy constants (kept for backward compat with dashboard/save)
        self.l_free = math.log(config.P_FREE / (1.0 - config.P_FREE))
        self.l_occ = math.log(config.P_OCCUPIED / (1.0 - config.P_OCCUPIED))
        self.l_prior = math.log(config.P_PRIOR / (1.0 - config.P_PRIOR))

        # History log
        self.history = []

    def _init_perimeter_walls(self):
        """Pre-fill outer border edges as -1 (wall) — matching occupancy_grid_mapping.py"""
        for i in range(self.width):
            self.set_edge((i, 0), (i, -1), -1)
            self.set_edge((i, self.height - 1), (i, self.height), -1)
        for i in range(self.height):
            self.set_edge((0, i), (-1, i), -1)
            self.set_edge((self.width - 1, i), (self.width, i), -1)

    # ══════════════════════════════════════════════════════════════════════
    #  Edge System (from MapSystem in occupancy_grid_mapping.py)
    # ══════════════════════════════════════════════════════════════════════

    def set_edge(self, c1, c2, status):
        """Set edge status between two adjacent cells.
        status: -1 = wall, 0 = unknown, 1 = open"""
        self.edges[tuple(sorted([c1, c2]))] = status

    def get_edge(self, c1, c2):
        """Get edge status between two adjacent cells.
        Returns: -1 = wall, 0 = unknown, 1 = open"""
        return self.edges.get(tuple(sorted([c1, c2])), 0)

    def update_cell(self, x, y, is_occ):
        """Update cell log-odds (matching MapSystem.update_cell).
        Skips if cell is already visited (confirmed FREE)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if (x, y) in self.visited:
            return

        update_val = L_OCC if is_occ else L_FREE
        self.occ_grid[x][y] = max(L_MIN, min(L_MAX, self.occ_grid[x][y] + update_val))
        # Sync numpy array
        self.log_odds[y, x] = self.occ_grid[x][y]

    def mark_visited(self, x, y):
        """Mark cell as visited and set to FREE (L_MIN).
        Matching MapSystem.mark_visited."""
        self.visited.add((x, y))
        self.occ_grid[x][y] = L_MIN
        # Sync numpy array
        self.log_odds[y, x] = L_MIN

    def get_cell_status(self, x, y):
        """Get cell status as string: 'OCC', 'FREE', 'UNK', or 'OUT'.
        Matching MapSystem.get_cell_status."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 'OUT'
        prob = self.log_odds_to_prob(self.occ_grid[x][y])
        if prob > 0.6:
            return 'OCC'
        if prob < 0.4:
            return 'FREE'
        return 'UNK'

    @staticmethod
    def log_odds_to_prob(l):
        """Convert log-odds to probability (matching MapSystem)."""
        return 1.0 - (1.0 / (1.0 + math.exp(l)))

    # ══════════════════════════════════════════════════════════════════════
    #  Backward-Compatible Methods (from old ogm.py)
    # ══════════════════════════════════════════════════════════════════════

    def log_odds_to_probability(self, l_val):
        """Convert Log-Odds to probability P in [0, 1] (backward compat)."""
        return 1.0 / (1.0 + np.exp(-l_val))

    def probability_to_log_odds(self, p_val):
        """Convert probability to Log-Odds (backward compat)."""
        p_val = np.clip(p_val, 1e-4, 1.0 - 1e-4)
        return np.log(p_val / (1.0 - p_val))

    def get_probabilities(self):
        """Return 4x4 probability matrix (backward compat for dashboard)."""
        return self.log_odds_to_probability(self.log_odds)

    def is_valid_cell(self, x, y):
        """Check if (x, y) is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def update_cell_status(self, x, y, is_occupied: bool, weight: float = 1.0):
        """Legacy Bayesian update (backward compat).
        New code should use update_cell() instead."""
        if not self.is_valid_cell(x, y):
            return
        l_sensor = self.l_occ if is_occupied else self.l_free
        delta = weight * (l_sensor - self.l_prior)
        new_l = self.log_odds[y, x] + delta
        self.log_odds[y, x] = np.clip(new_l, config.L_CLAMP_MIN, config.L_CLAMP_MAX)
        # Sync edge-system occ_grid
        self.occ_grid[x][y] = float(self.log_odds[y, x])

    def update_direction_sensor(self, robot_x, robot_y, heading_deg, sensor_rel_angle, distance_cm):
        """Update OGM from directional sensor reading (backward compat)."""
        if distance_cm is None:
            return

        global_angle = (heading_deg + sensor_rel_angle) % 360
        if 315 <= global_angle or global_angle < 45:
            dx, dy = 0, 1   # North
        elif 45 <= global_angle < 135:
            dx, dy = 1, 0   # East
        elif 135 <= global_angle < 225:
            dx, dy = 0, -1  # South
        else:
            dx, dy = -1, 0  # West

        target_x = robot_x + dx
        target_y = robot_y + dy
        is_occupied = (distance_cm < config.WALL_DETECT_CM)

        if self.is_valid_cell(target_x, target_y):
            self.update_cell_status(target_x, target_y, is_occupied=is_occupied)
        self.update_cell_status(robot_x, robot_y, is_occupied=False, weight=1.5)

    # ══════════════════════════════════════════════════════════════════════
    #  Display & Output
    # ══════════════════════════════════════════════════════════════════════

    def print_ascii_map(self, robot_x=None, robot_y=None, heading_deg=0):
        """ASCII map display on terminal."""
        probs = self.get_probabilities()
        heading_icons = {0: "^", 90: ">", 180: "v", 270: "<"}
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
        """Save probability matrix to CSV."""
        probs = self.get_probabilities()
        np.savetxt(filepath, probs, delimiter=",", fmt="%.4f",
                   header="Occupancy Probabilities 4x4 (Row: Y=0..3, Col: X=0..3)", comments="")
        print(f"[OGM] [OK] CSV saved: {filepath}")

    def save_log_file(self, filepath="ogm_log.txt"):
        """Save experiment log to text file."""
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

            # Edge summary
            f.write("\n--- EDGE STATUS ---\n")
            for edge_key, status in sorted(self.edges.items()):
                status_str = {-1: "WALL", 0: "UNK", 1: "OPEN"}.get(status, "?")
                f.write(f"  {edge_key[0]} <-> {edge_key[1]}: {status_str}\n")

            f.write("\n--- STEP HISTORY ---\n")
            for entry in self.history:
                f.write(f"{entry}\n")
        print(f"[OGM] [OK] Log saved: {filepath}")

    def save_plot_image(self, filepath="ogm_result.png", robot_path=None):
        """Generate heatmap image with wall edges from edge system."""
        probs = self.get_probabilities()

        fig, ax = plt.subplots(figsize=(8, 7))

        cax = ax.imshow(probs, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, origin="lower")
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Occupancy Probability P(m)", fontsize=11)

        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels([f"X={x}" for x in range(self.width)])
        ax.set_yticklabels([f"Y={y}" for y in range(self.height)])
        ax.set_title("RoboMaster EP -- 4x4 Grid OGM & Wall Edges", fontsize=13, pad=12, fontweight="bold")
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

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

        # Draw wall edges from edge system (Golden Yellow #FDCB6E)
        for x in range(self.width):
            for y in range(self.height):
                # North edge
                if self.get_edge((x, y), (x, y + 1)) == -1:
                    ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color="#FDCB6E", linewidth=5, zorder=5)
                # South edge
                if self.get_edge((x, y), (x, y - 1)) == -1:
                    ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="#FDCB6E", linewidth=5, zorder=5)
                # East edge
                if self.get_edge((x, y), (x + 1, y)) == -1:
                    ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color="#FDCB6E", linewidth=5, zorder=5)
                # West edge
                if self.get_edge((x, y), (x - 1, y)) == -1:
                    ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="#FDCB6E", linewidth=5, zorder=5)

        # Outer perimeter
        ax.plot([-0.5, 3.5, 3.5, -0.5, -0.5], [-0.5, -0.5, 3.5, 3.5, -0.5],
                color="#FDCB6E", linewidth=5.5, zorder=4)

        # Robot trajectory
        if robot_path and len(robot_path) > 1:
            px = [p[0] for p in robot_path]
            py = [p[1] for p in robot_path]
            ax.plot(px, py, color="darkorange", linewidth=2.8, linestyle="-", marker="o", markersize=7, label="Robot Trajectory", zorder=6)
            ax.scatter([px[0]], [py[0]], color="lime", s=130, zorder=7, edgecolors="black", label="Start")
            ax.scatter([px[-1]], [py[-1]], color="gold", s=130, zorder=7, edgecolors="black", label=f"Current ({px[-1]},{py[-1]})")
            ax.legend(loc="upper right", framealpha=0.9)

        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(-0.6, 3.6)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close(fig)
        print(f"[OGM] [OK] Plot saved: {filepath}")
