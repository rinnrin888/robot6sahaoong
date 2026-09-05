# -*- coding: utf-8 -*-
"""
explorer.py — A* Planner for Autonomous Maze Exploration & Goal Navigation.
Logic matches a_star_planner() from occupancy_grid_mapping.py exactly.

Modes:
  - "explore": find shortest path to nearest unvisited cell
  - "goal":    find shortest path to a specific goal cell

Edge costs:
  - open (1):    1.0
  - unknown (0): 5.0 (allow risky exploration)
  - wall (-1):   blocked (not traversable)

Turn penalties:
  - straight:    0.0
  - 90 turn:     0.8
  - 180 u-turn:  1.5
"""

import heapq
import config

GRID_SIZE = config.GRID_WIDTH   # 4
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
DIR_NAMES = ['NORTH', 'EAST', 'SOUTH', 'WEST']


def a_star_planner(ogm, start_x, start_y, start_heading, target_mode="explore", target_pos=None):
    """
    A* pathfinding matching occupancy_grid_mapping.py logic.

    Args:
        ogm:            OccupancyGridMap instance (with .edges, .visited)
        start_x, start_y: current robot position
        start_heading:  heading index (0=N, 1=E, 2=S, 3=W)
        target_mode:    "explore" (find unvisited cell) or "goal" (go to target_pos)
        target_pos:     (x, y) tuple for "goal" mode

    Returns:
        list of (x, y) from start to target, or None if no path found.
    """
    pq = [(0, start_x, start_y, start_heading, [(start_x, start_y)])]
    visited_states = set()

    while pq:
        cost, cx, cy, c_head, path = heapq.heappop(pq)

        if target_mode == "explore":
            # Target: any unvisited cell (not the start itself)
            if (cx, cy) not in ogm.visited and (cx, cy) != (start_x, start_y):
                return path
        else:
            # Target: specific goal position
            if (cx, cy) == target_pos:
                return path

        state = (cx, cy, c_head)
        if state in visited_states:
            continue
        visited_states.add(state)

        for d_idx in range(4):
            nx, ny = cx + DIRS[d_idx][0], cy + DIRS[d_idx][1]
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                e_stat = ogm.get_edge((cx, cy), (nx, ny))
                if e_stat != -1:  # Not a confirmed wall
                    # Known open edge = 1.0, unknown = 5.0 (risky but allowed)
                    move_cost = 1.0 if e_stat == 1 else 5.0
                    turn_diff = (d_idx - c_head) % 4
                    turn_penalty = 0.0 if turn_diff == 0 else (1.5 if turn_diff == 2 else 0.8)

                    new_cost = cost + move_cost + turn_penalty
                    h = abs(target_pos[0] - nx) + abs(target_pos[1] - ny) if target_pos else 0

                    heapq.heappush(pq, (new_cost + h, nx, ny, d_idx, path + [(nx, ny)]))

    return None


def heading_deg_to_idx(heading_deg):
    """Convert heading degrees (0/90/180/270) to direction index (0/1/2/3).
    0° (North) -> 0, 90° (East) -> 1, 180° (South) -> 2, 270° (West) -> 3"""
    norm = int(round(heading_deg / 90.0) * 90) % 360
    return {0: 0, 90: 1, 180: 2, 270: 3}.get(norm, 0)


def heading_idx_to_deg(heading_idx):
    """Convert direction index (0/1/2/3) to heading degrees.
    0 -> 0° (North), 1 -> 90° (East), 2 -> 180° (South), 3 -> 270° (West)"""
    return [0, 90, 180, 270][heading_idx % 4]
