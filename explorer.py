# -*- coding: utf-8 -*-
"""
explorer.py — Autonomous Unknown Maze Exploration & Exit Finder for Lab 6.
Uses Edge-based MazeWalls + Frontier-based Exploration + DFS/BFS Backtracking.
"""

from collections import deque
import config


class AutonomousExplorer:
    """
    ระบบนำทางสำรวจเขาวงกตที่ไม่รู้แผนที่มาก่อนเพื่อหาทางออก (Exit):
      - ใช้ข้อมูลกำแพงรอบด้าน (North, East, South, West) จาก MazeWalls
      - สำรวจช่องว่างที่ยังไม่เคยเดิน (Unvisited Frontiers) โดยมุ่งหน้าไปยังทิศของทางออก
      - มีระบบ Backtrack อัตโนมัติเมื่อเจอทางตัน (Dead End)
      - เมื่อแผนที่เปิดทางจนพบเส้นทางตรงไปยังทางออก จะเดินเข้าสู่ทางออกทันที
    """

    def __init__(self, exit_cell=config.EXIT_CELL):
        self.exit_cell = exit_cell
        self.visited_cells = set()

    def set_exit_cell(self, new_exit):
        """เปลี่ยนตำแหน่งทางออกเป้าหมาย"""
        self.exit_cell = tuple(new_exit)

    def mark_visited(self, x, y):
        """บันทึกว่าช่อง (x, y) ได้รับการสำรวจแล้ว"""
        self.visited_cells.add((x, y))

    def reset_visited(self):
        """ล้างประวัติการสำรวจ"""
        self.visited_cells.clear()

    def is_at_exit(self, x, y):
        """ตรวจสอบว่าถึงทางออกหรือยัง"""
        return (x, y) == self.exit_cell

    def get_adjacent_neighbors(self, x, y):
        """คืนค่า 4 ช่องรอบข้าง (North, East, South, West)"""
        return [
            (x, y + 1, "NORTH", 0),
            (x + 1, y, "EAST", 90),
            (x, y - 1, "SOUTH", 180),
            (x - 1, y, "WEST", 270)
        ]

    def bfs_shortest_path(self, start, target, maze_walls):
        """
        หาเส้นทางสั้นที่สุดจากจุด start ไปยัง target ผ่านขอบที่ไม่มีกำแพงกั้น
        คืนค่า: list ของ (x, y)
        """
        if start == target:
            return [start]

        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            cx, cy = path[-1]

            if (cx, cy) == target:
                return path

            for nx, ny, dir_name, _ in self.get_adjacent_neighbors(cx, cy):
                if (nx, ny) not in visited:
                    # ตรวจสอบว่าขอบกั้นระหว่าง (cx, cy) และ (nx, ny) ไม่ถูกบล็อกด้วยกำแพง
                    if maze_walls.can_move(cx, cy, dir_name):
                        visited.add((nx, ny))
                        queue.append(path + [(nx, ny)])

        return None

    def find_direct_path_to_exit(self, current, maze_walls):
        """ถ้ามีเส้นทางที่รู้จักแล้วเชื่อมไปยังทางออก ให้คืนเส้นทางนั้น"""
        return self.bfs_shortest_path(current, self.exit_cell, maze_walls)

    def decide_next_move(self, current, maze_walls):
        """
        ตัดสินใจเลือกช่องถัดไปที่ต้องเดิน:
        คืนค่า: (next_cell, decision_reason)
        """
        cx, cy = current
        self.mark_visited(cx, cy)

        # 1. ถึงทางออกแล้ว
        if self.is_at_exit(cx, cy):
            return None, "GOAL_REACHED"

        # 2. เช็คว่ามีทางเคลียร์ไปยังทางออกผ่านช่องและขอบเปิดหรือยัง
        direct_path = self.find_direct_path_to_exit(current, maze_walls)
        if direct_path and len(direct_path) > 1:
            next_step = direct_path[1]
            return next_step, f"EXIT_PATH_FOUND -> เดินตรงสู่เป้าหมาย {self.exit_cell}"

        # 3. หาช่องติดกันที่ยังไม่เคยเดิน และไม่มีกำแพงกั้น (Unvisited Walkable Neighbors)
        unvisited_candidates = []
        for nx, ny, dir_name, _ in self.get_adjacent_neighbors(cx, cy):
            if (nx, ny) not in self.visited_cells and maze_walls.can_move(cx, cy, dir_name):
                # คำนวณระยะห่างไปยังทางออก (Manhattan distance heuristic)
                dist_to_exit = abs(nx - self.exit_cell[0]) + abs(ny - self.exit_cell[1])
                unvisited_candidates.append((dist_to_exit, (nx, ny), dir_name))

        # ถ้ามีช่องรอบข้างที่ยังไม่เคยสำรวจ ให้เลือกช่องที่ใกล้ทางออกที่สุด
        if unvisited_candidates:
            unvisited_candidates.sort(key=lambda item: item[0])
            best_cell = unvisited_candidates[0][1]
            dir_name = unvisited_candidates[0][2]
            return best_cell, f"EXPLORE_NEW -> สำรวจขอบทางใหม่ ({best_cell[0]},{best_cell[1]}) ทิศ {dir_name}"

        # 4. หากเจอทางตัน (Dead End): ทุกด้านรอบตัวมีกำแพงกั้นหรือสำรวจครบแล้ว -> ทำการ Backtrack
        nearest_branch = None
        shortest_branch_path = None

        for cell in self.visited_cells:
            has_unvisited = any(
                (nx, ny) not in self.visited_cells and maze_walls.can_move(cell[0], cell[1], dir_name)
                for nx, ny, dir_name, _ in self.get_adjacent_neighbors(cell[0], cell[1])
            )
            if has_unvisited:
                path = self.bfs_shortest_path(current, cell, maze_walls)
                if path:
                    if shortest_branch_path is None or len(path) < len(shortest_branch_path):
                        shortest_branch_path = path
                        nearest_branch = cell

        if shortest_branch_path and len(shortest_branch_path) > 1:
            next_step = shortest_branch_path[1]
            return next_step, f"DEAD_END_BACKTRACK -> ถอยกลับไปหาทางแยกที่ {nearest_branch}"

        # 5. หากสำรวจจนครบทุกเส้นทางแล้วไม่พบทางออก
        return None, "MAZE_EXHAUSTED"
