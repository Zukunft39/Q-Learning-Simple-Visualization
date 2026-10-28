from collections import deque


class MapChecker:
    """用广度优先搜索检查地图，不修改地图，也不控制玩家动作。"""

    def get_reachable_positions(self, map_grid, start, blocked_position=None):
        """返回起点可达的坐标集合；可额外指定一个禁止经过的位置。"""
        reachable_positions = set()

        if len(map_grid) == 0:
            return reachable_positions

        rows = len(map_grid)
        cols = len(map_grid[0])
        if cols == 0:
            return reachable_positions

        for row in map_grid:
            if len(row) != cols:
                raise ValueError("地图的每一行必须具有相同的列数")

        start_row = start[0]
        start_col = start[1]
        if start_row < 0 or start_row >= rows:
            return reachable_positions
        if start_col < 0 or start_col >= cols:
            return reachable_positions
        if map_grid[start_row][start_col] == "#":
            return reachable_positions
        if start == blocked_position:
            return reachable_positions

        # deque 是队列：从右侧加入，从左侧取出，实现先进先出。
        pending_positions = deque()
        pending_positions.append(start)
        # 加入队列时就标记，避免相邻格子反复把对方加入队列。
        reachable_positions.add(start)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while len(pending_positions) > 0:
            position = pending_positions.popleft()
            row_index = position[0]
            col_index = position[1]

            for direction in directions:
                next_row = row_index + direction[0]
                next_col = col_index + direction[1]
                next_position = (next_row, next_col)

                if next_row < 0 or next_row >= rows:
                    continue
                if next_col < 0 or next_col >= cols:
                    continue
                if map_grid[next_row][next_col] == "#":
                    continue
                if next_position == blocked_position:
                    continue
                if next_position in reachable_positions:
                    continue

                reachable_positions.add(next_position)
                pending_positions.append(next_position)

        return reachable_positions

    def can_reach_goal(self, map_grid, start, goal):
        """判断从起点是否存在一条到终点的通路。"""
        reachable_positions = self.get_reachable_positions(map_grid, start)
        if goal in reachable_positions:
            return True
        return False
