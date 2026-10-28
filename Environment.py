import random

from MapChecker import MapChecker


class GridWorld:
    """保存地图和游戏规则；键盘输入由类外的 play_game 负责。"""

    # 类属性：字典保存每个动作对应的“行变化、列变化”。
    ACTIONS = {
        "w": (-1, 0),
        "s": (1, 0),
        "a": (0, -1),
        "d": (0, 1),
    }

    def __init__(self, mode="fixed", seed=42):
        if mode not in ("fixed", "random"):
            raise ValueError("mode 必须是 fixed 或 random")

        # self 类似 C# 的 this；这些属性由每个环境对象各自保存。
        self.mode = mode
        self.seed = seed
        self.rows = 10
        self.cols = 10
        self.start = (0, 0)
        self.goal = (9, 9)
        self.credit = 0
        self.reward = 5
        self.resume = -1
        self.hit_wall_num = 0
        self.goal_reward = 10
        self.rng = random.Random(seed)
        self.map_checker = MapChecker()
        self.max_generation_attempts = 1000

        self.map_grid = []
        self.player_row = 0
        self.player_col = 0
        self.done = False
        self.reset()

    def reset(self):
        """生成可解地图，重置玩家和积分，返回初始状态。"""
        if self.mode == "fixed":
            # 固定模式只在本次 reset 开头重置种子，重试时继续抽取。
            rng = random.Random(self.seed)
        else:
            rng = self.rng

        protected = {
            self.start,
            self.goal,
            (1, 0),
            (0, 1),
            (8, 9),
            (9, 8),
        }

        available_positions = []
        for row_index in range(self.rows):
            for col_index in range(self.cols):
                position = (row_index, col_index)
                if position not in protected:
                    available_positions.append(position)

        map_generated = False
        for attempt_index in range(self.max_generation_attempts):
            candidate_map = []
            for row_index in range(self.rows):
                new_row = []
                for col_index in range(self.cols):
                    new_row.append(".")
                candidate_map.append(new_row)

            obstacle_positions = rng.sample(available_positions, 15)
            for position in obstacle_positions:
                row_index = position[0]
                col_index = position[1]
                candidate_map[row_index][col_index] = "#"

            # 没有通路就丢弃候选地图，重新抽取障碍。
            can_reach_goal = self.map_checker.can_reach_goal(
                candidate_map, self.start, self.goal
            )
            if not can_reach_goal:
                continue

            # 到终点就会结束，所以奖励必须能在不经过终点时领取。
            reachable_before_goal = self.map_checker.get_reachable_positions(
                candidate_map, self.start, blocked_position=self.goal
            )

            reward_positions = []
            # 保持按行、列的顺序收集候选位置，保证固定种子可复现。
            for row_index in range(self.rows):
                for col_index in range(self.cols):
                    position = (row_index, col_index)
                    if position == self.start:
                        continue
                    if position == self.goal:
                        continue
                    if position not in reachable_before_goal:
                        continue
                    reward_positions.append(position)

            if len(reward_positions) < 5:
                continue

            selected_rewards = rng.sample(reward_positions, 5)
            for position in selected_rewards:
                row_index = position[0]
                col_index = position[1]
                candidate_map[row_index][col_index] = "$"

            goal_row = self.goal[0]
            goal_col = self.goal[1]
            candidate_map[goal_row][goal_col] = "P"
            self.map_grid = candidate_map
            map_generated = True
            break

        if not map_generated:
            raise RuntimeError(
                f"尝试 {self.max_generation_attempts} 次后仍无法生成可解地图"
            )

        self.player_row = self.start[0]
        self.player_col = self.start[1]
        self.credit = 0
        self.hit_wall_num = 0
        self.done = False

        initial_state = self.get_state()
        return initial_state

    def get_state(self):
        """返回玩家位置和地图快照，包含尚未领取的奖励。"""
        # 先把每一行转换成不可变元组，再组合成整张不可变地图。
        immutable_rows = []
        for row in self.map_grid:
            immutable_row = tuple(row)
            immutable_rows.append(immutable_row)

        immutable_map = tuple(immutable_rows)
        state = (self.player_row, self.player_col, immutable_map)
        return state

    def step(self, action):
        """返回 (下一状态, 本次奖励, 是否结束)，不读取键盘或打印。"""
        if action not in self.ACTIONS:
            raise ValueError("action 必须是 w、a、s、d 之一")

        if self.done:
            current_state = self.get_state()
            return current_state, 0, True

        movement = self.ACTIONS[action]
        row_delta = movement[0]
        col_delta = movement[1]
        next_row = self.player_row + row_delta
        next_col = self.player_col + col_delta

        if next_row < 0 or next_row >= self.rows:
            self.hit_wall_num += 1
            reward = -1
            self.credit += reward
            return self.get_state(), reward, False
        if next_col < 0 or next_col >= self.cols:
            self.hit_wall_num += 1
            reward = -1
            self.credit += reward
            return self.get_state(), reward, False
        if self.map_grid[next_row][next_col] == "#":
            self.hit_wall_num += 1
            reward = -1
            self.credit += reward
            return self.get_state(), reward, False

        if self.map_grid[next_row][next_col] == "$":
            reward = self.reward
            self.map_grid[next_row][next_col] = "."
        else:
            reward = self.resume

        self.player_row = next_row
        self.player_col = next_col

        new_position = (self.player_row, self.player_col)
        if new_position == self.goal:
            self.done = True
            # 普通移动 -1，再额外 +11，终点净奖励为 +10。
            goal_bonus = self.goal_reward + 1
            reward += goal_bonus

        self.credit += reward
        next_state = self.get_state()
        return next_state, reward, self.done

    def render(self):
        """玩家只画在显示列表里，不修改地图数据。"""
        for row_index in range(self.rows):
            display_row = []
            for col_index in range(self.cols):
                is_player_row = row_index == self.player_row
                is_player_col = col_index == self.player_col

                if is_player_row and is_player_col:
                    display_row.append("0")
                else:
                    cell = self.map_grid[row_index][col_index]
                    display_row.append(cell)

            line = " ".join(display_row)
            print(line)

        print(f"当前积分：{self.credit}")


def play_game(env):
    """人类玩家使用的 Windows 终端键盘循环。"""
    import msvcrt
    import subprocess

    env.reset()
    while True:
        subprocess.run("cls", shell=True)
        env.render()

        if env.done:
            print("你到达了终点！")
            print(f"最终积分：{env.credit}")
            break

        print("WASD 移动，# 是障碍，$ 是奖励，P 是终点，Q 退出")
        key = msvcrt.getwch()
        direction = key.lower()

        if direction == "q":
            break
        if direction in env.ACTIONS:
            env.step(direction)

    return env.credit


if __name__ == "__main__":
    env = GridWorld(mode="fixed", seed=42)
    play_game(env)
