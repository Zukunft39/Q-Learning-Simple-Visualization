import random
import unittest
from unittest.mock import patch

from Environment import GridWorld
from MapChecker import MapChecker


class MapCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = MapChecker()

    def test_open_and_blocked_maps(self):
        open_map = [[".", "."], ["#", "P"]]
        self.assertTrue(self.checker.can_reach_goal(open_map, (0, 0), (1, 1)))

        blocked_map = [[".", "#"], ["#", "P"]]
        self.assertFalse(self.checker.can_reach_goal(blocked_map, (0, 0), (1, 1)))

    def test_reward_behind_terminal_is_excluded(self):
        # 通往奖励的唯一道路经过终点：这种奖励不能被领取。
        corridor = [[".", "P", ".", "$"]]
        self.assertTrue(self.checker.can_reach_goal(corridor, (0, 0), (0, 1)))
        reachable = self.checker.get_reachable_positions(
            corridor, (0, 0), blocked_position=(0, 1)
        )
        self.assertEqual(reachable, {(0, 0)})
        self.assertEqual(corridor, [[".", "P", ".", "$"]])

    def test_search_visits_each_open_cell_once(self):
        grid = [[".", "."], [".", "."]]
        reachable = self.checker.get_reachable_positions(grid, (0, 0))
        self.assertEqual(reachable, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_invalid_start_and_empty_map(self):
        self.assertEqual(self.checker.get_reachable_positions([], (0, 0)), set())
        self.assertEqual(self.checker.get_reachable_positions([["#"]], (0, 0)), set())
        self.assertEqual(self.checker.get_reachable_positions([["."]], (-1, 0)), set())
        with self.assertRaises(ValueError):
            self.checker.get_reachable_positions([[".", "."], ["."]], (0, 0))


class MapGenerationTests(unittest.TestCase):
    def reachable_without_goal(self, env):
        # 测试使用独立的栈搜索，交叉验证生成器的 BFS 检查结果。
        pending = [env.start]
        visited = {env.start}
        while len(pending) > 0:
            row, col = pending.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row = row + dr
                next_col = col + dc
                position = (next_row, next_col)
                if next_row < 0 or next_row >= env.rows:
                    continue
                if next_col < 0 or next_col >= env.cols:
                    continue
                if position == env.goal:
                    continue
                if env.map_grid[next_row][next_col] == "#":
                    continue
                if position in visited:
                    continue
                visited.add(position)
                pending.append(position)
        return visited

    def assert_valid_map(self, env):
        reachable = self.reachable_without_goal(env)
        obstacle_count = 0
        reward_count = 0
        for row_index in range(env.rows):
            for col_index in range(env.cols):
                cell = env.map_grid[row_index][col_index]
                if cell == "#":
                    obstacle_count += 1
                elif cell == "$":
                    reward_count += 1
                    self.assertIn((row_index, col_index), reachable)

        goal_row, goal_col = env.goal
        goal_has_reachable_neighbor = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (goal_row + dr, goal_col + dc) in reachable:
                goal_has_reachable_neighbor = True
        self.assertTrue(goal_has_reachable_neighbor)
        self.assertEqual(obstacle_count, 15)
        self.assertEqual(reward_count, 5)
        self.assertEqual(env.map_grid[goal_row][goal_col], "P")

    def test_many_fixed_seeds_are_valid_and_repeatable(self):
        for seed in range(200):
            with self.subTest(seed=seed):
                env = GridWorld(seed=seed)
                self.assert_valid_map(env)
                initial = env.get_state()
                self.assertEqual(env.reset(), initial)

    def test_known_unsolvable_seed_is_retried(self):
        # 旧生成器的种子 122 会生成起终点不连通的地图。
        checker = MapChecker()
        original_check = checker.can_reach_goal
        with patch.object(MapChecker, "can_reach_goal", side_effect=original_check) as check:
            env = GridWorld(seed=122)
        self.assertGreater(check.call_count, 1)
        self.assert_valid_map(env)
        initial = env.get_state()
        self.assertEqual(env.reset(), initial)

    def test_random_resets_are_valid_and_reproducible(self):
        first = GridWorld(mode="random", seed=42)
        second = GridWorld(mode="random", seed=42)
        for _ in range(20):
            initial = first.get_state()
            self.assertEqual(initial, second.get_state())
            self.assert_valid_map(first)
            self.assertNotEqual(first.reset(), initial)
            second.reset()

    def test_failed_generation_stops_at_limit(self):
        env = GridWorld()
        env.max_generation_attempts = 3
        with patch.object(env.map_checker, "can_reach_goal", return_value=False) as check:
            with self.assertRaises(RuntimeError):
                env.reset()
        self.assertEqual(check.call_count, 3)

    def test_insufficient_reward_space_is_retried(self):
        env = GridWorld()
        env.max_generation_attempts = 3
        with patch.object(env.map_checker, "can_reach_goal", return_value=True):
            with patch.object(
                env.map_checker, "get_reachable_positions", return_value={env.start}
            ) as check:
                with self.assertRaises(RuntimeError):
                    env.reset()
        self.assertEqual(check.call_count, 3)

    def test_movement_rewards_and_reset_are_preserved(self):
        env = GridWorld()
        initial = env.get_state()
        self.assertEqual(env.step("w"), (initial, -1, False))
        self.assertEqual(env.credit, -1)
        self.assertEqual(env.hit_wall_num, 1)

        env.map_grid[0][1] = "$"
        self.assertEqual(env.step("d")[1], 5)
        self.assertEqual(env.step("a")[1], -1)
        self.assertEqual(env.step("d")[1], -1)

        env.player_row, env.player_col = (9, 8)
        self.assertEqual(env.step("d")[1:], (10, True))
        self.assertEqual(env.step("a")[1:], (0, True))
        self.assertEqual(env.reset(), initial)
        self.assertEqual(env.credit, 0)
        self.assertEqual(env.hit_wall_num, 0)
        self.assertFalse(env.done)


if __name__ == "__main__":
    unittest.main()
