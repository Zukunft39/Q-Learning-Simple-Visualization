"""桌面实验的训练逻辑。没有界面依赖，也可独立用于测试。"""
import csv
import json
import math
import queue
import random
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from Environment import GridWorld
from QAgent import QAgent


DEFAULT_CONFIG = {
    "mode": "fixed",
    "map_seed": 99999999,
    "agent_seed": 0,
    "episodes": 1000,
    "max_steps": 300,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.2,
    "eval_interval": 50,
    "eval_episodes": 20,
    "window": 20,
}
LIVE_KEYS = ("alpha", "gamma", "epsilon")


def validate_config(values):
    config = dict(values)
    if config["mode"] not in ("fixed", "random"):
        raise ValueError("地图模式必须为 fixed 或 random")
    for key in ("map_seed", "agent_seed", "episodes", "max_steps",
                "eval_interval", "eval_episodes", "window"):
        config[key] = int(config[key])
    for key in ("episodes", "max_steps", "eval_interval", "eval_episodes", "window"):
        if config[key] < 1:
            raise ValueError(key + " 必须是正整数")
    for key in LIVE_KEYS:
        config[key] = float(config[key])
        if not math.isfinite(config[key]):
            raise ValueError(key + " 必须是有限数值")
        if config[key] < 0 or config[key] > 1:
            raise ValueError(key + " 必须在 0 到 1 之间")
    if config["alpha"] == 0:
        raise ValueError("学习率 alpha 必须大于 0")
    return config


def rolling_means(values, window):
    """显式计算滑动平均，避免每个点都重新累加整个窗口。"""
    result = []
    total = 0.0
    for index in range(len(values)):
        total += values[index]
        if index >= window:
            total -= values[index - window]
        count = min(index + 1, window)
        result.append(total / count)
    return result


def stability_status(evaluations):
    """启发式表现稳定判据，不代表 Q 值收敛或策略最优。"""
    if len(evaluations) < 5:
        return "等待至少 5 个评估点"
    recent = evaluations[-5:]
    version = recent[0]["version"]
    for point in recent:
        if point["version"] != version:
            return "参数已变化，重新积累评估点"
        if point["success_rate"] < 95:
            return "仍在学习：近期评估成功率未稳定达到 95%"
    rewards = []
    steps = []
    for point in recent:
        rewards.append(point["reward"])
        steps.append(point["steps"])
    if max(rewards) - min(rewards) > 2:
        return "仍在学习：近期平均奖励波动超过 2"
    if max(steps) - min(steps) > 2:
        return "仍在学习：近期平均步数波动超过 2"
    return "表现稳定：近 5 点成功率≥95%，奖励/步数极差≤2（非收敛证明）"


class TrainingSession:
    """在工作线程运行；只通过 emit 传数据，不直接操作 Tk 控件。"""

    def __init__(self, config, emit, output_root):
        self.config = validate_config(config)
        self.initial_config = dict(self.config)
        self.emit = emit
        self.output_root = Path(output_root)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.commands = queue.Queue()
        self.evaluations = []
        self.changes = []
        self.version = 0
        self.completed = 0
        self.agent = None
        self.run_dir = None

    def gate(self):
        while self.pause_event.is_set():
            if self.stop_event.wait(0.05):
                return False
        return not self.stop_event.is_set()

    def log(self, message):
        line = datetime.now().strftime("%H:%M:%S") + "  " + message
        self.log_file.write(line + "\n")
        self.log_file.flush()
        self.emit("log", line)

    def write_record(self, phase, episode, rollout, result):
        row = dict(result)
        row.update({
            "phase": phase, "episode": episode, "rollout": rollout,
            "alpha": self.agent.alpha, "gamma": self.agent.gamma,
            "epsilon": self.agent.epsilon, "version": self.version,
            "q_entries": len(self.agent.q_table),
        })
        self.writer.writerow(row)

    def apply_commands(self, episode):
        while True:
            try:
                changes = self.commands.get_nowait()
            except queue.Empty:
                break
            candidate = dict(self.config)
            for key in LIVE_KEYS:
                candidate[key] = changes[key]
            candidate = validate_config(candidate)
            changed = False
            for key in LIVE_KEYS:
                if candidate[key] != self.config[key]:
                    changed = True
            if not changed:
                continue
            self.version += 1
            for key in LIVE_KEYS:
                self.config[key] = candidate[key]
                setattr(self.agent, key, candidate[key])
            record = {"episode": episode, "version": self.version}
            for key in LIVE_KEYS:
                record[key] = candidate[key]
            self.changes.append(record)
            self.log("下一局参数已应用：" + str(record))
            self.emit("parameters", record)

    def run_episode(self, env, learn):
        state = env.reset()
        total_reward = 0
        for index in range(self.config["max_steps"]):
            if not self.gate():
                return None
            action = self.agent.choose_action(state)
            next_state, reward, done = env.step(action)
            if learn:
                self.agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        return {
            "reward": total_reward, "steps": index + 1,
            "success": int(done), "collisions": env.hit_wall_num,
        }

    def evaluate(self, env, eval_rng, episode):
        """替换生成器而非消耗训练随机数；finally 保证停止时也能恢复。"""
        train_epsilon = self.agent.epsilon
        train_rng = self.agent.rng
        self.agent.epsilon = 0.0
        self.agent.rng = eval_rng
        results = []
        try:
            for rollout in range(1, self.config["eval_episodes"] + 1):
                result = self.run_episode(env, learn=False)
                if result is None:
                    return None
                results.append(result)
                self.write_record("evaluation", episode, rollout, result)
        finally:
            self.agent.epsilon = train_epsilon
            self.agent.rng = train_rng

        success_sum = 0
        reward_sum = 0
        steps_sum = 0
        collisions_sum = 0
        for result in results:
            success_sum += result["success"]
            reward_sum += result["reward"]
            steps_sum += result["steps"]
            collisions_sum += result["collisions"]
        count = len(results)
        point = {
            "episode": episode, "success_rate": success_sum / count * 100,
            "reward": reward_sum / count, "steps": steps_sum / count,
            "collisions": collisions_sum / count, "version": self.version,
        }
        self.evaluations.append(point)
        point["stability"] = stability_status(self.evaluations)
        self.log(
            f"评估 @{episode}：成功率 {point['success_rate']:.1f}%｜"
            f"平均奖励 {point['reward']:.2f}｜平均步数 {point['steps']:.1f}｜"
            f"平均碰撞 {point['collisions']:.1f}"
        )
        self.emit("evaluation", point)
        return point

    def run(self):
        status = "failed"
        error = None
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
            self.run_dir = self.output_root / name
            self.run_dir.mkdir()
            with (self.run_dir / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as data_file:
                with (self.run_dir / "run.log").open("w", encoding="utf-8") as self.log_file:
                    fields = ["phase", "episode", "rollout", "reward", "steps",
                              "success", "collisions", "alpha", "gamma", "epsilon",
                              "version", "q_entries"]
                    self.writer = csv.DictWriter(data_file, fieldnames=fields)
                    self.writer.writeheader()
                    self.log("开始实验；配置：" + str(self.config))
                    self.log("输出目录：" + str(self.run_dir))
                    if self.config["mode"] == "random":
                        self.log("随机地图评估使用独立地图序列；表格 Q-learning 通常不擅长新地图泛化。")

                    train_env = GridWorld(self.config["mode"], self.config["map_seed"])
                    eval_map_seed = self.config["map_seed"]
                    if self.config["mode"] == "random":
                        eval_map_seed += 1000003
                    eval_env = GridWorld(self.config["mode"], eval_map_seed)
                    self.agent = QAgent(
                        ["w", "a", "s", "d"], self.config["alpha"],
                        self.config["gamma"], self.config["epsilon"],
                        rng=random.Random(self.config["agent_seed"]),
                    )
                    eval_rng = random.Random(self.config["agent_seed"] + 271828)
                    for episode in range(1, self.config["episodes"] + 1):
                        if not self.gate():
                            break
                        self.apply_commands(episode)
                        result = self.run_episode(train_env, learn=True)
                        if result is None:
                            break
                        self.completed = episode
                        self.write_record("training", episode, 1, result)
                        point = dict(result)
                        point["episode"] = episode
                        point["version"] = self.version
                        self.emit("training", point)
                        self.log(
                            f"训练 {episode}/{self.config['episodes']}｜"
                            f"奖励 {result['reward']}｜步数 {result['steps']}｜"
                            f"成功 {bool(result['success'])}｜碰撞 {result['collisions']}"
                        )
                        if episode % self.config["eval_interval"] == 0 or episode == self.config["episodes"]:
                            if self.evaluate(eval_env, eval_rng, episode) is None:
                                break
                        data_file.flush()
                    status = "completed"
                    if self.stop_event.is_set():
                        status = "stopped"
                    self.log("实验结束：" + status + f"，已完成 {self.completed} 局")
        except Exception:
            error = traceback.format_exc()
            self.emit("log", error)
            self.emit("error", str(error))
        finally:
            summary = {
                "status": status, "completed_episodes": self.completed,
                "initial_config": self.initial_config, "final_config": self.config,
                "parameter_changes": self.changes, "evaluations": self.evaluations,
                "error": error,
            }
            if self.run_dir is not None:
                try:
                    (self.run_dir / "summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except OSError as save_error:
                    self.emit("error", "保存汇总失败：" + str(save_error))
            self.emit("finished", {
                "status": status, "completed": self.completed,
                "path": str(self.run_dir or self.output_root),
            })
