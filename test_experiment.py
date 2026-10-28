import csv
import json
import queue
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from Environment import GridWorld
from Experiment import DEFAULT_CONFIG, TrainingSession, rolling_means, stability_status, validate_config
from QAgent import QAgent


class ExperimentTests(unittest.TestCase):
    def config(self):
        config = dict(DEFAULT_CONFIG)
        config.update({"episodes": 12, "max_steps": 30,
                       "eval_interval": 4, "eval_episodes": 2})
        return config

    def test_evaluation_is_frozen_and_restores_randomness(self):
        config = self.config()
        session = TrainingSession(config, lambda *args: None, ".")
        session.agent = QAgent(["w", "a", "s", "d"], rng=random.Random(123))
        state = GridWorld().reset()
        session.agent.q_table[(state, "w")] = 2.5
        original_rng = session.agent.rng
        original_rng_state = original_rng.getstate()
        before = session.agent.q_table.copy()
        session.log = lambda message: None
        session.write_record = lambda *args: None
        with patch.object(session.agent, "learn", side_effect=AssertionError("evaluation learned")):
            point = session.evaluate(GridWorld(), random.Random(456), 4)
        self.assertIsNotNone(point)
        self.assertEqual(before, session.agent.q_table)
        self.assertIs(session.agent.rng, original_rng)
        self.assertEqual(original_rng_state, original_rng.getstate())
        self.assertEqual(session.agent.epsilon, 0.2)

    def test_evaluation_frequency_does_not_change_training(self):
        results = []
        with tempfile.TemporaryDirectory() as temporary:
            for interval in (3, 6):
                config = self.config()
                config["eval_interval"] = interval
                training = []
                def emit(kind, value):
                    if kind == "training":
                        training.append(value)
                session = TrainingSession(config, emit, temporary)
                session.run()
                results.append((training, session.agent.q_table))
        self.assertEqual(results[0], results[1])

    def test_live_parameters_and_saved_files(self):
        config = self.config()
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            session = TrainingSession(config, lambda *event: events.append(event), temporary)
            session.commands.put({"alpha": 0.3, "gamma": 0.9, "epsilon": 0.1})
            session.run()
            summary = json.loads((session.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_episodes"], 12)
            self.assertEqual(summary["parameter_changes"][0]["episode"], 1)
            self.assertEqual(session.agent.alpha, 0.3)
            with (session.run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as handle:
                records = list(csv.DictReader(handle))
            training_count = 0
            evaluation_count = 0
            for row in records:
                if row["phase"] == "training":
                    training_count += 1
                else:
                    evaluation_count += 1
                    self.assertEqual(float(row["epsilon"]), 0)
            self.assertEqual(training_count, 12)
            self.assertEqual(evaluation_count, 6)
            self.assertIn("开始实验", (session.run_dir / "run.log").read_text(encoding="utf-8"))
            self.assertEqual(events[-1][0], "finished")

    def test_stop_while_paused(self):
        session = TrainingSession(self.config(), lambda *args: None, ".")
        session.pause_event.set()
        timer = threading.Timer(0.03, session.stop_event.set)
        timer.start()
        self.assertFalse(session.gate())
        timer.join()

    def test_random_map_run(self):
        config = self.config()
        config["mode"] = "random"
        with tempfile.TemporaryDirectory() as temporary:
            session = TrainingSession(config, lambda *args: None, temporary)
            session.run()
            self.assertEqual(session.completed, 12)
            self.assertEqual(len(session.evaluations), 3)

    def test_validation_and_rolling_mean(self):
        self.assertEqual(rolling_means([1, 3, 5], 2), [1, 2, 4])
        for changes in ({"alpha": 0}, {"gamma": float("nan")},
                        {"max_steps": 0}, {"epsilon": 1.1}):
            config = self.config()
            config.update(changes)
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_stability_requires_success_and_same_parameters(self):
        points = []
        for episode in range(5):
            points.append({"success_rate": 100, "reward": 11,
                           "steps": 30, "version": 0})
        self.assertIn("表现稳定", stability_status(points))
        points[-1]["version"] = 1
        self.assertNotIn("表现稳定：", stability_status(points))
        points[-1]["version"] = 0
        points[-1]["success_rate"] = 0
        self.assertNotIn("表现稳定：", stability_status(points))


if __name__ == "__main__":
    unittest.main()
