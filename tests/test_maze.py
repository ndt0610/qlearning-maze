import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "q_learning_maze", Path(__file__).resolve().parents[1] / "q_learning_maze.py"
)
maze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze)


class MazeTests(unittest.TestCase):
    def test_default_policy_reaches_goal_without_walls(self):
        q_table, rewards, completed = maze.train()
        route = maze.greedy_path(q_table)
        self.assertEqual(len(rewards), 2000)
        self.assertGreater(completed, 0)
        self.assertEqual(route[0], maze.START)
        self.assertEqual(route[-1], maze.GOAL)
        self.assertEqual(len(route), len(set(route)))
        self.assertTrue(all(maze.MAZE[position] == 0 for position in route))

    def test_seed_repeats_rewards_and_policy(self):
        first, rewards1, done1 = maze.train(episodes=80, max_steps=40)
        second, rewards2, done2 = maze.train(episodes=80, max_steps=40)
        self.assertTrue((first == second).all())
        self.assertEqual(rewards1, rewards2)
        self.assertEqual(done1, done2)

    def test_finite_budget(self):
        _, rewards, completed = maze.train(episodes=10, max_steps=1)
        self.assertEqual(len(rewards), 10)
        self.assertEqual(completed, 0)


if __name__ == "__main__":
    unittest.main()
