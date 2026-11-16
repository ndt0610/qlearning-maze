"""A small, reproducible Q-learning maze demo."""

from __future__ import annotations

import numpy as np

MAZE = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)
START = (0, 0)
GOAL = (9, 9)
ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def valid_actions(state: tuple[int, int]) -> list[int]:
    """Return moves that stay inside the maze and avoid walls."""
    result = []
    for index, (dr, dc) in enumerate(ACTIONS):
        row, col = state[0] + dr, state[1] + dc
        if 0 <= row < MAZE.shape[0] and 0 <= col < MAZE.shape[1] and MAZE[row, col] == 0:
            result.append(index)
    return result


def step(state: tuple[int, int], action: int) -> tuple[int, int]:
    dr, dc = ACTIONS[action]
    return state[0] + dr, state[1] + dc


def train(
    alpha: float = 0.1,
    episodes: int = 2000,
    seed: int = 42,
    max_steps: int = 400,
) -> tuple[np.ndarray, list[int], int]:
    """Train one Q table. Each episode has a finite step budget."""
    if not 0 < alpha <= 1 or episodes < 1 or max_steps < 1:
        raise ValueError("alpha must be in (0, 1], and episode/step counts must be positive")
    rng = np.random.default_rng(seed)
    q_table = np.zeros((*MAZE.shape, len(ACTIONS)), dtype=float)
    rewards: list[int] = []
    completed = 0

    for episode in range(episodes):
        state = START
        total = 0
        epsilon = max(0.05, 0.5 * (0.997**episode))
        for _ in range(max_steps):
            allowed = valid_actions(state)
            if rng.random() < epsilon:
                action = int(rng.choice(allowed))
            else:
                values = q_table[state][allowed]
                best = [a for a, value in zip(allowed, values) if value == values.max()]
                action = int(rng.choice(best))

            next_state = step(state, action)
            terminal = next_state == GOAL
            reward = 50 if terminal else -1
            future = 0.0 if terminal else float(q_table[next_state][valid_actions(next_state)].max())
            q_table[state][action] += alpha * (reward + 0.9 * future - q_table[state][action])
            total += reward
            state = next_state
            if terminal:
                completed += 1
                break
        rewards.append(total)
    return q_table, rewards, completed


def greedy_path(q_table: np.ndarray, max_steps: int = 100) -> list[tuple[int, int]]:
    """Return a valid goal path, or an empty list if the policy loops."""
    state = START
    path = [state]
    visited = {state}
    for _ in range(max_steps):
        if state == GOAL:
            return path
        allowed = valid_actions(state)
        action = max(allowed, key=lambda a: q_table[state][a])
        state = step(state, action)
        if state in visited:
            return []
        path.append(state)
        visited.add(state)
    return path if state == GOAL else []


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.widgets import Button, Slider

    fig, (maze_ax, reward_ax) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.23, wspace=0.3)
    maze_ax.imshow(MAZE, cmap=ListedColormap(["#f3f7f4", "#465e56"]))
    maze_ax.scatter(START[1], START[0], color="#247a53", s=140, label="Start")
    maze_ax.scatter(GOAL[1], GOAL[0], color="#df8b2d", s=180, marker="*", label="Goal")
    path_line, = maze_ax.plot([], [], color="#2563a6", linewidth=3, label="Learned path")
    maze_ax.set_title("Q-learning maze")
    maze_ax.set_xticks(range(MAZE.shape[1]))
    maze_ax.set_yticks(range(MAZE.shape[0]))
    maze_ax.grid(color="#9aa9a2", linewidth=0.4)
    maze_ax.legend(loc="upper right")

    reward_line, = reward_ax.plot([], [], color="#2563a6", linewidth=0.8)
    reward_ax.set_xlabel("Episode")
    reward_ax.set_ylabel("Total reward")
    reward_ax.grid(alpha=0.25)
    slider = Slider(fig.add_axes((0.18, 0.08, 0.47, 0.035)), "Learning rate", 0.01, 1.0, valinit=0.1)
    button = Button(fig.add_axes((0.73, 0.07, 0.17, 0.055)), "Train again")

    def refresh(_event=None) -> None:
        q_table, rewards, completed = train(alpha=float(slider.val))
        path = greedy_path(q_table)
        if path:
            rows, cols = zip(*path)
            path_line.set_data(cols, rows)
        else:
            path_line.set_data([], [])
        reward_line.set_data(range(len(rewards)), rewards)
        reward_ax.relim()
        reward_ax.autoscale_view()
        reward_ax.set_title(f"Goal reached: {completed}/2000; path: {len(path) - 1 if path else 'none'}")
        fig.canvas.draw_idle()

    button.on_clicked(refresh)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
