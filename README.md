# Q-learning Maze

A reproducible tabular reinforcement-learning demo on a fixed 10×10 maze. The app trains a Q-table, plots episode rewards, and shows the greedy route learned from the final table.

## Learning loop

~~~mermaid
flowchart LR
  S[Current cell] --> A[Epsilon-greedy valid move]
  A --> R[Reward and next cell]
  R --> U[Q-learning update]
  U --> S
  U --> G[Goal reached or step budget]
~~~

The goal state is terminal, each episode has a finite step budget, and the route viewer rejects loops and invalid wall crossings.

## Reproducible example

With the default seed (42), the fixed maze reached the goal in 1,998 of 2,000 training episodes and the greedy policy found a 24-step route. This describes the included maze and configuration only.

## Run

    python -m pip install -r requirements.txt
    python q_learning_maze.py

Run checks without opening the visualization:

    python -m unittest discover -s tests -v

- q_learning_maze.py — maze, training loop, plotting, and path extraction.
- tests/test_maze.py — checks move validity and route behavior.

This is a compact teaching example, not a benchmark on unseen mazes.
