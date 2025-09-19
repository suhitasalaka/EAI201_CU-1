"""
Fixed-grid Pathfinding with Visualization
- Fixed grid defined in the program.
- Menu prompts: choose algorithm (Greedy / A*), choose heuristic (Manhattan / Euclidean / Diagonal).
- Runs the search, prints path, nodes expanded, and displays a matplotlib visualization.
"""

import math
import heapq
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Fixed grid (S=start, G=goal, 0=open, 1=wall)
# ---------------------------
grid = [
    ['S', '0', '0', '0', '0'],
    ['1', '1', '0', '1', 'G'],
    ['0', '0', '0', '1', '0'],
    ['1', '1', '0', '1', '1'],
    ['0', '0', '0', '0', '0']
]

# Find start and goal coordinates, then treat them as open cells
start = None
goal = None
for r in range(len(grid)):
    for c in range(len(grid[0])):
        if grid[r][c] == 'S':
            start = (r, c)
            grid[r][c] = '0'
        elif grid[r][c] == 'G':
            goal = (r, c)
            grid[r][c] = '0'

if start is None or goal is None:
    raise RuntimeError("Grid must contain one 'S' and one 'G'.")

# ---------------------------
# Heuristics
# ---------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def diagonal_chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

HEURISTICS = {
    1: ("Manhattan", manhattan),
    2: ("Euclidean", euclidean),
    3: ("Diagonal", diagonal_chebyshev)
}

# ---------------------------
# Grid helpers
# ---------------------------
ROWS = len(grid)
COLS = len(grid[0])

def in_bounds(pos):
    r, c = pos
    return 0 <= r < ROWS and 0 <= c < COLS

def is_wall(pos):
    r, c = pos
    return grid[r][c] == '1'

def neighbors(pos):
    r, c = pos
    # 4-neighbor (up, down, left, right)
    cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    return [p for p in cand if in_bounds(p)]

def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return list(reversed(path))

def path_steps(path):
    return max(0, len(path)-1) if path else None

# ---------------------------
# Greedy Best-First Search (priority = h(n))
# ---------------------------
def greedy_best_first(start, goal, heuristic):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))
    came_from = {start: None}
    visited = set([start])
    nodes_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, nodes_expanded
        for nb in neighbors(current):
            if is_wall(nb) or nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = current
            heapq.heappush(open_heap, (heuristic(nb, goal), nb))

    return None, nodes_expanded

# ---------------------------
# A* Search (priority = g + h)
# ---------------------------
def a_star(start, goal, heuristic):
    open_heap = []
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    heapq.heappush(open_heap, (f_score[start], start))
    came_from = {start: None}
    nodes_expanded = 0
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, nodes_expanded

        for nb in neighbors(current):
            if is_wall(nb):
                continue
            tentative_g = g_score[current] + 1  # cost of each orthogonal step = 1
            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score[nb], nb))

    return None, nodes_expanded

# ---------------------------
# Visualization (matplotlib)
# ---------------------------
def visualize(grid, start, goal, path=None, title="Pathfinding"):
    h = len(grid)
    w = len(grid[0])
    # image: white for open, black for wall
    img = np.ones((h, w, 3), dtype=float)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '1':
                img[r, c] = [0.0, 0.0, 0.0]  # wall black
            else:
                img[r, c] = [1.0, 1.0, 1.0]  # open white

    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(img, origin='upper')

    # start & goal markers
    sr, sc = start
    gr, gc = goal
    ax.scatter(sc, sr, marker='o', color='green', s=150, label='Start')
    ax.scatter(gc, gr, marker='X', color='red', s=150, label='Goal')

    # overlay path
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color='blue', linewidth=3, label='Path')
        ax.scatter(xs, ys, color='blue', s=30)

    # grid lines
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.gca().invert_yaxis()
    plt.show()

# ---------------------------
# Main menu & run
# ---------------------------
def main():
    print("Choose algorithm:")
    print("1. Greedy Best-First Search")
    print("2. A* Search")
    while True:
        try:
            algo_choice = int(input("Enter 1 or 2: ").strip())
            if algo_choice in (1,2):
                break
        except ValueError:
            pass
        print("Invalid input. Enter 1 or 2.")

    print("\nChoose heuristic:")
    print("1. Manhattan")
    print("2. Euclidean")
    print("3. Diagonal")
    while True:
        try:
            heur_choice = int(input("Enter 1, 2, or 3: ").strip())
            if heur_choice in (1,2,3):
                break
        except ValueError:
            pass
        print("Invalid input. Enter 1, 2, or 3.")

    heur_name, heur_func = HEURISTICS[heur_choice]

    if algo_choice == 1:
        print(f"\nRunning Greedy Best-First Search with {heur_name} heuristic...")
        path, expanded = greedy_best_first(start, goal, heur_func)
        algo_name = "Greedy Best-First"
    else:
        print(f"\nRunning A* Search with {heur_name} heuristic...")
        path, expanded = a_star(start, goal, heur_func)
        algo_name = "A*"

    if path:
        print("\nPath found!")
        print("Path (row,col) sequence:", path)
        print("Path length (steps):", path_steps(path))
        print("Nodes expanded:", expanded)
    else:
        print("\nNo path found.")
        print("Nodes expanded:", expanded)

    # visualize automatically
    title = f"{algo_name} with {heur_name} heuristic"
    visualize(grid, start, goal, path=path, title=title)

if __name__ == "__main__":
    main()
