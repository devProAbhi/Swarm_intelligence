# Importing Libraries
from visualizer import get_obstacles_list, get_botPose_list, get_greenZone_list, set_new_map, getMap
from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
from queue import Queue
from PIL import Image
import time
import matplotlib.pyplot as plt

set_new_map()

# Movement directions (8 directions)
moves = [[-1, 0], [0, 1], [1, 0], [0, -1],[-1, -1], [-1, 1], [1, 1],  [1, -1]]
grid_size = 200  # 200x200 grid

def makeGrid():
    """Creates a 2D grid where obstacles are marked as 0 (non-walkable) and other cells as 1 (walkable)."""
    grid = np.ones((grid_size, grid_size), dtype=int)
    obstacles = get_obstacles_list()
    for obs in obstacles:
        x1, y1 = obs[0]
        x2, y2 = obs[2]
        grid[x1:x2 + 1, y1:y2 + 1] = 0  # Mark obstacles
    return grid

def calculate_centroid(points):
    """Calculate the centroid of a rectangular zone."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (sum(x_coords) // 4, sum(y_coords) // 4)

def bfs(grid, start, goal):
    """BFS to find the shortest path from start to goal, returning path as a list of coordinates."""
    queue = Queue()
    queue.put([start])
    visited = set([start])

    while not queue.empty():
        path = queue.get()
        x, y = path[-1]

        # Check if goal is reached
        if (x, y) == goal:
            return path  # Return path to goal

        # Explore neighbors
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx, ny] == 1 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.put(path + [(nx, ny)])  # Extend path with new node

    return None  # No path found

def planBotPath():
    start_time = time.time()
    grid = makeGrid()
    botsPose = get_botPose_list()
    greenZones = get_greenZone_list()

    nGreenZones = len(greenZones)
    matrixLen = nGreenZones + 1
    scoreMatrix = np.zeros((matrixLen, matrixLen))

    # Initial bot position
    start_pos = botsPose[0]
    start = (start_pos[0], start_pos[1])

    # Calculate BFS cost for each green zone pair (TSP approach)
    for i in range(1, matrixLen):
        centroid = calculate_centroid(greenZones[i - 1])
        path = bfs(grid, start, centroid)
        if path:
            scoreMatrix[i, 0] = scoreMatrix[0, i] = len(path) - 1  # Distance from start to each green zone

    for i in range(1, matrixLen):
        for j in range(1, matrixLen):
            if i == j:
                continue
            start_centroid = calculate_centroid(greenZones[i - 1])
            goal_centroid = calculate_centroid(greenZones[j - 1])
            path = bfs(grid, start_centroid, goal_centroid)
            if path:
                scoreMatrix[i, j] = len(path) - 1

    # Solve TSP with dynamic programming
    botTaskSeq, _ = solve_tsp_dynamic_programming(scoreMatrix)

    # Move bot to each green zone in the sequence
    current_start = start
    combined_path = []
    for idx in botTaskSeq[1:]:
        green_idx = idx - 1
        if green_idx < 0:
            continue
        centroid = calculate_centroid(greenZones[green_idx])
        path = bfs(grid, current_start, centroid)

        if path:
            combined_path.extend(path)
            current_start = centroid
        else:
            print(f"No valid path to green zone {green_idx}, skipping.")

        # Update start to current goal
    total_time_taken = time.time() - start_time
    grid_new = getMap()
    for x, y in combined_path:
        grid_new[x][y] = np.array([0, 0, 255])
    plt.imshow(grid_new)
    plt.title(f"BFS GRID MAP (SINGLE AGENT) - Path Distance: {len(combined_path)}, Time Taken: {total_time_taken:.2f} s")

    plt.show()
planBotPath()

