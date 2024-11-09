import numpy as np
import time
from sklearn.cluster import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming
from PIL import Image
import random
import matplotlib.pyplot as plt
from visualizer import get_obstacles_list, get_botPose_list, get_greenZone_list, set_new_map, getMap
set_new_map()

# Define movement directions (4 cardinal directions)
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
grid_size = 200  # Assuming a 200x200 grid
robot_colors = [
    [255, 0, 0],    # Red
    [0, 0, 255],    # Blue
    [255, 0, 255],  # Magenta
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
   
]

# Define the grid


def makeGrid():
    """Creates a 2D grid where obstacles are marked as 0 (non-walkable) and other cells as 1 (walkable)."""
    grid = np.ones((grid_size, grid_size), dtype=int)
    obstacles = get_obstacles_list()
    for obs in obstacles:
        x1, y1 = obs[0]
        x2, y2 = obs[2]
        grid[x1:x2 + 1, y1:y2 + 1] = 0  # Mark obstacles
    return grid

# Calculate Manhattan distance (heuristic for TSP)


def calculate_heuristic(start, goal):
    """Calculate the Manhattan distance (heuristic) between two points."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

# RRT algorithm for pathfinding
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class RRT:
    def __init__(self, start, goal, grid, max_iter=10000, step_size=1.0, goal_sample_rate=0.05):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.tree = [start]
        self.parents = {tuple(start): None}

    def get_random_node(self):
        """Generate a random node, with a higher probability to sample near the goal."""
        if random.random() < self.goal_sample_rate:
            return self.goal  # Sample the goal with a higher probability
        return [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]

    def get_nearest_node(self, node):
        """Find the node in the tree closest to the given node."""
        return min(self.tree, key=lambda x: np.linalg.norm(np.array(x) - np.array(node)))

    def valid_node(self, node):
        """Check if a node is valid (i.e., within bounds and not in an obstacle)."""
        x, y = node
        if 0 <= x < grid_size and 0 <= y < grid_size and self.grid[x, y] == 1:
            return True
        return False

    def step_towards(self, current_node, random_node):
        """Take a step towards the random node."""
        direction = np.array(random_node) - np.array(current_node)
        norm = np.linalg.norm(direction)
        if norm > self.step_size:
            direction = direction / norm * self.step_size
        next_node = (current_node[0] + direction[0],
                     current_node[1] + direction[1])
        return (int(round(next_node[0])), int(round(next_node[1])))

    def build(self):
        """Build the tree using RRT algorithm."""
        for _ in range(self.max_iter):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)

            new_node = self.step_towards(nearest_node, random_node)
            if self.valid_node(new_node) and tuple(new_node) not in self.parents:
                self.tree.append(new_node)
                self.parents[tuple(new_node)] = nearest_node

                if np.array_equal(new_node, self.goal):
                    return self.reconstruct_path(new_node)
        return None  # If no path found

    def reconstruct_path(self, node):
        """Reconstruct the path from the goal to the start."""
        path = [node]
        while tuple(node) in self.parents:
            node = self.parents[tuple(node)]
            if node is None:
                break
            path.append(node)
        path.reverse()
        return path

# Plan the bot's path to visit green zones exactly once


def planBotPath(botsPose, greenZones):
    start_time = time.time()
    grid = makeGrid()

    # Start position (the bot's initial position)
    start_pos = botsPose
    start = (start_pos[0], start_pos[1])

    # Initialize the path and visited green zones list
    path = []
    visited_green_zones = set()
    current_pos = start

    # Get the center of each green zone (or boundary point if you prefer)
    green_centers = []
    for green_zone in greenZones:
        center_x = (green_zone[0][0] + green_zone[2][0]) // 2
        center_y = (green_zone[0][1] + green_zone[2][1]) // 2
        green_centers.append((center_x, center_y))

    # Calculate pairwise distances between green zones
    distance_matrix = []
    for i in range(len(green_centers)):
        row = []
        for j in range(len(green_centers)):
            distance = calculate_heuristic(green_centers[i], green_centers[j])
            row.append(distance)
        distance_matrix.append(row)

    # Use TSP to find the optimal order to visit green zones
    perm_order, _ = solve_tsp_dynamic_programming(np.array(distance_matrix))

    # Loop through and plan paths to the green zones in the optimal order

    while len(visited_green_zones) < len(greenZones):
        for i in perm_order:
            if i not in visited_green_zones:
                goal = green_centers[i]
                rrt = RRT(current_pos, goal, grid)
                segment_path = rrt.build()

                if segment_path:
                    # Skip the current position in the next segment
                    path.extend(segment_path[1:])
                    # Update the bot's current position to the goal
                    current_pos = segment_path[-1]
                    # Mark the green zone as visited
                    visited_green_zones.add(i)
                    break
                else:
                    print(f"Failed to find path to green zone {i}")

        else:
            print("No valid path found to any green zone")
            break  # No more reachable green zones

    return path
# Cluster green zones based on the number of robots


def cluster_green_zones(green_zones, num_robots):
    green_centers = []
    for zone in green_zones:
        center_x = (zone[0][0] + zone[2][0]) // 2
        center_y = (zone[0][1] + zone[2][1]) // 2
        green_centers.append([center_x, center_y])

    kmeans = KMeans(n_clusters=num_robots, random_state=0).fit(green_centers)
    clusters = [[] for _ in range(num_robots)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(green_zones[i])
    return clusters


def assign_clusters_to_robots(botsPose, clusters):
    num_robots = len(botsPose)
    cluster_assignments = [None] * num_robots
    unassigned_clusters = set(range(num_robots))

    for i, bot_pose in enumerate(botsPose):
        min_dist = float('inf')
        best_cluster = None
        for j in unassigned_clusters:
            cluster_center = np.mean(
                [[(zone[0][0] + zone[2][0]) // 2, (zone[0][1] + zone[2][1]) // 2] for zone in clusters[j]], axis=0)
            dist = manhattan_distance(bot_pose, cluster_center)
            if dist < min_dist:
                min_dist = dist
                best_cluster = j

        cluster_assignments[i] = best_cluster
        unassigned_clusters.remove(best_cluster)

    return cluster_assignments


def main():
    start_time = time.time()
    botsPose = get_botPose_list()
    greenZones = get_greenZone_list()
    num_robots = len(botsPose)
    clusters = cluster_green_zones(greenZones, num_robots)
    cluster_assignments = assign_clusters_to_robots(botsPose, clusters)

    whole_path = [[] for _ in range(num_robots)]


    for i, bot_pose in enumerate(botsPose):
        cluster_idx = cluster_assignments[i]
        path = planBotPath(bot_pose, clusters[cluster_idx])
        whole_path[i] = path

    total_path_length = sum(len(path) for path in whole_path)

    end_time = time.time()

    grid_new = getMap()
    for robot_idx, sub_path in enumerate(whole_path):
        color = robot_colors[robot_idx % len(robot_colors)]
        for x, y in sub_path:
            grid_new[x][y] = np.array(color)

    plt.imshow(grid_new)
    plt.title(f"RRT GRID MAP (MULTI AGENT) - Path Distance: {total_path_length}, Time Taken: {end_time - start_time:.2f} s")
    plt.show()

main()
