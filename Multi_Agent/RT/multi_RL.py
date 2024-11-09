import numpy as np
import time
import random
from sklearn.cluster import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming
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

# Manhattan distance (heuristic for Q-learning reward)
def calculate_heuristic(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

# Define Q-learning parameters
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
EPSILON = 0.1  # Exploration factor

# Q-learning agent
class QLearningAgent:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.q_table = np.zeros((grid_size, grid_size, len(moves)))  # (x, y, action) -> Q-value table

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            # Exploration: choose a random action
            return random.choice(range(len(moves)))
        else:
            # Exploitation: choose the action with the highest Q-value
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_value(self, state, action, reward, next_state):
        # Bellman equation
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + GAMMA * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += ALPHA * td_error

# Environment setup
def step(state, action, grid, goal):
    """Take a step in the environment and return the next state and reward."""
    next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
    
    # Check if the new state is within bounds and not an obstacle
    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size and grid[next_state[0], next_state[1]] == 1:
        # Reward: negative for every step to encourage shortest path
        reward = -1
        if next_state == goal:
            reward = 100  # High reward for reaching the goal (green zone)
        return next_state, reward
    else:
        # If it's a wall or out of bounds, stay in the same position and penalize
        return state, -100  # Large negative penalty for invalid moves

# Plan the bot's path to visit green zones exactly once
def planBotPath(botsPose, greenZones):
    start_time = time.time()
    grid = makeGrid()

    # Start position (the bot's initial position)
    start_pos = botsPose
    start = (start_pos[0], start_pos[1])

    # Initialize Q-learning agent
    agent = QLearningAgent(grid_size)

    # Get the center of each green zone (or boundary point if you prefer)
    green_centers = []
    for green_zone in greenZones:
        center_x = (green_zone[0][0] + green_zone[2][0]) // 2
        center_y = (green_zone[0][1] + green_zone[2][1]) // 2
        green_centers.append((center_x, center_y))

    # Train the agent to learn the best path using Q-learning
    for episode in range(1000):  # Number of training episodes
        current_pos = start
        visited_green_zones = set()

        while len(visited_green_zones) < len(greenZones):
            # Choose an action
            action = agent.choose_action(current_pos)
            next_state, reward = step(current_pos, action, grid, green_centers[len(visited_green_zones)])

            # Update Q-values based on the action taken
            agent.update_q_value(current_pos, action, reward, next_state)

            # If the agent reached the green zone, mark it as visited
            if next_state == green_centers[len(visited_green_zones)]:
                visited_green_zones.add(len(visited_green_zones))

            # Move to the next state
            current_pos = next_state

    # After training, the agent should know the optimal path
    path = []
    current_pos = start
    visited_green_zones = set()

    while len(visited_green_zones) < len(greenZones):
        action = agent.choose_action(current_pos)
        next_state, _ = step(current_pos, action, grid, green_centers[len(visited_green_zones)])

        # Record the path
        path.append(next_state)
        current_pos = next_state
        visited_green_zones.add(len(visited_green_zones))

    end_time = time.time()

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
            cluster_center = np.mean([[ (zone[0][0] + zone[2][0]) // 2, (zone[0][1] + zone[2][1]) // 2 ] for zone in clusters[j]], axis=0)
            dist = calculate_heuristic(bot_pose, cluster_center)
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
            grid_new[x][y] = color
        grid_new[sub_path[-1][0], sub_path[-1][1]] = color

    plt.imshow(grid_new)
    plt.show()

if __name__ == "__main__":
    main()
