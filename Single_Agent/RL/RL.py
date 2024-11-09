import numpy as np
import random
from visualizer import get_obstacles_list, get_botPose_list, get_greenZone_list,set_new_map

set_new_map()

# Define possible movement directions (8 directions)
moves = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
grid_size = 200  # Temporarily reduce grid size for testing

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

class QLearningAgent:
    def __init__(self, grid_size, moves, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.grid_size = grid_size
        self.moves = moves
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = np.zeros((grid_size, grid_size, len(moves)))  # Q-table (state-action values)

    def choose_action(self, state):
        """Choose an action using greedy strategy (exploitation only)"""
        # Exploitation: Best action based on Q-table (no random exploration)
        return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        # Q-value update
        self.q_table[state[0], state[1], action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action] - self.q_table[state[0], state[1], action]
        )

    def train(self, grid, start_pos, greenZones, episodes=100):
        for episode in range(episodes):
            state = start_pos
            total_reward = 0

            while True:
                action = self.choose_action(state)
                next_state = (state[0] + self.moves[action][0], state[1] + self.moves[action][1])

                # Check if the move is valid (within bounds and not an obstacle)
                if 0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size and grid[next_state[0], next_state[1]] == 1:
                    state = next_state
                else:
                    # Invalid move, stay in the same position
                    state = state

                # Check if bot reached a green zone
                reward = -1  # Small penalty for each step
                for zone in greenZones:
                    centroid = calculate_centroid(zone)
                    if state == centroid:
                        reward = 100  # Large reward for reaching a green zone
                        break

                # Update Q-table
                self.update_q_table(state, action, reward, state)

                total_reward += reward

                # If all green zones are visited (you can stop or add other criteria)
                if all(calculate_centroid(zone) == state for zone in greenZones):
                    break

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

def planBotPath():
    grid = makeGrid()
    botsPose = get_botPose_list()
    greenZones = get_greenZone_list()
    agent = QLearningAgent(grid_size=grid_size, moves=moves)

    # Start position of the bot
    start_pos = (botsPose[0][0], botsPose[0][1])

    # Train the agent using Q-learning for fewer episodes
    agent.train(grid, start_pos, greenZones, episodes=100)

    # Now, the agent has learned the best policy, let's move the bot through the grid
    state = start_pos
    map_display = set_map()  # 20x20 np.array for visualization (temporarily smaller)

    # Simulate the bot's path based on the learned policy
    combined_path = []
    for _ in range(50):  # Limiting to 50 steps for demonstration
        action = agent.choose_action(state)
        next_state = (state[0] + moves[action][0], state[1] + moves[action][1])

        # Check if the move is valid (within bounds and not an obstacle)
        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size and grid[next_state[0], next_state[1]] == 1:
            state = next_state
        else:
            # If the move is invalid, the bot stays in the same position
            state = state

        combined_path.append(state)

    # Draw combined path on map display
    for x, y in combined_path:
        map_display[x, y] = (0, 0, 255)  # Set path color to blue

    # Display the final map with combined path
    plt.title(f"Robot Paths on Map\nTotal Path Distance: {total_path_length}, Time Taken: {total_time_taken:.2f} seconds")


planBotPath()
