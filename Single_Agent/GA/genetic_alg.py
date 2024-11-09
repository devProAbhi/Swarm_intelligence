import numpy as np
import random
import time
import matplotlib.pyplot as plt
from visualizer import get_obstacles_list, get_botPose_list, get_greenZone_list, set_new_map, getMap

set_new_map()

grid_size = 200  # Assuming a 200x200 grid

# Define movement directions (4 cardinal directions)
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

def makeGrid():
    """Creates a 2D grid where obstacles are marked as 0 (non-walkable) and other cells as 1 (walkable)."""
    grid = np.ones((grid_size, grid_size), dtype=int)
    obstacles = get_obstacles_list()
    for obs in obstacles:
        x1, y1 = obs[0]
        x2, y2 = obs[2]
        grid[x1:x2 + 1, y1:y2 + 1] = 0  # Mark obstacles
    return grid

# Calculate Manhattan distance
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Calculate the total distance of the path starting from botsPose
def calculate_path_distance(path, start_pos, green_centers):
    distance = manhattan_distance(start_pos, green_centers[path[0]])  # Start from bot's position
    for i in range(len(path) - 1):
        distance += manhattan_distance(green_centers[path[i]], green_centers[path[i + 1]])
    return distance

# Fitness function: Lower distance = better fitness
def fitness(path, start_pos, green_centers):
    return 1 / (calculate_path_distance(path, start_pos, green_centers) + 1)  # We add 1 to avoid division by zero

# Crossover: Ordered Crossover (OX)
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    crossover_point1 = random.randint(0, size // 2)
    crossover_point2 = random.randint(size // 2, size)

    child1 = [None] * size
    child2 = [None] * size

    # Copy the section from parent1 to child1
    for i in range(crossover_point1, crossover_point2):
        child1[i] = parent1[i]

    # Fill the remaining spots with elements from parent2, ensuring no duplicates
    current_pos = crossover_point2
    for i in range(size):
        if parent2[i] not in child1:
            while child1[current_pos % size] is not None:
                current_pos += 1
            child1[current_pos % size] = parent2[i]

    # Repeat for child2
    for i in range(crossover_point1, crossover_point2):
        child2[i] = parent2[i]

    current_pos = crossover_point2
    for i in range(size):
        if parent1[i] not in child2:
            while child2[current_pos % size] is not None:
                current_pos += 1
            child2[current_pos % size] = parent1[i]

    return child1, child2

# Mutation: Swap mutation
def mutate(path):
    size = len(path)
    i, j = random.sample(range(size), 2)  # Pick two distinct indices
    path[i], path[j] = path[j], path[i]
    return path

# Tournament selection
def tournament_selection(population, fitness_scores, tournament_size=50):
    selected = random.sample(range(len(population)), tournament_size)
    best = selected[0]
    for idx in selected[1:]:
        if fitness_scores[idx] > fitness_scores[best]:
            best = idx
    return population[best]

# Create initial population
def create_population(size, num_green_zones):
    population = []
    for _ in range(size):
        population.append(random.sample(range(num_green_zones), num_green_zones))
    return population

# Genetic Algorithm
def genetic_algorithm(start_pos, green_centers, population_size=1000, generations=1000, mutation_rate=0.1):
    population = create_population(population_size, len(green_centers))
    best_path = None
    best_fitness = -float('inf')

    for gen in range(generations):
        fitness_scores = [fitness(path, start_pos, green_centers) for path in population]

        # Track the best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_path = population[max_fitness_idx]

        # Selection
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child1, child2 = ordered_crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return best_path

# Plan the bot's path using the genetic algorithm
def planBotPath():
    start_time = time.time()
    grid = makeGrid()
    botsPose = get_botPose_list()
    greenZones = get_greenZone_list()

    # Start position (the bot's initial position)
    start_pos = botsPose[0]  # Bot's starting position
    start = (start_pos[0], start_pos[1])

    # Get the center of each green zone
    green_centers = []
    for green_zone in greenZones:
        center_x = (green_zone[0][0] + green_zone[2][0]) // 2
        center_y = (green_zone[0][1] + green_zone[2][1]) // 2
        green_centers.append((center_x, center_y))

    # Run the genetic algorithm to find the optimal path
    best_path = genetic_algorithm(start_pos, green_centers)

    # Plot the path on the map
    grid_new = getMap()
    for i in range(len(best_path) - 1):
        start = green_centers[best_path[i]]
        end = green_centers[best_path[i + 1]]
        plt.plot([start[1], end[1]], [start[0], end[0]], color="blue")

    total_time_taken = time.time() - start_time
    plt.imshow(grid_new)
    plt.title(f"GENETIC ALGORITHM GRID MAP (Path Distance: {len(best_path)}, Time Taken: {total_time_taken:.2f} s)")
    plt.show()

planBotPath()
