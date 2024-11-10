# Multi agent path optimization
### Tree Search Algorithms in Pathfinding

In pathfinding and robotics, tree search algorithms are used to explore a space (often represented as a graph) to find the optimal (shortest or best) path from a starting point to a goal. These algorithms are crucial for solving various types of search problems, including robotics navigation, AI decision-making, and more. The algorithms discussed below help find the best path in a graph, which is typically represented as a set of nodes (or points) connected by edges (or paths).

### 1. **Dijkstra’s Algorithm (1959)**

**Dijkstra's Algorithm** is one of the most famous shortest-path algorithms. It efficiently finds the shortest path from a starting node to all other nodes in a weighted graph, guaranteeing the optimal solution. The algorithm works by gradually expanding the explored area of the graph using the following steps:

#### How it works:
1. **Initialization:**
   - The starting node is assigned a distance of `0`.
   - All other nodes are initially assigned a distance of infinity.
   - Create two sets of nodes:
     - **Visited set**: This set holds nodes that have already been processed.
     - **Unvisited set**: This set holds nodes that haven't been processed yet.

2. **Distance updates:**
   - From the current node, examine each neighboring node. If moving to a neighbor via the current node offers a shorter path than previously known, update the neighbor’s distance.

3. **Moving to the next node:**
   - After all neighbors are evaluated, move the current node to the "visited" set and select the unvisited node with the smallest distance value to continue the process.

4. **Repeat:** 
   - The algorithm continues until all nodes have been moved to the visited set, and their shortest distances are known.

#### Example:
Given a graph with nodes I, II, III, IV, V, VI, the algorithm would work by updating distances from the starting node (I) to the rest of the nodes. The shortest path from node I to node VI can be determined by following the shortest paths updated at each step.

In the example:
- From node I, the distances to IV, III, and II are updated as 7, 5, and 3 respectively.
- From node II, distances to III and VI are updated as 4 and 15 respectively.
- The algorithm continues expanding and updating until it finds the shortest path: I → II → III → V → VI.

### 2. **A* Algorithm (1968)**

The **A* Algorithm** is an extension of Dijkstra’s Algorithm. It improves performance by incorporating **heuristics**, which help guide the search towards the goal more efficiently. A* does this by considering both the cost to reach a node from the start (like Dijkstra’s) and an estimate of the remaining cost to the goal.

#### How it works:
1. **Heuristic Function**:
   - Each node is assigned two values:
     - `g(n)`: The exact cost from the starting node to the current node.
     - `h(n)`: A heuristic estimate of the cost to the goal from the current node (e.g., Euclidean distance).
   - The total cost to move through a node is given by `f(n) = g(n) + h(n)`, where `f(n)` is the total estimated cost to the goal.

2. **Search Process**:
   - Like Dijkstra’s, A* maintains a list of unvisited nodes and evaluates each one, but it uses the `f(n)` value to prioritize nodes that are more likely to lead to a faster solution.
   - Nodes with lower `f(n)` values are explored first, which reduces unnecessary exploration of distant or irrelevant paths.

This makes A* much faster than Dijkstra’s in many scenarios, especially when you have a good heuristic function.

### 3. **D* Algorithm (1994)**

The **D* Algorithm** (also known as Dynamic A*) is a variation of A* designed for situations where the environment changes during the search. It is an **incremental search algorithm** that can efficiently re-plan a path in response to dynamic changes (such as obstacles moving in a robot’s environment).

#### How it works:
1. **Incremental Updates**:
   - D* adapts A* by incrementally updating its search when new information becomes available (e.g., obstacles).
   - Instead of recalculating the entire path from scratch, D* reuses previous search results, updating only the parts of the graph affected by the new changes.

2. **Use of Heuristics**:
   - D* incorporates heuristic estimates to guide the search in a similar way to A*, but it also allows for faster re-planning when the environment changes.

3. **Application**:
   - It is particularly useful in robotics, where obstacles or other dynamic elements can cause the robot to adjust its path on-the-fly without having to restart the search from the beginning.

### 4. **Sampling-based Planning**

In real-world scenarios, particularly in robotics, it’s often impractical to search through all possible paths. **Sampling-based algorithms** take a more probabilistic approach to pathfinding, focusing on generating random samples to find a feasible path rather than searching exhaustively.

#### Characteristics:
- These algorithms are **probabilistically complete**, meaning that as the amount of time and samples increase, the probability of finding an optimal path approaches 1.

### 5. **Rapidly-exploring Random Trees (RRT) (1998)**

**RRT** is a powerful algorithm used for pathfinding in high-dimensional spaces (such as in robotics) where traditional search algorithms like A* might not be efficient. It rapidly explores the search space by building a tree from the start point and expanding it randomly toward the goal.

#### How it works:
1. **Initialization**:
   - A tree is initialized with the starting point.
   
2. **Random Expansion**:
   - Random points are sampled in the space, and the tree is expanded towards these points.
   - Each expansion step checks if the new point is valid (i.e., no obstacles) and adds it to the tree if it is.

3. **Reaching the Goal**:
   - The tree continues to expand until it reaches the goal or a valid path is found.
   - **RRT* (RRT Star)** is an extension of RRT that tries to optimize the path by minimizing the overall distance, which is done by rewiring the tree and making connections between points to improve the path quality.

#### Example:
- Starting from a robot’s initial position, random points are selected, and the tree expands toward them. The algorithm doesn’t guarantee the shortest path, but it’s efficient and works well in high-dimensional spaces.
- **Collision checking** is a crucial part of RRT, and lazy evaluation techniques are often used to reduce computational costs. Only the new segments of the tree are checked for collisions, rather than checking the entire path at each step.

### Summary of Algorithms:

- **Dijkstra’s Algorithm**: Finds the shortest path in a graph with guaranteed optimality but may be slow for large graphs.
- **A* Algorithm**: Improves Dijkstra by adding heuristics to speed up search toward the goal, useful in scenarios where a good heuristic is available.
- **D* Algorithm**: An extension of A* for dynamic environments, enabling efficient re-planning as the environment changes.
- **RRT**: A probabilistic path-planning algorithm that efficiently explores large, high-dimensional spaces, often used in robotics.
- **Sampling-based Planning**: Uses probabilistic methods to search through random samples and is especially useful when full exploration of the space is computationally infeasible.

Each of these algorithms has its strengths and is suited to different types of problems, particularly in robotics and AI, where navigating dynamic or high-dimensional spaces is essential.
