from collections import deque
import heapq
from queue import PriorityQueue
import copy
import random
import math
import sys

# --- Constants ---
Start_State = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]  # Trạng thái bắt đầu ví dụ
Goal_State = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]   # Trạng thái đích
Goal = Goal_State
Moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # Up, Down, Left, Right
Move_Names = {(-1, 0): "Up", (1, 0): "Down", (0, -1): "Left", (0, 1): "Right"} # For Q-Learning actions

# --- Helper Functions ---
def tim_X(x, goal_state=Goal):
    """Finds the coordinates (row, col) of value x in the goal state."""
    for i in range(3):
        for j in range(3):
            if goal_state[i][j] == x:
                return i, j
    return None

def khoang_cach_mahathan(matran_hientai, goal_state=Goal):
    """Calculates the Manhattan distance heuristic."""
    h_sum = 0
    if matran_hientai is None: # Handle cases where a state might be None
        return float('inf')
    for i in range(3):
        for j in range(3):
            val = matran_hientai[i][j]
            if val != 0:
                try:
                    pos_x, pos_y = tim_X(val, goal_state)
                    if pos_x is None: # Value not in goal state
                        # print(f"Warning: Value {val} not found in goal state for heuristic calculation.")
                        return float('inf') # Or handle as an error
                    h_sum += abs(i - pos_x) + abs(j - pos_y)
                except TypeError:
                    # This might happen if tim_X returns None and we don't check before unpacking
                    # print(f"Warning: Could not find value {val} in goal state for heuristic calculation.")
                    return float('inf')
    return h_sum

def Tim_0(matran_hientai):
    """Finds the coordinates (row, col) of the blank tile (0)."""
    if matran_hientai is None: return None
    for i in range(3):
        for j in range(3):
            if matran_hientai[i][j] == 0:
                return i, j
    return None

def Check(x, y):
    """Checks if coordinates (x, y) are within the 3x3 grid."""
    return 0 <= x < 3 and 0 <= y < 3

def Chiphi(matran_hientai, goal_state=Goal):
    """Calculates the number of misplaced tiles heuristic."""
    if matran_hientai is None: return float('inf')
    dem = 0
    for i in range(3):
        for j in range(3):
            if matran_hientai[i][j] != 0 and matran_hientai[i][j] != goal_state[i][j]:
                dem += 1
    return dem

def DiChuyen(matran_hientai, x, y, new_x, new_y):
    """Creates a new state by moving the blank tile."""
    new_state = copy.deepcopy(matran_hientai)
    new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
    return new_state

def print_matrix(matran):
    """Prints the matrix state."""
    if matran is None:
        print("None")
        return
    for row in matran:
        print(" ".join(map(str, row)))
    print()

def state_to_tuple(matran):
    """Converts a list-of-lists state to a tuple-of-tuples for hashing."""
    if not isinstance(matran, list):
        # print(f"Warning: Invalid state format for state_to_tuple: {matran}")
        return None
    try:
        return tuple(tuple(row) for row in matran)
    except TypeError: # If matran contains non-iterable rows or elements
        # print(f"Warning: TypeError during state_to_tuple conversion for: {matran}")
        return None


def get_neighbors(matran_hientai):
    """Generates all valid neighbor states."""
    neighbors = []
    zero_pos = Tim_0(matran_hientai)
    if zero_pos is None:
        return []
    x, y = zero_pos
    for dx, dy in Moves:
        new_x = x + dx
        new_y = y + dy
        if Check(new_x, new_y):
            new_matran = DiChuyen(matran_hientai, x, y, new_x, new_y)
            neighbors.append(new_matran)
    return neighbors

def apply_move_to_state(state_list, move_dir):
    """Applies a move direction to a state. Returns new state or original if move invalid."""
    # state_list should be a mutable list of lists
    new_state_list = copy.deepcopy(state_list) # Work on a copy
    blank_pos = Tim_0(new_state_list)
    if not blank_pos:
        # print(f"Warning: No blank tile found in state: {state_list}")
        return state_list # Return original if no blank tile

    blank_r, blank_c = blank_pos
    move_r, move_c = move_dir # move_dir is like (-1, 0) for UP

    # The tile to move INTO the blank space is at (blank_r + move_r, blank_c + move_c)
    # This is confusing. Let's re-think: move_dir is the direction the BLANK tile moves.
    # So, new blank position is (blank_r + move_r, blank_c + move_c)
    # The tile at that new position moves to the old blank position.

    tile_to_move_r, tile_to_move_c = blank_r - move_r, blank_c - move_c # This is the tile that will move into current blank
    
    # Correction: `move_dir` typically indicates where the blank space *itself* goes.
    # Or, equivalently, which adjacent tile moves *into* the blank space.
    # If move_dir = (-1,0) (Up), it means the tile *below* the blank moves *Up* into the blank space.
    # The blank space effectively moves Down.

    # Let's assume `move_dir` is the (dr, dc) for the blank tile.
    new_blank_r, new_blank_c = blank_r + move_r, blank_c + move_c

    if Check(new_blank_r, new_blank_c):
        # Swap the blank tile with the tile at the new blank position
        new_state_list[blank_r][blank_c], new_state_list[new_blank_r][new_blank_c] = \
            new_state_list[new_blank_r][new_blank_c], new_state_list[blank_r][blank_c]
        return new_state_list
    else:
        # print(f"Warning: Move {move_dir} from {blank_pos} is invalid for state {state_list}")
        return state_list # Return original if move is invalid

def belief_state_to_tuple_for_hashing(belief_state):
    """Converts belief state (iterable of state tuples) into sorted hashable frozenset."""
    if not belief_state:
        return frozenset()
    try:
        # Ensure all elements in belief_state are already tuples or convert them
        valid_state_tuples = []
        for s in belief_state:
            if isinstance(s, list): # If it's a list of lists (matrix)
                s_tuple = state_to_tuple(s)
                if s_tuple: valid_state_tuples.append(s_tuple)
            elif isinstance(s, tuple): # If it's already a tuple of tuples
                valid_state_tuples.append(s)
            # else: ignore malformed states
        return frozenset(valid_state_tuples)

    except Exception as e:
        # print(f"Error converting belief state to hashable: {e}, belief_state: {belief_state}")
        return None

# --- Search Algorithms ---
# (BFS, UCS, DFS, IDDFS, Greedy, A_Star, IDA_Star,
# Simple_HC, Steepest_HC, Stochastic_HC, Beam_Search, Simulated_Annealing, AND-OR Search)
# ... (Existing algorithms from the provided ThuatToan.py file) ...
def BFS(start, goal=Goal):
    """Breadth-First Search."""
    queue = deque([(start, [])])
    visited = {state_to_tuple(start)}
    while queue:
        matran_hientai, path = queue.popleft()
        if matran_hientai == goal:
            return path + [matran_hientai]
        # x, y = Tim_0(matran_hientai) # Not needed if using get_neighbors
        for new_matran in get_neighbors(matran_hientai):
            new_matran_tuple = state_to_tuple(new_matran)
            if new_matran_tuple not in visited:
                visited.add(new_matran_tuple)
                new_path = path + [matran_hientai]
                queue.append((new_matran, new_path))
    return None

def UCS(start, goal=Goal):
    """Uniform Cost Search."""
    qp = PriorityQueue()
    start_tuple = state_to_tuple(start)
    qp.put((0, start, [])) # (cost, state, path_to_state)
    visited = {start_tuple: 0} # Store min cost to reach a state

    while not qp.empty():
        cost, matran_hientai, path = qp.get()

        if matran_hientai == goal:
            return path + [matran_hientai]

        current_tuple = state_to_tuple(matran_hientai)
        if cost > visited.get(current_tuple, float('inf')): # If we found a shorter path already
            continue

        for new_matran in get_neighbors(matran_hientai):
            new_matran_tuple = state_to_tuple(new_matran)
            new_cost = cost + 1 # Assuming cost of each move is 1
            if new_cost < visited.get(new_matran_tuple, float('inf')):
                visited[new_matran_tuple] = new_cost
                new_path = path + [matran_hientai]
                qp.put((new_cost, new_matran, new_path))
    return None

def DFS(start, goal=Goal):
    """Depth-First Search (Iterative)."""
    stack = [(start, [])] # (state, path_to_state)
    visited = {state_to_tuple(start)}

    while stack:
        matran_hientai, path = stack.pop()

        if matran_hientai == goal:
            return path + [matran_hientai]
        
        # To explore in a consistent order (optional, but good for reproducibility)
        # Neighbors are typically generated in Up, Down, Left, Right order.
        # For DFS, we usually add them to stack in reversed order of exploration preference.
        neighbors = get_neighbors(matran_hientai)
        for new_matran in reversed(neighbors): # Process last generated neighbor first
            new_matran_tuple = state_to_tuple(new_matran)
            if new_matran_tuple not in visited:
                visited.add(new_matran_tuple) # Mark visited when adding to stack to avoid cycles
                new_path = path + [matran_hientai]
                stack.append((new_matran, new_path))
    return None

def DFS_limited(current_matran, goal, limit, path_so_far, visited_in_current_path_tuples):
    """Recursive helper for DLS and IDDFS."""
    if current_matran == goal:
        return path_so_far + [current_matran]
    if limit <= 0:
        return "cutoff" # Indicate that depth limit was reached

    current_matran_tuple = state_to_tuple(current_matran)
    # visited_in_current_path_tuples.add(current_matran_tuple) # Add current node to path visited

    any_remaining_path = False
    for new_matran in get_neighbors(current_matran):
        new_matran_tuple = state_to_tuple(new_matran)
        if new_matran_tuple not in visited_in_current_path_tuples: # Avoid cycles in current path
            # Create a new set for the recursive call to avoid interference
            new_visited_tuples = visited_in_current_path_tuples.copy()
            new_visited_tuples.add(current_matran_tuple)

            result = DFS_limited(new_matran, goal, limit - 1, path_so_far + [current_matran], new_visited_tuples)
            if result == "cutoff":
                any_remaining_path = True
            elif result is not None: # Solution found
                return result
    
    # visited_in_current_path_tuples.remove(current_matran_tuple) # Backtrack: remove current node from path visited
    return "cutoff" if any_remaining_path else None # None if no solution and no cutoff (fully explored this branch)


def IDDFS(start, goal=Goal, max_depth=30): # Max depth for 8-puzzle usually around 30
    """Iterative Deepening Depth-First Search."""
    for depth in range(max_depth + 1):
        # print(f"IDDFS: Trying depth {depth}")
        # For each depth, we need a fresh set of visited_in_path for the DLS call.
        # The main `visited` in IDDFS is implicitly handled by the depth limit and path cycle check.
        result = DFS_limited(start, goal, depth, [], set()) # Path starts empty, visited in path is initially empty
        if result is None: # Fully explored up to depth, no solution
            # print(f"IDDFS: No solution found up to depth {depth}, and no cutoff occurred.")
            return None 
        if result != "cutoff": # Solution found
            # print(f"IDDFS: Solution found at depth {depth}")
            return result
        # If result is "cutoff", continue to next depth
    # print(f"IDDFS: No solution found within max_depth {max_depth}")
    return None


def Greedy(start, goal=Goal, h_func=khoang_cach_mahathan):
    """Greedy Best-First Search."""
    qp = PriorityQueue() # Stores (heuristic_value, state, path)
    start_tuple = state_to_tuple(start)
    qp.put((h_func(start, goal), start, []))
    visited = {start_tuple} # To avoid cycles and redundant explorations

    while not qp.empty():
        h_cost, matran_hientai, path = qp.get()

        if matran_hientai == goal:
            return path + [matran_hientai]
        
        for new_matran in get_neighbors(matran_hientai):
            new_matran_tuple = state_to_tuple(new_matran)
            if new_matran_tuple not in visited:
                visited.add(new_matran_tuple)
                new_h_cost = h_func(new_matran, goal)
                new_path = path + [matran_hientai]
                qp.put((new_h_cost, new_matran, new_path))
    return None

def A_Star(start, goal=Goal, h_func=khoang_cach_mahathan):
    """A* Search."""
    # Priority queue stores (f_cost, g_cost, state, path)
    # g_cost is included for tie-breaking or re-opening nodes if a shorter path is found
    qp = PriorityQueue()
    start_tuple = state_to_tuple(start)
    
    g_costs = {start_tuple: 0} # Cost from start to node
    initial_h = h_func(start, goal)
    if initial_h == float('inf'): return None # Unsolvable from start based on heuristic (e.g. invalid state)

    qp.put((g_costs[start_tuple] + initial_h, g_costs[start_tuple], start, []))
    # visited is implicitly handled by checking g_costs: if a state is in g_costs, it has been visited/queued.

    while not qp.empty():
        f_n, cost_so_far, matran_hientai, path = qp.get()

        if matran_hientai == goal:
            return path + [matran_hientai]

        current_tuple = state_to_tuple(matran_hientai)
        # If we pulled a state from PQ for which we've already found a
        # shorter or equal path and processed it, skip.
        if cost_so_far > g_costs.get(current_tuple, float('inf')):
            continue
        
        for new_matran in get_neighbors(matran_hientai):
            new_g_cost = cost_so_far + 1 # Cost of each step is 1
            new_matran_tuple = state_to_tuple(new_matran)

            if new_g_cost < g_costs.get(new_matran_tuple, float('inf')):
                g_costs[new_matran_tuple] = new_g_cost
                h_val = h_func(new_matran, goal)
                if h_val == float('inf') : continue # Don't add unsolvable states to PQ

                new_f_cost = new_g_cost + h_val
                new_path = path + [matran_hientai]
                qp.put((new_f_cost, new_g_cost, new_matran, new_path))
    return None

def ida_star_search_recursive(current_path, g_cost, bound, goal, h_func):
    """Recursive step for IDA*."""
    current_node = current_path[-1]
    h_cost = h_func(current_node, goal)
    f_cost = g_cost + h_cost

    if f_cost > bound:
        return f_cost # This is the new minimum bound if this path was over
    if current_node == goal:
        return "FOUND" 
    
    min_exceeded_cost = float('inf') # Smallest f-cost that exceeded the bound

    for neighbor in get_neighbors(current_node):
        # Avoid going directly back to the previous state in the path
        if len(current_path) > 1 and neighbor == current_path[-2]:
            continue

        current_path.append(neighbor)
        result = ida_star_search_recursive(current_path, g_cost + 1, bound, goal, h_func)
        
        if result == "FOUND":
            return "FOUND"
        if result < min_exceeded_cost:
            min_exceeded_cost = result
        
        current_path.pop() # Backtrack

    return min_exceeded_cost


def IDA_Star(start, goal=Goal, h_func=khoang_cach_mahathan):
    """Iterative Deepening A* Search."""
    bound = h_func(start, goal)
    if bound == float('inf'): return None # Unsolvable from start

    path = [start] # Current path being explored

    while True:
        # print(f"IDA*: Trying bound {bound}")
        result = ida_star_search_recursive(path, 0, bound, goal, h_func)
        
        if result == "FOUND":
            return path # The path list is modified in-place and holds the solution
        if result == float('inf'): # No solution found, and no further paths to explore
            return None 
        
        bound = result # Update bound to the smallest f-cost that exceeded the previous bound
        # Reset path for next iteration to just the start state, as the recursive function modifies it.
        path = [start]


def Simple_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_iterations=1000):
    """Simple Hill Climbing."""
    current = start
    path = [current]
    iterations = 0
    while iterations < max_iterations :
        iterations +=1
        current_h = h_func(current, goal)
        if current_h == 0 and current == goal: # Check both heuristic and state equality
            return path
        
        neighbors = get_neighbors(current)
        if not neighbors: return path # Stuck

        best_neighbor_found = False
        for neighbor in neighbors: # Order of neighbors might matter
            if h_func(neighbor, goal) < current_h:
                current = neighbor
                path.append(current)
                best_neighbor_found = True
                break # Take the first better neighbor

        if not best_neighbor_found: # No better neighbor found
            return path # Reached local optimum or plateau
    return path # Max iterations reached


def Steepest_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_iterations=1000):
    """Steepest Ascent Hill Climbing."""
    current = start
    path = [current]
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        current_h = h_func(current, goal)

        if current_h == 0 and current == goal:
            return path

        neighbors = get_neighbors(current)
        if not neighbors: return path

        best_neighbor = None
        best_h = current_h

        for neighbor in neighbors:
            neighbor_h = h_func(neighbor, goal)
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbor = neighbor
        
        if best_neighbor is None or best_h >= current_h: # No better neighbor or stuck on plateau/local optimum
            return path
        
        current = best_neighbor
        path.append(current)
    return path # Max iterations reached


def Stochastic_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_steps=1000):
    """Stochastic Hill Climbing."""
    current = start
    path = [current]
    steps = 0
    while steps < max_steps:
        current_h = h_func(current, goal)
        if current == goal:
            return path
        
        neighbors = get_neighbors(current)
        better_neighbors = []
        for neighbor in neighbors:
            if h_func(neighbor, goal) < current_h:
                better_neighbors.append(neighbor)
        
        if not better_neighbors: # No strictly better neighbors
            # Option: allow sideways moves or stop. Here, we stop.
            return path # Reached local optimum or plateau
            
        chosen_neighbor = random.choice(better_neighbors)
        current = chosen_neighbor
        path.append(current)
        steps += 1
    return path # Max steps reached


def Beam_Search(start, goal=Goal, h_func=khoang_cach_mahathan, beam_width_k=3, max_iterations=100):
    """Beam Search."""
    # Each element in the beam: (heuristic_value, state, path_to_state)
    beam = [(h_func(start, goal), start, [start])]
    
    for iteration in range(max_iterations):
        if not beam: return None # No candidates left

        # Check if goal is in current beam
        for h_val, state, pth in beam:
            if state == goal:
                return pth

        successors = [] # Stores (heuristic, state, path) for all children of states in beam
        
        # Generate all successors for states in the current beam
        for h_cost, current_state, path_taken in beam:
            if current_state == goal: # Should have been caught above, but good check
                return path_taken

            for neighbor in get_neighbors(current_state):
                # Avoid cycles within a path for beam search (optional, but can help)
                # A global visited set is usually not used in pure beam search to allow re-exploration
                # if a path falls out of the beam and a new path reaches the same state via better route.
                # However, simple cycle check in path is useful.
                is_cycle = False
                for node_in_path in path_taken:
                    if neighbor == node_in_path:
                        is_cycle = True
                        break
                if not is_cycle:
                    new_h = h_func(neighbor, goal)
                    new_path = path_taken + [neighbor]
                    successors.append((new_h, neighbor, new_path))
        
        if not successors: # No valid successors generated
            # Return the best path found so far from the current beam (last state before no successors)
            beam.sort(key=lambda item: item[0])
            return beam[0][2] if beam else None 

        # Sort all successors by heuristic and select the best k
        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width_k]

    # Max iterations reached, return the best path from the current beam
    if beam:
        beam.sort(key=lambda item: item[0]) # Sort by heuristic
        return beam[0][2] # Return path of the best state
    return None


def Simulated_Annealing(start, goal=Goal, h_func=khoang_cach_mahathan, 
                        initial_temp=100.0, cooling_rate=0.95, min_temp=0.1, 
                        max_iterations_per_temp=100): # Max iterations overall controlled by cooling
    """Simulated Annealing Search."""
    current_state = start
    current_h = h_func(current_state, goal)
    
    # Path tracking is tricky for SA as it can move to worse states.
    # We can track the sequence of *accepted* states, or just the best found.
    # For 8-puzzle, if we want a path, we usually want the path to the *first time goal is hit*
    # or the path to the *best state if goal not hit*.
    # The GUI expects a path of states.
    
    path_taken = [current_state] # Tracks the sequence of states the algorithm visits
    best_state_so_far = current_state
    best_h_so_far = current_h
    path_to_best_so_far = [current_state]

    temp = initial_temp
    
    iteration = 0
    max_total_iterations = 50000 # Safety break for total iterations
    
    while temp > min_temp and iteration < max_total_iterations:
        if current_state == goal:
            # print(f"SA: Goal found at iteration {iteration}, temp {temp:.2f}")
            return path_taken # Return the path that led to the goal

        for _ in range(max_iterations_per_temp): # Iterations at current temperature
            iteration += 1
            if iteration >= max_total_iterations: break

            neighbors = get_neighbors(current_state)
            if not neighbors:
                break # Stuck, no neighbors to move to

            next_state = random.choice(neighbors)
            next_h = h_func(next_state, goal)
            
            delta_e = next_h - current_h # Change in "energy" (heuristic value)

            accept_move = False
            if delta_e < 0: # Better state
                accept_move = True
            else: # Worse state, accept with probability
                if temp > 1e-9: # Avoid division by zero if temp is extremely small
                    probability = math.exp(-delta_e / temp)
                    accept_move = random.random() < probability
            
            if accept_move:
                current_state = next_state
                current_h = next_h
                path_taken.append(current_state) # Add accepted state to path

                if current_h < best_h_so_far:
                    best_state_so_far = current_state
                    best_h_so_far = current_h
                    path_to_best_so_far = list(path_taken) # Save copy of path to this new best
                
                if current_state == goal: # Check again after move
                    # print(f"SA: Goal found during temp iter at iteration {iteration}, temp {temp:.2f}")
                    return path_taken
        
        temp *= cooling_rate
        if not neighbors: break # Break outer loop if stuck

    # If goal not found, return path to the best state encountered, or full path taken
    # print(f"SA: Max temp/iterations reached. Best H found: {best_h_so_far}")
    if best_state_so_far == goal : return path_to_best_so_far
    return path_taken # Returns the full exploration path if goal not found, or path to best if preferred

def and_or_search_8puzzle_recursive(current_state, goal_state, path_visited_tuples, path_so_far_states):
    """Recursive AND-OR search simulation (DFS-based)."""
    current_tuple = state_to_tuple(current_state)
    if current_tuple is None: return None # Invalid state

    if current_state == goal_state:
        return path_so_far_states + [current_state]

    if current_tuple in path_visited_tuples: # Cycle detected in current exploration branch
        return None

    path_visited_tuples.add(current_tuple) # Mark as visited for this branch

    # This is more like a DFS than a true AND-OR graph search which handles AND/OR nodes explicitly.
    # For 8-puzzle, moves from a state are OR choices.
    
    # OR part: try moves one by one
    for neighbor in get_neighbors(current_state):
        # Recursive call
        # For the recursive call, pass a copy of path_visited if it's specific to the branch
        # If path_visited is global for the DFS, don't copy, but then it's standard DFS.
        # For this "simulation", assume it's exploring alternatives.
        
        # Create a new path_visited set for the sub-problem to allow re-visiting nodes via different paths
        # if this were a more complex AND-OR. But for simple path finding, a global visited set is more efficient (like DFS).
        # Let's stick to the spirit of simple DFS-like exploration for this placeholder.
        
        result_path = and_or_search_8puzzle_recursive(
            neighbor, goal_state, path_visited_tuples.copy(), path_so_far_states + [current_state]
        ) # .copy() on visited if we want to explore alternative paths to same node fully.

        if result_path is not None: # Solution found down this OR branch
            return result_path 
            # For true AND-OR, if this was an AND node, we'd need all children to return solutions.
            # But 8-puzzle actions are OR.

    # If no move from current_state leads to solution in this branch
    # path_visited_tuples.remove(current_tuple) # Backtrack: remove from visited if it was branch-specific
    return None

def solve_with_and_or_8puzzle(start_state, goal_state=Goal):
    """Wrapper for AND-OR search simulation (DFS-like)."""
    # Using a set for visited_tuples to prevent cycles effectively during the search.
    # The path_so_far_states tracks the actual sequence of states in the current path.
    result_path = and_or_search_8puzzle_recursive(start_state, goal_state, set(), [])
    return result_path

def Backtracking_Search(start, goal=Goal):
    """
    Implements Backtracking search for the 8-puzzle.
    This is essentially a Depth-First Search.
    """
    # Stack stores (current_state, path_to_this_state)
    stack = [(start, [])]
    # Visited set stores tuples of states to avoid cycles and redundant work
    visited_tuples = {state_to_tuple(start)}

    while stack:
        current_state, path = stack.pop()

        if current_state == goal:
            return path + [current_state]

        # Generate neighbors (potential next states)
        # Process in reverse order so that when popped, it mimics typical recursive DFS order
        for neighbor in reversed(get_neighbors(current_state)): 
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited_tuples:
                visited_tuples.add(neighbor_tuple)
                stack.append((neighbor, path + [current_state]))
    
    return None # No solution found

# --- Genetic Algorithm (Simple Placeholder) ---
def initialize_population_8puzzle(size, start_state, goal_state, max_random_moves=20):
    population = []
    for _ in range(size):
        current = copy.deepcopy(start_state)
        path = [current]
        # Create an individual by a random walk, not necessarily reaching goal
        for _ in range(random.randint(5, max_random_moves)): # Random length of path
            neighbors = get_neighbors(current)
            if not neighbors: break
            current = random.choice(neighbors)
            # Avoid immediate cycles in random walk to make paths more diverse
            if len(path) > 1 and current == path[-2]: 
                # Try another neighbor if possible, or just take it
                if len(neighbors) > 1:
                    current = random.choice([n for n in neighbors if n != path[-2]] or neighbors)
            path.append(current)
        population.append(path) # Each individual is a path (list of states)
    return population

def fitness_8puzzle(individual_path, goal_state, h_func=khoang_cach_mahathan):
    if not individual_path: return -float('inf')
    final_state = individual_path[-1]
    
    heuristic_val = h_func(final_state, goal_state)
    if heuristic_val == float('inf') : return -float('inf') # Very unfit for invalid states

    # Fitness: higher is better. Lower heuristic is better.
    # Consider path length: shorter paths to good states are better.
    fitness = 1.0 / (1.0 + heuristic_val + 0.05 * len(individual_path)) 
    
    if final_state == goal_state:
        fitness *= 2 # Bonus for reaching goal
    return fitness

def select_parents_8puzzle(population, fitness_values, num_parents):
    # Tournament selection or roulette wheel could be used.
    # Simple: sort by fitness and pick top N.
    sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    parents = [ind for ind, fit in sorted_population[:num_parents]]
    return parents if parents else [random.choice(population)] # Ensure at least one parent

def crossover_8puzzle(parent1_path, parent2_path, start_state):
    # Crossover for paths is complex to ensure validity.
    # Simple approach: single point crossover on state sequences.
    # Then, try to make the children valid paths from start_state. This is hard.
    # A more common GA for pathfinding might evolve sequences of *actions*.
    
    # Let's try a simpler crossover: take prefix from one, suffix from other,
    # but this often creates invalid discontinuous paths for 8-puzzle states.
    # For simplicity, let's make children by picking one parent and slightly modifying it,
    # or just return copies of parents if crossover is too complex for this placeholder.
    
    # A very basic crossover:
    # child1 starts like parent1, child2 starts like parent2
    # Try to append a segment from the other parent IF the connection is valid
    # This is still non-trivial.
    
    # Fallback: return perturbed versions of parents or just copies.
    # For a placeholder, we might just return slightly mutated parents as children.
    # Or, a very naive crossover:
    p1_len = len(parent1_path)
    p2_len = len(parent2_path)
    if p1_len < 2 or p2_len < 2: return list(parent1_path), list(parent2_path)

    # Try to find a common state or a valid connection point (hard)
    # Simplest: one-point crossover on the sequence of states
    point1 = random.randint(1, p1_len -1)
    point2 = random.randint(1, p2_len -1)

    child1_path = parent1_path[:point1] + parent2_path[point2:]
    child2_path = parent2_path[:point2] + parent1_path[point1:]
    
    # These children might be invalid (disconnected).
    # A true GA for 8-puzzle would likely evolve action sequences or use repairing.
    # For this placeholder, we accept potentially disconnected paths and rely on fitness/mutation.
    return child1_path, child2_path


def mutate_8puzzle(individual_path, mutation_rate, start_state, goal_state, h_func, max_perturb_moves=3):
    if not individual_path or random.random() > mutation_rate:
        return individual_path

    mutated_path = list(individual_path) # Work on a copy
    if not mutated_path: return mutated_path

    # Mutation strategy:
    # 1. Pick a random point in the path.
    # 2. From that state, perform a few random valid moves.
    # 3. Replace the rest of the path with this new segment.
    
    if len(mutated_path) == 0 : return []

    mutation_idx = random.randint(0, len(mutated_path) - 1)
    current_state_at_mutation = mutated_path[mutation_idx]
    
    new_segment = []
    temp_state = copy.deepcopy(current_state_at_mutation)

    for _ in range(random.randint(1, max_perturb_moves)):
        neighbors = get_neighbors(temp_state)
        if not neighbors: break
        
        # Prefer moves that don't immediately undo, or head towards goal slightly
        best_neighbor = None
        min_h = h_func(temp_state, goal_state)

        # Simple random choice, could be smarter
        next_s = random.choice(neighbors)

        # Avoid going back to the immediate previous state in the new segment if possible
        if new_segment and next_s == new_segment[-1]:
            if len(neighbors)>1:
                next_s = random.choice([n for n in neighbors if n != new_segment[-1]] or neighbors)
        
        temp_state = next_s
        new_segment.append(temp_state)
        if temp_state == goal_state: break # If mutation reaches goal, stop early

    # Construct new path
    final_mutated_path = mutated_path[:mutation_idx] + new_segment
    if not final_mutated_path : # If mutation resulted in empty path (e.g. from index 0 and no new segment)
        return [start_state] # Return at least the start state
    return final_mutated_path


def Genetic_Algorithm_8Puzzle(start_state, goal_state=Goal,
                             population_size=50, generations=30, # GA needs more generations usually
                             mutation_rate=0.2, num_parents_mating_ratio=0.4, # Ratio of pop to be parents
                             h_func=khoang_cach_mahathan):
    """
    Genetic Algorithm for 8-Puzzle. (Simplified placeholder)
    Returns the best path found. Might not reach the goal.
    """
    # print(f"Running GA: Pop={population_size}, Gens={generations}, MutRate={mutation_rate}")
    population = initialize_population_8puzzle(population_size, start_state, goal_state)
    best_overall_individual = None
    best_overall_fitness = -float('inf')

    num_parents = int(population_size * num_parents_mating_ratio)
    if num_parents < 2 : num_parents = 2 # Need at least 2 parents for crossover
    if num_parents % 2 != 0 : num_parents +=1 # Ensure even number for pairing if needed by crossover

    for gen in range(generations):
        fitness_values = [fitness_8puzzle(ind, goal_state, h_func) for ind in population]

        current_gen_best_idx = -1
        if any(fitness_values): # Check if there are any valid fitness values
            current_gen_best_fitness = -float('inf')
            for i, f_val in enumerate(fitness_values):
                if f_val > current_gen_best_fitness:
                    current_gen_best_fitness = f_val
                    current_gen_best_idx = i
            
            if current_gen_best_idx != -1 and current_gen_best_fitness > best_overall_fitness:
                best_overall_fitness = current_gen_best_fitness
                best_overall_individual = copy.deepcopy(population[current_gen_best_idx])
                # print(f"Gen {gen}: New best. Fitness: {best_overall_fitness:.4f}, PathLen: {len(best_overall_individual)}, EndH: {h_func(best_overall_individual[-1], goal_state) if best_overall_individual else 'N/A'}")

        # Check for goal state
        for i, ind_path in enumerate(population):
            if ind_path and ind_path[-1] == goal_state:
                # print(f"Gen {gen}: Goal found by individual! Path length {len(ind_path)}")
                return ind_path

        parents = select_parents_8puzzle(population, fitness_values, num_parents)
        if not parents: # Should be handled by select_parents ensuring at least one/two
            # print("Warning: No parents selected. Restarting with new random population.")
            population = initialize_population_8puzzle(population_size, start_state, goal_state)
            continue

        offspring_population = []
        # Elitism: carry over some best parents directly
        num_elites = max(1, int(0.1 * num_parents)) # e.g., 10% of parents are elites
        elites = sorted(parents, key=lambda p: fitness_8puzzle(p, goal_state, h_func), reverse=True)
        offspring_population.extend(elites[:num_elites])


        while len(offspring_population) < population_size:
            p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            child1, child2 = crossover_8puzzle(p1, p2, start_state)
            offspring_population.append(child1)
            if len(offspring_population) < population_size:
                offspring_population.append(child2)
        
        population = offspring_population[:population_size]

        for i in range(len(population)):
            population[i] = mutate_8puzzle(population[i], mutation_rate, start_state, goal_state, h_func)
            # Ensure paths are not empty after mutation
            if not population[i]: population[i] = [start_state]


        # if gen % (generations // 5 if generations >=5 else 1) == 0:
        #      final_h_val = 'N/A'
        #      if best_overall_individual and best_overall_individual[-1]:
        #         final_h_val = h_func(best_overall_individual[-1], goal_state)
        #      print(f"Gen {gen} done. Overall best H at end: {final_h_val}")

    # print(f"GA finished. Best path length: {len(best_overall_individual) if best_overall_individual else 'N/A'}")
    # If goal found, it would have returned earlier. Otherwise, return best found.
    return best_overall_individual


def conformant_bfs(initial_belief_state_list, goal_list):
    """BFS on the belief state space. Returns path of move_dirs or None."""
    if not initial_belief_state_list:
        # print("Error: Initial belief state empty.")
        return None # Indicates no plan or error
        
    goal_tuple = state_to_tuple(goal_list)
    if goal_tuple is None:
        # print("Error: Invalid goal state.")
        return None

    # Convert all initial states to tuples
    initial_state_tuples = []
    for s_list in initial_belief_state_list:
        s_tuple = state_to_tuple(s_list)
        if s_tuple:
            initial_state_tuples.append(s_tuple)
    
    if not initial_state_tuples:
        # print("Error: No valid initial states in belief list.")
        return None

    initial_belief_fset = frozenset(initial_state_tuples)

    # Check if all states in the initial belief state are already the goal
    if all(st == goal_tuple for st in initial_belief_fset):
        # print("Initial belief state is already goal.")
        return [] # Empty list of moves

    # Queue stores (belief_state_frozenset, path_of_moves)
    queue = deque([(initial_belief_fset, [])])
    visited_belief_states = {initial_belief_fset} 

    MAX_BFS_NODES_CONFORMANT = 20000 # Limit to prevent excessive computation
    nodes_processed = 0

    while queue:
        nodes_processed += 1
        if nodes_processed > MAX_BFS_NODES_CONFORMANT:
            # print(f"Conformant BFS: Node limit ({MAX_BFS_NODES_CONFORMANT}) reached.")
            return None # No plan found within limits

        current_belief_fset, current_path_moves = queue.popleft()

        for move_dir in Moves: # Moves = [(-1,0), (1,0), (0,-1), (0,1)]
            next_individual_states_set = set() # Use a set to store resulting states (as tuples)

            possible_successor_belief = True
            for state_tuple in current_belief_fset:
                # Convert tuple state back to list of lists for apply_move_to_state
                state_list = [list(row) for row in state_tuple]
                
                # apply_move_to_state expects the blank to move by move_dir
                # Or, if it means the tile at (blank_pos + move_dir) moves into blank_pos
                # Let's test apply_move_to_state's behavior carefully.
                # Assuming apply_move_to_state correctly moves the blank:
                next_s_list = apply_move_to_state(state_list, move_dir)
                
                next_s_tuple = state_to_tuple(next_s_list)
                if next_s_tuple is None: # Should not happen if apply_move is robust
                    possible_successor_belief = False; break 
                next_individual_states_set.add(next_s_tuple)
            
            if not possible_successor_belief or not next_individual_states_set:
                continue # This move leads to an invalid/empty belief state

            next_belief_fset = frozenset(next_individual_states_set)

            if next_belief_fset in visited_belief_states:
                continue

            # Check if all states in the new belief state are the goal
            if all(st == goal_tuple for st in next_belief_fset):
                return current_path_moves + [move_dir] # Solution found

            visited_belief_states.add(next_belief_fset)
            queue.append((next_belief_fset, current_path_moves + [move_dir]))
            
    # print("Conformant BFS: No solution found.")
    return None # No plan found

# --- CSP Solver (Placeholder) ---
# ... (CSP_Solver_Placeholder and helpers - kept as is from user's ThuatToan.py) ...
def is_consistent_csp(assignment, var, value, constraints):
    temp_assignment = assignment.copy()
    temp_assignment[var] = value
    for constraint_func in constraints:
        if not constraint_func(temp_assignment):
            return False 
    return True

def backtracking_csp_recursive(assignment, variables, domains, constraints):
    if len(assignment) == len(variables): 
        return assignment 

    unassigned_vars = [v for v in variables if v not in assignment]
    first_unassigned = unassigned_vars[0] 

    for value in domains[first_unassigned]: 
        if is_consistent_csp(assignment, first_unassigned, value, constraints):
            assignment[first_unassigned] = value
            result = backtracking_csp_recursive(assignment, variables, domains, constraints)
            if result is not None:
                return result
            del assignment[first_unassigned] 
    return None

def CSP_Solver_Placeholder(problem_definition_dict): # Changed arg name to avoid clash
    variables = problem_definition_dict.get('variables')
    domains = problem_definition_dict.get('domains')
    constraints = problem_definition_dict.get('constraints')

    if not (variables and domains and constraints):
        # print("CSP problem definition is incomplete.")
        return None

    # print("CSP_Solver_Placeholder: Solving a generic CSP.")
    solution_assignment = backtracking_csp_recursive({}, variables, domains, constraints)

    if solution_assignment:
        # print(f"CSP found a solution: {solution_assignment}")
        # This needs to be mapped to a path if used for 8-puzzle pathfinding
        # For now, returning a list containing the goal state if a solution is found.
        return [Goal_State] # Placeholder return
    else:
        # print("CSP: No solution found for the given constraints.")
        return None


# --- Trust-Based and Trust-Partial (Placeholders) ---
def Trust_Based_Algorithm(start, goal=Goal, params=None):
    # print("Trust-Based Algorithm not implemented. Requires specification.")
    return None

def Trust_Partial_Algorithm(start, goal=Goal, params=None):
    # print("Trust-Partial Algorithm not implemented. Requires specification.")
    return None

# --- NEW ALGORITHMS ---

# --- 1. Belief Search (generic, similar to conformant_bfs for now) ---
def Belief_Search(initial_belief_state_list, goal_list, search_limit=20000):
    """
    A generic Belief State Search algorithm.
    Currently implemented like conformant_bfs.
    Returns a list of moves or None.
    """
    if not initial_belief_state_list:
        # print("Belief_Search Error: Initial belief state empty.")
        return None
        
    goal_tuple = state_to_tuple(goal_list)
    if goal_tuple is None:
        # print("Belief_Search Error: Invalid goal state.")
        return None

    initial_state_tuples = [s_tuple for s in initial_belief_state_list if (s_tuple := state_to_tuple(s))]
    if not initial_state_tuples:
        # print("Belief_Search Error: No valid initial states in belief list.")
        return None

    initial_belief_fset = frozenset(initial_state_tuples)

    if all(st == goal_tuple for st in initial_belief_fset):
        # print("Belief_Search: Initial belief state is already goal.")
        return [] 

    queue = deque([(initial_belief_fset, [])]) # (belief_fset, path_of_moves)
    visited_belief_states = {initial_belief_fset} 
    nodes_processed = 0

    while queue:
        nodes_processed += 1
        if nodes_processed > search_limit:
            # print(f"Belief_Search: Node limit ({search_limit}) reached.")
            return None 

        current_belief_fset, current_path_moves = queue.popleft()

        for move_dir in Moves: 
            next_individual_states_set = set()
            possible_successor = True
            for state_tuple in current_belief_fset:
                state_list = [list(row) for row in state_tuple]
                next_s_list = apply_move_to_state(state_list, move_dir)
                next_s_tuple = state_to_tuple(next_s_list)
                if next_s_tuple is None: 
                    possible_successor = False; break
                next_individual_states_set.add(next_s_tuple)
            
            if not possible_successor or not next_individual_states_set:
                continue

            next_belief_fset = frozenset(next_individual_states_set)

            if next_belief_fset in visited_belief_states:
                continue

            if all(st == goal_tuple for st in next_belief_fset):
                return current_path_moves + [move_dir]

            visited_belief_states.add(next_belief_fset)
            queue.append((next_belief_fset, current_path_moves + [move_dir]))
            
    # print("Belief_Search: No solution found.")
    return None


# --- 2. Q-Learning for 8-Puzzle ---
def Q_Learning_8Puzzle(start_state, goal_state, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps_per_episode=100):
    """
    Q-Learning algorithm for the 8-puzzle problem.
    Returns a path of states if a solution is found from the learned policy.
    Note: Q-Learning might require significant tuning and episodes to learn effectively.
    """
    q_table = {} # Using dict: {(state_tuple, action_tuple): q_value}
    goal_tuple = state_to_tuple(goal_state)
    start_tuple = state_to_tuple(start_state)

    if not start_tuple or not goal_tuple:
        # print("Q-Learning Error: Invalid start or goal state.")
        return None

    # Actions are the same as Moves: [(-1,0), (1,0), (0,-1), (0,1)]
    
    def get_q_value(s_tuple, a_tuple):
        return q_table.get((s_tuple, a_tuple), 0.0) # Default Q-value is 0

    def choose_action(s_tuple, current_epsilon):
        if random.random() < current_epsilon: # Explore
            return random.choice(Moves)
        else: # Exploit
            q_values = [get_q_value(s_tuple, action) for action in Moves]
            max_q = -float('inf')
            # Find max Q, breaking ties randomly
            best_actions = []
            for i, q_val in enumerate(q_values):
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [Moves[i]]
                elif q_val == max_q:
                    best_actions.append(Moves[i])
            return random.choice(best_actions) if best_actions else random.choice(Moves)


    # --- Training Phase ---
    # print(f"Q-Learning: Starting training for {episodes} episodes...")
    for episode in range(episodes):
        current_s_list = copy.deepcopy(start_state)
        current_s_tuple = state_to_tuple(current_s_list)
        
        # Adaptive epsilon (optional, decays over time)
        # current_epsilon = epsilon * (1 - episode / episodes) # Linear decay
        current_epsilon = epsilon * math.exp(-0.005 * episode) # Exponential decay

        for step in range(max_steps_per_episode):
            if current_s_tuple == goal_tuple:
                break # Reached goal in this episode

            action_tuple = choose_action(current_s_tuple, current_epsilon)
            
            # Simulate action
            next_s_list = apply_move_to_state(current_s_list, action_tuple) # apply_move_to_state returns new state
            next_s_tuple = state_to_tuple(next_s_list)

            reward = -1 # Default cost for a step
            if next_s_tuple == goal_tuple:
                reward = 100 # Large reward for reaching goal
            elif next_s_tuple == current_s_tuple : # Invalid move or no change
                reward = -10 # Penalty for invalid/ineffective moves
            
            # Q-Learning update rule:
            # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
            old_q_value = get_q_value(current_s_tuple, action_tuple)
            
            # Find max Q-value for the next state s'
            next_max_q = -float('inf')
            if next_s_tuple != goal_tuple : # If next state is not terminal
                 for next_action in Moves:
                    next_max_q = max(next_max_q, get_q_value(next_s_tuple, next_action))
            else: # If next state is terminal, future reward is 0
                next_max_q = 0.0
            
            if next_max_q == -float('inf'): next_max_q = 0.0 # If no Q-values for next state (e.g. new state)

            new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
            q_table[(current_s_tuple, action_tuple)] = new_q_value

            current_s_list = next_s_list
            current_s_tuple = next_s_tuple
        
        # if episode % (episodes // 10 if episodes >=10 else 1) == 0:
        #     print(f"Q-Learning: Episode {episode} finished. Q-table size: {len(q_table)}")

    # print("Q-Learning: Training finished.")

    # --- Path Extraction Phase (Greedy policy from Q-table) ---
    path = [start_state]
    current_s_list = copy.deepcopy(start_state)
    current_s_tuple = state_to_tuple(current_s_list)
    
    extraction_steps = 0
    max_extraction_steps = max_steps_per_episode * 2 # Safety break for path extraction

    while current_s_tuple != goal_tuple and extraction_steps < max_extraction_steps:
        extraction_steps += 1
        # Choose best action based on learned Q-values (epsilon = 0)
        q_values = [get_q_value(current_s_tuple, action) for action in Moves]
        
        max_q_val = -float('inf')
        best_action = None
        
        # Find best action, break ties (e.g. first one, or random among best)
        candidate_actions = []
        for i, q_val in enumerate(q_values):
            if q_val > max_q_val:
                max_q_val = q_val
                candidate_actions = [Moves[i]]
            elif q_val == max_q_val:
                candidate_actions.append(Moves[i])
        
        if not candidate_actions: # No Q-values known, or all are -inf (should not happen if default is 0)
            # print("Q-Learning Path Extraction: Stuck, no valid action from Q-table.")
            return path # Return current path, may not be solution
        
        best_action = random.choice(candidate_actions)

        next_s_list = apply_move_to_state(current_s_list, best_action)
        next_s_tuple = state_to_tuple(next_s_list)

        if next_s_tuple == current_s_tuple and current_s_tuple != goal_tuple:
            # Stuck in a loop or invalid move based on learned policy.
            # This can happen if the Q-table is sparse or learning was insufficient.
            # print("Q-Learning Path Extraction: Stuck, policy leads to no change or invalid move.")
            # Try a random move to escape? Or just return current path.
            # For now, just return path, may indicate failure to find full solution.
            if len(path) > 1 and next_s_list == path[-2]: # Trying to go back and forth
                 return path # Stuck in a 2-cycle
            # Try to pick a different action from candidates if available and not the one causing stuck
            remaining_candidates = [ca for ca in candidate_actions if ca != best_action]
            if remaining_candidates:
                best_action = random.choice(remaining_candidates)
                next_s_list = apply_move_to_state(current_s_list, best_action)
                next_s_tuple = state_to_tuple(next_s_list)
            else: # Only one candidate action led to being stuck
                return path


        path.append(next_s_list)
        current_s_list = next_s_list
        current_s_tuple = next_s_tuple

        if len(path) > max_steps_per_episode * 1.5 : # Path too long, likely stuck
            # print("Q-Learning Path Extraction: Path too long, assuming stuck.")
            return path # Return current path

    if path[-1] == goal_state:
        # print(f"Q-Learning: Path to goal found with {len(path)-1} steps.")
        return path
    else:
        # print("Q-Learning: Goal not reached after path extraction. Returning best path found.")
        return path # Return the path even if goal not reached (might be partial)


# --- 3. Belief1p Search (Greedy Belief State Search) ---
def heuristic_for_belief_state(belief_fset, goal_tuple, h_func=khoang_cach_mahathan):
    """
    Calculates a heuristic for a belief state.
    Example: Average heuristic of all states in the belief set.
    Or, max heuristic (pessimistic), or min heuristic (optimistic).
    """
    if not belief_fset: return float('inf')
    
    total_h = 0
    num_states = 0
    max_h = 0 # For max heuristic
    
    for state_tuple in belief_fset:
        state_list = [list(row) for row in state_tuple]
        h_val = h_func(state_list, [list(r) for r in goal_tuple]) # h_func needs list-of-lists
        if h_val == float('inf'): return float('inf') # If any state is very bad
        total_h += h_val
        if h_val > max_h: max_h = h_val
        num_states += 1
        
    # return total_h / num_states # Average heuristic
    return max_h # Pessimistic: heuristic is the max heuristic of any state in belief

def Belief1p_Search(initial_belief_state_list, goal_list, max_steps=100, h_func_belief=heuristic_for_belief_state):
    """
    Belief1p Search: A greedy search in the belief state space.
    "1p" could imply 1-ply lookahead, which is what greedy search does.
    Returns a list of moves or None.
    """
    if not initial_belief_state_list: return None
        
    goal_tuple = state_to_tuple(goal_list) # Goal is a single, fully observable state
    if goal_tuple is None: return None

    initial_state_tuples = [s_tuple for s in initial_belief_state_list if (s_tuple := state_to_tuple(s))]
    if not initial_state_tuples: return None

    current_belief_fset = frozenset(initial_state_tuples)
    
    path_of_moves = []
    visited_belief_states_in_path = {current_belief_fset} # To avoid immediate cycles in this greedy path

    for step in range(max_steps):
        if all(st == goal_tuple for st in current_belief_fset):
            # print(f"Belief1p: Goal reached in {step} moves.")
            return path_of_moves

        best_next_belief_fset = None
        best_move = None
        min_h_next_belief = float('inf')

        for move_dir in Moves:
            next_individual_states_set = set()
            possible_successor = True
            for state_tuple in current_belief_fset:
                state_list = [list(row) for row in state_tuple]
                next_s_list = apply_move_to_state(state_list, move_dir)
                next_s_tuple = state_to_tuple(next_s_list)
                if next_s_tuple is None: 
                    possible_successor = False; break
                next_individual_states_set.add(next_s_tuple)
            
            if not possible_successor or not next_individual_states_set:
                continue

            potential_next_belief_fset = frozenset(next_individual_states_set)
            
            if potential_next_belief_fset in visited_belief_states_in_path: # Avoid cycles in this path
                continue

            h_val_next_belief = h_func_belief(potential_next_belief_fset, goal_tuple)

            if h_val_next_belief < min_h_next_belief:
                min_h_next_belief = h_val_next_belief
                best_next_belief_fset = potential_next_belief_fset
                best_move = move_dir
            elif h_val_next_belief == min_h_next_belief and best_move is None : # Tie-breaking (e.g. take first one)
                best_next_belief_fset = potential_next_belief_fset
                best_move = move_dir


        if best_move is None or best_next_belief_fset is None:
            # print(f"Belief1p: Stuck at step {step}. No better or unvisited move.")
            return path_of_moves # Return current path, may not be solution (local optimum)
        
        current_belief_fset = best_next_belief_fset
        path_of_moves.append(best_move)
        visited_belief_states_in_path.add(current_belief_fset)

    # print(f"Belief1p: Max steps ({max_steps}) reached. Goal not found or path incomplete.")
    # Check if final state is goal, though it should have been caught in loop
    if all(st == goal_tuple for st in current_belief_fset):
         return path_of_moves
    return path_of_moves # Return whatever path was constructed


# --- Example Usage Block ---
if __name__ == "__main__":
    Start_Test = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
    Goal_Test = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    Start_Test_Harder = [[8,6,7],[2,5,4],[3,0,1]] # Solvable, needs many moves
    Goal_Test_Harder = [[1,2,3],[4,5,6],[7,8,0]]


    print("--- Running ThuatToan.py directly for testing ---")
    # ... (existing test cases) ...

    print("\n--- Testing Belief_Search ---")
    belief_states_test = [
        [[1, 0, 3], [4, 2, 5], [7, 8, 6]], # One move from goal for one of them
        [[1, 2, 3], [0, 4, 6], [7, 5, 8]]  # Standard start
    ]
    belief_search_plan = Belief_Search(belief_states_test, Goal_Test)
    if belief_search_plan is not None:
        print(f"Belief_Search found a plan with {len(belief_search_plan)} moves: {belief_search_plan}")
    else:
        print("Belief_Search: No plan found.")

    print("\n--- Testing Q-Learning_8Puzzle ---")
    # Q-Learning can be slow and results vary. Use fewer episodes for quick test.
    # For real use, episodes should be much higher (e.g., 10000+).
    q_learning_path = Q_Learning_8Puzzle(Start_Test, Goal_Test, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2, max_steps_per_episode=50)
    if q_learning_path:
        print(f"Q-Learning found a path of length {len(q_learning_path)-1}.")
        if q_learning_path[-1] == Goal_Test:
            print("Q-Learning reached the goal.")
            # print_matrix(q_learning_path[-1])
        else:
            print("Q-Learning did not reach the goal. Final state:")
            # print_matrix(q_learning_path[-1])
    else:
        print("Q-Learning: No path returned (possibly error or no solution found).")
    
    # Test with a state known to be one step from goal for Q-learning
    one_step_start = [[1,2,3],[4,5,0],[7,8,6]] # Blank needs to move right
    q_learning_path_simple = Q_Learning_8Puzzle(one_step_start, Goal_Test, episodes=100, alpha=0.2, gamma=0.9, epsilon=0.1, max_steps_per_episode=10)
    if q_learning_path_simple and q_learning_path_simple[-1] == Goal_Test:
        print(f"Q-Learning (simple case) reached goal. Path length {len(q_learning_path_simple)-1}")
    else:
        print(f"Q-Learning (simple case) did not reach goal reliably. Path len: {len(q_learning_path_simple) if q_learning_path_simple else 'N/A'}")


    print("\n--- Testing Belief1p_Search ---")
    belief1p_plan = Belief1p_Search(belief_states_test, Goal_Test, max_steps=20)
    if belief1p_plan is not None: # It will always return a list of moves, check if it's non-empty if stuck at start
        print(f"Belief1p_Search generated a plan with {len(belief1p_plan)} moves: {belief1p_plan}")
        # To verify, one would need to apply these moves to the initial belief state
    else: # Should not happen unless initial belief state is empty/invalid
        print("Belief1p_Search: Error or no plan generated.")

    # Test Belief1p on a single state belief set, harder start
    single_belief_hard = [Start_Test_Harder]
    belief1p_plan_hard = Belief1p_Search(single_belief_hard, Goal_Test_Harder, max_steps=50)
    print(f"Belief1p_Search (hard single) generated plan ({len(belief1p_plan_hard)} moves): {belief1p_plan_hard}")