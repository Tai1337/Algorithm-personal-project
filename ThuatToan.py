from collections import deque
import heapq
from queue import PriorityQueue
import copy
import random
import math
import sys

Start_State = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
Goal_State = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
Goal = Goal_State
Moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
Move_Names = {(-1, 0): "Up", (1, 0): "Down", (0, -1): "Left", (0, 1): "Right"}

def tim_X(x, goal_state=Goal):
    for i in range(3):
        for j in range(3):
            if goal_state[i][j] == x:
                return i, j
    return None

def khoang_cach_mahathan(matran_hientai, goal_state=Goal):
    h_sum = 0
    if matran_hientai is None:
        return float('inf')
    for i in range(3):
        for j in range(3):
            val = matran_hientai[i][j]
            if val != 0:
                try:
                    pos_x, pos_y = tim_X(val, goal_state)
                    if pos_x is None:
                        return float('inf')
                    h_sum += abs(i - pos_x) + abs(j - pos_y)
                except TypeError:
                    return float('inf')
    return h_sum

def Tim_0(matran_hientai):
    if matran_hientai is None: return None
    for i in range(3):
        for j in range(3):
            if matran_hientai[i][j] == 0:
                return i, j
    return None

def Check(x, y):
    return 0 <= x < 3 and 0 <= y < 3

def Chiphi(matran_hientai, goal_state=Goal):
    if matran_hientai is None: return float('inf')
    dem = 0
    for i in range(3):
        for j in range(3):
            if matran_hientai[i][j] != 0 and matran_hientai[i][j] != goal_state[i][j]:
                dem += 1
    return dem

def DiChuyen(matran_hientai, x, y, new_x, new_y):
    new_state = copy.deepcopy(matran_hientai)
    new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
    return new_state

def print_matrix(matran):
    if matran is None:
        print("None")
        return
    for row in matran:
        print(" ".join(map(str, row)))
    print()

def state_to_tuple(matran):
    if not isinstance(matran, list):
        return None
    try:
        return tuple(tuple(row) for row in matran)
    except TypeError:
        return None

def get_neighbors(matran_hientai):
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
    new_state_list = copy.deepcopy(state_list)
    blank_pos = Tim_0(new_state_list)
    if not blank_pos:
        return state_list

    blank_r, blank_c = blank_pos
    move_r, move_c = move_dir

    new_blank_r, new_blank_c = blank_r + move_r, blank_c + move_c

    if Check(new_blank_r, new_blank_c):
        new_state_list[blank_r][blank_c], new_state_list[new_blank_r][new_blank_c] = \
            new_state_list[new_blank_r][new_blank_c], new_state_list[blank_r][blank_c]
        return new_state_list
    else:
        return state_list

def belief_state_to_tuple_for_hashing(belief_state):
    if not belief_state:
        return frozenset()
    try:
        valid_state_tuples = []
        for s in belief_state:
            if isinstance(s, list):
                s_tuple = state_to_tuple(s)
                if s_tuple: valid_state_tuples.append(s_tuple)
            elif isinstance(s, tuple):
                valid_state_tuples.append(s)
        return frozenset(valid_state_tuples)

    except Exception as e:
        return None

def BFS(start, goal=Goal):
    queue = deque([(start, [])])
    visited = {state_to_tuple(start)}
    while queue:
        matran_hientai, path = queue.popleft()
        if matran_hientai == goal:
            return path + [matran_hientai]
        for new_matran in get_neighbors(matran_hientai):
            new_matran_tuple = state_to_tuple(new_matran)
            if new_matran_tuple not in visited:
                visited.add(new_matran_tuple)
                new_path = path + [matran_hientai]
                queue.append((new_matran, new_path))
    return None

def UCS(start, goal=Goal):
    qp = PriorityQueue()
    start_tuple = state_to_tuple(start)
    qp.put((0, start, []))
    visited = {start_tuple: 0}

    while not qp.empty():
        cost, matran_hientai, path = qp.get()

        if matran_hientai == goal:
            return path + [matran_hientai]

        current_tuple = state_to_tuple(matran_hientai)
        if cost > visited.get(current_tuple, float('inf')):
            continue

        for new_matran in get_neighbors(matran_hientai):
            new_matran_tuple = state_to_tuple(new_matran)
            new_cost = cost + 1
            if new_cost < visited.get(new_matran_tuple, float('inf')):
                visited[new_matran_tuple] = new_cost
                new_path = path + [matran_hientai]
                qp.put((new_cost, new_matran, new_path))
    return None

def DFS(start, goal=Goal):
    stack = [(start, [])]
    visited = {state_to_tuple(start)}

    while stack:
        matran_hientai, path = stack.pop()

        if matran_hientai == goal:
            return path + [matran_hientai]

        neighbors = get_neighbors(matran_hientai)
        for new_matran in reversed(neighbors):
            new_matran_tuple = state_to_tuple(new_matran)
            if new_matran_tuple not in visited:
                visited.add(new_matran_tuple)
                new_path = path + [matran_hientai]
                stack.append((new_matran, new_path))
    return None

def DFS_limited(current_matran, goal, limit, path_so_far, visited_in_current_path_tuples):
    if current_matran == goal:
        return path_so_far + [current_matran]
    if limit <= 0:
        return "cutoff"

    current_matran_tuple = state_to_tuple(current_matran)

    any_remaining_path = False
    for new_matran in get_neighbors(current_matran):
        new_matran_tuple = state_to_tuple(new_matran)
        if new_matran_tuple not in visited_in_current_path_tuples:
            new_visited_tuples = visited_in_current_path_tuples.copy()
            new_visited_tuples.add(current_matran_tuple)

            result = DFS_limited(new_matran, goal, limit - 1, path_so_far + [current_matran], new_visited_tuples)
            if result == "cutoff":
                any_remaining_path = True
            elif result is not None:
                return result
    
    return "cutoff" if any_remaining_path else None


def IDDFS(start, goal=Goal, max_depth=30):
    for depth in range(max_depth + 1):
        result = DFS_limited(start, goal, depth, [], set())
        if result is None:
            return None
        if result != "cutoff":
            return result
    return None


def Greedy(start, goal=Goal, h_func=khoang_cach_mahathan):
    qp = PriorityQueue()
    start_tuple = state_to_tuple(start)
    qp.put((h_func(start, goal), start, []))
    visited = {start_tuple}

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
    qp = PriorityQueue()
    start_tuple = state_to_tuple(start)
    
    g_costs = {start_tuple: 0}
    initial_h = h_func(start, goal)
    if initial_h == float('inf'): return None

    qp.put((g_costs[start_tuple] + initial_h, g_costs[start_tuple], start, []))

    while not qp.empty():
        f_n, cost_so_far, matran_hientai, path = qp.get()

        if matran_hientai == goal:
            return path + [matran_hientai]

        current_tuple = state_to_tuple(matran_hientai)
        if cost_so_far > g_costs.get(current_tuple, float('inf')):
            continue
        
        for new_matran in get_neighbors(matran_hientai):
            new_g_cost = cost_so_far + 1
            new_matran_tuple = state_to_tuple(new_matran)

            if new_g_cost < g_costs.get(new_matran_tuple, float('inf')):
                g_costs[new_matran_tuple] = new_g_cost
                h_val = h_func(new_matran, goal)
                if h_val == float('inf') : continue

                new_f_cost = new_g_cost + h_val
                new_path = path + [matran_hientai]
                qp.put((new_f_cost, new_g_cost, new_matran, new_path))
    return None

def ida_star_search_recursive(current_path, g_cost, bound, goal, h_func):
    current_node = current_path[-1]
    h_cost = h_func(current_node, goal)
    f_cost = g_cost + h_cost

    if f_cost > bound:
        return f_cost
    if current_node == goal:
        return "FOUND"
    
    min_exceeded_cost = float('inf')

    for neighbor in get_neighbors(current_node):
        if len(current_path) > 1 and neighbor == current_path[-2]:
            continue

        current_path.append(neighbor)
        result = ida_star_search_recursive(current_path, g_cost + 1, bound, goal, h_func)
        
        if result == "FOUND":
            return "FOUND"
        if result < min_exceeded_cost:
            min_exceeded_cost = result
        
        current_path.pop()

    return min_exceeded_cost


def IDA_Star(start, goal=Goal, h_func=khoang_cach_mahathan):
    bound = h_func(start, goal)
    if bound == float('inf'): return None

    path = [start]

    while True:
        result = ida_star_search_recursive(path, 0, bound, goal, h_func)
        
        if result == "FOUND":
            return path
        if result == float('inf'):
            return None
        
        bound = result
        path = [start]


def Simple_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_iterations=1000):
    current = start
    path = [current]
    iterations = 0
    while iterations < max_iterations :
        iterations +=1
        current_h = h_func(current, goal)
        if current_h == 0 and current == goal:
            return path
        
        neighbors = get_neighbors(current)
        if not neighbors: return path

        best_neighbor_found = False
        for neighbor in neighbors:
            if h_func(neighbor, goal) < current_h:
                current = neighbor
                path.append(current)
                best_neighbor_found = True
                break

        if not best_neighbor_found:
            return path
    return path


def Steepest_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_iterations=1000):
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
        
        if best_neighbor is None or best_h >= current_h:
            return path
        
        current = best_neighbor
        path.append(current)
    return path


def Stochastic_HC(start, goal=Goal, h_func=khoang_cach_mahathan, max_steps=1000):
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
        
        if not better_neighbors:
            return path
            
        chosen_neighbor = random.choice(better_neighbors)
        current = chosen_neighbor
        path.append(current)
        steps += 1
    return path


def Beam_Search(start, goal=Goal, h_func=khoang_cach_mahathan, beam_width_k=3, max_iterations=100):
    beam = [(h_func(start, goal), start, [start])]
    
    for iteration in range(max_iterations):
        if not beam: return None

        for h_val, state, pth in beam:
            if state == goal:
                return pth

        successors = []
        
        for h_cost, current_state, path_taken in beam:
            if current_state == goal:
                return path_taken

            for neighbor in get_neighbors(current_state):
                is_cycle = False
                for node_in_path in path_taken:
                    if neighbor == node_in_path:
                        is_cycle = True
                        break
                if not is_cycle:
                    new_h = h_func(neighbor, goal)
                    new_path = path_taken + [neighbor]
                    successors.append((new_h, neighbor, new_path))
        
        if not successors:
            beam.sort(key=lambda item: item[0])
            return beam[0][2] if beam else None

        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width_k]

    if beam:
        beam.sort(key=lambda item: item[0])
        return beam[0][2]
    return None


def Simulated_Annealing(start, goal=Goal, h_func=khoang_cach_mahathan,
                        initial_temp=100.0, cooling_rate=0.95, min_temp=0.1,
                        max_iterations_per_temp=100):
    current_state = start
    current_h = h_func(current_state, goal)
    
    path_taken = [current_state]
    best_state_so_far = current_state
    best_h_so_far = current_h
    path_to_best_so_far = [current_state]

    temp = initial_temp
    
    iteration = 0
    max_total_iterations = 50000
    
    while temp > min_temp and iteration < max_total_iterations:
        if current_state == goal:
            return path_taken

        for _ in range(max_iterations_per_temp):
            iteration += 1
            if iteration >= max_total_iterations: break

            neighbors = get_neighbors(current_state)
            if not neighbors:
                break

            next_state = random.choice(neighbors)
            next_h = h_func(next_state, goal)
            
            delta_e = next_h - current_h

            accept_move = False
            if delta_e < 0:
                accept_move = True
            else:
                if temp > 1e-9:
                    probability = math.exp(-delta_e / temp)
                    accept_move = random.random() < probability
            
            if accept_move:
                current_state = next_state
                current_h = next_h
                path_taken.append(current_state)

                if current_h < best_h_so_far:
                    best_state_so_far = current_state
                    best_h_so_far = current_h
                    path_to_best_so_far = list(path_taken)
                
                if current_state == goal:
                    return path_taken
        
        temp *= cooling_rate
        if not neighbors: break

    if best_state_so_far == goal : return path_to_best_so_far
    return path_taken

def and_or_search_8puzzle_recursive(current_state, goal_state, path_visited_tuples, path_so_far_states):
    current_tuple = state_to_tuple(current_state)
    if current_tuple is None: return None

    if current_state == goal_state:
        return path_so_far_states + [current_state]

    if current_tuple in path_visited_tuples:
        return None

    path_visited_tuples.add(current_tuple)
    
    for neighbor in get_neighbors(current_state):
        
        result_path = and_or_search_8puzzle_recursive(
            neighbor, goal_state, path_visited_tuples.copy(), path_so_far_states + [current_state]
        )

        if result_path is not None:
            return result_path
            
    return None

def solve_with_and_or_8puzzle(start_state, goal_state=Goal):
    result_path = and_or_search_8puzzle_recursive(start_state, goal_state, set(), [])
    return result_path

def Backtracking_Search(start, goal=Goal):
    stack = [(start, [])]
    visited_tuples = {state_to_tuple(start)}

    while stack:
        current_state, path = stack.pop()

        if current_state == goal:
            return path + [current_state]

        for neighbor in reversed(get_neighbors(current_state)):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited_tuples:
                visited_tuples.add(neighbor_tuple)
                stack.append((neighbor, path + [current_state]))
    
    return None

def initialize_population_8puzzle(size, start_state, goal_state, max_random_moves=20):
    population = []
    for _ in range(size):
        current = copy.deepcopy(start_state)
        path = [current]
        for _ in range(random.randint(5, max_random_moves)):
            neighbors = get_neighbors(current)
            if not neighbors: break
            current = random.choice(neighbors)
            if len(path) > 1 and current == path[-2]:
                if len(neighbors) > 1:
                    current = random.choice([n for n in neighbors if n != path[-2]] or neighbors)
            path.append(current)
        population.append(path)
    return population

def fitness_8puzzle(individual_path, goal_state, h_func=khoang_cach_mahathan):
    if not individual_path: return -float('inf')
    final_state = individual_path[-1]
    
    heuristic_val = h_func(final_state, goal_state)
    if heuristic_val == float('inf') : return -float('inf')

    fitness = 1.0 / (1.0 + heuristic_val + 0.05 * len(individual_path))
    
    if final_state == goal_state:
        fitness *= 2
    return fitness

def select_parents_8puzzle(population, fitness_values, num_parents):
    sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    parents = [ind for ind, fit in sorted_population[:num_parents]]
    return parents if parents else [random.choice(population)]

def crossover_8puzzle(parent1_path, parent2_path, start_state):
    p1_len = len(parent1_path)
    p2_len = len(parent2_path)
    if p1_len < 2 or p2_len < 2: return list(parent1_path), list(parent2_path)

    point1 = random.randint(1, p1_len -1)
    point2 = random.randint(1, p2_len -1)

    child1_path = parent1_path[:point1] + parent2_path[point2:]
    child2_path = parent2_path[:point2] + parent1_path[point1:]
    
    return child1_path, child2_path


def mutate_8puzzle(individual_path, mutation_rate, start_state, goal_state, h_func, max_perturb_moves=3):
    if not individual_path or random.random() > mutation_rate:
        return individual_path

    mutated_path = list(individual_path)
    if not mutated_path: return mutated_path

    if len(mutated_path) == 0 : return []

    mutation_idx = random.randint(0, len(mutated_path) - 1)
    current_state_at_mutation = mutated_path[mutation_idx]
    
    new_segment = []
    temp_state = copy.deepcopy(current_state_at_mutation)

    for _ in range(random.randint(1, max_perturb_moves)):
        neighbors = get_neighbors(temp_state)
        if not neighbors: break
        
        best_neighbor = None
        min_h = h_func(temp_state, goal_state)

        next_s = random.choice(neighbors)

        if new_segment and next_s == new_segment[-1]:
            if len(neighbors)>1:
                next_s = random.choice([n for n in neighbors if n != new_segment[-1]] or neighbors)
        
        temp_state = next_s
        new_segment.append(temp_state)
        if temp_state == goal_state: break

    final_mutated_path = mutated_path[:mutation_idx] + new_segment
    if not final_mutated_path :
        return [start_state]
    return final_mutated_path


def Genetic_Algorithm_8Puzzle(start_state, goal_state=Goal,
                             population_size=50, generations=30,
                             mutation_rate=0.2, num_parents_mating_ratio=0.4,
                             h_func=khoang_cach_mahathan):
    population = initialize_population_8puzzle(population_size, start_state, goal_state)
    best_overall_individual = None
    best_overall_fitness = -float('inf')

    num_parents = int(population_size * num_parents_mating_ratio)
    if num_parents < 2 : num_parents = 2
    if num_parents % 2 != 0 : num_parents +=1

    for gen in range(generations):
        fitness_values = [fitness_8puzzle(ind, goal_state, h_func) for ind in population]

        current_gen_best_idx = -1
        if any(fitness_values):
            current_gen_best_fitness = -float('inf')
            for i, f_val in enumerate(fitness_values):
                if f_val > current_gen_best_fitness:
                    current_gen_best_fitness = f_val
                    current_gen_best_idx = i
            
            if current_gen_best_idx != -1 and current_gen_best_fitness > best_overall_fitness:
                best_overall_fitness = current_gen_best_fitness
                best_overall_individual = copy.deepcopy(population[current_gen_best_idx])

        for i, ind_path in enumerate(population):
            if ind_path and ind_path[-1] == goal_state:
                return ind_path

        parents = select_parents_8puzzle(population, fitness_values, num_parents)
        if not parents:
            population = initialize_population_8puzzle(population_size, start_state, goal_state)
            continue

        offspring_population = []
        num_elites = max(1, int(0.1 * num_parents))
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
            if not population[i]: population[i] = [start_state]

    return best_overall_individual


def conformant_bfs(initial_belief_state_list, goal_list):
    if not initial_belief_state_list:
        return None
        
    goal_tuple = state_to_tuple(goal_list)
    if goal_tuple is None:
        return None

    initial_state_tuples = []
    for s_list in initial_belief_state_list:
        s_tuple = state_to_tuple(s_list)
        if s_tuple:
            initial_state_tuples.append(s_tuple)
    
    if not initial_state_tuples:
        return None

    initial_belief_fset = frozenset(initial_state_tuples)

    if all(st == goal_tuple for st in initial_belief_fset):
        return []

    queue = deque([(initial_belief_fset, [])])
    visited_belief_states = {initial_belief_fset}

    MAX_BFS_NODES_CONFORMANT = 20000
    nodes_processed = 0

    while queue:
        nodes_processed += 1
        if nodes_processed > MAX_BFS_NODES_CONFORMANT:
            return None

        current_belief_fset, current_path_moves = queue.popleft()

        for move_dir in Moves:
            next_individual_states_set = set()

            possible_successor_belief = True
            for state_tuple in current_belief_fset:
                state_list = [list(row) for row in state_tuple]
                
                next_s_list = apply_move_to_state(state_list, move_dir)
                
                next_s_tuple = state_to_tuple(next_s_list)
                if next_s_tuple is None:
                    possible_successor_belief = False; break
                next_individual_states_set.add(next_s_tuple)
            
            if not possible_successor_belief or not next_individual_states_set:
                continue

            next_belief_fset = frozenset(next_individual_states_set)

            if next_belief_fset in visited_belief_states:
                continue

            if all(st == goal_tuple for st in next_belief_fset):
                return current_path_moves + [move_dir]

            visited_belief_states.add(next_belief_fset)
            queue.append((next_belief_fset, current_path_moves + [move_dir]))
            
    return None

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

def CSP_Solver_Placeholder(problem_definition_dict):
    variables = problem_definition_dict.get('variables')
    domains = problem_definition_dict.get('domains')
    constraints = problem_definition_dict.get('constraints')

    if not (variables and domains and constraints):
        return None

    solution_assignment = backtracking_csp_recursive({}, variables, domains, constraints)

    if solution_assignment:
        return [Goal_State]
    else:
        return None


def Trust_Based_Algorithm(start, goal=Goal, params=None):
    return None

def Trust_Partial_Algorithm(start, goal=Goal, params=None):
    return None


def Belief_Search(initial_belief_state_list, goal_list, search_limit=20000):
    if not initial_belief_state_list:
        return None
        
    goal_tuple = state_to_tuple(goal_list)
    if goal_tuple is None:
        return None

    initial_state_tuples = [s_tuple for s in initial_belief_state_list if (s_tuple := state_to_tuple(s))]
    if not initial_state_tuples:
        return None

    initial_belief_fset = frozenset(initial_state_tuples)

    if all(st == goal_tuple for st in initial_belief_fset):
        return []

    queue = deque([(initial_belief_fset, [])])
    visited_belief_states = {initial_belief_fset}
    nodes_processed = 0

    while queue:
        nodes_processed += 1
        if nodes_processed > search_limit:
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
            
    return None


def Q_Learning_8Puzzle(start_state, goal_state, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps_per_episode=100):
    q_table = {}
    goal_tuple = state_to_tuple(goal_state)
    start_tuple = state_to_tuple(start_state)

    if not start_tuple or not goal_tuple:
        return None

    def get_q_value(s_tuple, a_tuple):
        return q_table.get((s_tuple, a_tuple), 0.0)

    def choose_action(s_tuple, current_epsilon):
        if random.random() < current_epsilon:
            return random.choice(Moves)
        else:
            q_values = [get_q_value(s_tuple, action) for action in Moves]
            max_q = -float('inf')
            best_actions = []
            for i, q_val in enumerate(q_values):
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [Moves[i]]
                elif q_val == max_q:
                    best_actions.append(Moves[i])
            return random.choice(best_actions) if best_actions else random.choice(Moves)

    for episode in range(episodes):
        current_s_list = copy.deepcopy(start_state)
        current_s_tuple = state_to_tuple(current_s_list)
        
        current_epsilon = epsilon * math.exp(-0.005 * episode)

        for step in range(max_steps_per_episode):
            if current_s_tuple == goal_tuple:
                break

            action_tuple = choose_action(current_s_tuple, current_epsilon)
            
            next_s_list = apply_move_to_state(current_s_list, action_tuple)
            next_s_tuple = state_to_tuple(next_s_list)

            reward = -1
            if next_s_tuple == goal_tuple:
                reward = 100
            elif next_s_tuple == current_s_tuple :
                reward = -10
            
            old_q_value = get_q_value(current_s_tuple, action_tuple)
            
            next_max_q = -float('inf')
            if next_s_tuple != goal_tuple :
                 for next_action in Moves:
                    next_max_q = max(next_max_q, get_q_value(next_s_tuple, next_action))
            else:
                next_max_q = 0.0
            
            if next_max_q == -float('inf'): next_max_q = 0.0

            new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
            q_table[(current_s_tuple, action_tuple)] = new_q_value

            current_s_list = next_s_list
            current_s_tuple = next_s_tuple
        
    path = [start_state]
    current_s_list = copy.deepcopy(start_state)
    current_s_tuple = state_to_tuple(current_s_list)
    
    extraction_steps = 0
    max_extraction_steps = max_steps_per_episode * 2

    while current_s_tuple != goal_tuple and extraction_steps < max_extraction_steps:
        extraction_steps += 1
        q_values = [get_q_value(current_s_tuple, action) for action in Moves]
        
        max_q_val = -float('inf')
        best_action = None
        
        candidate_actions = []
        for i, q_val in enumerate(q_values):
            if q_val > max_q_val:
                max_q_val = q_val
                candidate_actions = [Moves[i]]
            elif q_val == max_q_val:
                candidate_actions.append(Moves[i])
        
        if not candidate_actions:
            return path
        
        best_action = random.choice(candidate_actions)

        next_s_list = apply_move_to_state(current_s_list, best_action)
        next_s_tuple = state_to_tuple(next_s_list)

        if next_s_tuple == current_s_tuple and current_s_tuple != goal_tuple:
            if len(path) > 1 and next_s_list == path[-2]:
                 return path
            remaining_candidates = [ca for ca in candidate_actions if ca != best_action]
            if remaining_candidates:
                best_action = random.choice(remaining_candidates)
                next_s_list = apply_move_to_state(current_s_list, best_action)
                next_s_tuple = state_to_tuple(next_s_list)
            else:
                return path


        path.append(next_s_list)
        current_s_list = next_s_list
        current_s_tuple = next_s_tuple

        if len(path) > max_steps_per_episode * 1.5 :
            return path

    if path[-1] == goal_state:
        return path
    else:
        return path


def heuristic_for_belief_state(belief_fset, goal_tuple, h_func=khoang_cach_mahathan):
    if not belief_fset: return float('inf')
    
    total_h = 0
    num_states = 0
    max_h = 0
    
    for state_tuple in belief_fset:
        state_list = [list(row) for row in state_tuple]
        h_val = h_func(state_list, [list(r) for r in goal_tuple])
        if h_val == float('inf'): return float('inf')
        total_h += h_val
        if h_val > max_h: max_h = h_val
        num_states += 1
        
    return max_h

def Belief1p_Search(initial_belief_state_list, goal_list, max_steps=100, h_func_belief=heuristic_for_belief_state):
    if not initial_belief_state_list: return None
        
    goal_tuple = state_to_tuple(goal_list)
    if goal_tuple is None: return None

    initial_state_tuples = [s_tuple for s in initial_belief_state_list if (s_tuple := state_to_tuple(s))]
    if not initial_state_tuples: return None

    current_belief_fset = frozenset(initial_state_tuples)
    
    path_of_moves = []
    visited_belief_states_in_path = {current_belief_fset}

    for step in range(max_steps):
        if all(st == goal_tuple for st in current_belief_fset):
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
            
            if potential_next_belief_fset in visited_belief_states_in_path:
                continue

            h_val_next_belief = h_func_belief(potential_next_belief_fset, goal_tuple)

            if h_val_next_belief < min_h_next_belief:
                min_h_next_belief = h_val_next_belief
                best_next_belief_fset = potential_next_belief_fset
                best_move = move_dir
            elif h_val_next_belief == min_h_next_belief and best_move is None :
                best_next_belief_fset = potential_next_belief_fset
                best_move = move_dir


        if best_move is None or best_next_belief_fset is None:
            
            return path_of_moves
        
        current_belief_fset = best_next_belief_fset
        path_of_moves.append(best_move)
        visited_belief_states_in_path.add(current_belief_fset)
    if all(st == goal_tuple for st in current_belief_fset):
         return path_of_moves
    return path_of_moves


if __name__ == "__main__":
    Start_Test = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
    Goal_Test = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    Start_Test_Harder = [[8,6,7],[2,5,4],[3,0,1]]
    Goal_Test_Harder = [[1,2,3],[4,5,6],[7,8,0]]


    print("--- Running ThuatToan.py directly for testing ---")
    

    print("\n--- Testing Belief_Search ---")
    belief_states_test = [
        [[1, 0, 3], [4, 2, 5], [7, 8, 6]],
        [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
    ]
    belief_search_plan = Belief_Search(belief_states_test, Goal_Test)
    if belief_search_plan is not None:
        print(f"Belief_Search found a plan with {len(belief_search_plan)} moves: {belief_search_plan}")
    else:
        print("Belief_Search: No plan found.")

    print("\n--- Testing Q-Learning_8Puzzle ---")
    q_learning_path = Q_Learning_8Puzzle(Start_Test, Goal_Test, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2, max_steps_per_episode=50)
    if q_learning_path:
        print(f"Q-Learning found a path of length {len(q_learning_path)-1}.")
        if q_learning_path[-1] == Goal_Test:
            print("Q-Learning reached the goal.")
            
        else:
            print("Q-Learning did not reach the goal. Final state:")
            
    else:
        print("Q-Learning: No path returned (possibly error or no solution found).")
    
    one_step_start = [[1,2,3],[4,5,0],[7,8,6]]
    q_learning_path_simple = Q_Learning_8Puzzle(one_step_start, Goal_Test, episodes=100, alpha=0.2, gamma=0.9, epsilon=0.1, max_steps_per_episode=10)
    if q_learning_path_simple and q_learning_path_simple[-1] == Goal_Test:
        print(f"Q-Learning (simple case) reached goal. Path length {len(q_learning_path_simple)-1}")
    else:
        print(f"Q-Learning (simple case) did not reach goal reliably. Path len: {len(q_learning_path_simple) if q_learning_path_simple else 'N/A'}")


    print("\n--- Testing Belief1p_Search ---")
    belief1p_plan = Belief1p_Search(belief_states_test, Goal_Test, max_steps=20)
    if belief1p_plan is not None:
        print(f"Belief1p_Search generated a plan with {len(belief1p_plan)} moves: {belief1p_plan}")
        
    else:
        print("Belief1p_Search: Error or no plan generated.")

    single_belief_hard = [Start_Test_Harder]
    belief1p_plan_hard = Belief1p_Search(single_belief_hard, Goal_Test_Harder, max_steps=50)
    print(f"Belief1p_Search (hard single) generated plan ({len(belief1p_plan_hard)} moves): {belief1p_plan_hard}")