import pygame
import numpy as np
import random
from collections import defaultdict
import os
import csv

# Pygame Settings
CELL_SIZE = 80

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

DEFAULT_MAZE = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

script_dir = os.path.dirname(os.path.abspath(__file__))

class MazeGame:
    def __init__(self, maze_layout, cheese_positions, start_pos, target_cheese_count=2, q_agent_params=None, q_table_file="q_table.csv"):
        pygame.init()
        self.maze = np.array(maze_layout)
        self.grid_height, self.grid_width = self.maze.shape

        self.screen_width = self.grid_width * CELL_SIZE
        self.screen_height = self.grid_height * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Q-Learning Mouse: Eat {target_cheese_count} Cheeses (5x5)")
        self.clock = pygame.time.Clock()

        self.initial_cheese_positions = frozenset(cheese_positions)
        self.start_pos = tuple(start_pos)
        self.target_cheese_count = target_cheese_count
        self.q_table_file = q_table_file  # Đường dẫn file CSV để lưu/tải Q-table

        self.font = pygame.font.Font(None, 30)

        mouse_image_path = os.path.join(script_dir, "assets", "mouse.png")
        cheese_image_path = os.path.join(script_dir, "assets", "cheese.png")

        try:
            original_mouse_image = pygame.image.load(mouse_image_path).convert_alpha()
            self.mouse_image = pygame.transform.scale(original_mouse_image, (int(CELL_SIZE * 0.7), int(CELL_SIZE * 0.7)))
        except pygame.error:
            self.mouse_image = None
            print(f"Cannot load mouse.png, using blue rectangle fallback.")

        try:
            original_cheese_image = pygame.image.load(cheese_image_path).convert_alpha()
            self.cheese_image = pygame.transform.scale(original_cheese_image, (int(CELL_SIZE * 0.5), int(CELL_SIZE * 0.5)))
        except pygame.error:
            self.cheese_image = None
            print(f"Cannot load cheese.png, using yellow ellipse fallback.")
        
        if q_agent_params is None:
            q_agent_params = {}

        self.q_learning_agent = QLearningAgent(
            actions=[0, 1, 2, 3],
            maze_shape=self.maze.shape,
            q_table_file=self.q_table_file,  # Truyền đường dẫn file Q-table
            **q_agent_params
        )
        # Tải Q-table nếu file tồn tại
        self.q_learning_agent.load_q_table()
        self.reset_game_state()

    def reset_game_state(self):
        self.mouse_pos = list(self.start_pos)
        self.remaining_cheese = set(self.initial_cheese_positions)
        self.score = 0
        self.steps = 0
        self.current_path = [tuple(self.mouse_pos)]
        return self.get_state()

    def get_state(self):
        return (self.mouse_pos[0], self.mouse_pos[1], frozenset(self.remaining_cheese))

    def perform_action(self, action):
        potential_pos = list(self.mouse_pos)

        if action == 0: potential_pos[0] -= 1
        elif action == 1: potential_pos[0] += 1
        elif action == 2: potential_pos[1] -= 1
        elif action == 3: potential_pos[1] += 1

        current_reward = -0.1 
        hit_wall = False

        if not (0 <= potential_pos[0] < self.grid_height and \
                0 <= potential_pos[1] < self.grid_width) or \
           self.maze[potential_pos[0], potential_pos[1]] == 1:
            current_reward = -2.0 
            hit_wall = True
        else:
            self.mouse_pos = potential_pos
            self.current_path.append(tuple(self.mouse_pos))

        current_pos_tuple = tuple(self.mouse_pos)
        if not hit_wall and current_pos_tuple in self.remaining_cheese:
            self.remaining_cheese.remove(current_pos_tuple)
            self.score += 1
            current_reward += 20

        done = False
        if self.score >= self.target_cheese_count:
            current_reward += 60 
            done = True
        elif not self.remaining_cheese and self.score < self.target_cheese_count:
            done = True

        num_initial_cheese_for_steps = len(self.initial_cheese_positions) if self.initial_cheese_positions else 0
        max_allowed_steps = self.grid_width * self.grid_height * (num_initial_cheese_for_steps + 1.2) 
        if max_allowed_steps < 15: max_allowed_steps = 15

        if not done and self.steps > max_allowed_steps:
            done = True

        self.steps += 1
        return self.get_state(), current_reward, done

    def draw_grid(self):
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.maze[r,c] == 1: pygame.draw.rect(self.screen, BLACK, rect)
                else: pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

    def draw_elements(self):
        for r_idx, c_idx in self.remaining_cheese:
            if self.cheese_image:
                img_w, img_h = self.cheese_image.get_size()
                x_pos = c_idx*CELL_SIZE+(CELL_SIZE-img_w)//2; y_pos = r_idx*CELL_SIZE+(CELL_SIZE-img_h)//2
                self.screen.blit(self.cheese_image, (x_pos, y_pos))
            else: 
                pygame.draw.ellipse(self.screen, YELLOW, (c_idx*CELL_SIZE+CELL_SIZE//3, r_idx*CELL_SIZE+CELL_SIZE//3, CELL_SIZE//3, CELL_SIZE//3))
        if self.mouse_image:
            img_w, img_h = self.mouse_image.get_size()
            x_pos = self.mouse_pos[1]*CELL_SIZE+(CELL_SIZE-img_w)//2; y_pos = self.mouse_pos[0]*CELL_SIZE+(CELL_SIZE-img_h)//2
            self.screen.blit(self.mouse_image, (x_pos, y_pos))
        else: 
            pygame.draw.rect(self.screen, BLUE, (self.mouse_pos[1]*CELL_SIZE+CELL_SIZE//4, self.mouse_pos[0]*CELL_SIZE+CELL_SIZE//4, CELL_SIZE//2, CELL_SIZE//2))

    def draw_path(self, path, color=GREEN, width=3):
        if len(path) > 1:
            path_points = [(c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2) for r,c in path]
            pygame.draw.lines(self.screen, color, False, path_points, width)

    def draw_text(self, text, position, color=RED): 
        try:
            img = self.font.render(text, True, color)
            self.screen.blit(img, position)
        except Exception: pass

    def run_episode(self, learn=True, display_speed=15):
        state = self.reset_game_state()
        done = False; total_reward = 0; episode_steps = 0
        
        num_initial_cheese_for_steps = len(self.initial_cheese_positions) if self.initial_cheese_positions else 0
        max_display_steps = self.grid_width * self.grid_height * (num_initial_cheese_for_steps + 1.2) * 1.2
        if max_display_steps < 20: max_display_steps = 20

        y_offset = 10 
        line_height = 30 

        while not done and episode_steps < max_display_steps:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: return None, None, True 
            action = self.q_learning_agent.choose_action(state, learn)
            next_state, reward, done = self.perform_action(action)
            if learn: self.q_learning_agent.learn(state, action, reward, next_state, done)
            state = next_state; total_reward += reward; episode_steps += 1

            self.screen.fill(WHITE); self.draw_grid(); self.draw_elements()
            if not learn: self.draw_path(self.current_path, RED, 3)
            
            current_y = y_offset
            self.draw_text(f"Steps: {episode_steps}", (10, current_y)) 
            current_y += line_height
            self.draw_text(f"Cheese Eaten: {self.score}/{self.target_cheese_count}", (10, current_y)) 
            current_y += line_height
            
            status_text = f"Epsilon: {self.q_learning_agent.epsilon:.4f}"
            if not learn:
                status_text = "Finding cheese..." 
                if self.score >= self.target_cheese_count: status_text = f"ATE {self.target_cheese_count} CHEESES!" 
            self.draw_text(status_text, (10, current_y))
            current_y += line_height

            if learn and hasattr(self.q_learning_agent, 'current_episode_num'):
                     self.draw_text(f"Episode: {self.q_learning_agent.current_episode_num}", (10, current_y)) 
            
            pygame.display.flip(); self.clock.tick(display_speed)
        return total_reward, episode_steps, False

    def train(self, num_episodes):
        print("Starting training...") 
        rewards_per_episode, steps_per_episode = [], []
        for episode in range(num_episodes):
            if hasattr(self.q_learning_agent, 'current_episode_num'):
                self.q_learning_agent.current_episode_num = episode + 1
            total_reward, episode_steps, quit_flag = self.run_episode(learn=True, display_speed=400)
            if quit_flag: 
                print("Training interrupted by user.")
                # Lưu Q-table trước khi thoát
                self.q_learning_agent.save_q_table()
                return 
            if total_reward is None: 
                print("Training stopped by user.")
                # Lưu Q-table trước khi thoát
                self.q_learning_agent.save_q_table()
                return 
            rewards_per_episode.append(total_reward); steps_per_episode.append(episode_steps)
            self.q_learning_agent.decay_epsilon()

            if (episode + 1) % 100 == 0:
                avg_r = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode)>=100 else np.mean(rewards_per_episode) if rewards_per_episode else 0
                avg_s = np.mean(steps_per_episode[-100:]) if len(steps_per_episode)>=100 else np.mean(steps_per_episode) if steps_per_episode else 0
                print(f"Ep {episode+1}/{num_episodes} - AvgReward: {avg_r:.2f}, AvgSteps: {avg_s:.2f}, Epsilon: {self.q_learning_agent.epsilon:.4f}, Q-size: {len(self.q_learning_agent.q_table)}")

            if episode > 200 and self.target_cheese_count > 0 : 
                wins_last_50 = 0; start_idx = max(0, len(rewards_per_episode) - 50)
                if len(rewards_per_episode) - start_idx >= 50:
                    reward_threshold_win = (self.target_cheese_count * 20 + 60) - (self.grid_width + self.grid_height) * 0.1 
                    for i in range(start_idx, len(rewards_per_episode)):
                        if rewards_per_episode[i] > reward_threshold_win: wins_last_50 +=1
                if wins_last_50 >= 45 : 
                    print(f"Consistently solving. Stopping training early at episode {episode+1}.") 
                    break
        print("Training finished.")
        # Lưu Q-table sau khi huấn luyện hoàn tất
        self.q_learning_agent.save_q_table()

    def demonstrate_optimal_path(self, max_steps_demo=50):
        print("\nDemonstrating optimal path (epsilon = 0)...") 
        self.q_learning_agent.epsilon = 0 
        total_reward, episode_steps, quit_flag = self.run_episode(learn=False, display_speed=10)
        if quit_flag: 
            print("Demonstration interrupted by user.")
            return 
        if total_reward is None: 
            print("Demonstration stopped by user.")
            return 

        final_message = "Goal not achieved in demo." 
        if self.score >= self.target_cheese_count:
            final_message = f"ATE {self.score} CHEESES! Steps: {episode_steps}, Reward: {total_reward:.2f}" 
        print(final_message)

        y_offset = 10
        line_height = 30
        current_y = y_offset

        self.screen.fill(WHITE); self.draw_grid(); self.draw_elements()
        self.draw_path(self.current_path, RED, 3)
        self.draw_text(f"Steps: {episode_steps}", (10, current_y)) 
        current_y += line_height
        self.draw_text(f"Cheese Eaten: {self.score}/{self.target_cheese_count}", (10, current_y)) 
        current_y += line_height
        status_text_demo = "Demo Ended." 
        if self.score >= self.target_cheese_count: status_text_demo = "COMPLETED!" 
        self.draw_text(status_text_demo, (10, current_y))
        pygame.display.flip()
        pygame.time.wait(4000)

class QLearningAgent:
    def __init__(self, actions, maze_shape, 
                 learning_rate=0.1,      
                 discount_factor=0.95,    
                 epsilon_start=1.0,
                 epsilon_end=0.05,       
                 epsilon_decay_rate=0.9995, 
                 initial_q_value=1.0,
                 q_table_file="q_table.csv"):   
        self.actions = actions; self.lr = learning_rate; self.gamma = discount_factor
        self.epsilon = epsilon_start; self.epsilon_min = epsilon_end; self.epsilon_decay = epsilon_decay_rate
        self.q_table = defaultdict(lambda: np.full(len(actions), initial_q_value, dtype=float))
        self.maze_rows, self.maze_cols = maze_shape; self.current_episode_num = 0
        self.q_table_file = q_table_file  # Đường dẫn file CSV để lưu/tải Q-table

    def choose_action(self, state, learn=True):
        if learn and random.uniform(0,1) < self.epsilon: return random.choice(self.actions)
        else:
            q_values = self.q_table[state]; max_q = np.max(q_values)
            return random.choice([i for i,q in enumerate(q_values) if q == max_q])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_future_q = 0 if done else np.max(self.q_table[next_state])
        target_q = reward + self.gamma * max_future_q
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        """Lưu Q-table vào file CSV."""
        try:
            with open(self.q_table_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['row', 'col', 'cheese_positions', 'action', 'q_value'])
                for (state, action), q_value in self.q_table.items():
                    row, col, cheese_fset = state
                    cheese_str = str(sorted(list(cheese_fset)))  # Chuyển frozenset thành chuỗi để lưu
                    for a, q in enumerate(q_value):
                        writer.writerow([row, col, cheese_str, a, q])
            print(f"Q-table saved to {self.q_table_file}")
        except Exception as e:
            print(f"Error saving Q-table to {self.q_table_file}: {e}")

    def load_q_table(self):
        """Tải Q-table từ file CSV nếu tồn tại."""
        if not os.path.exists(self.q_table_file):
            print(f"No Q-table found at {self.q_table_file}. Starting with empty Q-table.")
            return
        try:
            with open(self.q_table_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                if not all(col in reader.fieldnames for col in ['row', 'col', 'cheese_positions', 'action', 'q_value']):
                    print(f"Invalid Q-table format in {self.q_table_file}")
                    return
                self.q_table = defaultdict(lambda: np.full(len(self.actions), 1.0, dtype=float))
                for row in reader:
                    try:
                        state = (
                            int(row['row']),
                            int(row['col']),
                            frozenset(eval(row['cheese_positions']))  # Chuyển chuỗi thành frozenset
                        )
                        action = int(row['action'])
                        q_value = float(row['q_value'])
                        self.q_table[state][action] = q_value
                    except (ValueError, SyntaxError) as e:
                        print(f"Skipping invalid row in Q-table CSV: {row}, Error: {e}")
                        continue
            print(f"Q-table loaded from {self.q_table_file}. Size: {len(self.q_table)}")
        except Exception as e:
            print(f"Error loading Q-table from {self.q_table_file}: {e}")

if __name__ == '__main__':
    maze_layout = DEFAULT_MAZE
    
    TARGET_CHEESE_TO_EAT = 2

    valid_positions = []
    for r_idx in range(len(maze_layout)):
        for c_idx in range(len(maze_layout[0])):
            if maze_layout[r_idx][c_idx] == 0:
                valid_positions.append((r_idx, c_idx))

    if not valid_positions: raise ValueError("Maze has no valid paths!") 
    start_pos = random.choice(valid_positions)
    
    num_cheese_on_map = TARGET_CHEESE_TO_EAT 
    cheese_positions = []
    available_for_cheese = list(set(valid_positions) - {start_pos})

    if len(available_for_cheese) < num_cheese_on_map:
        print(f"Warning: Not enough valid spots for {num_cheese_on_map} cheeses, reducing to {len(available_for_cheese)}.") 
        num_cheese_on_map = len(available_for_cheese)
        if TARGET_CHEESE_TO_EAT > num_cheese_on_map:
             print(f"NOTE: Target is to eat {TARGET_CHEESE_TO_EAT} cheeses, but only {num_cheese_on_map} are on the map.") 

    if num_cheese_on_map > 0:
        cheese_positions = random.sample(available_for_cheese, num_cheese_on_map)

    print(f"Mouse starting at: {start_pos}") 
    print(f"Cheese ({len(cheese_positions)} pieces): {cheese_positions}") 
    print(f"Goal: Eat {TARGET_CHEESE_TO_EAT} cheese pieces.") 

    game = None 
    try:
        q_agent_custom_params = {
            "learning_rate": 0.1, 
            "discount_factor": 0.95, 
            "epsilon_decay_rate": 0.9995, 
            "initial_q_value": 1.0
        }
        game = MazeGame(maze_layout, cheese_positions, start_pos, 
                        target_cheese_count=TARGET_CHEESE_TO_EAT,
                        q_agent_params=q_agent_custom_params,
                        q_table_file="q_table.csv")

        num_training_episodes = 1500 
        if not cheese_positions : 
            num_training_episodes = 10 
        
        game.train(num_training_episodes)
        game.demonstrate_optimal_path(max_steps_demo=int(game.grid_width * game.grid_height * 2.5))

    except KeyboardInterrupt: print("\nProgram interrupted by user (Ctrl+C).") 
    except Exception as e: 
        print(f"\nUnexpected error: {e}"); import traceback; traceback.print_exc() 
    finally:
        if pygame.get_init(): pygame.quit()