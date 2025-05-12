# -*- coding: utf-8 -*-
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, simpledialog
import time
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import traceback
import pygame 
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from maze_game_module.mouse_cheese_game import MazeGame
    MAZE_GAME_AVAILABLE = True
except ImportError as e:
    print(f"Could not import MouseCheeseGame: {e}")
    MAZE_GAME_AVAILABLE = False

try:
    import pygame_gui 
    PYGAME_GUI_AVAILABLE = True
except ImportError:
    PYGAME_GUI_AVAILABLE = False


try:

    from sudoku_game_module.sudoku_solver_game import SudokuPygame
    SUDOKU_GAME_AVAILABLE = True
except ImportError as e:
    print(f"Could not import SudokuPygame: {e}")
    SUDOKU_GAME_AVAILABLE = False


try:
    import ThuatToan
except ImportError:
    messagebox.showerror("Lỗi (Error)", "Không tìm thấy file ThuatToan.py.\nVui lòng đặt file ThuatToan.py cùng thư mục.")
    exit()


DEFAULT_START_STATE = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
DEFAULT_GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
ANIMATION_DELAY_MS = 400
GRAPH_FILENAME = "heuristic_graph.png"
Q_TABLE_FILE = os.path.join(script_dir, "q_table.csv")

SINGLE_STATE_ALGORITHMS = {
    "BFS": ThuatToan.BFS,
    "UCS": ThuatToan.UCS,
    "DFS": ThuatToan.DFS,
    "IDDFS": ThuatToan.IDDFS,
    "Backtracking": ThuatToan.Backtracking_Search,
    "Greedy (Manhattan)": lambda start, goal: ThuatToan.Greedy(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "Greedy (Misplaced)": lambda start, goal: ThuatToan.Greedy(start, goal, h_func=ThuatToan.Chiphi),
    "A* (Manhattan)": lambda start, goal: ThuatToan.A_Star(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "A* (Misplaced)": lambda start, goal: ThuatToan.A_Star(start, goal, h_func=ThuatToan.Chiphi),
    "IDA* (Manhattan)": lambda start, goal: ThuatToan.IDA_Star(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "IDA* (Misplaced)": lambda start, goal: ThuatToan.IDA_Star(start, goal, h_func=ThuatToan.Chiphi),
    "Simple HC (Manhattan)": lambda start, goal: ThuatToan.Simple_HC(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "Steepest HC (Manhattan)": lambda start, goal: ThuatToan.Steepest_HC(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "Stochastic HC (Manhattan)": lambda start, goal: ThuatToan.Stochastic_HC(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "Beam Search (k=3, Manhattan)": lambda start, goal: ThuatToan.Beam_Search(start, goal, h_func=ThuatToan.khoang_cach_mahathan, beam_width_k=3),
    "Simulated Annealing (Manhattan)": lambda start, goal: ThuatToan.Simulated_Annealing(start, goal, h_func=ThuatToan.khoang_cach_mahathan),
    "AND-OR Search (DFS Sim)": ThuatToan.solve_with_and_or_8puzzle,
    "Genetic Algorithm (8-Puzzle)": ThuatToan.Genetic_Algorithm_8Puzzle,
    "Q-Learning (8-Puzzle)": ThuatToan.Q_Learning_8Puzzle,
}

BELIEF_STATE_ALGORITHMS = {
    "Conformant BFS": ThuatToan.conformant_bfs,
    "Belief Search (BFS-like)": ThuatToan.Belief_Search,
    "Belief1p Search (Greedy)": ThuatToan.Belief1p_Search,
}

def validate_puzzle_state(matrix):
    flat_list = [item for sublist in matrix for item in sublist]
    if len(flat_list) != 9: return False, "State must have 9 numbers."
    seen = set()
    for num in flat_list:
        if not isinstance(num, int) or not (0 <= num <= 8): return False, f"Invalid number: {num}."
        if num in seen: return False, f"Duplicate number: {num}."
        seen.add(num)
    if len(seen) != 9: return False, "Numbers 0-8 must appear exactly once."
    return True, ""

def get_inversion_count(arr):
    count = 0; arr_no_zero = [x for x in arr if x != 0]; n = len(arr_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            if arr_no_zero[i] > arr_no_zero[j]: count += 1
    return count

def is_solvable(matrix):
    flat_list = [item for sublist in matrix for item in sublist]
    inversion_count = get_inversion_count(flat_list)
    return inversion_count % 2 == 0

def generate_random_solvable_state():
    while True:
        nums = list(range(9)); random.shuffle(nums)
        matrix = [nums[0:3], nums[3:6], nums[6:9]]
        if is_solvable(matrix): return matrix

def plot_heuristic_graph(path, goal_state, filename=GRAPH_FILENAME):
    if not path or len(path) < 1 or not all(isinstance(s, list) for s in path):
        return False
    try:
        valid_path_for_heuristic = []
        for state in path:
            is_valid_state, _ = validate_puzzle_state(state)
            if is_valid_state:
                valid_path_for_heuristic.append(state)
            else:
                pass

        if not valid_path_for_heuristic:
            return False

        heuristics = [ThuatToan.khoang_cach_mahathan(state, goal_state) for state in valid_path_for_heuristic]
        heuristics = [h for h in heuristics if h != float('inf')]

        if not heuristics:
            return False

        plt.figure(figsize=(5, 3)); plt.plot(range(len(heuristics)), heuristics, marker='o', linestyle='-', color='#4682B4')
        plt.title("Heuristic Value (Manhattan) vs. Step"); plt.xlabel("Step Number"); plt.ylabel("Heuristic Value")
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig(filename); plt.close()
        return True
    except Exception as e:
        return False

class InputStateDialog(ctk.CTkToplevel):
    def __init__(self, parent, title="Nhập Trạng Thái", current_state=None):
        super().__init__(parent)
        self.transient(parent); self.title(title); self.geometry("280x350"); self.lift()
        self.attributes("-topmost", True); self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.resizable(False, False); self.grab_set()
        self.result_state = None; self.parent = parent
        self.label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold")); self.label.pack(pady=10)
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent"); self.grid_frame.pack(pady=5)
        self.entries = []
        for i in range(3):
            for j in range(3):
                entry = ctk.CTkEntry(self.grid_frame, width=60, height=60, justify='center', font=ctk.CTkFont(size=20))
                entry.grid(row=i, column=j, padx=5, pady=5)
                if current_state:
                    try: entry.insert(0, str(current_state[i][j]))
                    except IndexError: pass
                self.entries.append(entry)
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent"); self.button_frame.pack(pady=15, fill="x", padx=20)
        self.button_frame.grid_columnconfigure((0, 1), weight=1)
        self.ok_button = ctk.CTkButton(self.button_frame, text="OK", command=self._on_ok); self.ok_button.grid(row=0, column=0, padx=10, sticky="ew")
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Hủy (Cancel)", command=self._on_cancel, fg_color="grey"); self.cancel_button.grid(row=0, column=1, padx=10, sticky="ew")
        self.entries[0].focus()
    def _on_ok(self):
        state = [[0]*3 for _ in range(3)]; temp_values = []
        try:
            for i in range(3):
                for j in range(3):
                    val_str = self.entries[i*3+j].get()
                    if not val_str.isdigit(): raise ValueError(f"Invalid input '{val_str}' at row {i+1}, col {j+1}.")
                    val = int(val_str); state[i][j] = val; temp_values.append(val)
            is_valid, msg = validate_puzzle_state(state)
            if not is_valid: messagebox.showerror("Lỗi Xác Thực (Validation Error)", msg, parent=self); return
            self.result_state = state; self.grab_release(); self.destroy()
        except ValueError as e: messagebox.showerror("Lỗi Đầu vào (Input Error)", str(e), parent=self)
        except Exception as e: messagebox.showerror("Lỗi (Error)", f"An unexpected error occurred: {e}", parent=self)
    def _on_cancel(self): self.result_state = None; self.grab_release(); self.destroy()
    def get_state(self): self.parent.wait_window(self); return self.result_state

class PuzzleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("8-Puzzle Solver & Mini Games - Visual Interface") 
        self.geometry("1250x800")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.belief_mode = tk.BooleanVar(value=False)
        self.belief_states = []
        self.current_start_state = copy.deepcopy(DEFAULT_START_STATE)
        self.current_goal_state = copy.deepcopy(DEFAULT_GOAL_STATE)
        self.solution_path = None
        self.current_step_index = 0
        self.is_running_auto = False
        self.elapsed_time = 0.0
        self.history = []
        self.selected_algo_name = ""
        self._animation_job = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)

        self.control_frame = ctk.CTkFrame(self, width=280, corner_radius=10)
        self.control_frame.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        self.control_frame.grid_rowconfigure(16, weight=1) 

        self.algo_label = ctk.CTkLabel(self.control_frame, text="Algorithm (8-Puzzle):", font=ctk.CTkFont(weight="bold"))
        self.algo_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        self.algo_combobox = ctk.CTkComboBox(self.control_frame, width=200, command=self.on_algo_select)
        self.algo_combobox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.solve_button = ctk.CTkButton(self.control_frame, text="Giải (Solve 8-Puzzle)", command=self.solve_puzzle_callback)
        self.solve_button.grid(row=2, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="ew")

        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray").grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.belief_mode_label = ctk.CTkLabel(self.control_frame, text="8-Puzzle Mode:", font=ctk.CTkFont(weight="bold"))
        self.belief_mode_label.grid(row=4, column=0, padx=(10,0), pady=(5, 5), sticky="w")
        self.belief_mode_checkbox = ctk.CTkCheckBox(self.control_frame, text="Belief State Mode",
                                                  variable=self.belief_mode, command=self.toggle_belief_mode)
        self.belief_mode_checkbox.grid(row=4, column=1, padx=(0,10), pady=(5, 5), sticky="w")

        self.state_control_label = ctk.CTkLabel(self.control_frame, text="Single Start State (8-Puzzle):", font=ctk.CTkFont(weight="bold"))
        self.state_control_label.grid(row=5, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.edit_add_start_button = ctk.CTkButton(self.control_frame, text="Edit Start State", command=self.edit_or_add_start_state)
        self.edit_add_start_button.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.clear_belief_button = ctk.CTkButton(self.control_frame, text="Clear Belief States", command=self.clear_belief_states, state="disabled")
        self.clear_belief_button.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.edit_goal_button = ctk.CTkButton(self.control_frame, text="Edit Goal State", command=self.edit_goal_state)
        self.edit_goal_button.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.randomize_button = ctk.CTkButton(self.control_frame, text="Random Start (Single Mode)", command=self.randomize_start_callback)
        self.randomize_button.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray").grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        
        self.mini_games_label = ctk.CTkLabel(self.control_frame, text="Mini Games:", font=ctk.CTkFont(weight="bold"))
        self.mini_games_label.grid(row=11, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.run_maze_game_button = ctk.CTkButton(self.control_frame, text="Chạy Game Chuột Tìm Phô Mai", command=self.run_mouse_cheese_game_callback)
        self.run_maze_game_button.grid(row=12, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        if not MAZE_GAME_AVAILABLE:
            self.run_maze_game_button.configure(state="disabled", text="Game Mê Cung (Lỗi Import)")

        
        self.run_sudoku_game_button = ctk.CTkButton(self.control_frame, text="Chạy Game Sudoku Solver", command=self.run_sudoku_game_callback)
        self.run_sudoku_game_button.grid(row=13, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        if not SUDOKU_GAME_AVAILABLE:
            self.run_sudoku_game_button.configure(state="disabled", text="Game Sudoku (Lỗi Import)")
        
        
        self.theme_label = ctk.CTkLabel(self.control_frame, text="Appearance:", font=ctk.CTkFont(weight="bold"))
        self.theme_label.grid(row=14, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="w")
        self.theme_menu = ctk.CTkOptionMenu(self.control_frame, values=["System", "Light", "Dark"], command=ctk.set_appearance_mode)
        self.theme_menu.grid(row=15, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        
        self.main_area = ctk.CTkTabview(self, corner_radius=10)
        self.main_area.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        self.tab_visualization = self.main_area.add("8-Puzzle Visualization")
        self.tab_belief_states = self.main_area.add("8-Puzzle Belief States")
        self.tab_history = self.main_area.add("History (8-Puzzle)")
        self.tab_statistics = self.main_area.add("Statistics (8-Puzzle)")
        self.tab_graph = self.main_area.add("Heuristic Graph (8-Puzzle)")

        self.belief_list_frame = ctk.CTkScrollableFrame(self.tab_belief_states, label_text="Current Belief States (Initial Set for 8-Puzzle)")
        self.belief_list_frame.pack(expand=True, fill="both", padx=10, pady=10)
        self.belief_list_widgets = []

        self.tab_visualization.grid_columnconfigure((0, 1, 2), weight=1)
        self.tab_visualization.grid_rowconfigure(0, weight=1)
        self.tab_visualization.grid_rowconfigure(1, minsize=50)
        self.tab_visualization.grid_rowconfigure(2, minsize=40)

        self.boards_frame = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        self.boards_frame.grid(row=0, column=0, columnspan=3, pady=(10, 5), sticky="nsew")
        self.boards_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_grid_frame, self.start_tiles = self.create_visual_grid(self.boards_frame, "Start State (Single Mode)")
        self.start_grid_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self.current_grid_frame, self.current_tiles = self.create_visual_grid(self.boards_frame, "Current State / Moves")
        self.current_grid_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.goal_grid_frame, self.goal_tiles = self.create_visual_grid(self.boards_frame, "Goal State")
        self.goal_grid_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        self.update_visual_grid(self.goal_tiles, self.current_goal_state)

        self.conformant_result_textbox = ctk.CTkTextbox(self.current_grid_frame, height=150, wrap="word", state="disabled", font=ctk.CTkFont(size=14))
        self.anim_controls_frame = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        self.anim_controls_frame.grid(row=1, column=0, columnspan=3, pady=5)
        self.prev_button = ctk.CTkButton(self.anim_controls_frame, text="◀ Previous", width=100, command=self.prev_step, state="disabled")
        self.prev_button.pack(side="left", padx=5)
        self.play_pause_button = ctk.CTkButton(self.anim_controls_frame, text="▶ Play", width=100, command=self.toggle_play_pause, state="disabled")
        self.play_pause_button.pack(side="left", padx=5)
        self.next_button = ctk.CTkButton(self.anim_controls_frame, text="Next ▶", width=100, command=self.next_step, state="disabled")
        self.next_button.pack(side="left", padx=5)

        self.status_frame = ctk.CTkFrame(self.tab_visualization, fg_color="transparent", corner_radius=10)
        self.status_frame.grid(row=2, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=ctk.CTkFont(size=14), anchor="w")
        self.status_label.pack(side="left", padx=10, pady=5)
        self.step_label = ctk.CTkLabel(self.status_frame, text="Step: 0/0", font=ctk.CTkFont(size=14), anchor="center")
        self.step_label.pack(side="left", padx=10, pady=5, expand=True)
        self.time_label = ctk.CTkLabel(self.status_frame, text="Time: 0.00s", font=ctk.CTkFont(size=14), anchor="e")
        self.time_label.pack(side="right", padx=10, pady=5)

        self.history_textbox = ctk.CTkTextbox(self.tab_history, wrap="none", state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.history_textbox.pack(expand=True, fill="both", padx=10, pady=10)

        self.stats_textbox = ctk.CTkTextbox(self.tab_statistics, wrap="none", state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.stats_textbox.pack(expand=True, fill="both", padx=10, pady=10)

        self.graph_frame = ctk.CTkFrame(self.tab_graph, fg_color="transparent")
        self.graph_frame.pack(expand=True, fill="both", padx=10, pady=10)
        self.graph_label = ctk.CTkLabel(self.graph_frame, text="Graph will appear here after solving (Single State Mode).", font=ctk.CTkFont(size=14))
        self.graph_label.pack(pady=20)
        self.canvas_widget = None

        self.toggle_belief_mode()
        self.update_history_display()
        self.update_statistics_display()


    
    def create_visual_grid(self, parent, label_text, grid_size=60, font_size=24):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        label = ctk.CTkLabel(container, text=label_text, font=ctk.CTkFont(weight="bold"))
        label.pack(pady=(0, 5))
        grid_frame = ctk.CTkFrame(container, fg_color="gray50", corner_radius=5)
        grid_frame.pack()
        tile_widgets = []
        for i in range(3):
            for j in range(3):
                tile = ctk.CTkLabel(grid_frame, text="", width=grid_size, height=grid_size,
                                    font=ctk.CTkFont(size=font_size, weight="bold"), corner_radius=5)
                tile.grid(row=i, column=j, padx=2, pady=2)
                tile_widgets.append(tile)
        return container, tile_widgets

    def update_visual_grid(self, tile_widgets, state_matrix):
        if not isinstance(state_matrix, list) or len(state_matrix) != 3 or not all(isinstance(row, list) and len(row) == 3 for row in state_matrix):
            for tile_widget in tile_widgets: tile_widget.configure(text="!", fg_color="red")
            return

        flat_state = [item for sublist in state_matrix for item in sublist]
        for i, tile_widget in enumerate(tile_widgets):
            num = flat_state[i]
            if num == 0: tile_widget.configure(text="", fg_color=("#E0E0E0", "#404040"))
            else: tile_widget.configure(text=str(num), fg_color=("#ADD8E6", "#306080"))

    def update_status(self, text, color="white"):
        status_colors = { "green": ("#008000", "#32CD32"), "red": ("#FF0000", "#FF6347"), "orange": ("#FFA500", "#FF8C00"), "blue": ("#0000FF", "#87CEEB"), "cyan": ("#00FFFF", "#AFEEEE"), "white": ("#000000", "#FFFFFF") }
        text_color = status_colors.get(color, status_colors["white"])
        self.status_label.configure(text=f"Status: {text}", text_color=text_color); self.update_idletasks()

    def update_step_display(self):
         if self.belief_mode.get() and isinstance(self.solution_path, list):
             total_steps = len(self.solution_path)
             self.step_label.configure(text=f"Moves: {total_steps}")
         elif not self.belief_mode.get() and isinstance(self.solution_path, list) and \
              all(isinstance(s, list) for s in self.solution_path):
             total_states = len(self.solution_path)
             self.step_label.configure(text=f"Step: {self.current_step_index + 1}/{total_states if total_states > 0 else 0}")
         else:
             self.step_label.configure(text="Step: 0/0")


    def update_time_display(self):
        self.time_label.configure(text=f"Time: {self.elapsed_time:.3f}s")

    def enable_animation_controls(self, enable=True):
        if self.belief_mode.get():
            state = "disabled"
        else:
            state = "normal" if enable else "disabled"

        self.prev_button.configure(state=state)
        self.play_pause_button.configure(state=state)
        self.next_button.configure(state=state)

        if enable and not self.belief_mode.get():
            self.prev_button.configure(state="disabled" if self.current_step_index == 0 else "normal")
            self.next_button.configure(state="disabled" if not self.solution_path or \
                                       not all(isinstance(s, list) for s in self.solution_path) or \
                                       self.current_step_index >= len(self.solution_path) - 1 else "normal")

    def update_history_display(self):
        self.history_textbox.configure(state="normal")
        self.history_textbox.delete("1.0", tk.END)
        if not self.history: self.history_textbox.insert("1.0", "No 8-Puzzle history yet.")
        else:
            header = f"{'Algorithm':<30} | {'Steps/Moves':>12} | {'Time (s)':>9}\n"
            header += "-" * (len(header) -1) + "\n"; self.history_textbox.insert("1.0", header)
            for _idx, (algo_name_hist, _start_data_hist, _goal_state_hist, path_hist, time_hist) in enumerate(reversed(self.history)):
                steps_str = "N/A"
                if algo_name_hist in BELIEF_STATE_ALGORITHMS:
                    if path_hist and isinstance(path_hist, list):
                        steps_str = f"{len(path_hist)} moves"
                    elif path_hist is None: steps_str = "No Plan"
                    else: steps_str = "ERR (Plan Type)"
                else:
                    if path_hist and isinstance(path_hist, list) and len(path_hist) > 0:
                        if all(isinstance(s, list) for s in path_hist):
                             steps_str = f"{len(path_hist)-1} steps"
                        else:
                             steps_str = "Invalid Path Content"
                    elif path_hist is None: steps_str = "No Path"
                    elif path_hist and isinstance(path_hist, list) and len(path_hist) == 0:
                        steps_str = "No Path"
                    else: steps_str = "ERR (Path Type)"

                line = f"{algo_name_hist:<30} | {steps_str:>12} | {time_hist:>9.3f}\n"; self.history_textbox.insert(tk.END, line)
        self.history_textbox.configure(state="disabled")

    def update_statistics_display(self):
        self.stats_textbox.configure(state="normal"); self.stats_textbox.delete("1.0", tk.END)
        if not self.history: self.stats_textbox.insert("1.0", "Run some 8-Puzzle algorithms to see statistics.")
        else:
            stats = {}
            for algo_name_hist, _start_data_hist, goal_state_hist, path_hist, time_hist in self.history:
                if algo_name_hist not in stats: stats[algo_name_hist] = {"count": 0, "total_time": 0.0, "total_steps": 0, "solved_count": 0}
                stats[algo_name_hist]["count"] += 1; stats[algo_name_hist]["total_time"] += time_hist

                solved = False; steps_or_moves = 0

                if algo_name_hist in BELIEF_STATE_ALGORITHMS:
                    if path_hist and isinstance(path_hist, list):
                        steps_or_moves = len(path_hist)
                        solved = True
                else:
                    if path_hist and isinstance(path_hist, list) and len(path_hist) > 0 and \
                       all(isinstance(s, list) for s in path_hist) and \
                       path_hist[-1] == goal_state_hist:
                        steps_or_moves = len(path_hist) - 1; solved = True

                if solved:
                    stats[algo_name_hist]["solved_count"] += 1
                    stats[algo_name_hist]["total_steps"] += steps_or_moves

            header = f"{'Algorithm':<32} | {'Runs':>5} | {'Solved':>6} | {'Avg Steps/Moves (Solved)':>22} | {'Avg Time (s)':>12}\n"
            header += "-" * (len(header)-1) + "\n"; self.stats_textbox.insert("1.0", header)
            for algo_name_stat, data in stats.items():
                count = data["count"]; solved_count = data["solved_count"]
                avg_time = data["total_time"] / count if count > 0 else 0.0
                avg_steps_or_moves = data["total_steps"] / solved_count if solved_count > 0 else 0.0
                line = f"{algo_name_stat:<32} | {count:>5} | {solved_count:>6} | {avg_steps_or_moves:>22.1f} | {avg_time:>12.3f}\n"
                self.stats_textbox.insert(tk.END, line)
        self.stats_textbox.configure(state="disabled")


    def display_heuristic_graph(self):
        for widget in self.graph_frame.winfo_children(): widget.destroy()
        self.canvas_widget = None
        if os.path.exists(GRAPH_FILENAME) and not self.belief_mode.get():
            try:
                fig = plt.figure(figsize=(6, 4))
                img = plt.imread(GRAPH_FILENAME)
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                fig.tight_layout(pad=0)

                self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
                self.canvas_widget = self.canvas.get_tk_widget()
                self.canvas_widget.pack(expand=True, fill="both")
                self.canvas.draw()
                plt.close(fig)
            except Exception as e:
                self.graph_label = ctk.CTkLabel(self.graph_frame, text="Error loading graph image.", font=ctk.CTkFont(size=14)); self.graph_label.pack(pady=20)
                self.update_status("Error displaying graph.", "red")
        else:
             self.graph_label = ctk.CTkLabel(self.graph_frame, text="Graph unavailable in Belief Mode or not generated.", font=ctk.CTkFont(size=14))
             self.graph_label.pack(pady=20)


    def toggle_belief_mode(self):
        is_belief = self.belief_mode.get()
        self.edit_add_start_button.configure(text="Add Belief State" if is_belief else "Edit Start State")
        self.clear_belief_button.configure(state="normal" if is_belief else "disabled")
        self.randomize_button.configure(state="disabled" if is_belief else "normal")
        self.state_control_label.configure(text="Belief States (Initial Set for 8-Puzzle):" if is_belief else "Single Start State (8-Puzzle):")

        if is_belief:
            available_algos = list(BELIEF_STATE_ALGORITHMS.keys())
            self.algo_combobox.configure(values=available_algos)
            if available_algos:
                self.algo_combobox.set(available_algos[0])
                self.selected_algo_name = available_algos[0]
            else:
                self.algo_combobox.set("No belief algos")
                self.selected_algo_name = ""
            self.algo_combobox.configure(state="normal" if available_algos else "disabled")
        else:
            available_algos = list(SINGLE_STATE_ALGORITHMS.keys())
            self.algo_combobox.configure(values=available_algos)
            if available_algos :
                self.algo_combobox.set(available_algos[0])
                self.selected_algo_name = available_algos[0]
            else:
                self.algo_combobox.set("No single-state algos")
                self.selected_algo_name = ""
            self.algo_combobox.configure(state="normal")


        if is_belief:
            self.start_grid_frame.grid_forget()
            self.current_grid_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="n")
            self.conformant_result_textbox.pack(expand=True, fill="both", padx=5, pady=5)
            self.current_grid_frame.winfo_children()[0].configure(text="Conformant Plan Output / Belief Moves (8-Puzzle)")
            self.update_visual_grid(self.current_tiles, [[0,0,0],[0,0,0],[0,0,0]])
            self.enable_animation_controls(False)
            if BELIEF_STATE_ALGORITHMS:
                self.main_area.set("8-Puzzle Belief States")
        else:
            self.start_grid_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
            self.current_grid_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
            self.conformant_result_textbox.pack_forget()
            self.current_grid_frame.winfo_children()[0].configure(text="Current State / Moves (8-Puzzle)")
            self.reset_visualization()

        self.update_belief_state_display()

    def update_belief_state_display(self):
        for widget in self.belief_list_frame.winfo_children():
            widget.destroy()
        self.belief_list_widgets = []

        if not self.belief_states:
            no_state_label = ctk.CTkLabel(self.belief_list_frame, text="No 8-Puzzle belief states added. Use 'Add Belief State'.")
            no_state_label.pack(pady=10)
            return

        cols = 3
        for i, state in enumerate(self.belief_states):
            row_num = i // cols
            col_num = i % cols
            state_frame, state_tiles = self.create_visual_grid(self.belief_list_frame, f"Belief State {i+1}", grid_size=40, font_size=16)
            state_frame.grid(row=row_num, column=col_num, padx=10, pady=10, sticky="n")
            self.update_visual_grid(state_tiles, state)
            self.belief_list_widgets.append(state_frame)

    def clear_belief_states(self):
        if not self.belief_mode.get(): return
        confirmed = messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa tất cả trạng thái niềm tin của 8-Puzzle?", parent=self)
        if confirmed:
            self.belief_states = []
            self.update_belief_state_display()
            self.update_status("Belief states (8-Puzzle) cleared.", "blue")

    def edit_or_add_start_state(self):
        if self.belief_mode.get():
            dialog = InputStateDialog(self, title="Add Belief State (8-Puzzle)")
            new_state = dialog.get_state()
            if new_state:
                valid_new, _ = validate_puzzle_state(new_state)
                if not valid_new:
                    messagebox.showerror("Lỗi (Error)", "Invalid state returned from dialog.", parent=self)
                    return
                self.belief_states.append(new_state)
                self.update_belief_state_display()
                self.update_status(f"Belief state {len(self.belief_states)} (8-Puzzle) added.", "green")
        else:
            self.edit_start_state()


    def on_algo_select(self, choice):
        self.selected_algo_name = choice

    def edit_start_state(self):
        if self.belief_mode.get(): return
        dialog = InputStateDialog(self, title="Edit Start State (8-Puzzle)", current_state=self.current_start_state)
        new_state = dialog.get_state()
        if new_state:
            valid_new, _ = validate_puzzle_state(new_state)
            if not valid_new: messagebox.showerror("Lỗi (Error)", "Invalid state returned from dialog.", parent=self); return
            if not is_solvable(new_state):
                 messagebox.showwarning("Cảnh báo (Warning)", "Trạng thái 8-Puzzle này có thể không giải được với mục tiêu mặc định.", parent=self)
            self.current_start_state = new_state
            self.reset_visualization()
            self.update_status("Start state (8-Puzzle) updated. Ready.", "green")

    def edit_goal_state(self):
        dialog = InputStateDialog(self, title="Edit Goal State (8-Puzzle)", current_state=self.current_goal_state)
        new_state = dialog.get_state()
        if new_state:
            valid_new, _ = validate_puzzle_state(new_state)
            if not valid_new: messagebox.showerror("Lỗi (Error)", "Invalid state returned from dialog.", parent=self); return
            self.current_goal_state = new_state
            self.reset_visualization()
            self.update_status("Goal state (8-Puzzle) updated. Ready.", "green")

    def randomize_start_callback(self):
        if self.belief_mode.get(): return
        self.update_status("Generating random 8-Puzzle state...", "orange")
        self.current_start_state = generate_random_solvable_state()
        self.reset_visualization()
        self.update_status("Random start state (8-Puzzle) generated. Ready.", "green")

    def reset_visualization(self):
        self.stop_animation()
        self.solution_path = None
        self.current_step_index = 0
        self.elapsed_time = 0.0
        self.update_visual_grid(self.goal_tiles, self.current_goal_state)

        if self.belief_mode.get():
             self.conformant_result_textbox.configure(state="normal")
             self.conformant_result_textbox.delete("1.0", tk.END)
             self.conformant_result_textbox.configure(state="disabled")
             self.update_visual_grid(self.current_tiles, [[0]*3]*3)
             self.enable_animation_controls(False)
        else:
            self.update_visual_grid(self.start_tiles, self.current_start_state)
            self.update_visual_grid(self.current_tiles, self.current_start_state)
            self.enable_animation_controls(False)

        self.update_step_display()
        self.update_time_display()

        if self.canvas_widget: self.canvas_widget.destroy(); self.canvas_widget = None
        if not any(isinstance(w, FigureCanvasTkAgg) for w in self.graph_frame.winfo_children()):
             for widget in self.graph_frame.winfo_children(): widget.destroy()
             self.graph_label = ctk.CTkLabel(self.graph_frame, text="Graph will appear here after solving (Single State Mode for 8-Puzzle).", font=ctk.CTkFont(size=14))
             self.graph_label.pack(pady=20)

    def solve_puzzle_callback(self): 
        self.reset_visualization()

        is_valid_goal, goal_msg = validate_puzzle_state(self.current_goal_state)
        if not is_valid_goal:
            messagebox.showerror("Lỗi Trạng Thái (State Error)", f"Goal state (8-Puzzle) is invalid:\n{goal_msg}")
            self.update_status("Error: Invalid Goal State (8-Puzzle).", "red"); return
        goal_state_copy = copy.deepcopy(self.current_goal_state)

        is_belief = self.belief_mode.get()
        algo_func = None
        start_data = None
        history_start_data = None

        if is_belief:
            algo_func = BELIEF_STATE_ALGORITHMS.get(self.selected_algo_name)
            if not algo_func:
                messagebox.showerror("Lỗi", f"Algorithm '{self.selected_algo_name}' not found for 8-Puzzle belief mode!"); return
            if not self.belief_states:
                messagebox.showerror("Lỗi", "No 8-Puzzle belief states defined. Please add states.", parent=self)
                self.update_status("Error: 8-Puzzle belief state list empty.", "red"); return

            for i, bs in enumerate(self.belief_states):
                is_valid_bs, bs_msg = validate_puzzle_state(bs)
                if not is_valid_bs:
                    messagebox.showerror("Lỗi Trạng Thái (State Error)", f"Belief state {i+1} (8-Puzzle) is invalid:\n{bs_msg}")
                    self.update_status(f"Error: Invalid Belief State {i+1} (8-Puzzle).", "red"); return

            start_data = copy.deepcopy(self.belief_states)
            history_start_data = copy.deepcopy(self.belief_states)
            self.update_status(f"Running {self.selected_algo_name} (8-Puzzle)...", "orange")

        else:
            algo_func = SINGLE_STATE_ALGORITHMS.get(self.selected_algo_name)
            if not algo_func:
                messagebox.showerror("Lỗi", f"Algorithm '{self.selected_algo_name}' not found for 8-Puzzle!")
                self.update_status("Error: Algorithm not selected/found (8-Puzzle).", "red"); return

            is_valid_start, start_msg = validate_puzzle_state(self.current_start_state)
            if not is_valid_start:
                messagebox.showerror("Lỗi Trạng Thái (State Error)", f"Start state (8-Puzzle) is invalid:\n{start_msg}")
                self.update_status("Error: Invalid Start State (8-Puzzle).", "red"); return

            start_data = copy.deepcopy(self.current_start_state)
            history_start_data = copy.deepcopy(self.current_start_state)
            self.update_status(f"Running {self.selected_algo_name} (8-Puzzle)...", "orange")


        solve_start_time = time.time()
        temp_path = None
        error_occurred = False
        try:
            temp_path = algo_func(start_data, goal_state_copy)

        except Exception as e:
             messagebox.showerror("Lỗi Thuật Toán (Algorithm Error)", f"Error during {self.selected_algo_name} (8-Puzzle):\n{e}\nSee console for details.", parent=self)
             error_occurred = True

        self.elapsed_time = time.time() - solve_start_time
        self.update_time_display()

        history_path_to_store = None

        if error_occurred:
            self.update_status(f"Error during {self.selected_algo_name} (8-Puzzle).", "red")
            self.solution_path = None
            self.enable_animation_controls(False)

        elif is_belief:
            self.enable_animation_controls(False)
            if temp_path is not None and isinstance(temp_path, list):
                self.solution_path = temp_path
                history_path_to_store = temp_path

                move_map = {(-1, 0): "Up", (1, 0): "Down", (0, -1): "Left", (0, 1): "Right"}
                valid_moves = [m for m in temp_path if isinstance(m, tuple) and len(m) == 2]
                moves_str = ", ".join([move_map.get(move, str(move)) for move in valid_moves])

                if not temp_path:
                     result_text = "Initial 8-Puzzle belief state already satisfies the goal. (0 moves)"
                     self.update_status(f"{self.selected_algo_name}: Goal from start! (8-Puzzle)", "green")
                else:
                     result_text = f"Conformant plan (8-Puzzle) found ({len(temp_path)} moves):\n{moves_str}"
                     self.update_status(f"{self.selected_algo_name}: Plan found! (8-Puzzle)", "green")

                self.conformant_result_textbox.configure(state="normal")
                self.conformant_result_textbox.delete("1.0", tk.END)
                self.conformant_result_textbox.insert("1.0", result_text)
                self.conformant_result_textbox.configure(state="disabled")
            else:
                self.update_status(f"{self.selected_algo_name}: No conformant plan found (8-Puzzle).", "blue")
                self.conformant_result_textbox.configure(state="normal")
                self.conformant_result_textbox.delete("1.0", tk.END)
                self.conformant_result_textbox.insert("1.0", "No conformant plan found (8-Puzzle).")
                self.conformant_result_textbox.configure(state="disabled")
                self.solution_path = None; history_path_to_store = None

        else:
             if temp_path and isinstance(temp_path, list) and len(temp_path) > 0 and \
                all(isinstance(s, list) and validate_puzzle_state(s)[0] for s in temp_path):
                 self.solution_path = temp_path
                 history_path_to_store = temp_path
                 self.current_step_index = 0
                 self.update_visual_grid(self.current_tiles, self.solution_path[0])
                 self.enable_animation_controls(True)

                 goal_reached = (self.solution_path[-1] == goal_state_copy)
                 if goal_reached:
                     self.update_status(f"{self.selected_algo_name}: Solution found! (8-Puzzle)", "green")
                     if plot_heuristic_graph(self.solution_path, goal_state_copy): self.display_heuristic_graph()
                     else:
                         if self.canvas_widget: self.canvas_widget.destroy(); self.canvas_widget = None
                         self.graph_label.configure(text="Error generating 8-Puzzle graph.");
                         if not self.graph_label.winfo_ismapped(): self.graph_label.pack(pady=20)
                 else:
                     self.update_status(f"{self.selected_algo_name}: Path found (Goal not reached) (8-Puzzle).", "orange")
                     self.current_step_index = len(self.solution_path) - 1
                     self.update_visual_grid(self.current_tiles, self.solution_path[self.current_step_index])
                     self.enable_animation_controls(True)
                     if plot_heuristic_graph(self.solution_path, goal_state_copy): self.display_heuristic_graph()

             else:
                 self.update_status(f"{self.selected_algo_name}: No solution or invalid path (8-Puzzle).", "blue")
                 self.solution_path = None; history_path_to_store = None
                 self.enable_animation_controls(False)

        self.update_step_display()
        self.history.append((self.selected_algo_name, history_start_data, goal_state_copy, history_path_to_store, self.elapsed_time))
        self.update_history_display()
        self.update_statistics_display()

    def next_step(self):
        if self.is_running_auto: self.stop_animation()
        if self.solution_path and not self.belief_mode.get() and \
           all(isinstance(s, list) for s in self.solution_path) and \
           self.current_step_index < len(self.solution_path) - 1:
            self.current_step_index += 1
            self.update_visual_grid(self.current_tiles, self.solution_path[self.current_step_index])
            self.update_step_display(); self.enable_animation_controls(True)

    def prev_step(self):
        if self.is_running_auto: self.stop_animation()
        if self.solution_path and not self.belief_mode.get() and \
           all(isinstance(s, list) for s in self.solution_path) and \
           self.current_step_index > 0:
            self.current_step_index -= 1
            self.update_visual_grid(self.current_tiles, self.solution_path[self.current_step_index])
            self.update_step_display(); self.enable_animation_controls(True)

    def toggle_play_pause(self):
        if not self.solution_path or self.belief_mode.get() or \
           not all(isinstance(s, list) for s in self.solution_path): return

        if self.is_running_auto: self.stop_animation()
        else:
            if self.current_step_index >= len(self.solution_path) - 1:
                self.current_step_index = -1
            self.start_animation()

    def start_animation(self):
        if not self.solution_path or self.is_running_auto or self.belief_mode.get() or \
           not all(isinstance(s, list) for s in self.solution_path): return

        self.is_running_auto = True; self.play_pause_button.configure(text="⏸ Pause")
        self.prev_button.configure(state="disabled"); self.next_button.configure(state="disabled")
        self.update_status(f"Playing {self.selected_algo_name} (8-Puzzle)...", "cyan")
        self._run_animation_step()

    def stop_animation(self):
        if self._animation_job:
            try: self.after_cancel(self._animation_job)
            except ValueError: pass
            self._animation_job = None
        self.is_running_auto = False; self.play_pause_button.configure(text="▶ Play")
        if self.solution_path and not self.belief_mode.get() and \
           all(isinstance(s, list) for s in self.solution_path):
            self.enable_animation_controls(True)

        current_status_text = self.status_label.cget("text")
        if current_status_text.startswith("Status: Playing"):
            self.update_status(f"Paused (8-Puzzle).", "blue")

    def _run_animation_step(self):
        if not self.is_running_auto or not self.solution_path or self.belief_mode.get() or \
           not all(isinstance(s, list) for s in self.solution_path) :
             if self._animation_job:
                 try: self.after_cancel(self._animation_job);
                 except ValueError: pass
                 self._animation_job = None
             self.stop_animation()
             return

        if self.current_step_index < len(self.solution_path) - 1:
            self.current_step_index += 1
            self.update_visual_grid(self.current_tiles, self.solution_path[self.current_step_index])
            self.update_step_display()
            self._animation_job = self.after(ANIMATION_DELAY_MS, self._run_animation_step)
        else:
            self.stop_animation()
            goal_for_this_run = self.history[-1][2] if self.history else self.current_goal_state

            goal_reached = (self.solution_path[-1] == goal_for_this_run)
            final_message = "Goal Reached! (8-Puzzle)" if goal_reached else "Animation Finished (End of Path for 8-Puzzle)."
            final_color = "green" if goal_reached else "blue"
            self.update_status(f"{self.selected_algo_name}: {final_message}", final_color)

    
    def run_mouse_cheese_game_callback(self):
        if not MAZE_GAME_AVAILABLE:
            messagebox.showerror("Lỗi Game Mê Cung",
                                 "Không thể chạy game do lỗi import MouseCheeseGame.\n"
                                 "Hãy kiểm tra cấu trúc thư mục 'maze_game_module' và các file phụ thuộc.")
            return

        training_episodes = simpledialog.askinteger(
            "Huấn luyện Agent (Game Mê Cung)",
            "Nhập số lượt huấn luyện cho Q-Learning Agent:\n(VD: 0 để không huấn luyện, 100, 1000).\n"
            "Q-table sẽ được lưu/tải từ 'q_table.csv' trong thư mục chương trình.",
            parent=self, minvalue=0, initialvalue=100
        )
        if training_episodes is None:  
            return

        self.update_status("Đang khởi tạo Game Mê Cung...", "orange")
        try:
            
            if not pygame.get_init():
                pygame.init()

            
            maze_layout = [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]
            ]
            TARGET_CHEESE_TO_EAT = 2 
            valid_positions = [(r, c) for r in range(len(maze_layout)) for c in range(len(maze_layout[0])) if maze_layout[r][c] == 0]
            if not valid_positions:
                raise ValueError("Mê cung không có ô trống hợp lệ!")
            start_pos = random.choice(valid_positions)
            available_for_cheese = list(set(valid_positions) - {start_pos})
            num_cheese = min(TARGET_CHEESE_TO_EAT, len(available_for_cheese))
            cheese_positions = random.sample(available_for_cheese, num_cheese) if num_cheese > 0 else []

            q_agent_params = {
                "learning_rate": 0.1, "discount_factor": 0.95, "epsilon_start": 1.0,
                "epsilon_end": 0.05, "epsilon_decay_rate": 0.9995, "initial_q_value": 1.0
            }

            game_instance = MazeGame(
                maze_layout=maze_layout, cheese_positions=cheese_positions, start_pos=start_pos,
                target_cheese_count=TARGET_CHEESE_TO_EAT, q_agent_params=q_agent_params,
                q_table_file=Q_TABLE_FILE 
            )

            self.update_status(f"Game Mê Cung: Đang huấn luyện ({training_episodes} lượt)..." if training_episodes > 0 else "Game Mê Cung: Bắt đầu.", "cyan")
            self.update_idletasks()

            if training_episodes > 0:
                game_instance.train(training_episodes)
            
            self.update_status("Game Mê Cung: AI đang chơi...", "cyan")
            game_instance.demonstrate_optimal_path(max_steps_demo=int(game_instance.grid_width * game_instance.grid_height * 2.5))
            
            
            
            self.update_status("Game Mê Cung đã kết thúc.", "blue")

        except pygame.error as e:
            messagebox.showerror("Lỗi Pygame (Maze)", f"Đã xảy ra lỗi với Pygame: {e}", parent=self)
            self.update_status("Lỗi Pygame khi chạy Game Mê Cung.", "red")
        except ImportError as e: 
             messagebox.showerror("Lỗi Import (Game Mê Cung)", f"Không thể import module cần thiết cho game: {e}\n"
                                 "Kiểm tra console để biết chi tiết.", parent=self)
             self.update_status(f"Lỗi import Game Mê Cung: {e}", "red")
        except Exception as e:
            messagebox.showerror("Lỗi Game Mê Cung", f"Đã xảy ra lỗi không xác định: {e}\n{traceback.format_exc()}", parent=self)
            self.update_status("Lỗi không xác định Game Mê Cung.", "red")


    
    def run_sudoku_game_callback(self):
        if not SUDOKU_GAME_AVAILABLE:
            messagebox.showerror("Lỗi Game Sudoku",
                                 "Không thể chạy game Sudoku do lỗi import SudokuPygame.\n"
                                 "Hãy kiểm tra cấu trúc thư mục 'sudoku_game_module'.")
            return

        self.update_status("Đang khởi tạo Game Sudoku...", "orange")
        try:
            
            
            
            
            
            

            sudoku_game_instance = SudokuPygame()
            sudoku_game_instance.run() 

            self.update_status("Game Sudoku đã kết thúc.", "blue")

        except pygame.error as e:
            messagebox.showerror("Lỗi Pygame (Sudoku)", f"Đã xảy ra lỗi với Pygame khi chạy Sudoku: {e}", parent=self)
            self.update_status("Lỗi Pygame khi chạy Game Sudoku.", "red")
        except Exception as e:
            messagebox.showerror("Lỗi Game Sudoku", f"Đã xảy ra lỗi không xác định trong game Sudoku: {e}\n{traceback.format_exc()}", parent=self)
            self.update_status("Lỗi không xác định Game Sudoku.", "red")


if __name__ == "__main__":
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    if current_script_dir not in sys.path:
        sys.path.append(current_script_dir)
    
    if os.path.exists(GRAPH_FILENAME):
        try: os.remove(GRAPH_FILENAME)
        except OSError as e: pass
    app = PuzzleApp()
    app.mainloop()