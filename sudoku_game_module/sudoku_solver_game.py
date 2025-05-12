import pygame
import sys
import time


def find_empty_location_solver_4x4(board):
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid_solver_4x4(board, num, pos):
    row, col = pos
    
    for j in range(4):
        if board[row][j] == num and col != j:
            return False

    for i in range(4):
        if board[i][col] == num and row != i:
            return False
    
    box_x = col // 2
    box_y = row // 2
    for i in range(box_y * 2, box_y * 2 + 2):
        for j in range(box_x * 2, box_x * 2 + 2):
            if board[i][j] == num and (i, j) != pos:
                return False
    return True

def solve_sudoku_recursive_visual_4x4(board_display_func, board):
    empty_pos = find_empty_location_solver_4x4(board)
    if not empty_pos:
        return True  

    row, col = empty_pos

    for num in range(1, 5):  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                return "QUIT"

        if is_valid_solver_4x4(board, num, (row, col)):
            board[row][col] = num
            board_display_func()
            pygame.time.delay(50) 

            result = solve_sudoku_recursive_visual_4x4(board_display_func, board)
            if result is True:
                return True
            if result == "QUIT":
                return "QUIT"

            board[row][col] = 0  
            board_display_func()
            pygame.time.delay(50)
    return False

GRID_DIMENSION = 4 
CELL_SIZE = 80 
GRID_SIZE = GRID_DIMENSION * CELL_SIZE 
WIDTH, HEIGHT = GRID_SIZE, GRID_SIZE + 120  

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (211, 211, 211)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)
GREEN_SOLVE = (0, 100, 0)
RED_MSG = (200, 0, 0)
BLUE_MSG = (0, 0, 200)
SELECTED_COLOR = (255, 165, 0)
BUTTON_COLOR = (100, 180, 100)
BUTTON_TEXT_COLOR = (0,0,0)

class SudokuPygame:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("4x4 Sudoku Game & Solver")
        self.font_cell = pygame.font.SysFont("Consolas", int(CELL_SIZE * 0.6), bold=True) 
        self.font_button = pygame.font.SysFont("Arial", 20) 
        self.font_message = pygame.font.SysFont("Arial", 18)

        self.board_initial = [
            [1, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 4]
        ]
        

        self.board_current = [row[:] for row in self.board_initial]
        self.board_solved_overlay = [[0]*GRID_DIMENSION for _ in range(GRID_DIMENSION)]

        self.selected_cell = None
        self.message_text = "4x4 Sudoku Ready!"
        self.message_color = BLUE_MSG
        self.is_solving = False

        button_width = 120 
        button_height = 35
        button_y_pos = GRID_SIZE + 15
        spacing = (WIDTH - 3 * button_width) // 4

        self.solve_button_rect = pygame.Rect(spacing, button_y_pos, button_width, button_height)
        self.reset_button_rect = pygame.Rect(spacing * 2 + button_width, button_y_pos, button_width, button_height)
        self.clear_button_rect = pygame.Rect(spacing * 3 + button_width * 2, button_y_pos, button_width, button_height)

    def draw_grid_lines(self):
        for i in range(GRID_DIMENSION + 1):
            line_width = 3 if i % 2 == 0 else 1 
            pygame.draw.line(self.screen, BLACK, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), line_width)
            pygame.draw.line(self.screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), line_width)

    def draw_numbers_and_selection(self):
        for r in range(GRID_DIMENSION):
            for c in range(GRID_DIMENSION):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if self.selected_cell and self.selected_cell == (r, c):
                    pygame.draw.rect(self.screen, SELECTED_COLOR, rect)
                else:
                    pygame.draw.rect(self.screen, LIGHT_GRAY, rect, 1)

                num_initial = self.board_initial[r][c]
                num_current = self.board_current[r][c]
                num_overlay = self.board_solved_overlay[r][c]

                display_num_str = ""
                color = BLACK

                if num_current != 0:
                    display_num_str = str(num_current)
                    color = LIGHT_BLUE if num_initial != 0 else DARK_BLUE
                elif num_overlay != 0:
                    display_num_str = str(num_overlay)
                    color = GREEN_SOLVE
                
                if display_num_str:
                    text_surface = self.font_cell.render(display_num_str, True, color)
                    text_rect = text_surface.get_rect(center=(c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2))
                    self.screen.blit(text_surface, text_rect)

    def draw_buttons(self):
        pygame.draw.rect(self.screen, BUTTON_COLOR if not self.is_solving else (150,150,150), self.solve_button_rect, border_radius=5)
        solve_text_img = self.font_button.render("Solve", True, BUTTON_TEXT_COLOR)
        self.screen.blit(solve_text_img, (self.solve_button_rect.centerx - solve_text_img.get_width() // 2,
                                         self.solve_button_rect.centery - solve_text_img.get_height() // 2))

        pygame.draw.rect(self.screen, BUTTON_COLOR, self.reset_button_rect, border_radius=5)
        reset_text_img = self.font_button.render("Reset", True, BUTTON_TEXT_COLOR)
        self.screen.blit(reset_text_img, (self.reset_button_rect.centerx - reset_text_img.get_width() // 2,
                                         self.reset_button_rect.centery - reset_text_img.get_height() // 2))

        pygame.draw.rect(self.screen, BUTTON_COLOR, self.clear_button_rect, border_radius=5)
        clear_text_img = self.font_button.render("Clear All", True, BUTTON_TEXT_COLOR) 
        self.screen.blit(clear_text_img, (self.clear_button_rect.centerx - clear_text_img.get_width() // 2,
                                         self.clear_button_rect.centery - clear_text_img.get_height() // 2))

    def draw_message(self):
        if self.message_text:
            msg_surface = self.font_message.render(self.message_text, True, self.message_color)
            msg_rect = msg_surface.get_rect(center=(WIDTH // 2, GRID_SIZE + 75)) 
            self.screen.blit(msg_surface, msg_rect)

    def display_board_for_solver(self):
        self.screen.fill(WHITE)
        self.draw_numbers_and_selection()
        self.draw_grid_lines()
        self.draw_buttons()
        self.draw_message()
        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if not self.is_solving:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = pygame.mouse.get_pos()
                        if my < GRID_SIZE:
                            grid_r, grid_c = my // CELL_SIZE, mx // CELL_SIZE
                            if 0 <= grid_r < GRID_DIMENSION and 0 <= grid_c < GRID_DIMENSION:
                                self.selected_cell = (grid_r, grid_c)
                                self.message_text = ""
                        elif self.solve_button_rect.collidepoint(mx, my):
                            self.is_solving = True
                            self.message_text = "Solving 4x4..."
                            self.message_color = BLUE_MSG
                            self.board_solved_overlay = [[0]*GRID_DIMENSION for _ in range(GRID_DIMENSION)]
                            board_to_solve = [row[:] for row in self.board_current]
                            
                            self.display_board_for_solver()
                            pygame.display.flip()
                            
                            solve_result = solve_sudoku_recursive_visual_4x4(self.display_board_for_solver, board_to_solve)
                            
                            if solve_result == True:
                                for r in range(GRID_DIMENSION):
                                    for c in range(GRID_DIMENSION):
                                        if self.board_current[r][c] == 0:
                                            self.board_solved_overlay[r][c] = board_to_solve[r][c]
                                self.message_text = "4x4 Solved!"
                                self.message_color = GREEN_SOLVE
                            elif solve_result == "QUIT":
                                running = False
                            else:
                                self.message_text = "No solution for 4x4."
                                self.message_color = RED_MSG
                            self.is_solving = False

                        elif self.reset_button_rect.collidepoint(mx, my):
                            self.board_current = [row[:] for row in self.board_initial]
                            self.board_solved_overlay = [[0]*GRID_DIMENSION for _ in range(GRID_DIMENSION)]
                            self.selected_cell = None
                            self.message_text = "4x4 Board Reset."
                            self.message_color = BLUE_MSG
                        elif self.clear_button_rect.collidepoint(mx, my):
                            for r in range(GRID_DIMENSION):
                                for c in range(GRID_DIMENSION):
                                    
                                    if self.board_initial[r][c] == 0:
                                        self.board_current[r][c] = 0
                            self.board_solved_overlay = [[0]*GRID_DIMENSION for _ in range(GRID_DIMENSION)]
                            self.message_text = "User inputs cleared."
                            self.message_color = BLUE_MSG

                    if event.type == pygame.KEYDOWN:
                        if self.selected_cell:
                            r, c = self.selected_cell
                            if self.board_initial[r][c] == 0: 
                                if pygame.K_1 <= event.key <= pygame.K_4: 
                                    num_entered = event.key - pygame.K_0
                                    temp_board_check = [row[:] for row in self.board_current]
                                    temp_board_check[r][c] = num_entered
                                    
                                    if is_valid_solver_4x4(temp_board_check, num_entered, (r,c)):
                                         self.board_current[r][c] = num_entered
                                         self.board_solved_overlay[r][c] = 0
                                         self.message_text = ""
                                    else:
                                        self.board_current[r][c] = num_entered
                                        self.board_solved_overlay[r][c] = 0
                                        self.message_text = f"Warning: {num_entered} might conflict."
                                        self.message_color = RED_MSG

                                elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE or event.key == pygame.K_0:
                                    self.board_current[r][c] = 0
                                    self.board_solved_overlay[r][c] = 0
                                    self.message_text = ""
                        if event.key == pygame.K_ESCAPE:
                            self.selected_cell = None
                            self.message_text = ""
            
            self.screen.fill(WHITE)
            self.draw_numbers_and_selection()
            self.draw_grid_lines()
            self.draw_buttons()
            self.draw_message()
            
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

if __name__ == '__main__':
    game = SudokuPygame()
    game.run()