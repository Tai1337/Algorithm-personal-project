
class BaseMiniGame:
    def __init__(self, screen_width, screen_height, ui_manager=None):
        pass

    def start_game(self, training_episodes=0):
        pass

    def handle_event(self, event):
        pass

    def update(self, time_delta):
        pass

    def draw(self, screen, main_game_time_remaining):
        pass

    def cleanup_ui(self):
        pass