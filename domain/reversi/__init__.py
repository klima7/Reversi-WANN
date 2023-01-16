import random
import time

import numpy as np
import pygame

from .simulation import Simulation
from .backend import LiveBackend
from .board import Color
from .environment import Environment


class ReversiEnv:

    FIELD_SIZE = 100
    DISC_SIZE = 80

    WHITE_COLOR = (255, 255, 255)
    BLACK_COLOR = (0, 0, 0)
    MIDDLE_COLOR = (127, 127, 127)
    BOARD_COLOR = (0, 127, 0)
    LINES_COLOR = (0, 150, 0)
    TEXT_COLOR = (0, 0, 0)

    def __init__(self, size, delay=0.2):
        self.__size = size
        self.__delay = delay
        self.__simulation = Simulation.create_initial(size, LiveBackend(size))
        self.__env = Environment(size, backend)
        self.__color = None
        self.__last_move = None
        self.__screen = None
        self.__show_gui = False
        self.reset()

    def reset(self):
        self.__simulation.reset()
        self.__last_move = None
        self.__color = random.choice([Color.WHITE, Color.BLACK])
        self.__move_opponent()
        return self.__get_state()

    def step(self, nn_predictions):
        assert self.__simulation.turn == self.__color

        nn_predictions = nn_predictions.reshape(self.__size)

        legal_moves = self.__simulation.get_moves()
        legal_moves_matrix = np.zeros(self.__size)
        for move in legal_moves:
            legal_moves_matrix[move[0], move[1]] = 1

        moves_values = nn_predictions * legal_moves_matrix
        best_value = np.max(moves_values)
        best_moves = np.argwhere(moves_values == best_value)
        best_move = random.choice(best_moves)
        self.__simulation.make_move(best_move)
        self.__last_move = best_move

        self.__move_opponent()

        return self.__get_state(), self.__get_reward(), self.__is_done(), {}

    def render(self, mode='human', close=False):
        self.__show_gui = True
        self.__init_gui_if_needed()
        for _ in pygame.event.get():
            continue
        self.__draw_screen()
        time.sleep(self.__delay)

    def close(self):
        pygame.quit()
        self.__screen = None

    def seed(self, seed=None):
        pass

    def __move_opponent(self):
        while not self.__simulation.is_finished() and self.__simulation.turn == -self.__color:
            if self.__show_gui:
                self.__draw_screen()
                time.sleep(self.__delay)

            action = random.choice(self.__simulation.get_moves())
            self.__simulation.make_move(action)
            self.__last_move = action

    def __get_state(self):
        return self.__simulation.board.to_relative(self.__color).to_vector()

    def __get_reward(self):
        if not self.__simulation.is_finished():
            return 0

        # Uncomment after training to interpret fitness as winning rate
        # return 1 if self.__simulation.get_winner() == self.__color else 0

        player_count = self.__simulation.board.get_discs_count(self.__color)
        opponent_count = self.__simulation.board.get_discs_count(-self.__color)

        return player_count - opponent_count

    def __is_done(self):
        return self.__simulation.is_finished()

    # ----------------- pure GUI stuff ---------------------

    def __init_gui_if_needed(self):
        if not pygame.get_init():
            pygame.init()

            screen_width = self.__size[1] * self.FIELD_SIZE
            screen_height = self.__size[0] * self.FIELD_SIZE + 40
            self.__screen = pygame.display.set_mode([screen_width, screen_height])
            pygame.display.set_caption(f'Reversi {self.__size[0]}x{self.__size[1]}')

            self.__turn_font = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.__winner_font = pygame.font.Font(pygame.font.get_default_font(), 40)

    def __draw_screen(self):
        if self.__simulation.is_finished():
            self.__draw_finish_screen()
        else:
            self.__draw_standard_screen()

        pygame.display.flip()

    def __draw_standard_screen(self):
        self.__draw_board()
        self.__draw_discs()
        self.__draw_last_move()
        self.__draw_turn()

    def __draw_finish_screen(self):
        winner = self.__simulation.get_winner()
        text = self.__get_winner_text(winner)
        color = self.__get_winner_color(winner)

        text_surface = self.__winner_font.render(text, True, self.TEXT_COLOR)
        text_pos = (
            (self.__screen.get_width() - text_surface.get_width()) // 2,
            (self.__screen.get_height() - text_surface.get_height()) // 2
        )
        circle_pos = (self.__screen.get_width() // 2, self.__screen.get_height() // 2 + 60)

        self.__screen.fill(self.BOARD_COLOR)
        self.__screen.blit(text_surface, text_pos)
        pygame.draw.circle(self.__screen, color, circle_pos, 20)

    def __draw_board(self):
        self.__screen.fill(self.BOARD_COLOR)
        pygame.draw.rect(self.__screen, self.LINES_COLOR,
                         (0, 0, self.__size[1] * self.FIELD_SIZE, self.__size[0] * self.FIELD_SIZE), width=3)
        for y in range(1, self.__size[0]):
            pygame.draw.line(self.__screen, self.LINES_COLOR, (0, y * self.FIELD_SIZE),
                             (self.__size[1] * self.FIELD_SIZE, y * self.FIELD_SIZE), width=3)
        for x in range(1, self.__size[1]):
            pygame.draw.line(self.__screen, self.LINES_COLOR, (x * self.FIELD_SIZE, 0),
                             (x * self.FIELD_SIZE, self.__size[0] * self.FIELD_SIZE), width=3)

    def __draw_discs(self):
        possible_moves = self.__simulation.get_moves()
        disc_center_offset = self.FIELD_SIZE // 2
        for y in range(self.__size[0]):
            for x in range(self.__size[1]):
                disc_color = self.__simulation.board[y, x]
                pos = (x * self.FIELD_SIZE + disc_center_offset, y * self.FIELD_SIZE + disc_center_offset)
                if disc_color == Color.ANY and (y, x) in possible_moves:
                    color = self.WHITE_COLOR if self.__simulation.turn == Color.WHITE else self.BLACK_COLOR
                    pygame.draw.circle(self.__screen, color, pos, self.DISC_SIZE // 2, width=3)
                elif disc_color != Color.ANY:
                    color = self.WHITE_COLOR if disc_color == Color.WHITE else self.BLACK_COLOR
                    pygame.draw.circle(self.__screen, color, pos, self.DISC_SIZE // 2)

    def __draw_last_move(self):
        if self.__last_move is not None:
            y, x = self.__last_move
            x_pos, y_pos = x * self.FIELD_SIZE + self.FIELD_SIZE // 2, y * self.FIELD_SIZE + self.FIELD_SIZE // 2
            pygame.draw.line(self.__screen, (255, 0, 0), (x_pos, 0), (x_pos, self.__size[0] * self.FIELD_SIZE), width=2)
            pygame.draw.line(self.__screen, (255, 0, 0), (0, y_pos), (self.__size[1] * self.FIELD_SIZE, y_pos), width=2)

    def __draw_turn(self):
        color = self.WHITE_COLOR if self.__simulation.turn == Color.WHITE else self.BLACK_COLOR
        name = 'WANN' if self.__simulation.turn == self.__color else 'Opponent'

        pygame.draw.circle(self.__screen, color, (20, self.__screen.get_height() - 21), 11)
        turn_text = self.__turn_font.render(name, True, self.TEXT_COLOR)
        self.__screen.blit(turn_text, (40, self.__screen.get_height() - 30))

    def __get_winner_text(self, winner):
        if winner == self.__color:
            return 'WANN'
        elif winner == -self.__color:
            return 'Opponent'
        else:
            return 'Draw'

    def __get_winner_color(self, winner):
        if winner == Color.BLACK:
            return self.BLACK_COLOR
        elif winner == Color.WHITE:
            return self.WHITE_COLOR
        else:
            return self.MIDDLE_COLOR
