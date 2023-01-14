import time
import signal
from abc import ABC, abstractmethod

import pygame

from simulation import Simulation
from environment import Environment
from board import Color
from exceptions import DomainException
from agents import PassiveAgent, ActiveAgent


class Gameplay(ABC):

    def __init__(self, size, delay, backend):
        self._size = size
        self._delay = delay

        self._simulation = Simulation.create_initial(size, backend)
        self._env = Environment(size, backend)

        self._player_black = None
        self._player_white = None

    def set_players(self, player_black, player_white):
        self._player_black = player_black
        self._player_white = player_white

        self.__config_player(player_black)
        self.__config_player(player_white)

    def swap_players(self):
        self._player_black, self._player_white = self._player_white, self._player_black

    def play(self):
        self.__before_gameplay()
        self._play()
        self.__after_gameplay()
        return self.__get_winner()

    def reset(self):
        self._simulation.reset()

    def dispose(self):
        pass

    @abstractmethod
    def _play(self):
        pass

    def _get_decisive_player(self):
        return self._player_white if self._simulation.turn == Color.WHITE else self._player_black

    def _get_state_for_player(self, player):
        color = Color.BLACK if player == self._player_black else Color.WHITE
        return self._env.cvt_board_to_state(self._simulation.board.to_relative(color))

    def _make_move(self, action):
        moving_player = self._get_decisive_player()

        # remember action and state to notify agent about action result later
        if moving_player is not None:
            state = self._get_state_for_player(moving_player)
            moving_player.last_state = state
            moving_player.last_action = action

        # just move
        self._simulation.make_move(action)

        # notify agents about previous move result
        if self._simulation.is_finished():
            self.__update_agent(self._player_black)
            self.__update_agent(self._player_white)
        else:
            self.__update_agent(self._get_decisive_player())

    def __get_winner(self):
        winner_color = self._simulation.get_winner()
        if winner_color == Color.WHITE:
            return self._player_white
        elif winner_color == Color.BLACK:
            return self._player_black
        else:
            return None

    def __config_player(self, player):
        if isinstance(player, PassiveAgent):
            player.env = self._env
        elif isinstance(player, ActiveAgent):
            player.get_possible_actions = self._env.get_possible_actions

        if player is not None:
            player.initialize()

    def __before_gameplay(self):
        if self._player_black is not None:
            self._player_black.before_gameplay()
        if self._player_white is not None:
            self._player_white.before_gameplay()

    def __after_gameplay(self):
        if self._player_black is not None:
            self._player_black.after_gameplay()
        if self._player_white is not None:
            self._player_white.after_gameplay()

    def __update_agent(self, player):
        if player is not None and player.last_action is not None:
            state = self._get_state_for_player(player)
            reward = self._env.get_reward(player.last_state, player.last_action, state)
            player.update(player.last_state, player.last_action, reward, state)


class NoGuiGameplay(Gameplay):

    def set_players(self, player_black, player_white):
        if None in [player_black, player_white]:
            raise DomainException('Human players are not allowed in games without GUI')
        super().set_players(player_black, player_white)

    def _play(self):
        while not self._simulation.is_finished():
            player = self._get_decisive_player()
            state = self._get_state_for_player(player)
            action = player.get_action(state)
            self._make_move(action)


class GuiGameplay(Gameplay):

    FIELD_SIZE = 100
    DISC_SIZE = 80

    WHITE_COLOR = (255, 255, 255)
    BLACK_COLOR = (0, 0, 0)
    MIDDLE_COLOR = (127, 127, 127)
    BOARD_COLOR = (0, 127, 0)
    LINES_COLOR = (0, 150, 0)
    TEXT_COLOR = (0, 0, 0)

    def __init__(self, size, delay, backend):
        super().__init__(size, delay, backend)

        self.__running = True
        self.__screen = None

        self.__turn_font = None
        self.__winner_font = None

        self.__last_move = None
        self.__pending_move = None

        self.__before_move_time = None
        self.__after_move_time = None
        self.__finish_time = None

    def reset(self):
        super().reset()
        self.__running = True
        self.__last_move = None
        self.__pending_move = None

        self.__before_move_time = None
        self.__after_move_time = None
        self.__finish_time = None

    def dispose(self):
        pygame.quit()
        self.__screen = None

    # ----------------- update logic stuff ---------------------

    def _play(self):
        self.__init_gui_if_needed()

        while self.__should_run():
            self.__collect_events()
            self.__update()
            self.__draw_screen()

    def __should_run(self):
        in_progress = not self._simulation.is_finished()
        cooldown_not_elapsed = self.__finish_time is not None and self.__finish_time + self._delay > time.time()
        return (in_progress or cooldown_not_elapsed) and self.__running

    def __update(self):
        if not self._simulation.is_finished():
            self.__update_move()
            if self._simulation.is_finished():
                self.__finish_time = time.time()

    def __update_move(self):
        current_time = time.time()

        # get move
        if self.__pending_move is None:
            decisive_player = self._get_decisive_player()
            self.__pending_move = self.__get_move_from_player(decisive_player)
            self.__before_move_time = current_time

        # draw move
        if self.__before_move_time is not None and self.__before_move_time + self._delay < current_time:
            self.__last_move = self.__pending_move
            self.__before_move_time = None
            self.__after_move_time = current_time

        # apply move
        if self.__after_move_time is not None and self.__after_move_time + self._delay < current_time:
            self._make_move(self.__pending_move)
            self.__pending_move = None
            self.__after_move_time = None

    def __get_move_from_player(self, player):
        if player is None:
            return self.__get_move_from_real_player()
        else:
            return self.__get_move_from_artificial_player(player)

    def __get_move_from_artificial_player(self, player):
        state = self._get_state_for_player(player)
        return player.get_action(state)

    def __get_move_from_real_player(self):
        possible_moves = self._simulation.get_moves()
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            mouse_pos = pygame.mouse.get_pos()
            move_pos = (mouse_pos[1] // self.FIELD_SIZE, mouse_pos[0] // self.FIELD_SIZE)
            if move_pos in possible_moves:
                return move_pos
            return None
        return None

    # ----------------- pure GUI stuff ---------------------

    def __init_gui_if_needed(self):
        if not pygame.get_init():
            pygame.init()

            screen_width = self._size[1] * self.FIELD_SIZE
            screen_height = self._size[0] * self.FIELD_SIZE + 40
            self.__screen = pygame.display.set_mode([screen_width, screen_height])
            pygame.display.set_caption(f'Reversi {self._size[0]}x{self._size[1]}')

            self.__turn_font = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.__winner_font = pygame.font.Font(pygame.font.get_default_font(), 40)

    def __collect_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False
                signal.raise_signal(signal.SIGINT)

    def __draw_screen(self):
        if self._simulation.is_finished():
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
        winner = self._simulation.get_winner()
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
                         (0, 0, self._size[1] * self.FIELD_SIZE, self._size[0] * self.FIELD_SIZE), width=3)
        for y in range(1, self._size[0]):
            pygame.draw.line(self.__screen, self.LINES_COLOR, (0, y * self.FIELD_SIZE),
                             (self._size[1] * self.FIELD_SIZE, y * self.FIELD_SIZE), width=3)
        for x in range(1, self._size[1]):
            pygame.draw.line(self.__screen, self.LINES_COLOR, (x * self.FIELD_SIZE, 0),
                             (x * self.FIELD_SIZE, self._size[0] * self.FIELD_SIZE), width=3)

    def __draw_discs(self):
        possible_moves = self._simulation.get_moves()
        disc_center_offset = self.FIELD_SIZE // 2
        for y in range(self._size[0]):
            for x in range(self._size[1]):
                disc_color = self._simulation.board[y, x]
                pos = (x * self.FIELD_SIZE + disc_center_offset, y * self.FIELD_SIZE + disc_center_offset)
                if disc_color == Color.ANY and (y, x) in possible_moves:
                    color = self.WHITE_COLOR if self._simulation.turn == Color.WHITE else self.BLACK_COLOR
                    pygame.draw.circle(self.__screen, color, pos, self.DISC_SIZE // 2, width=3)
                elif disc_color != Color.ANY:
                    color = self.WHITE_COLOR if disc_color == Color.WHITE else self.BLACK_COLOR
                    pygame.draw.circle(self.__screen, color, pos, self.DISC_SIZE // 2)

    def __draw_last_move(self):
        if self.__last_move is not None:
            y, x = self.__last_move
            x_pos, y_pos = x * self.FIELD_SIZE + self.FIELD_SIZE // 2, y * self.FIELD_SIZE + self.FIELD_SIZE // 2
            pygame.draw.line(self.__screen, (255, 0, 0), (x_pos, 0), (x_pos, self._size[0] * self.FIELD_SIZE), width=2)
            pygame.draw.line(self.__screen, (255, 0, 0), (0, y_pos), (self._size[1] * self.FIELD_SIZE, y_pos), width=2)

    def __draw_turn(self):
        color = self.WHITE_COLOR if self._simulation.turn == Color.WHITE else self.BLACK_COLOR
        name = self.__get_player_name(self._get_decisive_player())

        pygame.draw.circle(self.__screen, color, (20, self.__screen.get_height() - 21), 11)
        turn_text = self.__turn_font.render(name, True, self.TEXT_COLOR)
        self.__screen.blit(turn_text, (40, self.__screen.get_height() - 30))

    def __get_winner_text(self, winner):
        if winner == Color.BLACK:
            return self.__get_player_name(self._player_black)
        elif winner == Color.WHITE:
            return self.__get_player_name(self._player_white)
        else:
            return 'draw'

    def __get_winner_color(self, winner):
        if winner == Color.BLACK:
            return self.BLACK_COLOR
        elif winner == Color.WHITE:
            return self.WHITE_COLOR
        else:
            return self.MIDDLE_COLOR

    @staticmethod
    def __get_player_name(player):
        if player is None:
            return 'Human'
        else:
            return player.NAME.replace('_', ' ').upper()
