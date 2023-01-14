import random

import numpy as np

from .simulation import Simulation
from .backend import LiveBackend
from .board import Color
from .environment import Environment


class ReversiEnv:

    def __init__(self, size):
        self.__size = size
        self.__simulation = Simulation.create_initial(size, LiveBackend(size))
        self.__env = Environment(size, backend)
        self.__color = None
        self.reset()

    def reset(self):
        self.__simulation.reset()
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

        self.__move_opponent()

        return self.__get_state(), self.__get_reward(), self.__is_done(), {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def __move_opponent(self):
        while not self.__simulation.is_finished() and self.__simulation.turn == -self.__color:
            action = random.choice(self.__simulation.get_moves())
            self.__simulation.make_move(action)

    def __get_state(self):
        return self.__simulation.board.to_relative(self.__color).to_vector()

    def __get_reward(self):
        if not self.__simulation.is_finished():
            return 0

        return 1 if self.__simulation.get_winner() == self.__color else 0

        player_count = self.__simulation.board.get_discs_count(self.__color)
        opponent_count = self.__simulation.board.get_discs_count(-self.__color)

        return player_count - opponent_count

    def __is_done(self):
        return self.__simulation.is_finished()


class SmallReversiEnv(ReversiEnv):
    def __init__(self):
        super().__init__(size=(6, 6))
