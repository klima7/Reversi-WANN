from abc import ABC, abstractmethod

from .simulation import Simulation


class Backend(ABC):

    def __init__(self, size):
        self._size = size

    @abstractmethod
    def get_all_possible_boards_numbers(self):
        pass

    @abstractmethod
    def get_moves(self, board, turn):
        pass

    @abstractmethod
    def make_move(self, board, turn, move):
        pass

    @abstractmethod
    def get_winner(self, board):
        pass

    def _generate_all_possible_boards(self):
        boards = set()
        simulations = {Simulation.create_initial(self._size, LiveBackend(self._size))}

        while simulations:
            simulation = simulations.pop()

            if simulation in simulations:
                continue

            if simulation.is_finished():
                boards.update([simulation.board_view, simulation.opposite_board_view])
            else:
                boards.add(simulation.board_view)

            for move in simulation.get_moves():
                next_simulation = simulation.copy().make_move(move)
                simulations.add(next_simulation)

        return boards


class LiveBackend(Backend):

    def __init__(self, size):
        super().__init__(size)
        self.__boards_numbers = None

    def get_all_possible_boards_numbers(self):
        if self.__boards_numbers is None:
            self.__boards_numbers = tuple(board.number for board in self._generate_all_possible_boards())
        return self.__boards_numbers

    def get_moves(self, board, turn):
        moves_array = board.get_legal_moves(turn)
        return tuple(map(tuple, moves_array))

    def make_move(self, board, turn, move):
        new_board = board.make_move(move, turn)
        new_turn = -turn if board.has_any_moves(-turn) else turn
        return new_board, new_turn

    def get_winner(self, board):
        if board.is_finished():
            return board.get_winner()
        return None
