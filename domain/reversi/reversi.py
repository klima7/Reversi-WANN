from pathlib import Path
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import click
import numpy as np

from agents import agents
from gameplay import GuiGameplay, NoGuiGameplay, Tournament
from backend import LiveBackend, PreparedBackend
from exceptions import DomainException


@click.command(help="Runs Reversi game of given size, given number of times, with selected players, "
                    "which are learning or not, with or without GUI and returns wins count")
@click.argument('p1', type=click.Choice(list(agents.keys())), default='human')
@click.argument('p2', type=click.Choice(list(agents.keys())), default='human')
@click.option('-l1', is_flag=True, default=False, help='Enable learning for first player')
@click.option('-l2', is_flag=True, default=False, help='Enable learning for second player')
@click.option('-s', '--size', nargs=2, type=int, default=(8, 8), help='Size of the map')
@click.option('-n', '--number', type=int, default=1, help='Number of game repeats')
@click.option('-d', '--delay', type=float, default=0.05, help='Minimum delay between player moves in ms')
@click.option('--live/--prepared', default=True, help='Whether use live or prepared backend')
@click.option('--gui/--nogui', default=True, help='Whether graphical interface should be shown')
def reversi(p1, p2, l1, l2, size, number, delay, live, gui):
    player1 = construct_agent(p1, l1, size)
    player2 = construct_agent(p2, l2, size)

    backend = LiveBackend(size) if live else PreparedBackend(size, get_path_to_backend_data(size))

    gameplay_class = GuiGameplay if gui else NoGuiGameplay
    gameplay = gameplay_class(size, delay, backend)

    tournament = Tournament(gameplay, number, player1, player2)
    results = tournament.play()
    percent_results = np.array(results) / np.sum(results) * 100

    print('------------RESULTS------------')
    print(f'  Player1 ({p1}) wins: {results[0]} ({percent_results[0]:.1f}%)')
    print(f'  Player2 ({p2}) wins: {results[1]} ({percent_results[1]:.1f}%)')
    print(f'  Draws: {results[2]} ({percent_results[2]:.1f}%)')
    print('-------------------------------')

    save_agent_data(player1, size)
    save_agent_data(player2, size)


def construct_agent(name, learn, size):
    agent_class = agents[name]

    if agent_class is None:     # real human - special case
        return None

    agent = agent_class()
    agent.learn = learn

    path_to_agent_data = get_path_to_agent_data(size, agent.NAME)
    agent.load_data(path_to_agent_data)

    return agent


def save_agent_data(agent, size):
    if agent is None or agent.learn is False:
        return
    path_to_agent_data = get_path_to_agent_data(size, agent.NAME)
    agent.save_data(path_to_agent_data)


def get_path_to_agent_data(size, agent_name):
    root_path = Path(__file__).parent.parent
    size_directory = f'{size[0]}x{size[1]}'
    filename = f'{agent_name}.pickle'
    directory = root_path / 'res' / size_directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def get_path_to_backend_data(size):
    root_path = Path(__file__).parent.parent
    size_directory = f'{size[0]}x{size[1]}'
    directory = root_path / 'res' / size_directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory / 'data.pickle'


if __name__ == '__main__':
    try:
        reversi()
    except DomainException as e:
        print(f'ERROR: {e.message}', file=sys.stderr)
