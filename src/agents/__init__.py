from .base import Agent, PassiveAgent, ActiveAgent


agents = {
    'human': None
}


def agent(cls):
    name = cls.NAME
    if name in agents.keys():
        raise Exception(f'Agent name was already taken: {name}')
    cls.agent_name = name
    agents[name] = cls
    return cls


from .random import RandomAgent
from .value_iteration import ValueIterAgent
from .mcts import MctsAgent
from .sarsa import SarsaAgent
from .expected_sarsa import ExpectedSarsaAgent
from .sarsa_lambda import SarsaLambdaAgent
from .q_learning import QLearningAgent
from .double_q_learning import DoubleQLearningAgent
from .value_approx import ValueApproximationAgent
