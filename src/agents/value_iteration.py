import numpy as np
from tqdm import tqdm

from . import PassiveAgent, agent
from exceptions import DomainException


@agent
class ValueIterAgent(PassiveAgent):

    NAME = 'value_iter'
    DEFAULT_GAMMA = 0.95
    DEFAULT_THETA = 1e-4

    def __init__(self, gamma=DEFAULT_GAMMA, theta=DEFAULT_THETA):
        super().__init__()

        self.__gamma = gamma
        self.__theta = theta
        self.__policy = None

    def initialize(self):
        super().initialize()
        if self.learn:
            print('learning policy...')
            self.__policy = self.__learn_policy(self.__gamma, self.__theta)
        if self.__policy is None:
            raise DomainException('ValueIterAgent must learn policy first')

    def get_action(self, state):
        return self.__policy[state]

    def get_data_to_save(self):
        return self.__policy

    def set_saved_data(self, data):
        self.__policy = data

    def __learn_policy(self, gamma, theta):
        values = dict()
        policy = dict()

        for current_state in tqdm(self.env.get_all_states(), desc='Values initialization'):
            values[current_state] = 0
            policy[current_state] = 0

        while True:
            values_prev = dict(values)
            for s in tqdm(self.env.get_all_states(), desc='Value iteration'):
                actions_values = []
                for a in self.env.get_possible_actions(s):
                    action_value = 0
                    for s_prim, p in self.env.get_next_states(s, a).items():
                        r = self.env.get_reward(s, a, s_prim)
                        action_value += p * (r + gamma * values[s_prim])
                    actions_values.append(action_value)

                if actions_values:
                    values[s] = max(actions_values)

            if ValueIterAgent.__should_stop_learning(values, values_prev, theta):
                break

        return self.__create_policy(values, gamma)

    @staticmethod
    def __should_stop_learning(values1, values2, theta):
        v1_values = np.array(list(values1.values()))
        v2_values = np.array(list(values2.values()))
        diff = np.abs(v1_values - v2_values)
        min_diff = np.max(diff)
        return min_diff < theta

    def __create_policy(self, value_function, gamma):
        policy = {}

        for state in tqdm(self.env.get_all_states(), desc='Creating policy'):
            actions = self.env.get_possible_actions(state)

            if len(actions) == 0:
                continue

            actions_values = []

            for action in actions:
                action_value = 0
                for new_state, probability in self.env.get_next_states(state, action).items():
                    reward = self.env.get_reward(state, action, new_state)
                    action_value += probability * (reward + gamma * value_function[new_state])
                actions_values.append(action_value)

            best_action_index = np.argmax(actions_values)
            best_action = actions[best_action_index]
            policy[state] = best_action

        return policy
