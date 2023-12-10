import numpy as np
import pandas as pd
import random, os, datetime, math
from collections import defaultdict

total_reward = 0
bandit_dict = {}

def set_seed(my_seed=42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)

def calculate_expectation(bandit_stats):
    """Calculate the expectation value for a given bandit."""
    wins, losses, opps = bandit_stats['win'], bandit_stats['loss'], bandit_stats['opp']
    return ((wins - 0.8 * losses + opps - (opps > 0) * 1.5) /
            (0.7 * wins + losses + opps) *
            math.pow(0.965, wins + losses + opps))

def get_next_bandit():
    """Determine the next bandit to choose based on expectations."""
    best_bandit, best_bandit_expected = 0, 0
    for bnd, stats in bandit_dict.items():
        expect = calculate_expectation(stats)
        if expect > best_bandit_expected:
            best_bandit_expected = expect
            best_bandit = bnd
    return best_bandit

my_action_list = []
op_action_list = []

def update_bandit_stats(last_reward, my_last_action, op_last_action):
    """Update the statistics of the bandits based on the last actions and reward."""
    if last_reward > 0:
        bandit_dict[my_last_action]['win'] += 1
    else:
        bandit_dict[my_last_action]['loss'] += 1
    bandit_dict[op_last_action]['opp'] += 1

def check_continuation(my_last_action, op_last_action):
    """Check and update the continuation stats for my and opponent's last action."""
    if len(my_action_list) >= 3 and my_action_list[-1] == my_action_list[-2]:
        bandit_dict[my_last_action]['my_continue'] += 1
    else:
        bandit_dict[my_last_action]['my_continue'] = 0

    if len(op_action_list) >= 3 and op_action_list[-1] == op_action_list[-2]:
        bandit_dict[op_last_action]['op_continue'] += 1
    else:
        bandit_dict[op_last_action]['op_continue'] = 0

def multi_armed_probabilities(observation, configuration):
    global total_reward, bandit_dict

    if observation['step'] == 0:
        set_seed()
        total_reward = 0
        bandit_dict = {i: {'win': 1, 'loss': 0, 'opp': 0, 'my_continue': 0, 'op_continue': 0}
                       for i in range(configuration['banditCount'])}
    else:
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']

        my_idx = observation['agentIndex']
        my_last_action = observation['lastActions'][my_idx]
        op_last_action = observation['lastActions'][1 - my_idx]

        my_action_list.append(my_last_action)
        op_action_list.append(op_last_action)

        update_bandit_stats(last_reward, my_last_action, op_last_action)
        check_continuation(my_last_action, op_last_action)

        if last_reward > 0:
            my_pull = my_last_action
        else:
            if len(my_action_list) >= 4 and my_action_list[-1] == my_action_list[-2] == my_action_list[-3]:
                my_pull = my_last_action if random.random() < 0.5 else get_next_bandit()
            else:
                my_pull = get_next_bandit()
    return my_pull

