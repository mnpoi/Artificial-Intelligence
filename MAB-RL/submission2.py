import json
import numpy as np
import pandas as pd

bandit_state = None  # 多臂赌博机的状态
total_reward = 0  # 总奖励
last_step = None  # 上一步的动作
epsilon = 0.2  # 探索率，控制探索和开发的平衡
alpha = 0.1  # 学习率，控制Beta分布参数的更新速度
gamma = 0.9  # 折扣率，考虑未来奖励的权重

def multi_armed_bandit_agent(observation, configuration):
    global bandit_state, total_reward, last_step

    step = 1.0  # 通过此参数可以调整探索/开发平衡
    decay_rate = 0.97  # 每次调用后我们衰减赢的次数的比率

    if observation.step == 0:
        # 初始化多臂赌博机的状态
        bandit_state = [[1, 1] for i in range(configuration["banditCount"])]
    else:
        # 使用上一步的结果更新多臂赌博机的状态
        last_reward = observation["reward"] - total_reward
        total_reward = observation["reward"]

        # 我们需要了解自己是玩家1还是玩家2
        player = int(last_step == observation.lastActions[1])

        if last_reward > 0:
            # 使用Q-learning更新Beta分布的参数
            bandit_state[observation.lastActions[player]][0] += last_reward * step
            bandit_state[observation.lastActions[player]][0] = (1 - alpha) * bandit_state[observation.lastActions[player]][0] + alpha * (last_reward + gamma * np.max(bandit_state, axis=0)[0])
        else:
            bandit_state[observation.lastActions[player]][1] += step

        bandit_state[observation.lastActions[0]][0] = (bandit_state[observation.lastActions[0]][0] - 1) * decay_rate + 1
        bandit_state[observation.lastActions[1]][0] = (bandit_state[observation.lastActions[1]][0] - 1) * decay_rate + 1

    # 为每个代理生成来自Beta分布的随机数，并选择最幸运的一个
    best_proba = -1
    best_agent = None
    for k in range(configuration["banditCount"]):
        proba = np.random.beta(bandit_state[k][0], bandit_state[k][1])
        if proba > best_proba:
            best_proba = proba
            best_agent = k

    last_step = best_agent
    return int(best_agent)  # Ensure that the returned action is an integer
