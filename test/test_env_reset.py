from collections.abc import Callable
import logging
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th
import torch.optim as optim

import regawa.wrappers.gym_utils as model_utils
from regawa.gnn import ActionMode, AgentConfig, GraphAgent
from regawa.gnn.agent_utils import GNNParams
from regawa.gnn.gnn_agent import heterostatedata_to_tensors
from vejde_rddl import register_env
from regawa.rl.util import evaluate, rollout, save_eval_data, update, update_vf_agent
from regawa import agent_from_env


def test_env_reset():
    domain = "Navigation_MDP_ippc2011"
    instance = 1
    remove_false = True
    env_id = register_env()
    env_func = lambda: gym.make(
        env_id, domain=domain, instance=instance, remove_false=remove_false
    )

    env = gym.vector.SyncVectorEnv([env_func] * 2)

    o, i = env.reset()

    obs = []
    rewards = []
    dones = []
    actions = []

    obs.append(o)
    rewards.append(0)
    dones.append(False)

    episode_length = 40
    num_steps = episode_length

    for step in range(num_steps):
        a = env.action_space.sample()
        actions.append(a)
        next_o, r, term, trunc, i = env.step(a)
        done = term | trunc

        obs.append(next_o)
        rewards.append(r)
        dones.append(done)

    assert dones[-1].all(), "The last step should be a terminal state"

    o, i = env.reset()
    obs = []
    rewards = []
    dones = []
    actions = []
    num_steps = episode_length * 2

    obs.append(o)
    rewards.append(0)
    dones.append(False)

    for step in range(num_steps):
        a = env.action_space.sample()
        actions.append(a)
        next_o, r, term, trunc, i = env.step(a)
        done = term | trunc

        obs.append(next_o)
        rewards.append(r)
        dones.append(done)

    assert dones[-1].all()

    pass


def example():
    import gymnasium as gym
    import numpy as np
    from collections import deque

    # Initialize environment, buffer and episode_start

    domain = "Navigation_MDP_ippc2011"
    instance = 1
    remove_false = True
    env_id = register_env()
    env_func = lambda: gym.make(
        env_id, domain=domain, instance=instance, remove_false=remove_false
    )

    episode_length = 40
    num_steps = 78
    num_envs = 1
    offset = num_steps // episode_length
    envs = gym.vector.SyncVectorEnv([env_func] * num_envs)
    replay_buffer = deque()
    episode_start = np.zeros(envs.num_envs, dtype=bool)

    observations, _ = envs.reset()
    for _ in range(num_steps + offset):  # Training loop
        actions = envs.action_space.sample()
        next_observations, rewards, terminations, truncations, infos = envs.step(
            actions
        )

        # Add to replay buffer
        for i in range(envs.num_envs):
            if not episode_start[i]:
                replay_buffer.append(
                    (
                        observations[i],
                        actions[i],
                        rewards[i],
                        terminations[i],
                        truncations[i],
                        next_observations[i],
                    )
                )
            else:
                pass

        # update observation and if episode starts
        observations = next_observations
        episode_start = np.logical_or(terminations, truncations)
    envs.close()


if __name__ == "__main__":
    example()
