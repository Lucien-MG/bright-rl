from copy import deepcopy
from random import sample
from collections import deque

import numpy as np

import torch


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:

    def __init__(
        self,
        policy,
        epsilon=0.9,
        epsilon_decay=4e-5,
        gamma=0.999,
        tau=0.005,
        learning_rate=1e-4,
        batch_size=32,
    ):
        self.policy = policy

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.optimizer = torch.optim.SGD(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.loss = 0

        self.replay_memory = ReplayMemory(capacity=10000)

    @torch.no_grad()
    def act(self, observation):
        observation = torch.from_numpy(observation)
        rand_choice = torch.torch.rand(1).item()

        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), 0.02)

        if rand_choice > self.epsilon:
            return torch.argmax(self.policy(observation)).item()
        else:
            high = self.policy(observation).size()[0]
            return torch.randint(low=0, high=high, size=(1, 1)).item()

    def observe(self, observation, action, reward, next_observtion, terminated):
        self.replay_memory.push((observation, action, reward, next_observtion))

        if len(self.replay_memory) >= self.batch_size:
            self.optimize()

    def optimize(self):
        transitions = self.replay_memory.sample(self.batch_size)

        state_batch = torch.tensor(np.array([t[0] for t in transitions]))
        action_batch = torch.Tensor(np.array([t[1] for t in transitions])).to(
            dtype=torch.int64
        )
        reward_batch = torch.Tensor(np.array([t[2] for t in transitions]))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, [t[3] for t in transitions])),
            dtype=torch.bool,
        )
        non_final_next_states = torch.Tensor(
            np.array([t[3] for t in transitions if t[3] is not None])
        )

        state_action_values_batch = self.policy(state_batch).gather(
            1, action_batch.unsqueeze(dim=0)
        )

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)

        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.policy(non_final_next_states).max(1).values
            )

        # Compute the expected Q values
        reward_batch = (reward_batch - reward_batch.mean()) / (
            reward_batch.std() + 1e-12
        )
    
        expected_state_action_values_batch = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.MSELoss()
        self.loss = criterion(
            state_action_values_batch,
            expected_state_action_values_batch.unsqueeze(dim=0),
        )

        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()

        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 10)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # target_net_state_dict = self.target.state_dict()
        # policy_net_state_dict = self.policy.state_dict()
        # for key in policy_net_state_dict:
        #    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        # self.target.load_state_dict(target_net_state_dict)
