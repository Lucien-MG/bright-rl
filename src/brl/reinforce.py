import torch
import numpy as np

class Reinforce():

    def __init__(self, policy, gamma=0.99, learning_rate=1e-3):
        self.policy = policy
        self.gamma = gamma

        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=learning_rate)
        self.loss = 0

        self.memory = []

    def act(self, observation):
        return self.policy.act(observation)

    def observe(self, observation, action, reward, next_state):
        self.memory.append((observation, action, reward))

    def optimize(self):
        state_batch = torch.tensor(np.array([t[0] for t in self.memory]))
        action_batch = torch.Tensor(np.array([t[1] for t in self.memory])).to(dtype=torch.int64)
        reward_batch = torch.Tensor(np.array([t[2] for t in self.memory]))

        for t in reversed(range(len(reward_batch) - 1)):
            reward_batch[t] = reward_batch[t] + self.gamma * reward_batch[t + 1]

        reward_batch = torch.nn.functional.normalize(reward_batch, dim=0)

        probability_batch = torch.nn.functional.softmax(self.policy(state_batch), dim=0)

        self.loss = torch.mean(-torch.log(probability_batch.gather(dim=1, index=action_batch.unsqueeze(dim=0))) * reward_batch)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.memory = []
