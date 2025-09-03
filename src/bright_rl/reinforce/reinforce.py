import torch
import numpy as np

class Reinforce():

    def __init__(self, policy, gamma=0.99, nb_episodes=4, optimizer=torch.optim.SGD, optimizer_parameters={"lr": 3e-4}):
        self.policy = policy
        self.gamma = gamma
        self.nb_episodes = nb_episodes

        self.optimizer = optimizer(self.policy.parameters(), **optimizer_parameters)

        self.loss = 0

        self.memory = []
        self.episodes_in_memory = 0
    
    @torch.inference_mode()
    def act(self, state):
        state = torch.from_numpy(state)
        probabilities = torch.nn.functional.softmax(self.policy(state), dim=0)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action

    def observe(self, observation, action, reward, next_state, terminated):
        self.memory.append((observation, action, reward, terminated))
        self.episodes_in_memory += terminated

        if self.episodes_in_memory == self.nb_episodes:
            self.optimize()

            self.memory = []
            self.episodes_in_memory = 0

    def optimize(self):
        state_batch = torch.as_tensor(np.array([t[0] for t in self.memory])).to(dtype=torch.float32)
        action_batch = torch.as_tensor(np.array([t[1] for t in self.memory])).to(dtype=torch.int64)
        reward_batch = torch.as_tensor(np.array([t[2] for t in self.memory])).to(dtype=torch.float32)
        terminated_batch = torch.as_tensor(np.array([t[3] for t in self.memory])).to(dtype=torch.bool)

        for t in reversed(range(len(reward_batch) - 1)):
            reward_batch[t] = reward_batch[t] + (self.gamma * reward_batch[t + 1]) * (torch.logical_not(terminated_batch[t]))

        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-12)

        log_prob = torch.nn.functional.log_softmax(self.policy(state_batch), dim=1)
        log_prob = log_prob.gather(dim=1, index=action_batch.unsqueeze(dim=1)).squeeze()

        self.loss = torch.sum(-log_prob * reward_batch) / self.nb_episodes

        self.optimizer.zero_grad()

        self.loss.backward()

        self.optimizer.step()
