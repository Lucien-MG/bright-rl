import torch


class Reinforce:

    def __init__(
        self,
        policy,
        gamma=0.99,
        nb_episodes=2,
        optimizer=torch.optim.SGD,
        optimizer_parameters={"lr": 3e-2},
    ):
        self.policy = policy
        self.gamma = gamma
        self.nb_episodes = nb_episodes

        self.optimizer = optimizer(self.policy.parameters(), **optimizer_parameters)

        self.loss = 0

        self.memory = []
        self.episodes_in_memory = 0

    @torch.inference_mode()
    def act(self, observation, _mask=None):
        observation = torch.from_numpy(observation)
        probabilities = torch.nn.functional.softmax(self.policy(observation), dim=0)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action

    def observe(self, observation, action, reward, next_state, terminated):
        self.memory.append(
            (
                torch.from_numpy(observation),
                torch.tensor(action),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(terminated),
            )
        )
        self.episodes_in_memory += terminated

        if self.episodes_in_memory == self.nb_episodes:
            self.optimize()

            self.episodes_in_memory = 0

    def optimize(self):
        # Unzip the list of tuples into separate lists for each data type.
        states, actions, rewards, terminated = zip(*self.memory)

        # Use torch.stack to convert the lists of tensors to a single batched tensor.
        state_batch = torch.stack(states)
        action_batch = torch.stack(actions)
        reward_batch = torch.stack(rewards)
        terminated_batch = torch.stack(terminated)

        reward_monitor = torch.sum(reward_batch)

        for t in reversed(range(len(reward_batch) - 1)):
            reward_batch[t] = reward_batch[t] + (self.gamma * reward_batch[t + 1]) * (
                torch.logical_not(terminated_batch[t])
            )

        reward_batch = (reward_batch - reward_batch.mean()) / (
            reward_batch.std() + 1e-12
        )

        log_prob = torch.nn.functional.log_softmax(self.policy(state_batch), dim=1)
        log_prob = log_prob.gather(dim=1, index=action_batch.unsqueeze(dim=1)).squeeze()

        self.loss = torch.mean(-log_prob * reward_batch) / self.nb_episodes

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.memory.clear()
