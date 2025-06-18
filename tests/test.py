import torch
import gymnasium

import brl

class MyPolicy(torch.nn.Module):

    def __init__(self):
        super(MyPolicy, self).__init__()

        self.linear1 = torch.nn.Linear(4, 32)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 2)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def act(self, state):
        state = torch.from_numpy(state)
        probabilities = self.forward(state)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action, probabilities # m.log_prob(action)

policy = MyPolicy()

agent = brl.dqn.DQN(policy=policy)
env = gymnasium.make("CartPole-v1")

def training(env, agent, nb_steps):
    for i in range(nb_steps):
        print(run_env(env, agent))
    
    return 0

def run_env(env, agent):
    sum_reward = 0

    terminated = False
    obs, info = env.reset()

    while not terminated:
        action = agent.act(obs)

        new_obs, reward, terminated, truncated, info = env.step(action)

        agent.observe(obs, action, reward, new_obs)

        obs = new_obs

        sum_reward += reward

        agent.optimize()

    return sum_reward

training(env, agent, nb_steps=1000)
