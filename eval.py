import gymnasium as gym
import torch

from src.agent import Agent

env = gym.make("Pendulum-v1", render_mode="human")
agent = Agent()
agent.load_state_dict(torch.load("agent.pt"))
agent.eval()

obs, _ = env.reset()
for _ in range(1000):
    obs_tensor = torch.from_numpy(obs).float()
    with torch.no_grad():
        dist = agent.act(obs_tensor)
        action = dist.mean

    obs, _, term, trunc, _ = env.step(action.numpy().clip(-2, 2))
    if term or trunc:
        obs, _ = env.reset()

env.close()
