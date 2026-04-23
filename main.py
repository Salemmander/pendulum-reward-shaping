import gymnasium as gym
from src.agent import Agent
from src.ppo import PPO
from src.wrappers import AngleBalanceWrapper

env = AngleBalanceWrapper(gym.make("Pendulum-v1"), 30)
agent = Agent()
ppo = PPO(agent, env, rollout_length=2048)

ppo.train(500)
