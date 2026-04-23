import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(1))

    def act(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

    def critique(self, obs):
        return self.critic(obs).squeeze(-1)
