import torch
import gymnasium as gym
from src.agent import Agent
from tqdm import trange


class PPO:
    def __init__(
        self,
        agent: Agent,
        env: gym.Env,
        rollout_length: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        lr: float = 3e-4,
    ) -> None:
        self.agent = agent
        self.env = env
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

        obs_dim = 3
        action_dim = 1

        self.obs_buf = torch.zeros(rollout_length, obs_dim)
        self.actions_buf = torch.zeros(rollout_length, action_dim)
        self.log_probs_buf = torch.zeros(rollout_length)
        self.values_buf = torch.zeros(rollout_length)
        self.rewards_buf = torch.zeros(rollout_length)
        self.dones_buf = torch.zeros(rollout_length)
        self.advantages_buf = torch.zeros(rollout_length)
        self.returns_buf = torch.zeros(rollout_length)

        self.last_obs = None

    def rollout(self):
        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_rewards = []

        for step in range(self.rollout_length):
            obs_tensor = torch.from_numpy(obs).float()
            with torch.no_grad():
                dist = self.agent.act(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                value = self.agent.critique(obs_tensor)

            action_np = action.numpy().clip(-2.0, 2.0)
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            self.obs_buf[step] = obs_tensor
            self.actions_buf[step] = action
            self.log_probs_buf[step] = log_prob
            self.values_buf[step] = value
            self.rewards_buf[step] = reward
            self.dones_buf[step] = float(done)

            ep_reward += reward
            obs = next_obs
            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

        self.last_obs = obs
        self.last_ep_rewards = ep_rewards

    def advantage(self):
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(self.last_obs).float()
            last_value = self.agent.critique(last_obs_tensor).item()

        last_gae = 0.0
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_value = last_value
            else:
                next_value = self.values_buf[t + 1]

            mask = 1.0 - self.dones_buf[t]
            delta = (
                self.rewards_buf[t]
                + self.gamma * next_value * mask
                - self.values_buf[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * mask
            self.advantages_buf[t] = last_gae

        self.returns_buf = self.advantages_buf + self.values_buf

    def update(self):
        advantages = (self.advantages_buf - self.advantages_buf.mean()) / (
            self.advantages_buf.std() + 1e-8
        )

        indices = torch.arange(self.rollout_length)

        for _ in range(self.update_epochs):
            perm = indices[torch.randperm(self.rollout_length)]

            for start in range(0, self.rollout_length, self.minibatch_size):
                mb_idx = perm[start : start + self.minibatch_size]

                mb_obs = self.obs_buf[mb_idx]
                mb_actions = self.actions_buf[mb_idx]
                mb_old_log_probs = self.log_probs_buf[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = self.returns_buf[mb_idx]

                dist = self.agent.act(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1)
                values = self.agent.critique(mb_obs)

                ratio = (new_log_probs - mb_old_log_probs).exp()

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - mb_returns) ** 2).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_iterations, save_path: str = "agent.pt"):
        pbar = trange(num_iterations)
        for _ in pbar:
            self.rollout()
            self.advantage()
            self.update()

            if self.last_ep_rewards:
                mean_reward = sum(self.last_ep_rewards) / len(self.last_ep_rewards)
                pbar.set_postfix(ep_reward=f"{mean_reward:.1f}")

        torch.save(self.agent.state_dict(), save_path)
        print(f"saved model to {save_path}")

    def load(self, path: str):
        self.agent.load_state_dict(torch.load(path))
