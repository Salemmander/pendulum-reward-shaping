# Pendulum Reward Shaping - Implementation Plan

## Goal

Learn RL fundamentals by implementing PPO from scratch, training it on vanilla Pendulum-v1, then experimenting with reward wrappers to teach non-trivial behaviors.

## The Plan

### 1. Neural Network Architecture

- Write `Actor` class (outputs a Gaussian distribution over actions)
- Write `Critic` class (outputs a scalar value)
- Sanity check: feed a dummy observation through each, verify shapes are right

### 2. Rollout Buffer

- A class/dict that stores `obs, action, reward, log_prob, value, done` for N steps
- Has methods to `store()` a transition and `get()` the collected batch

### 3. Rollout Loop

- Step through env for N steps using current policy
- For each step, sample action, compute log_prob and value, store everything
- Sanity check: run it, print the buffer contents, make sure they look reasonable

### 4. GAE (Advantage Computation)

- Take the filled buffer, compute advantages and returns
- Sanity check: advantages should have mean ~0 after normalization

### 5. PPO Update Step

- Compute the clipped policy loss
- Compute value loss (MSE)
- Backprop, update both networks for K epochs
- Sanity check: losses should decrease within an update phase

### 6. Tie It Together (Training Loop)

- `for iteration in range(N): rollout -> GAE -> update`
- Log episode rewards to see if it's learning
- Train on vanilla Pendulum. Expect average reward to climb from ~-1500 toward ~-200 over time

### 7. Write Reward Wrapper

- Subclass `gym.Wrapper`, override `step()`
- Pick one variant from project_ideas.md (e.g., balance at 30 degrees off vertical)

### 8. Train on Wrapped Env

- Point the existing training loop at the wrapped env
- Observe whether the policy learns the new behavior
- This is where the real learning happens - reward hacking, unexpected behaviors, etc.

## Tips

- Test each piece in isolation before moving on. If the buffer is broken, you'll never catch it later.
- Start with small rollout sizes (like 256 steps) for fast iteration, then scale up.
- Print and plot things - it's the only way to diagnose RL bugs.
- Expect it to be frustrating - RL bugs are silent. The code runs, it just doesn't learn. That's normal.

## Common RL Bugs (Debugging Checklist)

When the policy isn't learning, it's almost always one of:

1. Advantages computed wrong (sign flipped, no normalization)
2. `log_prob` mismatch between rollout and update
3. Exploding gradients (clip gradients at norm 0.5)
4. Learning rate too high for this architecture

## Reference Hyperparameters (starting points from CleanRL)

- `rollout_length`: 2048 steps
- `update_epochs`: 10
- `minibatch_size`: 64
- `clip_epsilon`: 0.2
- `learning_rate`: 3e-4
- `gamma` (discount): 0.99
- `gae_lambda`: 0.95

## Key PPO Concepts (Quick Reference)

### The Three Phases per Iteration

1. **Rollout**: run current policy, collect transitions (no gradients)
2. **Advantages**: compute GAE from the buffer
3. **Update**: K epochs of minibatch gradient updates on policy + value

### The Clipped Objective

```
ratio = pi_new(a|s) / pi_old(a|s)
L = min( ratio * A, clip(ratio, 1-eps, 1+eps) * A )
```

Prevents the policy from drifting too far in a single update, which lets you safely reuse the same data for multiple gradient epochs.

### Actor-Critic Structure

- **Actor**: state -> distribution over actions (Gaussian for continuous control)
- **Critic**: state -> scalar value estimate (used to compute advantages)

### On-Policy

PPO is on-policy - data collected from the old policy is only used for this iteration's updates, then discarded.
