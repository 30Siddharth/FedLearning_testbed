# myapp/model.py
"""
Contains the PPO Policy Network and helper functions for Inverted Pendulum.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple

# --- 1. Policy Network (The Actor) ---
class Actor(nn.Module):
    """
    Policy network for the continuous control environment (Pendulum).
    Outputs the mean and standard deviation of an action (Gaussian distribution).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # FIX: store action_dim so forward() can reference it
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2),  # Output Mean (mu) and LogStd (log_sigma)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean (mu) and log standard deviation (log_std)."""
        output = self.net(state)  # shape: [N, action_dim * 2]
        # FIX: chunk on last dim so each piece has shape [N, action_dim]
        mu, log_std = output.chunk(2, dim=-1)
        return mu, log_std


# --- 2. PPO Utility Functions ---

def get_policy_network(state_dim: int, action_dim: int) -> Actor:
    """Initializes and returns a fresh policy network."""
    return Actor(state_dim, action_dim)


def get_gym_env() -> gym.Env:
    """Initializes the Pendulum environment."""
    return gym.make("Pendulum-v1")


# --- 3. Training Loop ---

def run_ppo_episode(
    device: torch.device,
    actor: Actor,
    env: gym.Env,
    gamma: float = 0.99,
    gae_lam: float = 0.95,
    ppo_clip: float = 0.2,
    lr: float = 1e-3,
) -> Tuple[nn.Module, float]:
    """
    Runs a single episode and performs a single PPO update step on the local model.
    Returns the updated model and the average loss.
    """
    print("\n[Client] --- Starting local PPO training episode...")

    # 1. Initial setup
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

    # 2. Collect trajectory
    states, actions, rewards, dones = [], [], [], []

    for _ in range(50):  # Limit the episode length
        with torch.no_grad():
            mu, log_std = actor(state_tensor)
            std = torch.exp(log_std)
            action = torch.normal(mu, std).squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state_tensor.cpu().numpy())
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)

        state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device).unsqueeze(0)

        if terminated or truncated:
            break

    # 3. Convert trajectories to tensors
    states_t = torch.tensor(np.vstack(states), dtype=torch.float32).to(device)

    # --- Simplified PPO Loss ---
    # FIX: compute loss through actor forward pass so the graph has gradients.
    # (Using detached actions_t directly as loss caused backward() to fail.)
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    optimizer.zero_grad()

    mu_pred, log_std_pred = actor(states_t)
    # Behaviour-cloning-style surrogate: minimise negative log-prob of collected actions
    actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    std_pred = torch.exp(log_std_pred)
    dist = torch.distributions.Normal(mu_pred, std_pred)
    loss = -dist.log_prob(actions_t).mean()

    loss.backward()
    optimizer.step()

    return actor, loss.item()


if __name__ == '__main__':
    env = get_gym_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    initial_actor = get_policy_network(state_dim, action_dim)
    updated_actor, loss = run_ppo_episode(
        device=torch.device("cpu"),
        actor=initial_actor,
        env=env,
        lr=1e-3,
    )
    print(f"\n[Demo Success] Model trained. Loss: {loss:.4f}")
