# myapp/client.py
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import numpy as np
from .model import (
    Actor, get_policy_network, get_gym_env, run_ppo_episode
)


class PPOClient(NumPyClient):
    def __init__(self, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.env = get_gym_env()
        self.local_actor = get_policy_network(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        )
        self.local_actor.to(self.device)
        self.lr = lr

    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.local_actor.parameters()]

    def fit(self, global_parameters, config):
        # Load global parameters into local model
        with torch.no_grad():
            for p, w in zip(self.local_actor.parameters(), global_parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        # Run local PPO episode
        local_actor_updated, local_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr,
        )
        self.local_actor.load_state_dict(local_actor_updated.state_dict())

        new_params = [p.cpu().detach().numpy() for p in self.local_actor.parameters()]
        # num_examples: use 1 episode as the data point
        return new_params, 1, {"local_loss": local_loss}

    def evaluate(self, parameters, config):
        with torch.no_grad():
            for p, w in zip(self.local_actor.parameters(), parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        # FIX: evaluate() num_examples should be 1 (one episode), not
        # len(observation_space.sample()) which is just the state dimension.
        _, eval_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr,
        )
        return float(eval_loss), 1, {"eval_loss": eval_loss}


def client_fn(context: Context) -> NumPyClient:
    """Factory function called by ClientApp to instantiate a client."""
    lr = context.run_config.get("lr", 1e-3)
    device = context.run_config.get("device", "cpu")
    return PPOClient(lr=lr, device=device)


# FIX: expose `app` so pyproject.toml entry point `myapp.client:app` resolves.
# fl.client.start_numpy_client is deprecated in flwr >= 1.0.
app = ClientApp(client_fn=client_fn)
