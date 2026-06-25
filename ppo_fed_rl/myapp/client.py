# myapp/client.py
import flwr as fl
import torch
import numpy as np
from .model import (
    Actor, get_policy_network, get_gym_env, run_ppo_episode
)

class PPOClient(fl.client.NumPyClient):
    def __init__(self, lr=1e-3, device="cpu"):
        self.device = torch.device(device)
        self.env = get_gym_env()
        # Initialize the local policy network
        self.local_actor = get_policy_network(state_dim=self.env.observation_space.shape[0], action_dim=self.env.action_space.shape[0])
        self.lr = lr

    def get_parameters(self, config):
        """Retrieves the current weights from the local model as a list of NumPy arrays."""
        # We must pass the parameters of the torch module, converting to numpy
        return [p.cpu().numpy() for p in self.local_actor.parameters()]

    def fit(self, global_parameters, config):
        """
        Loads global parameters, runs local PPO training, and returns updated weights.
        """
        # 1. Load global parameters into the local model
        with torch.no_grad():
            # Manually load weights from numpy arrays back into the PyTorch model
            for p, w in zip(self.local_actor.parameters(), global_parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        # 2. Run the local training simulation
        # The global parameters are used to initialize the client's state.
        # The return value is the newly updated local model and its loss.
        local_actor_updated, local_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr
        )

        # 3. Update the local model with the trained weights
        self.local_actor.load_state_dict(local_actor_updated.state_dict())

        # 4. Return updated parameters and metadata
        new_params = [p.cpu().numpy() for p in self.local_actor.parameters()]
        return new_params, len(self.env.observation_space.sample()), {"local_loss": local_loss}

    def evaluate(self, parameters, config):
        """
        Evaluates the model performance using the global parameters.
        (Simulated by running a test episode)
        """
        # 1. Load global parameters
        with torch.no_grad():
            for p, w in zip(self.local_actor.parameters(), parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        # 2. Run a test episode using the global weights
        # We calculate the loss, but the network does *not* train.
        _, eval_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr # LR doesn't matter for evaluation
        )
        
        return float(eval_loss), len(self.env.observation_space.sample()), {"eval_loss": eval_loss}

if __name__ == "__main__":
    # The boilerplate client startup logic
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    client = PPOClient(lr=args.lr, device=args.device)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
