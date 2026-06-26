# myapp/client.py
# Import paths updated for flwr v1.31+ restructure:
#   Context -> flwr.app
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.app import Context
import torch
from .model import get_policy_network, get_gym_env, run_ppo_episode


class PPOClient(NumPyClient):
    def __init__(self, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.env = get_gym_env()
        self.local_actor = get_policy_network(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        ).to(self.device)
        self.lr = lr

    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.local_actor.parameters()]

    def fit(self, global_parameters, config):
        with torch.no_grad():
            for p, w in zip(self.local_actor.parameters(), global_parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        local_actor_updated, local_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr,
        )
        self.local_actor.load_state_dict(local_actor_updated.state_dict())

        new_params = [p.cpu().detach().numpy() for p in self.local_actor.parameters()]
        return new_params, 1, {"local_loss": local_loss}

    def evaluate(self, parameters, config):
        with torch.no_grad():
            for p, w in zip(self.local_actor.parameters(), parameters):
                p.data.copy_(torch.from_numpy(w).to(self.device))

        _, eval_loss = run_ppo_episode(
            device=self.device,
            actor=self.local_actor,
            env=self.env,
            lr=self.lr,
        )
        return float(eval_loss), 1, {"eval_loss": eval_loss}


def client_fn(context: Context) -> NumPyClient:
    """Factory called by ClientApp to instantiate a client per SuperNode."""
    lr = float(context.run_config.get("lr", 1e-3))
    device = str(context.run_config.get("device", "cpu"))
    return PPOClient(lr=lr, device=device)


# Entrypoint referenced by pyproject.toml: myapp.client:app
app = ClientApp(client_fn=client_fn)
