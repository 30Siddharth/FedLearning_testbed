# PPO Federated RL — Flower App

A minimal Federated Reinforcement Learning testbed using [Flower](https://flower.ai) (flwr >= 1.32) and PyTorch PPO on the `Pendulum-v1` Gymnasium environment.

## Setup

```bash
cd ppo_fed_rl
pip install -e myapp
```

This installs the `myapp` package and all its dependencies (`flwr[simulation]`, `torch`, `gymnasium`, `numpy`).

## Run (simulation)

```bash
# From the ppo_fed_rl/ directory:
flwr run myapp
```

Flower's simulation engine will spin up 2 virtual SuperNodes by default (flwr v1.32 default).
To override settings at runtime:

```bash
flwr run myapp --run-config "num-server-rounds=10 lr=5e-4"
```

## Federation config (SuperLink connection)

As of flwr v1.26+, federation/SuperLink connection config is **no longer in `pyproject.toml`**.
It lives in `~/.flwr/config.toml` (the central Flower CLI config).
For local simulation, no extra config file is needed — `flwr run myapp` is sufficient.

## Structure

```
ppo_fed_rl/
├── README.md
└── myapp/
    ├── __init__.py
    ├── pyproject.toml   # Flower app config + Python package metadata
    ├── client.py        # PPOClient + ClientApp
    ├── server.py        # FedAvg strategy + ServerApp
    └── model.py         # Actor network + PPO episode loop
```
