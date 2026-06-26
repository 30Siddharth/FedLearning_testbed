# myapp/server.py
# FIX: correct import paths — `flwr.serverapp` does not exist in flwr >= 1.0
from flwr.server import ServerApp, ServerConfig
from flwr.server.app import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context


def server_fn(context: Context) -> ServerAppComponents:
    """Factory function that builds the ServerApp runtime components."""
    strategy = FedAvg(
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )

    # Read total rounds from context (set in pyproject.toml)
    num_rounds = int(context.run_config.get("num-server-rounds", 5))

    # FIX: ServerAppComponents expects a ServerConfig object, not a plain dict
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Expose entrypoint that `flwr run` looks for via pyproject.toml
app = ServerApp(server_fn=server_fn)
