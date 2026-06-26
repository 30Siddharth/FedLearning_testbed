# myapp/server.py
# Import paths updated for flwr v1.31+ restructure:
#   ServerApp -> flwr.serverapp
#   Context   -> flwr.app
from flwr.serverapp import ServerApp, ServerAppComponents
from flwr.app import Context
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context) -> ServerAppComponents:
    """Factory called by ServerApp to build the server runtime."""
    strategy = FedAvg(
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )

    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Entrypoint referenced by pyproject.toml: myapp.server:app
app = ServerApp(server_fn=server_fn)
