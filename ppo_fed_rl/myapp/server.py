from flwr.serverapp import ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context

def server_fn(context: Context) -> ServerAppComponents:
    """Factory function that builds the ServerApp runtime components."""
    
    # Define your standard aggregation strategy parameters here
    strategy = FedAvg(
        fraction_fit=0.1,
        min_fit_clients=3,
        min_available_clients=3,
        # Note: 'fit_rounds' was renamed to 'num_rounds' in the server configuration
    )
    
    # Read total rounds configuration from context (defined in pyproject.toml)
    # default value fallback if not explicitly set in configuration tables
    num_rounds = context.run_config.get("num-server-rounds", 5) 
    
    # Configure the server runtime components
    config = {"num_rounds": num_rounds}
    
    return ServerAppComponents(strategy=strategy, config=config)

# This object exposes the entrypoint that 'flwr run' looks for
app = ServerApp(server_fn=server_fn)
