# main.py
import flwr as fl
import tensorflow as tf

@fl.app.main
def main():
    # Build the strategy – you can use the same helper classes
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=5,
        min_available_clients=5,
        fit_rounds=10,
    )
    return strategy