# import tensorflow as tf  # required for compilation on local device
import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    df_test = pd.read_csv("/Users/architg/Desktop/Work/Misc./Federated Learning Tutorial (DLW22)/Network Intrusion Dataset/script_test.csv")
    df_test = utils.dataProcessing(df_test)  # data processing
    x_test = utils.featureSelection(df_test)[0]  # feature selection
    y_test = utils.featureSelection(df_test)[1]

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(x_test))
        accuracy = model.score(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    # Make a logistic regression model
    model = LogisticRegression(
        penalty='l2',
        max_iter=1,  # local epoch
        warm_start=True
    )
    utils.set_initial_params(model)

    # create the strategy
    # FedAvg
    fed_avg_strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round()
    )

    fl.server.start_server(server_address="0.0.0.0:8080",
                           strategy=fed_avg_strategy,
                           config=fl.server.ServerConfig(num_rounds=3)
    )
