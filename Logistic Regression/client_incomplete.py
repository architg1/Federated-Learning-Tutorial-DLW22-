import warnings

import flwr as fl
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    # loading the dataset

    # loading the train set
    df_train = pd.read_csv("/Users/architg/Desktop/Work/Misc./Federated Learning Tutorial (DLW22)/Network Intrusion Dataset/script_train.csv")
    df_train = utils.dataProcessing(df_train)  # data processing
    x_train = utils.featureSelection(df_train)[0]  # feature selection
    y_train = utils.featureSelection(df_train)[1]

    # loading the test set
    df_test = pd.read_csv("/Users/architg/Desktop/Work/Misc./Federated Learning Tutorial (DLW22)/Network Intrusion Dataset/script_test.csv")
    df_test = utils.dataProcessing(df_test)  # data processing
    x_test = utils.featureSelection(df_test)[0]  # feature selection
    y_test = utils.featureSelection(df_test)[1]

    # Make a Logistic Regression model
    model = -1

    # Setting initial parameters
    utils.set_initial_params(model)


    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            # set model to learned parameters
            utils.set_model_params()
            # calculate loss
            loss = -1
            # calculate accuracy
            accuracy = -1
            return loss, len(x_test), {"accuracy": accuracy}


    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=Client())
