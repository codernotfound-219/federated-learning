import pickle
import pandas as pd
import numpy as np
import random 
from collections import OrderedDict
import torch
from itertools import combinations
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from collections import deque


class ClientSelection:
   
    def __init__(self):
        """
        Load raw CSV files. Processing is deferred to the selection phase.
        """
        self.stats_df = pd.read_csv("/Users/aswatt/Development/Python/Workshop_ML_IISc/source/profile_on_rbpi/batch_size_ablation.csv")
        self.label_df = pd.read_csv("/Users/aswatt/Development/Python/Workshop_ML_IISc/source/profile_on_rbpi/client_label_distribution_int.csv")
        self.historical_scores = {}
        self.initialized = False  # will trigger lazy setup

    def _initialize_once(self):
        """Precomputes necessary fields for client scoring."""
        self.stats_df["client_id"] = self.stats_df["client_id"].apply(lambda x: f"part_{x}")
        self.df = pd.merge(self.label_df.copy(), self.stats_df.copy(), on="client_id")

        self.df["norm_speed"] = 1 - (self.df["training_time"] / self.df["training_time"].max())
        self.df["norm_size"] = self.df["num_items"] / self.df["num_items"].max()

        def entropy(row):
            labels = np.array([row[f"label_{i}"] for i in range(10)])
            total = labels.sum()
            probs = labels / total if total > 0 else np.zeros_like(labels)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])

        self.df["entropy"] = self.df[[f"label_{i}" for i in range(10)]].apply(entropy, axis=1)
        self.df["norm_entropy"] = self.df["entropy"] / self.df["entropy"].max()

        # Rare label logic
        global_label_counts = self.df[[f"label_{i}" for i in range(10)]].sum()
        self.rare_labels = set(global_label_counts[global_label_counts < global_label_counts.mean()].index.map(lambda x: int(x.split("_")[1])))
        self.client_label_sets = {
            row["client_id"]: {i for i in range(10) if row[f"label_{i}"] > 0}
            for _, row in self.df.iterrows()
        }

        # Initialize historical scores
        self.historical_scores = {cid: 0.5 for cid in self.df["client_id"]}
        self.initialized = True

    def client_selection_random(self, clients, CS_args: dict) -> list:
        return np.random.choice(
            [client.cid for client in clients],
            CS_args["num_clients_per_round"],
            replace=False
        ).tolist()

    def client_selection_adaptive(self, clients, CS_args: dict) -> list:
        """
        Selects top clients based on adaptive utility score using:
        - speed, entropy, sample size, rare label coverage, historical contribution
        """
        if not self.initialized:
            self._initialize_once()

        df = self.df.copy()
        round_num = CS_args["round"]
        alpha = CS_args.get("alpha", 0.6)
        k = CS_args["num_clients_per_round"]

        df["historical_score"] = df["client_id"].map(self.historical_scores)
        df["norm_hist"] = df["historical_score"] / df["historical_score"].max()
        df["rare_label_boost"] = df["client_id"].apply(
            lambda cid: len(self.client_label_sets[cid] & self.rare_labels) / 10.0
        )

        # Dynamic weights
        beta1 = 0.3 + (0.1 if round_num > 2 else 0)  # speed
        beta2 = 0.2 + (0.1 if round_num <= 2 else 0) # entropy
        beta3 = 0.2                                 # sample size
        beta4 = 0.3 + (0.1 if round_num > 2 else 0) # history
        beta5 = 0.05                                # rare labels

        df["score"] = (
            beta1 * df["norm_speed"] +
            beta2 * df["norm_entropy"] +
            beta3 * df["norm_size"] +
            beta4 * df["norm_hist"] +
            beta5 * df["rare_label_boost"] +
            np.random.uniform(-0.01, 0.01, len(df))
        )

        top_df = df.sort_values("score", ascending=False).head(k)
        selected_clients = top_df["client_id"].tolist()

        for cid in selected_clients:
            delta = np.random.uniform(0.01, 0.05)
            self.historical_scores[cid] = alpha * delta + (1 - alpha) * self.historical_scores[cid]

        return selected_clients

    
class Aggregation:
    def __init__(self):
        pass
    """
    Aggregation Algorithms
    """
    
    def aggregate_fedavg(self, round, selected_cids, client_list, update_client_models = True):
        
        global_model = OrderedDict()
        client_local_weights = client_list[0].model.to("cpu").state_dict()
        
        for layer in client_local_weights:
            shape = client_local_weights[layer].shape
            global_model[layer] = torch.zeros(shape)

        client_weights = list()
        
        n_k = list()
        for client_id in selected_cids:
            client_weights.append(client_list[client_id].model.to("cpu").state_dict())
            n_k.append(client_list[client_id].num_items)

        n_k = np.array(n_k)
        n_k = n_k / sum(n_k)
        
        for i, weights in enumerate(client_weights):
            for layer in weights.keys():
                # fmt: off
                global_model[layer] += (weights[layer] * n_k[i])
                # fmt: on

        # print("Global Model :: ", global_model["conv1.weight"][0])
        if update_client_models:
            for client in client_list:
                client.model.load_state_dict(global_model)

        return global_model, client_list
    
    
class Server(ClientSelection, Aggregation):
    def __init__(self, logger, device, model_class, model_args, data_path, dataset_id, test_batch_size):
        ClientSelection.__init__(self)
        Aggregation.__init__(self)
        
        self.id = "server"
        self.device = device
        self.logger = logger
        self.model = model_class(self.id, model_args)
        
        # Load normal test data for evaluation
        _, self.test_data = self.model.load_data(logger, data_path, dataset_id, self.id, None, test_batch_size)

        self.test_metrics = dict()  


    def test(self, round_id):
        data = self.test_data
        self.test_metrics[round_id] = self.model.test_model(self.logger, data)

            
            
