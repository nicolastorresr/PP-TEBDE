import torch
import numpy as np
from typing import List, Dict
from collections import OrderedDict

class FederatedLearning:
    def __init__(self, config):
        self.num_clients = config.num_clients
        self.client_fraction = config.client_fraction
        self.num_rounds = config.num_rounds
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate
        self.privacy_epsilon = config.privacy_epsilon

    def get_client_data(self, data, client_id):
        """
        Simulate data distribution among clients.
        In a real-world scenario, this would fetch data from actual clients.
        """
        num_samples = len(data)
        samples_per_client = num_samples // self.num_clients
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client
        return data[start_idx:end_idx]

    def client_update(self, client_model, client_data, client_id):
        """
        Perform local update on a client's model.
        """
        optimizer = torch.optim.SGD(client_model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(self.local_epochs):
            for batch in client_data:
                users, items, timestamps, labels = batch
                optimizer.zero_grad()
                outputs = client_model(users, items, timestamps)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return client_model.state_dict()

    def aggregate_models(self, global_model, client_models: List[Dict[str, torch.Tensor]]):
        """
        Aggregate updates from multiple clients using FedAvg algorithm.
        """
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
        
        global_model.load_state_dict(global_dict)
        return global_model

    def add_noise(self, model, epsilon):
        """
        Add Gaussian noise to model parameters for differential privacy.
        """
        for param in model.parameters():
            noise = torch.randn_like(param) * (1.0 / epsilon)
            param.add_(noise)
        return model

    def train(self, global_model, train_data):
        """
        Perform federated learning training process.
        """
        for round in range(self.num_rounds):
            # Select a fraction of clients to participate in this round
            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(range(self.num_clients), num_selected_clients, replace=False)

            # Client update
            client_models = []
            for client_id in selected_clients:
                client_data = self.get_client_data(train_data, client_id)
                client_model = type(global_model)()  # Create a new instance of the model
                client_model.load_state_dict(global_model.state_dict())  # Copy global model parameters
                updated_client_model = self.client_update(client_model, client_data, client_id)
                client_models.append(updated_client_model)

            # Aggregate models
            global_model = self.aggregate_models(global_model, client_models)

            # Add noise for differential privacy
            global_model = self.add_noise(global_model, self.privacy_epsilon)

            # Optionally, evaluate global model performance here

        return global_model

    def secure_aggregation(self, client_models: List[Dict[str, torch.Tensor]]):
        """
        Implement secure aggregation to protect privacy during model averaging.
        This is a simplified version and should be replaced with a proper secure aggregation protocol in production.
        """
        aggregated_model = OrderedDict()
        for key in client_models[0].keys():
            encrypted_tensors = [self.simulate_encryption(model[key]) for model in client_models]
            aggregated_tensor = torch.mean(torch.stack(encrypted_tensors), dim=0)
            aggregated_model[key] = self.simulate_decryption(aggregated_tensor)
        return aggregated_model

    def simulate_encryption(self, tensor):
        """
        Simulate homomorphic encryption. In a real-world scenario, this should be replaced with actual HE.
        """
        return tensor + torch.randn_like(tensor) * 0.01  # Add small random noise to simulate encryption

    def simulate_decryption(self, tensor):
        """
        Simulate homomorphic decryption. In a real-world scenario, this should be replaced with actual HE.
        """
        return tensor  # In this simulation, we just return the tensor as is
