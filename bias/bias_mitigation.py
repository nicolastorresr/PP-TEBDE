import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

class BiasMitigator:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()

    def mitigate_bias(self, model, user_item_matrix, user_features, item_features, detected_bias):
        """
        Mitigate detected bias in the recommendation model.
        
        Args:
        model: The recommendation model
        user_item_matrix (np.array): User-item interaction matrix
        user_features (np.array): User feature matrix
        item_features (np.array): Item feature matrix
        detected_bias (float): Level of detected bias
        
        Returns:
        tuple: Updated model parameters and bias mitigation metrics
        """
        if detected_bias < self.config.bias_threshold:
            return model, {"mitigation_applied": False}

        # Apply different mitigation strategies based on the level of detected bias
        if detected_bias < 0.3:
            return self.apply_soft_regularization(model, user_item_matrix, detected_bias)
        elif detected_bias < 0.6:
            return self.apply_reranking(model, user_item_matrix, user_features, item_features)
        else:
            return self.apply_adversarial_debiasing(model, user_item_matrix, user_features, item_features)

    def apply_soft_regularization(self, model, user_item_matrix, detected_bias):
        """
        Apply soft regularization to model parameters to reduce bias.
        """
        lambda_reg = self.config.regularization_strength * detected_bias
        
        for param in model.parameters():
            param.data -= lambda_reg * param.data
        
        return model, {"mitigation_method": "soft_regularization", "lambda": lambda_reg}

    def apply_reranking(self, model, user_item_matrix, user_features, item_features):
        """
        Apply a reranking strategy to balance exposure across items.
        """
        # Get model predictions
        with torch.no_grad():
            predictions = model(torch.tensor(user_features), torch.tensor(item_features))
            predictions = predictions.cpu().numpy()
        
        # Calculate item popularity
        item_popularity = np.sum(user_item_matrix, axis=0)
        item_popularity = self.scaler.fit_transform(item_popularity.reshape(-1, 1)).flatten()
        
        # Calculate diversity score (inverse of popularity)
        diversity_score = 1 - item_popularity
        
        # Combine prediction scores with diversity scores
        alpha = self.config.diversity_weight
        combined_scores = (1 - alpha) * predictions + alpha * diversity_score
        
        # Rerank items based on combined scores
        reranked_indices = np.argsort(combined_scores, axis=1)[:, ::-1]
        
        return model, {"mitigation_method": "reranking", "alpha": alpha}

    def apply_adversarial_debiasing(self, model, user_item_matrix, user_features, item_features):
        """
        Apply adversarial debiasing to reduce bias in model predictions.
        """
        # Implement a simple adversarial network
        class Adversary(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 64)
                self.fc2 = torch.nn.Linear(64, 1)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return torch.sigmoid(self.fc2(x))

        adversary = Adversary(model.embedding_dim)
        optimizer_adv = torch.optim.Adam(adversary.parameters(), lr=self.config.adversarial_lr)
        
        # Training loop
        for _ in range(self.config.adversarial_epochs):
            # Train adversary
            embeddings = model.get_embeddings(torch.tensor(user_features), torch.tensor(item_features))
            sensitive_attribute = torch.tensor(item_features[:, self.config.sensitive_attribute_index])
            
            adv_loss = F.binary_cross_entropy(adversary(embeddings), sensitive_attribute.float())
            optimizer_adv.zero_grad()
            adv_loss.backward()
            optimizer_adv.step()
            
            # Update model to fool adversary
            embeddings = model.get_embeddings(torch.tensor(user_features), torch.tensor(item_features))
            fool_loss = -F.binary_cross_entropy(adversary(embeddings), sensitive_attribute.float())
            
            model.optimizer.zero_grad()
            fool_loss.backward()
            model.optimizer.step()
        
        return model, {"mitigation_method": "adversarial_debiasing", "adv_loss": adv_loss.item()}

    def post_processing_fairness(self, recommendations, user_item_matrix):
        """
        Apply post-processing fairness constraints to recommendations.
        """
        n_users, n_items = user_item_matrix.shape
        fairness_matrix = np.zeros((n_users, n_items))
        
        # Calculate fairness scores
        item_popularity = np.sum(user_item_matrix, axis=0)
        max_popularity = np.max(item_popularity)
        fairness_scores = 1 - (item_popularity / max_popularity)
        
        for i in range(n_users):
            fairness_matrix[i] = fairness_scores
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(fairness_matrix, maximize=True)
        
        # Create fair recommendations
        fair_recommendations = [(user, item) for user, item in zip(row_ind, col_ind)]
        
        return fair_recommendations

    def evaluate_bias_mitigation(self, original_recommendations, mitigated_recommendations, user_item_matrix):
        """
        Evaluate the effectiveness of bias mitigation.
        """
        original_exposure = self.calculate_exposure(original_recommendations, user_item_matrix)
        mitigated_exposure = self.calculate_exposure(mitigated_recommendations, user_item_matrix)
        
        original_gini = self.gini_coefficient(original_exposure)
        mitigated_gini = self.gini_coefficient(mitigated_exposure)
        
        ndcg_change = self.calculate_ndcg_change(original_recommendations, mitigated_recommendations, user_item_matrix)
        
        return {
            "original_gini": original_gini,
            "mitigated_gini": mitigated_gini,
            "gini_improvement": original_gini - mitigated_gini,
            "ndcg_change": ndcg_change
        }

    def calculate_exposure(self, recommendations, user_item_matrix):
        """
        Calculate item exposure based on recommendations.
        """
        n_items = user_item_matrix.shape[1]
        exposure = np.zeros(n_items)
        
        for user, item in recommendations:
            exposure[item] += 1
        
        return exposure / len(recommendations)

    def gini_coefficient(self, array):
        """
        Calculate the Gini coefficient of a numpy array.
        """
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def calculate_ndcg_change(self, original_recommendations, mitigated_recommendations, user_item_matrix):
        """
        Calculate the change in NDCG after bias mitigation.
        """
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))

        def ndcg_at_k(r, k):
            dcg_max = dcg_at_k(sorted(r, reverse=True), k)
            if not dcg_max:
                return 0.
            return dcg_at_k(r, k) / dcg_max

        original_ndcg = 0
        mitigated_ndcg = 0
        n_users = user_item_matrix.shape[0]

        for user in range(n_users):
            true_relevance = user_item_matrix[user]
            original_relevance = [true_relevance[item] for _, item in original_recommendations if _== user]
            mitigated_relevance = [true_relevance[item] for _, item in mitigated_recommendations if _== user]
            
            original_ndcg += ndcg_at_k(original_relevance, len(original_relevance))
            mitigated_ndcg += ndcg_at_k(mitigated_relevance, len(mitigated_relevance))

        original_ndcg /= n_users
        mitigated_ndcg /= n_users

        return mitigated_ndcg - original_ndcg
