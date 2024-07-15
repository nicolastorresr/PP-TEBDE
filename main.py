import argparse
import logging
from utils.config import load_config
from utils.logger import setup_logger
from data.data_loader import load_dataset
from models.temporal_cf import TemporalCF
from models.attention_mechanism import TemporalAttention
from privacy.federated_learning import FederatedLearning
from bias.bias_detector import BiasDetector
from bias.bias_mitigation import BiasMitigator
from explainable.explanation_generator import ExplanationGenerator
from evaluation.experiment import run_experiment
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from privacy.differential_privacy import add_noise
from evaluation.metrics import compute_ndcg, compute_gini_coefficient
import numpy as np
from collections import defaultdict
from privacy.differential_privacy import add_noise
from explainable.explanation_generator import generate_counterfactual, generate_feature_importance

def parse_arguments():
    parser = argparse.ArgumentParser(description='PP-TEBDE: Privacy-Preserving Temporal Exposure Bias Detection and Explanation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, choices=['edurec', 'movielens', 'amazon'], default='edurec', help='Dataset to use')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'explain'], default='train', help='Mode of operation')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    logger = setup_logger(config.logging)

    logger.info(f"Starting PP-TEBDE with dataset: {args.dataset} in {args.mode} mode")

    # Load and preprocess data
    train_data, val_data, test_data = load_dataset(args.dataset, config.data)

    # Initialize model components
    temporal_cf = TemporalCF(config.model)
    attention_mechanism = TemporalAttention(config.attention)
    federated_learning = FederatedLearning(config.privacy)
    bias_detector = BiasDetector(config.bias)
    bias_mitigator = BiasMitigator(config.bias)
    explanation_generator = ExplanationGenerator(config.explainable)

    # Create PP-TEBDE model
    pp_tebde = PPTEBDE(
        temporal_cf,
        attention_mechanism,
        federated_learning,
        bias_detector,
        bias_mitigator,
        explanation_generator
    )

    if args.mode == 'train':
        # Train the model
        pp_tebde.train(train_data, val_data)
    elif args.mode == 'evaluate':
        # Evaluate the model
        results = run_experiment(pp_tebde, test_data, config.evaluation)
        logger.info(f"Evaluation results: {results}")
    elif args.mode == 'explain':
        # Generate explanations
        explanations = pp_tebde.generate_explanations(test_data)
        logger.info(f"Generated explanations: {explanations[:5]}")  # Log first 5 explanations

    logger.info("PP-TEBDE execution completed successfully")

class PPTEBDE(nn.Module):
    def __init__(self, temporal_cf, attention_mechanism, federated_learning, 
                 bias_detector, bias_mitigator, explanation_generator):
        self.temporal_cf = temporal_cf
        self.attention_mechanism = attention_mechanism
        self.federated_learning = federated_learning
        self.bias_detector = bias_detector
        self.bias_mitigator = bias_mitigator
        self.explanation_generator = explanation_generator

    def train(self, train_data, val_data):
        logger = logging.getLogger(__name__)
        logger.info("Starting PP-TEBDE training process")

        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val_ndcg = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            self.train()
            total_loss = 0
            
            # Federated Learning: Simulate multiple clients
            for client_id in range(self.config.num_clients):
                client_data = self.federated_learning.get_client_data(train_data, client_id)
                
                for batch in tqdm(client_data, desc=f"Epoch {epoch+1}/{self.config.num_epochs}, Client {client_id+1}/{self.config.num_clients}"):
                    users, items, timestamps, labels = batch
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.forward(users, items, timestamps)
                    
                    # Add differential privacy noise
                    noisy_outputs = add_noise(outputs, self.config.privacy.epsilon)
                    
                    # Compute loss
                    loss = criterion(noisy_outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            # Aggregate model updates using federated learning
            self.federated_learning.aggregate_models(self)
            
            # Validation
            val_ndcg, val_gini = self.evaluate(val_data)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Loss: {total_loss:.4f}, "
                        f"Val NDCG: {val_ndcg:.4f}, "
                        f"Val Gini: {val_gini:.4f}")
            
            # Bias detection and mitigation
            detected_bias = self.bias_detector.detect(val_data)
            if detected_bias > self.config.bias_threshold:
                logger.info(f"Bias detected: {detected_bias:.4f}. Applying mitigation.")
                self.bias_mitigator.mitigate(self, val_data)
            
            # Early stopping
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info("Training completed")

    def forward(self, users, items, timestamps):
        # Temporal Collaborative Filtering
        cf_output = self.temporal_cf(users, items, timestamps)
        
        # Apply Temporal Attention
        attended_output = self.attention_mechanism(cf_output, timestamps)
        
        return attended_output

    def evaluate(self, data):
        self.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data:
                users, items, timestamps, labels = batch
                outputs = self.forward(users, items, timestamps)
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        ndcg = compute_ndcg(all_predictions, all_labels)
        gini = compute_gini_coefficient(all_predictions)
        
        return ndcg, gini

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, user, items, timestamp):
        """
        Predict the likelihood of a user interacting with given items at a specific timestamp.
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user]).repeat(len(items))
            items_tensor = torch.LongTensor(items)
            timestamp_tensor = torch.FloatTensor([timestamp]).repeat(len(items))
            
            outputs = self.forward(user_tensor, items_tensor, timestamp_tensor)
            
            # Add differential privacy noise to predictions
            noisy_outputs = add_noise(outputs, self.config.privacy.epsilon)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(noisy_outputs).cpu().numpy()
        
        return probabilities

    def detect_bias(self, data):
        """
        Detect temporal exposure bias in the given data.
        """
        self.eval()
        user_exposure = defaultdict(lambda: defaultdict(float))
        item_popularity = defaultdict(float)
        
        with torch.no_grad():
            for batch in data:
                users, items, timestamps, _ = batch
                outputs = self.forward(users, items, timestamps)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                for user, item, prob in zip(users.cpu().numpy(), items.cpu().numpy(), probabilities):
                    user_exposure[user][item] += prob
                    item_popularity[item] += prob
        
        # Compute Gini coefficient for item popularity
        item_popularity_list = list(item_popularity.values())
        gini_coefficient = self.compute_gini_coefficient(item_popularity_list)
        
        # Compute temporal skew
        temporal_skew = self.compute_temporal_skew(user_exposure)
        
        # Combine metrics to get overall bias score
        bias_score = 0.5 * gini_coefficient + 0.5 * temporal_skew
        
        return bias_score

    def mitigate_bias(self, recommendations, detected_bias):
        """
        Mitigate detected bias in recommendations.
        """
        if detected_bias < self.config.bias_threshold:
            return recommendations  # No mitigation needed
        
        mitigated_recommendations = []
        for user, items in recommendations:
            # Rerank items to reduce popularity bias
            popularity_scores = [self.item_popularity[item] for item in items]
            diversity_boost = 1 - np.array(popularity_scores) / max(popularity_scores)
            
            # Combine original scores with diversity boost
            alpha = self.config.diversity_weight * detected_bias  # Adaptive diversity weight
            combined_scores = (1 - alpha) * np.array([score for _, score in items]) + alpha * diversity_boost
            
            # Sort items by combined score
            reranked_items = [item for item, _ in sorted(zip(items, combined_scores), key=lambda x: x[1], reverse=True)]
            
            mitigated_recommendations.append((user, reranked_items))
        
        return mitigated_recommendations

    def generate_explanations(self, data):
        """
        Generate explanations for recommendations.
        """
        explanations = []
        self.eval()
        
        with torch.no_grad():
            for batch in data:
                users, items, timestamps, _ = batch
                outputs = self.forward(users, items, timestamps)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                for user, item, timestamp, prob in zip(users.cpu().numpy(), items.cpu().numpy(), timestamps.cpu().numpy(), probabilities):
                    # Generate counterfactual explanation
                    counterfactual = generate_counterfactual(self, user, item, timestamp)
                    
                    # Generate feature importance
                    feature_importance = generate_feature_importance(self, user, item, timestamp)
                    
                    # Generate natural language explanation
                    nl_explanation = self.generate_nl_explanation(user, item, prob, counterfactual, feature_importance)
                    
                    explanations.append({
                        'user': user,
                        'item': item,
                        'probability': prob,
                        'counterfactual': counterfactual,
                        'feature_importance': feature_importance,
                        'natural_language': nl_explanation
                    })
        
        return explanations

    def compute_gini_coefficient(self, values):
        """
        Compute the Gini coefficient of the given values.
        """
        sorted_values = np.sort(values)
        index = np.arange(1, len(values) + 1)
        return (np.sum((2 * index - len(values) - 1) * sorted_values)) / (len(values) * np.sum(sorted_values))

    def compute_temporal_skew(self, user_exposure):
        """
        Compute temporal skew in user exposures.
        """
        user_skews = []
        for user, item_exposures in user_exposure.items():
            timestamps = sorted(item_exposures.keys())
            exposures = [item_exposures[t] for t in timestamps]
            
            # Compute skew as the difference between the area under the exposure curve
            # and the area under a uniform exposure curve
            uniform_area = 0.5 * len(timestamps) * len(timestamps)
            actual_area = np.trapz(exposures, timestamps)
            skew = abs(actual_area - uniform_area) / uniform_area
            user_skews.append(skew)
        
        return np.mean(user_skews)

    def generate_nl_explanation(self, user, item, probability, counterfactual, feature_importance):
        """
        Generate a natural language explanation for a recommendation.
        """
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"User {user} is recommended item {item} with a probability of {probability:.2f}. "
        explanation += "This recommendation is based on the following factors:\n"
        
        for feature, importance in top_features:
            explanation += f"- {feature}: {importance:.2f}\n"
        
        explanation += f"\nIf {counterfactual['changed_feature']} had been {counterfactual['changed_value']}, "
        explanation += f"the recommendation probability would have been {counterfactual['new_probability']:.2f}."
        
        return explanation

if __name__ == '__main__':
    main()
