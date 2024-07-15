import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from lime import lime_tabular
import shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class ExplanationGenerator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def generate_explanation(self, user_id, item_id, user_features, item_features, interaction_history):
        """
        Generate a comprehensive explanation for a recommendation.

        Args:
        user_id (int): ID of the user
        item_id (int): ID of the recommended item
        user_features (np.array): Features of the user
        item_features (np.array): Features of the item
        interaction_history (list): User's interaction history

        Returns:
        dict: A dictionary containing various types of explanations
        """
        explanations = {}
        
        explanations['feature_importance'] = self.feature_importance_explanation(user_features, item_features)
        explanations['similar_items'] = self.similar_items_explanation(item_id, item_features)
        explanations['temporal_factors'] = self.temporal_factors_explanation(user_id, interaction_history)
        explanations['counterfactual'] = self.counterfactual_explanation(user_features, item_features)
        explanations['natural_language'] = self.natural_language_explanation(explanations)
        explanations['visual'] = self.visual_explanation(explanations)

        return explanations

    def feature_importance_explanation(self, user_features, item_features):
        """
        Generate feature importance explanation using SHAP values.
        """
        explainer = shap.DeepExplainer(self.model, [user_features, item_features])
        shap_values = explainer.shap_values([user_features, item_features])
        
        user_feature_importance = dict(zip(self.config.user_feature_names, shap_values[0][0]))
        item_feature_importance = dict(zip(self.config.item_feature_names, shap_values[1][0]))
        
        return {
            'user_features': user_feature_importance,
            'item_features': item_feature_importance
        }

    def similar_items_explanation(self, item_id, item_features):
        """
        Find and explain similar items based on cosine similarity.
        """
        item_embeddings = self.model.get_item_embeddings(torch.tensor(item_features))
        similarities = cosine_similarity(item_embeddings[item_id].reshape(1, -1), item_embeddings)
        similar_indices = similarities.argsort()[0][-6:-1][::-1]  # Top 5 similar items
        
        similar_items = []
        for idx in similar_indices:
            similar_items.append({
                'item_id': idx.item(),
                'similarity_score': similarities[0][idx].item()
            })
        
        return similar_items

    def temporal_factors_explanation(self, user_id, interaction_history):
        """
        Explain temporal factors affecting the recommendation.
        """
        recent_interactions = interaction_history[-self.config.n_recent_interactions:]
        
        temporal_explanation = {
            'recency_effect': self.calculate_recency_effect(recent_interactions),
            'diversity_trend': self.calculate_diversity_trend(interaction_history),
            'seasonal_pattern': self.detect_seasonal_pattern(interaction_history)
        }
        
        return temporal_explanation

    def counterfactual_explanation(self, user_features, item_features):
        """
        Generate counterfactual explanations using LIME.
        """
        def predict_fn(x):
            return self.model(torch.tensor(x[:len(user_features)]), 
                              torch.tensor(x[len(user_features):])).detach().numpy()
        
        explainer = lime_tabular.LimeTabularExplainer(
            np.concatenate([user_features, item_features]),
            feature_names=self.config.user_feature_names + self.config.item_feature_names,
            class_names=['not recommended', 'recommended'],
            mode='regression'
        )
        
        exp = explainer.explain_instance(
            np.concatenate([user_features, item_features]),
            predict_fn,
            num_features=5
        )
        
        return exp.as_list()

    def natural_language_explanation(self, explanations):
        """
        Generate a natural language explanation based on the other explanation types.
        """
        nl_explanation = "This item was recommended to you because:\n"
        
        # Feature importance
        top_user_features = sorted(explanations['feature_importance']['user_features'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True)[:3]
        top_item_features = sorted(explanations['feature_importance']['item_features'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True)[:3]
        
        nl_explanation += "1. Your profile matches this item well, especially in terms of:\n"
        for feature, importance in top_user_features:
            nl_explanation += f"   - {feature.replace('_', ' ').title()}\n"
        
        nl_explanation += "2. This item has the following appealing characteristics:\n"
        for feature, importance in top_item_features:
            nl_explanation += f"   - {feature.replace('_', ' ').title()}\n"
        
        # Similar items
        nl_explanation += "3. It is similar to other items you've liked in the past.\n"
        
        # Temporal factors
        if explanations['temporal_factors']['recency_effect'] > 0.5:
            nl_explanation += "4. It aligns with your recent interests.\n"
        if explanations['temporal_factors']['diversity_trend'] > 0:
            nl_explanation += "5. It adds diversity to your recent interactions.\n"
        if explanations['temporal_factors']['seasonal_pattern']:
            nl_explanation += "6. It fits with your seasonal preferences.\n"
        
        # Counterfactual
        nl_explanation += "7. If you're not interested, consider adjusting these factors:\n"
        for feature, effect in explanations['counterfactual'][:2]:
            nl_explanation += f"   - {feature.replace('_', ' ').title()}: {effect:.2f}\n"
        
        return nl_explanation

    def visual_explanation(self, explanations):
        """
        Generate visual explanations (e.g., charts, word clouds).
        """
        visuals = {}
        
        # Feature importance bar chart
        plt.figure(figsize=(10, 6))
        features = list(explanations['feature_importance']['user_features'].keys()) + \
                   list(explanations['feature_importance']['item_features'].keys())
        importances = list(explanations['feature_importance']['user_features'].values()) + \
                      list(explanations['feature_importance']['item_features'].values())
        plt.bar(features, importances)
        plt.title('Feature Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        visuals['feature_importance_chart'] = plt

        # Word cloud of important features
        feature_importance_dict = {**explanations['feature_importance']['user_features'], 
                                   **explanations['feature_importance']['item_features']}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(feature_importance_dict)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Important Features Word Cloud')
        visuals['feature_wordcloud'] = plt

        # Temporal factors line plot
        plt.figure(figsize=(10, 6))
        x = range(len(explanations['temporal_factors']['diversity_trend']))
        plt.plot(x, explanations['temporal_factors']['diversity_trend'], label='Diversity Trend')
        plt.title('Temporal Diversity Trend')
        plt.xlabel('Time')
        plt.ylabel('Diversity Score')
        plt.legend()
        visuals['temporal_trend_chart'] = plt

        return visuals

    def calculate_recency_effect(self, recent_interactions):
        """
        Calculate the effect of recent interactions on the recommendation.
        """
        weights = np.linspace(0.1, 1, len(recent_interactions))
        recency_score = np.sum(weights * np.array(recent_interactions)) / np.sum(weights)
        return recency_score

    def calculate_diversity_trend(self, interaction_history):
        """
        Calculate the trend in interaction diversity over time.
        """
        window_size = min(30, len(interaction_history) // 3)
        diversity_scores = []
        
        for i in range(len(interaction_history) - window_size + 1):
            window = interaction_history[i:i+window_size]
            diversity = len(set(window)) / window_size
            diversity_scores.append(diversity)
        
        return diversity_scores

    def detect_seasonal_pattern(self, interaction_history):
        """
        Detect if there's a seasonal pattern in the user's interactions.
        """
        if len(interaction_history) < 365:
            return False
        
        daily_counts = np.zeros(365)
        for interaction in interaction_history:
            day = interaction % 365
            daily_counts[day] += 1
        
        autocorr = np.correlate(daily_counts, daily_counts, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
        if len(peaks) > 0 and (peaks[0] > 300 or peaks[0] < 400):
            return True
        
        return False
