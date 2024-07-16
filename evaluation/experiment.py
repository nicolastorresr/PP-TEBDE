import logging
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from models.pptebde import PPTEBDE
from data.data_loader import load_dataset
from metrics import (
    compute_ndcg, compute_map, compute_gini_coefficient, 
    compute_temporal_exposure_deviation, compute_bias_score,
    compute_diversity, compute_novelty, compute_serendipity,
    compute_coverage, compute_privacy_metric, compute_explainability_score
)
from visualization import (
    plot_bias_heatmap, plot_temporal_bias_trend, plot_feature_importance,
    plot_counterfactual_comparison, plot_recommendation_diversity,
    plot_bias_mitigation_effect, plot_privacy_utility_tradeoff
)

logger = logging.getLogger(__name__)

def run_experiment(model: PPTEBDE, test_data: Any, config: Dict) -> Dict[str, float]:
    """
    Run a full experiment with the PP-TEBDE model.

    Args:
    model (PPTEBDE): The PP-TEBDE model
    test_data (Any): Test dataset
    config (Dict): Configuration parameters

    Returns:
    Dict[str, float]: Dictionary of evaluation metrics
    """
    logger.info("Starting experiment")

    # Generate recommendations
    recommendations, scores = generate_recommendations(model, test_data)

    # Compute evaluation metrics
    metrics = compute_metrics(model, test_data, recommendations, scores, config)

    # Generate and evaluate explanations
    explanations = model.generate_explanations(test_data)
    metrics['explainability'] = evaluate_explanations(explanations, config)

    # Run ablation studies
    ablation_results = run_ablation_studies(model, test_data, config)
    metrics['ablation_studies'] = ablation_results

    # Visualize results
    visualize_results(metrics, recommendations, explanations, config)

    logger.info("Experiment completed")
    return metrics

def generate_recommendations(model: PPTEBDE, test_data: Any) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Generate recommendations for test data.

    Args:
    model (PPTEBDE): The PP-TEBDE model
    test_data (Any): Test dataset

    Returns:
    Tuple[List[List[int]], List[List[float]]]: Recommendations and their scores
    """
    recommendations = []
    scores = []
    for user, items, timestamps in tqdm(test_data, desc="Generating recommendations"):
        user_recs = model.predict(user, items, timestamps)
        sorted_items = sorted(zip(items, user_recs), key=lambda x: x[1], reverse=True)
        recommendations.append([item for item, _ in sorted_items])
        scores.append([score for _, score in sorted_items])
    return recommendations, scores

def compute_metrics(model: PPTEBDE, test_data: Any, recommendations: List[List[int]], 
                    scores: List[List[float]], config: Dict) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
    model (PPTEBDE): The PP-TEBDE model
    test_data (Any): Test dataset
    recommendations (List[List[int]]): Generated recommendations
    scores (List[List[float]]): Recommendation scores
    config (Dict): Configuration parameters

    Returns:
    Dict[str, float]: Dictionary of computed metrics
    """
    metrics = {}

    # Accuracy metrics
    ground_truth = [labels for _, _, labels in test_data]
    metrics['ndcg'] = np.mean([compute_ndcg(s, gt) for s, gt in zip(scores, ground_truth)])
    metrics['map'] = np.mean([compute_map(s, gt) for s, gt in zip(scores, ground_truth)])

    # Bias metrics
    item_exposures = compute_item_exposures(recommendations)
    metrics['gini_coefficient'] = compute_gini_coefficient(list(item_exposures.values()))
    user_exposures = compute_user_exposures(test_data, recommendations)
    metrics['temporal_exposure_deviation'] = compute_temporal_exposure_deviation(user_exposures)
    metrics['bias_score'] = compute_bias_score(metrics['gini_coefficient'], metrics['temporal_exposure_deviation'])

    # Diversity and novelty metrics
    metrics['diversity'] = compute_diversity(recommendations)
    item_popularity = compute_item_popularity(test_data)
    metrics['novelty'] = compute_novelty(recommendations, item_popularity)
    user_profiles = compute_user_profiles(test_data)
    metrics['serendipity'] = compute_serendipity(recommendations, user_profiles)
    metrics['coverage'] = compute_coverage(recommendations, config['total_items'])

    # Privacy metric
    metrics['privacy'] = compute_privacy_metric(model.privacy_param)

    return metrics

def evaluate_explanations(explanations: List[Dict], config: Dict) -> float:
    """
    Evaluate the quality of generated explanations.

    Args:
    explanations (List[Dict]): List of generated explanations
    config (Dict): Configuration parameters

    Returns:
    float: Explainability score
    """
    # In a real-world scenario, you would collect human ratings here
    # For this example, we'll simulate human ratings
    simulated_human_ratings = np.random.uniform(0.5, 1.0, len(explanations))
    return compute_explainability_score(explanations, simulated_human_ratings)

def run_ablation_studies(model: PPTEBDE, test_data: Any, config: Dict) -> Dict[str, Dict[str, float]]:
    """
    Run ablation studies by disabling different components of the model.

    Args:
    model (PPTEBDE): The PP-TEBDE model
    test_data (Any): Test dataset
    config (Dict): Configuration parameters

    Returns:
    Dict[str, Dict[str, float]]: Results of ablation studies
    """
    ablation_results = {}
    components = ['temporal_cf', 'attention_mechanism', 'federated_learning', 'bias_mitigation']

    for component in components:
        logger.info(f"Running ablation study: disabling {component}")
        model_copy = model.get_copy()
        model_copy.disable_component(component)
        recommendations, scores = generate_recommendations(model_copy, test_data)
        metrics = compute_metrics(model_copy, test_data, recommendations, scores, config)
        ablation_results[component] = metrics

    return ablation_results

def visualize_results(metrics: Dict[str, float], recommendations: List[List[int]], 
                      explanations: List[Dict], config: Dict):
    """
    Visualize experimental results.

    Args:
    metrics (Dict[str, float]): Computed evaluation metrics
    recommendations (List[List[int]]): Generated recommendations
    explanations (List[Dict]): Generated explanations
    config (Dict): Configuration parameters
    """
    # Plot bias heatmap
    bias_scores = compute_bias_scores(recommendations, config)
    plot_bias_heatmap(bias_scores, "Temporal Exposure Bias Heatmap")

    # Plot temporal bias trend
    bias_over_time = compute_bias_over_time(recommendations, config)
    plot_temporal_bias_trend(bias_over_time, "Temporal Bias Trend")

    # Plot feature importance
    feature_importance = explanations[0]['feature_importance']  # Using the first explanation as an example
    plot_feature_importance(feature_importance, "Feature Importance in Recommendations")

    # Plot counterfactual comparison
    cf_example = explanations[0]['counterfactual']  # Using the first explanation as an example
    plot_counterfactual_comparison(cf_example['original_prob'], cf_example['cf_prob'], 
                                   cf_example['feature'], cf_example['original_value'], 
                                   cf_example['cf_value'], "Counterfactual Comparison")

    # Plot recommendation diversity
    diversity_scores = [compute_diversity(rec) for rec in recommendations]
    plot_recommendation_diversity(diversity_scores, "Recommendation Diversity Over Time")

    # Plot bias mitigation effect
    before_mitigation = metrics['ablation_studies']['bias_mitigation']['bias_score']
    after_mitigation = metrics['bias_score']
    plot_bias_mitigation_effect([before_mitigation], [after_mitigation], "Effect of Bias Mitigation")

    # Plot privacy-utility tradeoff
    privacy_levels = [0.1, 0.5, 1.0, 1.5, 2.0]  # Example privacy levels
    utility_scores = [run_privacy_experiment(model, test_data, epsilon, config) for epsilon in privacy_levels]
    plot_privacy_utility_tradeoff(privacy_levels, utility_scores, "Privacy-Utility Tradeoff")

def compute_item_exposures(recommendations: List[List[int]]) -> Dict[int, float]:
    """Compute item exposure frequencies."""
    exposures = {}
    for rec_list in recommendations:
        for item in rec_list:
            exposures[item] = exposures.get(item, 0) + 1
    return exposures

def compute_user_exposures(test_data: Any, recommendations: List[List[int]]) -> Dict[int, Dict[int, float]]:
    """Compute user exposures over time."""
    user_exposures = {}
    for (user, _, timestamp), rec_list in zip(test_data, recommendations):
        if user not in user_exposures:
            user_exposures[user] = {}
        user_exposures[user][timestamp] = len(set(rec_list))
    return user_exposures

def compute_item_popularity(test_data: Any) -> Dict[int, float]:
    """Compute item popularity based on interaction frequency."""
    popularity = {}
    for _, items, _ in test_data:
        for item in items:
            popularity[item] = popularity.get(item, 0) + 1
    return popularity

def compute_user_profiles(test_data: Any) -> Dict[int, List[int]]:
    """Compute user profiles based on historical interactions."""
    profiles = {}
    for user, items, _ in test_data:
        if user not in profiles:
            profiles[user] = []
        profiles[user].extend(items)
    return profiles

def run_privacy_experiment(model: PPTEBDE, test_data: Any, epsilon: float, config: Dict) -> float:
    """Run experiment with different privacy levels."""
    model_copy = model.get_copy()
    model_copy.set_privacy_param(epsilon)
    recommendations, scores = generate_recommendations(model_copy, test_data)
    metrics = compute_metrics(model_copy, test_data, recommendations, scores, config)
    return metrics['ndcg']  # Return NDCG as utility score

if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yaml')

    # Load data
    train_data, val_data, test_data = load_dataset(config['dataset'])

    # Initialize and train model
    model = PPTEBDE(config)
    model.train(train_data, val_data)

    # Run experiment
    results = run_experiment(model, test_data, config)

    # Log results
    logger.info(f"Experiment results: {results}")
