import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score
from typing import List, Dict, Tuple

def compute_ndcg(predictions: List[float], ground_truth: List[float], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
    predictions (List[float]): Predicted scores or probabilities
    ground_truth (List[float]): True relevance scores
    k (int): Number of top items to consider

    Returns:
    float: NDCG@k score
    """
    return ndcg_score([ground_truth], [predictions], k=k)

def compute_map(predictions: List[float], ground_truth: List[float]) -> float:
    """
    Compute Mean Average Precision (MAP).

    Args:
    predictions (List[float]): Predicted scores or probabilities
    ground_truth (List[float]): True relevance scores (binary: 0 or 1)

    Returns:
    float: MAP score
    """
    return average_precision_score(ground_truth, predictions)

def compute_gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient of the given values.

    Args:
    values (List[float]): List of values

    Returns:
    float: Gini coefficient
    """
    sorted_values = np.sort(values)
    index = np.arange(1, len(values) + 1)
    n = len(values)
    return (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(values))

def compute_temporal_exposure_deviation(user_exposures: Dict[int, Dict[int, float]]) -> float:
    """
    Compute Temporal Exposure Deviation (TED) for all users.

    Args:
    user_exposures (Dict[int, Dict[int, float]]): Dictionary of user exposures over time

    Returns:
    float: Average TED across all users
    """
    user_teds = []
    for user, exposures in user_exposures.items():
        timestamps = sorted(exposures.keys())
        exposure_values = [exposures[t] for t in timestamps]
        
        # Compute area under the exposure curve
        actual_area = np.trapz(exposure_values, timestamps)
        
        # Compute area under uniform exposure
        uniform_area = 0.5 * (timestamps[-1] - timestamps[0]) * (exposure_values[0] + exposure_values[-1])
        
        # Compute TED for this user
        ted = abs(actual_area - uniform_area) / uniform_area
        user_teds.append(ted)
    
    return np.mean(user_teds)

def compute_bias_score(gini: float, ted: float, alpha: float = 0.5) -> float:
    """
    Compute overall bias score combining Gini coefficient and TED.

    Args:
    gini (float): Gini coefficient
    ted (float): Temporal Exposure Deviation
    alpha (float): Weight for combining Gini and TED (default: 0.5)

    Returns:
    float: Overall bias score
    """
    return alpha * gini + (1 - alpha) * ted

def compute_diversity(recommendations: List[List[int]]) -> float:
    """
    Compute diversity of recommendations based on unique items.

    Args:
    recommendations (List[List[int]]): List of recommendation lists for each user

    Returns:
    float: Diversity score
    """
    all_items = set()
    for rec_list in recommendations:
        all_items.update(rec_list)
    return len(all_items) / sum(len(rec) for rec in recommendations)

def compute_novelty(recommendations: List[List[int]], item_popularity: Dict[int, float]) -> float:
    """
    Compute novelty of recommendations based on item popularity.

    Args:
    recommendations (List[List[int]]): List of recommendation lists for each user
    item_popularity (Dict[int, float]): Dictionary of item popularities

    Returns:
    float: Novelty score
    """
    novelty_scores = []
    for rec_list in recommendations:
        novelty = -np.mean([np.log2(item_popularity.get(item, 1e-10)) for item in rec_list])
        novelty_scores.append(novelty)
    return np.mean(novelty_scores)

def compute_serendipity(recommendations: List[List[int]], user_profiles: Dict[int, List[int]], k: int = 10) -> float:
    """
    Compute serendipity of recommendations based on user profiles.

    Args:
    recommendations (List[List[int]]): List of recommendation lists for each user
    user_profiles (Dict[int, List[int]]): Dictionary of user profiles (items they've interacted with)
    k (int): Number of top recommendations to consider

    Returns:
    float: Serendipity score
    """
    serendipity_scores = []
    for user_id, rec_list in enumerate(recommendations):
        profile = set(user_profiles.get(user_id, []))
        unexpected_items = [item for item in rec_list[:k] if item not in profile]
        serendipity = len(unexpected_items) / k
        serendipity_scores.append(serendipity)
    return np.mean(serendipity_scores)

def compute_coverage(recommendations: List[List[int]], total_items: int) -> float:
    """
    Compute catalog coverage of recommendations.

    Args:
    recommendations (List[List[int]]): List of recommendation lists for each user
    total_items (int): Total number of items in the catalog

    Returns:
    float: Coverage score
    """
    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)
    return len(recommended_items) / total_items

def compute_privacy_metric(epsilon: float) -> float:
    """
    Compute privacy metric based on differential privacy epsilon.

    Args:
    epsilon (float): Differential privacy parameter

    Returns:
    float: Privacy score (higher is better)
    """
    return 1 / (1 + epsilon)

def compute_explainability_score(explanations: List[Dict], human_ratings: List[float]) -> float:
    """
    Compute explainability score based on human ratings of explanations.

    Args:
    explanations (List[Dict]): List of explanation dictionaries
    human_ratings (List[float]): List of human-provided ratings for explanations

    Returns:
    float: Explainability score
    """
    return np.mean(human_ratings)
