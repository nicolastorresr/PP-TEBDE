import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Tuple
from collections import defaultdict

class BiasDetector:
    def __init__(self, config):
        self.config = config
        self.temporal_window = config.temporal_window
        self.bias_threshold = config.bias_threshold

    def detect_temporal_exposure_bias(self, user_item_interactions: Dict[int, Dict[int, List[float]]]) -> float:
        """
        Detect temporal exposure bias in user-item interactions.
        
        :param user_item_interactions: Dictionary of user interactions, where each user has a dictionary
                                       of items they interacted with and the timestamps of interactions.
        :return: A bias score between 0 and 1, where higher values indicate more bias.
        """
        exposure_distributions = self._compute_exposure_distributions(user_item_interactions)
        temporal_skew = self._compute_temporal_skew(exposure_distributions)
        return temporal_skew

    def _compute_exposure_distributions(self, user_item_interactions: Dict[int, Dict[int, List[float]]]) -> Dict[int, np.ndarray]:
        """
        Compute exposure distributions for each user over time.
        """
        exposure_distributions = {}
        for user, items in user_item_interactions.items():
            timestamps = [t for item_times in items.values() for t in item_times]
            min_time, max_time = min(timestamps), max(timestamps)
            time_range = max_time - min_time
            num_bins = int(time_range / self.temporal_window) + 1
            
            hist, _ = np.histogram(timestamps, bins=num_bins, range=(min_time, max_time))
            exposure_distributions[user] = hist / np.sum(hist)
        
        return exposure_distributions

    def _compute_temporal_skew(self, exposure_distributions: Dict[int, np.ndarray]) -> float:
        """
        Compute temporal skew based on exposure distributions.
        """
        skews = []
        for user, distribution in exposure_distributions.items():
            skew = stats.skew(distribution)
            skews.append(abs(skew))
        
        avg_skew = np.mean(skews)
        normalized_skew = 2 * (avg_skew - np.min(skews)) / (np.max(skews) - np.min(skews)) - 1
        return (normalized_skew + 1) / 2  # Scale to [0, 1]

    def detect_popularity_bias(self, item_interactions: Dict[int, int]) -> float:
        """
        Detect popularity bias in item interactions.
        
        :param item_interactions: Dictionary of item interaction counts.
        :return: Gini coefficient as a measure of popularity bias.
        """
        values = sorted(item_interactions.values())
        cumulative = np.cumsum(values)
        return (np.sum((2 * np.arange(1, len(values) + 1) - len(values) - 1) * values) / 
                (len(values) * np.sum(values)))

    def detect_demographic_bias(self, user_demographics: Dict[int, Dict[str, str]], 
                                item_interactions: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Detect demographic bias in item interactions.
        
        :param user_demographics: Dictionary of user demographics.
        :param item_interactions: Dictionary of user-item interactions.
        :return: Dictionary of bias scores for each demographic attribute.
        """
        demographic_attributes = list(next(iter(user_demographics.values())).keys())
        bias_scores = {}
        
        for attribute in demographic_attributes:
            attribute_distribution = defaultdict(int)
            interaction_distribution = defaultdict(int)
            
            for user, demographics in user_demographics.items():
                attribute_value = demographics[attribute]
                attribute_distribution[attribute_value] += 1
                if user in item_interactions:
                    interaction_distribution[attribute_value] += len(item_interactions[user])
            
            attribute_dist = np.array(list(attribute_distribution.values()))
            interaction_dist = np.array(list(interaction_distribution.values()))
            
            attribute_dist = attribute_dist / np.sum(attribute_dist)
            interaction_dist = interaction_dist / np.sum(interaction_dist)
            
            bias_scores[attribute] = np.sum(np.abs(attribute_dist - interaction_dist)) / 2
        
        return bias_scores

    def is_biased(self, bias_score: float) -> bool:
        """
        Determine if the system is biased based on the computed bias score.
        """
        return bias_score > self.bias_threshold

    def analyze_bias(self, user_item_interactions: Dict[int, Dict[int, List[float]]],
                     item_interactions: Dict[int, int],
                     user_demographics: Dict[int, Dict[str, str]]) -> Dict[str, float]:
        """
        Perform a comprehensive bias analysis.
        
        :param user_item_interactions: Dictionary of user-item interactions with timestamps.
        :param item_interactions: Dictionary of item interaction counts.
        :param user_demographics: Dictionary of user demographics.
        :return: Dictionary of different bias scores.
        """
        temporal_bias = self.detect_temporal_exposure_bias(user_item_interactions)
        popularity_bias = self.detect_popularity_bias(item_interactions)
        demographic_bias = self.detect_demographic_bias(user_demographics, user_item_interactions)
        
        overall_bias = (temporal_bias + popularity_bias + np.mean(list(demographic_bias.values()))) / 3
        
        return {
            "temporal_exposure_bias": temporal_bias,
            "popularity_bias": popularity_bias,
            "demographic_bias": demographic_bias,
            "overall_bias": overall_bias
        }
