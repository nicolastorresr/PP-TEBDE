import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def plot_bias_heatmap(bias_scores: Dict[Tuple[int, int], float], 
                      title: str = "Temporal Exposure Bias Heatmap"):
    """
    Plot a heatmap of temporal exposure bias scores.
    
    Args:
    bias_scores (Dict[Tuple[int, int], float]): Dictionary of bias scores keyed by (user, item) tuples
    title (str): Title of the plot
    """
    # Convert bias scores to a DataFrame
    df = pd.DataFrame([(u, i, s) for (u, i), s in bias_scores.items()], 
                      columns=['User', 'Item', 'Bias Score'])
    pivot_df = df.pivot(index='User', columns='Item', values='Bias Score')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, cmap='YlOrRd', annot=False)
    plt.title(title)
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.show()

def plot_temporal_bias_trend(bias_over_time: List[Tuple[int, float]],
                             title: str = "Temporal Bias Trend"):
    """
    Plot the trend of temporal bias over time.
    
    Args:
    bias_over_time (List[Tuple[int, float]]): List of (timestamp, bias_score) tuples
    title (str): Title of the plot
    """
    timestamps, bias_scores = zip(*bias_over_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, bias_scores, marker='o')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Bias Score')
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_importance: Dict[str, float],
                            title: str = "Feature Importance in Recommendations"):
    """
    Plot a bar chart of feature importance scores.
    
    Args:
    feature_importance (Dict[str, float]): Dictionary of feature importance scores
    title (str): Title of the plot
    """
    features = list(feature_importance.keys())
    scores = list(feature_importance.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(features, scores)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_counterfactual_comparison(original: float, counterfactual: float, 
                                   feature: str, original_value: str, cf_value: str,
                                   title: str = "Counterfactual Comparison"):
    """
    Plot a comparison between original and counterfactual recommendation probabilities.
    
    Args:
    original (float): Original recommendation probability
    counterfactual (float): Counterfactual recommendation probability
    feature (str): The feature that was changed
    original_value (str): Original value of the changed feature
    cf_value (str): Counterfactual value of the changed feature
    title (str): Title of the plot
    """
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Original', 'Counterfactual'], [original, counterfactual], color=['blue', 'orange'])
    plt.title(title)
    plt.ylabel('Recommendation Probability')
    
    # Adding the text on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.annotate(f"{feature}: {original_value}", xy=(0, original), xytext=(0, original+0.1),
                 ha='center', va='bottom', arrowprops=dict(arrowstyle='->'))
    plt.annotate(f"{feature}: {cf_value}", xy=(1, counterfactual), xytext=(1, counterfactual+0.1),
                 ha='center', va='bottom', arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()

def plot_recommendation_diversity(diversity_scores: List[float], 
                                  title: str = "Recommendation Diversity Over Time"):
    """
    Plot the trend of recommendation diversity over time.
    
    Args:
    diversity_scores (List[float]): List of diversity scores over time
    title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(diversity_scores)), diversity_scores, marker='o')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Diversity Score')
    plt.grid(True)
    plt.show()

def plot_bias_mitigation_effect(before_mitigation: List[float], after_mitigation: List[float],
                                title: str = "Effect of Bias Mitigation"):
    """
    Plot a comparison of bias scores before and after mitigation.
    
    Args:
    before_mitigation (List[float]): Bias scores before mitigation
    after_mitigation (List[float]): Bias scores after mitigation
    title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot([before_mitigation, after_mitigation], labels=['Before Mitigation', 'After Mitigation'])
    plt.title(title)
    plt.ylabel('Bias Score')
    plt.grid(True)
    plt.show()

def plot_privacy_utility_tradeoff(privacy_levels: List[float], utility_scores: List[float],
                                  title: str = "Privacy-Utility Tradeoff"):
    """
    Plot the tradeoff between privacy levels and utility scores.
    
    Args:
    privacy_levels (List[float]): List of privacy levels (e.g., epsilon values in differential privacy)
    utility_scores (List[float]): Corresponding utility scores (e.g., NDCG)
    title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(privacy_levels, utility_scores, marker='o')
    plt.title(title)
    plt.xlabel('Privacy Level (ε)')
    plt.ylabel('Utility Score')
    plt.grid(True)
    plt.show()
