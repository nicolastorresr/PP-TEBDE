import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class CausalInference:
    def __init__(self, config):
        self.config = config

    def estimate_causal_effect(self, treatment, outcome, covariates):
        """
        Estimate the causal effect of a treatment on an outcome using linear regression.
        
        Args:
        treatment (np.array): Binary treatment assignment
        outcome (np.array): Observed outcome
        covariates (np.array): Covariate matrix
        
        Returns:
        float: Estimated causal effect
        """
        X = np.column_stack((treatment, covariates))
        model = LinearRegression()
        model.fit(X, outcome)
        return model.coef_[0]

    def propensity_score_matching(self, treatment, covariates):
        """
        Perform propensity score matching to balance treatment and control groups.
        
        Args:
        treatment (np.array): Binary treatment assignment
        covariates (np.array): Covariate matrix
        
        Returns:
        np.array: Matched indices for treatment and control groups
        """
        propensity_model = LinearRegression()
        propensity_model.fit(covariates, treatment)
        propensity_scores = propensity_model.predict(covariates)
        
        treated = np.where(treatment == 1)[0]
        control = np.where(treatment == 0)[0]
        
        matches = []
        for t in treated:
            match = control[np.argmin(np.abs(propensity_scores[t] - propensity_scores[control]))]
            matches.append((t, match))
        
        return np.array(matches)

    def difference_in_differences(self, pre_treatment, post_treatment, treatment_group):
        """
        Perform difference-in-differences analysis.
        
        Args:
        pre_treatment (np.array): Outcome values before treatment
        post_treatment (np.array): Outcome values after treatment
        treatment_group (np.array): Binary indicator for treatment group
        
        Returns:
        float: Estimated causal effect
        """
        diff_treated = np.mean(post_treatment[treatment_group == 1] - pre_treatment[treatment_group == 1])
        diff_control = np.mean(post_treatment[treatment_group == 0] - pre_treatment[treatment_group == 0])
        return diff_treated - diff_control

    def instrumental_variable(self, treatment, outcome, instrument):
        """
        Perform instrumental variable analysis.
        
        Args:
        treatment (np.array): Observed treatment
        outcome (np.array): Observed outcome
        instrument (np.array): Instrumental variable
        
        Returns:
        float: Estimated causal effect
        """
        first_stage = LinearRegression().fit(instrument.reshape(-1, 1), treatment)
        predicted_treatment = first_stage.predict(instrument.reshape(-1, 1))
        
        second_stage = LinearRegression().fit(predicted_treatment.reshape(-1, 1), outcome)
        return second_stage.coef_[0]

    def regression_discontinuity(self, running_variable, outcome, cutoff):
        """
        Perform regression discontinuity analysis.
        
        Args:
        running_variable (np.array): The variable determining treatment assignment
        outcome (np.array): Observed outcome
        cutoff (float): The cutoff value for treatment assignment
        
        Returns:
        float: Estimated causal effect
        """
        treatment = (running_variable >= cutoff).astype(int)
        
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(np.column_stack((running_variable, treatment)))
        
        model = LinearRegression().fit(X, outcome)
        
        effect = model.coef_[2]  # Coefficient of the treatment indicator
        return effect

    def sensitivity_analysis(self, treatment, outcome, covariates, effect_estimate, num_simulations=1000):
        """
        Perform sensitivity analysis to assess the robustness of causal estimates.
        
        Args:
        treatment (np.array): Binary treatment assignment
        outcome (np.array): Observed outcome
        covariates (np.array): Covariate matrix
        effect_estimate (float): Original causal effect estimate
        num_simulations (int): Number of simulations to run
        
        Returns:
        dict: Results of sensitivity analysis
        """
        effects = []
        for _ in range(num_simulations):
            # Add random noise to covariates
            noisy_covariates = covariates + np.random.normal(0, 0.1, covariates.shape)
            
            # Re-estimate causal effect
            noisy_effect = self.estimate_causal_effect(treatment, outcome, noisy_covariates)
            effects.append(noisy_effect)
        
        return {
            "original_estimate": effect_estimate,
            "mean_estimate": np.mean(effects),
            "std_estimate": np.std(effects),
            "ci_lower": np.percentile(effects, 2.5),
            "ci_upper": np.percentile(effects, 97.5)
        }

    def detect_temporal_bias(self, user_exposures, item_interactions, timestamps):
        """
        Detect temporal exposure bias using causal inference techniques.
        
        Args:
        user_exposures (np.array): User exposure to items over time
        item_interactions (np.array): User interactions with items
        timestamps (np.array): Timestamps of interactions
        
        Returns:
        dict: Results of temporal bias analysis
        """
        # Compute exposure intensity
        exposure_intensity = np.sum(user_exposures, axis=1) / (np.max(timestamps) - np.min(timestamps))
        
        # Define treatment as high exposure intensity
        treatment = (exposure_intensity > np.median(exposure_intensity)).astype(int)
        
        # Outcome is the diversity of item interactions
        outcome = np.array([len(np.unique(interactions)) for interactions in item_interactions])
        
        # Use timestamps as an instrumental variable
        iv_effect = self.instrumental_variable(treatment, outcome, timestamps)
        
        # Perform difference-in-differences analysis
        mid_point = np.median(timestamps)
        pre_treatment = outcome[timestamps < mid_point]
        post_treatment = outcome[timestamps >= mid_point]
        treatment_group = treatment[timestamps >= mid_point]
        did_effect = self.difference_in_differences(pre_treatment, post_treatment, treatment_group)
        
        # Sensitivity analysis
        sensitivity_results = self.sensitivity_analysis(treatment, outcome, timestamps.reshape(-1, 1), iv_effect)
        
        return {
            "iv_effect": iv_effect,
            "did_effect": did_effect,
            "sensitivity_analysis": sensitivity_results
        }
