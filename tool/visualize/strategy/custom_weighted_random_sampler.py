from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

    def update_weights(self, selected_indices, focus_mode):
        """
        Dynamically adjust weights based on TTAV focus mode.
        """
        # Reset to base weights first (e.g., 1.0 for all points)
        new_weights = self.base_weights.copy()
        
        if not selected_indices:
            self.weights = new_weights
            return

        # Define weight multipliers based on your final plan
        multiplier = 1.0
        if focus_mode == "balanced":
            multiplier = 2.0  # Focus area x2
        elif focus_mode == "fine":
            multiplier = 5.0  # Focus area x5 (preparing for adaptation)
            
        # Apply the multiplier to the selected focus center
        for idx in selected_indices:
            if idx < len(new_weights):
                new_weights[idx] *= multiplier
        
        # Update the actual sampler weights
        self.weights = new_weights
        print(f"[TTAV Sampler] Weights updated. Focus multiplier: {multiplier}")