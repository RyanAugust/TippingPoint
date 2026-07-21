import numpy as np
from scipy.optimize import minimize
from .models import MarketingReturnCurve

class PortfolioAllocator:
    """Optimizes budget allocation across multiple MarketingReturnCurve models."""

    def __init__(self, models):
        if not models:
            raise ValueError("At least one model must be provided.")
        self.models = models
        self.channel_names = [m.channel_name for m in models]

        # Ensure channel names are unique
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("All models must have unique channel_names.")

    def allocate_budget(self, total_budget, channel_bounds=None):
        """
        Finds the optimal spend distribution to maximize total return.

        Args:
            total_budget (float): Total budget to allocate.
            channel_bounds (dict, optional): Dictionary of (min_spend, max_spend) bounds
                                             keyed by channel_name.

        Returns:
            dict: The optimal allocation, marginal ROAS, and expected return.
        """
        n = len(self.models)

        # Determine bounds for each channel
        bounds = []
        for model in self.models:
            b = (0.0, total_budget)
            if channel_bounds and model.channel_name in channel_bounds:
                provided_b = channel_bounds[model.channel_name]
                b = (provided_b[0], min(provided_b[1], total_budget))
            bounds.append(b)

        def objective(spends):
            total_return = 0.0
            for i, model in enumerate(self.models):
                total_return += model.predict_incremental_return(spends[i])
            return -total_return

        def constraint(spends):
            return np.sum(spends) - total_budget

        cons = {'type': 'eq', 'fun': constraint}

        # Global optimization or multi-start can be better for S-curves.
        # We will use SLSQP with a simple proportional start, but if it fails to find
        # a good optimum, we will try a few random starts.

        best_res = None
        best_return = float('inf') # We are minimizing negative return

        # Start points: proportional, and budget-heavy on individual channels
        start_points = []

        # Proportional
        x0_prop = np.zeros(n)
        for i, b in enumerate(bounds):
            x0_prop[i] = b[0] # satisfy min bounds
        rem_budget = total_budget - np.sum(x0_prop)
        if rem_budget > 0:
            x0_prop += rem_budget / n
        start_points.append(x0_prop)

        # Random starts
        for _ in range(5):
            x_rand = np.random.rand(n)
            x_rand = x_rand / np.sum(x_rand) * total_budget
            # Clamping to bounds
            for i in range(n):
                x_rand[i] = np.clip(x_rand[i], bounds[i][0], bounds[i][1])
            start_points.append(x_rand)

        for x0 in start_points:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'disp': False, 'ftol': 1e-6, 'maxiter': 500}
            )

            # Strict budget check, sometimes SLSQP drifts slightly
            if np.abs(np.sum(res.x) - total_budget) < 1e-2 and res.fun < best_return:
                best_return = res.fun
                best_res = res

        # If all failed or drifted, just use the last one
        if best_res is None:
            best_res = res

        allocation = {self.models[i].channel_name: float(best_res.x[i]) for i in range(n)}
        mroas = {self.models[i].channel_name: float(self.models[i].predict_marginal_return(best_res.x[i])) for i in range(n)}
        expected_return = -float(best_res.fun)

        return {
            "total_budget": total_budget,
            "expected_total_return": expected_return,
            "overall_roas": expected_return / total_budget if total_budget > 0 else 0.0,
            "allocation": allocation,
            "marginal_roas_at_allocation": mroas,
            "success": best_res.success,
            "message": best_res.message
        }
