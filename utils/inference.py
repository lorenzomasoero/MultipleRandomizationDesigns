"""
Inference utilities for Multiple Randomization Designs (MRDs)

This module contains all experiment classes and inference computation methods
for running double randomized experiments and computing causal effects.

Classes:
    - Simple_Double_Randomized_Experiment: Core MRD experiment class
    - SimpleDoubleRandomizedExperiment: Extended experiment class with data generators  
    - SimpleDoubleRandomizedExperimentWithLocalInterference: Local interference version
    - Simple_Double_Randomized_Experiment_Spending: Spending/utility-based experiments

Functions:
    - Helper functions for utility computation, budget optimization, etc.
"""

import numpy as np
import scipy.stats as spst
import time
from tqdm import tqdm
from tqdm import tqdm_notebook
import multiprocessing as mp
from numba import jit, njit
import typing as t


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_utility_parameters(I=321, J=19, num_purchases=5, **kwargs):
    """Draw utility parameters for spending experiments"""
    # Simplified version - returns basic parameters for spending experiments
    return {
        'I': I,
        'J': J, 
        'num_purchases': num_purchases,
        'seed': kwargs.get('seed', 42)
    }


def max_utility_within_budget(utilities, prices, budget):
    """Dynamic programming solution for knapsack problem"""
    n = len(utilities)
    budget = int(np.round(budget))
    prices = np.round(prices).astype(int)
    
    # Initialize DP array
    dp = np.zeros(budget + 1, dtype=int)
    keep = np.zeros((n, budget + 1), dtype=bool)
    
    # Fill the DP array
    for i in range(n):
        for j in range(budget, prices[i] - 1, -1):
            if dp[j] < utilities[i] + dp[j - prices[i]]:
                dp[j] = utilities[i] + dp[j - prices[i]]
                keep[i][j] = True
    
    # Backtrack to find the items included in the optimal solution
    result = []
    remaining_budget = budget
    for i in range(n - 1, -1, -1):
        if remaining_budget >= prices[i] and keep[i][remaining_budget]:
            result.append(i)
            remaining_budget -= prices[i]
    
    result.reverse()
    return result

def greedy_approximation_max_utility(utilities, prices, budget):
    """Greedy approximation for knapsack problem"""
    ratio = utilities / prices
    indices = np.argsort(ratio)[::-1]
    
    total_utility = 0
    total_price = 0
    selected_indices = []
    
    for i in indices:
        if total_price + prices[i] <= budget:
            total_price += prices[i]
            total_utility += utilities[i]
            selected_indices.append(i)
    
    return selected_indices

# Numba-accelerated helper functions
@njit
def numba_set_seed(seed: int) -> None:
    np.random.seed(seed)

@njit
def exact_random_sample(n_units: int, size: int) -> np.ndarray:
    sample = np.zeros(n_units)
    sampled_indices = np.random.choice(n_units, size=size, replace=False)
    sample[sampled_indices] = 1
    return sample

@njit
def double_randomized_sample(I: int, I1: int, J: int, J1: int) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_I = exact_random_sample(I, I1)
    w_J = exact_random_sample(J, J1)
    return w_I, w_J, np.outer(w_I, w_J)

@njit
def get_types(w_I: np.ndarray, w_J: np.ndarray) -> np.ndarray:
    I, J = w_I.shape[0], w_J.shape[0]
    return np.outer(w_I, np.ones(J)) + 2 * np.outer(np.ones(I), w_J)

# Constants
SDRD_TYPES = [0, 1, 2, 3]


# ============================================================================
# MAIN EXPERIMENT CLASSES
# ============================================================================

class Simple_Double_Randomized_Experiment(object):
    """
    Core class for simple double randomized experiments with potential outcomes.
    
    The 4 treatment types are ordered as "c, im, iv, t" where:
    - c (0): control-control 
    - im (1): inactive movies, active viewers
    - iv (2): active movies, inactive viewers  
    - t (3): treated (both active)
    
    Args:
        potential_outcomes: numpy array of shape (4, I, J) containing potential outcomes
        active: array [I_1, J_1] specifying number of active units
        num_treatments: number of Monte Carlo experiments
        seed: random seed for reproducibility
    """
    
    def __init__(self, potential_outcomes, active, num_treatments, seed):
        self.potential_outcomes = potential_outcomes
        self.active = active
        self.num_treatments = num_treatments
        self.seed = seed
        self.total = np.asarray(potential_outcomes.shape[:2])
        
        assert (self.active[0] <= self.potential_outcomes.shape[0]) and \
               (self.active[1] <= self.potential_outcomes.shape[1]), 'Need I_1 <= I and J_1 <= J'
    
    def draw_treatments(self):
        """Draw random treatment assignments for all experiments"""
        # Ensure proper array types
        if type(self.total) == list:
            self.total = np.asarray(self.total)
        if self.total.dtype != int:
            self.total = self.total.astype(int)
        if type(self.active) == list:
            self.active = np.asarray(self.active)
        if self.active.dtype != int:
            self.active = self.active.astype(int)

        assert len(self.active) == 2 and self.active.shape == self.total.shape, \
            'Active and Total must both have length 2'
        assert (self.active <= self.total).all(), 'Active must be <= Total'

        I, J = self.total
        np.random.seed(self.seed)
        
        # Initialize arrays
        active_sets_M = np.zeros([self.num_treatments, I], dtype=int)
        active_sets_V = np.zeros([self.num_treatments, J], dtype=int)
        treatments = np.zeros([self.num_treatments, I, J], dtype=int)
        types = np.zeros([self.num_treatments, I, J], dtype=int)
        
        print('Drawing random assignments...')
        for n_mc in tqdm(range(self.num_treatments)):
            # Draw active sets
            active_sets_M[n_mc][np.random.choice(I, self.active[0], replace=False)] = 1
            active_sets_V[n_mc][np.random.choice(J, self.active[1], replace=False)] = 1
            
            # Create treatment matrix
            treatments[n_mc] = np.multiply(active_sets_M[n_mc][:, np.newaxis], 
                                         active_sets_V[n_mc][np.newaxis, :])
            
            # Compute types: 0=cc, 1=im, 2=iv, 3=t
            types[n_mc] = (active_sets_M[n_mc][:, np.newaxis] + 
                          2 * active_sets_V[n_mc][np.newaxis, :])
                          
        return treatments, types, active_sets_M, active_sets_V

    def draw_types(self):
        """Draw treatment types for all experiments"""
        return self.draw_treatments()[1]
    
    def draw_realized_outcomes(self, types):
        """Draw realized outcomes given treatment types"""
        print('Instantiating array of observed data across experiments...')
        I, J = self.total
        num_outcomes = np.max(types[0]) + 1
        
        # Convert types to one-hot encoding
        flat_types = np.ndarray.flatten(types)
        flat_types_vec = np.zeros((flat_types.size, num_outcomes))
        flat_types_vec[np.arange(flat_types.size), flat_types] = 1
        types_vec = flat_types_vec.reshape(self.num_treatments, I, J, num_outcomes)
        
        # Select appropriate potential outcomes
        realized_outcomes = np.multiply(self.potential_outcomes[np.newaxis], types_vec).sum(axis=-1)
        return realized_outcomes

    def compute_population_average_effects(self):
        """Compute population average effects for each treatment type"""
        return np.apply_over_axes(np.mean, self.potential_outcomes, (0, 1)).reshape(4, 1)

    def compute_population_spillover_effects(self, population_average_effects):
        """Compute population spillover effects"""
        spillover_direct = (population_average_effects[-1] + population_average_effects[0] - 
                           population_average_effects[1] - population_average_effects[2])
        spillover_M = population_average_effects[1] - population_average_effects[0]
        spillover_V = population_average_effects[2] - population_average_effects[0]
        spillover_pairs = population_average_effects[-1] - population_average_effects[0]
        return spillover_direct, spillover_M, spillover_V, spillover_pairs

    def compute_covariance_coefficients(self):
        """Compute covariance coefficients for variance estimation"""
        I_1, J_1 = self.active
        I, J = self.total
        I_0, J_0 = self.total - self.active

        alphas = np.zeros([4, 2])
        alphas[:, 0] = [I_1/(I_0*(I-1)), I_0/(I_1*(I-1)), I_1/(I_0*(I-1)), I_0/(I_1*(I-1))]
        alphas[:, 1] = [J_1/(J_0*(J-1)), J_1/(J_0*(J-1)), J_0/(J_1*(J-1)), J_0/(J_1*(J-1))]
        
        sample_sizes = np.zeros([4, 2], dtype=str)
        sample_sizes[:, 0] = ['0', '1', '0', '1']
        sample_sizes[:, 1] = ['0', '0', '1', '1']

        coefs = np.zeros([4, 4, 2])
        for v_ in range(4):
            for v__ in range(4):
                coefs[v_, v__] = (alphas[v_] * (sample_sizes[v_] == sample_sizes[v__]) - 
                                np.asarray([1/I, 1/J]) * (sample_sizes[v_] != sample_sizes[v__]))
        return coefs

    def compute_population_covariance_average_effects(self):
        """Compute population covariance matrix of average effects"""
        I, J = self.total
        covariance_average_effects = np.zeros([4, 4])
        coefs = self.compute_covariance_coefficients()

        for v_ in range(4):
            for v__ in range(4):
                beta_M, beta_V = coefs[v_, v__]

                # Movie effects
                y_dot_i_M_s = (self.potential_outcomes[:, :, v_].mean(axis=1) - 
                              self.potential_outcomes[:, :, v_].mean())
                y_dot_i_M_s2 = (self.potential_outcomes[:, :, v__].mean(axis=1) - 
                               self.potential_outcomes[:, :, v__].mean())
                y_dot_M = np.mean(y_dot_i_M_s * y_dot_i_M_s2)

                # Viewer effects
                y_dot_j_V_s = (self.potential_outcomes[:, :, v_].mean(axis=0) - 
                              self.potential_outcomes[:, :, v_].mean())
                y_dot_j_V_s2 = (self.potential_outcomes[:, :, v__].mean(axis=0) - 
                               self.potential_outcomes[:, :, v__].mean())
                y_dot_V = np.mean(y_dot_j_V_s * y_dot_j_V_s2)

                # Mixed terms
                y_dot_ij_MV_s = (self.potential_outcomes[:, :, v_] - 
                                np.mean(self.potential_outcomes[:, :, v_], axis=0)[np.newaxis, :] - 
                                np.mean(self.potential_outcomes[:, :, v_], axis=1)[:, np.newaxis] + 
                                self.potential_outcomes[:, :, v_].mean())
                y_dot_ij_MV_s2 = (self.potential_outcomes[:, :, v__] - 
                                 np.mean(self.potential_outcomes[:, :, v__], axis=0)[np.newaxis, :] - 
                                 np.mean(self.potential_outcomes[:, :, v__], axis=1)[:, np.newaxis] + 
                                 self.potential_outcomes[:, :, v__].mean())
                y_dot_MV = np.sum(y_dot_ij_MV_s * y_dot_ij_MV_s2) / (I * J)

                covariance_average_effects[v_, v__] = (beta_M * y_dot_M + beta_V * y_dot_V + 
                                                     beta_M * beta_V * y_dot_MV)

        return covariance_average_effects

    def compute_population_variance_spillovers(self):
        """Compute population variance of spillover effects"""
        covariance_mat = self.compute_population_covariance_average_effects()
        coefs = np.asarray([1, -1, -1, 1]).reshape(4, 1)
        coefs = coefs * coefs.T
        
        var_direct = np.sum(covariance_mat * coefs)
        var_spillover_M = np.sum(covariance_mat[:2, :2] * coefs[:2, :2])
        cv__ = covariance_mat[[0, 2]]
        var_spillover_V = np.sum(cv__[:, [0, 2]] * coefs[:2, :2])
        cv__, cf__ = covariance_mat[[0, -1]], coefs[[0, -1]]
        var_spillover_pairs = np.sum(cv__[:, [0, -1]] * cf__[:, [0, -1]])
        
        variance_spillovers = var_direct, var_spillover_M, var_spillover_V, var_spillover_pairs
        return np.asarray(variance_spillovers).T

    def compute_sample_average_effects(self, realized_outcomes, types):
        """Compute sample average effects for each experiment"""
        unique_values = [0, 1, 2, 3]
        sample_average_effect = np.zeros([self.num_treatments, len(unique_values)])
        
        for n_mc in tqdm(range(self.num_treatments)):
            Y = realized_outcomes[n_mc]
            types_ = types[n_mc]
            for v_, v in enumerate(unique_values):
                sample_average_effect[n_mc, v_] = Y[np.where(types_ == v)].mean()
                
        return sample_average_effect

    def compute_sample_spillover_effects(self, sample_average_effect):
        """Compute sample spillover effects"""
        sample_tau_direct = (sample_average_effect[:, -1] - sample_average_effect[:, 1] - 
                            sample_average_effect[:, 2] + sample_average_effect[:, 0])
        sample_tau_spillover_M = sample_average_effect[:, 1] - sample_average_effect[:, 0]
        sample_tau_spillover_V = sample_average_effect[:, 2] - sample_average_effect[:, 0]
        sample_tau_spillover_pairs = sample_average_effect[:, -1] - sample_average_effect[:, 0]
        
        return np.asarray([sample_tau_direct, sample_tau_spillover_M, 
                          sample_tau_spillover_V, sample_tau_spillover_pairs]).T

    def compute_sample_variance_average_effects(self, realized_outcomes, types):
        """Compute sample variance of average effects with bias correction"""
        I, J = self.total
        I_1, J_1 = self.active
        I_0, J_0 = I - I_1, J - J_1
        I_s_ls = np.asarray([I_0, I_1, I_0, I_1])
        J_s_ls = np.asarray([J_0, J_0, J_1, J_1])
        alpha_M_ls = (I - I_s_ls) / ((I - 1) * I_s_ls)
        alpha_V_ls = (J - J_s_ls) / ((J - 1) * J_s_ls)
        
        sample_variance_average_effects = np.zeros([self.num_treatments, 4])
        bias_correction_M = np.zeros([self.num_treatments, 4])
        bias_correction_V = np.zeros([self.num_treatments, 4])
        
        for n_mc in tqdm(range(self.num_treatments)):
            Y = realized_outcomes[n_mc]
            types_ = types[n_mc]

            for s in range(4):
                I_s, J_s = I_s_ls[s], J_s_ls[s]
                if I_s * J_s < 2:
                    continue
                    
                Y_s = Y[types_ == s].reshape([I_s, J_s])
                alpha_M, alpha_V = alpha_M_ls[s], alpha_V_ls[s]
                xi = 1 - alpha_M - alpha_V + alpha_M * alpha_V

                hat_Y_s = Y_s.mean()
                hat_Y_s_M = Y_s.mean(axis=1)
                hat_Y_s_V = Y_s.mean(axis=0)

                tilde_sigma_M = np.sum((hat_Y_s_M - hat_Y_s)**2)
                tilde_sigma_V = np.sum((hat_Y_s_V - hat_Y_s)**2)
                tilde_sigma_MV = np.sum((Y_s - hat_Y_s_M[:, np.newaxis] - 
                                       hat_Y_s_V[np.newaxis, :] + hat_Y_s)**2)

                sample_variance_average_effects[n_mc, s] = (
                    (alpha_M / I_s * tilde_sigma_M) + 
                    (alpha_V / J_s * tilde_sigma_V) + 
                    (alpha_M * alpha_V / (I_s * J_s) * tilde_sigma_MV))

                if J_s > 1:
                    bias_correction_M[n_mc, s] = (1/I_s * (J-J_s)/(J_s * (J_s-1) * J) * 
                                                 np.sum((Y_s - Y_s.mean(axis=1)[:, np.newaxis])**2))
                if I_s > 1:
                    bias_correction_V[n_mc, s] = (1/J_s * (I-I_s)/(I_s * (I_s-1) * I) * 
                                                 np.sum((Y_s - Y_s.mean(axis=0)[np.newaxis, :])**2))

        hat_sig = (sample_variance_average_effects / xi - 
                  alpha_M_ls / (1 - alpha_M_ls) * bias_correction_M - 
                  alpha_V_ls / (1 - alpha_V_ls) * bias_correction_V)
        
        if np.min(hat_sig) < 0:
            hat_sig[hat_sig <= 0] = -np.max(hat_sig[hat_sig <= 0])
        return hat_sig

    def compute_sample_variance_spillovers(self, sample_variance_average_effects):
        """Compute sample variance of spillover effects using bounds"""
        bound_on_covariance = np.zeros([self.num_treatments, 4, 4])

        for s in range(4):
            for s_ in range(4):
                bound_on_covariance[:, s, s_] = np.sqrt(
                    sample_variance_average_effects[:, s] * sample_variance_average_effects[:, s_])

        sample_var_direct = self.compute_sample_variance_spillover_direct(bound_on_covariance)
        sample_var_im = self.compute_sample_variance_spillover_im(bound_on_covariance)
        sample_var_iv = self.compute_sample_variance_spillover_iv(bound_on_covariance)
        sample_var_spillover_pairs = self.compute_sample_variance_spillover_treated_pairs(bound_on_covariance)

        sample_variance_spillovers_bnds = np.zeros([self.num_treatments, 2, 4])
        sample_variance_spillovers_bnds[:, :, 0] = np.clip(sample_var_direct, 1e-10, np.inf)
        sample_variance_spillovers_bnds[:, :, 1] = np.clip(sample_var_im, 1e-10, np.inf)
        sample_variance_spillovers_bnds[:, :, 2] = np.clip(sample_var_iv, 1e-10, np.inf)
        sample_variance_spillovers_bnds[:, :, 3] = np.clip(sample_var_spillover_pairs, 1e-10, np.inf)

        return sample_variance_spillovers_bnds

    def compute_sample_variance_spillover_direct(self, bound_covariances):
        """Compute bounds for direct spillover variance"""
        cf = np.asarray([1, 1, 1, 1]).reshape(4, 1)
        cf2 = -cf * cf.T
        np.fill_diagonal(cf2, 1)
        
        lower_bound = np.apply_over_axes(np.sum, cf2[np.newaxis] * bound_covariances, (1, 2))
        upper_bound = np.apply_over_axes(np.sum, np.abs(cf2)[np.newaxis] * bound_covariances, (1, 2))

        bnds = np.zeros([self.num_treatments, 2])
        bnds[:, 0] = lower_bound.reshape(self.num_treatments)
        bnds[:, 1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def compute_sample_variance_spillover_im(self, bound_covariances):
        """Compute bounds for IM spillover variance"""
        lower_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, 1, 1] - 
                      2 * bound_covariances[:, 0, 1])
        upper_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, 1, 1] + 
                      2 * bound_covariances[:, 0, 1])

        bnds = np.zeros([self.num_treatments, 2])
        bnds[:, 0] = lower_bound.reshape(self.num_treatments)
        bnds[:, 1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def compute_sample_variance_spillover_iv(self, bound_covariances):
        """Compute bounds for IV spillover variance"""
        lower_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, 2, 2] - 
                      2 * bound_covariances[:, 0, 2])
        upper_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, 2, 2] + 
                      2 * bound_covariances[:, 0, 2])

        bnds = np.zeros([self.num_treatments, 2])
        bnds[:, 0] = lower_bound.reshape(self.num_treatments)
        bnds[:, 1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def compute_sample_variance_spillover_treated_pairs(self, bound_covariances):
        """Compute bounds for treated pairs spillover variance"""
        lower_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, -1, -1] - 
                      2 * bound_covariances[:, 0, -1])
        upper_bound = (bound_covariances[:, 0, 0] + bound_covariances[:, -1, -1] + 
                      2 * bound_covariances[:, 0, -1])

        bnds = np.zeros([self.num_treatments, 2])
        bnds[:, 0] = lower_bound.reshape(self.num_treatments)
        bnds[:, 1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def run_experiment(self):
        """Run complete experiment and return all results"""
        start_time = time.time()
        
        results = {}
        results['potential_outcomes'] = self.potential_outcomes
        results['types'] = self.draw_types()
        results['realized_outcomes'] = self.draw_realized_outcomes(results['types'])
        results['total'] = self.total
        results['active'] = self.active

        print('Computing average effects...')
        # Population quantities
        results['population_average_effects'] = self.compute_population_average_effects()
        results['population_spillover_effects'] = self.compute_population_spillover_effects(
            results['population_average_effects'])
        
        # Sample quantities
        results['sample_average_effects'] = self.compute_sample_average_effects(
            results['realized_outcomes'], results['types'])
        results['sample_spillover_effects'] = self.compute_sample_spillover_effects(
            results['sample_average_effects'])

        print('Computing estimates of variance from the sample...')
        # Population variances
        results['population_variance_average_effects'] = np.diagonal(
            self.compute_population_covariance_average_effects())
        results['population_covariance_average_effects'] = self.compute_population_covariance_average_effects()
        results['population_variance_spillover_effects'] = self.compute_population_variance_spillovers()
        
        # Sample variances
        results['sample_variance_average_effects'] = self.compute_sample_variance_average_effects(
            results['realized_outcomes'], results['types'])
        results['sample_variance_spillover_effects'] = self.compute_sample_variance_spillovers(
            results['sample_variance_average_effects'])

        print('Experiment completed; Total computing time : ', str(time.time() - start_time)[:6], ' seconds')
        return results


class SimpleDoubleRandomizedExperimentWithLocalInterference(Simple_Double_Randomized_Experiment):
    """
    Extension of Simple_Double_Randomized_Experiment for local interference experiments.
    
    This class handles experiments where potential outcomes are pre-computed and
    interference effects are modeled through local interactions.
    """
    
    def __init__(self, potential_outcomes, active, num_treatments, seed):
        super().__init__(potential_outcomes, active, num_treatments, seed)
    
    def run_experiment(self):
        """Run complete experiment with local interference"""
        start_time = time.time()
        
        results = {}
        results['potential_outcomes'] = self.potential_outcomes
        results['types'] = self.draw_types()
        results['realized_outcomes'] = self.draw_realized_outcomes(results['types'])
        results['total'] = self.total
        results['active'] = self.active

        print('Computing average effects...')
        # Population quantities
        results['population_average_effects'] = self.compute_population_average_effects()
        results['population_spillover_effects'] = self.compute_population_spillover_effects(
            results['population_average_effects'])
        
        # Sample quantities
        results['sample_average_effects'] = self.compute_sample_average_effects(
            results['realized_outcomes'], results['types'])
        results['sample_spillover_effects'] = self.compute_sample_spillover_effects(
            results['sample_average_effects'])

        print('Computing estimates of variance from the sample...')
        # Population variances
        results['population_variance_average_effects'] = np.diagonal(
            self.compute_population_covariance_average_effects())
        results['population_variance_spillover_effects'] = self.compute_population_variance_spillovers()
        
        # Sample variances
        results['sample_variance_average_effects'] = self.compute_sample_variance_average_effects(
            results['realized_outcomes'], results['types'])
        results['sample_variance_spillover_effects'] = self.compute_sample_variance_spillovers(
            results['sample_variance_average_effects'])

        print('Experiment completed; Total computing time : ', str(time.time() - start_time)[:6], ' seconds')
        return results


# ============================================================================
# EXTENDED EXPERIMENT CLASSES WITH DATA GENERATORS
# ============================================================================

class SimpleDoubleRandomizedExperiment(object):
    """
    Extended experiment class that works with data generators for more complex experiments.
    
    This version uses data generators for generating outcomes dynamically
    rather than pre-computed potential outcomes.
    """
    
    def __init__(self, total: np.ndarray, active: np.ndarray, num_treatments: int, 
                 data_generator, seed: int):
        self.data_generator = data_generator
        self.total = total
        self.active = active
        self.num_treatments = num_treatments
        self.seed = seed

        assert all(self.total - self.active >= 0), 'Need I_1 <= I and J_1 <= J'

    def draw_treatments(self):
        """Draw random treatment assignments using numba-accelerated functions"""
        if type(self.total) == list:
            self.total = np.asarray(self.total)
        if self.total.dtype != int:
            self.total = self.total.astype(int)
        if type(self.active) == list:
            self.active = np.asarray(self.active)
        if self.active.dtype != int:
            self.active = self.active.astype(int)

        I, J = self.total
        I1, J1 = self.active

        # Set random seeds
        numba_set_seed(self.seed)
        np.random.seed(self.seed) 
        
        active_sets_I = np.zeros([self.num_treatments, I], dtype=int)
        active_sets_J = np.zeros([self.num_treatments, J], dtype=int)
        treatments = np.zeros([self.num_treatments, I, J], dtype=int)
        types = np.zeros([self.num_treatments, I, J], dtype=int)
        
        print('Drawing random assignments...')
        for n_mc in tqdm(range(self.num_treatments)):
            active_sets_I[n_mc], active_sets_J[n_mc], treatments[n_mc] = double_randomized_sample(I, I1, J, J1)
            types[n_mc] = get_types(active_sets_I[n_mc], active_sets_J[n_mc])

        return treatments, types, active_sets_I, active_sets_J

    def draw_types(self):
        """Draw treatment types for all experiments"""
        return self.draw_treatments()[1] 
    
    def draw_realized_outcomes(self, types, is_parallel=False):
        """Draw realized outcomes using the data generator"""
        I, J = self.total
        treatments = np.array(types >= 3, dtype=int)

        if is_parallel:
            pool = mp.Pool()
            realized_outcomes = pool.map(self.data_generator.simulate_results, treatments)
        else:
            realized_outcomes = np.array([
                self.data_generator.simulate_results(treatment_draw) 
                for treatment_draw in tqdm(treatments)
            ])
            
        return realized_outcomes
        
    def compute_sample_average_effects(self, realized_outcomes, types):   
        """Compute sample average effects for each experiment"""
        sample_average_effect = np.zeros([self.num_treatments, len(SDRD_TYPES)])
        for n_mc in tqdm(range(self.num_treatments)): 
            Y = realized_outcomes[n_mc]
            types_ = types[n_mc]
            for v_, v in enumerate(SDRD_TYPES):
                sample_average_effect[n_mc, v_] = Y[np.where(types_==v)].mean()

        return sample_average_effect
    
    def compute_sample_spillover_effects(self, sample_average_effect):
        """Compute sample spillover effects"""
        sample_tau_direct = (sample_average_effect[:,-1] - sample_average_effect[:,1] - 
                            sample_average_effect[:,2] + sample_average_effect[:,0])
        sample_tau_spillover_M = sample_average_effect[:,1] - sample_average_effect[:,0]
        sample_tau_spillover_V = sample_average_effect[:,2] - sample_average_effect[:,0]
        sample_tau_spillover_pairs = sample_average_effect[:,-1] - sample_average_effect[:,0]
        
        return np.asarray([sample_tau_direct, sample_tau_spillover_M, 
                          sample_tau_spillover_V, sample_tau_spillover_pairs]).T

    def compute_covariance_coefficients(self):
        """Compute covariance coefficients (same as base class)"""
        I_1, J_1 = self.active
        I, J = self.total
        I_0, J_0 = self.total - self.active

        alphas = np.zeros([4,2])
        alphas[:,0] = [I_1/(I_0*(I-1)), I_0/(I_1*(I-1)), I_1/(I_0*(I-1)), I_0/(I_1*(I-1))]
        alphas[:,1] = [J_1/(J_0*(J-1)), J_1/(J_0*(J-1)), J_0/(J_1*(J-1)), J_0/(J_1*(J-1))]
        
        sample_sizes = np.zeros([4,2], dtype=str)
        sample_sizes[:, 0] = ['0', '1', '0', '1']
        sample_sizes[:, 1] = ['0', '0', '1', '1']

        coefs = np.zeros([4,4,2])
        for v_ in range(4):
            for v__ in range(4):
                coefs[v_, v__] = (alphas[v_] * (sample_sizes[v_] == sample_sizes[v__]) - 
                                np.asarray([1/I, 1/J]) * (sample_sizes[v_] != sample_sizes[v__]))
        return coefs

    def compute_sample_variance_average_effects(self, realized_outcomes, types):
        """Compute sample variance of average effects (same implementation as base class)"""
        I, J = self.total
        I_1, J_1 = self.active
        I_0, J_0 = I - I_1, J - J_1
        I_s_ls = np.asarray([I_0, I_1, I_0, I_1])
        J_s_ls = np.asarray([J_0, J_0, J_1, J_1])
        alpha_M_ls = (I-I_s_ls)/((I-1)*I_s_ls)
        alpha_V_ls = (J-J_s_ls)/((J-1)*J_s_ls)
        sample_variance_average_effects = np.zeros([self.num_treatments, 4])
        bias_correction_M = np.zeros([self.num_treatments, 4])
        bias_correction_V = np.zeros([self.num_treatments, 4])
        
        for n_mc in tqdm(range(self.num_treatments)): 
            Y = realized_outcomes[n_mc]
            types_ = types[n_mc]

            for s in range(4):
                I_s, J_s = I_s_ls[s], J_s_ls[s]
                if I_s * J_s < 2:
                    pass
                else:
                    Y_s = Y[types_ == s].reshape([I_s, J_s])
                    alpha_M, alpha_V = alpha_M_ls[s], alpha_V_ls[s] 
                    xi = 1 - alpha_M - alpha_V + alpha_M * alpha_V

                    hat_Y_s = Y_s.mean()
                    hat_Y_s_M = Y_s.mean(axis = 1)
                    hat_Y_s_V = Y_s.mean(axis = 0)

                    tilde_sigma_M = np.sum((hat_Y_s_M - hat_Y_s)**2)
                    tilde_sigma_V = np.sum((hat_Y_s_V - hat_Y_s)**2)
                    tilde_sigma_MV = np.sum((Y_s - hat_Y_s_M[:,np.newaxis] - hat_Y_s_V[np.newaxis,:] + hat_Y_s)**2)

                    sample_variance_average_effects[n_mc,s] = (alpha_M/I_s*tilde_sigma_M) + (alpha_V/J_s*tilde_sigma_V) + (alpha_M * alpha_V / (I_s * J_s) * tilde_sigma_MV)

                    bias_correction_M[n_mc, s] = 1/I_s * (J-J_s)/(J_s * (J_s-1) * J) * np.sum((Y_s - Y_s.mean(axis = 1)[:, np.newaxis])**2)
                    bias_correction_V[n_mc, s] = 1/J_s * (I-I_s)/(I_s * (I_s-1) * I) * np.sum((Y_s - Y_s.mean(axis = 0)[np.newaxis, :])**2)

        hat_sig = sample_variance_average_effects/xi - alpha_M_ls/(1-alpha_M_ls)*bias_correction_M - alpha_V_ls/(1-alpha_V_ls)*bias_correction_V
        if np.min(hat_sig)<0:
            hat_sig[hat_sig<=0] = - np.max(hat_sig[hat_sig<=0])
        return hat_sig

    def compute_sample_variance_spillovers(self, sample_variance_average_effects):
        """Compute sample variance of spillover effects using bounds"""
        bound_on_covariance = np.zeros([self.num_treatments, 4, 4])

        for s in range(4):
            for s_ in range(4):
                bound_on_covariance[:, s, s_] = np.sqrt(sample_variance_average_effects[:,s]*sample_variance_average_effects[:,s_])

        sample_var_direct = self.compute_sample_variance_spillover_direct(bound_on_covariance)
        sample_var_im = self.compute_sample_variance_spillover_im(bound_on_covariance)
        sample_var_iv = self.compute_sample_variance_spillover_iv(bound_on_covariance)
        sample_var_spillover_pairs = self.compute_sample_variance_spillover_treated_pairs(bound_on_covariance)

        sample_variance_spillovers_bnds = np.zeros([self.num_treatments, 2, 4])
        sample_variance_spillovers_bnds[:,:,0] = np.clip(sample_var_direct, 1e-10, np.inf) 
        sample_variance_spillovers_bnds[:,:,1] = np.clip(sample_var_im, 1e-10, np.inf)  
        sample_variance_spillovers_bnds[:,:,2] = np.clip(sample_var_iv, 1e-10, np.inf)  
        sample_variance_spillovers_bnds[:,:,3] = np.clip(sample_var_spillover_pairs, 1e-10, np.inf)  

        return sample_variance_spillovers_bnds

    def compute_sample_variance_spillover_direct(self, bound_covariances):
        """Compute bounds for direct spillover variance"""
        cf = np.asarray([1,1,1,1]).reshape(4,1)
        cf2 = -cf*cf.T
        np.fill_diagonal(cf2, 1)
        lower_bound = np.apply_over_axes(np.sum, cf2[np.newaxis] * bound_covariances, (1,2)) 
        upper_bound = np.apply_over_axes(np.sum, np.abs(cf2)[np.newaxis] * bound_covariances, (1,2))

        bnds = np.zeros([self.num_treatments,2])
        bnds[:,0] = lower_bound.reshape(self.num_treatments)
        bnds[:,1] = upper_bound.reshape(self.num_treatments) 
        return bnds

    def compute_sample_variance_spillover_im(self, bound_covariances):
        """Compute bounds for IM spillover variance"""
        lower_bound = bound_covariances[:, 0, 0] + bound_covariances[:, 1, 1] - 2 * bound_covariances[:,0,1] 
        upper_bound = bound_covariances[:, 0, 0] + bound_covariances[:, 1, 1] + 2 * bound_covariances[:,0,1]

        bnds = np.zeros([self.num_treatments,2])
        bnds[:,0] = lower_bound.reshape(self.num_treatments)
        bnds[:,1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def compute_sample_variance_spillover_iv(self, bound_covariances):
        """Compute bounds for IV spillover variance"""
        lower_bound = bound_covariances[:, 0, 0] + bound_covariances[:, 2, 2] - 2 * bound_covariances[:,0,2] 
        upper_bound = bound_covariances[:, 0, 0] + bound_covariances[:, 2, 2] + 2 * bound_covariances[:,0,2]

        bnds = np.zeros([self.num_treatments,2])
        bnds[:,0] = lower_bound.reshape(self.num_treatments)
        bnds[:,1] = upper_bound.reshape(self.num_treatments)
        return bnds

    def compute_sample_variance_spillover_treated_pairs(self, bound_covariances):
        """Compute bounds for treated pairs spillover variance"""
        lower_bound = bound_covariances[:,0, 0] + bound_covariances[:,-1,-1] - 2 * bound_covariances[:, 0, -1] 
        upper_bound = bound_covariances[:,0, 0] + bound_covariances[:,-1,-1] + 2 * bound_covariances[:, 0, -1]

        bnds = np.zeros([self.num_treatments,2])
        bnds[:,0] = lower_bound.reshape(self.num_treatments)
        bnds[:,1] = upper_bound.reshape(self.num_treatments)
        return bnds

