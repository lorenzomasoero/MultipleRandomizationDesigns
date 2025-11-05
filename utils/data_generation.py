#!/usr/bin/env python
"""
Data generation utilities for Multiple Randomization Designs (MRDs)

This module contains all classes and functions for generating synthetic data
and potential outcomes for double randomized experiments.

Classes:
    - Synthethic_Data_ALI: Core synthetic data generation with Additive Local Interference
    - InteractionDGP: Base class for data generating processes
    - InteractionDGPWithLocalInterference: Base class for local interference DGPs  
    - CESMarketplaceFlatRateDiscount: CES marketplace with flat rate discount
    - CreatorAdvertiserMarketplaceWithCompilmentarity: Creator-advertiser marketplace

Functions:
    - get_shares: Compute market shares for CES utility
"""

import numpy as np
import scipy.stats as spst
import time

# Default parameters
N_PRODUCTS = 200
N_CUSTOMERS = 500
TIME_PERIODS = 10
CES_PARAMETER = 2
DISCOUNT = 3
INCOME_SCALE = 100
INCOME_TAIL = 5
DEFAULT_MU = 1
DEFAULT_BMAX = 0.5

TOTAL = np.array([N_CUSTOMERS, N_PRODUCTS])
ACTIVE = np.array([N_CUSTOMERS//2, N_PRODUCTS//2])
MC_REPETITIONS = 1000


# ============================================================================
# CORE SYNTHETIC DATA GENERATION
# ============================================================================

class Synthethic_Data_ALI(object):
    """
    Synthetic Data generation under Additive Local Interference (ALI) assumption.
    
    Generates potential outcomes for 4 treatment types: c, im, iv, t
    under the ALI model where outcomes are additive across interference sources.
    
    Args:
        rv_type: Distribution type ('normal', 'laplace', 'cauchy')
        params: Array of 8 parameters [mu_c, sigma_c, mu_im, sigma_im, mu_iv, sigma_iv, mu_t, sigma_t]
        total: Array [I, J] specifying total number of movies and viewers
        seed: Random seed for reproducibility
    """

    def __init__(self, rv_type, params, total, seed):
        self.rv_type = rv_type
        self.params = params
        self.total = total
        self.seed = seed
        
    def rv_dict(self):
        """Return dictionary with parametric specification for ALI model"""
        if self.rv_type in ['normal', 'laplace', 'cauchy']:
            assert len(self.params) == 8, 'Need 8 parameters'
            assert min(self.params[1::2]) >= 0, 'Standard deviation needs to be non-negative'

            mu_C, sigma_C, mu_IM, sigma_IM, mu_IV, sigma_IV, mu_T, sigma_T = self.params 
            parameter_dict = {
                'rv_type': self.rv_type,
                'mu_C': mu_C, 'sigma_C': sigma_C,
                'mu_IM': mu_IM, 'sigma_IM': sigma_IM,
                'mu_IV': mu_IV, 'sigma_IV': sigma_IV,
                'mu_T': mu_T, 'sigma_T': sigma_T
            }
        else:
            print('This type of random variable has not been implemented yet.')
            return False
        return parameter_dict

    def draw_potential_outcomes(self): 
        """
        Draw potential outcomes under ALI assumption.
        
        Returns:
            H_all: Array of shape [I, J, 4] containing potential outcomes
                   where the last dimension indexes treatment types (c, im, iv, t)
        """
        start_time = time.time()
        rvs_dict = self.rv_dict()
        
        # Ensure total is numpy array of ints
        if type(self.total) == list:
            self.total = np.asarray(self.total)
        if self.total.dtype != int:
            self.total = self.total.astype(int)

        np.random.seed(self.seed)
        I, J = self.total

        assert rvs_dict['rv_type'] in ['normal', 'laplace', 'cauchy'], \
            'This type of random variable has not been implemented yet.'

        # Extract parameters
        mu_C, sigma_C = rvs_dict['mu_C'], rvs_dict['sigma_C']
        mu_T, sigma_T = rvs_dict['mu_T'], rvs_dict['sigma_T']
        mu_IM, sigma_IM = rvs_dict['mu_IM'], rvs_dict['sigma_IM']
        mu_IV, sigma_IV = rvs_dict['mu_IV'], rvs_dict['sigma_IV']
        
        # Generate base components based on distribution type
        if rvs_dict['rv_type'] == 'normal':
            H_C = np.random.normal(mu_C, sigma_C, size=I*J).reshape([I, J])
            H_T = np.random.normal(mu_T, sigma_T, size=I*J).reshape([I, J])
            H_IM = np.random.normal(mu_IM, sigma_IM, size=I*J).reshape([I, J])
            H_IV = np.random.normal(mu_IV, sigma_IV, size=I*J).reshape([I, J])

        elif rvs_dict['rv_type'] == 'laplace':
            H_C = np.random.laplace(mu_C, sigma_C, size=I*J).reshape([I, J])
            H_T = np.random.laplace(mu_T, sigma_T, size=I*J).reshape([I, J])
            H_IM = np.random.laplace(mu_IM, sigma_IM, size=I*J).reshape([I, J])
            H_IV = np.random.laplace(mu_IV, sigma_IV, size=I*J).reshape([I, J])

        elif rvs_dict['rv_type'] == 'cauchy':
            # Note: these are loc, scale parameters, not mean, variance
            H_C = spst.cauchy.rvs(mu_C, sigma_C, size=I*J).reshape([I, J])
            H_T = spst.cauchy.rvs(mu_T, sigma_T, size=I*J).reshape([I, J])
            H_IM = spst.cauchy.rvs(mu_IM, sigma_IM, size=I*J).reshape([I, J])
            H_IV = spst.cauchy.rvs(mu_IV, sigma_IV, size=I*J).reshape([I, J])
            
        # Construct potential outcomes under ALI assumption
        H_all = np.zeros([I, J, 4])
        H_all[:, :, 0] = H_C                    # Control
        H_all[:, :, 1] = H_C + H_IM             # Inactive movie
        H_all[:, :, 2] = H_C + H_IV             # Inactive viewer
        H_all[:, :, 3] = H_T + H_IV + H_IM      # Treated
        
        print('Data drawn; elapsed time:', str(time.time() - start_time)[:6], 'seconds')
        return H_all


# ============================================================================
# BASE DATA GENERATING PROCESS CLASSES
# ============================================================================

class InteractionDGP(object):
    """
    Base class for data generating processes with customer-product interactions.
    
    Provides the interface that all DGPs should implement for use with
    the experiment classes.
    """

    def __init__(self, n_customers, n_products, seed):
        self.n_customers = n_customers
        self.n_products = n_products
        self.rng = np.random.default_rng(seed)   

    def get_total(self):
        """Return total number of customers and products"""
        return np.array([self.n_customers, self.n_products])

    def simulate_results(self, treatment):
        """
        Simulate outcomes given treatment assignment.
        
        Args:
            treatment: Binary treatment matrix of shape [n_customers, n_products]
            
        Returns:
            outcomes: Outcome matrix of same shape as treatment
        """
        raise NotImplementedError("Subclasses must implement simulate_results")


class InteractionDGPWithLocalInterference(InteractionDGP):
    """
    Base class for DGPs with local interference effects.
    
    Extends InteractionDGP to handle cases where treatment effects
    depend on local neighborhood treatment status.
    """

    def __init__(self, n_customers, n_products, seed):
        super().__init__(n_customers, n_products, seed)

    def get_potential_outcomes(self, active):
        """
        Get potential outcomes for all treatment combinations.
        
        Args:
            active: Array [n_active_customers, n_active_products]
            
        Returns:
            potential_outcomes: Array of shape [4, n_customers, n_products]
        """
        raise NotImplementedError("Subclasses must implement get_potential_outcomes")


# ============================================================================
# MARKETPLACE DATA GENERATING PROCESSES
# ============================================================================

def get_shares(prices: np.ndarray, weights: np.ndarray, discount: float, 
               ces_parameter: float, treatment: np.ndarray) -> np.ndarray:
    """
    Compute market shares under CES utility with discounts.
    
    Args:
        prices: Product prices
        weights: Customer preference weights
        discount: Discount rate applied to treated products  
        ces_parameter: CES parameter (elasticity)
        treatment: Binary treatment matrix
        
    Returns:
        shares: Market share matrix
    """
    effective_prices = prices - prices * discount * treatment
    shares = (
        np.power(effective_prices, -ces_parameter) * weights
        / np.sum(np.multiply(weights, np.power(prices, 1-ces_parameter)), axis=1).reshape(-1, 1))
    return shares


class CESMarketplaceFlatRateDiscount(InteractionDGP):
    """
    CES marketplace with flat rate discount data generating process.
    
    Simulates a marketplace where customers have CES utility functions
    and products can receive flat rate discounts as treatment.
    """

    def __init__(self, 
                 n_products=N_PRODUCTS,
                 n_customers=N_CUSTOMERS,
                 time_periods=TIME_PERIODS,
                 ces_parameter=CES_PARAMETER,
                 discount_rate=DISCOUNT,
                 income_scale=INCOME_SCALE,
                 income_tail=INCOME_TAIL,
                 seller_elasticity=1,
                 seed=12):
        
        super().__init__(n_customers, n_products, seed)

        self.time_periods = time_periods
        self.ces_parameter = ces_parameter
        self.discount_rate = discount_rate
        self.seller_elasticity = seller_elasticity

        # Preference parameters for each customer's utility function
        self.preference_weights = self.rng.dirichlet(np.ones(self.n_products), size=self.n_customers)

        # Initial product prices
        self.initial_prices = np.clip(
            5 + self.rng.exponential(10, size=self.n_products).astype(np.float32), 1, 100)

        # Customers' incomes
        self.incomes = income_scale + income_scale * self.rng.pareto(income_tail, size=self.n_customers) // 1

        # Initial purchases and demand
        no_treatment = np.zeros((self.n_customers, self.n_products))
        self.initial_quantities = (self.incomes.reshape((-1, 1)) * 
                                  get_shares(self.initial_prices, self.preference_weights, 
                                           self.discount_rate, self.ces_parameter, no_treatment))
        self.initial_demand = self.initial_quantities.sum(axis=0)

    def simulate_results(self, treatment: np.ndarray, metric='revenue') -> np.ndarray:
        """
        Simulate marketplace outcomes given treatment.
        
        Args:
            treatment: Binary treatment matrix
            metric: Outcome metric ('revenue', 'sales', 'all')
            
        Returns:
            outcomes: Simulated outcomes based on specified metric
        """
        # Arrays to store results for each time period
        spend = np.zeros((self.n_customers, self.n_products, self.time_periods))
        effective_spend = np.zeros((self.n_customers, self.n_products, self.time_periods))
        quantities = np.zeros((self.n_customers, self.n_products, self.time_periods))
        demand = np.zeros((self.n_products, self.time_periods))
        prices = np.zeros((self.n_products, self.time_periods))

        # Record initial prices and quantities
        prices[:, 0] = self.initial_prices
        quantities[:, :, 0] = self.initial_quantities
        demand[:, 0] = self.initial_demand

        # Simulate the market forward
        for T in range(1, self.time_periods):
            # Update prices based on treatment
            prices[:, T] = self.initial_prices * (1 + self.seller_elasticity * np.max(treatment, axis=0))

            shares = get_shares(prices[:, T], self.preference_weights, 
                              self.discount_rate, self.ces_parameter, treatment)
                        
            # Compute new quantities demanded
            quantities[:, :, T] = self.incomes.reshape((-1, 1)) * shares

            # Compute aggregate demand, spending and effective spending
            demand[:, T] = quantities[:, :, T].sum(axis=0)
            spend[:, :, T] = quantities[:, :, T] * prices[:, T]
            effective_spend[:, :, T] = quantities[:, :, T] * (
                prices[:, T] - self.discount_rate * treatment)

        if metric == 'all':
            return {
                'spend': spend,
                'effective_spend': effective_spend,
                'quantities': quantities, 
                'demand': demand, 
                'prices': prices
            }
        elif metric == 'revenue':
            return np.sum(spend, axis=2)
        elif metric == 'sales':
            return np.sum(quantities, axis=2)
        else:
            raise ValueError('Unsupported metric.')


class CreatorAdvertiserMarketplaceWithCompilmentarity(InteractionDGPWithLocalInterference):
    """
    Creator-advertiser marketplace with complementarity between actions.
    
    Models a two-sided market where outcomes exhibit complementarity
    between customer and product treatments.
    """

    def __init__(self, n_customers, n_products, seed,
                 avg_mu=DEFAULT_MU,
                 bmax_customer=DEFAULT_BMAX,
                 bmax_product=DEFAULT_BMAX,
                 direct_lift=1,
                 outcome='total'):
        
        super().__init__(n_customers, n_products, seed)

        self.mu = self.rng.exponential(avg_mu, size=(n_customers, n_products))
        self.beta_customer = bmax_customer * self.rng.random(size=n_customers)
        self.beta_product = bmax_product * self.rng.random(size=n_products)
        self.direct_lift = direct_lift
        self.outcome = outcome

    def simulate_results(self, treatment: np.ndarray) -> np.ndarray:
        """
        Simulate outcomes given treatment assignment.
        
        Args:
            treatment: Binary treatment matrix
            
        Returns:
            outcomes: Outcome matrix with local interference effects
        """
        types = (np.zeros_like(treatment) 
                + np.max(treatment, axis=1).reshape(-1, 1) 
                + 2 * np.max(treatment, axis=0))
        
        type_mask = np.array([types == _ for _ in range(4)])

        active_products = np.max(np.sum(treatment, axis=1))
        active_customers = np.max(np.sum(treatment, axis=0))

        potential_outcomes = self.get_potential_outcomes(active_products, active_customers)

        return np.where(potential_outcomes, type_mask, 0).sum(axis=0)

    def get_potential_outcomes(self, active_products: int, active_customers: int) -> np.ndarray:
        """
        Generate potential outcomes with complementarity effects.
        
        Args:
            active_products: Number of active products
            active_customers: Number of active customers
            
        Returns:
            potential_outcomes: Array [4, n_customers, n_products] of potential outcomes
        """
        # Control outcomes
        c = self.mu * (
            np.zeros_like(self.mu) 
            + (np.sum(self.mu, axis=1) * self.beta_customer).reshape(-1, 1)  
            + (np.sum(self.mu, axis=0) * self.beta_product))

        # Inactive customer treatment
        iC = self.mu * (
            np.zeros_like(self.mu) 
            + ((np.sum(self.mu, axis=1) + active_products) * self.beta_customer).reshape(-1, 1)  
            + (np.sum(self.mu, axis=0) * self.beta_product))

        # Inactive product treatment
        iP = self.mu * (
            np.zeros_like(self.mu) 
            + (np.sum(self.mu, axis=1) * self.beta_customer).reshape(-1, 1)  
            + ((np.sum(self.mu, axis=0) + active_customers) * self.beta_product))

        # Full treatment
        t = (self.mu + self.direct_lift) * (
            np.zeros_like(self.mu) 
            + ((np.sum(self.mu, axis=1) + active_products) * self.beta_customer).reshape(-1, 1)  
            + ((np.sum(self.mu, axis=0) + active_customers) * self.beta_product))
        
        if self.outcome == 'total':
            return np.array([c, iC, iP, t])

        elif self.outcome == 'profit':
            margins = np.ones_like(t) - self.beta_customer.reshape(-1, 1) - self.beta_product
            t = (margins - self.direct_lift / (self.mu + self.direct_lift)) * t
            return np.array([margins * c, margins * iC, margins * iP, t])
        
        else:
            raise ValueError('Unsupported outcome type.')
