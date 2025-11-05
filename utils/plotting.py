#!/usr/bin/env python
"""
Plotting utilities for Multiple Randomization Designs (MRDs)

This module contains all plotting functions for visualizing experimental results,
including main paper figures, appendix figures, and diagnostic plots.

Functions:
    Main Paper Plots:
    - plot_avg_and_variance_average_effects: Main paper average effects plot
    - plot_mean_var_spillovers: Main paper spillover effects plot
    
    Appendix Plots:
    - appendix_plot_avg_and_variance_average_effects: Individual type average effects
    - appendix_plot_mean_var_spillovers: Individual spillover type plots
    
    Diagnostic Plots:
    - plot_t_stat_qq_spillover: T-statistics and QQ plots
    - compare_to_buyer_randomized: Comparison with single randomization
    - plot_pvalue_distribution_under_null: P-value distribution under null
    
    Helper Functions:
    - plot_hist: Histogram plotting with quantile bars
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from scipy.stats import norm
import seaborn as sns


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plot_hist(data, ax, quantile_bars, variance=None):
    """
    Helper function for plotting histograms with quantile bars.
    
    Args:
        data: Data to plot
        ax: Matplotlib axis object
        quantile_bars: Boolean, whether to show quantile bars
        variance: Optional variance for theoretical confidence bounds
    """
    sns.histplot(ax=ax, data=data)
    if quantile_bars:
        ax.axvline(np.mean(data), color='red', linewidth=5)
        ax.axvline(np.quantile(data, 0.025), color='black', linewidth=5)
        ax.axvline(np.quantile(data, 0.975), color='black', linewidth=5)
    if variance:
        ax.axvline(np.mean(data) + 1.96*np.sqrt(variance), color='green', linewidth=2)
        ax.axvline(np.mean(data) - 1.96*np.sqrt(variance), color='green', linewidth=2)


# ============================================================================
# MAIN PAPER PLOTS
# ============================================================================

def plot_avg_and_variance_average_effects(results, save): 
    """
    Plot average effects and variances for main paper figures.
    
    Args:
        results: Dictionary containing experimental results
        save: String path to save plot (without .pdf extension), or False to not save
    """
    population_variance_average_effects = results['population_variance_average_effects']
    sample_variance_average_effects = results['sample_variance_average_effects']
    population_average_effects = results['population_average_effects']
    sample_average_effects = results['sample_average_effects']
    active = results['active']
    total = results['total']
    
    lo, hi = spst.norm.ppf(0.025), spst.norm.ppf(0.975)
    title_ls = ['cc', 'ib', 'is', 'tr']
    color_ls = ['red', 'green', 'blue', 'gray']
    type_ = 0  # Focus on CC type for main plot

    plt.figure(figsize=(18, 6))    
    I, J = total
    I_1, J_1 = active

    # Left subplot: Average effects distribution
    ax = plt.subplot(1, 2, 1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    sample_average_effect = sample_average_effects[:, type_]
    y_ls, x_ls, desc = plt.hist(sample_average_effect, alpha=.25, color=color_ls[type_], 
                               bins=50, label=r'$\widehat{\overline{\overline{Y}}}_{'+title_ls[type_]+r'}$')
    max_x, max_y = max(x_ls), max(y_ls)

    # Plot sample quantities
    plt.scatter(np.mean(sample_average_effect), -max_y/20, alpha=.95, 
               color=color_ls[type_], marker='o', s=500)
    plt.vlines(x=np.mean(sample_average_effect), ymin=0, alpha=.95, ymax=max_y, 
              color=color_ls[type_], linewidth=5)
    
    # Empirical quantiles
    empirical_lb = np.quantile(sample_average_effect, .025)
    empirical_ub = np.quantile(sample_average_effect, .975)
    
    plt.scatter(empirical_lb, -max_y/20, alpha=.95, color=color_ls[type_], marker='o', s=500)
    plt.vlines(x=empirical_lb, ymin=0, alpha=.95, ymax=max_y, color=color_ls[type_], linewidth=5)
    plt.scatter(empirical_ub, -max_y/20, alpha=.95, color=color_ls[type_], marker='o', s=500)
    plt.vlines(x=empirical_ub, ymin=0, alpha=.95, ymax=max_y, color=color_ls[type_], linewidth=5)

    # Plot population quantities
    plt.scatter(population_average_effects[type_], -max_y/20, color='k', marker='X', s=300, 
               label=r'$\bar{\bar{y}}_{cc} \pm 1.96 \sqrt{Var\left[\hat{\overline{\overline{Y}}}_{cc}\right]}$')
    plt.vlines(x=population_average_effects[type_], ymin=0, ymax=max_y, color='k', 
              linestyle='-.', linewidth=5)
    
    # Population confidence bounds
    pop_lb = population_average_effects[type_] + lo * np.sqrt(population_variance_average_effects[type_])
    pop_ub = population_average_effects[type_] + hi * np.sqrt(population_variance_average_effects[type_])
    
    plt.scatter(pop_lb, -max_y/20, color='k', marker='^', s=300)
    plt.vlines(x=pop_lb, ymin=0, ymax=max_y, color='k', linewidth=5, linestyle='-.')
    plt.scatter(pop_ub, -max_y/20, color='k', marker='^', s=300)
    plt.vlines(x=pop_ub, ymin=0, ymax=max_y, color='k', linewidth=5, linestyle='-.')

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    
    plt.ylabel('# Randomizations', fontsize=40)
    plt.legend(fontsize=28, ncol=2, bbox_to_anchor=(1.05, -0.1))
    
    # Right subplot: Variance distribution
    ax = plt.subplot(1, 2, 2)
    y, x, _ = plt.hist(sample_variance_average_effects[:, 0], color=color_ls[0], alpha=.2, 
                      bins=50, label=r'$\widehat{\Sigma}_{cc}$')
    ymax = max(y)
    
    plt.vlines(sample_variance_average_effects[:, 0].mean(), ymin=0, ymax=ymax, 
              color=color_ls[0], linewidth=5)
    plt.vlines(population_variance_average_effects[0], ymin=0, ymax=ymax, 
              color='k', linestyle='--', linewidth=5)
    
    plt.scatter(population_variance_average_effects[type_], -max_y/20, color='k', 
               label=r'$Var\left(\hat{\overline{\overline{Y}}}_{cc}\right)$', marker='*', s=300)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    plt.legend(fontsize=30, ncol=2, bbox_to_anchor=(0.9, -0.1))

    if save != False:
        print(save)
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def plot_mean_var_spillovers(results, save):
    """
    Plot mean and variance of spillover effects for main paper.
    
    Args:
        results: Dictionary containing experimental results
        save: String path to save plot (without .pdf extension), or False to not save
    """
    population_spillover_effects = results['population_spillover_effects']    
    sample_spillover_effects = results['sample_spillover_effects']
    population_variance_spillover_effects = results['population_variance_spillover_effects']
    sample_variance_spillover_effects = results['sample_variance_spillover_effects']
    
    lo, hi = spst.norm.ppf(0.025), spst.norm.ppf(0.975)
    
    plt.figure(figsize=(18, 6))
    chosen_e = 1  # Focus on buyer spillover
    
    for e_, spillover in enumerate(sample_spillover_effects.T):
        if e_ != chosen_e:
            continue
            
        # Left subplot: Spillover distribution
        ax = plt.subplot(1, 2, 1)
        y_ls, x_ls, desc = plt.hist(spillover, alpha=.25, bins=50, color='cadetblue', 
                                   label=r'$\widehat{\tau}_{\rm{spill}}^B$')
        max_x, max_y = max(x_ls), max(y_ls)

        # Sample quantiles
        plt.scatter(np.quantile(spillover, .5), -max_y/20, color='cadetblue', marker='o', s=500)
        plt.vlines(x=np.quantile(spillover, .5), ymin=0, ymax=max_y, color='cadetblue', linewidth=5)
        plt.scatter(np.quantile(spillover, .025), -max_y/20, color='cadetblue', marker='o', s=500)
        plt.vlines(x=np.quantile(spillover, .025), ymin=0, ymax=max_y, color='cadetblue', linewidth=5)
        plt.scatter(np.quantile(spillover, .975), -max_y/20, color='cadetblue', marker='o', s=500)  
        plt.vlines(x=np.quantile(spillover, .975), ymin=0, ymax=max_y, color='cadetblue', linewidth=5)

        # Population quantities
        plt.scatter(population_spillover_effects[e_], -max_y/20, color='k', marker='X', s=300)
        plt.vlines(x=population_spillover_effects[e_], ymin=0, ymax=max_y, color='k', 
                  linestyle='-.', linewidth=5, 
                  label=r'${\tau}_{\rm{spill}}^B \pm 1.96 \sqrt{{Var}(\hat{\tau}_{spill}^B)} $')
        
        # Population confidence bounds
        pop_lb = population_spillover_effects[e_] + lo * np.sqrt(population_variance_spillover_effects[e_])
        pop_ub = population_spillover_effects[e_] + hi * np.sqrt(population_variance_spillover_effects[e_])
        
        plt.scatter(pop_lb, -max_y/20, color='k', marker='^', s=300)
        plt.vlines(x=pop_lb, ymin=0, ymax=max_y, color='k', linestyle='-.', linewidth=5)
        plt.scatter(pop_ub, -max_y/30, color='k', marker='^', s=300)
        plt.vlines(x=pop_ub, ymin=0, ymax=max_y, color='k', linestyle='-.', linewidth=5)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        xticks = ticker.MaxNLocator(4)
        yticks = ticker.MaxNLocator(4)
        ax.xaxis.set_major_locator(xticks)
        ax.yaxis.set_major_locator(yticks)

    plt.ylabel('# Randomizations', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    plt.legend(fontsize=24, ncol=2, bbox_to_anchor=(1., -0.1))
    
    # Right subplot: Variance distribution
    ax = plt.subplot(1, 2, 2)
    sample_variance_spillover_im = sample_variance_spillover_effects[:, :, 1]
    y1, x, _ = plt.hist(sample_variance_spillover_im[:, 1], alpha=.2, bins=50, color='cadetblue', 
                       label=r'$\widehat{Var}^{hi}(\hat{\tau}_{spill}^B)$')
    ym = max(y1)

    plt.vlines(sample_variance_spillover_im[:, 1].mean(), ymin=0, ymax=ym, 
              color='cadetblue', linewidth=5)
    plt.vlines(population_variance_spillover_effects[1], ymin=0, ymax=ym, color='k', 
              linestyle='-.', linewidth=5, label=r'${Var}(\hat{\tau}_{spill}^B)$')
    
    plt.scatter(population_variance_spillover_effects[1], -max_y/30, color='k', marker='X', s=300)
    plt.scatter(sample_variance_spillover_im[:, 1].mean(), -max_y/30, color='cadetblue', s=300)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    plt.legend(fontsize=25, ncol=2, bbox_to_anchor=(.85, -0.1))

    if save != False:
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()


# ============================================================================
# APPENDIX PLOTS
# ============================================================================

def appendix_plot_avg_and_variance_average_effects(results, type_idx, save=False): 
    """
    Plot average effects and variances for individual treatment types (appendix).
    
    Args:
        results: Dictionary containing experimental results
        type_idx: Index of treatment type (0=cc, 1=ib, 2=is, 3=tr)
        save: String path to save plot (without .pdf extension), or False to not save
    """
    population_variance_average_effects = results['population_variance_average_effects']
    sample_variance_average_effects = results['sample_variance_average_effects']
    population_average_effects = results['population_average_effects']
    sample_average_effects = results['sample_average_effects']
    active = results['active']
    total = results['total']
    
    labels_mean_pop = [
        r'$\bar{\bar{y}}_{cc} \pm 1.96 \sqrt{Var\left[\hat{\overline{\overline{Y}}}_{cc}\right]}$', 
        r'$\bar{\bar{y}}_{ib} \pm 1.96 \sqrt{Var\left[\hat{\overline{\overline{Y}}}_{ib}\right]}$', 
        r'$\bar{\bar{y}}_{is} \pm 1.96 \sqrt{Var\left[\hat{\overline{\overline{Y}}}_{is}\right]}$', 
        r'$\bar{\bar{y}}_{tr} \pm 1.96 \sqrt{Var\left[\hat{\overline{\overline{Y}}}_{tr}\right]}$']
    
    labels_variance = [r'$\widehat{\Sigma}_{cc}$', r'$\widehat{\Sigma}_{ib}$', 
                      r'$\widehat{\Sigma}_{is}$', r'$\widehat{\Sigma}_{tr}$']
    
    population_variance_labels = [
        r'$Var\left(\hat{\overline{\overline{Y}}}_{cc}\right)$',
        r'$Var\left(\hat{\overline{\overline{Y}}}_{ib}\right)$',
        r'$Var\left(\hat{\overline{\overline{Y}}}_{is}\right)$',
        r'$Var\left(\hat{\overline{\overline{Y}}}_{tr}\right)$']
    
    lo, hi = spst.norm.ppf(0.025), spst.norm.ppf(0.975)
    title_ls = ['cc', 'ib', 'is', 'tr']
    color_ls = ['red', 'green', 'blue', 'gray']

    plt.figure(figsize=(18, 6))    
    I, J = total

    # Left subplot: Average effects distribution
    ax = plt.subplot(1, 2, 1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    sample_average_effect = sample_average_effects[:, type_idx]
    y_ls, x_ls, desc = plt.hist(sample_average_effect, alpha=.25, color=color_ls[type_idx], 
                               bins=50, label=r'$\widehat{\overline{\overline{Y}}}_{'+title_ls[type_idx]+r'}$')
    max_x, max_y = max(x_ls), max(y_ls)
    max_y = 700
    plt.ylim([-75, 750])

    # Sample quantities
    plt.scatter(np.mean(sample_average_effect), -max_y/20, alpha=.95, 
               color=color_ls[type_idx], marker='o', s=500)
    plt.vlines(x=np.mean(sample_average_effect), ymin=0, alpha=.95, ymax=max_y, 
              color=color_ls[type_idx], linewidth=5)
    
    # Sample quantiles
    plt.scatter(np.quantile(sample_average_effect, .025), -max_y/20, alpha=.95, 
               color=color_ls[type_idx], marker='o', s=500)
    plt.vlines(x=np.quantile(sample_average_effect, .025), ymin=0, alpha=.95, ymax=max_y, 
              color=color_ls[type_idx], linewidth=5)
    plt.scatter(np.quantile(sample_average_effect, .975), -max_y/20, alpha=.95, 
               color=color_ls[type_idx], marker='o', s=500)
    plt.vlines(x=np.quantile(sample_average_effect, .975), ymin=0, alpha=.95, ymax=max_y, 
              color=color_ls[type_idx], linewidth=5)

    # Population quantities
    plt.scatter(population_average_effects[type_idx], -max_y/20, color='k', marker='X', s=300, 
               label=labels_mean_pop[type_idx])
    plt.vlines(x=population_average_effects[type_idx], ymin=0, ymax=max_y, color='k', 
              linestyle='-.', linewidth=5)
    
    # Population confidence bounds
    pop_lb = population_average_effects[type_idx] + lo * np.sqrt(population_variance_average_effects[type_idx])
    pop_ub = population_average_effects[type_idx] + hi * np.sqrt(population_variance_average_effects[type_idx])
    
    plt.scatter(pop_lb, -max_y/20, color='k', marker='^', s=300)
    plt.vlines(x=pop_lb, ymin=0, ymax=max_y, color='k', linewidth=5, linestyle='-.')
    plt.scatter(pop_ub, -max_y/20, color='k', marker='^', s=300)
    plt.vlines(x=pop_ub, ymin=0, ymax=max_y, color='k', linewidth=5, linestyle='-.')

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)

    plt.ylabel('# Randomizations', fontsize=40)
    plt.legend(fontsize=28, ncol=2, bbox_to_anchor=(1.05, -0.1))
    
    # Right subplot: Variance distribution
    ax = plt.subplot(1, 2, 2)
    y, x, _ = plt.hist(sample_variance_average_effects[:, type_idx], color=color_ls[type_idx], 
                      alpha=.2, bins=50, label=labels_variance[type_idx])
    ymax = 700
    
    plt.vlines(sample_variance_average_effects[:, type_idx].mean(), ymin=0, ymax=ymax, 
              color=color_ls[type_idx], linewidth=5)
    plt.vlines(population_variance_average_effects[type_idx], ymin=0, ymax=ymax, 
              color='k', linestyle='--', linewidth=5)
    
    plt.scatter(population_variance_average_effects[type_idx], -max_y/20, color='k', 
               label=population_variance_labels[type_idx], marker='*', s=300)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    plt.ylim([-75, 750])
    
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    plt.legend(fontsize=30, ncol=2, bbox_to_anchor=(0.9, -0.1))

    if save != False:
        print(save)
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def appendix_plot_mean_var_spillovers(results, type_idx, save):
    """
    Plot mean and variance of individual spillover types (appendix).
    
    Args:
        results: Dictionary containing experimental results
        type_idx: Index of spillover type (0=direct, 1=buyer, 2=seller, 3=ATE)
        save: String path to save plot (without .pdf extension), or False to not save
    """
    population_spillover_effects = results['population_spillover_effects']    
    sample_spillover_effects = results['sample_spillover_effects']
    population_variance_spillover_effects = results['population_variance_spillover_effects']
    sample_variance_spillover_effects = results['sample_variance_spillover_effects']

    lo, hi = spst.norm.ppf(0.025), spst.norm.ppf(0.975)
    title_ls = [r'$\hat{\tau}_{\rm{direct}}$', r'$\hat{\tau}_{\rm{spill}}^{\rm{B}}$',  
                r'$\hat{\tau}_{\rm{spill}}^{\rm{S}}$', r'$\hat{\tau}_{\rm{ATE}}$']
    label_var = [r'$\widehat{Var}^{hi}(\hat{\tau}_{\rm{direct}})$',
                 r'$\widehat{Var}^{hi}(\hat{\tau}_{spill}^B)$',
                 r'$\widehat{Var}^{hi}(\hat{\tau}_{spill}^S)$',
                 r'$\widehat{Var}^{hi}(\hat{\tau}_{ATE})$']
    color_ls = ['orange', 'cadetblue', 'magenta', 'peru']    
    
    plt.figure(figsize=(18, 6))
    
    # Get the specific spillover type we want to plot
    spillover = sample_spillover_effects[:, type_idx]
    
    # LEFT SUBPLOT: Spillover Effect Distribution
    ax = plt.subplot(1, 2, 1)
    y_ls, x_ls, desc = plt.hist(spillover, alpha=.25, bins=50, 
                               color=color_ls[type_idx], label=title_ls[type_idx])
    max_x, max_y = max(x_ls), max(y_ls)
    max_y = 700
    
    # Sample quantiles for spillover effect
    plt.scatter(np.quantile(spillover, .5), -max_y/20, color=color_ls[type_idx], marker='o', s=500)
    plt.vlines(x=np.quantile(spillover, .5), ymin=0, ymax=max_y, color=color_ls[type_idx], linewidth=5)
    plt.scatter(np.quantile(spillover, .025), -max_y/20, color=color_ls[type_idx], marker='o', s=500)
    plt.vlines(x=np.quantile(spillover, .025), ymin=0, ymax=max_y, color=color_ls[type_idx], linewidth=5)
    plt.scatter(np.quantile(spillover, .975), -max_y/20, color=color_ls[type_idx], marker='o', s=500)  
    plt.vlines(x=np.quantile(spillover, .975), ymin=0, ymax=max_y, color=color_ls[type_idx], linewidth=5)

    # Population spillover quantities
    plt.scatter(population_spillover_effects[type_idx], -max_y/20, color='k', marker='X', s=300)
    plt.vlines(x=population_spillover_effects[type_idx], ymin=0, ymax=max_y, color='k', 
              linestyle='-.', linewidth=5, label=title_ls[type_idx])
    
    # Population confidence bounds for spillover
    pop_lb = population_spillover_effects[type_idx] + lo * np.sqrt(population_variance_spillover_effects[type_idx])
    pop_ub = population_spillover_effects[type_idx] + hi * np.sqrt(population_variance_spillover_effects[type_idx])
    
    plt.scatter(pop_lb, -max_y/20, color='k', marker='^', s=300)
    plt.vlines(x=pop_lb, ymin=0, ymax=max_y, color='k', linestyle='-.', linewidth=5)
    plt.scatter(pop_ub, -max_y/30, color='k', marker='^', s=300)
    plt.vlines(x=pop_ub, ymin=0, ymax=max_y, color='k', linestyle='-.', linewidth=5)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    plt.ylim([-75, 750])
    plt.ylabel('# Randomizations', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    plt.legend(fontsize=24, ncol=2, bbox_to_anchor=(.7, -0.1))
    
    # RIGHT SUBPLOT: Variance Distribution
    ax = plt.subplot(1, 2, 2)
    y1, x, _ = plt.hist(sample_variance_spillover_effects[:, 1, type_idx], alpha=.2, bins=50, 
                       color=color_ls[type_idx], label=label_var[type_idx])
    ym = max(y1)

    plt.vlines(sample_variance_spillover_effects[:, 1, type_idx].mean(), ymin=0, ymax=ym, 
              color=color_ls[type_idx], linewidth=5)
    plt.vlines(population_variance_spillover_effects[type_idx], ymin=0, ymax=ym, color='k', 
              linestyle='-.', linewidth=5, label=label_var[type_idx])
    plt.scatter(population_variance_spillover_effects[type_idx], -max_y/30, color='k', marker='X', s=300)
    plt.scatter(sample_variance_spillover_effects[:, 1, type_idx].mean(), -max_y/30, 
               color=color_ls[type_idx], s=300)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    xticks = ticker.MaxNLocator(4)
    yticks = ticker.MaxNLocator(4)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(24)
    ax.yaxis.offsetText.set_fontsize(24)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)
    plt.legend(fontsize=25, ncol=2, bbox_to_anchor=(.85, -0.1))

    if save != False:
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()

# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================

def plot_t_stat_qq_spillover(results, save, _spill=1):
    """
    Plot t-statistics and QQ plots for spillover effects.
    
    Args:
        results: Dictionary containing experimental results
        save: String path to save plot (without .pdf extension), or False to not save
        _spill: Which spillover to test (0=direct, 1=buyer, 2=seller, 3=ATE)
    """
    from scipy.stats import probplot
    
    svars = results['sample_variance_spillover_effects'][:, 1, _spill]
    ests = results['sample_spillover_effects'][:, _spill]
    zs = ests / np.sqrt(svars)

    proper_zs = (zs - np.mean(zs)) / np.std(zs)

    lo, hi = spst.norm.ppf(0.025), spst.norm.ppf(0.975)
    est_ls = [r'\hat{\tau}_{\rm{direct}}', r'\hat{\tau}_{\rm{spill}}^{\rm{B}}',  
              r'\hat{\tau}_{\rm{spill}}^{\rm{S}}', r'\hat{\tau}_{\rm{ATE}}']
    color_ls = ['orange', 'green', 'red', 'blue']  

    plt.figure(figsize=(18, 6)) 
    
    # Left subplot: T-statistics distribution
    ax = plt.subplot(1, 2, 1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    y_ls, x_ls, desc = plt.hist(zs, alpha=.25, color=color_ls[_spill], bins=50, 
                               label=r'$'+est_ls[_spill]+r'/\sqrt{\widehat{Var}^{hi}('+est_ls[_spill]+r')}$')
    max_x, max_y = max(x_ls), max(y_ls)

    plt.scatter(hi, -max_y/20, alpha=.95, color='k', marker='>', s=500)
    plt.vlines(x=hi, ymin=0, alpha=.95, ymax=max_y, color='k', linestyle='-.', 
              linewidth=5, label='Rejection threshold')

    plt.ylabel('# Randomizations', fontsize=30)
    plt.legend(fontsize=20, ncol=2, bbox_to_anchor=(1., -0.1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    print('Population value:', *results['population_spillover_effects'][_spill], ', Power:', np.mean(zs > hi))

    # Right subplot: QQ plot
    ax = plt.subplot(1, 2, 2)
    _ = ax.plot(*probplot(proper_zs, dist="norm")[0], color=color_ls[_spill], alpha=.25, linewidth=10)
    x = np.linspace(*ax.get_xlim())
    _ = ax.plot(x, x, color='k', alpha=.95, linestyle='-.', linewidth=3)
    plt.ylabel('Sample Quantiles', fontsize=30)
    plt.xlabel('Normal Quantiles', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if save != False:
        print(save)
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def compare_to_buyer_randomized(results, ce_effects, save):
    """
    Compare double randomization to buyer randomized design.
    
    Args:
        results: Dictionary containing experimental results
        ce_effects: Customer randomized effects array
        save: String path to save plot (without .pdf extension), or False to not save
    """
    fig = plt.figure(figsize=(18, 6)) 
    ates = results['sample_spillover_effects'][:, 3]
    
    y_ls, x_ls, desc = plt.hist(ce_effects, alpha=.25, color='orange', bins=50, label=r'$\hat{\tau}_{\rm{CR}}$')
    y_ls, x_ls, desc = plt.hist(ates, alpha=.25, color='red', bins=50, label=r'$\hat{\tau}_{\rm{ATE}}$')
    max_x, max_y = max(x_ls), max(y_ls)
    
    plt.legend(fontsize=30, ncol=2, bbox_to_anchor=(0.9, -0.1))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel('# Randomizations', fontsize=30)
    fig.axes[0].xaxis.offsetText.set_fontsize(24)

    if save != False:
        print(save)
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def plot_pvalue_distribution_under_null(results, save):
    """
    Plot p-value distributions under the null hypothesis.
    
    Args:
        results: Dictionary containing experimental results (used for structure only)
        save: String path to save plot (without .pdf extension), or False to not save
    """
    # Under true null hypothesis, all potential outcomes should be identical
    total = results['total']
    active = results['active'] 
    num_treatments = len(results['sample_average_effects'])
    
    # Get original potential outcomes
    original_potential_outcomes = results['potential_outcomes']
    
    # Create null potential outcomes - all treatments identical to control
    null_potential_outcomes = np.copy(original_potential_outcomes)
    
    # Make all treatment types identical to the control type (index 0)
    for s in range(1, 4):  # Skip s=0 (control), copy to s=1,2,3
        null_potential_outcomes[s] = null_potential_outcomes[0]
    
    # Import here to avoid circular imports
    from utils.inference import Simple_Double_Randomized_Experiment
    
    # Run experiment with null potential outcomes
    null_experiment = Simple_Double_Randomized_Experiment(
        null_potential_outcomes, active, num_treatments, 42)
    null_results = null_experiment.run_experiment()
    
    # Extract sample means and variances from null experiment
    sample_average_effects = null_results['sample_average_effects']
    sample_variance_average_effects = null_results['sample_variance_average_effects']
    
    # Calculate means for different treatment types 
    mean_c = sample_average_effects[:, 0]   # Control (cc)
    mean_ib = sample_average_effects[:, 1]  # ib type
    
    # Calculate standard errors
    se = np.sqrt(sample_variance_average_effects[:, 0] + 
                 sample_variance_average_effects[:, 1])
    
    # Avoid division by zero
    se = np.where(se == 0, 1e-10, se)
    
    # Calculate t-statistics and p-values
    tstat = (mean_ib - mean_c) / se
    pvals = 2 * norm.sf(np.abs(tstat))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(pvals, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlim([0, 1])
    plt.xlabel('P-Values', fontsize=14)
    plt.ylabel('# Randomizations', fontsize=14) 
    plt.title('P-Value Distribution Under Null Hypothesis', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    if save != False:
        print(f"Saving p-value distribution plot to {save}.pdf")
        plt.savefig(save + '.pdf', dpi=100, bbox_inches='tight')
    
    plt.show()
