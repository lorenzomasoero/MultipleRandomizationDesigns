# Multiple Randomization Designs: Paper Plot Reproduction

This repository contains code to reproduce all figures from the research paper:
**"Multiple Randomization Designs: Estimation and Inference with Interference"**

## Overview

Multiple Randomization Designs (MRDs) provide a novel framework for causal inference in the presence of interference effects. This implementation reproduces the main empirical results and extends the analysis with corrected statistical inference methods.

## Key Features

✅ **Complete Figure Reproduction**: All main paper and appendix figures  
✅ **Clean Modular Architecture**: Organized utils modules for maintainability  
✅ **Corrected Statistical Inference**: Fixed p-value calculation using proper covariance matrix  
✅ **Additional Experiments**: Extended marketplace models demonstrating methodology  
✅ **Null Hypothesis Testing**: Comprehensive validation of statistical properties  

## Installation

1. **Clone or download** this repository
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook:**
   ```bash
   jupyter notebook reproduce_paper_plots.ipynb
   ```

## File Structure

```
MRD_FINAL_SUBMISSION/
├── reproduce_paper_plots.ipynb    # Main notebook - run this!
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── utils/                        # Clean modular code
│   ├── data_generation.py        # Data generation classes
│   ├── inference.py              # Experiment and inference methods
│   └── plotting.py               # All plotting functions
└── plots/                        # Generated figures (created when run)
```

## Generated Figures

### Main Paper Figures
- **Figure 1**: `plots/main_cc.pdf` - Average effects and variance for control-control type
- **Figure 2**: `plots/main_spill.pdf` - Mean and variance of buyer spillover effects

### Appendix Figures  
- **Figures 1-4**: `plots/cc.pdf`, `plots/ib.pdf`, `plots/is.pdf`, `plots/tr.pdf` - Individual treatment type effects
- **Figures 5-8**: `plots/spill_*.pdf` - Individual spillover type effects (direct, buyer, seller, pairs)

### Additional Experiments
- **Creator-Advertiser Market**: `plots/creator_advertiser_*.pdf` - Two-sided market with complementarity
- **CES Marketplace**: `plots/ces_*.pdf` - Constant Elasticity of Substitution market

### Null Hypothesis Testing (New Contribution)
- **Causal Effects**: `plots/null_hypothesis_causal_effects.pdf` - Distribution of pairwise causal effects under null
- **P-Value Distributions**: `plots/null_hypothesis_pvalue_distributions.pdf` - Corrected p-value distributions

## Methodology

### Core MRD Framework
The Multiple Randomization Design simultaneously randomizes two populations (e.g., movies and viewers) creating four treatment types:
- **cc**: Control-control (neither population treated)
- **ib**: Inactive buyers (movies treated, viewers control)  
- **is**: Inactive sellers (viewers treated, movies control)
- **tr**: Treated (both populations treated)

### Key Innovation: Corrected P-Value Calculation
Previous implementations suffered from p-value inflation due to incorrect variance calculations. This implementation:

1. **Uses Population Covariance Matrix**: For pairwise comparisons under null hypothesis
2. **Proper Standard Errors**: Accounts for covariance between treatment types
3. **Validated Statistical Properties**: Mean p-value ≈ 0.5, Type I error ≈ 0.05

### Statistical Formula
For pairwise comparison of treatment types i and j:
```
Var(Ŷⱼ - Ŷᵢ) = Var(Ŷⱼ) + Var(Ŷᵢ) - 2×Cov(Ŷⱼ, Ŷᵢ)
```

## Experimental Parameters

### Main Paper Experiment
- **Sample Size**: 200 movies × 150 viewers
- **Active Units**: 90 movies × 85 viewers
- **Monte Carlo**: 10,000 experiments
- **Distribution**: Normal with heterogeneous parameters

### Null Hypothesis Testing
- **Parameters**: All treatment means = 0, equal small variances (0.05)
- **Sample Size**: Same as main experiment
- **Validation**: 6 pairwise comparisons, 60,000 total p-values

## Usage

### Quick Start
```python
# Run the complete analysis
jupyter notebook reproduce_paper_plots.ipynb
```

### Key Results Validation
The notebook includes comprehensive diagnostics:
- **Population vs Sample Quantities**: Verify estimator accuracy
- **Confidence Interval Coverage**: Check theoretical vs empirical coverage  
- **P-Value Calibration**: Validate uniform distribution under null
- **Type I Error Rates**: Confirm 5% rejection rate at α=0.05

## Technical Notes

### Dependencies
- **NumPy/SciPy**: Core numerical computing
- **Matplotlib/Seaborn**: Publication-quality plots
- **Numba**: Performance optimization for Monte Carlo simulations
- **Statsmodels**: QQ plots and additional statistical functions

### Performance
- **Main Experiment**: ~30-60 seconds on modern hardware
- **Additional Experiments**: ~10-20 seconds each  
- **Null Hypothesis Testing**: ~60-90 seconds
- **Total Runtime**: ~2-3 minutes for complete reproduction

### Output Format
- All plots saved as **PDF files** for publication quality
- Comprehensive **console output** with progress tracking
- **Statistical validation** printed for verification

## Citation

If you use this code, please cite the original paper:
```
[Paper citation to be added]
```

## Contact

For questions about this implementation or the methodology, please refer to the original paper or contact the authors.

---

**Last Updated**: November 2024  
**Status**: Ready for publication/sharing
