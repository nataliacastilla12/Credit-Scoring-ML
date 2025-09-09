# Credit Risk Modeling and Interpretability Framework

## Abstract
This study designs and validates an end-to-end framework for developing credit scoring mod-
els, overcoming the traditional dichotomy between machine learning performance and the
need for regulatory interpretability. The methodology is distinguished by its innovations,
including an exhaustive feature engineering and selection framework that employs multiple
methods. This framework is complemented by a risk-based economic validation to quan-
tify financial impact and a hybrid interpretability system (SHAP, LIME, WoE) to explain
complex model predictions. This process was applied to develop and compare four models:
a baseline Logistic Regression (LR), two LR variants to mitigate class imbalance, and an
optimized XGBoost model. Results revealed that the XGBoost model achieved superior
performance, with an AUC of 0.7012 and a default recall of 70.5%. The economic analysis
quantified the value of this accuracy at $4.2 million USD in potential savings. This work
not only presents a superior predictive model but offers a replicable paradigm for financial
institutions to responsibly adopt machine learning solutions, ensuring they are robust, eco-
nomically viable, and transparent.
## Thesis Project Documentation

This repository contains the code and notebooks developed for a comprehensive credit risk modeling thesis, focusing on model development, validation, and interpretability techniques.

## Project Structure

The project is organized into several key directories:

### 1. data_exploration_analysis/
Contains exploratory data analysis tools and utilities:
- eda_utils/eda_utils.py: Core functions for data exploration and preprocessing
- eda_utils/utils_plot.py: Visualization utilities for exploratory analysis

### 2. feature_eng/
Feature engineering notebooks and utilities for credit risk modeling.

### 3. models/
Model development and training:
- model_utils/model_calibration.py: Functions for calibrating probability estimates
- model_utils/modeling_utils.py: Core utilities for model training, evaluation, and hyperparameter tuning

### 4. model_risk_validations/
Risk validation framework and analysis:
- validations_models.ipynb: Main notebook for model risk validation analysis
- risk_validations_utils/risk_validation_phase1.py: Utilities for Phase 1 risk validation
- images_risk_validations/: Directory containing risk validation visualizations

### 5. performance_validations/
Model performance tracking and stability analysis:
- performance_validations.ipynb: Main notebook for performance validation over time
- performance_model_utils/stability_utils.py: Utilities for calculating stability metrics (PSI, etc.)
- performance_model_utils/stability_plots.py: Visualization functions for stability metrics

### 6. interpretability_validations/
Model interpretability framework:
- model_interpretability_framework.ipynb: Main notebook implementing LIME and SHAP interpretability
- images_interpretability/: Directory containing interpretability visualizations

## Key Components

### Risk Validation Framework
The risk validation framework evaluates model performance across risk deciles, analyzing:
- Default rates and loss rates by decile
- Expected loss calculations
- Economic value analysis with cumulative savings
- Model comparison across different sampling techniques (Baseline, Weighted, Undersampled, XGBoost)

### Performance Validation
Tracks model performance over time using:
- AUC, KS, and other discrimination metrics
- Calibration metrics (O/E ratio, Brier Score, ECE)
- Population Stability Index (PSI)
- Decile distribution stability

### Interpretability Framework
Implements and compares two key interpretability methods:
1. SHAP (SHapley Additive exPlanations):
   - Global feature importance based on average absolute SHAP values
   - Individual prediction explanations with force plots
   - Dependency plots for key features

2. LIME (Local Interpretable Model-agnostic Explanations):
   - Custom implementation for global feature importance through aggregation
   - Local explanations for individual credit decisions
   - Integration with WoE-transformed variables

### WoE Variable Analysis
Visualizes Weight of Evidence distributions for key variables identified by SHAP/LIME to enhance interpretability of transformed features.

## Usage Instructions

### Model Risk Validation
Run the validations_models.ipynb notebook to:
- Calculate comprehensive risk metrics by decile
- Generate economic value analysis
- Compare model performance across different sampling techniques

### Performance Validation
Run the performance_validations.ipynb notebook to:
- Track discrimination and calibration metrics over time
- Analyze model stability using PSI and decile distributions
- Generate stability visualizations

### Model Interpretability
Run the model_interpretability_framework.ipynb notebook to:
- Generate global feature importance using SHAP and LIME
- Create individual prediction explanations
- Visualize WoE distributions for key variables

## Implementation Notes

1. Color Scheme:
   - Baseline: #1560bd
   - Weighted: #75caed
   - Undersampled: #8B7EC8
   - XGBoost: #d62728

2. Visualization Standards:
   - All plots use Plotly with white background
   - Consistent color scheme across all visualizations
   - Passive voice descriptions for thesis documentation

3. Key Metrics:
   - Default rate
   - Expected loss
   - Net savings vs baseline
   - Interest revenue

## Dependencies
The project requires the following key libraries:
- pandas, numpy: Data manipulation
- scikit-learn: Model training and evaluation
- xgboost: XGBoost implementation
- shap: SHAP implementation for model interpretability
- lime: LIME implementation for model interpretability
- plotly: Interactive visualizations
- matplotlib, seaborn: Static visualizations

## Contact
For questions regarding this thesis project, please contact the author.
