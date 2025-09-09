"""
Model Calibration and Temporal Performance Metrics

This module provides functions to calculate and visualize model calibration metrics
and performance over time for credit risk models.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import warnings

# Set default plotly template to white background
import plotly.io as pio
pio.templates.default = "plotly_white"


def calculate_auc_by_month(df, score_cols, target_col='target', date_col='d_vintage'):
    """
    Calculate AUC scores by month for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing AUC scores by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_auc'] = []
    
    # Calculate AUC for each month and model
    for month in sorted(df['month_year'].unique()):
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data or no defaults
        if len(month_df) < 100 or month_df[target_col].sum() == 0:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                auc = roc_auc_score(month_df[target_col], month_df[col])
                results[f'{col}_auc'].append(auc)
            except:
                results[f'{col}_auc'].append(np.nan)
    
    return pd.DataFrame(results)


def calculate_ks_by_month(df, score_cols, target_col='target', date_col='d_vintage'):
    """
    Calculate Kolmogorov-Smirnov statistic by month for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing KS scores by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_ks'] = []
    
    # Calculate KS for each month and model
    for month in sorted(df['month_year'].unique()):
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data or no defaults
        if len(month_df) < 100 or month_df[target_col].sum() == 0:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                # Calculate KS statistic
                pos_scores = month_df.loc[month_df[target_col] == 1, col]
                neg_scores = month_df.loc[month_df[target_col] == 0, col]
                
                # Use scipy's ks_2samp function
                ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
                results[f'{col}_ks'].append(ks_stat)
            except:
                results[f'{col}_ks'].append(np.nan)
    
    return pd.DataFrame(results)


def calculate_brier_score_by_month(df, score_cols, target_col='target', date_col='d_vintage'):
    """
    Calculate Brier score by month for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores (probabilities)
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing Brier scores by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_brier'] = []
    
    # Calculate Brier score for each month and model
    for month in sorted(df['month_year'].unique()):
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data
        if len(month_df) < 100:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                brier = brier_score_loss(month_df[target_col], month_df[col])
                results[f'{col}_brier'].append(brier)
            except:
                results[f'{col}_brier'].append(np.nan)
    
    return pd.DataFrame(results)


def calculate_observed_expected_ratio(df, score_cols, target_col='target', date_col='d_vintage', n_bins=10):
    """
    Calculate Observed vs Expected ratio by month for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores (probabilities)
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
    n_bins : int, default=10
        Number of bins to use for calculating O/E ratio
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing O/E ratios by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_oe_ratio'] = []
    
    # Calculate O/E ratio for each month and model
    for month in sorted(df['month_year'].unique()):
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data
        if len(month_df) < 100:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                # Calculate observed default rate
                observed = month_df[target_col].mean()
                
                # Calculate expected default rate (average predicted probability)
                expected = month_df[col].mean()
                
                # Calculate O/E ratio
                oe_ratio = observed / expected if expected > 0 else np.nan
                results[f'{col}_oe_ratio'].append(oe_ratio)
            except:
                results[f'{col}_oe_ratio'].append(np.nan)
    
    return pd.DataFrame(results)


def calculate_expected_calibration_error(df, score_cols, target_col='target', date_col='d_vintage', n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) by month for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores (probabilities)
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
    n_bins : int, default=10
        Number of bins to use for calculating ECE
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing ECE by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_ece'] = []
    
    # Calculate ECE for each month and model
    for month in sorted(df['month_year'].unique()):
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data
        if len(month_df) < 100:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                # Create bins based on predicted probabilities
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_indices = np.digitize(month_df[col], bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                
                # Calculate ECE
                ece = 0
                for bin_idx in range(n_bins):
                    bin_mask = (bin_indices == bin_idx)
                    if np.sum(bin_mask) > 0:
                        bin_prob = np.mean(month_df.loc[bin_mask, col])
                        bin_acc = np.mean(month_df.loc[bin_mask, target_col])
                        bin_size = np.sum(bin_mask) / len(month_df)
                        ece += bin_size * np.abs(bin_acc - bin_prob)
                
                results[f'{col}_ece'].append(ece)
            except:
                results[f'{col}_ece'].append(np.nan)
    
    return pd.DataFrame(results)


def plot_metric_over_time(df, metric_cols, title, y_axis_title, height=500, width=900):
    """
    Plot metrics over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the metrics and month_year column
    metric_cols : list
        List of column names containing metrics to plot
    title : str
        Title of the plot
    y_axis_title : str
        Y-axis title
    height : int, default=500
        Height of the plot
    width : int, default=900
        Width of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Convert month_year to string for better display
    df = df.copy()
    df['month_year_str'] = df['month_year'].astype(str)
    
    # Define colors for each model
    colors = {
        'baseline': '#1560bd',   
        'weighted': '#75caed',     
        'undersampled':  '#8B7EC8', 
        'xgboost': '#d62728'        
    }

    # Create figure
    fig = go.Figure()
    
    # Add traces for each metric
    for col in metric_cols:
        model_name = col.split('_')[0]  # Extract model name from column name
        fig.add_trace(
            go.Scatter(
                x=df['month_year_str'],
                y=df[col],
                mode='lines+markers',
                name=model_name.capitalize(),
                line=dict(width=2, color=colors.get(model_name, '#213092')),
                marker=dict(size=8, color=colors.get(model_name, '#213092'))
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title=y_axis_title,
        legend_title='Model',
        height=height,
        width=width,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_calibration_curve(df, score_cols, target_col='target', n_bins=10, height=500, width=900):
    """
    Plot calibration curves for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores (probabilities)
    target_col : str, default='target'
        Name of the column containing target variable
    n_bins : int, default=10
        Number of bins to use for calibration curve
    height : int, default=500
        Height of the plot
    width : int, default=900
        Width of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Define colors for each model
    colors = {
        'baseline': '#1560bd',   
        'weighted': '#75caed',     
        'undersampled':  '#8B7EC8', 
        'xgboost': '#d62728'        
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line (perfect calibration)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(color='black', dash='dash', width=1.5)
        )
    )
    
    # Calculate and add calibration curve for each model
    for col in score_cols:
        # Create bins based on predicted probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(df[col], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate mean predicted probability and observed frequency for each bin
        mean_predicted_probs = []
        observed_freqs = []
        
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:
                mean_predicted_probs.append(np.mean(df.loc[bin_mask, col]))
                observed_freqs.append(np.mean(df.loc[bin_mask, target_col]))
        
        # Add trace for this model
        model_name = col.split('_')[0]  # Extract model name from column name
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_probs,
                y=observed_freqs,
                mode='lines+markers',
                name=model_name.capitalize(),
                line=dict(width=2, color=colors.get(model_name, '#213092')),
                marker=dict(size=8, color=colors.get(model_name, '#213092'))
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Observed Frequency',
        legend_title='Model',
        height=height,
        width=width,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    Parameters:
    -----------
    expected : array-like
        Expected (reference) distribution
    actual : array-like
        Actual (current) distribution
    bins : int or array-like, default=10
        Number of bins or bin edges
        
    Returns:
    --------
    float
        PSI value
    """
    # Create bins if integer is provided
    if isinstance(bins, int):
        # Determine bin edges based on the expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        # Add a small epsilon to the last bin edge to include the maximum value
        if len(bin_edges) > 1:
            bin_edges[-1] += 1e-8
    else:
        bin_edges = bins
    
    # Calculate bin counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert counts to percentages
    expected_pct = expected_counts / np.sum(expected_counts)
    actual_pct = actual_counts / np.sum(actual_counts)
    
    # Replace zeros with a small epsilon to avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi


def calculate_psi_over_time(df, score_cols, date_col='d_vintage', reference_date=None, bins=10):
    """
    Calculate PSI over time for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores
    date_col : str, default='d_vintage'
        Name of the column containing date information
    reference_date : str or None, default=None
        Reference date to use as the expected distribution
        If None, the first date will be used
    bins : int, default=10
        Number of bins to use for PSI calculation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing PSI values by month for each model
    """
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Sort unique months
    unique_months = sorted(df['month_year'].unique())
    
    # Set reference date
    if reference_date is None:
        reference_month = unique_months[0]
    else:
        reference_date = pd.to_datetime(reference_date)
        reference_month = reference_date.to_period('M')
    
    # Get reference data
    reference_data = {}
    for col in score_cols:
        reference_data[col] = df.loc[df['month_year'] == reference_month, col].values
    
    # Initialize results dictionary
    results = {'month_year': []}
    for col in score_cols:
        results[f'{col}_psi'] = []
    
    # Calculate PSI for each month and model
    for month in unique_months:
        # Skip reference month
        if month == reference_month:
            continue
            
        month_df = df[df['month_year'] == month]
        
        # Skip months with insufficient data
        if len(month_df) < 100:
            continue
            
        results['month_year'].append(month)
        
        for col in score_cols:
            try:
                # Calculate PSI
                psi = calculate_psi(reference_data[col], month_df[col].values, bins=bins)
                results[f'{col}_psi'].append(psi)
            except:
                results[f'{col}_psi'].append(np.nan)
    
    return pd.DataFrame(results)


def plot_score_distribution_by_month(df, score_col, date_col='d_vintage', n_months=6, height=500, width=900):
    """
    Plot score distribution for multiple months.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_col : str
        Column name containing model scores
    date_col : str, default='d_vintage'
        Name of the column containing date information
    n_months : int, default=6
        Number of most recent months to plot
    height : int, default=500
        Height of the plot
    width : int, default=900
        Width of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Define colors for each model
    colors = {
        'baseline': '#1560bd',   
        'weighted': '#75caed',     
        'undersampled':  '#8B7EC8', 
        'xgboost': '#d62728'        
    }
    
    # Convert date column to datetime if it's not already
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract month and year
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Get the most recent n_months
    unique_months = sorted(df['month_year'].unique())
    if len(unique_months) > n_months:
        selected_months = unique_months[-n_months:]
    else:
        selected_months = unique_months
    
    # Filter data for selected months
    filtered_df = df[df['month_year'].isin(selected_months)].copy()
    filtered_df['month_year_str'] = filtered_df['month_year'].astype(str)
    
    # Create figure
    fig = go.Figure()
    
    # Get model name from score column
    model_name = score_col.split('_')[0].lower()
    model_color = colors.get(model_name, '#213092')
    
    # Add histogram for each month
    for month in selected_months:
        month_df = filtered_df[filtered_df['month_year'] == month]
        month_str = month.strftime('%Y-%m')
        
        fig.add_trace(
            go.Histogram(
                x=month_df[score_col],
                name=month_str,
                opacity=0.7,
                nbinsx=30,
                marker_color=model_color
            )
        )
    
    # Update layout
    model_name = score_col.split('_')[0].capitalize()
    fig.update_layout(
        title=f'{model_name} Score Distribution Over Time',
        xaxis_title='Score',
        yaxis_title='Count',
        legend_title='Month',
        height=height,
        width=width,
        template='plotly_white',
        barmode='overlay'
    )
    
    return fig


def run_temporal_analysis(df, score_cols, target_col='target', date_col='d_vintage'):
    """
    Run comprehensive temporal analysis for multiple models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    score_cols : list
        List of column names containing model scores
    target_col : str, default='target'
        Name of the column containing target variable
    date_col : str, default='d_vintage'
        Name of the column containing date information
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames with temporal metrics and figures
    """
    results = {}
    
    # Calculate metrics over time
    print("Calculating AUC by month...")
    results['auc_by_month'] = calculate_auc_by_month(df, score_cols, target_col, date_col)
    
    print("Calculating KS by month...")
    results['ks_by_month'] = calculate_ks_by_month(df, score_cols, target_col, date_col)
    
    print("Calculating Brier score by month...")
    results['brier_by_month'] = calculate_brier_score_by_month(df, score_cols, target_col, date_col)
    
    print("Calculating O/E ratio by month...")
    results['oe_ratio_by_month'] = calculate_observed_expected_ratio(df, score_cols, target_col, date_col)
    
    print("Calculating ECE by month...")
    results['ece_by_month'] = calculate_expected_calibration_error(df, score_cols, target_col, date_col)
    
    print("Calculating PSI over time...")
    results['psi_by_month'] = calculate_psi_over_time(df, score_cols, date_col)
    
    # Create plots
    print("Creating plots...")
    auc_cols = [col for col in results['auc_by_month'].columns if col.endswith('_auc')]
    results['auc_plot'] = plot_metric_over_time(
        results['auc_by_month'], 
        auc_cols, 
        'AUC Over Time', 
        'AUC'
    )
    
    ks_cols = [col for col in results['ks_by_month'].columns if col.endswith('_ks')]
    results['ks_plot'] = plot_metric_over_time(
        results['ks_by_month'], 
        ks_cols, 
        'KS Statistic Over Time', 
        'KS'
    )
    
    brier_cols = [col for col in results['brier_by_month'].columns if col.endswith('_brier')]
    results['brier_plot'] = plot_metric_over_time(
        results['brier_by_month'], 
        brier_cols, 
        'Brier Score Over Time', 
        'Brier Score'
    )
    
    oe_cols = [col for col in results['oe_ratio_by_month'].columns if col.endswith('_oe_ratio')]
    results['oe_plot'] = plot_metric_over_time(
        results['oe_ratio_by_month'], 
        oe_cols, 
        'Observed/Expected Ratio Over Time', 
        'O/E Ratio'
    )
    
    ece_cols = [col for col in results['ece_by_month'].columns if col.endswith('_ece')]
    results['ece_plot'] = plot_metric_over_time(
        results['ece_by_month'], 
        ece_cols, 
        'Expected Calibration Error Over Time', 
        'ECE'
    )
    
    psi_cols = [col for col in results['psi_by_month'].columns if col.endswith('_psi')]
    results['psi_plot'] = plot_metric_over_time(
        results['psi_by_month'], 
        psi_cols, 
        'Population Stability Index Over Time', 
        'PSI'
    )
    
    # Create calibration curve
    results['calibration_curve'] = plot_calibration_curve(df, score_cols, target_col)
    
    # Create score distribution plots
    results['score_distribution_plots'] = {}
    for col in score_cols:
        results['score_distribution_plots'][col] = plot_score_distribution_by_month(df, col, date_col)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("This module provides functions for model calibration and temporal performance analysis.")
    print("Import this module and use the functions in your notebook.")
