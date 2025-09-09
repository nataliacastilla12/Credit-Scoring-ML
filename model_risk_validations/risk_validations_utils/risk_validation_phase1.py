import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHASE 1: RISK VALIDATION ANALYSIS
# =============================================================================

def create_simple_qcut_deciles(df, 
                               score_columns=['baseline_score', 'weighted_score', 'undersampled_score', 'xgboost_score'],
                               set_col='Set', 
                               target_col='target'):
    """
    Creates deciles using pandas qcut for each model independently
    This is the standard approach that properly captures model differences
    """
    print("=" * 70)
    print("PHASE 1: RISK VALIDATION - STANDARD QCUT DECILES")
    print("=" * 70)
    
    # Filter to Test set with target only for reference
    test_df = df[(df[set_col] == 'Test') & (df[target_col].notna())].copy()
    print(f"Using {len(test_df):,} Test observations with target for validation")
    print()
    
    df_with_deciles = df.copy()
    decile_info = {}
    
    for score_col in score_columns:
        if score_col not in df.columns:
            print(f"WARNING: Column {score_col} not found. Skipping...")
            continue
            
        model_name = score_col.replace('_score', '').title()
        decile_col_name = f'{model_name}_decile'
        
        try:
            # Use pandas qcut to create deciles for the FULL dataset
            df_with_deciles[decile_col_name] = pd.qcut(
                df[score_col], 
                10, 
                labels=range(1, 11),
                duplicates='drop'  # Handle ties properly
            )
            
            # Verify distribution in test set
            test_distribution = df_with_deciles[df_with_deciles[set_col] == 'Test'][decile_col_name].value_counts().sort_index()
            
            print(f"Decile distribution for {model_name} (Test set):")
            for decile, count in test_distribution.items():
                pct = count / len(df_with_deciles[df_with_deciles[set_col] == 'Test']) * 100
                print(f"  Decile {decile}: {count:,} observations ({pct:.1f}%)")
            
            # Get score boundaries for reference
            test_with_deciles = df_with_deciles[df_with_deciles[set_col] == 'Test']
            boundaries = []
            for decile in range(1, 11):
                decile_data = test_with_deciles[test_with_deciles[decile_col_name] == decile]
                if len(decile_data) > 0:
                    min_score = decile_data[score_col].min()
                    max_score = decile_data[score_col].max()
                    boundaries.append((decile, min_score, max_score))
            
            print(f"Score boundaries for {model_name}:")
            for decile, min_val, max_val in boundaries[:3]:  # Show first 3
                print(f"  Decile {decile}: {min_val:.4f} to {max_val:.4f}")
            print(f"  ... (showing first 3 deciles)")
            
            # Store decile information
            decile_info[model_name] = {
                'boundaries': boundaries,
                'test_distribution': test_distribution,
                'method': 'pandas_qcut'
            }
            
            print(f"✓ Successfully created {decile_col_name}")
            print()
            
        except Exception as e:
            print(f"ERROR creating deciles for {score_col}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Verify model differences
    print("=" * 50)
    print("VERIFICATION OF MODEL DIFFERENCES")
    print("=" * 50)
    
    test_with_deciles = df_with_deciles[df_with_deciles[set_col] == 'Test']
    decile_cols = [col for col in test_with_deciles.columns if col.endswith('_decile')]
    
    if len(decile_cols) >= 2:
        # Compare first two models
        col1, col2 = decile_cols[0], decile_cols[1]
        same_deciles = (test_with_deciles[col1] == test_with_deciles[col2]).sum()
        total = len(test_with_deciles)
        
        print(f"Comparison {col1} vs {col2}:")
        print(f"Same decile: {same_deciles:,}/{total:,} ({same_deciles/total*100:.1f}%)")
        print(f"Different decile: {total-same_deciles:,}/{total:,} ({(total-same_deciles)/total*100:.1f}%)")
        
        if len(decile_cols) >= 4:
            # Compare Baseline vs XGBoost (most different)
            baseline_col = [col for col in decile_cols if 'Baseline' in col][0]
            xgboost_col = [col for col in decile_cols if 'Xgboost' in col][0]
            
            same_bx = (test_with_deciles[baseline_col] == test_with_deciles[xgboost_col]).sum()
            print(f"\nComparison {baseline_col} vs {xgboost_col}:")
            print(f"Same decile: {same_bx:,}/{total:,} ({same_bx/total*100:.1f}%)")
            print(f"Different decile: {total-same_bx:,}/{total:,} ({(total-same_bx)/total*100:.1f}%)")
    
    print("\n✅ Decile creation completed with proper model differentiation!")
    
    return df_with_deciles, decile_info

def calculate_comprehensive_risk_metrics(df, 
                                       target_col='target',
                                       real_loan_amount_col='real_loan_amnt',
                                       set_col='Set',
                                       lgd_col=None,
                                       interest_rate_col=None,
                                       term_col=None,
                                       default_lgd=0.6,
                                       default_interest_rate=0.15,
                                       default_term=36):
    """
    Calculate comprehensive risk metrics at loan level and then aggregate by decile
    """
    # Filter to Test set with target
    test_df = df[(df[set_col] == 'Test') & (df[target_col].notna())].copy()
    
    print("Filtering diagnostics:")
    print(f"Total rows in dataset: {len(df):,}")
    print(f"Test set rows: {len(df[df[set_col] == 'Test']):,} ({len(df[df[set_col] == 'Test'])/len(df)*100:.1f}% of total)")
    print(f"Test set with non-null target: {len(test_df):,} ({len(test_df)/len(df[df[set_col] == 'Test'])*100:.1f}% of Test set)")
    
    print("\nCalculating comprehensive risk metrics for Test set...")
    print(f"Test set size: {len(test_df):,} observations")
    print()
    
    # Find decile columns
    decile_columns = [col for col in test_df.columns if col.endswith('_decile')]
    
    if not decile_columns:
        # Try alternative naming patterns for decile columns
        decile_columns = [col for col in test_df.columns if 'decile' in col.lower()]
        if not decile_columns:
            raise ValueError("No decile columns found!")
    
    print(f"Found {len(decile_columns)} decile columns: {decile_columns}")
    
    # Overall statistics
    overall_default_rate = test_df[target_col].mean()
    total_defaults = test_df[target_col].sum()
    total_exposure = test_df[real_loan_amount_col].sum() if real_loan_amount_col in test_df.columns else 0
    
    print(f"Overall Statistics:")
    print(f"  Default Rate: {overall_default_rate:.3f} ({overall_default_rate*100:.1f}%)")
    print(f"  Total Defaults: {total_defaults:,}")
    if total_exposure > 0:
        print(f"  Total Exposure: ${total_exposure:,.0f}")
    print()
    
    # Calculate loan-level metrics first
    test_df['default'] = test_df[target_col].astype(int)  # Ensure binary
    
    # Add loan-specific metrics or use defaults
    if real_loan_amount_col in test_df.columns:
        test_df['loan_amount'] = test_df[real_loan_amount_col]
    else:
        # If no real loan amount is available, use a default value but warn the user
        print("WARNING: No loan amount column found. Using default value of 1.0.")
        test_df['loan_amount'] = 1.0  # Default value of 1.0 instead of 1000
    
    # LGD - either loan-specific or default
    if lgd_col and lgd_col in test_df.columns:
        if isinstance(test_df[lgd_col].iloc[0], str) and '%' in str(test_df[lgd_col].iloc[0]):
            # Convert percentage strings to float (e.g., '60%' to 0.6)
            test_df['lgd'] = test_df[lgd_col].str.rstrip('%').astype(float) / 100
        else:
            test_df['lgd'] = test_df[lgd_col]
    else:
        test_df['lgd'] = default_lgd
    
    # Interest rate - either loan-specific or default
    if interest_rate_col and interest_rate_col in test_df.columns:
        if isinstance(test_df[interest_rate_col].iloc[0], str) and '%' in str(test_df[interest_rate_col].iloc[0]):
            # Convert percentage strings to float (e.g., '5.2%' to 0.052)
            test_df['interest_rate'] = test_df[interest_rate_col].str.rstrip('%').astype(float) / 100
        else:
            test_df['interest_rate'] = test_df[interest_rate_col]
    else:
        test_df['interest_rate'] = default_interest_rate
    
    # Term - either loan-specific or default
    if term_col and term_col in test_df.columns:
        if not pd.api.types.is_numeric_dtype(test_df[term_col]):
            # Extract numeric part from strings like '36 months'
            test_df['term'] = test_df[term_col].str.extract('(\d+)').astype(float).values.flatten()
        else:
            test_df['term'] = test_df[term_col]
    else:
        test_df['term'] = default_term
    
    # Calculate loan-level metrics
    # Expected loss (default * loan_amount * lgd)
    test_df['expected_loss'] = test_df['default'] * test_df['loan_amount'] * test_df['lgd']

    # Annualized expected loss
    test_df['annualized_expected_loss'] = test_df['expected_loss'] * (12 / test_df['term'])
    
    # Interest revenue (annualized)
    test_df['annual_interest_revenue'] = test_df['loan_amount'] * test_df['interest_rate']
    
    # Interest revenue over the full term
    test_df['total_interest_revenue'] = test_df['annual_interest_revenue'] * (test_df['term'] / 12)
    
    # Results dictionary to store all model metrics
    results = {}
    
    for decile_col in decile_columns:
        model_name = decile_col.replace('_decile', '')
        print(f"Analyzing {model_name} model...")
        
        # Check if decile column has valid values
        decile_counts = test_df[decile_col].value_counts().sort_index()
        print(f"Decile distribution for {model_name}:")
        for decile, count in decile_counts.items():
            print(f"  Decile {decile}: {count:,} loans ({count/len(test_df)*100:.1f}%)")
        
        # Add decile-specific metrics for this model
        model_df = test_df.copy()
        model_df['decile'] = model_df[decile_col]
        
        # Initialize columns for cutoff metrics
        model_df['loss_savings'] = 0.0
        model_df['revenue_opportunity_cost'] = 0.0
        model_df['net_savings'] = 0.0
        
        # Calculate loss savings and revenue opportunity cost for each decile
        # These metrics represent what would happen if we reject loans with decile > current decile
        for decile in range(1, 11):
            # For each decile, calculate what would happen if we reject all loans with higher deciles
            higher_deciles_mask = model_df['decile'] > decile
            
            # Calculate metrics for this decile
            # Loss savings = expected loss of rejected loans (what we avoid by rejecting)
            loss_savings = model_df.loc[higher_deciles_mask, 'annualized_expected_loss'].sum()
            
            # Revenue opportunity cost = interest revenue of rejected loans (what we give up by rejecting)
            revenue_opportunity_cost = model_df.loc[higher_deciles_mask, 'annual_interest_revenue'].sum()
            
            # Net savings = loss savings - revenue opportunity cost
            net_savings = loss_savings - revenue_opportunity_cost
            
            # Store these values in the corresponding decile row
            model_df.loc[model_df['decile'] == decile, 'loss_savings'] = loss_savings
            model_df.loc[model_df['decile'] == decile, 'revenue_opportunity_cost'] = revenue_opportunity_cost
            model_df.loc[model_df['decile'] == decile, 'net_savings'] = net_savings
        
        # Define helper functions for weighted calculations to avoid lambda issues
        def weighted_lgd(x):
            return (x['lgd'] * x['loan_amount']).sum() / x['loan_amount'].sum() if x['loan_amount'].sum() > 0 else 0
        
        def weighted_interest_rate(x):
            return (x['interest_rate'] * x['loan_amount']).sum() / x['loan_amount'].sum() if x['loan_amount'].sum() > 0 else 0
        
        def weighted_term(x):
            return (x['term'] * x['loan_amount']).sum() / x['loan_amount'].sum() if x['loan_amount'].sum() > 0 else 0
        
        # Create a new DataFrame to store the aggregated metrics by decile
        decile_metrics = pd.DataFrame(index=range(1, 11))
        decile_metrics.index.name = 'decile'
        decile_metrics = decile_metrics.reset_index()
        
        # Calculate metrics for each decile manually
        for decile in range(1, 11):
            decile_df = model_df[model_df['decile'] == decile]
            
            # Basic metrics
            decile_metrics.loc[decile-1, 'total_loans'] = len(decile_df)
            decile_metrics.loc[decile-1, 'total_exposure'] = decile_df['loan_amount'].sum()
            decile_metrics.loc[decile-1, 'avg_loan_amount'] = decile_df['loan_amount'].mean()
            decile_metrics.loc[decile-1, 'median_loan_amount'] = decile_df['loan_amount'].median()
            decile_metrics.loc[decile-1, 'defaults'] = decile_df['default'].sum()
            decile_metrics.loc[decile-1, 'default_rate'] = decile_df['default'].mean()
            decile_metrics.loc[decile-1, 'expected_loss'] = decile_df['expected_loss'].sum()
            decile_metrics.loc[decile-1, 'annualized_expected_loss'] = decile_df['annualized_expected_loss'].sum()
            decile_metrics.loc[decile-1, 'annual_interest_revenue'] = decile_df['annual_interest_revenue'].sum()
            decile_metrics.loc[decile-1, 'total_interest_revenue'] = decile_df['total_interest_revenue'].sum()
            decile_metrics.loc[decile-1, 'avg_lgd'] = decile_df['lgd'].mean()
            decile_metrics.loc[decile-1, 'avg_interest_rate'] = decile_df['interest_rate'].mean()
            decile_metrics.loc[decile-1, 'avg_term'] = decile_df['term'].mean()
            
            # Cutoff metrics (first loan with this decile)
            first_loan = decile_df.iloc[0] if len(decile_df) > 0 else None
            if first_loan is not None:
                decile_metrics.loc[decile-1, 'loss_savings_first'] = first_loan['loss_savings']
                decile_metrics.loc[decile-1, 'revenue_opportunity_cost_first'] = first_loan['revenue_opportunity_cost']
                decile_metrics.loc[decile-1, 'net_savings_first'] = first_loan['net_savings']
            else:
                decile_metrics.loc[decile-1, 'loss_savings_first'] = 0
                decile_metrics.loc[decile-1, 'revenue_opportunity_cost_first'] = 0
                decile_metrics.loc[decile-1, 'net_savings_first'] = 0
            
            # Weighted metrics
            if len(decile_df) > 0:
                decile_metrics.loc[decile-1, 'weighted_lgd'] = weighted_lgd(decile_df)
                decile_metrics.loc[decile-1, 'weighted_interest_rate'] = weighted_interest_rate(decile_df)
                decile_metrics.loc[decile-1, 'weighted_term'] = weighted_term(decile_df)
            else:
                decile_metrics.loc[decile-1, 'weighted_lgd'] = 0
                decile_metrics.loc[decile-1, 'weighted_interest_rate'] = 0
                decile_metrics.loc[decile-1, 'weighted_term'] = 0
            
            # Additional metrics
            decile_metrics.loc[decile-1, 'non_defaults'] = decile_metrics.loc[decile-1, 'total_loans'] - decile_metrics.loc[decile-1, 'defaults']
            decile_metrics.loc[decile-1, 'defaulted_exposure'] = decile_df[decile_df['default'] == 1]['loan_amount'].sum()
            
            # Loss rate calculation
            if decile_metrics.loc[decile-1, 'total_exposure'] > 0:
                decile_metrics.loc[decile-1, 'loss_rate'] = (decile_metrics.loc[decile-1, 'defaulted_exposure'] * 
                                                           decile_metrics.loc[decile-1, 'weighted_lgd'] / 
                                                           decile_metrics.loc[decile-1, 'total_exposure'] * 100)
            else:
                decile_metrics.loc[decile-1, 'loss_rate'] = 0
            
            # Portfolio percentages
            decile_metrics.loc[decile-1, 'pct_of_loans'] = (decile_metrics.loc[decile-1, 'total_loans'] / len(model_df)) * 100
            decile_metrics.loc[decile-1, 'pct_of_defaults'] = (decile_metrics.loc[decile-1, 'defaults'] / total_defaults) * 100 if total_defaults > 0 else 0
            decile_metrics.loc[decile-1, 'pct_of_exposure'] = (decile_metrics.loc[decile-1, 'total_exposure'] / total_exposure) * 100 if total_exposure > 0 else 0
            
            # Lift
            decile_metrics.loc[decile-1, 'lift'] = decile_metrics.loc[decile-1, 'default_rate'] / overall_default_rate if overall_default_rate > 0 else 0
        
        # Add cumulative metrics
        decile_metrics['cumulative_defaults_pct'] = decile_metrics['pct_of_defaults'].cumsum()
        decile_metrics['cumulative_exposure_pct'] = decile_metrics['pct_of_exposure'].cumsum()
        decile_metrics['cumulative_expected_loss'] = decile_metrics['expected_loss'].cumsum()
        
        # Add model name
        decile_metrics['model'] = model_name
        
        # Store both the aggregated results and the loan-level data
        results[model_name] = {
            'decile_metrics': decile_metrics,
            'loan_data': model_df  # Include the loan-level data with all metrics
        }
        
        # Print summary statistics for verification
        print(f"\nSummary for {model_name}:")
        print(f"  Total loans: {len(model_df):,}")
        print(f"  Total defaults: {total_defaults:,}")
        print(f"  Overall default rate: {overall_default_rate*100:.2f}%")
        
        # Calculate metrics for decile 6 cutoff
        decile_6_loans = model_df[model_df['decile'] <= 6].shape[0]
        decile_6_defaults = model_df[(model_df['decile'] <= 6) & (model_df['default'] == 1)].shape[0]
        decile_6_default_rate = decile_6_defaults / decile_6_loans if decile_6_loans > 0 else 0
        decile_6_exposure = model_df[model_df['decile'] <= 6]['loan_amount'].sum()
        decile_6_defaulted_exposure = model_df[(model_df['decile'] <= 6) & (model_df['default'] == 1)]['loan_amount'].sum()
        decile_6_loss_rate = (decile_6_defaulted_exposure / decile_6_exposure * 100) if decile_6_exposure > 0 else 0
        decile_6_expected_loss = model_df[(model_df['decile'] <= 6)]['expected_loss'].sum()
        
        print(f"\n  Decile 6 cutoff metrics:")
        print(f"    Approved loans: {decile_6_loans:,} ({decile_6_loans/len(model_df)*100:.1f}% of total)")
        print(f"    Default rate: {decile_6_default_rate*100:.2f}%")
        print(f"    Loss rate: {decile_6_loss_rate:.2f}%")
        print(f"    Expected loss: ${decile_6_expected_loss/1000000:.2f}M")
        
        print(f"✓ Completed risk analysis for {model_name}")
    
    return results

    
def create_risk_validation_summary_table(risk_results):
    """
    Create comprehensive summary tables for risk validation
    """
    print("=" * 80)
    print("RISK VALIDATION SUMMARY TABLES")
    print("=" * 80)
    
    # 1. Default Rate and Lift Comparison
    print("\n1. DEFAULT RATE AND LIFT BY DECILE:")
    print("-" * 50)
    
    default_lift_data = []
    for model, df in risk_results.items():
        for _, row in df['decile_metrics'].iterrows():
            default_lift_data.append({
                'Model': model,
                'Decile': row['decile'],
                'Default_Rate': f"{row['default_rate']:.3f}",
                'Lift': f"{row['lift']:.2f}"
            })
    
    default_lift_df = pd.DataFrame(default_lift_data)
    default_pivot = default_lift_df.pivot_table(
        index='Decile', 
        columns='Model', 
        values=['Default_Rate', 'Lift'], 
        aggfunc='first'
    )
    print(default_pivot)
    
    # 2. Loss Rate Comparison
    print("\n\n2. LOSS RATE BY DECILE (%):")
    print("-" * 40)
    
    loss_data = []
    for model, df in risk_results.items():
        for _, row in df['decile_metrics'].iterrows():
            loss_data.append({
                'Model': model,
                'Decile': row['decile'],
                'Loss_Rate': f"{row['loss_rate']:.2f}%"
            })
    
    loss_df = pd.DataFrame(loss_data)
    loss_pivot = loss_df.pivot_table(
        index='Decile', 
        columns='Model', 
        values='Loss_Rate', 
        aggfunc='first'
    )
    print(loss_pivot)
    
    # 3. Expected Loss Comparison
    print("\n\n3. EXPECTED LOSS BY DECILE ($):")
    print("-" * 40)
    
    exp_loss_data = []
    for model, df in risk_results.items():
        for _, row in df['decile_metrics'].iterrows():
            exp_loss_data.append({
                'Model': model,
                'Decile': row['decile'],
                'Expected_Loss': f"${row['expected_loss']:,.0f}"
            })
    
    exp_loss_df = pd.DataFrame(exp_loss_data)
    exp_loss_pivot = exp_loss_df.pivot_table(
        index='Decile', 
        columns='Model', 
        values='Expected_Loss', 
        aggfunc='first'
    )
    print(exp_loss_pivot)
    
    # 4. Portfolio Distribution
    print("\n\n4. PORTFOLIO DISTRIBUTION:")
    print("-" * 30)
    
    portfolio_data = []
    for model, df in risk_results.items():
        for _, row in df['decile_metrics'].iterrows():
            portfolio_data.append({
                'Model': model,
                'Decile': row['decile'],
                'Loans': f"{row['total_loans']:,}",
                'Portfolio_%': f"{row['pct_of_loans']:.1f}%",
                'Avg_Loan': f"${row['avg_loan_amount']:,.0f}",
                'Exposure_%': f"{row['pct_of_exposure']:.1f}%"
            })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # Show first model as example
    first_model = list(risk_results.keys())[0]
    first_model_data = portfolio_df[portfolio_df['Model'] == first_model]
    print(f"\nPortfolio Distribution ({first_model} Model):")
    print(first_model_data[['Decile', 'Loans', 'Portfolio_%', 'Avg_Loan', 'Exposure_%']].to_string(index=False))
    
    return default_pivot, loss_pivot, exp_loss_pivot, portfolio_df

    

def create_risk_validation_visualizations(risk_results):
    """
    Create consolidated risk validation visualizations
    
    Parameters:
    - risk_results: Dictionary with risk analysis results by model
    
    Returns:
    - Four plotly figures with risk validation visualizations
    """
    # Print the data values used in the visualizations
    print("\n===== RISK VALIDATION VISUALIZATION DATA VALUES =====\n")
    
    # Print default rate by decile for each model
    print("DEFAULT RATE BY DECILE (%)")
    headers = ["Decile"] + list(risk_results.keys())
    print(f"{headers[0]:<10} " + " ".join([f"{model:<15}" for model in headers[1:]]))
    print("-" * 70)
    
    for decile in range(1, 11):
        row = [f"{decile:<10}"] 
        for model, df in risk_results.items():
            model_decile = df['decile_metrics'][df['decile_metrics']['decile'] == decile]
            default_rate = model_decile['default_rate'].values[0] * 100 if len(model_decile) > 0 else 0
            row.append(f"{default_rate:<15.2f}")
        print(" ".join(row))
    
    # Print loss rate by decile for each model
    print("\nLOSS RATE BY DECILE (%)")
    print(f"{headers[0]:<10} " + " ".join([f"{model:<15}" for model in headers[1:]]))
    print("-" * 70)
    
    for decile in range(1, 11):
        row = [f"{decile:<10}"] 
        for model, df in risk_results.items():
            model_decile = df['decile_metrics'][df['decile_metrics']['decile'] == decile]
            loss_rate = model_decile['loss_rate'].values[0] if len(model_decile) > 0 else 0
            row.append(f"{loss_rate:<15.2f}")
        print(" ".join(row))
    
    # Print expected loss by decile for each model
    print("\nEXPECTED LOSS BY DECILE ($)")
    print(f"{headers[0]:<10} " + " ".join([f"{model:<15}" for model in headers[1:]]))
    print("-" * 70)
    
    for decile in range(1, 11):
        row = [f"{decile:<10}"] 
        for model, df in risk_results.items():
            model_decile = df['decile_metrics'][df['decile_metrics']['decile'] == decile]
            expected_loss = model_decile['expected_loss'].values[0] if len(model_decile) > 0 else 0
            row.append(f"{expected_loss:<15.2f}")
        print(" ".join(row))
    # Color palette for models
    colors = {
        'Baseline':  '#1560bd',   
        'Weighted': '#75caed',     
        'Undersampled':  '#8B7EC8', 
        'Xgboost': '#d62728'        
    }
    
    # 1. BASIC METRICS DASHBOARD (2x2 subplots)
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Default Rate by Decile', 'Loss Rate by Decile', 
                       'Expected Loss by Decile ($)', 'Risk Concentration Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Default Rate (top-left)
    for model, df in risk_results.items():
        fig1.add_trace(go.Scatter(
            x=df['decile_metrics']['decile'], y=df['decile_metrics']['default_rate'] * 100,
            mode='lines+markers', name=f'{model}',
            line=dict(color=colors.get(model, '#000000'), width=2),
            marker=dict(size=4), legendgroup=model
        ), row=1, col=1)
    
    # Loss Rate (top-right)
    for model, df in risk_results.items():
        fig1.add_trace(go.Scatter(
            x=df['decile_metrics']['decile'], y=df['decile_metrics']['loss_rate'],
            mode='lines+markers', name=f'{model}',
            line=dict(color=colors.get(model, '#000000'), width=2),
            marker=dict(size=4), showlegend=False, legendgroup=model
        ), row=1, col=2)
    
    # Expected Loss (bottom-left)
    for model, df in risk_results.items():
        fig1.add_trace(go.Bar(
            x=df['decile_metrics']['decile'], y=df['decile_metrics']['expected_loss'],
            name=f'{model}', marker_color=colors.get(model, '#000000'),
            showlegend=False, legendgroup=model, opacity=0.7
        ), row=2, col=1)
    
    # Risk Concentration Analysis (bottom-right)
    concentration_data = {}
    for model, df in risk_results.items():
        top1 = df['decile_metrics']['cumulative_defaults_pct'].iloc[0]  # Top 1 decile
        top2 = df['decile_metrics']['cumulative_defaults_pct'].iloc[1]  # Top 2 deciles  
        top3 = df['decile_metrics']['cumulative_defaults_pct'].iloc[2]  # Top 3 deciles
        concentration_data[model] = [top1, top2, top3]
    
    x_conc = ['Top 1 Decile', 'Top 2 Deciles', 'Top 3 Deciles']
    for i, model in enumerate(concentration_data.keys()):
        fig1.add_trace(go.Bar(
            x=x_conc, y=concentration_data[model],
            name=f'{model}', marker_color=colors.get(model, '#000000'),
            showlegend=False, legendgroup=model, opacity=0.7
        ), row=2, col=2)
    
    # Update layout for dashboard
    fig1.update_layout(
        title='Risk Metrics Dashboard - Model Comparison',
        template='plotly_white',
        font=dict(family="Arial", size=11),
        height=700,
        legend=dict(x=1.02, y=1, xanchor='left'),
        barmode='group'
    )
    
    # Update axes
    fig1.update_xaxes(title_text="Risk Decile", dtick=1, gridcolor='lightgray', row=1, col=1)
    fig1.update_xaxes(title_text="Risk Decile", dtick=1, gridcolor='lightgray', row=1, col=2)
    fig1.update_xaxes(title_text="Risk Decile", dtick=1, gridcolor='lightgray', row=2, col=1)
    fig1.update_xaxes(title_text="Risk Concentration", gridcolor='lightgray', row=2, col=2)
    
    fig1.update_yaxes(title_text="Default Rate (%)", gridcolor='lightgray', row=1, col=1)
    fig1.update_yaxes(title_text="Loss Rate (%)", gridcolor='lightgray', row=1, col=2)
    fig1.update_yaxes(title_text="Expected Loss ($)", gridcolor='lightgray', row=2, col=1)
    fig1.update_yaxes(title_text="% of Total Defaults", gridcolor='lightgray', row=2, col=2)
    
    # 2. LIFT HEATMAP (keep as separate - it's perfect!)
    fig2 = go.Figure()
    
    # Prepare data for heatmap
    lift_data = []
    model_names = []
    
    for model, df in risk_results.items():
        lift_data.append(df['decile_metrics']['lift'].values)
        model_names.append(f'{model} Model')
    
    # Create heatmap
    fig2.add_trace(go.Heatmap(
        z=lift_data,
        x=[f'Decile {i}' for i in range(1, 11)],
        y=model_names,
        colorscale='RdYlBu_r',
        colorbar=dict(title="Lift Value"),
        text=[[f'{val:.2f}' for val in row] for row in lift_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Lift: %{z:.2f}<br><extra></extra>'
    ))
    
    fig2.update_layout(
        title='Lift Heatmap by Model and Decile',
        xaxis_title='Risk Decile (1=Lowest Risk, 10=Highest Risk)',
        yaxis_title='Model',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        height=500,
        margin=dict(l=100, r=50, t=80, b=120),
        annotations=[dict(
            x=0.5, y=-0.25, xref='paper', yref='paper',
            text="Higher lift values (red) indicate better risk discrimination",
            showarrow=False, font=dict(size=10, color="gray")
        )]
    )
    
    # Fix x-axis text overlap
    fig2.update_xaxes(tickangle=0, tickfont=dict(size=10))
    fig2.update_yaxes(tickfont=dict(size=10))
    
    # 3. DEFAULT CAPTURE EFFICIENCY CURVE (separate for better visibility)
    fig3 = go.Figure()
    
    for model, df in risk_results.items():
        # Calculate cumulative portfolio and default capture
        cumulative_portfolio = [0,10,20,30,40,50,60,70,80,90,100]
        cumulative_defaults = list(df['decile_metrics']['cumulative_defaults_pct'].values)
        cumulative_defaults.insert(0,0)
        print('default.{}'.format(model), cumulative_defaults)

        
        fig3.add_trace(go.Scatter(
            x=cumulative_portfolio, y=cumulative_defaults,
            mode='lines', name=f'{model} Model',
            line=dict(color=colors.get(model, '#000000'), width=2),
            marker=dict(size=6)
        ))
    
    # Add diagonal line (random model)
    fig3.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode='lines', name='Random Model',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    fig3.update_layout(
        title='Default Capture Efficiency Curve - Portfolio vs Default Capture',
        xaxis_title='Cumulative % of Portfolio (by Risk Score)',
        yaxis_title='Cumulative % of Defaults Captured',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500,
        margin=dict(l=60, r=60, t=80, b=60),
      
    )
    
    fig3.update_xaxes(dtick=10, gridcolor='lightgray')
    fig3.update_yaxes(dtick=10, gridcolor='lightgray')
    
    # 4. LOSS CAPTURE EFFICIENCY CURVE (separate for better visibility)
    fig4 = go.Figure()
    
    for model, df in risk_results.items():
        # Calculate cumulative loss capture
        cumulative_portfolio = [0,10,20,30,40,50,60,70,80,90,100]
        cumulative_loss_capture = df['decile_metrics']['cumulative_expected_loss'] / df['decile_metrics']['cumulative_expected_loss'].iloc[-1] * 100
        cumulative_loss_capture = list(cumulative_loss_capture.values)
        cumulative_loss_capture.insert(0,0)
        print('CumLoss.{}'.format(model),  cumulative_loss_capture)
        print( 'Cum Loss',)

        fig4.add_trace(go.Scatter(
            x=cumulative_portfolio, y=cumulative_loss_capture,
            mode='lines', name=f'{model} Model',
            line=dict(color=colors.get(model, '#000000'), width=2),
            marker=dict(size=6)
        ))
    
    # Add diagonal line (random model)
    fig4.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode='lines', name='Random Model',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    fig4.update_layout(
        title='Loss Capture Efficiency Curve - Portfolio vs Loss Capture',
        xaxis_title='Cumulative % of Portfolio (by Risk Score)',
        yaxis_title='Cumulative % of Loss Captured',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    fig4.update_xaxes(dtick=10, gridcolor='lightgray')
    fig4.update_yaxes(dtick=10, gridcolor='lightgray')
    
    return fig1, fig2, fig3, fig4

def create_advanced_xgboost_analysis(risk_results):
    """
    Create advanced visualizations highlighting XGBoost advantages
    
    Parameters:
    - risk_results: Dictionary with risk analysis results by model
    
    Returns:
    - Three plotly figures with advanced XGBoost analysis
    """
    # Print the data values used in the visualizations
    print("\n===== ADVANCED XGBOOST ANALYSIS DATA VALUES =====\n")
    
    # Print efficiency curve data
    print("MODEL EFFICIENCY CURVE DATA")
    print("Percentage of portfolio vs. percentage of defaults captured")
    headers = ["Portfolio %"] + list(risk_results.keys())
    print(f"{headers[0]:<15} " + " ".join([f"{model:<15}" for model in headers[1:]]))
    print("-" * 80)
    
    # Get portfolio percentages (x-axis values)
    portfolio_pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for pct in portfolio_pcts:
        row = [f"{pct:<15}"] 
        for model, df in risk_results.items():
            # Find closest portfolio percentage
            closest_idx = (df['decile_metrics']['cumulative_exposure_pct'] - pct).abs().idxmin()
            
            # Calcular defaults acumulados si no existe la columna
            if 'cumulative_defaults' not in df['decile_metrics'].columns:
                # Ordenar por deciles (de mejor a peor)
                sorted_df = df['decile_metrics'].sort_values('decile')
                # Calcular defaults acumulados
                sorted_df['cumulative_defaults'] = sorted_df['defaults'].cumsum()
                default_capture = sorted_df.loc[closest_idx, 'cumulative_defaults'] / sorted_df['defaults'].sum() * 100
            else:
                default_capture = df['decile_metrics'].loc[closest_idx, 'cumulative_defaults'] / df['decile_metrics']['defaults'].sum() * 100
                
            row.append(f"{default_capture:<15.2f}")
        print(" ".join(row))
    
    # Print risk concentration data
    print("\nRISK CONCENTRATION DATA")
    print("Percentage of portfolio vs. percentage of expected loss")
    print(f"{headers[0]:<15} " + " ".join([f"{model:<15}" for model in headers[1:]]))
    print("-" * 80)
    
    for pct in portfolio_pcts:
        row = [f"{pct:<15}"] 
        for model, df in risk_results.items():
            # Find closest portfolio percentage
            closest_idx = (df['decile_metrics']['cumulative_exposure_pct'] - pct).abs().idxmin()
            
            # Calcular pérdidas acumuladas si no existe la columna
            if 'cumulative_expected_loss' not in df['decile_metrics'].columns:
                # Ordenar por deciles (de mejor a peor)
                sorted_df = df['decile_metrics'].sort_values('decile')
                # Calcular pérdidas acumuladas
                sorted_df['cumulative_expected_loss'] = sorted_df['expected_loss'].cumsum()
                loss_capture = sorted_df.loc[closest_idx, 'cumulative_expected_loss'] / sorted_df['expected_loss'].sum() * 100
            else:
                loss_capture = df['decile_metrics'].loc[closest_idx, 'cumulative_expected_loss'] / df['decile_metrics']['expected_loss'].sum() * 100
                
            row.append(f"{loss_capture:<15.2f}")
        print(" ".join(row))
    # Color palette for models
    colors = {
        'Baseline':  '#1560bd',   
        'Weighted': '#75caed',     
        'Undersampled':  '#8B7EC8', 
        'Xgboost': '#d62728'        
    }
    
    # 1. MODEL EFFICIENCY CURVE (Gains Chart / Lift Curve)
    fig_efficiency = go.Figure()
    
    for model, df in risk_results.items():
        # Calculate cumulative portfolio % and cumulative defaults %
        portfolio_pct = df['decile_metrics']['pct_of_loans'].cumsum()
        defaults_pct = df['decile_metrics']['cumulative_defaults_pct']
        
        fig_efficiency.add_trace(go.Scatter(
            x=portfolio_pct,
            y=defaults_pct,
            mode='lines+markers',
            name=f'{model} Model',
            line=dict(color=colors.get(model, '#000000'), width=2),
            marker=dict(size=6)
        ))
    
    # Add diagonal line (random model)
    fig_efficiency.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Random Model',
        line=dict(color='gray', dash='dash', width=1),
        showlegend=True
    ))
    
    fig_efficiency.update_layout(
        title='Model Efficiency Curve - Portfolio vs Default Capture',
        xaxis_title='Cumulative % of Portfolio (by Risk Score)',
        yaxis_title='Cumulative % of Defaults Captured',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500,
        margin=dict(l=60, r=60, t=80, b=60),
        annotations=[
            dict(
                x=0.7, y=0.3,
                xref='paper', yref='paper',
                text="Better models capture more<br>defaults with less portfolio<br>(area above diagonal)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                ax=20, ay=-30,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    fig_efficiency.update_xaxes(dtick=10, gridcolor='lightgray')
    fig_efficiency.update_yaxes(dtick=10, gridcolor='lightgray')
    
    # 2. RISK CONCENTRATION ANALYSIS
    fig_concentration = go.Figure()
    
    concentration_data = []
    for model, df in risk_results.items():
        # Calculate risk concentration in top deciles
        top_3_defaults = df['decile_metrics'][df['decile_metrics']['decile'].isin([8, 9, 10])]['pct_of_defaults'].sum()
        top_2_defaults = df['decile_metrics'][df['decile_metrics']['decile'].isin([9, 10])]['pct_of_defaults'].sum()
        top_1_defaults = df['decile_metrics'][df['decile_metrics']['decile'] == 10]['pct_of_defaults'].iloc[0]
        
        concentration_data.append({
            'Model': model,
            'Top_1_Decile': top_1_defaults,
            'Top_2_Deciles': top_2_defaults,
            'Top_3_Deciles': top_3_defaults
        })
    
    concentration_df = pd.DataFrame(concentration_data)
    
    # Create grouped bar chart
    x_pos = np.arange(len(concentration_df))
    width = 0.25
    
    fig_concentration.add_trace(go.Bar(
        x=[f"{row['Model']}" for _, row in concentration_df.iterrows()],
        y=concentration_df['Top_1_Decile'],
        name='Top 1 Decile (10% portfolio)',
        marker_color='#d62728',
        opacity=0.8
    ))
    
    fig_concentration.add_trace(go.Bar(
        x=[f"{row['Model']}" for _, row in concentration_df.iterrows()],
        y=concentration_df['Top_2_Deciles'],
        name='Top 2 Deciles (20% portfolio)',
        marker_color='#ff7f0e',
        opacity=0.8
    ))
    
    fig_concentration.add_trace(go.Bar(
        x=[f"{row['Model']}" for _, row in concentration_df.iterrows()],
        y=concentration_df['Top_3_Deciles'],
        name='Top 3 Deciles (30% portfolio)',
        marker_color='#2ca02c',
        opacity=0.8
    ))
    
    fig_concentration.update_layout(
        title='Risk Concentration Analysis - % of Defaults in Top Deciles',
        xaxis_title='Model',
        yaxis_title='% of Total Defaults Captured',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        barmode='group',
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_concentration.update_xaxes(gridcolor='lightgray')
    fig_concentration.update_yaxes(gridcolor='lightgray')
    
    # 3. ECONOMIC VALUE COMPARISON
    fig_economic = go.Figure()
    
    # Calculate economic value vs Baseline
    baseline_results = risk_results['Baseline']['decile_metrics']
    economic_data = []
    
    for model, df in risk_results.items():
        if model == 'Baseline':
            continue
            
        economic_value = []
        cumulative_value = 0
        
        for i, row in df['decile_metrics'].iterrows():
            baseline_row = baseline_results.iloc[i]
            
            # Economic benefit = reduction in expected loss vs baseline
            expected_loss_reduction = baseline_row['expected_loss'] - row['expected_loss']
            cumulative_value += expected_loss_reduction
            economic_value.append(cumulative_value)
        
        economic_data.append({
            'model': model,
            'deciles': list(range(1, 11)),
            'cumulative_value': economic_value
        })
    
    for data in economic_data:
        fig_economic.add_trace(go.Scatter(
            x=data['deciles'],
            y=data['cumulative_value'],
            mode='lines+markers',
            name=f'{data["model"]} vs Baseline',
            line=dict(color=colors.get(data['model'], '#000000'), width=2),
            marker=dict(size=6)
        ))
    
    # Add zero line
    fig_economic.add_hline(y=0, line_dash="dash", line_color="gray", 
                          annotation_text="Break-even vs Baseline")
    
    fig_economic.update_layout(
        title='Economic Value Analysis - Cumulative Expected Loss Savings vs Baseline',
        xaxis_title='Risk Decile (1=Lowest Risk, 10=Highest Risk)',
        yaxis_title='Cumulative Expected Loss Savings ($)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_economic.update_xaxes(dtick=1, gridcolor='lightgray')
    fig_economic.update_yaxes(gridcolor='lightgray')
    
    return fig_efficiency, fig_concentration, fig_economic

def analyze_cutoff_policy(risk_results, cutoff_decile=4, interest_rate=0.15, interest_rate_col=None, lgd=None, lgd_col=None, term_col=None, default_term=36):
    """
    Analyze economic impact of implementing a cutoff policy at specific decile
    
    Parameters:
    - risk_results: Dictionary with risk analysis results by model
    - cutoff_decile: Approve loans up to this decile (1-10)
    - interest_rate: Annual interest rate to calculate opportunity cost (used if interest_rate_col is None)
    - interest_rate_col: Column name with loan-specific interest rates (if None, uses the fixed interest_rate)
    - lgd: Loss Given Default value to use for loss calculations (if None, uses expected_loss directly)
    - term_col: Column name with loan-specific terms in months (if None, uses default_term)
    - default_term: Default loan term in months to use if term_col is not provided (default: 36)
    
    Returns:
    - Dictionary with cutoff analysis results by model
    """
    cutoff_analysis = {}
    
    for model, df in risk_results.items():
        # Add cutoff-specific metrics to the decile DataFrame
        df = df.copy()
        
        # Calculate metrics for each decile based on the cutoff
        # A decile is rejected if its value is > cutoff_decile
        df['is_rejected'] = df['decile'] > cutoff_decile
        
        # Approved portfolio (deciles 1 to cutoff_decile)
        approved_df = df[df['decile'] <= cutoff_decile]
        rejected_df = df[df['decile'] > cutoff_decile]
        
        # Calculate approved portfolio metrics
        approved_loans = approved_df['total_loans'].sum()
        approved_amount = approved_df['total_exposure'].sum()
        approved_expected_loss = approved_df['expected_loss'].sum()
        approved_defaults = approved_df['defaults'].sum()
        
        # Calculate rejected portfolio metrics
        rejected_loans = rejected_df['total_loans'].sum()
        rejected_amount = rejected_df['total_exposure'].sum()
        rejected_expected_loss = rejected_df['expected_loss'].sum()
        rejected_defaults = rejected_df['defaults'].sum()
        
        # Calculate total portfolio metrics
        total_loans = df['total_loans'].sum()
        total_amount = df['total_exposure'].sum()
        total_expected_loss = df['expected_loss'].sum()
        total_defaults = df['defaults'].sum()
        
        # Handle loan terms for annualization
        if term_col and term_col in rejected_df.columns:
            # Use loan-specific terms if available
            if not pd.api.types.is_numeric_dtype(rejected_df[term_col]):
                # Extract numeric part from strings like '36 months'
                rejected_terms = rejected_df[term_col].str.extract('(\d+)').astype(float).values.flatten()
            else:
                rejected_terms = rejected_df[term_col].values
        else:
            # Use default term if no loan-specific terms
            rejected_terms = np.full(len(rejected_df), default_term)
        
        # Calculate interest revenue with annualization
        if interest_rate_col is not None and interest_rate_col in rejected_df.columns:
            # Use loan-specific interest rates if available
            if isinstance(rejected_df[interest_rate_col].iloc[0], str) and '%' in str(rejected_df[interest_rate_col].iloc[0]):
                # Convert percentage strings to float (e.g., '5.2%' to 0.052)
                interest_rates = rejected_df[interest_rate_col].str.rstrip('%').astype(float) / 100
            else:
                interest_rates = rejected_df[interest_rate_col]
                
            # Calculate weighted interest revenue (annualized)
            rejected_interest = ((rejected_df['total_exposure'] * interest_rates) * (12 / rejected_terms.mean())).sum() if len(rejected_df) > 0 else 0
        else:
            # Use fixed interest rate if no loan-specific rates (annualized)
            rejected_interest = (rejected_amount * interest_rate * (12 / rejected_terms.mean())) if len(rejected_df) > 0 else 0
        
        # Apply LGD to expected loss if provided
        if lgd_col is not None and lgd_col in rejected_df.columns:
            # Use loan-specific LGD values if available
            if isinstance(rejected_df[lgd_col].iloc[0], str) and '%' in str(rejected_df[lgd_col].iloc[0]):
                # Convert percentage strings to float (e.g., '60%' to 0.6)
                lgd_values = rejected_df[lgd_col].str.rstrip('%').astype(float) / 100
            else:
                lgd_values = rejected_df[lgd_col]
                
            # Apply weighted LGD to expected loss
            rejected_expected_loss = rejected_expected_loss * lgd_values.mean() if len(rejected_df) > 0 else 0
        elif lgd is not None:
            # Use fixed LGD value if no loan-specific LGD values
            rejected_expected_loss = rejected_expected_loss * lgd
            
        # Annualize expected loss based on loan terms
        if term_col and term_col in rejected_df.columns and term_col is not None and len(rejected_df) > 0:
            # Annualize the expected loss
            rejected_expected_loss = rejected_expected_loss * (12 / rejected_terms.mean())
        
        # Calculate savings and opportunity cost
        loss_savings = rejected_expected_loss  # Money saved by not approving risky loans
        revenue_opportunity_cost = rejected_interest  # Income forgone by not approving loans
        
        # Calculate net savings
        net_savings = loss_savings - revenue_opportunity_cost
        
        # Calculate rates
        approved_default_rate = approved_defaults / approved_loans if approved_loans > 0 else 0
        rejected_default_rate = rejected_defaults / rejected_loans if rejected_loans > 0 else 0
        total_default_rate = total_defaults / total_loans if total_loans > 0 else 0
        
        approved_loss_rate = approved_expected_loss / approved_amount if approved_amount > 0 else 0
        rejected_loss_rate = rejected_expected_loss / rejected_amount if rejected_amount > 0 else 0
        total_loss_rate = total_expected_loss / total_amount if total_amount > 0 else 0
        
        # Store effective interest rate used in calculations
        if interest_rate_col and interest_rate_col in rejected_df.columns:
            # Calculate weighted average interest rate for rejected loans
            if len(rejected_df) > 0:
                if isinstance(rejected_df[interest_rate_col].iloc[0], str):
                    # If interest rates are strings, convert to float
                    interest_rates = rejected_df[interest_rate_col].str.rstrip('%').astype(float) / 100
                else:
                    interest_rates = rejected_df[interest_rate_col]
                effective_interest_rate = interest_rates.mean()
            else:
                effective_interest_rate = interest_rate
        else:
            effective_interest_rate = interest_rate

        # Store results
        cutoff_analysis[model] = {
            'cutoff_decile': cutoff_decile,
            'effective_interest_rate': effective_interest_rate,
            
            # Approved portfolio
            'approved_loans': approved_loans,
            'approved_amount': approved_amount,
            'approved_expected_loss': approved_expected_loss,
            'approved_default_rate': approved_default_rate,  # Store as decimal (0-1)
            'approved_loss_rate': approved_loss_rate,  # Store as decimal (0-1)
            'approved_pct_loans': (approved_loans / total_loans),  # Store as decimal (0-1)
            'approved_pct_amount': (approved_amount / total_amount),  # Store as decimal (0-1)
            'approved_count': approved_loans,  # Explicitly store count for visualization
            
            # Rejected portfolio
            'rejected_loans': rejected_loans,
            'rejected_amount': rejected_amount,
            'rejected_expected_loss': rejected_expected_loss,
            'rejected_default_rate': rejected_default_rate,  # Store as decimal (0-1)
            'rejected_loss_rate': rejected_loss_rate,  # Store as decimal (0-1)
            'rejected_pct_loans': (rejected_loans / total_loans),  # Store as decimal (0-1)
            'rejected_pct_amount': (rejected_amount / total_amount),  # Store as decimal (0-1)
            'rejected_count': rejected_loans,  # Explicitly store count for visualization
            
            # Economic impact
            'loss_savings': loss_savings,
            'revenue_opportunity_cost': revenue_opportunity_cost,
            'net_savings': net_savings,
            'savings_rate': (loss_savings / total_expected_loss) if total_expected_loss > 0 else 0,  # Decimal (0-1)
            'roi_cutoff_policy': (net_savings / total_amount) if total_amount > 0 else 0,  # Decimal (0-1)
            
            # Comparison vs total portfolio
            'loss_reduction_pct': ((total_expected_loss - approved_expected_loss) / total_expected_loss) * 100,
            'default_rate_improvement': (total_default_rate - approved_default_rate) * 100,
            'loss_rate_improvement': (total_loss_rate - approved_loss_rate) * 100
        }
    
    # Print the data values used in the analysis
    print("\n===== CUTOFF POLICY ANALYSIS DATA VALUES =====\n")
    print(f"Cutoff Decile: {cutoff_decile} (Approve up to decile {cutoff_decile})")
    
    # Print economic impact metrics for each model
    print("\nECONOMIC IMPACT METRICS:")
    headers = ["Model", "Loss Savings ($)", "Opportunity Cost ($)", "Net Savings ($)", "Effective Int. Rate (%)"]
    print(f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20} {headers[4]:<20}")
    print("-" * 95)
    
    for model, data in cutoff_analysis.items():
        print(f"{model:<15} {data['loss_savings']:<20.2f} {data['revenue_opportunity_cost']:<20.2f} {data['net_savings']:<20.2f} {data['effective_interest_rate']*100:<20.2f}")
    
    # Print portfolio metrics
    print("\nPORTFOLIO METRICS:")
    headers = ["Model", "Approved Loans (%)", "Approved Amount (%)", "Approved Default Rate (%)", "Loss Reduction (%)"]
    print(f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20} {headers[4]:<20}")
    print("-" * 95)
    
    for model, data in cutoff_analysis.items():
        print(f"{model:<15} {data['approved_pct_loans']:<20.2f} {data['approved_pct_amount']:<20.2f} {data['approved_default_rate']:<20.2f} {data['loss_reduction_pct']:<20.2f}")
    
    return cutoff_analysis

def create_cutoff_analysis_visualization(cutoff_analysis):
    """
    Create visualization for cutoff policy analysis
    
    Parameters:
    - cutoff_analysis: Dictionary with cutoff policy analysis results by model
    
    Returns:
    - Plotly figure with cutoff policy analysis visualization
    """
    # Print the data values used in the visualization
    print("\n===== CUTOFF POLICY ANALYSIS VISUALIZATION DATA VALUES =====\n")
    
    # Print portfolio composition data
    print("PORTFOLIO COMPOSITION (APPROVED/REJECTED LOANS)")
    print(f"{'Model':<15} {'Approved %':<15} {'Rejected %':<15} {'Approved #':<15} {'Rejected #':<15}")
    print("-" * 75)
    
    for model, data in cutoff_analysis.items():
        # Obtener porcentaje de préstamos aprobados (convertir a porcentaje para visualización)
        if 'approved_pct_loans' in data:
            approved_pct = data['approved_pct_loans'] * 100  # Convertir de decimal a porcentaje
        elif 'approved_pct' in data:
            approved_pct = data['approved_pct'] * 100  # Convertir de decimal a porcentaje
        else:
            approved_pct = 0
            print(f"Warning: No approved percentage found for {model}")
            
        rejected_pct = 100 - approved_pct  # El complemento para llegar al 100%
        
        # Obtener conteo de préstamos aprobados
        if 'approved_count' in data:
            approved_count = data['approved_count']
        elif 'approved_loans' in data:
            approved_count = data['approved_loans']
        else:
            approved_count = 0
            print(f"Warning: No approved count found for {model}")
            
        # Obtener conteo de préstamos rechazados
        if 'rejected_count' in data:
            rejected_count = data['rejected_count']
        elif 'rejected_loans' in data:
            rejected_count = data['rejected_loans']
        else:
            rejected_count = 0
            print(f"Warning: No rejected count found for {model}")
            
        print(f"{model:<15} {approved_pct:<15.2f} {rejected_pct:<15.2f} {approved_count:<15} {rejected_count:<15}")
    
    # Print economic impact data
    print("\nECONOMIC IMPACT DATA")
    print(f"{'Model':<15} {'Loss Savings ($)':<20} {'Opportunity Cost ($)':<20} {'Net Savings ($)':<20}")
    print("-" * 75)
    
    for model, data in cutoff_analysis.items():
        # Verificar y obtener loss_savings
        if 'loss_savings' in data:
            loss_savings = data['loss_savings']
        else:
            loss_savings = 0
            print(f"Warning: No loss savings found for {model}")
        
        # Verificar y obtener opportunity_cost (puede estar como revenue_opportunity_cost)
        if 'opportunity_cost' in data:
            opportunity_cost = data['opportunity_cost']
        elif 'revenue_opportunity_cost' in data:
            opportunity_cost = data['revenue_opportunity_cost']
        else:
            opportunity_cost = 0
            print(f"Warning: No opportunity cost found for {model}")
        
        # Verificar y obtener net_savings
        if 'net_savings' in data:
            net_savings = data['net_savings']
        else:
            net_savings = 0
            print(f"Warning: No net savings found for {model}")
            
        print(f"{model:<15} {loss_savings:<20.2f} {opportunity_cost:<20.2f} {net_savings:<20.2f}")
    
    # Print default rate comparison
    print("\nDEFAULT RATE COMPARISON (%)")
    print(f"{'Model':<15} {'Approved DR':<15} {'Rejected DR':<15} {'Total DR':<15} {'Effective IR (%)':<15}")
    print("-" * 75)
    
    for model, data in cutoff_analysis.items():
        # Verificar y obtener approved_default_rate (ya está en formato decimal 0-1)
        if 'approved_default_rate' in data:
            # Convertir a porcentaje para visualización
            approved_dr = data['approved_default_rate'] * 100
        else:
            approved_dr = 0
            print(f"Warning: No approved default rate found for {model}")
        
        # Verificar y obtener rejected_default_rate (ya está en formato decimal 0-1)
        if 'rejected_default_rate' in data:
            # Convertir a porcentaje para visualización
            rejected_dr = data['rejected_default_rate'] * 100
        else:
            rejected_dr = 0
            print(f"Warning: No rejected default rate found for {model}")
        
        # Verificar y obtener total_default_rate (ya está en formato decimal 0-1)
        if 'total_default_rate' in data:
            # Convertir a porcentaje para visualización
            total_dr = data['total_default_rate'] * 100
        else:
            # Intentar calcular si tenemos los datos necesarios
            if 'approved_loans' in data and 'rejected_loans' in data and 'approved_default_rate' in data and 'rejected_default_rate' in data:
                total_loans = data['approved_loans'] + data['rejected_loans']
                total_defaults = (data['approved_default_rate'] * data['approved_loans']) + (data['rejected_default_rate'] * data['rejected_loans'])
                total_dr = (total_defaults / total_loans) * 100 if total_loans > 0 else 0
            else:
                total_dr = 0
                print(f"Warning: No total default rate found for {model} and couldn't calculate it")
        
        # Verificar y obtener effective_interest_rate (ya está en formato decimal 0-1)
        if 'effective_interest_rate' in data:
            # Convertir a porcentaje para visualización
            effective_ir = data['effective_interest_rate'] * 100
        elif 'interest_rate' in data:
            # Convertir a porcentaje para visualización
            effective_ir = data['interest_rate'] * 100
        else:
            effective_ir = 0
            print(f"Warning: No effective interest rate found for {model}")
            
        print(f"{model:<15} {approved_dr:<15.2f} {rejected_dr:<15.2f} {total_dr:<15.2f} {effective_ir:<15.2f}")
    
    colors = {
        'Baseline':  '#1560bd',   
        'Weighted': '#75caed',     
        'Undersampled':  '#8B7EC8', 
        'Xgboost': '#d62728'        
    }
    
    # Define consistent colors for different metrics
    approved_color =   '#c9ceb7'
    rejected_color = '#f3c6c0'  
    savings_color = '#e78c80'   
    cost_color =   '#9da77e'    
    net_color = '#cf7e73'    

    # Create subplots (2x2)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Composition (%)', 'Economic Impact ($M)', 
                       'Loss Reduction (%)', 'Default Rate Comparison (%)'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    models = list(cutoff_analysis.keys())
    
    # 1. Portfolio Composition (top-left)
    approved_pct = [cutoff_analysis[model]['approved_pct_amount'] for model in models]
    rejected_pct = [cutoff_analysis[model]['rejected_pct_amount'] for model in models]
    
    fig.add_trace(go.Bar(
        x=models, y=approved_pct, name='Approved (%)',
        marker_color=approved_color
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=models, y=rejected_pct, name='Rejected (%)',
        marker_color=rejected_color
    ), row=1, col=1)
    
    # 2. Economic Impact in Millions (top-right)
    loss_savings_m = [cutoff_analysis[model]['loss_savings'] / 1_000_000 for model in models]
    opportunity_cost_m = [cutoff_analysis[model]['revenue_opportunity_cost'] / 1_000_000 for model in models]
    net_savings_m = [cutoff_analysis[model]['net_savings'] / 1_000_000 for model in models]
    
    # Create a separate subplot for Loss Savings due to scale difference
    fig.add_trace(go.Bar(
        x=models, y=[-x for x in opportunity_cost_m], name='Opportunity Cost ($M)',
        marker_color=cost_color
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=models, y=net_savings_m, name='Net Savings ($M)',
        marker_color=net_color, opacity=0.8
    ), row=1, col=2)
    
    # 3. Loss Reduction (bottom-left)
    loss_reduction = [cutoff_analysis[model]['loss_reduction_pct'] for model in models]
    
    fig.add_trace(go.Bar(
        x=models, y=loss_reduction, name='Loss Reduction (%)',
        #marker_color=[colors.get(model, '#000000') for model in models]
        marker_color='lightgray'
    ), row=2, col=1)
    
    # 4. Default Rate Comparison (bottom-right)
    approved_default_rate = [cutoff_analysis[model]['approved_default_rate'] for model in models]
    rejected_default_rate = [cutoff_analysis[model]['rejected_default_rate'] for model in models]
    
    fig.add_trace(go.Bar(
        x=models, y=approved_default_rate, name='Approved Default Rate (%)',
        marker_color=approved_color
    ), row=2, col=2)
    
    fig.add_trace(go.Bar(
        x=models, y=rejected_default_rate, name='Rejected Default Rate (%)',
        marker_color=rejected_color
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'Cutoff Policy Analysis - Approve up to Decile {cutoff_analysis[models[0]]["cutoff_decile"]}',
        template='plotly_white',
        font=dict(family="Arial", size=11),
        height=700,
        showlegend=True,
        barmode='group'
    )
    
    # Update axes
    fig.update_yaxes(title_text="Portfolio %", row=1, col=1)
    fig.update_yaxes(title_text="Amount ($M)", row=1, col=2)
    fig.update_yaxes(title_text="Loss Reduction (%)", row=2, col=1)
    fig.update_yaxes(title_text="Default Rate (%)", row=2, col=2)
    
    return fig

def execute_phase1_risk_validation(df, 
                                  reference_score='baseline_score',
                                  score_columns=['baseline_score', 'weighted_score', 'undersampled_score', 'xgboost_score'],
                                  target_col='target',
                                  real_loan_amount_col='real_loan_amnt',
                                  set_col='Set',
                                  cutoff_decile=4,
                                  interest_rate=0.15,
                                  interest_rate_col=None,
                                  lgd_col=None,
                                  lgd=None,
                                  term_col=None,
                                  default_term=36):
    """
    Execute complete Phase 1: Risk Validation Analysis
    
    Parameters:
    - df: DataFrame with model scores and target
    - reference_score: Column name of reference score (usually baseline model)
    - score_columns: List of column names with model scores
    - target_col: Column name with target variable
    - real_loan_amount_col: Column name with actual loan amount
    - set_col: Column name indicating data split (Test, Val)
    - cutoff_decile: Decile cutoff for policy analysis (default: 4)
    - interest_rate: Default interest rate if loan-specific rates not available (default: 0.15)
    - interest_rate_col: Column name with loan-specific interest rates (default: None)
    
    Returns:
    - Dictionary with all analysis results
    """
    print("🎯 EXECUTING PHASE 1: RISK VALIDATION ANALYSIS")
    print("=" * 70)
    
    # Step 1: Create simple qcut deciles
    df_with_deciles, decile_info = create_simple_qcut_deciles(
        df, score_columns, set_col, target_col
    )
    
    # Step 2: Calculate comprehensive risk metrics
    risk_results = calculate_comprehensive_risk_metrics(
        df_with_deciles, target_col, real_loan_amount_col, set_col
    )
    
    # Step 3: Create summary tables
    default_pivot, loss_pivot, exp_loss_pivot, portfolio_df = create_risk_validation_summary_table(risk_results)
    
    # Step 4: Create basic visualizations
    fig1, fig2, fig3, fig4 = create_risk_validation_visualizations(risk_results)
    
    # Step 5: Create advanced XGBoost analysis
    fig_efficiency, fig_concentration, fig_economic = create_advanced_xgboost_analysis(risk_results)
    
    # Step 6: Analyze cutoff policy
    #cutoff_analysis = analyze_cutoff_policy(
       # risk_results, 
       # cutoff_decile=cutoff_decile, 
       # interest_rate=interest_rate,
       # interest_rate_col=interest_rate_col,
       # lgd=lgd,
       # lgd_col = lgd_col,
       # term_col=term_col,
       # default_term=default_term
    #)
    #fig_cutoff = create_cutoff_analysis_visualization(cutoff_analysis)
    
    print("\n" + "=" * 70)
    print("✅ PHASE 1: RISK VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("- Simple qcut deciles")
    print("- Comprehensive risk metrics by decile")
    print("- Summary tables (default rate, loss rate, expected loss)")
    print("- 4 consolidated interactive visualizations")
    print("- 3 advanced XGBoost analysis visualizations")
    print(f"- Cutoff policy analysis (decile {cutoff_decile})")
    print("\nNext: Execute Phase 2 (Temporal Performance & Stability)")
    
    return {
        'df_with_deciles': df_with_deciles,
        'decile_info': decile_info,
        'risk_results': risk_results,
        'summary_tables': {
            'default_pivot': default_pivot,
            'loss_pivot': loss_pivot,
            'exp_loss_pivot': exp_loss_pivot,
            'portfolio_df': portfolio_df
        },
        'visualizations': {
            'dashboard': fig1,
            'lift_heatmap': fig2,
            'default_capture_efficiency': fig3,
            'loss_capture_efficiency': fig4
        },
        'advanced_analysis': {
            'efficiency_curve': fig_efficiency,
            'risk_concentration': fig_concentration,
            'economic_value': fig_economic
        },
       # 'cutoff_policy': {
       #     'analysis': cutoff_analysis,
        #    'visualization': fig_cutoff
        #}
    }

def visualize_economic_value(economic_value_data, baseline_model='Baseline'):
    """
    Visualize economic value analysis results
    
    Parameters:
    - economic_value_data: Dictionary with economic value data by model
    - baseline_model: Name of the baseline model
    
    Returns:
    - Plotly figure
    """
    # Print the data values used in the chart
    print("\n===== ECONOMIC VALUE ANALYSIS DATA VALUES =====\n")
    print("CUMULATIVE EXPECTED LOSS SAVINGS VS BASELINE ($):")
    print("Decile    ", end="")
    
    # Get models excluding baseline
    models = [model for model in economic_value_data.keys() if model != baseline_model]
    
    # Print header with model names
    for model in models:
        print(f"{model:<15}", end="")
    print()
    print("-" * (8 + 15 * len(models)))
    
    # Print values for each decile
    for decile in range(1, 11):
        print(f"{decile:<9}", end="")
        
        for model in models:
            if decile <= len(economic_value_data[model]):
                val = economic_value_data[model][decile-1]
                print(f"${val:<14,.2f}", end="")
            else:
                print("N/A            ", end="")
        print()
    
    print("\n" + "=" * 50 + "\n")
    
    # Create figure
    fig = go.Figure()
    
    # Colors for models
    colors = {
        'Baseline': '#1560bd',   
        'Weighted': '#75caed',     
        'Undersampled':  '#8B7EC8', 
        'Xgboost': '#d62728'        
    }
    
    # Add traces for each model
    for model in economic_value_data.keys():
        if model != baseline_model:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(economic_value_data[model]) + 1)),
                y=economic_value_data[model],
                mode='lines+markers',
                name=f'{model} vs {baseline_model}',
                line=dict(color=colors.get(model, '#000000'), width=2),
                marker=dict(size=8)
            ))
    
    # Add break-even line
    fig.add_trace(go.Scatter(
        x=[1, 10],
        y=[0, 0],
        mode='lines',
        name='Break-even vs Baseline',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Economic Value Analysis - Cumulative Expected Loss Savings vs Baseline',
        xaxis_title='Risk Decile (1=Lowest Risk, 10=Highest Risk)',
        yaxis_title='Cumulative Expected Loss Savings ($)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig.update_xaxes(dtick=1, gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    return fig


def analyze_optimal_cutoff(risk_results, interest_rate=0.15, interest_rate_col=None, lgd=None, term_col=None, default_term=36):
    """
    Analyze optimal cutoff policy for each model by evaluating all possible decile cutoffs
    
    Parameters:
    - risk_results: Dictionary with risk analysis results by model
    - interest_rate: Annual interest rate to calculate opportunity cost (used if interest_rate_col is None)
    - interest_rate_col: Column name with loan-specific interest rates (if None, uses the fixed interest_rate)
    - lgd: Loss Given Default value to use for loss calculations (if None, uses expected_loss directly)
    - term_col: Column name with loan-specific terms in months (if None, uses default_term)
    - default_term: Default loan term in months to use if term_col is not provided (default: 36)
    
    Returns:
    - Dictionary with optimal cutoff analysis results by model
    """
    optimal_cutoff_results = {}
    
    # For each model, analyze all possible cutoff deciles (1-10)
    for model, df in risk_results.items():
        model_results = []
        
        # Analyze each possible cutoff decile
        for cutoff_decile in range(1, 11):
            # Approved portfolio (deciles 1 to cutoff_decile)
            approved_df = df[df['decile'] <= cutoff_decile].copy()
            rejected_df = df[df['decile'] > cutoff_decile].copy()
            
            # Calculate approved portfolio metrics
            approved_loans = approved_df['total_loans'].sum()
            approved_amount = approved_df['total_exposure'].sum()
            approved_defaults = approved_df['defaults'].sum()
            
            # Calculate rejected portfolio metrics
            rejected_loans = rejected_df['total_loans'].sum()
            rejected_amount = rejected_df['total_exposure'].sum()
            rejected_defaults = rejected_df['defaults'].sum()
            
            # Calculate total portfolio metrics
            total_loans = df['total_loans'].sum()
            total_amount = df['total_exposure'].sum()
            total_defaults = df['defaults'].sum()
            
            # Get average term for annualization
            if term_col and term_col in rejected_df.columns:
                # Use weighted average term for rejected loans
                avg_term_rejected = (rejected_df[term_col] * rejected_df['total_exposure']).sum() / rejected_amount if rejected_amount > 0 else default_term
                avg_term_approved = (approved_df[term_col] * approved_df['total_exposure']).sum() / approved_amount if approved_amount > 0 else default_term
            else:
                # Use default term
                avg_term_rejected = default_term
                avg_term_approved = default_term
            
            # Annualization factor (convert term in months to years)
            annualization_factor_rejected = 12 / avg_term_rejected if avg_term_rejected > 0 else 1
            annualization_factor_approved = 12 / avg_term_approved if avg_term_approved > 0 else 1
            
            # Calculate expected loss (annualized)
            if lgd is not None:
                # Calculate expected loss using LGD
                approved_expected_loss = approved_defaults * lgd * annualization_factor_approved
                rejected_expected_loss = rejected_defaults * lgd * annualization_factor_rejected
            else:
                # Use the expected_loss from the DataFrame (ensure it's annualized)
                approved_expected_loss = approved_df['expected_loss'].sum() * annualization_factor_approved
                rejected_expected_loss = rejected_df['expected_loss'].sum() * annualization_factor_rejected
            
            total_expected_loss = approved_expected_loss + rejected_expected_loss
            
            # Calculate savings and opportunity cost
            loss_savings = rejected_expected_loss  # Money saved by not approving risky loans (annualized)
            
            # Calculate opportunity cost based on loan-specific interest rates if available
            if interest_rate_col and interest_rate_col in rejected_df.columns:
                # Calculate weighted average interest rate for rejected loans
                weighted_avg_rate = (rejected_df[interest_rate_col] * rejected_df['total_exposure']).sum() / rejected_amount if rejected_amount > 0 else interest_rate
                revenue_opportunity_cost = rejected_amount * weighted_avg_rate
                
                # Store the actual weighted average rate used
                effective_interest_rate = weighted_avg_rate
            else:
                # Use fixed interest rate
                revenue_opportunity_cost = rejected_amount * interest_rate
                effective_interest_rate = interest_rate
                
            net_savings = loss_savings - revenue_opportunity_cost
            
            # Calculate rates
            approved_default_rate = approved_defaults / approved_loans if approved_loans > 0 else 0
            rejected_default_rate = rejected_defaults / rejected_loans if rejected_loans > 0 else 0
            total_default_rate = total_defaults / total_loans if total_loans > 0 else 0
            
            approved_loss_rate = approved_expected_loss / approved_amount if approved_amount > 0 else 0
            rejected_loss_rate = rejected_expected_loss / rejected_amount if rejected_amount > 0 else 0
            total_loss_rate = total_expected_loss / total_amount if total_amount > 0 else 0
            
            # Store results for this cutoff decile
            cutoff_result = {
                'cutoff_decile': cutoff_decile,
                'interest_rate': effective_interest_rate,
                'avg_term_rejected': avg_term_rejected,
                'avg_term_approved': avg_term_approved,
                
                # Approved portfolio
                'approved_loans': approved_loans,
                'approved_amount': approved_amount,
                'approved_expected_loss': approved_expected_loss,
                'approved_default_rate': approved_default_rate,
                'approved_loss_rate': approved_loss_rate,
                'approved_pct_loans': (approved_loans / total_loans) if total_loans > 0 else 0,
                'approved_pct_amount': (approved_amount / total_amount) if total_amount > 0 else 0,
                
                # Rejected portfolio
                'rejected_loans': rejected_loans,
                'rejected_amount': rejected_amount,
                'rejected_expected_loss': rejected_expected_loss,
                'rejected_default_rate': rejected_default_rate,
                'rejected_loss_rate': rejected_loss_rate,
                'rejected_pct_loans': (rejected_loans / total_loans) if total_loans > 0 else 0,
                'rejected_pct_amount': (rejected_amount / total_amount) if total_amount > 0 else 0,
                
                # Economic impact
                'loss_savings': loss_savings,
                'revenue_opportunity_cost': revenue_opportunity_cost,
                'net_savings': net_savings,
                'savings_rate': (loss_savings / total_expected_loss) if total_expected_loss > 0 else 0,
                'roi_cutoff_policy': (net_savings / total_amount) if total_amount > 0 else 0,
            }
            
            model_results.append(cutoff_result)
        
        # Find the optimal cutoff (maximizing net_savings)
        optimal_cutoff_results[model] = model_results
    
    return optimal_cutoff_results


def compare_cutoff_policies(cutoff_results_dict, model_colors=None):
    """
    Create comparison visualization for cutoff policies across models
    
    Parameters:
    - cutoff_results_dict: Dictionary with cutoff policy analysis results by model
    - model_colors: Dictionary with colors for each model
    
    Returns:
    - Plotly figure with comparison visualization
    """
    if model_colors is None:
        model_colors = {
            'Baseline': '#1560bd',
            'Weighted': '#75caed',
            'Undersampled': '#8B7EC8',
            'XGBoost': '#d62728'
        }
    
    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Approval Rate by Loan Count',
            'Default Rate in Approved Portfolio',
            'Net Savings from Cutoff Policy',
            'ROI of Cutoff Policy'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add traces for each model
    for model, results in cutoff_results_dict.items():
        color = model_colors.get(model, '#000000')
        
        # Approval Rate (%)
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[results.get('approved_pct_loans', 0) * 100],
                name=model,
                marker_color=color,
                text=[f"{results.get('approved_pct_loans', 0) * 100:.1f}%"],
                textposition='auto',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Default Rate (%)
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[results.get('approved_default_rate', 0) * 100],
                name=model,
                marker_color=color,
                text=[f"{results.get('approved_default_rate', 0) * 100:.1f}%"],
                textposition='auto',
                hovertemplate='%{y:.1f}%<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Net Savings ($)
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[results.get('net_savings', 0)],
                name=model,
                marker_color=color,
                text=[f"${results.get('net_savings', 0):,.0f}"],
                textposition='auto',
                hovertemplate='$%{y:,.0f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # ROI (%)
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[results.get('roi_cutoff_policy', 0) * 100],
                name=model,
                marker_color=color,
                text=[f"{results.get('roi_cutoff_policy', 0) * 100:.2f}%"],
                textposition='auto',
                hovertemplate='%{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'Cutoff Policy Analysis (Approve Deciles 1-{cutoff_results_dict[list(cutoff_results_dict.keys())[0]].get("cutoff_decile", "?")})',
        template='plotly_white',
        height=600,
        width=900,
        font=dict(family="Arial", size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Approval Rate (%)', row=1, col=1)
    fig.update_yaxes(title_text='Default Rate (%)', row=1, col=2)
    fig.update_yaxes(title_text='Net Savings ($)', row=2, col=1)
    fig.update_yaxes(title_text='ROI (%)', row=2, col=2)
    
    # Print the data values used in the chart
    print("\n===== CUTOFF POLICY ANALYSIS DATA VALUES =====\n")
    
    print("APPROVAL RATE (% of loans):")
    for model, results in cutoff_results_dict.items():
        print(f"{model}: {results.get('approved_pct_loans', 0) * 100:.1f}%")
    
    print("\nDEFAULT RATE IN APPROVED PORTFOLIO:")
    for model, results in cutoff_results_dict.items():
        print(f"{model}: {results.get('approved_default_rate', 0) * 100:.1f}%")
    
    print("\nNET SAVINGS FROM CUTOFF POLICY ($):")
    for model, results in cutoff_results_dict.items():
        print(f"{model}: ${results.get('net_savings', 0):,.2f}")
    
    print("\nROI OF CUTOFF POLICY (%):")
    for model, results in cutoff_results_dict.items():
        print(f"{model}: {results.get('roi_cutoff_policy', 0) * 100:.2f}%")
    
    print("\n" + "=" * 50 + "\n")
    
    return fig


def compare_optimal_cutoffs(optimal_cutoff_results, model_colors=None):
    """
    Create comparison visualizations for optimal cutoff analysis
    
    Parameters:
    - optimal_cutoff_results: Dictionary with optimal cutoff analysis results by model
    - model_colors: Dictionary with colors for each model
    
    Returns:
    - Dictionary with Plotly figures for different metrics
    """
    if model_colors is None:
        model_colors = {
            'Baseline': '#1560bd',
            'Weighted': '#75caed',
            'Undersampled': '#8B7EC8',
            'XGBoost': '#d62728'
        }
    
    # Create figures for different metrics
    figures = {}
    
    # 1. Net Savings by Cutoff Decile
    fig_net_savings = go.Figure()
    
    for model, results in optimal_cutoff_results.items():
        color = model_colors.get(model, '#000000')
        
        # Extract data
        deciles = [r['cutoff_decile'] for r in results]
        net_savings = [r['net_savings'] for r in results]
        
        # Add trace
        fig_net_savings.add_trace(
            go.Scatter(
                x=deciles,
                y=net_savings,
                mode='lines+markers',
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            )
        )
    
    # Update layout
    fig_net_savings.update_layout(
        title='Net Savings by Cutoff Decile',
        xaxis_title='Cutoff Decile (Approve up to this decile)',
        yaxis_title='Net Savings ($)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_net_savings.update_xaxes(dtick=1, gridcolor='lightgray')
    fig_net_savings.update_yaxes(gridcolor='lightgray')
    
    figures['net_savings'] = fig_net_savings
    
    # 2. ROI by Cutoff Decile
    fig_roi = go.Figure()
    
    for model, results in optimal_cutoff_results.items():
        color = model_colors.get(model, '#000000')
        
        # Extract data
        deciles = [r['cutoff_decile'] for r in results]
        roi = [r['roi_cutoff_policy'] * 100 for r in results]  # Convert to percentage
        
        # Add trace
        fig_roi.add_trace(
            go.Scatter(
                x=deciles,
                y=roi,
                mode='lines+markers',
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            )
        )
    
    # Update layout
    fig_roi.update_layout(
        title='ROI by Cutoff Decile',
        xaxis_title='Cutoff Decile (Approve up to this decile)',
        yaxis_title='ROI (%)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_roi.update_xaxes(dtick=1, gridcolor='lightgray')
    fig_roi.update_yaxes(gridcolor='lightgray')
    
    figures['roi'] = fig_roi
    
    # 3. Approval Rate by Cutoff Decile
    fig_approval = go.Figure()
    
    for model, results in optimal_cutoff_results.items():
        color = model_colors.get(model, '#000000')
        
        # Extract data
        deciles = [r['cutoff_decile'] for r in results]
        approval_rates = [r['approved_pct_loans'] * 100 for r in results]  # Convert to percentage
        
        # Add trace
        fig_approval.add_trace(
            go.Scatter(
                x=deciles,
                y=approval_rates,
                mode='lines+markers',
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            )
        )
    
    # Update layout
    fig_approval.update_layout(
        title='Approval Rate by Cutoff Decile',
        xaxis_title='Cutoff Decile (Approve up to this decile)',
        yaxis_title='Approval Rate (% of loans)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_approval.update_xaxes(dtick=1, gridcolor='lightgray')
    fig_approval.update_yaxes(gridcolor='lightgray')
    
    figures['approval_rate'] = fig_approval
    
    # 4. Default Rate by Cutoff Decile
    fig_default = go.Figure()
    
    for model, results in optimal_cutoff_results.items():
        color = model_colors.get(model, '#000000')
        
        # Extract data
        deciles = [r['cutoff_decile'] for r in results]
        default_rates = [r['approved_default_rate'] * 100 for r in results]  # Convert to percentage
        
        # Add trace
        fig_default.add_trace(
            go.Scatter(
                x=deciles,
                y=default_rates,
                mode='lines+markers',
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            )
        )
    
    # Update layout
    fig_default.update_layout(
        title='Default Rate in Approved Portfolio by Cutoff Decile',
        xaxis_title='Cutoff Decile (Approve up to this decile)',
        yaxis_title='Default Rate (%)',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig_default.update_xaxes(dtick=1, gridcolor='lightgray')
    fig_default.update_yaxes(gridcolor='lightgray')
    
    figures['default_rate'] = fig_default
    
    # Print the data values used in the charts
    print("\n===== OPTIMAL CUTOFF ANALYSIS DATA VALUES =====\n")
    
    # Print Net Savings data
    print("NET SAVINGS BY CUTOFF DECILE ($):")
    print("Decile    ", end="")
    
    for model in optimal_cutoff_results.keys():
        print(f"{model:<15}", end="")
    print()
    print("-" * (8 + 15 * len(optimal_cutoff_results)))
    
    for decile in range(1, 11):
        print(f"{decile:<9}", end="")
        
        for model, results in optimal_cutoff_results.items():
            if decile <= len(results):
                val = results[decile-1]['net_savings']
                print(f"${val:<14,.2f}", end="")
            else:
                print("N/A            ", end="")
        print()
    
    print("\n" + "=" * 50 + "\n")
    
    return figures


def create_optimal_cutoff_summary_table(optimal_cutoff_results):
    """
    Create summary table for optimal cutoff analysis
    
    Parameters:
    - optimal_cutoff_results: Dictionary with optimal cutoff analysis results by model
    
    Returns:
    - DataFrame with summary table
    """
    # Find optimal cutoff for each model (maximizing net savings)
    summary_data = []
    
    for model, results in optimal_cutoff_results.items():
        # Find cutoff with maximum net savings
        max_net_savings_idx = max(range(len(results)), key=lambda i: results[i]['net_savings'])
        optimal_cutoff = results[max_net_savings_idx]
        
        # Add to summary data
        summary_data.append({
            'Model': model,
            'Optimal Cutoff Decile': optimal_cutoff['cutoff_decile'],
            'Approval Rate (%)': optimal_cutoff['approved_pct_loans'] * 100,
            'Default Rate (%)': optimal_cutoff['approved_default_rate'] * 100,
            'Net Savings ($)': optimal_cutoff['net_savings'],
            'ROI (%)': optimal_cutoff['roi_cutoff_policy'] * 100,
            'Loss Savings ($)': optimal_cutoff['loss_savings'],
            'Revenue Opportunity Cost ($)': optimal_cutoff['revenue_opportunity_cost']
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Format numeric columns
    format_dict = {
        'Approval Rate (%)': '{:.1f}',
        'Default Rate (%)': '{:.1f}',
        'Net Savings ($)': '${:,.2f}',
        'ROI (%)': '{:.2f}',
        'Loss Savings ($)': '${:,.2f}',
        'Revenue Opportunity Cost ($)': '${:,.2f}'
    }
    
    for col, fmt in format_dict.items():
        summary_df[col] = summary_df[col].apply(lambda x: fmt.format(x))
    
    # Print the data values used in the table
    print("\n===== OPTIMAL CUTOFF SUMMARY TABLE =====\n")
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 50 + "\n")
    
    return summary_df
