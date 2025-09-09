"""
Utility functions for the modeling process.

This module will contain functions for training, evaluating, and interpreting
the machine learning models for the credit risk assessment task.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              C: float = 1.0,
                              class_weight: str = 'balanced',
                              random_state: int = 42) -> LogisticRegression:
    """
    Trains a Logistic Regression model with recommended settings for credit risk.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.
        C (float, optional): Inverse of regularization strength. Defaults to 1.0.
        class_weight (str, optional): Handles class imbalance. Defaults to 'balanced'.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        LogisticRegression: The trained logistic regression model object.
    """
    # Initialize the model with best-practice parameters for credit scoring
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty='l2',
        solver='liblinear',  # Good solver for smaller datasets
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    print("Logistic Regression model trained successfully.")

    return model


def evaluate_model(model,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Evaluates the model performance on the test set, prints key metrics, and plots the ROC curve.

    Args:
        model: A trained model object with a `predict_proba` method.
        X_test (pd.DataFrame): The test feature matrix.
        y_test (pd.Series): The test target vector.

    Returns:
        dict: A dictionary containing the AUC and KS statistic.
    """
    # Predict probabilities for the positive class (target=1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 1. Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)

    # 2. Calculate KS Statistic
    probas_target_1 = y_pred_proba[y_test == 1]
    probas_target_0 = y_pred_proba[y_test == 0]
    ks_stat, p_value = ks_2samp(probas_target_1, probas_target_0)

    print("\nModel Evaluation Results:")
    print(f"  - AUC: {auc:.4f}")
    print(f"  - K-S Statistic: {ks_stat:.4f}")

    # 3. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00529B', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#BDBDBD', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return {'auc': auc, 'ks': ks_stat}


def display_logistic_regression_coefficients(model: LogisticRegression, feature_names: list) -> pd.DataFrame:
    """
    Displays the coefficients of a trained Logistic Regression model as a bar plot
    and returns them as a DataFrame.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        feature_names (list): A list of the feature names corresponding to the coefficients.
        
    Returns:
        pd.DataFrame: A DataFrame containing features and their corresponding coefficients.
    """
    if not hasattr(model, 'coef_'):
        print("The provided model object does not have coefficients (coef_ attribute).")
        return pd.DataFrame()

    # Create a DataFrame of features and their coefficients
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False).reset_index(drop=True)

    print("\nLogistic Regression Coefficients (Top 15 positive and negative):")
    print("-" * 60)
    print("Top Positive (Higher probability of default):")
    print(coefficients_df.head(15))
    print("\nTop Negative (Lower probability of default):")
    print(coefficients_df.tail(15))
    print("-" * 60)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    
    # Use a diverging color scheme
    colors = ['#d62728' if c < 0 else '#2ca02c' for c in coefficients_df['Coefficient']]
    
    sns.barplot(x='Coefficient', y='Feature', data=coefficients_df, palette=colors, orient='h')
    
    plt.xlabel('Coefficient Value (Impact on Log-Odds)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Logistic Regression Coefficients', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', lw=0.8)
    plt.tight_layout()
    plt.show()

    return coefficients_df


def check_multicollinearity_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each feature to check for multicollinearity.

    Args:
        X (pd.DataFrame): The feature matrix.

    Returns:
        pd.DataFrame: A DataFrame with features and their corresponding VIF scores, sorted descending.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    # Calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    vif_data = vif_data.sort_values(by='VIF', ascending=False).reset_index(drop=True)
    
    print("Variance Inflation Factor (VIF) Analysis:")
    print("-" * 50)
    print(vif_data)
    print("-" * 50)
    
    high_vif_features = vif_data[vif_data['VIF'] > 5]
    if not high_vif_features.empty:
        print("\nWarning: The following features have a VIF > 5, indicating potential multicollinearity:")
        print(high_vif_features['feature'].tolist())
    else:
        print("\nSuccess: No significant multicollinearity detected (all VIF scores are <= 5).")
        
    return vif_data


def optimize_hyperparameters(X, y, model_class, param_grid, n_trials=50, n_splits=5):
    # Asegurarse de que 'y' tenga el formato correcto (1D array) para evitar warnings
    y = y.values.ravel()

    """
    Optimiza los hiperparámetros de un modelo usando Optuna (Optimización Bayesiana).

    Args:
        X (pd.DataFrame): DataFrame de características.
        y (pd.Series): Serie del target.
        model_class: La clase del modelo a optimizar (ej. LogisticRegression, XGBClassifier).
        param_grid (function): Una función que toma un `trial` de Optuna y devuelve un diccionario de parámetros.
        n_trials (int): Número de iteraciones de optimización.
        n_splits (int): Número de folds para la validación cruzada estratificada.

    Returns:
        dict: Un diccionario con los mejores hiperparámetros encontrados.
    """
    def objective(trial):
        # Genera un conjunto de hiperparámetros para este trial
        params = param_grid(trial)

        # Instancia el modelo con los parámetros sugeridos
        model = model_class(**params)

        # Configura la validación cruzada estratificada
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Calcula el score (AUC) promedio a través de la validación cruzada
        # Usamos 'roc_auc' como métrica de evaluación
        score = cross_val_score(model, X, y, n_jobs=-1, cv=cv, scoring='roc_auc').mean()

        return score

    # Crea un estudio de Optuna para maximizar el AUC
    study = optuna.create_study(direction='maximize')
    
    # Inicia la búsqueda de hiperparámetros
    # Se desactiva el log para no llenar el output, se puede activar con `show_progress_bar=True`
    study.optimize(objective, n_trials=n_trials)

    print(f"Mejor score AUC (cross-validation): {study.best_value:.4f}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params


# --- Funciones de Evaluación y Visualización de Modelos ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_model_evaluation_summary(base_model, X_data, y_true, calibrator=None, model_name='Model', threshold=0.5, custom_color='#b4a7d6'):
    """
    Generates a 3-panel visualization to evaluate a classification model in English.
    Calculates probabilities internally and handles calibrated models.

    Panels:
    1. Confusion Matrix.
    2. Distribution of predicted probabilities for actual classes.
    3. ROC Curve.

    Args:
        base_model: The pre-trained base classification model.
        X_data (DataFrame): The feature data for evaluation.
        y_true (array-like): True class labels.
        calibrator (object, optional): The trained calibrator (e.g., IsotonicRegression).
                                       If provided, it will be used to transform probabilities.
        model_name (str): Name of the model for plot titles.
        threshold (float): Decision threshold for classifying as positive.
        custom_color (str): Main color for the visualizations.
    """
    # --- 1. Generate Predictions ---
    y_pred_proba_raw = base_model.predict_proba(X_data)[:, 1]

    if calibrator:
        y_pred_proba = calibrator.transform(y_pred_proba_raw)
        plot_title_suffix = f'({model_name} - Calibrated)'
    else:
        y_pred_proba = y_pred_proba_raw
        plot_title_suffix = f'({model_name} - Uncalibrated)'

    # --- 2. Prepare Data for Plotting ---
    y_true = np.asarray(y_true).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    probs_pos = y_pred_proba[y_true == 1]
    probs_neg = y_pred_proba[y_true == 0]

    # --- 3. Create Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Panel 1: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    group_names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    group_counts = [f"{value:0,d}" for value in cm.flatten()]
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_counts, group_names)]
    labels = np.asarray(labels).reshape(2, 2)
    
    cmap = sns.light_palette(custom_color, as_cmap=True)
    
    sns.heatmap(cm, ax=axes[0], annot=labels, fmt='', cmap=cmap, cbar=False, 
                annot_kws={"size": 14, "color": "white" if custom_color < '#888888' else 'black'})
    axes[0].set_title(f'Confusion Matrix {plot_title_suffix}', fontsize=16)
    axes[0].set_ylabel('True Values', fontsize=12)
    axes[0].set_xlabel('Predicted Values', fontsize=12)

    # --- Panel 2: Distribution of Predictions ---
    sns.histplot(probs_neg, ax=axes[1], bins=30, stat='density', color='darkgrey', label='Negatives')
    sns.histplot(probs_pos, ax=axes[1], bins=30, stat='density', color=custom_color, alpha=0.7, label='Positives')
    axes[1].axvline(threshold, color='#00A0A0', linestyle='--', label='Boundary')
    axes[1].set_title(f'Distributions of Predictions {plot_title_suffix}', fontsize=16)
    axes[1].set_xlabel('Positive Probability (predicted)', fontsize=12)
    axes[1].set_ylabel('Samples (normalized scale)', fontsize=12)
    axes[1].legend()

    # --- Panel 3: ROC Curve ---
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[2].plot(fpr, tpr, color=custom_color, lw=2, label=f'ROC curve (area = {auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    
    idx = np.argmin(np.abs(thresholds_roc - threshold))
    if thresholds_roc[idx] < 1.0:
        axes[2].plot(fpr[idx], tpr[idx], 'o', markersize=10, color='#2c3e50', label='Decision Point')

    axes[2].set_xlim([-0.02, 1.0])
    axes[2].set_ylim([0.0, 1.02])
    axes[2].set_xlabel('False Positive Rate', fontsize=12)
    axes[2].set_ylabel('True Positive Rate', fontsize=12)
    axes[2].set_title(f'ROC Curve {plot_title_suffix}', fontsize=16)
    axes[2].legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

def generate_classification_report(base_model, X_train, y_train, X_test, y_test, calibrator=None, threshold=0.5):
    """
    Genera un DataFrame con un reporte de métricas de clasificación para train y test.

    Incluye AUC, Gini, Brier Score, y métricas del classification_report de scikit-learn.

    Args:
        base_model: El modelo de clasificación base ya entrenado.
        X_train (DataFrame): Datos de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        X_test (DataFrame): Datos de prueba.
        y_test (array-like): Etiquetas de prueba.
        calibrator (object, optional): El calibrador entrenado (p. ej., IsotonicRegression).
        threshold (float): Umbral de decisión para la clasificación.

    Returns:
        DataFrame: Un DataFrame de pandas con las métricas comparativas.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
    import pandas as pd

    sets = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test)
    }

    all_metrics = {}

    for set_name, (X, y) in sets.items():
        # Asegurar que y_true sea un array 1D de enteros para consistencia de tipos
        y_true = np.asarray(y).ravel().astype(int)

        # Generar probabilidades
        y_proba_raw = base_model.predict_proba(X)[:, 1]
        y_proba = calibrator.transform(y_proba_raw) if calibrator else y_proba_raw
        y_pred = (y_proba >= threshold).astype(int)

        # Calcular métricas
        auc = roc_auc_score(y_true, y_proba)
        gini = 2 * auc - 1
        brier = brier_score_loss(y_true, y_proba)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Almacenar métricas de forma segura usando .get()
        metrics = {
            'AUC': auc,
            'Gini': gini,
            'Brier Score': brier,
            'Precision (Class 0)': report.get('0', {}).get('precision', 0.0),
            'Recall (Class 0)': report.get('0', {}).get('recall', 0.0),
            'F1-Score (Class 0)': report.get('0', {}).get('f1-score', 0.0),
            'Precision (Class 1)': report.get('1', {}).get('precision', 0.0),
            'Recall (Class 1)': report.get('1', {}).get('recall', 0.0),
            'F1-Score (Class 1)': report.get('1', {}).get('f1-score', 0.0),
            'Accuracy': report.get('accuracy', 0.0)
        }
        all_metrics[set_name] = metrics

    # Crear DataFrame y transponer para el formato deseado
    report_df = pd.DataFrame(all_metrics).T
    return report_df.T


