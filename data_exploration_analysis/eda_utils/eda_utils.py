import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_correlation_matrix_plotly(df, columns=None, method='pearson',
                                   color_continuous_scale='RdBu_r', # Plotly continuous color scale
                                   title="Matriz de Correlación (Plotly)"):
    """
    Calcula y visualiza una matriz de correlación interactiva usando Plotly.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada.
    columns : list, opcional
        Lista de nombres de columnas a incluir en la matriz de correlación.
        Si es None, se utilizarán todas las columnas numéricas del DataFrame.
    method : str, opcional (default='pearson')
        Método de correlación a utilizar ('pearson', 'kendall', 'spearman').
    color_continuous_scale : str, opcional (default='RdBu_r')
        Paleta de colores continua de Plotly para el heatmap. Ejemplos: 'Viridis', 'Cividis', 'Blues', 'Greens', etc.
        El sufijo '_r' invierte la paleta (e.g., 'RdBu_r' va de azul a rojo).
    title : str, opcional (default="Matriz de Correlación (Plotly)")
        Título del gráfico.

    Retorna:
    --------
    plotly.graph_objects.Figure
        El objeto de figura de Plotly, que se mostrará en entornos compatibles (como Jupyter).
    """
    if columns:
        # Asegurarse de que solo se seleccionan columnas existentes en el DataFrame
        valid_columns = [col for col in columns if col in df.columns]
        if len(valid_columns) < len(columns):
            print(f"Advertencia: Algunas columnas especificadas no existen en el DataFrame y serán excluidas.")
        
        numeric_df = df[valid_columns].select_dtypes(include=np.number)
        
        if numeric_df.shape[1] < len(valid_columns):
            non_numeric_in_selection = [col for col in valid_columns if col not in numeric_df.columns]
            print(f"Advertencia: Las siguientes columnas especificadas no son numéricas y serán excluidas: {non_numeric_in_selection}")
        if numeric_df.empty:
            print("Error: Ninguna de las columnas especificadas (y existentes) es numérica.")
            return None
    else:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("Error: El DataFrame no contiene columnas numéricas para calcular la correlación.")
            return None

    if numeric_df.shape[1] < 2:
        print(f"Error: Se necesitan al menos dos columnas numéricas para calcular la correlación. Columnas numéricas encontradas: {numeric_df.columns.tolist()}")
        return None

    print(f"Calculando correlación para {numeric_df.shape[1]} columnas numéricas: {numeric_df.columns.tolist()}")
    corr_matrix = numeric_df.corr(method=method)

    fig = px.imshow(corr_matrix,
                    text_auto=".2f", # Muestra los valores con 2 decimales
                    aspect="auto",
                    color_continuous_scale=color_continuous_scale,
                    title=title,
                    labels=dict(color="Correlación"))
    
    fig.update_layout(
        title_x=0.5, # Centrar título
        xaxis_tickangle=-45
    )
    fig.update_xaxes(side="bottom") 
    return fig


def plot_distribution_plotly(df, column, target_column=None, plot_type='hist',
                             color_discrete_sequence=px.colors.qualitative.Plotly, 
                             bins=None, 
                             title_suffix=" (Plotly)"):
    """
    Visualiza la distribución de una columna usando Plotly, opcionalmente segmentada por una columna objetivo.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Nombre de la columna a visualizar.
    target_column : str, opcional
        Nombre de la columna objetivo para segmentar la visualización (e.g., 'loan_status').
    plot_type : str, opcional ('hist', 'box', 'violin', 'bar')
        Tipo de gráfico a generar:
        - 'hist': Histograma (para numéricas).
        - 'box': Diagrama de caja (para numéricas, puede agruparse por `target_column`).
        - 'violin': Diagrama de violín (similar a box, pero muestra densidad).
        - 'bar': Diagrama de barras (para categóricas, cuenta frecuencias).
    color_discrete_sequence : list of str, opcional
        Secuencia de colores discretos de Plotly.
    bins : int, opcional (default=None, Plotly elige automáticamente)
        Número de bins para histogramas (usado si plot_type='hist').
    title_suffix : str, opcional
        Sufijo para el título del gráfico.

    Retorna:
    --------
    plotly.graph_objects.Figure
        El objeto de figura de Plotly.
    """
    if column not in df.columns:
        print(f"Error: La columna '{column}' no existe en el DataFrame.")
        return None
    if target_column and target_column not in df.columns:
        print(f"Advertencia: La columna objetivo '{target_column}' no existe en el DataFrame. Se procederá sin segmentación.")
        target_column = None

    fig = None
    base_title = f'Distribución de {column}'
    if target_column:
        base_title += f' por {target_column}'
    
    final_title = base_title + title_suffix

    # Determinar si la columna es categórica o numérica para la lógica de 'bar'
    is_categorical_col = df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column])

    if plot_type == 'bar' or (is_categorical_col and plot_type not in ['box', 'violin']):
        # Usar bar para categóricas o si se especifica 'bar'
        # Ordenar por frecuencia para mejor visualización
        value_counts_sorted = df[column].value_counts()

        if target_column:
            # Agrupar y contar, luego asegurar el orden
            count_df = df.groupby([target_column, column]).size().reset_index(name='counts')
            # Para ordenar las barras de la columna principal según su frecuencia total:
            count_df[column] = pd.Categorical(count_df[column], categories=value_counts_sorted.index, ordered=True)
            count_df = count_df.sort_values(column)

            fig = px.bar(count_df, x=column, y='counts', color=target_column,
                         title=final_title, barmode='group',
                         color_discrete_sequence=color_discrete_sequence)
        else:
            count_df = value_counts_sorted.reset_index()
            count_df.columns = [column, 'counts']
            fig = px.bar(count_df, x=column, y='counts', title=final_title,
                         color_discrete_sequence=color_discrete_sequence)
        fig.update_layout(xaxis_title=column, yaxis_title='Cantidad')

    elif plot_type == 'hist' and not is_categorical_col:
        fig = px.histogram(df, x=column, color=target_column,
                           title=final_title,
                           marginal="box" if target_column else None, 
                           nbins=bins,
                           color_discrete_sequence=color_discrete_sequence,
                           barmode='overlay' if target_column else 'relative') # Overlay para comparar distribuciones
        if target_column:
             fig.update_traces(opacity=0.75) # Mejorar visibilidad con overlay
        fig.update_layout(xaxis_title=column, yaxis_title='Frecuencia')

    elif plot_type == 'box' and not is_categorical_col:
        fig = px.box(df, y=column, x=target_column, color=target_column,
                     title=final_title,
                     color_discrete_sequence=color_discrete_sequence,
                     points="outliers")
        fig.update_layout(yaxis_title=column, xaxis_title=target_column if target_column else '')

    elif plot_type == 'violin' and not is_categorical_col:
        fig = px.violin(df, y=column, x=target_column, color=target_column,
                        title=final_title,
                        box=True, 
                        points="outliers",
                        color_discrete_sequence=color_discrete_sequence)
        fig.update_layout(yaxis_title=column, xaxis_title=target_column if target_column else '')
        
    else:
        print(f"Error: Tipo de gráfico '{plot_type}' no reconocido o no aplicable para el tipo de dato de la columna '{column}' (Tipo: {df[column].dtype}).")
        return None

    if fig:
        fig.update_layout(title_x=0.5) 
    return fig


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_psi(expected_series, actual_series, bins=10, variable_type=None, clip_perc=0.00001):
    """
    Calcula el Índice de Estabilidad de la Población (PSI) entre dos series.

    El PSI mide cuánto ha cambiado la distribución de una variable entre una población
    de referencia (esperada) y una población actual.
    - PSI < 0.1: Sin cambio significativo. La variable es estable.
    - 0.1 <= PSI < 0.25: Cambio moderado. Requiere investigación.
    - PSI >= 0.25: Cambio significativo. La variable es inestable.

    Parámetros:
    -----------
    expected_series : pd.Series
        Serie con los datos de la población de referencia (ej. datos de entrenamiento, periodo anterior).
    actual_series : pd.Series
        Serie con los datos de la población actual (ej. datos de scoring, periodo actual).
    bins : int o list/array-like, opcional (default=10)
        Para variables numéricas:
        - Si es int: Número de cuantiles a usar para crear los bins, basados en `expected_series`.
        - Si es list/array-like: Bordes de los bins a utilizar.
        Para variables categóricas, este parámetro se ignora.
    variable_type : str, opcional ('numerical' o 'categorical')
        Especifica el tipo de variable. Si es None, se infiere del dtype de `expected_series`.
    clip_perc : float, opcional (default=0.00001)
        Pequeño valor para reemplazar porcentajes de 0 y evitar errores de log(0) o división por cero.
        Se suma este valor si el porcentaje es 0.

    Retorna:
    --------
    float
        El valor total del PSI.
    pd.DataFrame
        Un DataFrame detallando la contribución de cada bin/categoría al PSI.
    """

    # Validar y limpiar series de entrada
    if not isinstance(expected_series, pd.Series) or not isinstance(actual_series, pd.Series):
        raise ValueError("Tanto expected_series como actual_series deben ser pd.Series.")

    expected_series_clean = expected_series.dropna()
    actual_series_clean = actual_series.dropna()

    if expected_series_clean.empty:
        print(f"Advertencia: expected_series para '{expected_series.name}' está vacía después de eliminar NaNs. No se puede calcular PSI.")
        return np.nan, pd.DataFrame()
    if actual_series_clean.empty:
        print(f"Advertencia: actual_series para '{actual_series.name}' está vacía después de eliminar NaNs. No se puede calcular PSI.")
        return np.nan, pd.DataFrame()

    # Inferir tipo de variable si no se proporciona
    if variable_type is None:
        if pd.api.types.is_numeric_dtype(expected_series_clean):
            variable_type = 'numerical'
        else:
            variable_type = 'categorical'

    # Procesamiento y binning
    if variable_type == 'numerical':
        if isinstance(bins, int): 
            try:
                bin_edges = pd.qcut(expected_series_clean, q=bins, retbins=True, duplicates='drop')[1]
            except ValueError: # Si qcut falla (e.g. no suficientes valores únicos para cuantiles)
                bin_edges = [] # Señal para usar fallback

            if len(bin_edges) <= 2: 
                print(f"Advertencia: No se pudieron crear {bins} cuantiles únicos para '{expected_series.name}'. Usando estrategia de fallback para binning.")
                unique_vals_sorted = np.unique(np.concatenate([expected_series_clean.unique(), actual_series_clean.unique()]))
                unique_vals_sorted.sort()
                
                if len(unique_vals_sorted) > 1:
                    min_val_data = unique_vals_sorted[0]
                    max_val_data = unique_vals_sorted[-1]
                    if len(unique_vals_sorted) <= bins : 
                        bin_edges = np.array([min_val_data - 0.5] + [(unique_vals_sorted[i] + unique_vals_sorted[i+1]) / 2 for i in range(len(unique_vals_sorted)-1)] + [max_val_data + 0.5])
                    else: 
                        bin_edges = np.linspace(min_val_data, max_val_data, bins + 1)
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 2:
                         bin_edges = np.array([min_val_data - 0.5 * abs(min_val_data) if min_val_data != 0 else -0.5, 
                                               max_val_data + 0.5 * abs(max_val_data) if max_val_data != 0 else 0.5])
                         if bin_edges[0] == bin_edges[1]: bin_edges[1] +=1 # Ensure distinct if single point after adjustment
                elif len(unique_vals_sorted) == 1: 
                    val = unique_vals_sorted[0]
                    bin_edges = np.array([val - 0.5 * abs(val) if val != 0 else -0.5, 
                                          val + 0.5 * abs(val) if val != 0 else 0.5])
                    if bin_edges[0] == bin_edges[1]: bin_edges[1] +=1
                else: 
                    return np.nan, pd.DataFrame()
            else: 
                min_val_combined = min(expected_series_clean.min(), actual_series_clean.min())
                max_val_combined = max(expected_series_clean.max(), actual_series_clean.max())
                bin_edges[0] = min(bin_edges[0], min_val_combined) 
                bin_edges[-1] = max(bin_edges[-1], max_val_combined)
                bin_edges = np.unique(bin_edges)

        elif isinstance(bins, (list, np.ndarray)):
            bin_edges = np.unique(np.array(bins))
            if len(bin_edges) < 2:
                raise ValueError("Si 'bins' es una lista, debe contener al menos dos bordes únicos.")
        else:
            raise ValueError("El parámetro 'bins' para variables numéricas debe ser un int o una lista/array de bordes.")

        expected_binned = pd.cut(expected_series_clean, bins=bin_edges, include_lowest=True, duplicates='drop')
        actual_binned = pd.cut(actual_series_clean, bins=bin_edges, include_lowest=True, duplicates='drop')
        
    elif variable_type == 'categorical':
        expected_binned = expected_series_clean.astype('str') # Convert to string to handle mixed types as categories
        actual_binned = actual_series_clean.astype('str')
        all_categories = pd.Index(expected_binned.unique()).union(actual_binned.unique())
        expected_binned = pd.Categorical(expected_binned, categories=all_categories)
        actual_binned = pd.Categorical(actual_binned, categories=all_categories)
    else:
        raise ValueError("variable_type debe ser 'numerical' o 'categorical'.")

    df_expected_dist = pd.DataFrame(expected_binned.value_counts(dropna=False)).rename(columns={'count': 'Expected_Count'})
    df_actual_dist = pd.DataFrame(actual_binned.value_counts(dropna=False)).rename(columns={'count': 'Actual_Count'})
    
    psi_df = df_expected_dist.join(df_actual_dist, how='outer').fillna(0)
    if isinstance(psi_df.index, pd.IntervalIndex):
        psi_df.index = psi_df.index.astype(str)
    psi_df.index.name = 'Category_or_Bin'
    
    total_expected = psi_df['Expected_Count'].sum()
    total_actual = psi_df['Actual_Count'].sum()

    if total_expected == 0:
        print(f"Advertencia: La suma de Expected_Count para '{expected_series.name}' es 0. No se puede calcular PSI.")
        return np.nan, psi_df
    if total_actual == 0:
        print(f"Advertencia: La suma de Actual_Count para '{actual_series.name}' es 0. No se puede calcular PSI.")
        return np.nan, psi_df

    psi_df['Expected_Perc'] = psi_df['Expected_Count'] / total_expected
    psi_df['Actual_Perc'] = psi_df['Actual_Count'] / total_actual

    psi_df['Expected_Perc_Clipped'] = np.where(psi_df['Expected_Perc'] == 0, clip_perc, psi_df['Expected_Perc'])
    psi_df['Actual_Perc_Clipped'] = np.where(psi_df['Actual_Perc'] == 0, clip_perc, psi_df['Actual_Perc'])
    
    psi_df['PSI_Contribution'] = (psi_df['Actual_Perc_Clipped'] - psi_df['Expected_Perc_Clipped']) * \
                                 np.log(psi_df['Actual_Perc_Clipped'] / psi_df['Expected_Perc_Clipped'])
    
    total_psi = psi_df['PSI_Contribution'].sum()
    
    psi_df = psi_df.sort_index()

    return total_psi, psi_df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def analyze_null_values_by_group(df, group_col, analyze_cols=None):
    """
    Calcula el número y porcentaje de valores nulos por columna, agrupados por una columna específica.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada a analizar.
    group_col : str
        El nombre de la columna por la cual agrupar (ej: 'd_vintage').
    analyze_cols : list, opcional
        Lista de nombres de columnas para analizar los nulos. 
        Si es None, se analizarán todas las columnas excepto la de agrupación.

    Retorna:
    --------
    pd.DataFrame
        Un DataFrame con `group_col` como índice y un MultiIndex en las columnas
        que contiene 'Null Count' y 'Null Percentage (%)' para cada `analyze_cols`.
        Las columnas estarán ordenadas alfabéticamente y el índice (group_col) también.
    """
    if group_col not in df.columns:
        print(f"Error: La columna de agrupación '{group_col}' no existe en el DataFrame.")
        return None

    if analyze_cols is None:
        analyze_cols = df.columns.tolist()
        if group_col in analyze_cols: # Asegurarse de removerla si está
            analyze_cols.remove(group_col)
    else:
        valid_analyze_cols = []
        for col in analyze_cols:
            if col not in df.columns:
                print(f"Advertencia: La columna a analizar '{col}' no existe y será omitida.")
            elif col == group_col:
                print(f"Advertencia: La columna de agrupación '{group_col}' no debe estar en `analyze_cols` y será omitida.")
            else:
                valid_analyze_cols.append(col)
        analyze_cols = sorted(valid_analyze_cols) # Ordenar para consistencia en columnas
        if not analyze_cols:
            print("Error: No hay columnas válidas para analizar después de las validaciones.")
            return None
    
    if not analyze_cols: # Chequeo adicional si analyze_cols quedó vacío después de remover group_col
        print(f"Error: No quedan columnas para analizar después de excluir '{group_col}'.")
        return None

    grouped = df.groupby(group_col)
    results = {}

    for name, group_df in grouped:
        if len(group_df) == 0: # Si un grupo está vacío
            group_null_counts = pd.Series(0, index=analyze_cols)
            group_null_percentages = pd.Series(0.0, index=analyze_cols)
        else:
            group_null_counts = group_df[analyze_cols].isnull().sum()
            group_null_percentages = (group_null_counts / len(group_df)) * 100
        
        group_results = {}
        for col in analyze_cols:
            group_results[(col, 'Null Count')] = group_null_counts[col]
            group_results[(col, 'Null Percentage (%)')] = group_null_percentages[col]
        results[name] = group_results

    if not results:
        print(f"No se encontraron grupos para '{group_col}' o no hay datos para analizar.")
        return pd.DataFrame()

    result_df = pd.DataFrame.from_dict(results, orient='index')

    if result_df.empty: # Si el DataFrame resultante está vacío
        print(f"El DataFrame resultante para el análisis de nulos por '{group_col}' está vacío.")
        return result_df

    # Ordenar el índice (group_col) y las columnas (alfabéticamente)
    result_df.sort_index(inplace=True)
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)

    return result_df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_woe_iv(df, feature_col, target_col, bins=10, clip_perc=0.00001):
    """
    Calcula el Peso de la Evidencia (WOE) y el Valor de Información (IV) para una variable.

    El IV es una medida de la capacidad predictiva de una variable para separar los buenos de los malos.
    - IV < 0.02: Inútil para la predicción.
    - 0.02 <= IV < 0.1: Predictor débil.
    - 0.1 <= IV < 0.3: Predictor medio.
    - 0.3 <= IV < 0.5: Predictor fuerte.
    - IV >= 0.5: Sospechoso o demasiado bueno para ser verdad.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada.
    feature_col : str
        Nombre de la columna de la característica (predictora).
    target_col : str
        Nombre de la columna objetivo (binaria, 0s y 1s).
    bins : int, opcional (default=10)
        Número de bins a crear para variables numéricas usando cuantiles.
        Se ignora para variables categóricas.
    clip_perc : float, opcional (default=0.00001)
        Pequeño valor para reemplazar conteos de 0 en eventos o no-eventos para evitar división por cero.

    Retorna:
    --------
    pd.DataFrame
        Un DataFrame que detalla el cálculo de WOE e IV para cada bin/categoría.
    float
        El valor total de IV para la característica.
    """
    # Validaciones iniciales
    if feature_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Las columnas '{feature_col}' o '{target_col}' no se encuentran en el DataFrame.")

    df_copy = df[[feature_col, target_col]].copy()
    df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')
    df_copy.dropna(subset=[target_col], inplace=True)

    if not pd.api.types.is_numeric_dtype(df_copy[target_col]):
        raise TypeError(f"La columna objetivo '{target_col}' debe ser numérica.")

    if df_copy[target_col].nunique() != 2:
        print(f"Advertencia: La columna objetivo '{target_col}' no es binaria. Resultados pueden no ser significativos.")

    # Si la variable es numérica pero tiene pocos valores únicos (ej. binaria), trátala como categórica.
    # De lo contrario, si es numérica con muchos valores, aplica binning.
    if pd.api.types.is_numeric_dtype(df_copy[feature_col]) and df_copy[feature_col].nunique() > bins:
        # Binning para variables numéricas continuas
        # Crear una categoría explícita para nulos
        df_copy[f'{feature_col}_binned'] = df_copy[feature_col].apply(lambda x: 'Missing' if pd.isnull(x) else x)
        numeric_part = df_copy[df_copy[f'{feature_col}_binned'] != 'Missing'][feature_col]

        if not numeric_part.empty:
            try:
                # Usar qcut en la parte no nula
                binned_data = pd.qcut(numeric_part, q=bins, duplicates='drop')
                df_copy.loc[numeric_part.index, f'{feature_col}_binned'] = binned_data.astype(str)
            except ValueError:
                # Si qcut falla, usar los valores únicos como categorías
                df_copy.loc[numeric_part.index, f'{feature_col}_binned'] = numeric_part.astype(str)
        
        feature_binned_col = f'{feature_col}_binned'
    else:
        # Tratar como categórica (incluye strings, object, y numéricas de baja cardinalidad)
        # Convertir nulos a una categoría explícita 'Missing'
        df_copy[f'{feature_col}_binned'] = df_copy[feature_col].apply(lambda x: 'Missing' if pd.isnull(x) else str(x))
        feature_binned_col = f'{feature_col}_binned'

    # Calcular eventos y no-eventos por bin/categoría
    grouped = df_copy.groupby(feature_binned_col)[target_col].agg(['count', lambda x: (x == 1).sum()])
    grouped.columns = ['Total', 'Events']
    grouped['Non_Events'] = grouped['Total'] - grouped['Events']

    # Evitar división por cero
    grouped['Events'] = np.where(grouped['Events'] == 0, clip_perc, grouped['Events'])
    grouped['Non_Events'] = np.where(grouped['Non_Events'] == 0, clip_perc, grouped['Non_Events'])

    total_events = grouped['Events'].sum()
    total_non_events = grouped['Non_Events'].sum()

    if total_events == 0 or total_non_events == 0:
        print(f"Advertencia: No hay eventos o no-eventos en los datos para '{feature_col}'. No se puede calcular WOE/IV.")
        return pd.DataFrame(), 0.0

    grouped['Distr_Events'] = grouped['Events'] / total_events
    grouped['Distr_Non_Events'] = grouped['Non_Events'] / total_non_events

    # Calcular WOE e IV
    grouped['WOE'] = np.log(grouped['Distr_Events'] / grouped['Distr_Non_Events'])
    grouped['IV'] = (grouped['Distr_Events'] - grouped['Distr_Non_Events']) * grouped['WOE']

    total_iv = grouped['IV'].sum()

    grouped.index.name = 'Category_or_Bin'
    grouped.reset_index(inplace=True)

    return grouped, total_iv

    result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)
    result_df.index.name = group_col
    
    return result_df.sort_index() # Ordenar por el índice (group_col)


def generate_null_alerts_by_vintage(df, 
                                    group_col, 
                                    analyze_cols=None,
                                    high_null_perc_threshold=50.0,
                                    min_increase_perc_threshold=10.0):
    """
    Analyzes null value percentages for specified columns grouped by a vintage/group column,
    and generates alerts if percentages exceed thresholds or show significant increases.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada.
    group_col : str
        Nombre de la columna para agrupar (ej. 'd_vintage'). Se espera que esta columna
        pueda ser ordenada cronológicamente.
    analyze_cols : list of str, opcional (default=None)
        Lista de nombres de columnas a analizar. Si es None, se analizan todas las columnas
        excepto group_col.
    high_null_perc_threshold : float, opcional (default=50.0)
        Umbral de porcentaje de nulos (0-100) por encima del cual se genera una alerta
        de 'High Null Percentage'.
    min_increase_perc_threshold : float, opcional (default=10.0)
        Aumento mínimo en puntos porcentuales (pp) en el porcentaje de nulos de un vintage
        al siguiente para generar una alerta de 'Significant Increase in Nulls'.

    Retorna:
    --------
    pd.DataFrame
        Un DataFrame con los detalles de las alertas generadas. Las columnas incluyen:
        'Vintage', 'Variable', 'Null_Percentage_Current_Vintage', 
        'Null_Percentage_Previous_Vintage', 'Increase_In_Nulls_pp', 
        'Alert_Type', 'Details'.
        Retorna un DataFrame vacío con estas columnas si no se generan alertas.
    """
    if analyze_cols is None:
        analyze_cols = [col for col in df.columns if col != group_col]

    # This function is assumed to return a DataFrame with group_col as index
    # and MultiIndex columns like (variable_name, 'Null Percentage (%)')
    null_summary_by_group = analyze_null_values_by_group(df, group_col, analyze_cols)

    DESIRED_COLUMNS = [
        'Vintage', 'Variable', 'Null_Percentage_Current_Vintage',
        'Null_Percentage_Previous_Vintage', 'Increase_In_Nulls_pp',
        'Alert_Type', 'Details'
    ]

    if null_summary_by_group.empty:
        return pd.DataFrame(columns=DESIRED_COLUMNS)

    alerts_list = []
    
    try:
        # Assuming group_col is the index from analyze_null_values_by_group
        sorted_vintages = sorted(null_summary_by_group.index.unique())
    except TypeError:
        print(f"Warning: Could not sort unique values of group_col '{group_col}'. Proceeding with unsorted order.")
        sorted_vintages = null_summary_by_group.index.unique()


    for col_to_analyze in analyze_cols:
        prev_null_perc = np.nan 

        for current_vintage_val in sorted_vintages:
            try:
                # Accessing data using .loc with MultiIndex columns
                current_data_point = null_summary_by_group.loc[current_vintage_val, (col_to_analyze, 'Null Percentage (%)')]
            except KeyError:
                # This can happen if a specific variable/vintage combination is not in the summary
                current_data_point = np.nan # Treat as NaN if not found

            if pd.isna(current_data_point):
                current_null_perc = np.nan 
            else:
                current_null_perc = current_data_point

            if pd.isna(current_null_perc):
                prev_null_perc = np.nan 
                continue

            # Alert 1: High null percentage
            if current_null_perc > high_null_perc_threshold:
                alerts_list.append({
                    'Vintage': current_vintage_val,
                    'Variable': col_to_analyze,
                    'Null_Percentage_Current_Vintage': current_null_perc,
                    'Null_Percentage_Previous_Vintage': np.nan, 
                    'Increase_In_Nulls_pp': np.nan, 
                    'Alert_Type': 'High Null Percentage',
                    'Details': (f"Variable '{col_to_analyze}' in vintage '{current_vintage_val}' has {current_null_perc:.2f}% nulls, exceeding threshold of {high_null_perc_threshold}%")
                })

            # Alert 2: Significant increase in nulls
            if not pd.isna(prev_null_perc): 
                increase = current_null_perc - prev_null_perc
                if increase > min_increase_perc_threshold:
                    alerts_list.append({
                        'Vintage': current_vintage_val,
                        'Variable': col_to_analyze,
                        'Null_Percentage_Current_Vintage': current_null_perc,
                        'Null_Percentage_Previous_Vintage': prev_null_perc,
                        'Increase_In_Nulls_pp': increase,
                        'Alert_Type': 'Significant Increase in Nulls',
                        'Details': (f"Variable '{col_to_analyze}' in vintage '{current_vintage_val}' increased nulls to "
                                    f"{current_null_perc:.2f}% from {prev_null_perc:.2f}% (increase of {increase:.2f}pp), "
                                    f"exceeding threshold of {min_increase_perc_threshold}pp.")
                    })
            
            prev_null_perc = current_null_perc # Update for the next vintage

    if not alerts_list:
        return pd.DataFrame(columns=DESIRED_COLUMNS)
    else:
        alerts_df = pd.DataFrame(alerts_list)
        # Ensure all desired columns exist and are in the correct order
        for col_name in DESIRED_COLUMNS:
            if col_name not in alerts_df.columns:
                alerts_df[col_name] = np.nan
        return alerts_df[DESIRED_COLUMNS]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def assess_variable_quality(df, 
                            null_perc_threshold=70.0, 
                            constant_perc_threshold=95.0):
    """
    Assesses variable quality based on null percentages and variability (constant-like behavior).

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada.
    null_perc_threshold : float, opcional (default=70.0)
        Umbral de porcentaje de nulos (0-100) por encima del cual una variable 
        es marcada como 'Flag_High_Nulls'.
    constant_perc_threshold : float, opcional (default=95.0)
        Umbral de porcentaje (0-100) para la frecuencia del valor más común (excluyendo NaNs).
        Si el valor más común excede este umbral, la variable es marcada como 'Flag_Constant_Like'.
        Un valor de 100 significa que la variable tiene un único valor (después de quitar NaNs).

    Retorna:
    --------
    pd.DataFrame
        Un DataFrame con las siguientes columnas para cada variable del df de entrada:
        - 'Variable': Nombre de la columna.
        - 'Dtype': Tipo de dato de la columna.
        - 'Null_Count': Cantidad de valores nulos.
        - 'Null_Percentage': Porcentaje de valores nulos.
        - 'Unique_Values_No_NA': Cantidad de valores únicos (excluyendo NaNs).
        - 'Most_Frequent_Value_No_NA': El valor más frecuente (excluyendo NaNs).
        - 'Most_Frequent_Value_Perc_No_NA': Porcentaje del valor más frecuente (excluyendo NaNs).
        - 'Flag_High_Nulls': True si Null_Percentage > null_perc_threshold.
        - 'Flag_Constant_Like': True si Most_Frequent_Value_Perc_No_NA > constant_perc_threshold.
        - 'Recommendation': Sugerencia ('Keep', 'Review: High Nulls', 'Review: Constant-Like', 
                          'Remove: High Nulls & Constant-Like', 'Remove: All Nulls', 'Remove: Constant').
        El DataFrame estará ordenado por 'Null_Percentage' y luego por 'Most_Frequent_Value_Perc_No_NA'.
    """
    
    results = []
    for col in df.columns:
        # Null analysis
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        # Variability analysis (excluding NaNs)
        col_no_na = df[col].dropna()
        num_unique_no_na = col_no_na.nunique()
        
        most_frequent_value_no_na = None
        most_frequent_value_perc_no_na = np.nan # Use NaN if all are null or empty
        
        if not col_no_na.empty:
            counts_no_na = col_no_na.value_counts(normalize=False)
            most_frequent_value_no_na = counts_no_na.index[0]
            most_frequent_value_perc_no_na = (counts_no_na.iloc[0] / len(col_no_na)) * 100
        elif null_percentage == 100: # All values are NaN
             most_frequent_value_perc_no_na = 0 # Or np.nan, but 0 helps in logic for constant if all null

        # Flagging
        flag_high_nulls = null_percentage > null_perc_threshold
        
        if num_unique_no_na == 0: # All values were NaN
            flag_constant_like = False 
        elif num_unique_no_na == 1:
            flag_constant_like = True 
        else: # More than 1 unique value
            flag_constant_like = most_frequent_value_perc_no_na > constant_perc_threshold

        # Recommendation logic
        recommendation = "Keep"
        if null_percentage == 100:
            recommendation = "Remove: All Nulls"
            flag_constant_like = False 
        elif flag_high_nulls and flag_constant_like:
            recommendation = "Remove: High Nulls & Constant-Like"
        elif flag_high_nulls:
            recommendation = "Review: High Nulls"
        elif flag_constant_like:
            if num_unique_no_na == 1: 
                 recommendation = "Remove: Constant (Single Value)"
            else:
                 recommendation = "Review: Constant-Like"

        results.append({
            'Variable': col,
            'Dtype': df[col].dtype,
            'Null_Count': null_count,
            'Null_Percentage': round(null_percentage, 2),
            'Unique_Values_No_NA': num_unique_no_na,
            'Most_Frequent_Value_No_NA': most_frequent_value_no_na,
            'Most_Frequent_Value_Perc_No_NA': round(most_frequent_value_perc_no_na, 2) if not np.isnan(most_frequent_value_perc_no_na) else np.nan,
            'Flag_High_Nulls': flag_high_nulls,
            'Flag_Constant_Like': flag_constant_like,
            'Recommendation': recommendation
        })
        
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(
        by=['Null_Percentage', 'Most_Frequent_Value_Perc_No_NA'], 
        ascending=[False, False]
    )
    
    return summary_df
