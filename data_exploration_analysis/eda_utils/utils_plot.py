import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

# Define custom_stopwords or ensure it's passed as an argument if used globally
custom_stopwords = [] # Placeholder, user should define this or pass it to plot_purpose_treemap

def plot_chart(
    df,
    chart_type="bar",             # "bar" o "line"
    x_col="x",
    y_col="y",
    text_col=None,
    group_col=None,              # para múltiples líneas o barras
    sort_by=None,
    ascending=True,
    x_label=None,
    y_label=None,
    title="Chart",
    color=None,                  # puede ser columna string o color fijo
    color_sequence=None,         # lista de colores para los grupos
    width=1200,
    height=700,
    text_as_percentage=True,
    text_position="outside",     # "outside" para barras, "top center" para líneas
    barmode="group",
    category_order=None              # solo para barras: "group", "stack"
):
    """
    Función flexible para graficar barras o líneas con agrupación opcional.

    Parámetros:
    - df: DataFrame de entrada
    - chart_type: "bar" o "line"
    - x_col, y_col: columnas para eje X e Y
    - text_col: columna con valores de texto (por ejemplo, porcentaje)
    - group_col: columna de agrupación para múltiples líneas o barras
    - sort_by: columna para ordenar (opcional)
    - ascending: orden ascendente o descendente
    - x_label, y_label: etiquetas personalizadas para los ejes
    - title: título del gráfico
    - color: nombre de la columna para colorear (o color fijo si no hay agrupamiento)
    - color_sequence: lista de colores para los grupos
    - width, height: dimensiones
    - text_as_percentage: si mostrar el texto como porcentaje
    - text_position: posición del texto
    - barmode: modo de barra: "group", "stack", etc.
    - category_order: list with the order that you want for each axes
    """

    df_plot = df.copy()
    if sort_by:
        df_plot = df_plot.sort_values(by=sort_by, ascending=ascending)

    labels = {x_col: x_label or x_col, y_col: y_label or y_col}

    color_args = dict()
    if group_col:
        color_args["color"] = group_col
    elif isinstance(color, str):
        color_args["color"] = color
    else:
        color_args["color_discrete_sequence"] = [color or "#b35151"]

    if color_sequence:
        color_args["color_discrete_sequence"] = color_sequence

    if chart_type == "bar":
        fig = px.bar(
            df_plot,
            x=x_col,
            y=y_col,
            text=text_col if text_col else None,
            labels=labels,
            **color_args
        )
        fig.update_layout(barmode=barmode)

    elif chart_type == "line":
        fig = px.line(
            df_plot,
            x=x_col,
            y=y_col,
            text=text_col if text_col else None,
            labels=labels,
            markers=True,
            **color_args
        )
    else:
        raise ValueError("chart_type debe ser 'bar' o 'line'.")

    # Layout general
    fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        gridcolor='lightgray',
        title=labels[x_col],
        tickfont=dict(size=12, family='Arial', color='black'),
        title_font=dict(size=19, family='Arial', color='black'),
        title_standoff=28,
        categoryorder="array",
        categoryarray=category_order
    ),
    yaxis=dict(
        gridcolor='lightgray',
        title=labels[y_col],
        tickfont=dict(size=14, family='Arial', color='black'),
        title_font=dict(size=19, family='Arial', color='black')
    ),
    width=width,
    height=height,
    title=dict(
        text=title,
        font=dict(size=19, family='Arial', color='black')
    )
)


    # Texto sobre barras o puntos
    if text_col:
        fig.update_traces(
            texttemplate='%{text:.0%}' if text_as_percentage else '%{text}',
            textposition=text_position,
            textfont=dict(size=18)
        )

    fig.show()


#--------------------------------------------------------------------------------------------------------------------------------------------


def heatmap_categoricas_vs_numerica(
    df, 
    cat_x, 
    cat_y, 
    num_var, 
    colorscale,
    width,
    height,
    title=None,
    x_title=None,
    y_title=None
):
    """
    Genera un heatmap que muestra la agregación de una variable numérica
    para cada combinación de dos variables categóricas.

    Parámetros:
    - df: DataFrame con los datos.
    - cat_x: nombre columna categórica para eje X.
    - cat_y: nombre columna categórica para eje Y.
    - num_var: nombre columna numérica para agregar.
    - colorscale: escala de colores para el heatmap.
    - width: Width of the figure in pixels.
    - height: Height of the figure in pixels.
    - title: título del gráfico (string).
    """


    
    
    fig = go.Figure(data=go.Heatmap(
    z=df[num_var],
    x=df[cat_x],
    y=df[cat_y],
    colorscale=    colorscale,
    text=df[num_var],                  
    texttemplate="%{text:.2f}",            
    textfont={"size": 12, "color": "white"} , zmin=0, zmax=df[num_var].quantile(0.95)), )
    

    fig.update_layout(
        title=title,
        width=width, 
        height=height,
        xaxis_title=x_title or cat_x,
        yaxis_title=y_title or cat_y
        
    )

    fig.show()



#--------------------------------------------------------------------------------------------------------------------------------------------



def plot_purpose_treemap(df, status_column='approved', purpose_column='purpose', custom_stopwords_list=None):
    # Si no se provee una lista de stopwords, se usa la global o una vacía.
    global custom_stopwords
    if custom_stopwords_list is not None:
        current_stopwords = custom_stopwords_list
    elif 'custom_stopwords' in globals() and isinstance(custom_stopwords, list):
        current_stopwords = custom_stopwords
    else:
        current_stopwords = [] # Default to empty list if not provided and not globally defined

    # Separar por estado
    df_approved = df[df[status_column] == 1]
    df_rejected = df[df[status_column] == 0]

    def clean_and_tokenize(text_series):
        text = ' '.join(text_series.dropna().astype(str)).lower()
        text = re.sub(r'\d+', '', text)                      # quitar números
        text = re.sub(r'[^\w\s]', '', text)                  # quitar puntuación
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'-', ' ', text).strip()    
        text = re.sub(r'_', ' ', text).strip()             
        tokens = text.split()
        tokens = [word for word in tokens if word not in current_stopwords]
        return tokens

    # Tokenización
    approved_tokens = clean_and_tokenize(df_approved[purpose_column])
    rejected_tokens = clean_and_tokenize(df_rejected[purpose_column])

    # Top palabras
    top_approved = Counter(approved_tokens).most_common(20)
    top_rejected = Counter(rejected_tokens).most_common(20)

    # DataFrames
    df_app = pd.DataFrame(top_approved, columns=['word', 'raw_count'])
    df_app['status'] = 'Approved'

    df_rej = pd.DataFrame(top_rejected, columns=['word', 'raw_count'])
    df_rej['status'] = 'Rejected'

    # Normalizar los conteos (suma 100 en cada grupo)
    if not df_app.empty and df_app['raw_count'].sum() > 0:
        df_app['count'] = df_app['raw_count'] / df_app['raw_count'].sum() * 100
    else:
        df_app['count'] = 0
        
    if not df_rej.empty and df_rej['raw_count'].sum() > 0:
        df_rej['count'] = df_rej['raw_count'] / df_rej['raw_count'].sum() * 100
    else:
        df_rej['count'] = 0

    # Concatenar
    df_all = pd.concat([df_app, df_rej], ignore_index=True)

    # Texto personalizado con porcentaje
    df_all['label'] = df_all['word'] + '<br>' + df_all['count'].round(1).astype(str) + '%'

    # Treemap
    fig = px.treemap(
        df_all,
        path=['status', 'label'],
        values='count',
        color='status',
        color_discrete_map={'Approved': '#c6c0cf', 'Rejected': '#716287'},
        title="Top Words in Loan Purpose (Normalized by Approval State)"
    )

    # Mostrar porcentaje como etiqueta
    fig.update_traces(texttemplate='%{label}',)

    fig.update_layout(width=1100, height=500)

    fig.show()


  
#----------------------------------------------------------------------------------------------------------------------------------------------------



def plot_comparative_distribution_plotly(df_train, df_test, column_name, plot_type='hist',
                                         color_discrete_sequence=None,
                                         bins=None,
                                         title_suffix=" - Train vs Test (Plotly)"):
    """
    Visualiza y compara la distribución de una columna entre df_train y df_test usando Plotly.

    Parámetros:
    -----------
    df_train : pd.DataFrame
        DataFrame de entrenamiento.
    df_test : pd.DataFrame
        DataFrame de prueba.
    column_name : str
        Nombre de la columna a visualizar.
    plot_type : str, opcional ('hist', 'box', 'violin', 'bar')
        Tipo de gráfico a generar:
        - 'hist': Histograma (para numéricas).
        - 'box': Diagrama de caja (para numéricas).
        - 'violin': Diagrama de violín (similar a box, pero muestra densidad).
        - 'bar': Diagrama de barras (para categóricas, cuenta frecuencias).
    color_discrete_sequence : list of str, opcional
        Secuencia de colores discretos de Plotly. Si None, usa ['rgb(0,0,255)', 'rgb(255,0,0)'].
    bins : int, opcional (default=None, Plotly elige automáticamente)
        Número de bins para histogramas (usado si plot_type='hist').
    title_suffix : str, opcional
        Sufijo para el título del gráfico.

    Retorna:
    --------
    plotly.graph_objects.Figure
        El objeto de figura de Plotly.
    """
    if column_name not in df_train.columns or column_name not in df_test.columns:
        print(f"Error: La columna '{column_name}' no existe en ambos DataFrames.")
        return None

    # Definir el mapeo de colores
    if color_discrete_sequence is None or len(color_discrete_sequence) < 2:
        # Azul para Train, Rojo para Test por defecto
        color_discrete_map_param = {'Train': 'rgb(0,0,255)', 'Test': 'rgb(255,0,0)'}
    else:
        color_discrete_map_param = {'Train': color_discrete_sequence[0], 'Test': color_discrete_sequence[1]}

    # Crear copias para no modificar los dataframes originales
    df_train_copy = df_train[[column_name]].copy()
    df_test_copy = df_test[[column_name]].copy()

    df_train_copy['Dataset'] = 'Train'
    df_test_copy['Dataset'] = 'Test'

    combined_df = pd.concat([df_train_copy, df_test_copy], ignore_index=True)

    fig = None
    base_title = f'Comparación de Distribución de {column_name}'
    final_title = base_title + title_suffix

    is_categorical_col = combined_df[column_name].dtype == 'object' or \
                         pd.api.types.is_categorical_dtype(combined_df[column_name])

    if plot_type == 'bar' or (is_categorical_col and plot_type not in ['box', 'violin']):
        train_counts = df_train_copy[column_name].value_counts(normalize=True).reset_index()
        train_counts.columns = [column_name, 'Proportion']
        train_counts['Dataset'] = 'Train'

        test_counts = df_test_copy[column_name].value_counts(normalize=True).reset_index()
        test_counts.columns = [column_name, 'Proportion']
        test_counts['Dataset'] = 'Test'
        
        plot_df = pd.concat([train_counts, test_counts])
        
        all_categories = pd.concat([df_train_copy[column_name], df_test_copy[column_name]]).unique()
        category_order = [cat for cat in df_train_copy[column_name].value_counts().index if cat in all_categories]
        for cat in all_categories:
            if cat not in category_order:
                category_order.append(cat)

        category_order = [c for c in category_order if pd.notna(c)] # Filter out NaN/NaT
        plot_df[column_name] = pd.Categorical(plot_df[column_name], categories=category_order, ordered=True)
        plot_df = plot_df.sort_values(by=[column_name, 'Dataset'])

        fig = px.bar(plot_df, x=column_name, y='Proportion', color='Dataset',
                     title=final_title, barmode='group',
                     color_discrete_map=color_discrete_map_param)
        fig.update_layout(yaxis_title='Proporción')

    elif plot_type == 'hist' and not is_categorical_col:
        fig = px.histogram(combined_df, x=column_name, color='Dataset',
                           title=final_title,
                           nbins=bins,
                           color_discrete_map=color_discrete_map_param,
                           barmode='overlay', 
                           opacity=0.7) 
        fig.update_layout(yaxis_title='Frecuencia')

    elif plot_type == 'box' and not is_categorical_col:
        fig = px.box(combined_df, y=column_name, x='Dataset', color='Dataset',
                     title=final_title,
                     color_discrete_map=color_discrete_map_param,
                     points="outliers")
        fig.update_layout(yaxis_title=column_name, xaxis_title='Dataset')

    elif plot_type == 'violin' and not is_categorical_col:
        fig = px.violin(combined_df, y=column_name, x='Dataset', color='Dataset',
                        title=final_title,
                        box=True,
                        points="outliers",
                        color_discrete_map=color_discrete_map_param)
        fig.update_layout(yaxis_title=column_name, xaxis_title='Dataset')
        
    else:
        print(f"Error: Tipo de gráfico '{plot_type}' no reconocido o no aplicable para el tipo de dato de la columna '{column_name}' (Tipo: {combined_df[column_name].dtype}).")
        return None

    if fig:
        fig.update_layout(
            title_x=0.5, 
            plot_bgcolor='white', 
            xaxis=dict(gridcolor='lightgrey', showline=True, linewidth=1, linecolor='black', mirror=True), 
            yaxis=dict(gridcolor='lightgrey', showline=True, linewidth=1, linecolor='black', mirror=True),
            legend_title_text='Dataset'
        )
    return fig