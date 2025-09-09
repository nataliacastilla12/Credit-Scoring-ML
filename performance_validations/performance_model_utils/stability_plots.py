import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Default model colors for consistency across reports
MODEL_COLOR_MAP: Dict[str, str] = {
    "Baseline": "#1560bd",
    "Weighted": "#75caed",
    "Undersampled": "#8B7EC8",
    "Xgboost": "#d62728",
}


def _sorted_periods_unique(values: pd.Series) -> List:
    """
    Safely sort period labels whether they are datetime-like or strings.
    Returns a list of unique sorted period values.
    """
    unique_vals = pd.Series(values).dropna().unique()
    try:
        return sorted(unique_vals, key=lambda x: pd.to_datetime(str(x)))
    except Exception:
        return sorted(unique_vals)


def _apply_white_theme(fig: go.Figure, width: int, height: int) -> go.Figure:
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, gridcolor="lightgrey"),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, gridcolor="lightgrey"),
        legend_title_text="",
        title_x=0.5,
    )
    return fig


def plot_psi_over_time(
    psi_df: pd.DataFrame,
    period_col: str = "period",
    psi_col: str = "psi",
    model_col: Optional[str] = None,
    title: str = "Score Stability (PSI) Over Time",
    width: int = 1000,
    height: int = 500,
    colors: Optional[Dict[str, str]] = None,
    thresholds: tuple = (0.10, 0.25),
) -> go.Figure:
    """
    Plot PSI over time. Supports a single series or multiple series if a model column is provided.

    Parameters
    ----------
    psi_df : pd.DataFrame
        DataFrame with at least [period_col, psi_col]. If model_col is provided, it should exist too.
    period_col : str, default='period'
    psi_col : str, default='psi'
    model_col : str or None, default=None
        If provided, multiple lines will be plotted and colored by this column using the given colors map.
    title : str
    width, height : int
    colors : dict or None
        Mapping from model name to hex color. Defaults to MODEL_COLOR_MAP when model_col is not None.
    thresholds : tuple(float, float)
        Horizontal lines for PSI interpretation (e.g., 0.10 and 0.25).
    """
    df = psi_df.copy()
    if period_col not in df.columns or psi_col not in df.columns:
        raise ValueError("psi_df must contain the specified period_col and psi_col.")

    # Sort by period for stable plotting
    periods_sorted = _sorted_periods_unique(df[period_col])
    df[period_col] = pd.Categorical(df[period_col], categories=periods_sorted, ordered=True)
    df = df.sort_values([period_col] + ([model_col] if model_col and model_col in df.columns else []))

    color_map = colors or (MODEL_COLOR_MAP.copy() if model_col else None)

    if model_col and model_col in df.columns:
        fig = px.line(
            df,
            x=period_col,
            y=psi_col,
            color=model_col,
            markers=True,
            color_discrete_map=color_map,
            title=title,
        )
    else:
        # Single line
        fig = px.line(
            df,
            x=period_col,
            y=psi_col,
            markers=True,
            title=title,
        )
        # Use Baseline color by default
        fig.update_traces(line=dict(color=MODEL_COLOR_MAP["Baseline"]))

    # Add threshold lines
    y0, y1 = thresholds
    x0 = 0
    fig.add_hline(y=y0, line_dash="dash", line_color="#999999", annotation_text=f"PSI {y0:.2f}", annotation_position="top left")
    fig.add_hline(y=y1, line_dash="dash", line_color="#666666", annotation_text=f"PSI {y1:.2f}", annotation_position="bottom left")

    # Adjust y range slightly above max psi
    max_psi = float(np.nanmax(df[psi_col].values)) if not df[psi_col].empty else 0.3
    ymax = max(max_psi * 1.2, thresholds[1] * 1.1)

    fig.update_yaxes(range=[0, ymax])
    fig = _apply_white_theme(fig, width, height)
    return fig


def plot_decile_distribution_heatmap_by_period(
    decile_df: pd.DataFrame,
    period_col: str = "period",
    decile_col: str = "decile",
    value_col: str = "proportion",
    title: str = "Decile Distribution by Period (Heatmap)",
    width: int = 1200,
    height: int = 500,
    colorscale: str = "Blues",
    show_text: bool = False,
    decimals: int = 2,
    decile_ascending: bool = True,
) -> go.Figure:
    """
    Plot a heatmap of decile proportions by period using the output of
    stability_utils.compute_decile_distribution_by_period.

    If value_col is not present but 'count' is, the function computes per-period proportions.
    """
    df = decile_df.copy()
    for col in [period_col, decile_col]:
        if col not in df.columns:
            raise ValueError(f"decile_df must contain column '{col}'.")

    # Ensure value column exists
    if value_col not in df.columns:
        if "count" in df.columns:
            # Normalize by period
            totals = df.groupby(period_col)["count"].transform("sum")
            df[value_col] = np.where(totals > 0, df["count"] / totals, 0.0)
        else:
            raise ValueError(f"'{value_col}' not found and 'count' not available to compute it.")

    # Clean decile labels to integers if possible
    if pd.api.types.is_float_dtype(df[decile_col]) or pd.api.types.is_object_dtype(df[decile_col]):
        try:
            df[decile_col] = df[decile_col].astype(float).astype(int)
        except Exception:
            pass

    periods_sorted = _sorted_periods_unique(df[period_col])
    df[period_col] = pd.Categorical(df[period_col], categories=periods_sorted, ordered=True)

    # Pivot for heatmap (y=decile, x=period)
    pivot = df.pivot_table(index=decile_col, columns=period_col, values=value_col, fill_value=0.0)

    # Sort deciles
    pivot = pivot.sort_index(ascending=decile_ascending)

    fig = px.imshow(
        pivot,
        color_continuous_scale=colorscale,
        aspect="auto",
        origin="lower" if decile_ascending else "upper",
        labels=dict(color="Proportion"),
        title=title,
        text_auto=f".{decimals}f" if show_text else False,
    )

    # Ensure proper tick labels
    fig.update_yaxes(title_text=decile_col)
    fig.update_xaxes(title_text=period_col)

    fig = _apply_white_theme(fig, width, height)
    return fig


def plot_decile_distribution_stacked_area(
    decile_df: pd.DataFrame,
    period_col: str = "period",
    decile_col: str = "decile",
    value_col: str = "proportion",
    title: str = "Decile Distribution by Period (Stacked Area)",
    width: int = 1200,
    height: int = 500,
    color_sequence: Optional[List[str]] = None,
) -> go.Figure:
    """
    Plot stacked area of decile proportions by period. Useful to visualize how the
    share of each decile changes over time. This is not using model color mapping; deciles
    are categorical layers and are better represented with a sequential palette.
    """
    df = decile_df.copy()
    for col in [period_col, decile_col]:
        if col not in df.columns:
            raise ValueError(f"decile_df must contain column '{col}'.")

    if value_col not in df.columns:
        if "proportion" in df.columns:
            value_col = "proportion"
        elif "count" in df.columns:
            totals = df.groupby(period_col)["count"].transform("sum")
            df[value_col] = np.where(totals > 0, df["count"] / totals, 0.0)
        else:
            raise ValueError("Neither 'proportion' nor 'count' available to compute stacked area.")

    # Coerce decile to ordered categorical for consistent legend
    try:
        df[decile_col] = df[decile_col].astype(float).astype(int)
    except Exception:
        pass

    periods_sorted = _sorted_periods_unique(df[period_col])
    df[period_col] = pd.Categorical(df[period_col], categories=periods_sorted, ordered=True)

    # Define color sequence if not provided
    if color_sequence is None:
        color_sequence = px.colors.sequential.Blues
        # Use a longer sequence by cycling if deciles > len palette
        unique_deciles = sorted(df[decile_col].dropna().unique())
        if len(unique_deciles) > len(color_sequence):
            # interpolate by repeating
            reps = int(np.ceil(len(unique_deciles) / len(color_sequence)))
            color_sequence = (color_sequence * reps)[: len(unique_deciles)]

    fig = px.area(
        df.sort_values([period_col, decile_col]),
        x=period_col,
        y=value_col,
        color=decile_col,
        line_group=decile_col,
        groupnorm="fraction",  # ensures stacking to 1 if not perfectly normalized
        color_discrete_sequence=color_sequence,
        title=title,
    )

    fig.update_yaxes(title_text="Proportion", range=[0, 1])
    fig.update_xaxes(title_text=period_col)

    fig = _apply_white_theme(fig, width, height)
    return fig
