import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Default model colors to keep consistency across plots and reports
MODEL_COLOR_MAP: Dict[str, str] = {
    "Baseline": "#1560bd",
    "Weighted": "#75caed",
    "Undersampled": "#8B7EC8",
    "Xgboost": "#d62728",
}


def get_fixed_bin_edges_from_reference(
    reference_series: pd.Series,
    n_bins: int = 10,
    method: str = "quantile",
) -> np.ndarray:
    """
    Compute fixed bin edges from a reference series (typically the training window) to
    be reused across time for stability and decile analyses.

    Parameters
    ----------
    reference_series : pd.Series
        Reference data used to derive the fixed bin edges.
    n_bins : int, default=10
        Number of bins (e.g., deciles if 10).
    method : str, default="quantile"
        If "quantile", uses pd.qcut to derive edges from quantiles.
        If "uniform", uses equal-width bins between min and max.

    Returns
    -------
    np.ndarray
        Sorted unique bin edges with length >= 2.
    """
    ref = reference_series.dropna()
    if ref.empty:
        raise ValueError("Reference series is empty after dropping NaNs; cannot compute edges.")

    if not pd.api.types.is_numeric_dtype(ref):
        raise ValueError("Only numerical variables are supported for fixed-bin decile edges.")

    if method == "quantile":
        try:
            edges = pd.qcut(ref, q=min(n_bins, max(1, ref.nunique() - 1)), retbins=True, duplicates="drop")[1]
        except ValueError:
            edges = np.array([])
    elif method == "uniform":
        edges = np.linspace(ref.min(), ref.max(), n_bins + 1)
    else:
        raise ValueError("method must be either 'quantile' or 'uniform'.")

    edges = np.unique(edges)
    if edges.size < 2:
        # Fallback: synthesize two edges around the single point
        v = ref.iloc[0]
        low = v - 0.5 * abs(v) if v != 0 else -0.5
        high = v + 0.5 * abs(v) if v != 0 else 0.5
        if low == high:
            high = low + 1.0
        edges = np.array([low, high])

    return edges


def assign_deciles_from_edges(
    series: pd.Series,
    bin_edges: np.ndarray,
    ascending: bool = True,
    decile_col_name: str = "decile",
) -> pd.Series:
    """
    Assign fixed-bin deciles (1..N) to a numeric series using previously computed edges.

    Parameters
    ----------
    series : pd.Series
        Numeric series to bin.
    bin_edges : np.ndarray
        Fixed bin edges, as returned by get_fixed_bin_edges_from_reference.
    ascending : bool, default=True
        If True, lower values map to lower deciles (1 = lowest values).
        If False, higher values map to lower deciles (1 = highest values).
    decile_col_name : str, default="decile"
        Only used for naming consistency when creating a Series.

    Returns
    -------
    pd.Series
        Integer decile labels in [1..N], where N = len(bin_edges) - 1.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("assign_deciles_from_edges expects a numeric series.")

    binned = pd.cut(series, bins=np.unique(bin_edges), include_lowest=True, duplicates="drop")
    # cat.codes: -1 for NaN; we will keep NaN as NaN in deciles
    codes = binned.cat.codes
    n_bins = binned.cat.categories.size
    if n_bins == 0:
        return pd.Series(index=series.index, dtype="float")

    # Map to 1..n
    if ascending:
        deciles = codes.replace({-1: np.nan}) + 1
    else:
        # Reverse: 1 corresponds to the highest interval
        deciles = (n_bins - 1 - codes).replace({n_bins: np.nan}) + 1
        deciles = deciles.replace({(n_bins + 1): np.nan})

    return pd.Series(deciles, index=series.index, name=decile_col_name)


def compute_decile_distribution_by_period(
    df: pd.DataFrame,
    score_col: str,
    period_col: str,
    bin_edges: np.ndarray,
    decile_col_name: str = "decile",
    normalize: bool = True,
    dropna_scores: bool = True,
) -> pd.DataFrame:
    """
    Compute decile distributions by period using fixed bin edges.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    score_col : str
        Column with the numeric score (or PD) to bin.
    period_col : str
        Column with time periods (e.g., month string or datetime).
    bin_edges : np.ndarray
        Fixed bin edges, usually computed from a reference period.
    decile_col_name : str, default="decile"
        Name for the decile column in the output.
    normalize : bool, default=True
        If True, also compute per-period proportions (summing to 1 per period).
    dropna_scores : bool, default=True
        If True, drop rows with missing scores before binning.

    Returns
    -------
    pd.DataFrame
        Columns: [period_col, decile_col_name, 'count', 'proportion' (if normalize)]
    """
    work = df[[period_col, score_col]].copy()
    if dropna_scores:
        work = work.dropna(subset=[score_col])

    work[decile_col_name] = assign_deciles_from_edges(work[score_col], bin_edges)

    grp = work.groupby([period_col, decile_col_name]).size().reset_index(name="count")

    if normalize:
        totals = grp.groupby(period_col)["count"].transform("sum")
        grp["proportion"] = np.where(totals > 0, grp["count"] / totals, 0.0)

    # Ensure deterministic ordering: period ascending, decile ascending
    grp = grp.sort_values(by=[period_col, decile_col_name]).reset_index(drop=True)
    return grp


def compute_psi_over_time(
    df: pd.DataFrame,
    score_col: str,
    period_col: str,
    reference_period: Optional[object] = None,
    bin_edges: Optional[np.ndarray] = None,
    n_bins: int = 10,
    clip_perc: float = 1e-6,
    dropna_scores: bool = True,
) -> Tuple[pd.DataFrame, Dict[object, pd.DataFrame], np.ndarray, object]:
    """
    Compute PSI of a score across periods using fixed bins derived from a reference period
    (unless bin_edges are provided). Internally reuses your existing eda_utils.calculate_psi.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the score and period columns.
    score_col : str
        Column name of the numeric score to monitor.
    period_col : str
        Column name of the period (e.g., 'vintage' or 'month').
    reference_period : optional
        Period to use as reference. If None, the earliest (sorted) period is used.
    bin_edges : optional
        Precomputed bin edges. If None, edges are derived from the reference period via quantiles.
    n_bins : int, default=10
        Number of bins (if deriving edges).
    clip_perc : float, default=1e-6
        Clipping value passed to calculate_psi to avoid log(0).
    dropna_scores : bool, default=True
        If True, drop NaNs in score.

    Returns
    -------
    psi_over_time : pd.DataFrame
        Columns: ['period', 'psi', 'n_expected', 'n_actual'].
    psi_details_by_period : Dict[period, pd.DataFrame]
        Detailed PSI contributions per bin for each period.
    edges : np.ndarray
        Bin edges used.
    reference_period_used : object
        The period actually used as reference.
    """
    # Local import to avoid circular dependency at module import time
    from eda_utils import calculate_psi  # type: ignore

    work = df[[period_col, score_col]].copy()
    if dropna_scores:
        work = work.dropna(subset=[score_col])

    # Sort periods safely whether they are datetime-like or strings
    periods = work[period_col].dropna().unique()
    try:
        # Try to sort as datetimes first
        periods_sorted = sorted(periods, key=lambda x: pd.to_datetime(str(x)))
    except Exception:
        periods_sorted = sorted(periods)

    ref_period_used = reference_period if reference_period is not None else periods_sorted[0]

    ref_mask = work[period_col] == ref_period_used
    ref_series = work.loc[ref_mask, score_col]
    if ref_series.empty:
        raise ValueError(f"Reference period '{ref_period_used}' has no non-null scores.")

    # Determine bin edges if not provided
    if bin_edges is None:
        bin_edges = get_fixed_bin_edges_from_reference(ref_series, n_bins=n_bins, method="quantile")

    psi_rows: List[Dict] = []
    details: Dict[object, pd.DataFrame] = {}

    for p in periods_sorted:
        act_series = work.loc[work[period_col] == p, score_col]
        total_psi, psi_df = calculate_psi(
            expected_series=ref_series,
            actual_series=act_series,
            bins=np.unique(bin_edges),
            variable_type="numerical",
            clip_perc=clip_perc,
        )
        details[p] = psi_df.copy()
        psi_rows.append(
            {
                "period": p,
                "psi": float(total_psi) if pd.notnull(total_psi) else np.nan,
                "n_expected": int(ref_series.shape[0]),
                "n_actual": int(act_series.shape[0]),
            }
        )

    psi_over_time = pd.DataFrame(psi_rows)
    return psi_over_time, details, np.unique(bin_edges), ref_period_used
