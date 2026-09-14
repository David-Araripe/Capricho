"""Module containing helper functions for manipulating pandas DataFrames"""

import functools
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import gmean, gstd, median_abs_deviation

from ..logger import logger
from .default_fields import (
    DATA_DROPPING_COMMENT,
    DATA_PROCESSING_COMMENT,
    SHARED_IDENTIFIER_GROUP,
)


def assign_shared_identifier_groups(
    df: pd.DataFrame,
    key_columns: tuple[str, ...] = ("connectivity", "target_chembl_id"),
) -> pd.DataFrame:
    """Label rows that share a simplified downstream identifier.

    CAPRICHO may intentionally keep several aggregated rows for one connectivity and
    target combination because mutation or user-supplied ID columns differ. Every member
    of such a group receives the same deterministic integer label; singleton rows remain
    missing. Calling ``value_counts()`` on the resulting column therefore reports each
    group's size.

    Args:
        df: Aggregated data to annotate.
        key_columns: Columns defining the simplified downstream identifier.

    Returns:
        A copy with a nullable-integer ``shared_identifier_group`` column.
    """
    missing_columns = [column for column in key_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing shared identifier key columns: {missing_columns}")

    result = df.copy()
    shared_mask = result.duplicated(subset=list(key_columns), keep=False)
    group_labels = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if shared_mask.any():
        group_labels.loc[shared_mask] = (
            result.loc[shared_mask]
            .groupby(list(key_columns), sort=True, dropna=False)
            .ngroup()
            .add(1)
            .astype("Int64")
        )

    if SHARED_IDENTIFIER_GROUP in result.columns:
        result = result.drop(columns=SHARED_IDENTIFIER_GROUP)
    insert_at = result.columns.get_loc("smiles") + 1 if "smiles" in result.columns else len(result.columns)
    result.insert(insert_at, SHARED_IDENTIFIER_GROUP, group_labels)
    return result


def filter_dropping_flags(
    df: pd.DataFrame,
    flags: list[str],
    column: str = DATA_DROPPING_COMMENT,
) -> pd.DataFrame:
    """Filter out rows whose comment column contains any of the specified quality flags.

    Args:
        df: DataFrame to filter.
        flags: List of flag strings to match against.
        column: Name of the comment column to check. Defaults to DATA_DROPPING_COMMENT.

    Returns:
        DataFrame with flagged rows removed.
    """
    if not flags:
        return df

    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame. No filtering applied.")
        return df

    mask = pd.Series(False, index=df.index)
    for flag in flags:
        flag_mask = df[column].fillna("").str.contains(flag, regex=False)
        mask = mask | flag_mask
        n_flagged = flag_mask.sum()
        if n_flagged > 0:
            logger.info(f"Filtering {n_flagged} rows with flag: '{flag}'")

    n_removed = mask.sum()
    if n_removed > 0:
        logger.info(f"Total rows filtered out: {n_removed}/{len(df)} ({n_removed / len(df) * 100:.1f}%)")

    return df[~mask].copy()


def save_dataframe(
    df: pd.DataFrame,
    path: Union[Path, str],
    compression: Optional[str] = "infer",
) -> None:
    """Saves a DataFrame to a file with optional compression.

    This function determines the file format from the file extension and uses
    the appropriate pandas function to save the DataFrame.

    Args:
        df: The DataFrame to be saved.
        path: The file path where the DataFrame will be saved. The file
              extension determines the format (.csv, .tsv, .parquet).
        compression: The compression format to use. For CSV/TSV, the default
            is 'infer', which deduces the compression from the file extension
            (e.g., '.gz', '.zip'). For Parquet, if 'infer' is passed, it
            defaults to 'snappy'. Use None for no compression.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_suffix = path.suffix

    if ".csv" in path.name or ".tsv" in path.name:
        sep = "\t" if ".tsv" in path.name else ","
        # For CSV and TSV, pandas can infer compression from the extension. [3, 4]
        df.to_csv(path, sep=sep, index=False, compression=compression)
    elif ".parquet" in path.name:
        if compression == "infer":
            if path.suffix.endswith(".gz"):
                compression_to_use = "gzip"
            else:
                compression_to_use = "snappy"
                path = path.with_suffix(".parquet")
        else:
            compression_to_use = compression
        df.to_parquet(path, index=False, compression=compression_to_use)
    else:
        raise ValueError(
            f"Unsupported file format: '{file_suffix}'. Supported formats are "
            "'.csv', '.tsv', and '.parquet'."
        )


def conflicting_duplicates(df, key_subset, diff_subset: Optional[list[str]] = None) -> pd.Series:
    """Return a boolean Series like `df.duplicated(...)`, where True marks rows
    that have the same values in `key_subset` but different values in `diff_subset`.
    If diff_subset is None, it behaves like `df.duplicated(subset=key_subset, keep=False)`.

    Args:
        df: pd.DataFrame to check for duplicates
        key_subset: list of columns to check for duplication
        diff_subset: list of columns that must be different within groups of duplicates.

    Returns:
        pd.Series: Boolean Series indicating rows with conflicting duplicates
    """
    if diff_subset is None:
        return df.duplicated(subset=key_subset, keep=False)
    nunique_diff = df.groupby(key_subset)[diff_subset].nunique(dropna=False)
    conflicting_keys = nunique_diff[(nunique_diff > 1).any(axis=1)].index
    mask = df.set_index(key_subset).index.isin(conflicting_keys)
    return pd.Series(mask, index=df.index)


def pchembl_to_molar(pchembl_value: float, unit: str = "nM") -> float:
    """Convert a pChEMBL value to molar units.

    Args:
        pchembl_value: pChEMBL value to be converted
        unit: unit of the pChEMBL value. Defaults to "nM".

    Returns:
        float: pChEMBL value converted to molar units
    """
    if unit == "nM":
        return 10 ** (-pchembl_value) * 10**9
    elif unit in ["uM", "µM"]:
        return 10 ** (6 - pchembl_value) * 10**6
    elif unit == "mM":
        return 10 ** (9 - pchembl_value) * 10**3
    elif unit == "M":
        return 10 ** (9 - pchembl_value)
    else:
        raise ValueError(f"Unit '{unit}' not recognized.")


def format_value(x) -> str:
    """Helper function to format a value to a string with 4 decimal places. This function
    is used to store the original pChEMBL values as strings separated by "|".

    Args:
        x: Value to be formatted

    Returns:
        x: Formatted value
    """
    if isinstance(x, float):
        if x.is_integer() and 1000 <= x <= 9999:
            return f"{int(x)}"  # Return as integer string if it's a year
        return f"{x:.2f}"
    elif isinstance(x, int):
        return str(x)
    else:
        return x


def aggr_val_series(series: pd.Series, sep: str = "|") -> str:
    """Aggregate a pd.Series into a string with values separated by a string (default: pipe)."""
    return sep.join([format_value(x) for x in series])


def get_mad(values) -> Union[float, np.float64]:
    """Calculate the MAD for a list of numerical values. If only one value, return NaN."""
    if len(values) > 1:
        return median_abs_deviation(values)
    else:
        return np.nan


def gmedian(values) -> Union[float, np.float64]:
    """Calculate the median of a list of -log transformed numerical values. If even number
    of values, return the geometric mean of the two middle values.

    Args:
        values: List of -log transformed numerical values

    Returns:
        float: Geometric median of the values, transformed back to the original scale.
    """
    if len(values) % 2 != 0:
        return np.median(values)

    # Even number of elements
    sorted_values = np.sort(values)
    mid_index = len(sorted_values) // 2
    return gmean([values[mid_index - 1], values[mid_index]])


def merge_dataframes(dfs, id_cols) -> pd.DataFrame:
    """
    Merge a list of DataFrames based on id_cols. Useful reference for merges:
    https://stackoverflow.com/questions/53645882/pandas-merging-101

    Args:
        dfs: List of DataFrames to be merged
        id_cols: List of columns to be used as keys for merging

    Returns:
        merged_df: Merged DataFrame
    """
    return functools.reduce(lambda left, right: pd.merge(left, right, on=id_cols, how="inner"), dfs)


def apply_func_grpd(grpd, func: callable, idcols: list, *cols: list) -> pd.DataFrame:
    """Apply a function to a list of columns (*cols) of a grouped dataframe and
    merge the results based on id_cols.

    Args:
        grpd: grouped dataframe
        func: callable function to be applied
        idcols: list of columns to be used as index
    """
    results = []
    for col in cols:
        try:
            results.append(grpd[col].apply(func).reset_index().set_index(idcols).copy())
        except Exception as e:
            logger.error(f"Error applying function to column {col}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            raise e
    return pd.concat(results, ignore_index=False, axis=1).reset_index()


def assign_stats(
    df: pd.DataFrame, sep: str = "|", value_col: str = "pchembl_value", use_geometric: bool = True
) -> pd.DataFrame:
    """Assign statistics to a DataFrame based on a column with multiple values separated by
    a particular separator, e.g. `|` (pipe).

    Args:
        df: pd.DataFrame to be processed.
        sep: string separating the values. Defaults to `|` (pipe).
        value_col: column containing the values to be processed. Defaults to "pchembl_value".
        use_geometric: if True, treats values as -log[unit] and converts them into the original
            scale to calculate the statistics. If False, transformation doesn't take place.
            Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with the statistics assigned. as columns:
    >>> new_cols = [
    >>>     f"{value_col}_mean",
    >>>     f"{value_col}_std",
    >>>     f"{value_col}_median",
    >>>     f"{value_col}_counts",
    >>> ]
    """

    value_series = df[value_col].astype(str).str.split(sep).apply(lambda x: list(map(float, x)))
    new_cols = [f"{value_col}{suffix}" for suffix in ["_mean", "_std", "_median", "_counts"]]

    if use_geometric:
        df[new_cols[0]] = value_series.apply(gmean)
        df[new_cols[1]] = value_series.apply(gstd)
        df[new_cols[2]] = value_series.apply(lambda x: -np.log10(10 ** (-np.median(x))))
        df[new_cols[3]] = value_series.apply(len)
    else:
        df[new_cols[0]] = value_series.apply(np.mean)
        df[new_cols[1]] = value_series.apply(np.std)
        df[new_cols[2]] = value_series.apply(np.median)
        df[new_cols[3]] = value_series.apply(len)

    # For censored measurements, use the first value instead of calculated statistics
    if "standard_relation" in df.columns:
        relation_series = df["standard_relation"].astype(str).str.split(sep)
        # A row is censored if the first relation value is not "="
        # standard_relation is always on the ID array so we can just check the first value
        is_censored = relation_series.apply(lambda x: x[0] != "=")

        if is_censored.any():
            # mean = first value
            df.loc[is_censored, new_cols[0]] = value_series[is_censored].apply(lambda x: x[0])
            df.loc[is_censored, new_cols[1]] = np.nan  # std = NaN
            # median = first value
            df.loc[is_censored, new_cols[2]] = value_series[is_censored].apply(lambda x: x[0])
            # counts stays the same, so we don't modify it

    return df


def find_dict_in_dataframe(df):
    cols_w_dicts = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            logger.info(f"Column '{col}' contains dictionaries.")
            logger.info(
                "Rows with dictionaries: "
                + " ".join(df[df[col].apply(lambda x: isinstance(x, dict))].index.astype(str))
            )
            cols_w_dicts.append(col)
    if cols_w_dicts:
        return cols_w_dicts


def add_comment(
    df: pd.DataFrame,
    comment: str,
    criteria_func: Optional[callable] = None,
    target_column: Optional[str] = None,
    comment_type: Literal["p", "d"] = "d",
) -> pd.DataFrame:
    """Marks rows in a DataFrame based on a given criteria or the entire DataFrame, adding a comment to:
        - `data_dropping_comment` if comment_type == `d` (drop)
        - `data_processing_comment` if comment_type == `p` (process).

    Args:
        df (pd.DataFrame): The input DataFrame.
        comment (str): The comment to add to the comment column for marked rows.
        criteria_func (callable, optional): A function that takes a pandas Series and returns a
            boolean Series. E.g.: pd.isna, lambda x: x == 'invalid', lambda x: x < 0.
            It's required if `target_column` is specified. If `target_column` is None,
            this argument is ignored and the comment is applied to all rows.
        target_column (str, optional): The name of the column to apply the `criteria_func` to.
            If None, the `comment` is applied to all rows of the DataFrame.
        comment_type (Literal["p", "d"]): The type of comment to add.
            'p' for data processing comment, 'd' for data dropping comment. Defaults to 'd'.

    Returns:
        pd.DataFrame: The DataFrame with the specified comment column added/updated.
    """
    if comment_type == "p":
        column_name = DATA_PROCESSING_COMMENT
    elif comment_type == "d":
        column_name = DATA_DROPPING_COMMENT
    else:
        raise ValueError("comment_type must be either 'p' or 'd'.")

    if column_name not in df.columns:  # make sure it exists, initialize empty string
        df[column_name] = ""

    # Preventively fill NaN values in the comment column to avoid "nan" string concatenation
    null_mask = df[column_name].isna()
    df.loc[null_mask, column_name] = ""

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame for comment '{comment}'.")
        if criteria_func is None:
            raise ValueError("criteria_func must be provided when target_column is specified.")

        mask = criteria_func(df[target_column])
    else:  # if no target_column is specified, have a full True mask
        if criteria_func is not None:
            pass
        mask = pd.Series(True, index=df.index)

    df.loc[mask, column_name] = df.loc[mask, column_name].apply(
        lambda x: f"{x} & {comment}" if x else comment
    )
    return df
