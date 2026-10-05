"""Module containing helper functions for processing repeated elements in a DataFrame"""

import numpy as np
import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer

from ..logger import logger
from .default_fields import multiple_value_cols
from .pandas_helper import aggr_val_series, apply_func_grpd, assign_stats, format_value


def repeated_indices_from_IDs_df(df: pd.DataFrame, columns: list) -> list[list[int]]:
    """Find repeated indices for given columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to search for repeats.
        columns (list): List of column names with external (non-chembl) IDs to identify
            repeats across. E.g.: ["JUMP_ID", "target_chembl_id"]

    Returns:
        list: A list of lists containing indices of repeated rows based on specified columns.
    """
    # Validate input
    if not all(col in df for col in columns):
        raise ValueError("One or more specified columns do not exist in the DataFrame.")

    # Concatenate values of specified columns into a single series
    concatenated_series = df[columns].astype(str).agg("-".join, axis=1)

    # Find duplicates using numpy operations
    def find_duplicate_index(series: pd.Series) -> list[list[int]]:
        arr = series.to_numpy()
        sidx = np.lexsort(arr.reshape(1, -1))
        sorted_series = series.iloc[sidx]
        sorted_arr = sorted_series.to_numpy()
        # Identify duplicates
        duplicates_mask = np.concatenate(([False], sorted_arr[1:] == sorted_arr[:-1], [False]))
        idx = np.flatnonzero(duplicates_mask[1:] != duplicates_mask[:-1])
        # Extract original indices for duplicates
        sorted_indices = sorted_series.index.tolist()
        return [sorted_indices[i:j] for i, j in zip(idx[::2], idx[1::2] + 1, strict=True)]

    final_repeat_idxs = find_duplicate_index(concatenated_series)
    return final_repeat_idxs


def repeated_indices_from_array_series(series: pd.Series) -> list[list[int]]:
    """Function to find repeated arrays from a list of arrays"""

    def find_duplicate_index(series: np.array) -> list[list[int]]:
        """Group indices of duplicate rows
        From https://stackoverflow.com/a/46629623

        Args:
            arr (numpy array): array of fingerprints

        Returns:
            list[list[int]]: list of lists of indices of duplicate rows
        """
        # Sort by rows
        arr = np.vstack(series)
        sidx = np.lexsort(arr.T)
        b = arr[sidx]
        # Get unique row mask
        m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False]))
        # Get start and stop indices for each group of duplicates
        idx = np.flatnonzero(m[1:] != m[:-1])
        # Get sorted indices
        sort_idxs = series.index[sidx].tolist()
        # Return list of lists of indices of duplicate rows
        return [sort_idxs[i:j] for i, j in zip(idx[::2], idx[1::2] + 1, strict=True)]

    final_repeat_idxs = find_duplicate_index(series)
    return final_repeat_idxs


def process_repeat_mols(
    df: pd.DataFrame,
    repeat_element_idxs: list[list[int]],
    solve_strat: str = "keep",
    multiple_value_cols: list[str] = multiple_value_cols,
    extra_id_cols: list[str] | None = None,
    extra_multival_cols: list[str] | None = None,
    chirality: bool = False,
    aggregate_mutants: bool = False,
    value_col: str = "pchembl_value",
) -> pd.DataFrame:
    """Aggregate repeated readouts while retaining their measurement-level values.

    The caller supplies groups of row indices identified by the selected compound
    equality method. Within these groups, readouts are aggregated separately by
    target, standard relation, extra identification columns, and (unless
    aggregate_mutants=True) mutation. Measurement-level values and metadata are
    retained as pipe-separated strings alongside summary statistics.

    With solve_strat='keep' (the default, used by the CLI), divergent measurements
    are retained even when their range is >= 1. Users can inspect the retained
    values and statistics and decide what to filter during downstream preparation.
    The range is logged; logging it does not remove or flag measurements here.

    When chirality=False, the output includes might_be_racemic. This boolean column
    indicates whether the row may contain readouts from different stereoisomers
    because stereochemistry was ignored during aggregation. It is True for rows
    processed as repeats and False for other rows. The column is omitted when
    chirality=True.

    Args:
        df: DataFrame containing bioactivity data.
        repeat_element_idxs: Groups of repeated row index labels identified by the caller.
        solve_strat: Repeat-handling strategy. 'keep' retains divergent measurements;
            'drop' is a legacy opt-in removal mode. Defaults to 'keep'.
        multiple_value_cols: Columns to retain as pipe-separated measurement-level values,
            when present in df. Defaults to the package's multivalue column list.
        extra_id_cols: Additional grouping columns to keep readouts separate and retain
            in the output. Defaults to [].
        extra_multival_cols: Additional columns to retain as pipe-separated strings.
            Defaults to [].
        chirality: Whether the caller's compound matching preserves stereochemistry.
            Also controls stereochemistry in the canonicalized output SMILES and
            inclusion of might_be_racemic. Defaults to False.
        aggregate_mutants: Whether to combine mutations within a group instead of
            treating mutation as a grouping column. Defaults to False.
        value_col: Measurement column to aggregate and summarize. Defaults to
            'pchembl_value'. Its summaries use a geometric mean; other columns use
            an arithmetic mean.

    Returns:
        DataFrame containing aggregated and singleton rows, retained measurement-level
        values, canonicalized SMILES, and mean, standard deviation, median, and count
        columns for value_col.
    """
    if extra_multival_cols is None:
        extra_multival_cols = []
    if extra_id_cols is None:
        extra_id_cols = []
    df = df.copy()
    repeat_mapping = {}
    for idx in range(len(repeat_element_idxs)):
        for i in repeat_element_idxs[idx]:
            repeat_mapping[i] = idx
    df = df.assign(repeat_mapping=lambda x: x.index.map(repeat_mapping))
    repeat_subset = df.query("~repeat_mapping.isna()").assign(
        **{value_col: lambda df, vc=value_col: df[vc].apply(lambda val: format_value(val))}
    )
    if repeat_subset.empty:
        logger.info(
            "Multiple readouts on the same compound not found within the dataset. Statistics "
            "columns (counts, mean, median) will be calculated solely for the sake of consistency."
        )
        activity_divergence_series = pd.Series(dtype=float)
    else:
        numeric_activity = (
            # concatenate grouped values and convert to numeric arrays
            repeat_subset.groupby(["repeat_mapping"])[value_col]
            .apply(lambda x: "|".join(x))
            .str.split("|")
            .apply(lambda x: np.array(x).astype(float))
        )
        max_series = numeric_activity.apply(lambda x: np.max(x))
        min_series = numeric_activity.apply(lambda x: np.min(x))
        activity_divergence_series = max_series - min_series

    # Diagnose ranges >= 1 in value_col units; removal is opt-in via solve_strat='drop'.
    high_diff_repeats = np.where(activity_divergence_series >= 1)[0]
    points_dropped = len(repeat_subset["repeat_mapping"].isin(high_diff_repeats))
    logger.info(f"Found {len(high_diff_repeats)} repeats with more than 1 log unit difference.")
    if not activity_divergence_series.empty:
        logger.info(f"Maximum difference between min & max values: {activity_divergence_series.max()}")
    if solve_strat == "drop":
        logger.info(f"{points_dropped} points will be removed from the dataset")

    id_cols = [*extra_id_cols, "repeat_mapping", "target_chembl_id", "standard_relation"]

    if value_col == "standard_type":
        id_cols.append("standard_units")

    # Build multivalue columns: include value_col and all multivalue columns except
    # pchembl_value when it's not the value_col (avoid aggregating unnecessary NaN values)
    effective_multival_cols = []
    for col in multiple_value_cols:
        if col == "pchembl_value" and value_col != "pchembl_value":
            # Skip pchembl_value when aggregating on other columns (e.g., standard_value)
            pass
        elif col == value_col:
            # Always include the value_col for aggregation
            effective_multival_cols.append(col)
        elif col not in id_cols and col in df.columns:
            # Include other multivalue columns
            effective_multival_cols.append(col)
    multival_cols = [*effective_multival_cols, *extra_multival_cols]

    if aggregate_mutants:
        multival_cols = [*multival_cols, "mutation"]
    else:
        id_cols = [*id_cols, "mutation"]
    multival_data = repeat_subset[multival_cols].astype(object)
    repeat_subset[multival_cols] = multival_data.where(multival_data.notna(), np.nan)

    if pd.__version__ >= "1.5.0":
        grouped = repeat_subset.groupby(id_cols, group_keys=True)
    else:
        grouped = repeat_subset.groupby(id_cols)

    try:
        updated_vals = apply_func_grpd(grouped, aggr_val_series, id_cols, *multival_cols)
    except TypeError as e:
        logger.error(
            f"Error while aggregating values for columns {multival_cols}. "
            f"Data shouldn't contain NaN values, check the columns: {df.isna().sum()}"
        )
        raise e

    updated_df = assign_stats(
        updated_vals, value_col=value_col, use_geometric=(value_col == "pchembl_value")
    ).merge(df, on=id_cols, how="left")
    rename_cols = {c: c.rstrip("_x") for c in updated_df.columns if c.endswith("_x")}
    todrop_cols = [c for c in updated_df.columns if c.endswith("_y")]
    updated_df = updated_df.drop(columns=todrop_cols).rename(columns=rename_cols)
    todrop_processed = updated_df["repeat_mapping"].isin(high_diff_repeats).index
    smiles_canonizer = ChemStandardizer(
        method="canon", from_smi=True, n_jobs=8, progress=True, isomeric=chirality, chunk_size=None
    )
    if solve_strat == "drop":
        updated_df = updated_df.drop(index=todrop_processed)

    # Convert multival_cols (except value_col) to strings for non-aggregated rows
    non_aggregated_df = df.drop(index=repeat_subset.index)
    if not chirality:
        non_aggregated_df = non_aggregated_df.assign(might_be_racemic=False)
    for col in multival_cols:
        if col in non_aggregated_df.columns and col != value_col:
            non_aggregated_df[col] = non_aggregated_df[col].apply(format_value)

    stats_cols = [f"{value_col}{suffix}" for suffix in ["_mean", "_std", "_median", "_counts"]]
    aggregated_df = updated_df
    if not chirality:
        aggregated_df = aggregated_df.assign(might_be_racemic=True)
    if updated_df.empty:
        single_values = pd.to_numeric(non_aggregated_df[value_col], errors="coerce")
        non_aggregated_df[stats_cols[0]] = single_values
        non_aggregated_df[stats_cols[1]] = np.nan
        non_aggregated_df[stats_cols[2]] = single_values
        non_aggregated_df[stats_cols[3]] = 1
        df = non_aggregated_df.reset_index(drop=True)
    elif non_aggregated_df.empty:
        df = aggregated_df.reset_index(drop=True)
    else:
        df = pd.concat([non_aggregated_df, aggregated_df], ignore_index=True)
    # the `smiles` column will be the final smiles column; to be used for modeling
    smiles = df["standard_smiles"].apply(
        lambda smi: smi if pd.isna(smi) or "|" not in smi else smi.split("|")[0]
    )
    logger.info("Canonicalizing smiles...")
    unique_smiles = smiles.drop_duplicates().tolist()
    canonical_by_smiles = dict(zip(unique_smiles, smiles_canonizer(unique_smiles), strict=True))
    df = df.assign(smiles=smiles.map(canonical_by_smiles))

    stereo_warning_cols = [] if chirality else ["might_be_racemic"]
    final_cols = [*id_cols, "smiles", *multival_cols, *stereo_warning_cols, *stats_cols]
    final_cols.pop(final_cols.index("repeat_mapping"))  # remove repeat_mapping from final_cols
    df = (
        df[final_cols]
        .rename(columns={"standard_smiles": "processed_smiles"})
        .reset_index(drop=True)
        .drop_duplicates()
    )
    logger.info(f"Final number of points: {len(df)}")
    # Also add the single-read points to the mean / median / counts values.
    # Convert explicitly instead of relying on pandas' version-dependent silent downcasting.
    fallback_values = pd.to_numeric(df[value_col], errors="coerce")
    for suffix in ("median", "mean"):
        column = f"{value_col}_{suffix}"
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fallback_values)
    counts_column = f"{value_col}_counts"
    df[counts_column] = pd.to_numeric(df[counts_column], errors="coerce").fillna(1).astype("int64")
    df[value_col] = df[value_col].apply(format_value)  # convert to str for consistency
    return df
