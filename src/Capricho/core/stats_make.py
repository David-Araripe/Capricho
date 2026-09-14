"""Module containing helper functions for processing repeated elements in a DataFrame"""

from typing import List

import numpy as np
import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer

from ..logger import logger
from .default_fields import multiple_value_cols
from .pandas_helper import aggr_val_series, apply_func_grpd, assign_stats, format_value


def repeated_indices_from_IDs_df(df: pd.DataFrame, columns: list) -> List[List[int]]:
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
    def find_duplicate_index(series: pd.Series) -> List[List[int]]:
        arr = series.to_numpy()
        sidx = np.lexsort(arr.reshape(1, -1))
        sorted_series = series.iloc[sidx]
        sorted_arr = sorted_series.to_numpy()
        # Identify duplicates
        duplicates_mask = np.concatenate(([False], sorted_arr[1:] == sorted_arr[:-1], [False]))
        idx = np.flatnonzero(duplicates_mask[1:] != duplicates_mask[:-1])
        # Extract original indices for duplicates
        sorted_indices = sorted_series.index.tolist()
        return [sorted_indices[i:j] for i, j in zip(idx[::2], idx[1::2] + 1)]

    final_repeat_idxs = find_duplicate_index(concatenated_series)
    return final_repeat_idxs


def repeated_indices_from_array_series(series: pd.Series) -> List[List[int]]:
    """Function to find repeated arrays from a list of arrays"""

    def find_duplicate_index(series: np.array) -> List[List[int]]:
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
        return [sort_idxs[i:j] for i, j in zip(idx[::2], idx[1::2] + 1)]

    final_repeat_idxs = find_duplicate_index(series)
    return final_repeat_idxs


def process_repeat_mols(
    df: pd.DataFrame,
    repeat_element_idxs: List[List[int]],
    solve_strat: str = "keep",
    multiple_value_cols: List[str] = multiple_value_cols,
    extra_id_cols: List[str] = [],
    extra_multival_cols: List[str] = [],
    chirality: bool = False,
    aggregate_mutants: bool = False,
    value_col: str = "pchembl_value",
) -> pd.DataFrame:
    """Process the dataframe according to repeated elements identified
    with the function `find_repeated_arr_from_series`. The standard criteria here
    will be that molecules with the same Fingerprint representation will be treated
    as a single entity, and will have their values aggregated. Upon aggregation, if the
    min & max values differ 1 or more log units, then those samples will be remioved
    from the dataset. Otherwise, values will be aggregated and a new column will be
    assigned, called `might_rancemic`. This column will be a boolean, indicating
    whether the molecule might be rancemic or not.

    Args:
        df: dataframe with the bioactivity data
        repeat_element_idxs: list of indices of repeated elements in the dataframe.
        solve_strat: strategy to solve the repeated elements. If 'drop', then both the
            points within >= 1 log unit difference will be dropped. If 'keep', then
            no values will be dropped.
        extra_id_cols: list of extra identification columns you might have for your own
            compounds that you'd like to use to avoid mixing data & to keep in the final
            dataframe. Defaults to [].
        extra_multival_cols: list of extra columns that you'd like to keep as aggregated
            values in the final dataframe. Caveat: these columns will be displayes as (str)
            separated by `|` (pipe) in the final dataframe. Defaults to [].
        chirality: boolean flag to indicate whether the fingerprints used to check for
            identical compounds is chirality-sensitive or not. Defaults to False

    Returns:
        df: dataframe with the repeated elements processed.
    """
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

    # Will drop the repeats with more than 1 log unit difference
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
    non_aggregated_df = df.drop(index=repeat_subset.index).assign(
        might_rancemic=lambda x: [False] * len(x),
    )
    for col in multival_cols:
        if col in non_aggregated_df.columns and col != value_col:
            non_aggregated_df[col] = non_aggregated_df[col].apply(format_value)

    stats_cols = [f"{value_col}{suffix}" for suffix in ["_mean", "_std", "_median", "_counts"]]
    aggregated_df = updated_df.assign(might_rancemic=lambda x: [True if not chirality else False] * len(x))
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
    canonical_by_smiles = dict(zip(unique_smiles, smiles_canonizer(unique_smiles)))
    df = df.assign(smiles=smiles.map(canonical_by_smiles))

    final_cols = [*id_cols, "smiles", *multival_cols, "might_rancemic", *stats_cols]
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
