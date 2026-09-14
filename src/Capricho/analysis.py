"""Analysis utilities for CAPRICHO bioactivity data comparability studies"""

from enum import Enum
from itertools import combinations
from typing import List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log
from matplotlib import colormaps
from scipy import stats

from .chembl.unit_conversions import is_unit_annotation_error_diff

DEFAULT_COV_BAR_COLOR = "#2a78d6"


def r2_score(y_true, y_pred):
    ss_res = np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


class ProcessingComment(str, Enum):
    """Processing comments added during data curation (non-dropping flags).

    Note: UNIT_CONVERTED is a pattern-based comment. Actual comments include the
    original and target units, e.g., "Unit converted to nM from uM".
    """

    CALCULATED_PCHEMBL = "Calculated pChEMBL"
    SALT_SOLVENT_REMOVED = "Salt/solvent removed"
    PCHEMBL_DUPLICATION_ACROSS_DOCUMENTS = "pChEMBL Duplication Across Documents"
    UNIT_CONVERTED = "Unit converted to"  # Example: "Unit converted to nM from uM"


class DroppingComment(str, Enum):
    """Dropping comments indicating data quality issues (dropping flags).

    Note: Assay size and insufficient overlap values shown are examples. Actual thresholds
    depend on user parameters and should be matched using pattern-based functions.
    """

    DATA_VALIDITY_COMMENT = "Data Validity Comment Present"
    POTENTIAL_DUPLICATE = "Potential Duplicate"
    UNDEFINED_STEREOCHEMISTRY = "Undefined Stereochemistry"
    MUTATION_KEYWORD = "Mutation keyword in assay description"
    ASSAY_SIZE_TOO_LARGE = "Assay size >"  # Example: "Assay size > 100"
    ASSAY_SIZE_TOO_SMALL = "Assay size <"  # Example: "Assay size < 20"
    UNIT_ANNOTATION_ERROR = "Unit Annotation Error"
    MISSING_DOCUMENT_DATE = "Missing document date"
    MIXTURE_IN_SMILES = "Mixture in SMILES"
    ACTIVITY_COMMENT_REVIEW = "Activity with exact standard relation and inactivity-like comment"
    INSUFFICIENT_ASSAY_OVERLAP = (
        "Insufficient assay overlap"  # Example: "Insufficient assay overlap (min_overlap=5)"
    )
    INSUFFICIENT_ASSAY_OVERLAP_WITH_METADATA = "Insufficient assay overlap with metadata matching"  # Example: "Insufficient assay overlap with metadata matching (min_overlap=5)"


def normalize_comment_pattern(comment: str) -> str:
    """Normalize comment by removing dynamic values (e.g., threshold numbers).

    Args:
        comment: Raw comment string from data.

    Returns:
        Normalized pattern for matching. For example:
        - "Assay size < 20" -> "Assay size <"
        - "Assay size > 100" -> "Assay size >"
        - "Insufficient assay overlap (min_overlap=5)" -> "Insufficient assay overlap"
        - "Insufficient assay overlap with metadata matching (min_overlap=5)" -> "Insufficient assay overlap with metadata matching"
        - "Unit converted to nM from uM" -> "Unit converted to"
        - "Unit Annotation Error" -> "Unit Annotation Error"
    """
    # Handle assay size patterns by removing the threshold number
    if comment.startswith("Assay size <"):
        return "Assay size <"
    elif comment.startswith("Assay size >"):
        return "Assay size >"
    # Handle insufficient assay overlap patterns by removing the parameter
    elif comment.startswith("Insufficient assay overlap with metadata matching"):
        return "Insufficient assay overlap with metadata matching"
    elif comment.startswith("Insufficient assay overlap"):
        return "Insufficient assay overlap"
    # Handle unit conversion patterns by removing the specific units
    elif comment.startswith("Unit converted to"):
        return "Unit converted to"
    return comment


def extract_assay_threshold(comment: str) -> str:
    """Extract the threshold value from an assay size comment.

    Args:
        comment: Comment like "Assay size < 20" or "Assay size > 100".

    Returns:
        The threshold number as string, or empty string if not found.
    """
    if comment.startswith("Assay size <"):
        return comment.split("<")[1].strip()
    elif comment.startswith("Assay size >"):
        return comment.split(">")[1].strip()
    return ""


def get_all_comments() -> list[str]:
    """Get all comment types in order for plotting.

    Returns:
        List of all comment patterns (dropping and processing combined).
        Assay size, insufficient overlap, and unit conversion comments are patterns
        without specific thresholds/units.
    """
    return [
        DroppingComment.DATA_VALIDITY_COMMENT.value,
        DroppingComment.POTENTIAL_DUPLICATE.value,
        DroppingComment.UNDEFINED_STEREOCHEMISTRY.value,
        DroppingComment.MUTATION_KEYWORD.value,
        DroppingComment.ASSAY_SIZE_TOO_LARGE.value,
        DroppingComment.ASSAY_SIZE_TOO_SMALL.value,
        DroppingComment.INSUFFICIENT_ASSAY_OVERLAP.value,
        DroppingComment.INSUFFICIENT_ASSAY_OVERLAP_WITH_METADATA.value,
        DroppingComment.UNIT_ANNOTATION_ERROR.value,
        DroppingComment.MISSING_DOCUMENT_DATE.value,
        DroppingComment.MIXTURE_IN_SMILES.value,
        DroppingComment.ACTIVITY_COMMENT_REVIEW.value,
        ProcessingComment.SALT_SOLVENT_REMOVED.value,
        ProcessingComment.CALCULATED_PCHEMBL.value,
        ProcessingComment.PCHEMBL_DUPLICATION_ACROSS_DOCUMENTS.value,
        ProcessingComment.UNIT_CONVERTED.value,
    ]


def deaggregate_data(data: pd.DataFrame, sep_str: str = "|") -> pd.DataFrame:
    """De-aggregate data by splitting pipe-delimited columns back into individual rows.

    Takes aggregated data where multiple values are stored in pipe-delimited strings
    and explodes them back into separate rows, effectively reversing the aggregation.
    Only rows containing the separator in at least one column are exploded.

    Args:
        data: DataFrame with potentially aggregated (pipe-delimited) columns.
        sep_str: Separator string used to delimit multiple values in columns.

    Returns:
        DataFrame with de-aggregated data where each measurement is a separate row.
    """
    cols_with_pipe = data.apply(lambda col: col.astype(str).str.contains(sep_str, regex=False)).any()
    cols_with_pipe = np.compress(cols_with_pipe.values, cols_with_pipe.index).tolist()

    if not cols_with_pipe:
        return data.copy()

    first_pipe_col = cols_with_pipe[0]
    mask = data[first_pipe_col].astype(str).str.contains(sep_str, regex=False)
    aggregated_rows = data[mask]

    if len(aggregated_rows) == 0:
        return data.copy()

    aggregated_rows = aggregated_rows.copy()
    for col in cols_with_pipe:
        aggregated_rows[col] = aggregated_rows[col].fillna("").astype(str).str.split(sep_str)

    exploded = aggregated_rows.explode(column=cols_with_pipe)
    deaggregated = pd.concat([data.drop(index=aggregated_rows.index), exploded])

    return deaggregated.reset_index(drop=True)


def deduplicate_aggregated_values(
    data: pd.DataFrame,
    value_column: str = "pchembl_value",
    sep_str: str = "|",
) -> pd.DataFrame:
    """Remove duplicate values within aggregated rows based on the value column.

    When data is aggregated, identical values from different documents may appear
    multiple times (e.g., "8.00|8.00|8.00|6.92"). This function removes duplicate
    values, keeping only the first occurrence of each unique value within each row.

    This is useful for removing cross-document duplicates before recalculating
    statistics (mean, median, std), as duplicate values can skew the distribution.

    Args:
        data: DataFrame with aggregated (pipe-delimited) columns.
        value_column: Column containing the values to deduplicate on.
            Defaults to "pchembl_value".
        sep_str: Separator string used to delimit multiple values in columns.

    Returns:
        DataFrame with deduplicated values. Rows that had duplicates will have
        fewer pipe-delimited entries. Single-value rows are unchanged.
    """
    if len(data) == 0:
        return data.copy()

    data = data.copy()

    # Find columns with pipe separators (multi-value columns)
    cols_with_pipe = data.apply(lambda col: col.astype(str).str.contains(sep_str, regex=False)).any()
    cols_with_pipe = np.compress(cols_with_pipe.values, cols_with_pipe.index).tolist()

    if not cols_with_pipe or value_column not in cols_with_pipe:
        return data

    # Process each row that has aggregated values
    mask = data[value_column].astype(str).str.contains(sep_str, regex=False)

    if not mask.any():
        return data

    def dedupe_row(row):
        """Deduplicate a single row based on value_column, keeping first occurrence."""
        values = str(row[value_column]).split(sep_str)

        # Find indices of first occurrence of each unique value
        seen = {}
        keep_indices = []
        for i, val in enumerate(values):
            if val not in seen:
                seen[val] = i
                keep_indices.append(i)

        # If no duplicates found, return row unchanged
        if len(keep_indices) == len(values):
            return row

        # Apply the same indices to all pipe-delimited columns
        new_row = row.copy()
        for col in cols_with_pipe:
            col_values = str(row[col]).split(sep_str)
            # Handle case where column has fewer values than value_column
            if len(col_values) >= len(values):
                new_values = [col_values[i] for i in keep_indices]
                new_row[col] = sep_str.join(new_values)

        return new_row

    # Apply deduplication to rows with aggregated values
    deduplicated_rows = data.loc[mask].apply(dedupe_row, axis=1)
    data.loc[mask] = deduplicated_rows

    return data


def filter_aggregated_dropping_flags(
    data: pd.DataFrame,
    flags: list[str],
    comment_column: str = "data_dropping_comment",
    value_column: str = "pchembl_value",
    sep_str: str = "|",
) -> pd.DataFrame:
    """Remove individual flagged measurements from aggregated rows and recalculate statistics.

    For aggregated data (pipe-separated columns), this function checks each measurement
    position individually. Only measurements whose comment matches a flag are removed.
    Rows where ALL measurements are flagged are removed entirely. Non-aggregated rows
    are filtered at the row level (same as filter_dropping_flags).

    Args:
        data: DataFrame with potentially aggregated (pipe-delimited) columns.
        flags: List of flag strings to match against (substring match).
        comment_column: Column containing quality flags. Defaults to "data_dropping_comment".
        value_column: Column containing activity values for stats recalculation.
            Defaults to "pchembl_value".
        sep_str: Separator string used to delimit multiple values in columns.

    Returns:
        DataFrame with flagged measurements removed and statistics recalculated.
    """
    from .core.pandas_helper import assign_stats, filter_dropping_flags

    if not flags:
        return data

    if comment_column not in data.columns:
        log.warning(f"Column '{comment_column}' not found in DataFrame. No filtering applied.")
        return data

    if len(data) == 0:
        return data.copy()

    # Detect pipe-separated columns
    cols_with_pipe = data.apply(lambda col: col.astype(str).str.contains(sep_str, regex=False)).any()
    cols_with_pipe = np.compress(cols_with_pipe.values, cols_with_pipe.index).tolist()

    if not cols_with_pipe or comment_column not in cols_with_pipe:
        # Non-aggregated data: fall back to row-level filtering
        return filter_dropping_flags(data, flags, column=comment_column)

    data = data.copy()

    # Per-flag logging (how many rows contain each flag)
    for flag in flags:
        flag_mask = data[comment_column].fillna("").astype(str).str.contains(flag, regex=False)
        n_flagged = flag_mask.sum()
        if n_flagged > 0:
            log.info(f"Rows with flag '{flag}': {n_flagged}")

    # Identify aggregated rows (value_column or comment_column has pipes)
    check_col = value_column if value_column in cols_with_pipe else comment_column
    is_aggregated = data[check_col].astype(str).str.contains(sep_str, regex=False)

    # Track rows to drop entirely
    rows_to_drop = []
    # Track rows that were modified (need stats recalculation)
    rows_modified = []

    for idx in data.index:
        if is_aggregated.loc[idx]:
            # Aggregated row: measurement-level filtering
            comments = str(data.at[idx, comment_column]).split(sep_str)
            keep_indices = []
            for i, comment in enumerate(comments):
                if not _measurement_has_flag(comment, flags):
                    keep_indices.append(i)

            if len(keep_indices) == 0:
                # All measurements flagged -> remove row
                rows_to_drop.append(idx)
            elif len(keep_indices) < len(comments):
                # Partial filtering: keep only clean measurements
                for col in cols_with_pipe:
                    col_values = str(data.at[idx, col]).split(sep_str)
                    if len(col_values) >= len(comments):
                        new_values = [col_values[i] for i in keep_indices]
                        data.at[idx, col] = sep_str.join(new_values)
                rows_modified.append(idx)
            # else: all clean, no change needed
        else:
            # Non-aggregated row: row-level filtering
            comment = str(data.at[idx, comment_column]) if pd.notna(data.at[idx, comment_column]) else ""
            if _measurement_has_flag(comment, flags):
                rows_to_drop.append(idx)

    n_removed = len(rows_to_drop)
    n_pruned = len(rows_modified)
    n_total = n_removed + n_pruned
    if n_removed > 0:
        log.info(f"Rows fully removed (all measurements flagged): {n_removed}")
    if n_pruned > 0:
        log.info(f"Rows with individual measurements pruned: {n_pruned}")
    if n_total > 0:
        log.info(f"Total rows affected: {n_total}/{len(data)} ({n_total / len(data) * 100:.1f}%)")

    data = data.drop(index=rows_to_drop)

    if len(data) == 0:
        return data.reset_index(drop=True)

    # Recalculate stats for modified rows (and all aggregated rows for consistency)
    if rows_modified:
        use_geometric = value_column == "pchembl_value"
        data = assign_stats(data, value_col=value_column, use_geometric=use_geometric)

    return data.reset_index(drop=True)


def _measurement_has_flag(comment: str, flags: list[str]) -> bool:
    """Check if a single measurement's comment matches any of the given flags.

    Uses substring matching, consistent with filter_dropping_flags semantics.

    Args:
        comment: Comment string for a single measurement.
        flags: List of flag patterns to check.

    Returns:
        True if any flag is found in the comment.
    """
    if not comment or comment == "nan":
        return False
    for flag in flags:
        if flag in comment:
            return True
    return False


def resolve_annotation_errors(
    data: pd.DataFrame,
    strategy: str = "first",
    mol_id_col: str = "molecule_chembl_id",
    assay_id_col: str = "assay_chembl_id",
    value_col: str = "pchembl_value",
    year_col: str = "year",
) -> pd.DataFrame:
    """Resolve unit annotation errors by keeping measurements from earliest documents.

    Unit annotation errors occur when measurements for the same molecule across different
    assays differ by exactly 3.0 or 6.0 log units (suggesting 1000x or 1,000,000x unit
    conversion errors). This function detects such pairs and keeps only the measurement
    from the earliest document, assuming later measurements copied the value incorrectly.

    This function operates on exploded (non-aggregated) data where each row is a single
    measurement.

    Args:
        data: DataFrame with exploded bioactivity data (one measurement per row).
        strategy: Resolution strategy. Currently only "first" is supported, which keeps
            the measurement from the earliest document year.
        mol_id_col: Column name for molecule identifiers.
        assay_id_col: Column name for assay identifiers.
        value_col: Column name for activity values (e.g., pchembl_value).
        year_col: Column name for document year.

    Returns:
        DataFrame with annotation errors resolved (problematic measurements removed).
    """
    if len(data) == 0:
        return data.copy()

    if strategy != "first":
        raise ValueError(f"Unknown strategy: {strategy}. Only 'first' is supported.")

    data = data.copy()

    # Convert value column to numeric for comparison
    data["__value_numeric__"] = pd.to_numeric(data[value_col], errors="coerce")

    # Convert year to numeric, treating 'nan' string as NaN
    data["__year_numeric__"] = data[year_col].replace("nan", np.nan)
    data["__year_numeric__"] = pd.to_numeric(data["__year_numeric__"], errors="coerce")

    # Track indices to remove
    indices_to_remove = set()
    pairs_detected = 0

    log.trace(f"Starting annotation error detection for {len(data)} measurements")

    # Group by molecule and find problematic pairs
    for mol_id, group in data.groupby(mol_id_col):
        if len(group) < 2:
            continue

        # Get all pairs of measurements from different assays
        for i, (idx_a, row_a) in enumerate(group.iterrows()):
            for idx_b, row_b in list(group.iterrows())[i + 1 :]:
                # Skip if same assay
                if row_a[assay_id_col] == row_b[assay_id_col]:
                    continue

                # Check if values differ by an exact multiple of 3 log units (3.0, 6.0, 9.0, ...)
                val_a = row_a["__value_numeric__"]
                val_b = row_b["__value_numeric__"]

                if pd.isna(val_a) or pd.isna(val_b):
                    continue

                diff = abs(val_a - val_b)
                if not bool(is_unit_annotation_error_diff(diff)):
                    continue

                pairs_detected += 1

                # Found a problematic pair - resolve by keeping earliest
                year_a = row_a["__year_numeric__"]
                year_b = row_b["__year_numeric__"]
                assay_a = row_a[assay_id_col]
                assay_b = row_b[assay_id_col]

                log.trace(
                    f"Detected annotation error pair for {mol_id}: "
                    f"assay {assay_a} (value={val_a:.2f}, year={year_a}) vs "
                    f"assay {assay_b} (value={val_b:.2f}, year={year_b}), "
                    f"diff={diff:.2f}"
                )

                # Handle NaN years: remove the one with NaN year, keep valid year
                if pd.isna(year_a) and pd.isna(year_b):
                    # Both NaN - can't resolve, remove both
                    log.trace("  -> Both years NaN, removing BOTH measurements")
                    indices_to_remove.add(idx_a)
                    indices_to_remove.add(idx_b)
                elif pd.isna(year_a):
                    # A has NaN year, keep B
                    log.trace(
                        f"  -> Year A is NaN, keeping B (assay {assay_b}, value={val_b:.2f}, year={year_b})"
                    )
                    indices_to_remove.add(idx_a)
                elif pd.isna(year_b):
                    # B has NaN year, keep A
                    log.trace(
                        f"  -> Year B is NaN, keeping A (assay {assay_a}, value={val_a:.2f}, year={year_a})"
                    )
                    indices_to_remove.add(idx_b)
                elif year_a <= year_b:
                    # A is earlier or same, remove B
                    log.trace(
                        f"  -> A is earlier/same ({year_a} <= {year_b}), "
                        f"keeping A (assay {assay_a}, value={val_a:.2f}), removing B"
                    )
                    indices_to_remove.add(idx_b)
                else:
                    # B is earlier, remove A
                    log.trace(
                        f"  -> B is earlier ({year_b} < {year_a}), "
                        f"keeping B (assay {assay_b}, value={val_b:.2f}), removing A"
                    )
                    indices_to_remove.add(idx_a)

    log.trace(
        f"Annotation error detection complete: found {pairs_detected} problematic pairs, "
        f"removing {len(indices_to_remove)} unique measurements"
    )

    # Remove problematic rows
    result = data.drop(index=list(indices_to_remove))

    # Clean up temporary columns
    result = result.drop(columns=["__value_numeric__", "__year_numeric__"])

    return result.reset_index(drop=True)


def recalculate_aggregated_stats(
    data: pd.DataFrame,
    value_column: str = "pchembl_value",
    sep_str: str = "|",
) -> pd.DataFrame:
    """Recalculate statistics (mean, median, std, counts) from pipe-delimited values.

    After deduplication or other modifications to the pipe-delimited value column,
    the pre-computed statistics (mean, median, std, counts) may be stale. This
    function recalculates them from the current pipe-delimited values.

    Args:
        data: DataFrame with aggregated data containing value_column and
            corresponding stats columns ({value_column}_mean, _median, _std, _counts).
        value_column: Column containing pipe-delimited values.
            Defaults to "pchembl_value".
        sep_str: Separator string used to delimit multiple values.

    Returns:
        DataFrame with recalculated statistics columns.
    """
    if len(data) == 0:
        return data.copy()

    data = data.copy()

    # Stats column names
    mean_col = f"{value_column}_mean"
    median_col = f"{value_column}_median"
    std_col = f"{value_column}_std"
    counts_col = f"{value_column}_counts"

    def calc_stats(val_str):
        """Calculate stats from pipe-delimited string."""
        if pd.isna(val_str) or val_str == "":
            return np.nan, np.nan, np.nan, 0

        vals = str(val_str).split(sep_str)
        vals = [float(v) for v in vals if v.strip()]

        if len(vals) == 0:
            return np.nan, np.nan, np.nan, 0
        elif len(vals) == 1:
            return vals[0], vals[0], 0.0, 1
        else:
            return np.mean(vals), np.median(vals), np.std(vals, ddof=1), len(vals)

    # Apply calculation to each row
    stats = data[value_column].apply(lambda x: pd.Series(calc_stats(x)))
    stats.columns = [mean_col, median_col, std_col, counts_col]

    # Update the stats columns
    for col in [mean_col, median_col, std_col, counts_col]:
        if col in data.columns:
            data[col] = stats[col]
        else:
            data[col] = stats[col]

    return data


def explode_assay_comparability(
    subset: pd.DataFrame,
    sep_str: str = "|",
    extra_multival_cols: Optional[list[str]] = None,
    value_column: str = "pchembl_value",
) -> pd.DataFrame:
    """Explode dataset to create pairwise comparisons between assays for the same compound.

    Takes a subset of data where compounds have measurements across multiple assays (indicated
    by separator in columns) and creates all pairwise combinations for comparability analysis.

    Args:
        subset: DataFrame with multi-valued columns separated by sep_str.
        sep_str: Separator string used to delimit multiple values in columns.
        extra_multival_cols: Additional columns to treat as multi-valued.
        value_column: The column containing activity values to compare. Defaults to
            "pchembl_value" but can be set to "standard_value" for non-pChEMBL data
            (e.g., Caco-2 permeability, percent inhibition).

    Returns:
        DataFrame with exploded pairwise comparisons, with _x and _y suffixes for each pair.
    """
    if extra_multival_cols is None:
        extra_multival_cols = []
    elif not isinstance(extra_multival_cols, list):
        raise ValueError("extra_multival_cols must be a list of column names or None")

    singleval_cols = [
        "connectivity",
        "target_chembl_id",
        "source_datapoint_id",
    ]
    multival_cols = [
        "activity_id",
        "assay_chembl_id",
        value_column,
        "data_processing_comment",
        "data_dropping_comment",
        "standard_type",
        "canonical_smiles",
        *extra_multival_cols,
    ]

    # Prevent overlap between singleval and multival columns
    for col in extra_multival_cols:
        if col in singleval_cols:
            singleval_cols.remove(col)

    # Auto-detect which columns are actually multi-valued (contain the separator).
    # Columns used as --id-columns during aggregation will be single-valued.
    # Skip this check for empty DataFrames to preserve original column classification.
    if len(subset) > 0:
        for col in list(multival_cols):
            if col in subset.columns:
                if not subset[col].astype(str).str.contains(sep_str, regex=False).any():
                    multival_cols.remove(col)
                    singleval_cols.append(col)

    # Filter to only include columns that exist in the input dataframe
    singleval_cols = [col for col in singleval_cols if col in subset.columns]
    multival_cols = [col for col in multival_cols if col in subset.columns]

    exploded_subset = subset[
        [
            *singleval_cols,
            *multival_cols,
        ]
    ].apply(lambda x: x.str.split(sep_str) if x.name not in singleval_cols else x)

    for col in multival_cols:
        exploded_subset[col] = exploded_subset[col].apply(
            lambda x: [sep_str.join(y) for y in combinations(x, 2)]
        )

    exploded_subset = exploded_subset.explode(multival_cols)

    # Handle empty DataFrame case
    if len(exploded_subset) == 0:
        # Create empty result with correct columns
        result_cols = [*singleval_cols]
        for col in multival_cols:
            result_cols.extend([f"{col}_x", f"{col}_y"])
        result_cols.extend(["processing_comment", "dropping_comment"])
        return pd.DataFrame(columns=result_cols)

    suffixes = ["_x", "_y"]
    for col in multival_cols:
        values = exploded_subset[col].apply(lambda x: x.split(sep_str)).values
        for idx, s in enumerate(suffixes):
            exploded_subset[f"{col}{s}"] = [v[idx] for v in values]
        exploded_subset.drop(columns=[col], inplace=True)

    # Combine processing and dropping comments from both assays
    exploded_subset = exploded_subset.assign(
        processing_comment=lambda x: x["data_processing_comment_x"].fillna("")
        + sep_str
        + x["data_processing_comment_y"].fillna(""),
        dropping_comment=lambda x: x["data_dropping_comment_x"].fillna("")
        + sep_str
        + x["data_dropping_comment_y"].fillna(""),
    ).query("assay_chembl_id_x != assay_chembl_id_y")

    # Clean up comment strings by removing duplicates and extra separators
    exploded_subset["dropping_comment"] = (
        exploded_subset.dropping_comment.str.rstrip(sep_str)
        .str.lstrip(sep_str)
        .apply(lambda x: sep_str.join(np.unique(x.split(sep_str))))
    )
    exploded_subset["processing_comment"] = (
        exploded_subset.processing_comment.str.rstrip(sep_str)
        .str.lstrip(sep_str)
        .apply(lambda x: sep_str.join(np.unique(x.split(sep_str))))
    )
    return exploded_subset


def format_units_latex(units: str) -> str:
    """Convert common unit patterns to LaTeX format.

    Handles scientific notation patterns like "10^-6 cm/s" and converts them
    to proper LaTeX math notation. Units already containing LaTeX ($) are
    passed through unchanged.

    Args:
        units: Unit string (e.g., "10^-6 cm/s", "nM", "$10^{-6}$ cm/s").

    Returns:
        Unit string with LaTeX math notation where applicable.
    """
    if not units:
        return ""

    # Already contains LaTeX - pass through
    if "$" in units:
        return units

    # Handle 10^N patterns (e.g., "10^-6 cm/s" → "$10^{-6}$ cm/s")
    import re

    pattern = r"10\^(-?\d+)"
    if re.search(pattern, units):
        return re.sub(pattern, r"$10^{\1}$", units)

    return units


def format_axis_label(
    property_name: str = "value",
    log_transform: bool = False,
    units: Optional[str] = None,
) -> str:
    """Format axis label with LaTeX math notation for publication-quality figures.

    Args:
        property_name: Name of the measured property (e.g., "Permeability", "pChEMBL").
        log_transform: If True, prepend -log10 in LaTeX format.
        units: Unit string. Supports patterns like "10^-6 cm/s" which are
            converted to LaTeX. Already-LaTeX units ("$...$") pass through.

    Returns:
        Formatted label string with LaTeX math notation.
    """
    parts = [property_name]

    if log_transform and units:
        # e.g., "Permeability ($-\log_{10}$ [$10^{-6}$ cm/s])"
        formatted_units = format_units_latex(units)
        parts.append(f"($-\\log_{{10}}$ [{formatted_units}])")
    elif log_transform:
        # e.g., "Standard Value ($-\log_{10}$)"
        parts.append(r"($-\log_{10}$)")
    elif units:
        # e.g., "Permeability (10^-6 cm/s)"
        formatted_units = format_units_latex(units)
        parts.append(f"({formatted_units})")

    return " ".join(parts)


def format_title_with_n(title: str, n_pairs: int, sep: str = " ") -> str:
    """Append the number of pairwise assay comparisons to a plot title.

    Each point in a comparability scatter plot is one assay-versus-assay comparison
    for a single compound, not a compound or a raw measurement. The unit is spelled
    out so the count cannot be misread as a compound count.

    Args:
        title: Plot title, possibly empty.
        n_pairs: Number of pairwise comparisons plotted.
        sep: Separator between the title and the sample size. Pass "\\n" to keep long
            multi-panel titles from overlapping their neighbours.

    Returns:
        Title with the sample size appended, e.g. "Cleaned Ki Data (n = 383 pairs)".
    """
    noun = "pair" if n_pairs == 1 else "pairs"
    sample_size = f"n = {n_pairs:,} {noun}"
    return f"{title}{sep}({sample_size})" if title else sample_size


def format_metrics_text(r2: float, rho: float, tau: float) -> str:
    """Format the comparability metrics annotation, omitting undefined values.

    Correlations are undefined for panels holding too few distinct points (a single
    repeated comparison has zero variance). Printing "nan" there reads as a plotting
    failure rather than as a property of the data, so undefined metrics are dropped
    and the panel's sample size in the title carries the explanation.

    Args:
        r2: R² score.
        rho: Spearman rho.
        tau: Kendall tau.

    Returns:
        Newline-separated LaTeX annotation, empty when no metric is defined.
    """
    metrics = [
        (r2, r"$R^2: {:.2f}$"),
        (rho, r"Spearman $\rho: {:.2f}$"),
        (tau, r"Kendall $\tau: {:.2f}$"),
    ]
    return "\n".join(template.format(value) for value, template in metrics if np.isfinite(value))


def _log_comparability_metrics(
    xp: np.ndarray,
    yp: np.ndarray,
    label: str = "",
    is_log_scale: bool = True,
    rho: Optional[float] = None,
    r2: Optional[float] = None,
    tau: Optional[float] = None,
) -> None:
    """Log quantitative comparability metrics for pairwise assay comparisons.

    For log-scale data (pChEMBL or -log10 transformed), reports the fraction of
    pairs within ±0.3 and ±1.0 log units. For all data, reports Spearman rho,
    Kendall tau, and R².

    Args:
        xp: Array of x-values (possibly log-transformed).
        yp: Array of y-values (possibly log-transformed).
        label: Description of the data subset (e.g., flag name).
        is_log_scale: If True, compute ±0.3/±1.0 statistics.
        rho: Pre-computed Spearman rho. Computed from data if None.
        r2: Pre-computed R² score. Computed from data if None.
        tau: Pre-computed Kendall tau. Computed from data if None.
    """
    n = len(xp)
    if n == 0:
        return

    abs_diff = np.abs(np.asarray(xp) - np.asarray(yp))

    try:
        if rho is None:
            rho, _ = stats.spearmanr(xp, yp)
        if r2 is None:
            r2 = r2_score(xp, yp)
        if tau is None:
            tau, _ = stats.kendalltau(xp, yp)
    except ValueError:
        return

    lines = [f"COMPARABILITY: {label} — {n:,} pairwise comparisons"]

    if is_log_scale:
        within_03 = int(np.sum(abs_diff <= 0.3))
        within_10 = int(np.sum(abs_diff <= 1.0))
        outside_10 = n - within_10
        lines.append(f"  Within ±0.3 log units: {within_03:>8,} ({within_03 / n * 100:5.1f}%)")
        lines.append(f"  Within ±1.0 log units: {within_10:>8,} ({within_10 / n * 100:5.1f}%)")
        lines.append(f"  Outside ±1.0 log units:{outside_10:>8,} ({outside_10 / n * 100:5.1f}%)")

    lines.append(f"  Spearman rho: {rho:.3f}  |  Kendall tau: {tau:.3f}  |  R²: {r2:.3f}")
    log.info("\n".join(lines))


def plot_subset(
    subset: pd.DataFrame,
    title: str = "",
    color: str = "slategray",
    alpha: float = 0.3,
    figsize: Tuple[float, float] = (5, 5),
    value_column: str = "pchembl_value",
    log_transform: bool = False,
    log_scale_factor: float = 1.0,
    axis_label: Optional[str] = None,
    axis_limits: Optional[Tuple[float, float]] = None,
    reference_lines: bool = True,
    units: Optional[str] = None,
    show_n: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create scatter plot comparing values across assays with correlation metrics.

    Args:
        subset: DataFrame with {value_column}_x and {value_column}_y columns.
        title: Plot title. The number of plotted pairwise comparisons is appended
            unless show_n is False.
        color: Color for scatter points.
        alpha: Transparency for scatter points.
        figsize: Figure size as (width, height) tuple.
        value_column: Base name of the value column (without _x/_y suffix).
            Defaults to "pchembl_value".
        log_transform: If True, apply -log10 transformation to values before plotting.
            Use for concentration values (nM, µM, etc.) but NOT for percentages or pChEMBL.
        log_scale_factor: Scale factor for the unit of measurement (e.g., 1e-6 for
            values in 10^-6 cm/s units). The transformation becomes -log10(value * factor).
            For example, a value of 5 in 10^-6 cm/s units with factor=1e-6 gives
            -log10(5e-6) ≈ 5.3, producing pChEMBL-like positive values.
        axis_label: Custom axis label. If None, uses format_axis_label() to generate
            a LaTeX-formatted label based on value_column, log_transform, and units.
        axis_limits: Tuple of (min, max) for both axes. If None, defaults to (3, 12) for
            pchembl_value, (0, 100) for percentage data, or auto-detects from data.
        reference_lines: If True, draw identity and ±1/±0.3 reference lines. These are
            most meaningful for pChEMBL-scale data.
        units: Unit string for axis labels (e.g., "10^-6 cm/s"). Converted to LaTeX
            format automatically. Only used when axis_label is None.
        show_n: If True, append the number of pairwise comparisons to the title.

    Returns:
        Tuple of (figure, axes) objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    subset = subset.copy()

    x_col = f"{value_column}_x"
    y_col = f"{value_column}_y"

    subset[x_col] = subset[x_col].astype(float)
    subset[y_col] = subset[y_col].astype(float)

    xp = subset[x_col].copy()
    yp = subset[y_col].copy()

    if log_transform:
        # Apply -log10(value * scale_factor) transformation
        # Math: -log10(x * factor) = -log10(x) - log10(factor) = -log10(x) + offset
        # For factor=1e-6: offset = -log10(1e-6) = 6, giving positive pChEMBL-like values
        offset = -np.log10(log_scale_factor) if log_scale_factor != 1.0 else 0
        # Use epsilon of 0.001 to handle near-zero values sensibly
        # (zeros would otherwise produce extreme outliers)
        xp = -np.log10(xp + 0.001) + offset
        yp = -np.log10(yp + 0.001) + offset

    ax.scatter(
        xp,
        yp,
        alpha=alpha,
        edgecolors="none",
        color=color,
        zorder=3,  # above the reference lines, which would otherwise hide points on y=x
    )
    ax.set_title(format_title_with_n(title, len(xp)) if show_n else title)

    # Determine axis limits
    if axis_limits is not None:
        lim_min, lim_max = axis_limits
    elif value_column == "pchembl_value":
        # pChEMBL values typically range from 3-12
        lim_min, lim_max = 3, 12
    else:
        # Auto-detect from (possibly transformed) data
        all_vals = np.concatenate([xp, yp])
        data_min, data_max = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (data_max - data_min) * 0.1
        lim_min = data_min - margin
        lim_max = data_max + margin

    if reference_lines and (value_column == "pchembl_value" or log_transform):
        # Reference lines are most meaningful for pChEMBL-scale data
        ax.plot((lim_min, lim_max), (lim_min, lim_max), "k-", label="Identity $(y=x)$")
        ax.plot((lim_min, lim_max), (lim_min - 1, lim_max - 1), "k--")
        ax.plot((lim_min, lim_max), (lim_min + 1, lim_max + 1), "k--", label="$y=x±1$")
        ax.plot((lim_min, lim_max), (lim_min - 0.3, lim_max - 0.3), "k-.")
        ax.plot((lim_min, lim_max), (lim_min + 0.3, lim_max + 0.3), "k-.", label="$y=x±0.3$")
    elif reference_lines:
        # For percentage or other linear data, just draw identity line
        ax.plot((lim_min, lim_max), (lim_min, lim_max), "k-", label="Identity $(y=x)$")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Calculate correlation metrics
    r, _ = stats.spearmanr(xp, yp)
    tau, _ = stats.kendalltau(xp, yp)
    r2 = r2_score(xp, yp)

    ax.text(
        1.0,
        0.175,
        format_metrics_text(r2, r, tau),
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Log quantitative comparability metrics
    is_log = value_column == "pchembl_value" or log_transform
    _log_comparability_metrics(
        xp.values, yp.values, label=title or "Overall", is_log_scale=is_log, rho=r, r2=r2, tau=tau
    )

    # Determine axis labels
    if axis_label is not None:
        label_base = axis_label
    elif value_column == "pchembl_value":
        label_base = "pChEMBL value"
    else:
        property_name = value_column.replace("_", " ").title()
        label_base = format_axis_label(property_name, log_transform=log_transform, units=units)

    ax.set_ylabel(f"Assay 2 {label_base}")
    ax.set_xlabel(f"Assay 1 {label_base}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.45, -0.15),
        ncol=3,
        frameon=False,
    )

    return fig, ax


class _CoveragePanel(NamedTuple):
    """One panel of :func:`plot_cross_assay_coverage`, with the columns it reads."""

    pct_col: str
    overlap_col: str
    total_col: str
    title: str
    subtitle: str
    remainder_label: str


#: The two panels, each reporting its share against its own denominator.
_COVERAGE_PANELS = [
    _CoveragePanel(
        pct_col="datapoint_overlap_pct",
        overlap_col="comparable_datapoints",
        total_col="aggregated_datapoints",
        title="Datapoint overlap",
        subtitle="Aggregated compound–target readouts measured in >1 assay",
        remainder_label="one assay",
    ),
    _CoveragePanel(
        pct_col="assay_overlap_pct",
        overlap_col="assays_with_overlap",
        total_col="represented_assays",
        title="Assay overlap",
        subtitle="Assay IDs sharing ≥1 aggregated datapoint with another assay",
        remainder_label="isolated",
    ),
]


def plot_cross_assay_coverage(
    coverage: pd.DataFrame,
    labels: Optional[dict] = None,
    color: Union[str, dict] = DEFAULT_COV_BAR_COLOR,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot how much of each dataset cross-assay comparability can reach.

    Comparability metrics say nothing about how much data they were computed on, so the
    two denominators are drawn side by side: the share of aggregated datapoints measured
    in more than one assay, and the share of represented assay identifiers that share a
    datapoint with another assay. Each bar is a filled share against a full-width track,
    so the part the analysis cannot see stays visible.

    Args:
        coverage: Output of :func:`~Capricho.flag_report.coverage_table`, one row per
            dataset. Rows are drawn top to bottom in the order given, so sort or filter
            it before plotting.
        labels: Maps a ``dataset`` value to the axis label shown for it, for example
            ``{"IC50 (ChEMBL 32)": r"IC$_{50}$"}``. Datasets without an entry keep their
            name.
        color: Colour of the filled bars: one colour for every dataset, or a dict keyed
            by ``dataset``.
        title: Figure-level title. No suptitle is drawn when None.
        figsize: Overrides the default size, whose height grows with the number of
            datasets. Label placement is tuned for the default width; below about 10
            inches the bar labels start to collide.

    Returns:
        Tuple of the figure and the array of its two axes.

    Raises:
        ValueError: If ``coverage`` has no rows.
    """

    #: Muted palette for the coverage bars:
    cov_track_color = "#f2f2f0"
    cov_track_edge_color = "#8b8b87"
    cov_text_color = "#0b0b0b"
    cov_2nd_text_color = "#52514e"
    cov_grid_color = "#e2e2df"
    cov_surf_color = "#fcfcfb"

    n_datasets = len(coverage)
    if not n_datasets:
        raise ValueError("`coverage` has no rows to plot.")

    if isinstance(color, dict):
        colors = [color.get(dataset, DEFAULT_COV_BAR_COLOR) for dataset in coverage["dataset"]]
    else:
        colors = [color] * n_datasets

    if figsize is None:
        # Width keeps the overlap label and the remainder label from running into each
        # other once the counts are long; height is a fixed allowance for the titles above
        # and the shared x label below, plus a row pitch wide enough for the two-line label
        # to fit inside a bar at any dataset count.
        figsize = (12, 1.5 + 0.55 * n_datasets + 0.9)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, facecolor=cov_surf_color, layout="constrained"
    )
    y_positions = np.arange(n_datasets)
    labels = labels or {}
    tick_labels = [labels.get(dataset, dataset) for dataset in coverage["dataset"]]

    for ax, panel in zip(axes, _COVERAGE_PANELS):
        ax.set_facecolor(cov_surf_color)
        ax.barh(
            y_positions,
            100,
            color=cov_track_color,
            edgecolor=cov_track_edge_color,
            linewidth=1.0,
            height=0.56,
        )
        ax.barh(y_positions, coverage[panel.pct_col], color=colors, height=0.56)

        for y, row in enumerate(coverage.itertuples()):
            overlap_pct = getattr(row, panel.pct_col)
            label_inside = overlap_pct >= 35
            ax.text(
                1.2 if label_inside else overlap_pct + 1.2,
                y,
                f"Overlap: {overlap_pct:.1f}%\n"
                f"({getattr(row, panel.overlap_col):,}/{getattr(row, panel.total_col):,})",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="semibold",
                color=cov_surf_color if label_inside else cov_text_color,
            )
            ax.text(
                98.8,
                y,
                f"{100 - overlap_pct:.1f}% {panel.remainder_label}",
                va="center",
                ha="right",
                fontsize=8.5,
                color=cov_text_color,
            )

        ax.set_title(panel.title, loc="left", fontsize=12, fontweight="bold", color=cov_text_color, pad=25)
        # Offset in points, not axes fractions, so the subtitle stays under the title
        # instead of drifting above it as the figure grows with the number of datasets.
        ax.annotate(
            panel.subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            va="bottom",
            fontsize=8.5,
            color=cov_2nd_text_color,
        )
        ax.set_yticks(y_positions, tick_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", colors=cov_text_color, length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=cov_grid_color, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supxlabel("Share within each panel denominator (%)", fontsize=10, color=cov_text_color)
    if title is not None:
        fig.suptitle(title, fontsize=16, color=cov_text_color)
    return fig, axes


def build_query_string(comment: str, value_column: str = "pchembl_value") -> str:
    """Build query string for filtering data by specific comment flags.

    This function replicates the original notebook logic for querying exploded datasets.
    It handles both exact matches and pattern-based matches (e.g., for assay size).

    Args:
        comment: The comment pattern to search for (may be normalized pattern like "Assay size <").
        value_column: Base name of the value column (without _x/_y suffix).
            Used for duplicate checking queries. Defaults to "pchembl_value".

    Returns:
        Query string suitable for pd.DataFrame.query().
    """
    # Since this data duplication is a flagged introduced by capricho, we'll just show the
    # data points that have x == y for this flag
    if comment == ProcessingComment.PCHEMBL_DUPLICATION_ACROSS_DOCUMENTS.value:
        return (
            "(data_processing_comment_x.str.contains('pChEMBL Duplication Across Documents', regex=False) & "
            "data_processing_comment_y.str.contains('pChEMBL Duplication Across Documents', regex=False) & "
            f"{value_column}_x == {value_column}_y)"
        )

    # Special handling for calculated pChEMBL (uses combined processing_comment column)
    if comment == ProcessingComment.CALCULATED_PCHEMBL.value:
        return "processing_comment.str.contains('Calculated pChEMBL', regex=False) & (dropping_comment == '')"

    # For other processing comments (uses combined processing_comment column)
    if comment == ProcessingComment.SALT_SOLVENT_REMOVED.value:
        return f"processing_comment.str.contains('{comment}', regex=False) & dropping_comment == ''"

    # Special handling for unit conversion (pattern-based, uses combined processing_comment column)
    if comment == ProcessingComment.UNIT_CONVERTED.value:
        return "processing_comment.str.contains('Unit converted to', regex=False) & dropping_comment == ''"

    # Special handling for potential duplicates (needs to be in both assays, exclude data validity issues)
    if comment == DroppingComment.POTENTIAL_DUPLICATE.value:
        return (
            "((data_dropping_comment_x.str.contains('Potential Duplicate', regex=False) & "
            "data_dropping_comment_y.str.contains('Potential Duplicate', regex=False)) | "
            "(data_dropping_comment_y.str.contains('Potential Duplicate', regex=False) & "
            "data_dropping_comment_x.str.contains('Potential Duplicate', regex=False))) & "
            "~dropping_comment.str.contains('Data Validity Comment Present', regex=False)"
        )

    # Special handling for unit annotation errors (needs to be in both assays, exclude data validity issues)
    if comment == DroppingComment.UNIT_ANNOTATION_ERROR.value:
        return (
            "((data_dropping_comment_x.str.contains('Unit Annotation Error', regex=False) & "
            "data_dropping_comment_y.str.contains('Unit Annotation Error', regex=False)) | "
            "(data_dropping_comment_y.str.contains('Unit Annotation Error', regex=False) & "
            "data_dropping_comment_x.str.contains('Unit Annotation Error', regex=False))) & "
            "~dropping_comment.str.contains('Data Validity Comment Present', regex=False)"
        )

    # we need to compare the documents with dates to the ones without to see if it's an issue
    if comment == DroppingComment.MISSING_DOCUMENT_DATE.value:
        return (
            "((data_dropping_comment_x.str.contains('Missing document date', regex=False) | "
            "data_dropping_comment_y.str.contains('Missing document date', regex=False)) & "
            "~dropping_comment.str.contains('Data Validity Comment Present', regex=False))"
        )

    # For Data Validity Comment, allow any processing comment
    if comment == DroppingComment.DATA_VALIDITY_COMMENT.value:
        return f"dropping_comment.str.contains('{comment}', regex=False)"

    # For other dropping comments, exclude Data Validity Comment
    return (
        f"dropping_comment.str.contains('{comment}', regex=False) & "
        "~dropping_comment.str.contains('Data Validity Comment Present', regex=False)"
    )


def plot_multi_panel_comparability(
    exploded_subset: pd.DataFrame,
    comments: List[str],
    title: str = "Comparability Across Flagged Data",
    figsize: Tuple[float, float] = (20, 8),
    ncols: int = 5,
    value_column: str = "pchembl_value",
    log_transform: bool = False,
    log_scale_factor: float = 1.0,
    axis_label: Optional[str] = None,
    axis_limits: Optional[Tuple[float, float]] = None,
    reference_lines: bool = True,
    units: Optional[str] = None,
    alpha: float = 0.5,
    show_n: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create multi-panel plot showing comparability for different data quality flags.

    Args:
        exploded_subset: DataFrame from explode_assay_comparability().
        comments: List of comment strings to plot.
        title: Overall figure title.
        figsize: Figure size as (width, height) tuple.
        ncols: Number of columns in subplot grid.
        value_column: Base name of the value column (without _x/_y suffix).
            Defaults to "pchembl_value".
        log_transform: If True, apply -log10 transformation to values before plotting.
            Use for concentration values (nM, µM, etc.) but NOT for percentages or pChEMBL.
        log_scale_factor: Scale factor for the unit of measurement (e.g., 1e-6 for
            values in 10^-6 cm/s units). The transformation becomes -log10(value * factor).
        axis_label: Custom axis label. If None, uses format_axis_label() to generate
            a LaTeX-formatted label based on value_column, log_transform, and units.
        axis_limits: Tuple of (min, max) for both axes. If None, defaults to (3, 12) for
            pchembl_value, or auto-detects from data.
        reference_lines: If True, draw identity and ±1/±0.3 reference lines.
        units: Unit string for axis labels (e.g., "10^-6 cm/s"). Converted to LaTeX
            format automatically. Only used when axis_label is None.
        alpha: Transparency for scatter points.
        show_n: If True, append each panel's number of pairwise comparisons to its title.

    Returns:
        Tuple of (figure, axes array).
    """
    x_col = f"{value_column}_x"
    y_col = f"{value_column}_y"

    comments_with_data = []  # only display the comments that have data
    n_data = []  # debugging info only
    for comment in comments:
        query_str = build_query_string(comment, value_column=value_column)
        subset = exploded_subset.query(query_str)
        if len(subset) > 0:
            comments_with_data.append(comment)
            n_data.append(len(subset))

    if len(comments_with_data) == 0:  # No data to plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No data available for any flags", ha="center", va="center")
        ax.axis("off")
        return fig, np.array([ax])

    nrows = int(np.ceil(len(comments_with_data) / ncols))
    # tab10 keeps every panel saturated. tab20 alternates dark/light pairs, tinting
    # every even-numbered panel too faintly to read.
    palette = colormaps["tab10"].colors
    colors = [tuple([*palette[i % len(palette)]] + [1]) for i in range(len(comments_with_data))]

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs_flat = axs.flatten() if nrows > 1 else [axs] if ncols == 1 else axs

    # Determine axis limits
    if axis_limits is not None:
        lim_min, lim_max = axis_limits
    elif value_column == "pchembl_value":
        # pChEMBL values typically range from 3-12
        lim_min, lim_max = 3, 12
    else:
        # Auto-detect from (possibly transformed) data
        all_x = exploded_subset[x_col].astype(float)
        all_y = exploded_subset[y_col].astype(float)
        if log_transform:
            offset = -np.log10(log_scale_factor) if log_scale_factor != 1.0 else 0
            all_x = -np.log10(all_x + 0.001) + offset
            all_y = -np.log10(all_y + 0.001) + offset
        all_vals = np.concatenate([all_x, all_y])
        data_min, data_max = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (data_max - data_min) * 0.1
        lim_min = data_min - margin
        lim_max = data_max + margin

    # Determine axis labels
    if axis_label is not None:
        label_base = axis_label
    elif value_column == "pchembl_value":
        label_base = "pChEMBL value"
    else:
        property_name = value_column.replace("_", " ").title()
        label_base = format_axis_label(property_name, log_transform=log_transform, units=units)

    for idx, color, obs, ax in zip(
        range(1, len(comments_with_data) + 1), colors, comments_with_data, axs_flat
    ):
        query_str = build_query_string(obs, value_column=value_column)
        subset = exploded_subset.query(query_str)

        # Extract actual title with dynamic threshold if it's a parametric comment
        title_str = obs
        if obs.startswith("Assay size") or obs.startswith("Insufficient assay overlap"):
            # Find first occurrence with actual threshold from data
            for col in ["data_dropping_comment_x", "data_dropping_comment_y"]:
                if col in subset.columns:
                    sample_comments = subset[col].dropna()
                    if len(sample_comments) > 0:
                        # Split by " & " since data_dropping_comment uses " & " as separator
                        for comment in sample_comments.iloc[0].split(" & "):
                            comment = comment.strip()
                            if normalize_comment_pattern(comment) == obs:
                                title_str = comment
                                break
                        if title_str != obs:
                            break

        subset = subset.copy()
        subset[x_col] = subset[x_col].astype(float)
        subset[y_col] = subset[y_col].astype(float)

        xp = subset[x_col].copy()
        yp = subset[y_col].copy()

        if log_transform:
            offset = -np.log10(log_scale_factor) if log_scale_factor != 1.0 else 0
            xp = -np.log10(xp + 0.001) + offset
            yp = -np.log10(yp + 0.001) + offset

        ax.scatter(
            xp,
            yp,
            alpha=alpha,
            edgecolors="none",
            label=title_str,
            color=color,
            zorder=3,  # above the reference lines, which would otherwise hide points on y=x
        )
        panel_title = f"{idx}. {title_str}"
        ax.set_title(
            format_title_with_n(panel_title, len(xp), sep="\n") if show_n else panel_title,
            fontsize="medium",
        )

        # Add reference lines
        if reference_lines and (value_column == "pchembl_value" or log_transform):
            ax.plot((lim_min, lim_max), (lim_min, lim_max), "k-", label="Identity $(y=x)$")
            ax.plot((lim_min, lim_max), (lim_min - 1, lim_max - 1), "k--")
            ax.plot((lim_min, lim_max), (lim_min + 1, lim_max + 1), "k--", label="$y=x±1$")
            ax.plot((lim_min, lim_max), (lim_min - 0.3, lim_max - 0.3), "k-.")
            ax.plot((lim_min, lim_max), (lim_min + 0.3, lim_max + 0.3), "k-.", label="$y=x±0.3$")
        elif reference_lines:
            ax.plot((lim_min, lim_max), (lim_min, lim_max), "k-", label="Identity $(y=x)$")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        try:
            r, _ = stats.spearmanr(xp, yp)
            tau, _ = stats.kendalltau(xp, yp)
            r2 = r2_score(xp, yp)
        except ValueError:
            continue

        ax.text(
            1.0,
            0.175,
            format_metrics_text(r2, r, tau),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )

        # Log per-panel comparability metrics
        is_log = value_column == "pchembl_value" or log_transform
        _log_comparability_metrics(
            xp.values, yp.values, label=title_str, is_log_scale=is_log, rho=r, r2=r2, tau=tau
        )

        if idx in [1, ncols + 1]:
            ax.set_ylabel(f"Assay 2 {label_base}")
        if idx > (nrows - 1) * ncols:
            ax.set_xlabel(f"Assay 1 {label_base}")

        if idx == len(comments_with_data):
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 1:
                ax.legend(
                    handles[1:],
                    labels[1:],
                    title="",
                    bbox_to_anchor=(1.05, 0, 1, 0.2),
                    loc="lower left",
                    mode="expand",
                    frameon=False,
                )

    # Hide unused subplots
    for ax in axs_flat[len(comments_with_data) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)

    return fig, axs
