"""Clean aggregated bioactivity data by removing quality flags and duplicates,
and pivot into activity matrices for multitask modeling.
"""

from typing import List, Optional

import pandas as pd

from ..core.pandas_helper import assign_shared_identifier_groups, assign_stats
from ..logger import logger


def clean_data(
    df: pd.DataFrame,
    drop_flags: Optional[List[str]] = None,
    deduplicate: bool = False,
    value_col: str = "pchembl_value",
    resolve_annotation_error: Optional[str] = None,
    compound_col: str = "connectivity",
) -> pd.DataFrame:
    """Clean aggregated bioactivity data by deduplicating, resolving errors, and filtering flags.

    Orchestrates cleaning steps in the correct order:
    1. Deduplicate (if requested): removes identical values within aggregated rows
       and recalculates statistics (mean, median, std, counts).
    2. Resolve annotation errors (if requested): detects measurements differing by
       exactly 3.0 or 6.0 log units (unit conversion errors), keeps the earliest
       document's value, then re-aggregates.
    3. Drop flags: removes individual flagged measurements from aggregated rows
       and recalculates statistics; rows where all measurements are flagged are
       removed entirely. Non-aggregated data is filtered at the row level.
    4. Recalculate shared-identifier group labels after any row removal.

    Appropriate flags for dropping include unit errors, undefined stereochemistry,
    assay size issues, and mixtures. For potential duplicates, prefer using
    ``deduplicate=True`` instead of dropping — this keeps one measurement while
    removing extras, rather than discarding the entire row.

    Example::

        from Capricho.cli.prepare import clean_data, prepare_multitask_data

        cleaned = clean_data(df, drop_flags=["Unit Annotation Error"], deduplicate=True)
        matrix = prepare_multitask_data(cleaned, task_col="target_chembl_id", ...)

    Args:
        df: Aggregated DataFrame from aggregate_data().
        drop_flags: List of quality flags to remove. Rows where data_dropping_comment
            contains any of these flags will be filtered out.
        deduplicate: If True, remove duplicate values within aggregated rows
            and recalculate statistics.
        value_col: Column containing the activity values (e.g., "pchembl_value"
            or "standard_value"). Used for deduplication and stats recalculation.
        resolve_annotation_error: Resolution strategy for unit annotation errors.
            Currently only "first" is supported (keep earliest document).
            Cannot be used together with dropping "Unit Annotation Error" flags.
        compound_col: Column defining compound identity for re-aggregation and
            ``shared_identifier_group``. Defaults to ``connectivity``.

    Returns:
        Cleaned DataFrame.

    Raises:
        ValueError: If both ``resolve_annotation_error`` and "Unit Annotation Error"
            in ``drop_flags`` are set, since these are contradictory operations.
    """
    from ..analysis import (
        DroppingComment,
        deaggregate_data,
        deduplicate_aggregated_values,
        filter_aggregated_dropping_flags,
        resolve_annotation_errors,
    )
    from ..flag_report import format_flag_summary, summarize_flags

    # Validate: can't both resolve and drop annotation errors
    if resolve_annotation_error is not None and drop_flags:
        if DroppingComment.UNIT_ANNOTATION_ERROR.value in drop_flags:
            raise ValueError(
                "Cannot both resolve and drop unit annotation errors. "
                "Use resolve_annotation_error to fix them, or drop_flags to remove them."
            )

    if resolve_annotation_error is not None and resolve_annotation_error != "first":
        raise ValueError(
            f"Unknown resolution strategy: {resolve_annotation_error}. Only 'first' is supported."
        )

    df = df.copy()
    input_rows = len(df)

    # Track steps for summary
    dedup_removed = 0
    annotation_removed = 0
    rows_after_dedup = input_rows
    rows_after_annotation = input_rows

    # Step 1: Deduplicate
    if deduplicate:
        logger.info("Deduplicating identical values within aggregated rows...")
        initial_total = df[value_col].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0).sum()
        df = deduplicate_aggregated_values(df, value_column=value_col)
        final_total = df[value_col].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0).sum()
        dedup_removed = initial_total - final_total
        logger.info(f"Deduplication removed {dedup_removed} duplicate values")

        logger.info("Recalculating statistics after deduplication...")
        df = assign_stats(df, value_col=value_col, use_geometric=(value_col == "pchembl_value"))
        rows_after_dedup = len(df)

    # Step 2: Resolve annotation errors
    if resolve_annotation_error is not None:
        logger.info("Resolving unit annotation errors (3.0 or 6.0 log unit differences)...")

        initial_rows = len(df)
        exploded = deaggregate_data(df)
        logger.info(f"Exploded {initial_rows} aggregated rows into {len(exploded)} individual measurements")

        resolved = resolve_annotation_errors(
            exploded,
            strategy=resolve_annotation_error,
            value_col=value_col,
        )
        annotation_removed = len(exploded) - len(resolved)
        logger.info(f"Removed {annotation_removed} measurements due to annotation error resolution")

        # Re-aggregate the data
        from .chembl_data_pipeline import re_aggregate_data

        # Detect extra ID columns placed between compound identifiers and SMILES.
        cols = list(df.columns)
        if "connectivity" in cols and "smiles" in cols:
            connectivity_idx = cols.index("connectivity")
            smiles_idx = cols.index("smiles")
            identifier_columns = {"connectivity", compound_col, "smiles"}
            detected_id_cols = [
                column
                for column in cols[connectivity_idx + 1 : smiles_idx]
                if column not in identifier_columns
            ]
            logger.info(f"Detected id_columns for re-aggregation: {detected_id_cols}")
        else:
            detected_id_cols = []

        df = re_aggregate_data(
            resolved,
            chirality=False,
            extra_id_cols=detected_id_cols,
            compound_equality=compound_col,
        )
        logger.info(f"Re-aggregated to {len(df)} rows")
        rows_after_annotation = len(df)

    # Step 3: Drop flags (measurement-level for aggregated data)
    rows_before_flags = len(df)
    measurements_before_flags = _count_measurements(df, value_col)
    omission_summary = None
    if drop_flags:
        omission_summary = summarize_flags(df, flags=drop_flags, per_measurement=True)
        df = filter_aggregated_dropping_flags(df, drop_flags, value_column=value_col)

    # Log consolidated summary
    lines = ["", "PREPARATION SUMMARY"]
    lines.append(f"  Input rows:                {input_rows:>8,}")
    if deduplicate:
        lines.append(
            f"  After deduplication:       {rows_after_dedup:>8,}  (removed {dedup_removed} duplicate values)"
        )
    if resolve_annotation_error is not None:
        lines.append(
            f"  After annotation resolution:{rows_after_annotation:>7,}  (removed {annotation_removed} measurements)"
        )
    if drop_flags:
        rows_removed_by_flags = rows_before_flags - len(df)
        lines.append(f"  After flag filtering:      {len(df):>8,}  (removed {rows_removed_by_flags} rows)")
    lines.append(f"  Final rows:                {len(df):>8,}")

    if omission_summary is not None:
        lines.append("")
        lines.append(
            format_flag_summary(
                omission_summary,
                title=(
                    f"OMITTED BY QUALITY FLAGS — share of {measurements_before_flags:,} "
                    "measurements pooled in the input"
                ),
            )
        )
    logger.info("\n".join(lines))

    shared_key = (compound_col, "target_chembl_id")
    if set(shared_key).issubset(df.columns):
        df = assign_shared_identifier_groups(df, key_columns=shared_key)
    return df


def _count_measurements(df: pd.DataFrame, value_col: str, sep_str: str = "|") -> int:
    """Count the individual measurements pooled in a value column."""
    if value_col not in df.columns or len(df) == 0:
        return len(df)
    return int(df[value_col].apply(lambda x: len(str(x).split(sep_str)) if pd.notna(x) else 0).sum())


def prepare_multitask_data(
    df: pd.DataFrame,
    task_col: str,
    value_col: str,
    compound_col: str,
    smiles_col: str,
    id_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Transform aggregated data to multitask format (activity matrix).

    This function pivots aggregated bioactivity data to create an activity matrix where:
    - Rows represent unique compounds (identified by compound_col)
    - Columns represent tasks (e.g., different targets)
    - Values are the bioactivity measurements (e.g., pchembl_value_mean)

    Use ``clean_data()`` before calling this function to filter quality flags
    and deduplicate values.

    Args:
        df: Aggregated DataFrame from aggregate_data() with bioactivity statistics.
        task_col: Column to use as task identifier (e.g., "target_chembl_id").
        value_col: Column containing values to pivot (e.g., "pchembl_value_mean").
        compound_col: Column defining compound identity (for example, ``connectivity``,
            ``inchi``, ``inchikey``, or ``smiles``).
        smiles_col: Column containing SMILES strings.
        id_columns: List of additional columns to combine with task_col for creating
            composite task identifiers. Use this when data was aggregated with
            --id-columns (e.g., ["assay_tissue"]) to prevent losing information.

    Returns:
        DataFrame with compounds as rows (indexed by compound_col), tasks as columns,
        and a smiles column. Missing values are represented as NaN.
    """
    df = df.copy()

    # Validate required columns exist
    required_cols = [compound_col, task_col, value_col, smiles_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create composite task column if id_columns are provided
    effective_task_col = task_col
    if id_columns is not None and len(id_columns) > 0:
        missing_id_cols = [col for col in id_columns if col not in df.columns]
        if missing_id_cols:
            raise ValueError(f"Missing id_columns: {missing_id_cols}")

        # Create composite task identifier by joining task_col with id_columns
        composite_name = "_composite_task"
        df[composite_name] = df[task_col].astype(str)
        for col in id_columns:
            df[composite_name] = df[composite_name] + "-" + df[col].fillna("").astype(str)
        effective_task_col = composite_name
        logger.info(f"Created composite task column from: {task_col} + {id_columns}")

    # Check for duplicates (multiple values per compound-task pair)
    dup_check = df.groupby([compound_col, effective_task_col]).size()
    duplicates = dup_check[dup_check > 1]
    if len(duplicates) > 0:
        n_dup_pairs = len(duplicates)
        n_extra_rows = duplicates.sum() - n_dup_pairs
        logger.warning(
            f"Found {n_dup_pairs} compound-task pairs with multiple values ({n_extra_rows} extra rows). "
            "Only the first value will be kept. If the rows were kept separate by mutation or "
            "--id-columns during aggregation, include those distinguishing columns here via --id-columns. "
            "Inspect shared_identifier_group in CAPRICHO output to identify the affected groups."
        )

    # Pivot the data to create activity matrix
    logger.info(
        f"Creating activity matrix with {df[compound_col].nunique()} compounds "
        f"and {df[effective_task_col].nunique()} tasks"
    )

    activity_matrix = df.pivot_table(
        index=compound_col,
        columns=effective_task_col,
        values=value_col,
        aggfunc="first",
    )

    # Reset the columns name to remove the task_col label
    activity_matrix.columns.name = None

    # Add SMILES column back by taking the first SMILES for each compound
    if smiles_col != compound_col:
        smiles_map = df.groupby(compound_col)[smiles_col].first()
        activity_matrix[smiles_col] = smiles_map

    # Compute sparsity (fraction of NaN cells, excluding smiles column)
    task_cols = [c for c in activity_matrix.columns if c != smiles_col]
    if task_cols:
        n_cells = activity_matrix[task_cols].size
        n_missing = activity_matrix[task_cols].isna().sum().sum()
        sparsity = n_missing / n_cells * 100 if n_cells > 0 else 0.0
    else:
        sparsity = 0.0

    logger.info(
        f"Activity matrix: {activity_matrix.shape[0]} compounds x {len(task_cols)} tasks "
        f"(sparsity: {sparsity:.1f}%)"
    )

    return activity_matrix
