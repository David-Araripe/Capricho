"""Contain functions for binarizing bioactivity data; handling censored data and validating agreement between discrete and censored measurements"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean, gstd

from ..core.pandas_helper import add_comment
from ..logger import logger


def _calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Calculate Matthews Correlation Coefficient from confusion matrix values.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives

    Returns:
        MCC value between -1.0 and 1.0, or 0.0 if denominator is zero
    """
    numerator = (tp * tn) - (fp * fn)
    denominator_parts = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if denominator_parts == 0:
        return 0.0

    import math

    denominator = math.sqrt(denominator_parts)

    return numerator / denominator


def _calculate_assay_compatibility_mcc(
    df: pd.DataFrame,
    compound_id_col: str,
    target_id_col: str,
    output_binary_col: str,
) -> float:
    """Calculate MCC measuring agreement between measurements for same compound-target pairs.

    Uses pairwise comparisons: for each compound-target pair with multiple measurements,
    compares all pairs of binary labels to build confusion matrix, then calculates MCC.

    Args:
        df: DataFrame with binarized data
        compound_id_col: Column name for compound IDs
        target_id_col: Column name for target IDs
        output_binary_col: Column name with binary labels

    Returns:
        MCC value between -1.0 and 1.0
    """
    groupby_cols = [compound_id_col, target_id_col]
    if "mutation" in df.columns:
        groupby_cols.append("mutation")

    tp = tn = fp = fn = 0

    for _, group_df in df.groupby(groupby_cols):
        binary_labels = group_df[output_binary_col].dropna().tolist()

        # Skip if less than 2 measurements
        if len(binary_labels) < 2:
            continue

        # Pairwise comparisons within this group
        for i in range(len(binary_labels)):
            for j in range(i + 1, len(binary_labels)):
                label_i = binary_labels[i]
                label_j = binary_labels[j]

                # Treat label_i as "truth" and label_j as "prediction"
                if label_i == 1 and label_j == 1:
                    tp += 1
                elif label_i == 0 and label_j == 0:
                    tn += 1
                elif label_i == 0 and label_j == 1:
                    fp += 1
                elif label_i == 1 and label_j == 0:
                    fn += 1

    return _calculate_mcc(tp, tn, fp, fn)


def _truncate_dataframe(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Truncate DataFrame values to a specified length for display purposes.

    Args:
        df: DataFrame to truncate
        limit: Maximum character length for each cell value

    Returns:
        DataFrame with truncated string values
    """
    if pd.__version__ > "2.1.0":  # applymap got deprecated in 2.1.0
        return df.map(lambda x: str(x)[:limit] + "..." if len(str(x)) > limit else str(x))
    else:
        return df.applymap(lambda x: str(x)[:limit] + "..." if len(str(x)) > limit else str(x))


def invert_relation_for_pchembl(relation: str) -> str:
    """Inverts comparison relation for pchembl values.

    Since pchembl = -log10(Molar), higher pchembl = more active (lower concentration).
    Therefore, standard_relation directions must be inverted:
    - standard_relation "<" (low concentration, active) → pchembl ">" (high value, active)
    - standard_relation ">" (high concentration, inactive) → pchembl "<" (low value, inactive)

    Args:
        relation: Original standard_relation from ChEMBL ("=", "<", ">", "<=", ">=", "~", ">>", "<<")

    Returns:
        Inverted relation for pchembl comparison
    """
    inversion_map = {
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
        "=": "=",
        "~": "~",
        ">>": "<<",
        "<<": ">>",
    }
    if relation not in inversion_map:
        raise ValueError(f"Unknown relation: {relation}")
    return inversion_map[relation]


def _classify_by_relation(value: float, relation: str, threshold: float) -> int:
    """Classify a single measurement as active (1) or inactive (0) based on relation type.

    Args:
        value: pchembl_value to classify
        relation: standard_relation ("=", "~", "<", ">", "<=", ">=", "<<", ">>")
        threshold: Activity threshold for binarization

    Returns:
        1 for active, 0 for inactive
    """
    if relation == "=":
        return 1 if value >= threshold else 0

    elif relation == "~":
        lower_bound = value - 0.5
        return 1 if lower_bound >= threshold else 0

    elif relation in ["<", "<=", "<<"]:
        return 1 if value >= threshold else 0

    elif relation in [">", ">=", ">>"]:
        return 0 if value <= threshold else 1

    else:
        raise ValueError(f"Unknown relation: {relation}")


VALID_CONFLICT_STRATEGIES = {"drop", "relation", "confidence", "majority"}
CENSORED_RELATIONS = {"<", ">", "<=", ">=", "<<", ">>"}


def _max_confidence_score(confidence_str) -> int:
    """Parse a pipe-separated confidence_score string and return the maximum value.

    Args:
        confidence_str: A single value like "9" or pipe-separated like "8|9", or NaN

    Returns:
        Maximum confidence score as integer, or 0 for NaN/empty
    """
    if pd.isna(confidence_str) or str(confidence_str).strip() == "":
        return 0
    parts = str(confidence_str).split("|")
    try:
        return max(int(p.strip()) for p in parts if p.strip())
    except ValueError:
        return 0


def _resolve_conflicts(
    df: pd.DataFrame,
    conflict_indices: list,
    strategy: str,
    compound_id_col: str,
    target_id_col: str,
    relation_col: str,
    output_binary_col: str,
    groupby_cols: list[str],
    value_column: str = "pchembl_value_mean",
    threshold: float | None = None,
) -> tuple[list, list[dict]]:
    """Apply a conflict resolution strategy to conflicting compound-target groups.

    Args:
        df: DataFrame with binarized data
        conflict_indices: List of row indices involved in conflicts
        strategy: One of "drop", "relation", "confidence", "majority"
        compound_id_col: Column name for compound IDs
        target_id_col: Column name for target IDs
        relation_col: Column with standard_relation values
        output_binary_col: Column with binary labels
        groupby_cols: Columns to group by for conflict detection
        value_column: Column used for binarization (used to derive counts column)
        threshold: Binarization threshold (used for measurement-level majority voting)

    Returns:
        Tuple of (indices_to_drop, resolution_details) where resolution_details is
        a list of dicts keyed by (compound_id, target_id) with resolution info
    """
    if strategy == "confidence" and "confidence_score" not in df.columns:
        raise ValueError(
            "Strategy 'confidence' requires a 'confidence_score' column in the DataFrame"
        )

    # Derive raw value column and counts column from value_column
    base = value_column.rsplit("_", 1)[0] if "_" in value_column else value_column
    counts_col = None
    raw_value_col = None
    if strategy == "majority":
        candidate_counts = f"{base}_counts"
        if candidate_counts in df.columns:
            counts_col = candidate_counts
        if base in df.columns and base != value_column:
            raw_value_col = base

    conflict_subset = df.loc[conflict_indices]
    indices_to_drop = []
    resolution_details = []

    for group_key, group_df in conflict_subset.groupby(groupby_cols):
        has_conflict, _ = _detect_conflicts(group_df, output_binary_col)
        if not has_conflict:
            continue

        if strategy == "drop":
            drop_idx, detail = _resolve_drop_group(group_key, group_df)
        elif strategy == "relation":
            drop_idx, detail = _resolve_relation_group(group_key, group_df, relation_col)
        elif strategy == "confidence":
            drop_idx, detail = _resolve_confidence_group(group_key, group_df)
        elif strategy == "majority":
            drop_idx, detail = _resolve_majority_group(
                group_key,
                group_df,
                output_binary_col,
                counts_col=counts_col,
                raw_value_col=raw_value_col,
                relation_col=relation_col,
                threshold=threshold,
            )
        else:
            raise ValueError(f"Unknown conflict resolution strategy: '{strategy}'")

        indices_to_drop.extend(drop_idx)
        resolution_details.append(detail)

    return indices_to_drop, resolution_details


def _resolve_drop_group(group_key, group_df) -> tuple[list, dict]:
    """Drop all rows in a conflicting group."""
    return group_df.index.tolist(), {
        "group_key": group_key,
        "strategy": "drop",
        "outcome": "dropped_all",
        "rows_kept": 0,
        "rows_dropped": len(group_df),
        "detail": f"Dropped all {len(group_df)} rows",
    }


def _resolve_relation_group(group_key, group_df, relation_col) -> tuple[list, dict]:
    """Keep '=' rows and drop censored rows. Fall back to dropping all if no '=' exists."""
    exact_mask = group_df[relation_col] == "="
    exact_rows = group_df[exact_mask]

    if len(exact_rows) == 0:
        logger.warning(
            f"Conflict for {group_key}: no exact (=) measurements found, dropping all rows"
        )
        return group_df.index.tolist(), {
            "group_key": group_key,
            "strategy": "relation",
            "outcome": "dropped_all",
            "rows_kept": 0,
            "rows_dropped": len(group_df),
            "detail": "No exact (=) measurements found, dropped all rows",
        }

    censored_indices = group_df[~exact_mask].index.tolist()
    return censored_indices, {
        "group_key": group_key,
        "strategy": "relation",
        "outcome": "kept_exact",
        "rows_kept": len(exact_rows),
        "rows_dropped": len(censored_indices),
        "detail": f"Kept {len(exact_rows)} exact (=) rows, dropped {len(censored_indices)} censored rows",
    }


def _resolve_confidence_group(group_key, group_df) -> tuple[list, dict]:
    """Keep row(s) with highest confidence_score. Drop all on tie across binary labels."""
    group_df = group_df.copy()
    group_df["_max_conf"] = group_df["confidence_score"].apply(_max_confidence_score)
    max_conf = group_df["_max_conf"].max()
    winners = group_df[group_df["_max_conf"] == max_conf]

    # If winners still have conflicting labels, it's a tie → drop all
    if len(winners["activity_binary"].dropna().unique()) > 1:
        return group_df.index.tolist(), {
            "group_key": group_key,
            "strategy": "confidence",
            "outcome": "dropped_all",
            "rows_kept": 0,
            "rows_dropped": len(group_df),
            "detail": f"Tied confidence scores ({max_conf}) with conflicting labels, dropped all rows",
        }

    losers = group_df[group_df["_max_conf"] != max_conf].index.tolist()
    outcome_label = "active" if winners.iloc[0]["activity_binary"] == 1 else "inactive"
    return losers, {
        "group_key": group_key,
        "strategy": "confidence",
        "outcome": f"kept_{outcome_label}",
        "rows_kept": len(winners),
        "rows_dropped": len(losers),
        "detail": f"Kept {len(winners)} rows with confidence_score={max_conf} ({outcome_label})",
    }


def _resolve_majority_group(
    group_key,
    group_df,
    output_binary_col,
    counts_col: str | None = None,
    raw_value_col: str | None = None,
    relation_col: str = "standard_relation",
    threshold: float | None = None,
) -> tuple[list, dict]:
    """Keep rows matching the majority binary label, weighted by measurement count.

    When the raw pipe-separated value column is available (e.g. pchembl_value), each
    individual measurement is classified against the threshold and gets one vote.
    Otherwise, falls back to count-weighted or row-based voting.
    """
    valid = group_df[output_binary_col].dropna()
    if len(valid.unique()) < 2:
        return [], {
            "group_key": group_key,
            "strategy": "majority",
            "outcome": "no_conflict",
            "rows_kept": len(group_df),
            "rows_dropped": 0,
            "detail": "No conflict after dropping NaN labels",
        }

    # Measurement-level voting: split pipe-separated values, classify each individually
    use_measurement_level = (
        raw_value_col
        and raw_value_col in group_df.columns
        and relation_col in group_df.columns
        and threshold is not None
    )

    if use_measurement_level:
        active_weight = 0.0
        inactive_weight = 0.0
        for _, row in group_df.iterrows():
            raw_str = str(row[raw_value_col])
            relation = str(row[relation_col])
            if pd.isna(row[raw_value_col]) or raw_str == "nan":
                continue
            values = raw_str.split("|")
            for v in values:
                try:
                    label = _classify_by_relation(float(v), relation, threshold)
                    if label == 1:
                        active_weight += 1
                    else:
                        inactive_weight += 1
                except (ValueError, TypeError):
                    continue
        weight_label = "individual measurements"
    elif counts_col and counts_col in group_df.columns:
        weights = group_df[counts_col].fillna(1).astype(float)
        active_weight = float(weights[group_df[output_binary_col] == 1].sum())
        inactive_weight = float(weights[group_df[output_binary_col] == 0].sum())
        weight_label = "measurements"
    else:
        active_weight = float((group_df[output_binary_col] == 1).sum())
        inactive_weight = float((group_df[output_binary_col] == 0).sum())
        weight_label = "rows"

    active_weight = float(active_weight)
    inactive_weight = float(inactive_weight)

    if active_weight == inactive_weight:
        return group_df.index.tolist(), {
            "group_key": group_key,
            "strategy": "majority",
            "outcome": "dropped_all",
            "rows_kept": 0,
            "rows_dropped": len(group_df),
            "detail": f"Tied votes: {active_weight:.0f} active vs {inactive_weight:.0f} inactive "
            f"{weight_label}, dropped all rows",
        }

    if active_weight > inactive_weight:
        majority_label = 1
        majority_weight = active_weight
        minority_weight = inactive_weight
    else:
        majority_label = 0
        majority_weight = inactive_weight
        minority_weight = active_weight

    minority_indices = group_df[group_df[output_binary_col] != majority_label].index.tolist()
    kept = len(group_df) - len(minority_indices)
    outcome_label = "active" if majority_label == 1 else "inactive"
    return minority_indices, {
        "group_key": group_key,
        "strategy": "majority",
        "outcome": f"kept_{outcome_label}",
        "rows_kept": kept,
        "rows_dropped": len(minority_indices),
        "detail": f"Majority vote: {majority_weight:.0f} {outcome_label} vs {minority_weight:.0f} "
        f"{weight_label}, dropped {len(minority_indices)} minority rows",
    }


def _classify_conflict_pattern(measurements: list[dict]) -> str:
    """Classify a conflict as exact_vs_censored or censored_vs_censored.

    Args:
        measurements: List of measurement dicts with 'standard_relation' key

    Returns:
        "exact_vs_censored" or "censored_vs_censored"
    """
    relations = {m.get("standard_relation", "=") for m in measurements}
    has_exact = "=" in relations
    has_censored = bool(relations & CENSORED_RELATIONS)

    if has_exact and has_censored:
        return "exact_vs_censored"
    return "censored_vs_censored"


def _detect_conflicts(
    group_df: pd.DataFrame,
    output_binary_col: str,
) -> tuple[bool, str]:
    """Detect conflicts within a compound-target group based on binarization outcomes.

    A conflict occurs when multiple measurements for the same compound-target pair
    result in different binary labels (active vs inactive).

    Args:
        group_df: DataFrame subset for one compound-target pair
        output_binary_col: Column name with binary labels

    Returns:
        Tuple of (has_conflict, conflict_type) where conflict_type is "binary_label_mismatch" or ""
    """
    binary_labels = group_df[output_binary_col].dropna()
    if len(binary_labels) > 1 and len(binary_labels.unique()) > 1:
        return True, "binary_label_mismatch"

    return False, ""


def _generate_conflict_details(
    df: pd.DataFrame,
    conflict_indices: list,
    compound_id_col: str,
    target_id_col: str,
    pchembl_relation_col: str,
    value_column: str,
    output_binary_col: str,
    threshold: float,
) -> list[dict]:
    """Generate detailed conflict information for each compound-target pair.

    Args:
        df: DataFrame with binarized data
        conflict_indices: List of row indices with conflicts
        compound_id_col: Column name for compound IDs
        target_id_col: Column name for target IDs
        pchembl_relation_col: Column name for pchembl_relation
        value_column: Column name for pchembl values
        output_binary_col: Column name for binary labels
        threshold: Binarization threshold

    Returns:
        List of conflict detail dictionaries
    """
    if not conflict_indices:
        return []

    conflict_subset = df.loc[conflict_indices].copy()
    groupby_cols = [compound_id_col, target_id_col]
    if "mutation" in conflict_subset.columns:
        groupby_cols.append("mutation")

    conflict_details = []
    for group_key, group_df in conflict_subset.groupby(groupby_cols):
        has_conflict, conflict_type = _detect_conflicts(group_df, output_binary_col)

        if not has_conflict:
            continue

        measurements = []
        for _, row in group_df.iterrows():
            measurement = {
                "value": float(row[value_column]) if not pd.isna(row[value_column]) else None,
                "pchembl_relation": row[pchembl_relation_col],
                "binary": int(row[output_binary_col]) if not pd.isna(row[output_binary_col]) else None,
            }

            if "standard_relation" in row and not pd.isna(row["standard_relation"]):
                measurement["standard_relation"] = row["standard_relation"]
            if "assay_chembl_id" in row:
                measurement["assay"] = row["assay_chembl_id"]
            if "molecule_chembl_id" in row:
                measurement["molecule"] = row["molecule_chembl_id"]

            measurements.append(measurement)

        conflict_detail = {
            "compound_id": group_key[0],
            "target_id": group_key[1],
            "conflict_type": conflict_type,
            "measurements": measurements,
            "threshold": threshold,
        }

        if len(groupby_cols) > 2:
            conflict_detail["mutation"] = group_key[2]

        # Group measurements by binary outcome
        active_measurements = [m for m in measurements if m["binary"] == 1]
        inactive_measurements = [m for m in measurements if m["binary"] == 0]

        # Create vote summary
        vote_summary = {
            "active_votes": len(active_measurements),
            "inactive_votes": len(inactive_measurements),
        }
        conflict_detail["vote_summary"] = vote_summary

        # Build explanation with vote counts and measurement details
        explanation_parts = []

        if active_measurements:
            active_strs = [f"{m['pchembl_relation']}{m['value']:.2f}" for m in active_measurements]
            explanation_parts.append(f"Active ({len(active_measurements)} votes): {', '.join(active_strs)}")

        if inactive_measurements:
            inactive_strs = [f"{m['pchembl_relation']}{m['value']:.2f}" for m in inactive_measurements]
            explanation_parts.append(
                f"Inactive ({len(inactive_measurements)} votes): {', '.join(inactive_strs)}"
            )

        conflict_detail["explanation"] = " | ".join(explanation_parts) + f" | Threshold: {threshold}"

        # Severity: distance and spread metrics
        values = [m["value"] for m in measurements if m["value"] is not None]
        if values:
            spread = max(values) - min(values)
            max_distance = max(abs(v - threshold) for v in values)
            if spread < 1.0:
                classification = "low"
            elif spread <= 2.0:
                classification = "medium"
            else:
                classification = "high"
            conflict_detail["severity"] = {
                "max_distance_from_threshold": round(max_distance, 4),
                "measurement_spread": round(spread, 4),
                "classification": classification,
            }

        # Recommendation based on relation types
        relations = {m.get("standard_relation", "=") for m in measurements}
        has_exact = "=" in relations
        has_censored = bool(relations & CENSORED_RELATIONS)
        if has_exact and has_censored:
            conflict_detail["recommendation"] = (
                "Exact measurement (=) is generally more reliable than censored bounds"
            )
        else:
            conflict_detail["recommendation"] = (
                "Manual review recommended -- all measurements are of the same type"
            )

        conflict_details.append(conflict_detail)

    return conflict_details


def _log_and_flag_conflicts(
    df: pd.DataFrame,
    conflict_indices: list,
    compound_id_col: str,
    target_id_col: str,
    relation_col: str,
    value_column: str,
    output_binary_col: str,
    conflict_resolution: str | None = None,
) -> pd.DataFrame:
    """Log conflict details and flag conflicting rows in the DataFrame.

    Args:
        df: DataFrame with binarized data
        conflict_indices: List of row indices with conflicts
        compound_id_col: Column name for compound IDs
        target_id_col: Column name for target IDs
        relation_col: Column name for relations
        value_column: Column name for pchembl values
        output_binary_col: Column name for binary labels
        conflict_resolution: If set, suppresses the detail table since conflicts will be resolved

    Returns:
        DataFrame with conflicts flagged via add_comment()
    """
    if not conflict_indices:
        return df

    if conflict_resolution:
        logger.warning(
            f"Found {len(conflict_indices)} measurements with disagreements. "
            f"Resolving with strategy '{conflict_resolution}'."
        )
    else:
        logger.warning(
            f"Found {len(conflict_indices)} measurements with disagreements. "
            "These compound-target pairs have inconsistent measurements across different relation types."
        )

        conflict_subset = df.loc[conflict_indices].copy()

        logging_cols = [compound_id_col, target_id_col, relation_col, value_column, "mutation"]
        optional_cols = ["molecule_chembl_id", "assay_chembl_id", output_binary_col, "standard_relation"]
        for col in optional_cols:
            if col in conflict_subset.columns:
                logging_cols.append(col)

        logging_cols = [col for col in logging_cols if col in conflict_subset.columns]
        conflict_display = conflict_subset[logging_cols].sort_values(
            by=[target_id_col, compound_id_col, value_column], ascending=[True, True, False]
        )

        truncated_df = _truncate_dataframe(conflict_display, limit=15)
        logger.warning(
            f"Sample of conflicting measurements (showing up to 20 rows):\n"
            f"{truncated_df.head(20).to_string(index=False)}"
        )

    df = add_comment(
        df=df,
        comment="Non-agreeing discrete and censored values",
        criteria_func=lambda x: x.index.isin(conflict_indices),
        target_column=value_column,
        comment_type="d",
    )

    return df


def save_conflict_report(
    conflict_details: list[dict],
    output_path: str | Path,
    threshold: float,
    total_rows: int = 0,
    active_count: int = 0,
    inactive_count: int = 0,
    mcc: float = 0.0,
    resolution_details: list[dict] | None = None,
    conflict_resolution: str | None = None,
) -> None:
    """Save conflict report to JSON file.

    Args:
        conflict_details: List of conflict detail dictionaries
        output_path: Path to save the JSON file
        threshold: Binarization threshold used
        total_rows: Total number of rows in the DataFrame
        active_count: Number of active rows
        inactive_count: Number of inactive rows
        mcc: Matthews Correlation Coefficient
        resolution_details: Resolution details from _resolve_conflicts
        conflict_resolution: Strategy name used for resolution
    """
    # Count conflict patterns
    pattern_counts = {"exact_vs_censored": 0, "censored_vs_censored": 0}
    for conflict in conflict_details:
        pattern = _classify_conflict_pattern(conflict.get("measurements", []))
        pattern_counts[pattern] += 1

    summary = {
        "total_rows": total_rows,
        "total_conflicts": len(conflict_details),
        "threshold": threshold,
        "active_count": active_count,
        "inactive_count": inactive_count,
        "mcc": round(mcc, 4),
        "conflict_patterns": pattern_counts,
    }

    # Add resolution summary if a strategy was used
    if conflict_resolution and resolution_details is not None:
        resolved = sum(1 for d in resolution_details if d["rows_dropped"] > 0)
        unresolved = sum(1 for d in resolution_details if d["rows_dropped"] == 0)
        total_dropped = sum(d["rows_dropped"] for d in resolution_details)
        summary["resolution_summary"] = {
            "strategy": conflict_resolution,
            "conflicts_resolved": resolved,
            "conflicts_unresolved": unresolved,
            "total_rows_dropped": total_dropped,
        }

    # Add per-conflict resolution info
    if conflict_resolution and resolution_details is not None:
        detail_by_key = {tuple(d["group_key"]): d for d in resolution_details}
        for conflict in conflict_details:
            key = (conflict["compound_id"], conflict["target_id"])
            if "mutation" in conflict:
                key = (*key, conflict["mutation"])
            if key in detail_by_key:
                d = detail_by_key[key]
                conflict["resolution"] = {
                    "strategy": d["strategy"],
                    "outcome": d["outcome"],
                    "rows_kept": d["rows_kept"],
                    "rows_dropped": d["rows_dropped"],
                    "detail": d["detail"],
                }

    report = {
        "summary": summary,
        "conflicts": conflict_details,
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Conflict report saved to {output_path}")


def _deduplicate_resolved_groups(
    df: pd.DataFrame,
    groupby_cols: list[str],
    raw_value_col: str,
    relation_col: str,
    output_binary_col: str,
    threshold: float,
    use_geometric: bool = True,
) -> pd.DataFrame:
    """Merge compound-target groups into one row per group, filtering disagreeing measurements.

    For each group with multiple rows (or rows with mixed individual measurements),
    splits pipe-separated values, classifies each measurement against the threshold,
    and keeps only measurements agreeing with the row's binary label. Then merges
    all rows in the group into a single row.

    Args:
        df: DataFrame after conflict resolution
        groupby_cols: Columns defining compound-target groups
        raw_value_col: Column with pipe-separated raw values (e.g. "pchembl_value")
        relation_col: Column with standard_relation
        output_binary_col: Column with binary labels
        threshold: Binarization threshold
        use_geometric: If True, use geometric mean/std for stats recalculation

    Returns:
        DataFrame with one row per compound-target group
    """
    if raw_value_col not in df.columns:
        return df

    # Detect pipe-separated columns
    pipe_cols = []
    for col in df.columns:
        if df[col].astype(str).str.contains("|", regex=False).any():
            pipe_cols.append(col)

    # Derive stats column names
    stats_suffixes = ["_mean", "_std", "_median", "_counts"]
    stats_cols = [f"{raw_value_col}{s}" for s in stats_suffixes]
    stats_cols = [c for c in stats_cols if c in df.columns]

    merged_rows = []
    indices_processed = set()

    for _, group_df in df.groupby(groupby_cols):
        if len(group_df) == 1:
            row = group_df.iloc[0]
            binary_label = row[output_binary_col]
            raw_str = str(row[raw_value_col])

            # Skip if no binary label or no raw values
            if pd.isna(binary_label) or pd.isna(row[raw_value_col]) or raw_str == "nan":
                continue

            # Single row: still filter disagreeing measurements within it
            if "|" in raw_str:
                values = raw_str.split("|")
                relation = str(row[relation_col])
                keep_indices = []
                for i, v in enumerate(values):
                    try:
                        label = _classify_by_relation(float(v), relation, threshold)
                        if label == int(binary_label):
                            keep_indices.append(i)
                    except (ValueError, TypeError):
                        keep_indices.append(i)  # Keep on error

                if len(keep_indices) == 0:
                    indices_processed.update(group_df.index.tolist())
                    continue  # All measurements disagree, drop the group

                if len(keep_indices) < len(values):
                    # Filter pipe-separated columns at the same positions
                    new_row = row.copy()
                    for col in pipe_cols:
                        col_values = str(new_row[col]).split("|")
                        if len(col_values) == len(values):
                            new_row[col] = "|".join(col_values[i] for i in keep_indices)

                    # Expand standard_relation to pipe-separated to match
                    new_row[relation_col] = "|".join(relation for _ in keep_indices)

                    # Recalculate stats
                    kept_values = [float(values[i]) for i in keep_indices]
                    new_row = _recalculate_stats(
                        new_row, raw_value_col, kept_values, use_geometric
                    )
                    merged_rows.append(new_row)
                    indices_processed.update(group_df.index.tolist())
                    continue
            # No filtering needed for single-value rows
            continue

        # Multi-row group: merge all rows into one
        binary_label = group_df[output_binary_col].dropna().iloc[0]
        all_kept_values = []
        all_kept_relations = []
        # For each pipe-separated column, collect kept entries
        pipe_col_kept = {col: [] for col in pipe_cols}

        for _, row in group_df.iterrows():
            raw_str = str(row[raw_value_col])
            relation = str(row[relation_col])

            if pd.isna(row[raw_value_col]) or raw_str == "nan":
                continue

            values = raw_str.split("|")
            n_values = len(values)
            relations = [relation] * n_values  # Expand single relation to match count

            keep_indices = []
            for i, v in enumerate(values):
                try:
                    label = _classify_by_relation(float(v), relations[i], threshold)
                    if label == int(binary_label):
                        keep_indices.append(i)
                except (ValueError, TypeError):
                    keep_indices.append(i)

            for i in keep_indices:
                all_kept_values.append(values[i])
                all_kept_relations.append(relations[i])

            # Filter corresponding positions in all pipe-separated columns
            for col in pipe_cols:
                col_values = str(row[col]).split("|")
                if len(col_values) == n_values:
                    pipe_col_kept[col].extend(col_values[i] for i in keep_indices)
                else:
                    # Column doesn't align with raw values; just concatenate all
                    pipe_col_kept[col].extend(col_values)

        if not all_kept_values:
            indices_processed.update(group_df.index.tolist())
            continue  # All measurements filtered out, drop the group

        # Build merged row from first row as template
        new_row = group_df.iloc[0].copy()
        new_row[output_binary_col] = binary_label

        # Set pipe-separated columns to merged values
        for col in pipe_cols:
            if pipe_col_kept[col]:
                new_row[col] = "|".join(pipe_col_kept[col])

        # Set standard_relation to pipe-separated per-measurement relations
        new_row[relation_col] = "|".join(all_kept_relations)

        # Recalculate stats
        kept_floats = [float(v) for v in all_kept_values]
        new_row = _recalculate_stats(new_row, raw_value_col, kept_floats, use_geometric)

        merged_rows.append(new_row)
        indices_processed.update(group_df.index.tolist())

    if not merged_rows and not indices_processed:
        return df

    # Build result: keep unprocessed rows as-is, append merged rows
    unprocessed = df.loc[~df.index.isin(indices_processed)]
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        result = pd.concat([unprocessed, merged_df], ignore_index=True)
    else:
        result = unprocessed.reset_index(drop=True)

    return result


def _recalculate_stats(
    row: pd.Series,
    raw_value_col: str,
    values: list[float],
    use_geometric: bool,
) -> pd.Series:
    """Recalculate mean/std/median/counts stats for a row from a list of values.

    Calculates directly instead of using assign_stats to avoid the censored-row
    override (which doesn't work with mixed pipe-separated relations).
    """
    n = len(values)
    arr = np.array(values)

    if use_geometric:
        mean_val = float(gmean(10 ** (-arr)))
        mean_val = -np.log10(mean_val)
        if n > 1:
            std_val = float(gstd(10 ** (-arr)))
        else:
            std_val = np.nan
        median_val = float(np.median(arr))
    else:
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr)) if n > 1 else np.nan
        median_val = float(np.median(arr))

    mean_col = f"{raw_value_col}_mean"
    std_col = f"{raw_value_col}_std"
    median_col = f"{raw_value_col}_median"
    counts_col = f"{raw_value_col}_counts"

    if mean_col in row.index:
        row[mean_col] = mean_val
    if std_col in row.index:
        row[std_col] = std_val
    if median_col in row.index:
        row[median_col] = median_val
    if counts_col in row.index:
        row[counts_col] = n

    return row


def binarize_aggregated_data(
    df: pd.DataFrame,
    threshold: float = 6.0,
    value_column: str = "pchembl_value_mean",
    compound_id_col: str = "inchikey",
    target_id_col: str = "target_chembl_id",
    relation_col: str = "standard_relation",
    output_binary_col: str = "activity_binary",
    compare_across_mutants: bool = False,
    conflict_report_path: str | Path | None = None,
    conflict_resolution: str | None = None,
) -> pd.DataFrame:
    """Binarize aggregated bioactivity data based on activity threshold and standard_relation.

    This function converts continuous pchembl values to binary labels (0=inactive, 1=active)
    while properly handling censored measurements and approximate values, and validating
    agreement between discrete and censored measurements for the same compound-target pair.

    Key logic:
    - standard_relation "=": compare value to threshold directly
    - standard_relation "~": approximate (±0.5 log units); uses lower bound for conservative classification
    - standard_relation "<", "<<" (low concentration): if pchembl >= threshold → active (1)
    - standard_relation ">", ">>" (high concentration): if pchembl <= threshold → inactive (0)
    - Mixed relations: validate agreement and flag conflicts

    Args:
        df: Aggregated DataFrame from aggregate_data() with pchembl statistics
        threshold: Activity threshold for binarization (default 6.0 = 1 µM)
        value_column: Which aggregated column to use (default: "pchembl_value_mean")
        compound_id_col: Column identifying compounds (default: "inchikey").
            Select "connectivity" explicitly for connectivity-only data.
        target_id_col: Column identifying targets (default: "target_chembl_id")
        relation_col: Column with standard_relation values (default: "standard_relation")
        output_binary_col: Name for output binary column (default: "activity_binary")
        compare_across_mutants: If False (default), different mutations are treated as separate
            compound-target pairs for conflict detection. If True, measurements on different
            mutants are compared and flagged if they disagree.
        conflict_report_path: Optional path to save detailed conflict report as JSON
        conflict_resolution: Strategy for resolving conflicts. One of:
            - None (default): flag only, keep all rows
            - "drop": remove all rows for conflicting pairs
            - "relation": keep '=' rows, drop censored; fall back to drop if no '='
            - "confidence": keep row with highest confidence_score; drop all on tie
            - "majority": keep rows matching majority binary label; drop all on tie

    Returns:
        DataFrame with binary activity labels, pchembl_relation column, and conflict flags
    """
    if conflict_resolution is not None and conflict_resolution not in VALID_CONFLICT_STRATEGIES:
        raise ValueError(
            f"Unknown conflict resolution strategy: '{conflict_resolution}'. "
            f"Valid options: {sorted(VALID_CONFLICT_STRATEGIES)}"
        )

    required_cols = [compound_id_col, target_id_col, value_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        message = f"Missing required columns: {missing_cols}"
        if compound_id_col in missing_cols:
            message += (
                ". Use compound_id_col='connectivity' (CLI: --compound-id-col connectivity) "
                "for connectivity-only data, or select the identifier used during aggregation."
            )
        raise ValueError(message)

    df = df.copy()

    if relation_col not in df.columns:
        logger.warning(f"Column '{relation_col}' not found. Assuming all measurements have '=' relation.")
        df[relation_col] = "="

    pchembl_relation_col = "pchembl_relation"
    df[pchembl_relation_col] = df[relation_col].apply(invert_relation_for_pchembl)

    groupby_cols = [compound_id_col, target_id_col]
    if not compare_across_mutants and "mutation" in df.columns:
        groupby_cols.append("mutation")

    df[output_binary_col] = np.nan
    conflict_indices = []

    for idx, row in df.iterrows():
        value = row[value_column]
        relation = row[relation_col]

        if pd.isna(value):
            continue

        try:
            df.loc[idx, output_binary_col] = _classify_by_relation(value, relation, threshold)
        except ValueError as e:
            logger.warning(f"{e} at index {idx}. Skipping binarization.")

    for group_key, group_df in df.groupby(groupby_cols):
        has_conflict, conflict_type = _detect_conflicts(group_df, output_binary_col)

        if has_conflict:
            conflict_indices.extend(group_df.index.tolist())
            logger.debug(
                f"Conflict ({conflict_type}) for {compound_id_col}={group_key[0]}, "
                f"{target_id_col}={group_key[1]}"
            )

    df = _log_and_flag_conflicts(
        df,
        conflict_indices,
        compound_id_col,
        target_id_col,
        pchembl_relation_col,
        value_column,
        output_binary_col,
        conflict_resolution=conflict_resolution,
    )

    # Generate conflict details before resolution (needs all rows present)
    conflict_details = None
    if conflict_report_path and conflict_indices:
        conflict_details = _generate_conflict_details(
            df,
            conflict_indices,
            compound_id_col,
            target_id_col,
            pchembl_relation_col,
            value_column,
            output_binary_col,
            threshold,
        )

    # Apply conflict resolution if requested
    resolution_details = None
    if conflict_resolution and conflict_indices:
        indices_to_drop, resolution_details = _resolve_conflicts(
            df,
            conflict_indices,
            conflict_resolution,
            compound_id_col,
            target_id_col,
            relation_col,
            output_binary_col,
            groupby_cols,
            value_column=value_column,
            threshold=threshold,
        )
        if indices_to_drop:
            df = df.drop(index=indices_to_drop).reset_index(drop=True)
            logger.info(
                f"Conflict resolution (strategy='{conflict_resolution}'): "
                f"dropped {len(indices_to_drop)} rows"
            )

    # Deduplicate: merge compound-target groups into one row, filtering disagreeing measurements
    base_value_col = value_column.rsplit("_", 1)[0] if "_" in value_column else value_column
    if conflict_resolution and base_value_col in df.columns and base_value_col != value_column:
        use_geometric = base_value_col == "pchembl_value"
        rows_before = len(df)
        df = _deduplicate_resolved_groups(
            df, groupby_cols, base_value_col, relation_col,
            output_binary_col, threshold, use_geometric=use_geometric,
        )
        # Regenerate pchembl_relation from the (now possibly pipe-separated) standard_relation
        df[pchembl_relation_col] = df[relation_col].astype(str).apply(
            lambda r: "|".join(invert_relation_for_pchembl(x) for x in r.split("|"))
        )
        rows_after = len(df)
        if rows_before != rows_after:
            logger.info(
                f"Deduplication: merged {rows_before} rows into {rows_after} rows"
            )

    # Save conflict report after resolution (so stats reflect final state)
    if conflict_report_path and conflict_details is not None:
        n_active_report = int((df[output_binary_col].dropna() == 1).sum())
        n_inactive_report = int((df[output_binary_col].dropna() == 0).sum())
        mcc_report = _calculate_assay_compatibility_mcc(
            df, compound_id_col, target_id_col, output_binary_col
        )

        save_conflict_report(
            conflict_details,
            conflict_report_path,
            threshold,
            total_rows=len(df),
            active_count=n_active_report,
            inactive_count=n_inactive_report,
            mcc=mcc_report,
            resolution_details=resolution_details,
            conflict_resolution=conflict_resolution,
        )

    df[output_binary_col] = df[output_binary_col].astype("Int64")

    # Calculate summary statistics
    n_active = (df[output_binary_col] == 1).sum()
    n_inactive = (df[output_binary_col] == 0).sum()
    n_missing = df[output_binary_col].isna().sum()
    n_total = len(df)

    # Calculate conflict statistics
    n_conflicting_measurements = len(conflict_indices)
    n_rows_dropped = 0
    if resolution_details:
        n_rows_dropped = sum(d["rows_dropped"] for d in resolution_details)
        n_conflicting_pairs = len(resolution_details)
    elif conflict_indices:
        remaining_conflict_idx = [i for i in conflict_indices if i in df.index]
        if remaining_conflict_idx:
            conflict_subset = df.loc[remaining_conflict_idx]
            conflict_groupby_cols = [compound_id_col, target_id_col]
            if not compare_across_mutants and "mutation" in conflict_subset.columns:
                conflict_groupby_cols.append("mutation")
            n_conflicting_pairs = conflict_subset.groupby(conflict_groupby_cols).ngroups
        else:
            n_conflicting_pairs = 0
    else:
        n_conflicting_pairs = 0

    # Calculate MCC for assay compatibility (only for pairs with multiple measurements)
    mcc = _calculate_assay_compatibility_mcc(df, compound_id_col, target_id_col, output_binary_col)

    # Log comprehensive summary
    if n_total > 0:
        conflict_lines = (
            f"  Conflicting measurements detected: {n_conflicting_measurements:>6}\n"
            f"  Compound-target pairs affected:    {n_conflicting_pairs:>6}\n"
        )
        if conflict_resolution:
            conflict_lines += (
                f"  Resolution strategy:               {conflict_resolution:>6}\n"
                f"  Rows dropped by resolution:        {n_rows_dropped:>6}\n"
            )
        else:
            conflict_lines += (
                f"  Rows at risk if conflicts dropped: {n_conflicting_measurements:>6} "
                f"({n_conflicting_measurements/n_total*100:>5.1f}%)\n"
            )

        logger.info(
            f"BINARIZATION SUMMARY\n"
            f"Threshold: {threshold}\n"
            f"\nBinary labels:\n"
            f"  Active (1):   {n_active:>6} ({n_active/n_total*100:>5.1f}%)\n"
            f"  Inactive (0): {n_inactive:>6} ({n_inactive/n_total*100:>5.1f}%)\n"
            f"  Missing:      {n_missing:>6} ({n_missing/n_total*100:>5.1f}%)\n"
            f"\nConflict analysis:\n"
            f"{conflict_lines}"
            f"\nAssay compatibility:\n"
            f"  MCC (agreement between measurements): {mcc:>6.3f}\n"
        )

    return df
