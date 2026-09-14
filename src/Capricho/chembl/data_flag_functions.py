"""Collection of functions to flag compounds based on specific criteria.

Not all the functions are annotating compounds to be removed from the dataset. Some are
used to annotate processing steps that occurred during the data processing pipeline, like:

- Salt/solvent removal (applied to the canonical SMILES since they're kept as-is by CompoundMapper)
- Calculated pChEMBL value (used when the this measure is absent in ChEMBL and calculated from nM ... etc readouts)
- Potential duplicates (when this one is found, CompoundMapper will keep only one of those in the dataset. If
  the user wants to investigate, please pass the `keep_duplicates` flag to True)

"""

import re
from typing import Optional

import pandas as pd

from ..core.default_fields import (
    ASSAY_ID,
    DATA_DROPPING_COMMENT,
    MOLECULE_ID,
    TARGET_ID,
)
from ..core.pandas_helper import add_comment, conflicting_duplicates
from ..logger import logger
from .unit_conversions import is_unit_annotation_error_diff

# Deliberately excludes activity_comment w/ ambiguous abbreviations such as "nd".
REVIEW_ACTIVITY_COMMENTS = (
    "not active",
    "inactive",
    "no activity",
    "below threshold",
    "below detection",
    "inconclusive",
    "not tested",
    "not determined",
)


### Marking readouts that will be DROPPED by Capricho ###
def flag_missing_canonical_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'canonical_smiles' is missing."""
    return add_comment(
        df,
        comment="Missing SMILES",
        criteria_func=lambda x: x.isna(),  # x is the Series df["canonical_smiles"]
        target_column="canonical_smiles",
        comment_type="d",
    )


def flag_missing_standard_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'standard_smiles' is missing."""
    return add_comment(
        df,
        comment="Missing Standard SMILES",
        criteria_func=lambda x: x.isna(),
        target_column="standard_smiles",
        comment_type="d",
    )


def flag_with_data_validity_comment(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'data_validity_comment' is present (not NA)."""
    return add_comment(
        df,
        comment="Data Validity Comment Present",
        criteria_func=lambda x: ~x.isna(),
        target_column="data_validity_comment",
        comment_type="d",
    )


def flag_potential_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'potential_duplicate' is 0."""
    return add_comment(
        df,
        comment="Potential Duplicate",
        criteria_func=lambda x: x == 1,
        target_column="potential_duplicate",
        comment_type="d",
    )


def flag_to_remove_mixture_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'mixture_compounds' is True."""
    return add_comment(
        df,
        comment="Mixture in SMILES",
        criteria_func=lambda x: x.str.contains(".", regex=False),
        target_column="standard_smiles",
        comment_type="d",
    )


def flag_undefined_stereochemistry(df: pd.DataFrame) -> pd.DataFrame:
    """Mark compounds with undefined stereochemistry based on a predefined boolean mask."""
    return add_comment(
        df,
        comment="Undefined Stereochemistry",
        criteria_func=lambda x: x > 0,
        target_column="undefined_stereocenters",
        comment_type="d",
    )


def flag_zero_values(df: pd.DataFrame, column: str = "standard_value") -> pd.DataFrame:
    """Mark rows where the measurement value is exactly zero.

    Zero values in bioactivity measurements are typically data quality issues -
    they may represent values below the limit of detection, data entry errors,
    or rounding artifacts. This flag helps identify such problematic data points.

    Args:
        df: DataFrame to process.
        column: Column name to check for zero values. Defaults to "standard_value".

    Returns:
        DataFrame with zero-value rows flagged in data_dropping_comment.
    """
    if column not in df.columns:
        logger.debug(f"Column '{column}' not found. Skipping zero value flagging.")
        return df

    return add_comment(
        df,
        comment="Zero Value",
        criteria_func=lambda x: x == 0,
        target_column=column,
        comment_type="d",
    )


def flag_min_assay_size(df: pd.DataFrame, min_assay_size: int = 0) -> pd.DataFrame:
    """Mark assays for removal based on size lower than the specified minimum assay size."""
    assert min_assay_size >= 0, "Minimum assay size must be a non-negative integer."

    if "assay_size" not in df.columns:
        logger.error("Column 'assay_size' not found in DataFrame. Skipping minimum assay size filtering.")
        return df
    if min_assay_size == 0:
        logger.info("Minimum assay size is set to 0. Skipping filtering based on assay size.")
        return df
    else:
        return add_comment(
            df,
            comment=f"Assay size < {min_assay_size}",
            criteria_func=lambda x: x < min_assay_size,
            target_column="assay_size",
            comment_type="d",
        )


def flag_max_assay_size(df: pd.DataFrame, max_assay_size: Optional[int] = None) -> pd.DataFrame:
    """Mark assays for removal based on size greater than the specified maximum assay size."""
    if max_assay_size is None:
        logger.info("Maximum assay size is not set. Skipping filtering based on maximum assay size.")
        return df

    assert max_assay_size > 0, "Maximum assay size must be a positive integer."

    if "assay_size" not in df.columns:
        logger.error("Column 'assay_size' not found in DataFrame. Skipping maximum assay size filtering.")
        return df
    if max_assay_size == 0:
        logger.info("Maximum assay size is set to 0. Skipping filtering based on assay size.")
        return df
    else:
        return add_comment(
            df,
            comment=f"Assay size > {max_assay_size}",
            criteria_func=lambda x: x > max_assay_size,
            target_column="assay_size",
            comment_type="d",
        )


def flag_strict_mutant_assays(df: pd.DataFrame, strict_mutant_removal: bool = False) -> pd.DataFrame:
    """Mark assays for removal if their description contains mutant-related keywords
    and strict_mutant_removal is True.
    """
    if not strict_mutant_removal:
        logger.debug("Strict mutant removal is False. Skipping assay description-based mutant flagging.")
        return df

    if "assay_description" not in df.columns:
        logger.warning(
            "Column 'assay_description' not found in DataFrame. " "Skipping strict mutant assay flagging."
        )
        return df

    keywords = ["mutant", "mutation", "variant"]
    keyword_pattern = "|".join(keywords)

    criteria = (  # noqa: E731 - lambda function saved in a variable here just for clarity
        lambda series: series.astype(str).str.lower().str.contains(keyword_pattern, regex=True, na=False)
    )

    # Calculate the mask for logging purposes before applying add_comment
    mask_to_flag = criteria(df["assay_description"])
    num_to_flag = mask_to_flag.sum()

    if num_to_flag > 0:
        logger.info(
            f"Flagging {num_to_flag} assays for removal based on strict mutant removal criteria "
            f"(keywords: {', '.join(keywords)} in 'assay_description')."
        )
        df = add_comment(
            df,
            comment="Mutation keyword in assay description",
            criteria_func=criteria,
            target_column="assay_description",
            comment_type="d",
        )
    else:
        logger.info(
            "No assays to flag for removal based on strict mutant removal criteria in 'assay_description'."
        )
    return df


def flag_missing_document_date(df: pd.DataFrame) -> pd.DataFrame:
    """Mark activities that lack a document date (year) in the processing comment.

    This function always flags missing document dates for transparency, regardless of whether
    they will be filtered out. Activities without document dates are flagged in the
    data_processing_comment column so users can see which data points lack temporal information.

    Args:
        df: DataFrame to be processed.

    Returns:
        pd.DataFrame: DataFrame with activities lacking document dates flagged in processing comment.
    """
    if "year" not in df.columns:
        logger.debug("Column 'year' not found in DataFrame. Skipping document date flagging.")
        return df

    mask = df["year"].isna()
    num_to_flag = mask.sum()

    if num_to_flag > 0:
        logger.info(f"Flagging {num_to_flag} activities with missing document date (year) for transparency.")
        df = add_comment(
            df,
            comment="Missing document date",
            criteria_func=lambda x: x.isna(),
            target_column="year",
            comment_type="d",
        )
    else:
        logger.debug("No activities with missing document dates found.")

    return df


def flag_incompatible_units(df: pd.DataFrame) -> pd.DataFrame:
    """Mark activities with units that cannot be converted to pChEMBL in the dropping comment.

    This function flags activities with standard_units that are incompatible with pChEMBL
    calculation (i.e., not nM, µM, uM, or mM). These activities will have pchembl_value=NaN
    and are flagged for transparency.

    Args:
        df: DataFrame to be processed.

    Returns:
        pd.DataFrame: DataFrame with incompatible units flagged in dropping comment.
    """
    if "standard_units" not in df.columns:
        logger.debug("Column 'standard_units' not found in DataFrame. Skipping incompatible units flagging.")
        return df

    compatible_units = ["nM", "µM", "uM", "mM"]
    # Flag rows where standard_units is NOT in compatible_units AND NOT null
    mask = ~df["standard_units"].isin(compatible_units) & df["standard_units"].notna()
    num_to_flag = mask.sum()

    if num_to_flag > 0:
        logger.info(f"Flagging {num_to_flag} activities with incompatible units for pChEMBL calculation.")
        df = add_comment(
            df,
            comment="Incompatible units for pChEMBL calculation",
            criteria_func=lambda x: ~x.isin(compatible_units) & x.notna(),
            target_column="standard_units",
            comment_type="d",
        )
    else:
        logger.debug("No activities with incompatible units found.")

    return df


def flag_insufficient_assay_overlap(
    df: pd.DataFrame,
    min_overlap: int = 0,
    molecule_col: str = MOLECULE_ID,
    assay_col: str = ASSAY_ID,
    target_col: str = TARGET_ID,
    comment_col: str = DATA_DROPPING_COMMENT,
) -> pd.DataFrame:
    """Mark activities from assay pairs (for the same target) that don't meet the minimum
    compound overlap criterium, useful for analysis assessing the comparability of assays
    reported in ChEMBL. Depending on the target, this filter can remove a significant
    amount of activities from the dataset, but it is useful to assess the comparability of
    the assays reported in the database.

    This function calculates overlap across ALL assays regardless of size flags, following
    CAPRICHO's principle of transparency. Overlap is only counted when:
    1. Compounds have DIFFERENT pChEMBL values across assays (same values indicate annotation errors)
    2. The difference is not an exact multiple of 3.0 log units (likely unit-annotation errors)
    3. Assays are from DIFFERENT documents (same-document overlaps are excluded)

    Args:
        df: DataFrame to be processed.
        min_overlap: Minimum number of overlapping compounds required. Defaults to 0
        molecule_col: Name of the molecule identifier column. Defaults to molecule_chembl_id.
        assay_col: Name of the assay identifier column. Defaults to assay_chembl_id.
        target_col: Name of the target identifier column. Defaults to target_chembl_id.
        comment_col: Name of the column to store dropping comments. Defaults to data_dropping_comment.

    Returns:
        pd.DataFrame: DataFrame with activities from low-overlap assay pairs flagged.
    """
    if min_overlap is None:
        raise ValueError("min_overlap must be provided and cannot be None.")
        return df
    elif min_overlap == 0:
        logger.info("min_overlap set to 0. Skipping insufficient assay overlap filtering.")
        return df

    # Check for required columns
    required_cols = [molecule_col, assay_col, target_col, "pchembl_value", "document_chembl_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(
            f"Required columns missing: {missing_cols}. " "Skipping insufficient assay overlap filtering."
        )
        return df

    if df.empty:
        logger.info("DataFrame is empty. Skipping insufficient assay overlap filtering.")
        return df

    if comment_col not in df.columns:  # Ensure comment_col exists
        df[comment_col] = pd.Series(dtype="object")

    # Log the overlap calculation strategy
    logger.info(
        "Calculating assay overlap with the following criteria:\n"
        "  - Only counting compounds with DIFFERENT pChEMBL values across assays\n"
        "  - Excluding differences that are exact multiples of 3.0 log units (likely unit annotation errors)\n"
        "  - Excluding overlaps within the same document"
    )

    import numpy as np

    assays_to_flag_globally = set()  # Keep track of assays that don't have a compatible partner

    # Filter out assays already flagged for size issues (goldilocks approach in the Landrum & Riniker paper)
    # These assays should be skipped in overlap checking
    size_flagged_mask = df[comment_col].notna() & df[comment_col].astype(str).str.contains(
        r"Assay size [<>]", regex=True, na=False
    )
    size_flagged_assays = set(df[size_flagged_mask][assay_col].unique())

    if size_flagged_assays:
        logger.debug(
            f"Skipping {len(size_flagged_assays)} assays already flagged for size issues in overlap checking."
        )

    for target_id, group_df in df.groupby(target_col):
        unique_assays = group_df[assay_col].unique()

        # Filter out size-flagged assays from overlap checking
        unique_assays_to_check = [a for a in unique_assays if a not in size_flagged_assays]

        if len(unique_assays_to_check) < 2:
            continue  # Not enough non-size-flagged assays for this target to form a pair

        # Only include non-size-flagged assays in the analysis
        group_df = group_df[~group_df[assay_col].isin(size_flagged_assays)]

        # Prepare data with necessary columns for vectorized operations
        target_data = group_df[[assay_col, molecule_col, "pchembl_value", "document_chembl_id"]].copy()

        # Self-join to create all pairs of (assay1, assay2) for the same molecule
        # This finds all overlapping molecules between assays
        pairs = target_data.merge(target_data, on=molecule_col, suffixes=("_1", "_2"))

        # Filter to only keep assay1 < assay2 to avoid duplicates and self-comparisons
        pairs = pairs[pairs[f"{assay_col}_1"] < pairs[f"{assay_col}_2"]]

        # Apply filters
        # 1. Different documents
        pairs = pairs[pairs["document_chembl_id_1"] != pairs["document_chembl_id_2"]]

        # 2. Different pChEMBL values, excluding differences that are exact multiples of 3 log
        #    units (3.0, 6.0, 9.0, ...) since those are likely unit-annotation errors, not overlap.
        pchembl_diff = np.abs(pairs["pchembl_value_1"] - pairs["pchembl_value_2"])
        pairs = pairs[
            (pairs["pchembl_value_1"] != pairs["pchembl_value_2"])
            & ~is_unit_annotation_error_diff(pchembl_diff)
        ]

        # Count overlapping compounds per assay pair
        overlap_counts = pairs.groupby([f"{assay_col}_1", f"{assay_col}_2"]).size()

        # Find assay pairs with sufficient overlap
        sufficient_pairs = overlap_counts[overlap_counts >= min_overlap]

        # Collect all assays that have at least one compatible partner
        assays_with_sufficient_overlap = set()
        for (assay1, assay2), count in sufficient_pairs.items():
            assays_with_sufficient_overlap.add(assay1)
            assays_with_sufficient_overlap.add(assay2)
            logger.trace(
                f"Target {target_id}: Assays {assay1} and {assay2} have {count} compounds "
                f"with conflicting pChEMBL values (>= {min_overlap})."
            )

        # Flag assays that don't have any compatible partner (only non-size-flagged assays)
        for assay_id in unique_assays_to_check:
            if assay_id not in assays_with_sufficient_overlap:
                assays_to_flag_globally.add(assay_id)
                logger.debug(
                    f"Target {target_id}: Assay {assay_id} has no compatible partner with >= {min_overlap} "
                    f"overlapping compounds with conflicting pChEMBL values. Will be flagged for removal."
                )

    if assays_to_flag_globally:
        num_assays_flagged = len(assays_to_flag_globally)
        comment_text = f"Insufficient assay overlap (min_overlap={min_overlap})"
        logger.info(
            f"Flagging activities from {num_assays_flagged} unique assays "
            f"that lack a partner with >= {min_overlap} overlapping compounds."
        )

        # Use add_comment helper for consistency
        df = add_comment(
            df,
            comment=comment_text,
            criteria_func=lambda x: x.isin(list(assays_to_flag_globally)),
            target_column=assay_col,
            comment_type="d",
        )
    else:
        logger.info(f"All assays have at least one partner with >= {min_overlap} overlapping compounds.")

    return df


def flag_censored_activity_comment(df: pd.DataFrame) -> pd.DataFrame:
    """Flag exact relations paired with inactivity-like activity comments.

    The source relation and value remain unchanged because the comment alone does not
    establish that the reported value is censored.
    """
    if "activity_comment" not in df.columns:
        logger.debug("Column 'activity_comment' not found. Skipping activity comment review flagging.")
        return df
    if "standard_relation" not in df.columns:
        logger.warning("Column 'standard_relation' not found. Cannot flag activity comment conflicts.")
        return df

    review_comment_pattern = re.compile(
        rf"\b(?:{'|'.join(map(re.escape, REVIEW_ACTIVITY_COMMENTS))})\b",
        re.IGNORECASE,
    )

    # An all-NULL SQL result can be inferred as float rather than string.
    activity_comments = df["activity_comment"].astype("string")
    mask = activity_comments.str.contains(review_comment_pattern, na=False) & df["standard_relation"].eq("=")
    if not mask.any():
        logger.debug("No exact relations with inactivity-like activity comments found.")
        return df

    logger.info(f"Flagging {mask.sum()} exact relations with inactivity-like activity comments.")
    return add_comment(
        df,
        comment="Activity with exact standard relation and inactivity-like comment",
        criteria_func=lambda _: mask,
        target_column="activity_comment",
        comment_type="d",
    )


### Marking readouts that were PROCESSED by CompoundMapper ###
def flag_calculated_pchembl(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows where 'calculated_pchembl' is True."""
    return add_comment(
        df,
        comment="Calculated pChEMBL",
        comment_type="p",
    )


def flag_salt_or_solvent_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows with a salt/mixture on the canonical SMILES (not modified by CompoundMapper)"""
    return add_comment(
        df,
        comment="Salt/solvent removed",
        criteria_func=lambda x: x.str.contains(".", regex=False),
        target_column="canonical_smiles",
        comment_type="p",
    )


def flag_inter_document_duplication(
    df: pd.DataFrame,
    key_subset: list[str] = [
        "molecule_chembl_id",
        "standard_smiles",
        "canonical_smiles",
        "pchembl_value",
        "standard_relation",
        "target_chembl_id",
        "mutation",
        "target_organism",
    ],
    diff_subset: Optional[list[str]] = ["document_chembl_id"],
) -> pd.DataFrame:
    """Marks rows with a potential duplication after SMILES standardization & salt removal.

    This function only flags duplicates for discrete measurements (standard_relation='=').
    Censored measurements (e.g., '<', '>') are not flagged as duplicates since the same
    bound can be independently reached in different studies without indicating true duplication.

    Args:
        df: DataFrame to be processed.
        key_subset: metadata columns used for identifying duplicates. Defaults to a
            list of columns typically used to identify a compound readout.
        diff_subset: optional metadata columns used to identify duplicates across
            different documents. If None, it identifies only based on `key_subset`.
            Defaults to a list containing 'document_chembl_id', which is used to identify
            duplicates across different documents.

    Returns:
        pd.DataFrame: DataFrame with duplicates marked in `data_processing_comment`
                      (or `data_dropping_comment` if comment_type='d' was used).
    """
    # Only check for duplicates in discrete measurements (standard_relation='=')
    if "standard_relation" not in df.columns:
        logger.warning(
            "Column 'standard_relation' not found in DataFrame. Skipping inter-document duplication flagging."
        )
        return df

    discrete_mask = df["standard_relation"] == "="
    num_discrete = discrete_mask.sum()
    num_censored = (~discrete_mask).sum()

    if num_discrete == 0:
        logger.info(
            f"No discrete measurements (standard_relation='=') found. "
            f"All {num_censored} measurements are censored. Skipping inter-document duplication flagging."
        )
        return df

    logger.debug(
        f"Checking for inter-document duplicates among {num_discrete} discrete measurements. "
        f"Ignoring {num_censored} censored measurements."
    )

    # Only check duplicates among discrete measurements
    df_discrete = df[discrete_mask]
    dupli_mask_discrete = conflicting_duplicates(df_discrete, key_subset=key_subset, diff_subset=diff_subset)
    n_duplicates = dupli_mask_discrete.sum()

    if n_duplicates > 0:
        logger.info(
            f"Flagged {n_duplicates} duplicates with same Mol identifiers across different Documents "
            f"(only among discrete measurements with standard_relation='=')."
        )
        # Map the duplicate indices back to the original dataframe
        dupli_idxs = df_discrete[dupli_mask_discrete].index
        return (
            df.assign(temp_dupli_flag=False)
            .pipe(lambda x: x.assign(temp_dupli_flag=x.index.isin(dupli_idxs)))
            .pipe(
                add_comment,
                comment="pChEMBL Duplication Across Documents",
                criteria_func=lambda x, idxs=dupli_idxs: x.index.isin(idxs),
                target_column="temp_dupli_flag",
                comment_type="p",
            )
            .drop(columns=["temp_dupli_flag"])
        )
    else:
        logger.debug("No inter-document duplicates found among discrete measurements.")
        return df


def flag_unit_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows where unit conversion was applied to standardize measurements.

    This function flags activities that had their standard_value and standard_units
    converted to a common unit by unit conversion functions (e.g., convert_permeability_units,
    convert_molar_concentration_units, etc.). The conversion_factor column (added during
    conversion) is used to identify which rows were converted.

    If an 'original_unit' column exists (added by newer conversion functions), it will be
    used to create a more informative comment showing the original -> target unit transformation.
    Both conversion_factor and original_unit columns are removed after flagging.

    This is a processing flag (comment_type='p') to document data transformations
    for transparency.

    Args:
        df: DataFrame to be processed. Must contain 'conversion_factor' column
            if unit conversion was applied. May optionally contain 'original_unit'
            for more detailed flagging.

    Returns:
        pd.DataFrame: DataFrame with converted rows flagged in data_processing_comment.
            The conversion_factor and original_unit columns are removed after flagging.
    """
    if "conversion_factor" not in df.columns:
        logger.debug("Column 'conversion_factor' not found. Skipping unit conversion flagging.")
        return df

    mask = df["conversion_factor"].notna()
    num_converted = mask.sum()

    if num_converted > 0:
        logger.info(f"Flagging {num_converted} activities with converted units.")

        # Check if we have original_unit and standard_units for detailed comment
        has_original_unit = "original_unit" in df.columns
        has_standard_units = "standard_units" in df.columns

        if not has_original_unit or not has_standard_units:
            logger.warning(
                "Unit conversion flagging requires 'original_unit' and 'standard_units' columns. "
                "Skipping flagging. This may indicate a bug in the conversion function."
            )
        else:
            # Create row-specific comments showing original -> target unit
            for idx in df[mask].index:
                original = df.loc[idx, "original_unit"]
                target = df.loc[idx, "standard_units"]
                comment = f"Unit converted to {target} from {original}"
                df = add_comment(
                    df,
                    comment=comment,
                    criteria_func=lambda x, i=idx: x.index == i,
                    target_column="conversion_factor",
                    comment_type="p",
                )
    else:
        logger.debug("No activities with converted units found.")

    # Clean up conversion_factor column
    df = df.drop(columns=["conversion_factor"])

    # Clean up original_unit column if it exists
    if "original_unit" in df.columns:
        df = df.drop(columns=["original_unit"])

    return df
