"""Module holding functionalities for the ChEMBL API."""

from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core.pandas_helper import add_comment
from ..logger import logger
from .api.downloader import get_assay_size_sql, get_full_activity_data_sql
from .data_flag_functions import (
    flag_calculated_pchembl,
    flag_incompatible_units,
    flag_potential_duplicate,
    flag_with_data_validity_comment,
)
from .exceptions import BioactivitiesNotFoundError
from .unit_conversions import is_unit_annotation_error_diff


def convert_to_log10(df: pd.DataFrame) -> pd.DataFrame:
    """Function to be applied to the whole DataFrame. Will convert the standard_value
    column to pchembl_value column, if the standard_units are in nM, µM or uM.

    Activities with incompatible units are flagged and preserved with pchembl_value=NaN
    for transparency.

    Args:
        df: a bioactivity DataFrame. e.g.: output from `get_activity_table`.

    Returns:
        pd.DataFrame: the DataFrame with the pchembl_value column added. Activities with
            incompatible units are flagged and have pchembl_value=NaN.
    """

    def compute_log(row):
        unit = row["standard_units"]
        if pd.isna(unit):
            return np.nan

        value = row["standard_value"]
        if value == 0:  # avoid division by zero
            return np.nan
        if unit == "mM":
            value_in_M = value * 1e-3
        elif unit == "µM" or unit == "uM":
            value_in_M = value * 1e-6
        elif unit == "nM":
            value_in_M = value * 1e-9
        else:
            return np.nan  # Unit not compatible with pChEMBL calculation

        return -np.log10(value_in_M)

    desired_units = ["nM", "µM", "uM", "mM"]  # noqa: F841
    df = df.copy().pipe(flag_incompatible_units)  # flag incompatible units w/ pchembl calculation e.g.; %

    # Filter to convertible units for calculation, but preserve incompatible ones after
    convertible_mask = df["standard_units"].isin(desired_units) | df["standard_units"].isna()
    convertible_df = df[convertible_mask].copy()
    incompatible_df = df[~convertible_mask].copy()

    if "pchembl_value" not in convertible_df.columns:
        raise ValueError("pchembl_value column not found in input DataFrame.")
    elif (~convertible_df.pchembl_value.isna()).any():
        raise ValueError(
            "input DataFrame should only have pchembl_value.isna() values. If not, use "
            "chembl.processing.process_bioactivities instead."
        )

    if convertible_df.shape[0] > 0:  # Calculate pChEMBL for convertible units
        convertible_df = convertible_df.pipe(flag_calculated_pchembl)
        convertible_df = convertible_df.assign(pchembl_value=lambda x: x.apply(compute_log, axis=1))
        pchembl_inf_or_nan = convertible_df.loc[~np.isfinite(convertible_df["pchembl_value"])]
        if not pchembl_inf_or_nan.empty:
            debug_cols = [
                "target_chembl_id",
                "assay_chembl_id",
                "assay_type",
                "molecule_chembl_id",
                "standard_units",
                "standard_value",
            ]
            _info = pchembl_inf_or_nan.loc[pchembl_inf_or_nan.index[:6], debug_cols]
            comment = "Infinite or NaN pchembl_value after calculation"
            logger.info(f"Flagging {len(pchembl_inf_or_nan)} rows: {comment}:\n{_info}")
            convertible_df = add_comment(
                df=convertible_df,
                comment=comment,
                criteria_func=lambda x: x.index.isin(pchembl_inf_or_nan.index),
                target_column="pchembl_value",  # Dummy target, criteria_func handles selection
                comment_type="d",
            )

    if incompatible_df.shape[0] > 0:  # Recombine convertible and incompatible dataframes
        # Ensure incompatible_df has pchembl_value column (will be NaN)
        if "pchembl_value" not in incompatible_df.columns:
            incompatible_df["pchembl_value"] = np.nan
        result_df = pd.concat([convertible_df, incompatible_df], ignore_index=True)
    else:
        result_df = convertible_df

    return result_df


def curate_activity_pairs(
    df: pd.DataFrame,
    mol_id_col: str = "molecule_chembl_id",
    assay_id_col: str = "assay_chembl_id",
    activity_value_col: str = "pchembl_value",
) -> pd.DataFrame:
    """Curate activity pairs for the same molecule across different assays. Removes or flags pairs
    of measurements if their activity values (e.g., pChEMBL values) differ by approximately 3.0 or 6.0

    Filter inspired on Landrum & Riniker, 2024, where the authors state:

    > Given the very low probability of two separate experiments producing exactly the same
    results, the exact matches are most likely cases where values from a previous paper are
    copied into a new one; this was discussed in the earlier work by Kramer et al. (10) and
    spot-checked with a number of assay pairs here.

    Args:
        df: DataFrame with bioactivity data.
        mol_id_col: column name for molecule IDs. Defaults to "molecule_chembl_id" for ChEMBL data.
        assay_id_col: column name for assay IDs. Defaults to "assay_chembl_id" for ChEMBL data.
        activity_value_col: column name for activity values. Defaults to "pchembl_value" for ChEMBL data.

    Returns:
        pd.DataFrame: The curated DataFrame.
    """
    df = df.copy()
    if not {mol_id_col, assay_id_col, activity_value_col}.issubset(df.columns):
        logger.warning(
            "Skipping activity pair curation: Required columns "
            f"({mol_id_col}, {assay_id_col}, {activity_value_col}) not found."
        )
        return df

    if df.empty:
        logger.info("Input DataFrame is empty. Skipping activity pair curation.")
        return df

    # Prepare DataFrame for merge by adding original index as a new column
    df_for_merge = df.copy()

    # Create a unique name for the temporary column holding original index values
    temp_orig_idx_col = "__original_index__"
    _i = 0
    while temp_orig_idx_col in df_for_merge.columns:
        temp_orig_idx_col = f"__original_index__{_i}"
        _i += 1
    df_for_merge[temp_orig_idx_col] = df.index  # Use original df.index

    # Self-merge on molecule ID
    merged_df = pd.merge(df_for_merge, df_for_merge, on=mol_id_col, suffixes=("_L", "_R"))

    # Define column names for easier access
    orig_idx_col_L = temp_orig_idx_col + "_L"
    orig_idx_col_R = temp_orig_idx_col + "_R"
    assay_col_L = assay_id_col + "_L"
    assay_col_R = assay_id_col + "_R"
    activity_col_L = activity_value_col + "_L"
    activity_col_R = activity_value_col + "_R"

    # Filter for pairs:
    # 1. From different assays
    condition_diff_assays = merged_df[assay_col_L] != merged_df[assay_col_R]
    # 2. Unique pairs of original rows (avoid self-comparison and duplicate (rowA,rowB)/(rowB,rowA) pairs)
    condition_unique_rows = merged_df[orig_idx_col_L] < merged_df[orig_idx_col_R]

    valid_pairs = merged_df[condition_diff_assays & condition_unique_rows].copy()

    if valid_pairs.empty:
        logger.info(
            "No potential activity pairs found after initial structural filtering (diff assays, unique rows)."
        )
        return df

    # Handle NaNs in activity values for the pairs
    valid_pairs.dropna(subset=[activity_col_L, activity_col_R], inplace=True)

    if valid_pairs.empty:
        logger.info("No valid pairs with non-NaN activity values found for curation.")
        return df

    # Calculate absolute difference in activity values
    valid_pairs["abs_diff"] = np.abs(valid_pairs[activity_col_L] - valid_pairs[activity_col_R])

    # Flag pairs whose difference is an exact multiple of 3 log units (3.0, 6.0, 9.0, ...),
    # the signature of a metric-prefix unit-annotation error (e.g. nM reported as uM).
    problematic_pairs = valid_pairs[is_unit_annotation_error_diff(valid_pairs["abs_diff"])]

    rows_to_flag_indices = set()
    if not problematic_pairs.empty:
        indices_L = problematic_pairs[orig_idx_col_L]
        indices_R = problematic_pairs[orig_idx_col_R]
        rows_to_flag_indices.update(indices_L.unique())
        rows_to_flag_indices.update(indices_R.unique())

        for _, row_pair in problematic_pairs.iterrows():  # iterrows on small problematic_pairs df for logging
            logger.debug(
                f"Marking/flagging rows for molecule {row_pair[mol_id_col]} (indices: {row_pair[orig_idx_col_L]}, {row_pair[orig_idx_col_R]}), "
                f"assays {row_pair[assay_col_L]} (value: {row_pair[activity_col_L]}) and "
                f"{row_pair[assay_col_R]} (value: {row_pair[activity_col_R]}) "
                f"due to activity value difference being an exact multiple of 3.0 log units."
            )

    if rows_to_flag_indices:
        comment = "Unit Annotation Error"
        to_rm_idxs = list(rows_to_flag_indices)

        logger.info(f"Activity Curation: Flagging {len(to_rm_idxs)} measurements due to {comment.lower()}")
        df = add_comment(
            df=df,
            comment=comment,
            criteria_func=lambda x: x.index.isin(to_rm_idxs),
            target_column=activity_value_col,
            comment_type="d",
        )
    else:
        logger.info("No activity pairs found meeting the curation criteria (activity diff ~3.0 | ~6.0).")

    return df


def process_bioactivities(
    bioactivities_df: pd.DataFrame,
    calculate_pchembl: bool = True,
    curate_annotation_errors: bool = True,
    require_document_date: bool = False,
    value_col: str = "pchembl_value",
) -> pd.DataFrame:
    """Processes the bioactivities DataFrame. Will convert the standard_value
    column to pchembl_value column if the standard_units are in mM, µM, uM, or nM. If the
    standard_units are in log, the original value in the pchembl_value is preserved.

    Args:
        bioactivities_df: bioactivity dataframe, e.g.: output from `get_activity_table`.
        calculate_pchembl: Whether to calculate pChEMBL values. When aggregating on
            standard_value (value_col != "pchembl_value"), this enables calculation of
            pchembl_value for compatible units (e.g., nM, µM, mM). Though those are available
            for high quality data, censored data does not have a pChEMBL value readily available
            and they'll need to be calculated.
        curate_annotation_errors: Whether to apply activity curation based on pChEMBL values
            diverging in exactly 3.0 (indicate possible annotation errors). Defaults to True.
        require_document_date: Whether to filter out activities without a document year.
        value_col: Column name for aggregation values. Defaults to "pchembl_value".
            When set to "standard_value", pchembl_value filtering is skipped.

    Returns:
        pd.DataFrame: the processed bioactivities DataFrame.
    """
    bioactivities_df = bioactivities_df.astype({"standard_value": "float32", "pchembl_value": "float32"})
    if int(pd.__version__.split(".", maxsplit=1)[0]) < 3:
        with pd.option_context("future.no_silent_downcasting", True):
            bioactivities_df = bioactivities_df.replace({None: np.nan}).infer_objects(copy=False)
    else:
        bioactivities_df = bioactivities_df.replace({None: np.nan}).infer_objects()
    bioactivities_df = (
        bioactivities_df.pipe(flag_with_data_validity_comment)
        # .query("data_validity_comment.isna()")
        .pipe(flag_potential_duplicate)
        # .query("potential_duplicate == 0")
        .drop(columns=["data_validity_comment", "potential_duplicate"])
    )

    # When aggregating on pchembl_value (default), opportunistically calculate missing values.
    # When aggregating on standard_value, skip this filtering to avoid unnecessary work.
    if value_col == "pchembl_value":
        # Separate activities with and without pChEMBL values
        with_pchembl = bioactivities_df.query("~pchembl_value.isna()").copy()
        without_pchembl = bioactivities_df.query("pchembl_value.isna()").copy()

        if calculate_pchembl and not without_pchembl.empty:
            # Calculate pChEMBL for activities without it (preserves incompatible units)
            without_pchembl = convert_to_log10(without_pchembl)
            bioactivities_df = pd.concat([with_pchembl, without_pchembl], ignore_index=True)
        elif not calculate_pchembl and not without_pchembl.empty:
            # Keep activities without pChEMBL (incompatible units already flagged)
            bioactivities_df = pd.concat([with_pchembl, without_pchembl], ignore_index=True)
        else:
            # Either calculate_pchembl=True and no missing values, or calculate_pchembl=False and all have values
            bioactivities_df = with_pchembl if without_pchembl.empty else bioactivities_df
    else:
        # When aggregating on non-pchembl values (e.g., standard_value), still opportunistically
        # calculate pchembl_value for compatible units if requested, but don't filter based on it
        if calculate_pchembl:
            without_pchembl = bioactivities_df.query("pchembl_value.isna()").copy()
            if not without_pchembl.empty:
                # Calculate pChEMBL for activities without it (preserves incompatible units)
                without_pchembl = convert_to_log10(without_pchembl)
                with_pchembl = bioactivities_df.query("~pchembl_value.isna()").copy()
                bioactivities_df = pd.concat([with_pchembl, without_pchembl], ignore_index=True)

    if curate_annotation_errors:
        bioactivities_df = curate_activity_pairs(bioactivities_df)

    if require_document_date:
        if "year" not in bioactivities_df.columns:
            logger.warning(
                "Document date curation enabled, but 'year' column not found. Skipping this curation."
            )
        else:
            original_count = len(bioactivities_df)
            bioactivities_df = bioactivities_df[bioactivities_df["year"].notna()]
            removed_count = original_count - len(bioactivities_df)
            if removed_count > 0:
                logger.info(
                    f"Document Date Curation: Removed {removed_count} measurements lacking a document year."
                )

    return bioactivities_df.reset_index(drop=True)


def get_bioactivities_workflow(
    molecule_chembl_ids: Optional[Union[list, str]] = None,
    target_chembl_ids: Optional[Union[list, str]] = None,
    assay_chembl_ids: Optional[Union[list, str]] = None,
    document_chembl_ids: Optional[Union[list, str]] = None,
    standard_relation: Optional[List[str]] = None,
    standard_type: Optional[List[str]] = None,
    standard_units: Optional[List[str]] = None,
    confidence_scores: Union[list, Tuple] = (9, 8),
    assay_types: Union[list, Tuple] = ("B", "F"),
    chembl_release: Optional[int] = None,
    additional_fields: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    calculate_pchembl: bool = False,
    curate_annotation_errors: bool = True,
    require_document_date: bool = False,
    backend: Literal["downloader", "webresource"] = "downloader",
    value_col: str = "pchembl_value",
):
    """Perform the first step of the bioactivity data retrieval workflow. These are:

    1. Get ChEMBL data using any of the input identifiers: molecule_chembl_ids, target_chembl_ids,
    assay_chembl_ids, or document_chembl_ids. Additional filters are supported by other input
    parameters.

    2. Once the data is retrieved, the bioactivities are processed according to the `calculate_pchembl`
    parameter. ChEMBL calculates pChEMBL values for activity data with the following criteria:

    - "standard_type" in "[IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50"];
    - "standard_relation" == "=" & "standard_units" == "nM";
    - "standard_value" > 0 & "data_validity_comment"].isnull()) | "data_validity_comment" == "Manually validated"

    By passing this parameter to True, pchembl values will be calculated for bioactivities reported in
    nM, µM or uM `standard_unit`, or -Log|Log `standard_type`.

    3. The first quality filter is applied. Data containing "data_validity_description" or
    "potential_duplicate" flags are immediatelly removed from the DataFrame.

    Args:
        molecule_chembl_ids: list of ChEMBL molecule IDs to fetch data for. Defaults to None.
        target_chembl_ids: list of ChEMBL target IDs to fetch data for. Defaults to None.
        assay_chembl_ids: list of ChEMBL assay IDs to fetch data for. Defaults to None.
        document_chembl_ids: list of ChEMBL document IDs to fetch data for. Defaults to None.
        standard_relation: Optional filter for standard relation types (e.g., ["=", "<", ">"])
        standard_type: Optional filter for activity types (e.g., ["IC50", "Ki", "EC50"])
        confidence_scores: list of confidence scores to filter the fetched assay data.
            Defaults to (9, 8).
        assay_types: list of assay types to be fetched from ChEMBL. Defaults to binding (B) and
            functional (F) data.
        chembl_release: Not to confuse for `version`. This is the ChEMBL release number used to
            filter the data. Defaults to None.
        additional_fields: `backend=="downloader"` only! "Optional list of additional fields to
            include in the sql query. E.g.: ["vs.sequence"], to retrieve the sequence of the
            variant, if available. Defaults to None.
        prefix: `backend=="downloader"` only! prefix to be used by pystow for storing the data
            on a custom directory. Defaults to None.
        version: `backend=="downloader"` only! version of the ChEMBL database to be downloaded by
            chembl_downloader. If left as None, the latest version will be downloaded. Defaults to None.
        curate_annotation_errors: Whether to apply activity curation based on pChEMBL values diverging
            in exactly 3.0 (indicate possible annotation errors). Defaults to True.
        calculate_pchembl: calculate pChEMBL values for bioactivities reported in nM, µM or uM `standard_unit`
            or -Log|Log `standard_type`. Defaults to False
        backend: the backend to be used for fetching the data. If downloader, the ChEMBL sql database
            is downloaded and extracted first. Defaults to "downloader".
        value_col: Column name for values to be used during aggregation. Defaults to "pchembl_value".
            When set to "standard_value", pchembl filtering is skipped and pchembl values are
            only calculated opportunistically for compatible units. Defaults to "pchembl_value".

        Raises:
            BioactivitiesNotFoundError: If the retrieved bioactivity dataframe is empty.
    """
    if backend == "downloader":
        bioactivities_df = get_full_activity_data_sql(
            molecule_chembl_ids=molecule_chembl_ids,
            target_chembl_ids=target_chembl_ids,
            assay_chembl_ids=assay_chembl_ids,
            document_chembl_ids=document_chembl_ids,
            standard_relation=standard_relation,
            standard_type=standard_type,
            standard_units=standard_units,
            confidence_scores=confidence_scores,
            assay_types=assay_types,
            chembl_release=chembl_release,
            additional_fields=additional_fields,
            prefix=prefix,
            version=version,
        )
        assay_sizes = get_assay_size_sql(
            assay_chembl_ids=bioactivities_df.assay_chembl_id.unique().tolist(),
            prefix=prefix,
            version=version,
        )
        bioactivities_df = bioactivities_df.merge(assay_sizes, on="assay_chembl_id")
    elif backend == "webresource":
        from .api.webresource import get_full_activity_data

        bioactivities_df = get_full_activity_data(
            molecule_chembl_ids=molecule_chembl_ids,
            target_chembl_ids=target_chembl_ids,
            assay_chembl_ids=assay_chembl_ids,
            document_chembl_ids=document_chembl_ids,
            confidence_scores=confidence_scores,
            assay_types=assay_types,
            chembl_release=chembl_release,
        ).sort_values(by=["molecule_chembl_id", "activity_id", "standard_value"])

    if bioactivities_df.empty:
        raise BioactivitiesNotFoundError(
            "No bioactivities found for the given query: "
            f"molecule_chembl_ids={molecule_chembl_ids}, "
            f"target_chembl_ids={target_chembl_ids}, "
            f"assay_chembl_ids={assay_chembl_ids}, "
            f"document_chembl_ids={document_chembl_ids}, "
        )

    bioactivities_df = process_bioactivities(
        bioactivities_df,
        calculate_pchembl=calculate_pchembl,
        curate_annotation_errors=curate_annotation_errors,
        require_document_date=require_document_date,
        value_col=value_col,
    )

    if bioactivities_df.empty:
        raise BioactivitiesNotFoundError(
            "No bioactivities found after calculating pchembl_values. To investigate, use "
            "either methods CompoundMapper.chembl.api.downloader.get_full_activity_data_sql or "
            "CompoundMapper.chembl.api.webresource.get_full_activity_data."
        )

    return bioactivities_df
