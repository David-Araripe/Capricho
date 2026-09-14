from inspect import signature
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer
from job_tqdflex import ParallelApplier
from rdkit import Chem
from rdkit.Chem import inchi
from tqdm.auto import tqdm

from ..chembl.data_flag_functions import (
    flag_censored_activity_comment,
    flag_insufficient_assay_overlap,
    flag_inter_document_duplication,
    flag_max_assay_size,
    flag_min_assay_size,
    flag_missing_canonical_smiles,
    flag_missing_document_date,
    flag_salt_or_solvent_removal,
    flag_strict_mutant_assays,
    flag_to_remove_mixture_compounds,
    flag_undefined_stereochemistry,
    flag_unit_conversion,
)
from ..chembl.exceptions import BioactivitiesNotFoundError
from ..chembl.processing import get_bioactivities_workflow
from ..chembl.unit_conversions import (
    convert_dose_units,
    convert_mass_concentration_units,
    convert_molar_concentration_units,
    convert_permeability_units,
    convert_time_units,
)
from ..core.default_fields import (
    ASSAY_ID,
    DATA_DROPPING_COMMENT,
    DATA_PROCESSING_COMMENT,
    MOLECULE_ID,
    SHARED_IDENTIFIER_GROUP,
    TARGET_ID,
)
from ..core.fp_utils import calculate_mixed_FPs
from ..core.pandas_helper import assign_shared_identifier_groups, save_dataframe
from ..core.smiles_utils import clean_mixtures
from ..core.stats_make import process_repeat_mols, repeated_indices_from_array_series
from ..core.stereo import find_undefined_stereocenters
from ..logger import logger

CompoundEqualityMethod = Literal["mixed_fp", "connectivity", "inchi", "inchikey", "smiles"]
InChIIdentifier = Literal["connectivity", "inchi", "inchikey"]
FULL_INCHI_IDENTIFIERS = {"inchi", "inchikey"}
INCHI_IDENTIFIERS = {"connectivity", *FULL_INCHI_IDENTIFIERS}
STEREO_SENSITIVE_EQUALITY_METHODS = {"smiles", *FULL_INCHI_IDENTIFIERS}

# when aggregated, some `activity_id` values will be strings and sorting won't work properly
AGGREGATE_SAVE_SORTED_BY = ["target_chembl_id", "assay_chembl_id"]


def _compound_identifier_column(compound_equality: CompoundEqualityMethod) -> str:
    """Return the output column representing the selected compound identity."""
    # Fingerprints are not persisted in tabular output, so mixed_fp continues to use
    # connectivity as its inspectable downstream identifier.
    return "connectivity" if compound_equality == "mixed_fp" else compound_equality


def _convert_smiles_to_identifier(
    smiles: list,
    identifier: InChIIdentifier,
    *,
    progress: bool = True,
) -> list:
    """Convert SMILES without serializing RDKit molecules to worker processes.

    InChI generation is implemented in C++ and is fast enough that joblib's process and
    RDKit-molecule pickling overhead dominates for the dataset sizes used by CAPRICHO.
    """
    if identifier not in INCHI_IDENTIFIERS:
        raise ValueError(f"Invalid InChI identifier: {identifier}")

    values = tqdm(smiles, desc=f"Converting to {identifier}") if progress else smiles
    identifiers = []
    for smiles_value in values:
        if smiles_value is None:
            identifiers.append(None)
            continue
        mol = Chem.MolFromSmiles(smiles_value)
        if mol is None:
            identifiers.append(None)
        elif identifier == "inchi":
            identifiers.append(inchi.MolToInchi(mol))
        else:
            inchikey = Chem.MolToInchiKey(mol)
            identifiers.append(inchikey if identifier == "inchikey" else inchikey.split("-")[0])
    return identifiers


def _identifier_map(smiles: pd.Series, identifier: InChIIdentifier) -> dict:
    """Calculate an identifier once per distinct SMILES string."""
    unique_smiles = smiles.drop_duplicates().tolist()
    identifiers = _convert_smiles_to_identifier(unique_smiles, identifier)
    return dict(zip(unique_smiles, identifiers))


def _connectivity_from_identifier(identifier, identifier_type: InChIIdentifier):
    """Derive connectivity without parsing the molecule again."""
    if identifier is None or pd.isna(identifier):
        return None
    if identifier_type == "inchi":
        return inchi.InchiToInchiKey(identifier).split("-")[0]
    if identifier_type == "inchikey":
        return identifier.split("-")[0]
    return identifier


def _assign_output_compound_identifiers(
    df: pd.DataFrame,
    compound_equality: CompoundEqualityMethod,
    identifier_by_smiles: Optional[dict] = None,
) -> pd.DataFrame:
    """Add inspectable compound identifiers without recalculating cached values."""
    result = df.copy()
    selected_identifier = compound_equality if compound_equality in INCHI_IDENTIFIERS else "connectivity"

    if identifier_by_smiles is None:
        identifier_by_smiles = {}
    result[selected_identifier] = result["smiles"].map(identifier_by_smiles).astype(object)
    missing_mask = result[selected_identifier].isna()
    if missing_mask.any():
        missing_smiles = result.loc[missing_mask, "smiles"]
        missing_identifiers = _identifier_map(missing_smiles, selected_identifier)
        result.loc[missing_mask, selected_identifier] = missing_smiles.map(missing_identifiers)

    if selected_identifier in FULL_INCHI_IDENTIFIERS:
        result["connectivity"] = result[selected_identifier].map(
            lambda value: _connectivity_from_identifier(value, selected_identifier)
        )
    return result


def _finalize_aggregated_output(
    df: pd.DataFrame,
    compound_equality: CompoundEqualityMethod,
    extra_id_cols: list[str],
    identifier_by_smiles: Optional[dict] = None,
) -> pd.DataFrame:
    """Add identifiers, order output columns, and assign shared-identifier groups."""
    comment_columns = [DATA_PROCESSING_COMMENT, DATA_DROPPING_COMMENT]
    for column in comment_columns:
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' is required in the DataFrame but is missing. "
                "Please ensure that the DataFrame contains all necessary columns."
            )
    result = _assign_output_compound_identifiers(
        df,
        compound_equality,
        identifier_by_smiles=identifier_by_smiles,
    )
    identifier_columns = [
        "connectivity",
        *([compound_equality] if compound_equality in FULL_INCHI_IDENTIFIERS else []),
    ]
    first_columns = [*identifier_columns, *extra_id_cols, "smiles"]
    last_columns = result.columns.difference(first_columns + comment_columns).tolist() + comment_columns
    result = result[[*first_columns, *last_columns]]
    result = result.sort_values(AGGREGATE_SAVE_SORTED_BY).reset_index(drop=True)
    selected_identifier = _compound_identifier_column(compound_equality)
    return assign_shared_identifier_groups(result, key_columns=(selected_identifier, "target_chembl_id"))


def _log_pipeline_summary(
    df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    pre_aggregation_count: int,
) -> None:
    """Log a structured summary of the pipeline run.

    Reports how much data carries each quality flag, how much would be omitted if every
    flag were dropped, and how much of the aggregated data is measured in more than one
    assay and can therefore be compared across assays.

    Args:
        df: The pre-aggregation DataFrame (one row per measurement).
        aggregated_df: The aggregated DataFrame (one row per compound-target readout).
        pre_aggregation_count: Rows carried into aggregation.
    """
    from ..flag_report import (
        format_cross_assay_coverage,
        format_flag_summary,
        summarize_cross_assay_coverage,
        summarize_flags,
    )

    lines = ["", "PIPELINE SUMMARY"]

    lines.append(f"  Measurements before aggregation: {pre_aggregation_count:>8,}")
    lines.append(f"  Aggregated datapoints:           {len(aggregated_df):>8,}")

    if len(df) > 0:
        # Pre-aggregation df uses molecule_chembl_id; post-aggregation uses connectivity
        cpd_col = "connectivity" if "connectivity" in df.columns else MOLECULE_ID
        n_compounds = df[cpd_col].nunique() if cpd_col in df.columns else 0
        n_targets = df[TARGET_ID].nunique() if TARGET_ID in df.columns else 0
        n_assays = df[ASSAY_ID].nunique() if ASSAY_ID in df.columns else 0
        lines.append(f"  Unique compounds:                {n_compounds:>8,}")
        lines.append(f"  Unique targets:                  {n_targets:>8,}")
        lines.append(f"  Unique assays:                   {n_assays:>8,}")

    total = len(df)

    if ASSAY_ID in aggregated_df.columns and len(aggregated_df) > 0:
        coverage = summarize_cross_assay_coverage(aggregated_df, n_retrieved=total or None)
        lines.append("")
        lines.append(format_cross_assay_coverage(coverage))

    # Quality flags: the union is the data omitted if every flag is dropped.
    if DATA_DROPPING_COMMENT in df.columns and total > 0:
        lines.append("")
        lines.append(
            format_flag_summary(
                summarize_flags(df, comment_column=DATA_DROPPING_COMMENT),
                title=(
                    f"QUALITY FLAGS ({DATA_DROPPING_COMMENT}) "
                    f"— share of {total:,} measurements before aggregation"
                ),
            )
        )

    # Processing flags record what was changed, not what would be removed.
    if DATA_PROCESSING_COMMENT in df.columns and total > 0:
        lines.append("")
        lines.append(
            format_flag_summary(
                summarize_flags(df, comment_column=DATA_PROCESSING_COMMENT),
                title=(
                    f"PROCESSING FLAGS ({DATA_PROCESSING_COMMENT}) "
                    f"— share of {total:,} measurements before aggregation"
                ),
            )
        )

    logger.info("\n".join(lines))


# after the workflow, `activity_id` is an integer, so we can sort by it to ensure consistent
# ordering on the aggregated datapoints -> assay1|assay2|...|assayN,activity_id1|...|activity_idN
WORKFLOW_SAVE_SORTED_BY = [*AGGREGATE_SAVE_SORTED_BY, "activity_id"]


def _warn_info_post_aggregation_repeats(
    df: pd.DataFrame,
    extra_id_cols: list[str],
    aggregate_mutants: bool = False,
    value_col: str = "pchembl_value",
    compound_equality: CompoundEqualityMethod = "connectivity",
    _limit: int = 30,
    _sample_rows: int = 5,
) -> None:
    """Report unexpected full-key duplicates and intentionally shared downstream IDs."""

    def _truncate_dataframe(data: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Truncate DataFrame values to a specified length."""

        def truncate(value):
            text = str(value)
            return text[:limit] + "..." if len(text) > limit else text

        if pd.__version__ > "2.1.0":  # applymap got deprecated in 2.1.0
            return data.map(truncate)
        return data.applymap(truncate)

    def _sample_complete_groups(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Return complete repeated-identifier groups up to the row limit when possible."""
        group_sizes = data[SHARED_IDENTIFIER_GROUP].value_counts(sort=False)
        selected_groups = []
        selected_rows = 0
        for group_id, group_size in group_sizes.items():
            if selected_groups and selected_rows + group_size > _sample_rows:
                break
            selected_groups.append(group_id)
            selected_rows += group_size
            if selected_rows >= _sample_rows:
                break
        sample = data[data[SHARED_IDENTIFIER_GROUP].isin(selected_groups)]
        truncated = len(sample) > _sample_rows
        return sample.head(_sample_rows), truncated

    selected_identifier = _compound_identifier_column(compound_equality)
    shared_key = (selected_identifier, "target_chembl_id")
    df = assign_shared_identifier_groups(df, key_columns=shared_key)

    aggregation_compound_key = "smiles" if compound_equality == "mixed_fp" else selected_identifier
    if aggregate_mutants:
        aggregation_key = [
            aggregation_compound_key,
            "target_chembl_id",
            *extra_id_cols,
            "standard_relation",
        ]
    else:
        aggregation_key = [
            aggregation_compound_key,
            "mutation",
            "target_chembl_id",
            *extra_id_cols,
            "standard_relation",
        ]
    aggregation_key = list(dict.fromkeys(column for column in aggregation_key if column in df.columns))

    value_mean_col = f"{value_col}_mean"
    display_columns = list(
        dict.fromkeys(
            [
                SHARED_IDENTIFIER_GROUP,
                selected_identifier,
                "target_chembl_id",
                *([] if selected_identifier == "smiles" else ["smiles"]),
                *([] if aggregate_mutants else ["mutation"]),
                *extra_id_cols,
                "standard_relation",
                "molecule_chembl_id",
                "assay_chembl_id",
                value_mean_col,
            ]
        )
    )
    display_columns = [column for column in display_columns if column in df.columns]

    # Rows should not remain repeated after applying CAPRICHO's configured aggregation key.
    # A censored value is itself part of that key; exact measurements aggregate regardless
    # of value and therefore share one sentinel here.
    aggregation_ids = df.loc[:, aggregation_key].copy()
    if "standard_relation" in df.columns and value_mean_col in df.columns:
        is_censored = df["standard_relation"].fillna("=").ne("=")
        aggregation_ids["_censored_value"] = df[value_mean_col].where(is_censored, "__exact__")
    unexpected_mask = aggregation_ids.duplicated(keep=False)
    if unexpected_mask.any():
        unexpected = df.loc[unexpected_mask, display_columns].sort_values(
            by=[
                column
                for column in ["target_chembl_id", selected_identifier, value_mean_col]
                if column in df.columns
            ],
            ascending=True,
        )
        logger.warning(
            f"Unexpectedly found {unexpected_mask.sum():,} rows sharing CAPRICHO's aggregation key "
            f"{aggregation_key} (and censored value where applicable). These rows should normally have been "
            "combined; please inspect the dataset. "
            "Sample rows (including every member where the display limit permits):\n"
            f"{_truncate_dataframe(unexpected.head(_sample_rows), _limit).to_string(index=False)}"
        )

    shared_mask = df[SHARED_IDENTIFIER_GROUP].notna()
    if not shared_mask.any():
        return

    shared_rows = df.loc[shared_mask]
    n_groups = shared_rows[SHARED_IDENTIFIER_GROUP].nunique()
    combination_word = "combination" if n_groups == 1 else "combinations"
    key_candidates = [
        *([] if aggregate_mutants else ["mutation"]),
        *extra_id_cols,
        "standard_relation",
    ]
    key_candidates = list(dict.fromkeys(column for column in key_candidates if column in shared_rows.columns))
    grouped = shared_rows.groupby(SHARED_IDENTIFIER_GROUP, dropna=False)
    varying_fields = [
        column for column in key_candidates if grouped[column].nunique(dropna=False).gt(1).any()
    ]
    if not varying_fields:
        fallback_candidates = [
            column for column in ["smiles", value_mean_col] if column in shared_rows.columns
        ]
        varying_fields = [
            column for column in fallback_candidates if grouped[column].nunique(dropna=False).gt(1).any()
        ]
    varying_text = ", ".join(f"`{column}`" for column in varying_fields) or "preserved readout fields"
    prepare_fields = [column for column in varying_fields if column in key_candidates]
    if prepare_fields:
        action = f"add `--id-columns {','.join(prepare_fields)}` to `capricho prepare`"
    else:
        action = "include the relevant distinguishing fields in the downstream task ID"
    compound_action = (
        f" Use `--compound-col {selected_identifier}` in `capricho prepare` to retain this compound identity."
        if selected_identifier != "connectivity"
        else ""
    )

    shared_display_columns = list(
        dict.fromkeys(
            [
                SHARED_IDENTIFIER_GROUP,
                selected_identifier,
                "target_chembl_id",
                *varying_fields,
                value_mean_col,
            ]
        )
    )
    shared_display_columns = [column for column in shared_display_columns if column in shared_rows.columns]
    shared_sample = shared_rows.loc[:, shared_display_columns].copy()
    if value_mean_col in shared_sample.columns:
        shared_sample[value_mean_col] = shared_sample[value_mean_col].round(3)
    shared_sample = shared_sample.sort_values(
        by=(
            [SHARED_IDENTIFIER_GROUP, value_mean_col]
            if value_mean_col in shared_rows.columns
            else [SHARED_IDENTIFIER_GROUP]
        ),
        ascending=True,
    )
    shared_sample, sample_truncated = _sample_complete_groups(shared_sample)
    truncation_note = " (the displayed group is truncated)" if sample_truncated else ""
    logger.info(
        f"CAPRICHO found {len(shared_rows):,} separate activity rows with {n_groups:,} repeated "
        f"`{selected_identifier}` + `target_chembl_id` {combination_word}; varying fields: {varying_text}. "
        f"These may be valid distinct readouts. `{SHARED_IDENTIFIER_GROUP}` labels rows sharing a combination "
        f"(NaN otherwise); group sizes: data.{SHARED_IDENTIFIER_GROUP}.value_counts(). If downstream tasks use "
        f"target only, {action} or resolve these combinations first.{compound_action} Sample"
        f"{truncation_note}:\n{_truncate_dataframe(shared_sample, _limit).to_string(index=False)}"
    )


def get_standardize_and_clean_workflow(
    molecule_ids: Optional[list[str]] = None,
    target_ids: Optional[list[str]] = None,
    assay_ids: Optional[list[str]] = None,
    document_ids: Optional[list[str]] = None,
    chirality: bool = True,
    calculate_pchembl: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    confidence_scores: list[str] = [7, 8, 9],
    bioactivity_type: Optional[list[str]] = None,
    standard_relation: list[str] = ["="],
    standard_units: Optional[list[str]] = None,
    assay_types: list[str] = ["B", "F"],
    chembl_release: Optional[int] = None,
    save_not_aggregated: bool = True,
    drop_unassigned_chiral: bool = False,
    version: Optional[Union[int, str]] = None,
    backend: Literal["downloader", "webresource"] = "downloader",
    curate_annotation_errors: bool = True,
    require_doc_date: bool = False,
    min_assay_size: Optional[int] = None,
    max_assay_size: Optional[int] = None,
    min_assay_overlap: int = 0,
    strict_mutant_removal: bool = False,
    value_col: str = "pchembl_value",
    enable_unit_conversion: bool = False,
) -> pd.DataFrame:  # Changed return type annotation to pd.DataFrame
    """Fetched the filtered data from ChEMBL based on the provided IDs, assay confidence,
    and bioactivity types. The fetched smiles are then standardized and chemical mixtures
    are removed from the dataset. Duplicate data is also removed and the remaining data
    is saved to a csv file.

    Args:
        molecule_ids: list of ChEMBL molecule IDs to filter data from
        target_ids: list of ChEMBL target IDs to filter data from
        assay_ids: list of ChEMBL assay IDs to filter data from
        document_ids: list of ChEMBL document IDs to filter data from
        calculate_pchembl: whether to calculate pchembl values when not found for assay
            results reported in nanomolar/micromolar units
        chirality: setting this to False will remove stereochemistry information from the
            SMILES on top of the standardization. Defaults to True.
        output_path: path to save the resulting csv file
        confidence_scores: list of confidence scores (assay-related) to filter data from
        bioactivity_type: list of bioactivity types (assay-related) to filter data from
        standard_relation: standard relation to filter data from. Currently only supports "="
            Defaults to "=".
        chembl_release: latest ChEMBL release to retrieve data from
        save_not_aggregated: whether to save the resulting data to the csv (output_path) before
        drop_unassigned_chiral: whether to drop data points with undefined stereocenters. Defaults to False.
        version: `backend=="downloader"` only! version of the ChEMBL database to be downloaded by
            chembl_downloader. If left as None, the latest version will be downloaded. Defaults to None.
        backend: the backend to be used for fetching the data. If downloader, the ChEMBL sql database
            is downloaded and extracted first. Defaults to "downloader".
        curate_annotation_errors: Whether to apply activity curation based on pChEMBL values diverging
            in exactly 3.0 (indicate possible annotation errors). Defaults to True.
        require_doc_date: Whether to filter out activities without a document year.
        max_assay_size: Minimum number of compounds in an assay. Assays smaller than this size will
            have their activities flagged for removal. Defaults to None (no filtering).
        max_assay_size: Maximum number of compounds in an assay. Assays exceeding this size will
            have their activities flagged for removal. Defaults to None (no filtering).
        min_assay_overlap: Minimum number of overlapping compounds between two assays for the same target
            for their activities to be considered. Defaults to 0 (no filtering).
        strict_mutant_removal: If True, assays with 'mutant', 'mutation', or 'variant' in their
            description will be flagged for removal. Defaults to False.

    Returns:
        pd.DataFrame: the filtered, standardized, and cleaned data
    """
    if output_path is not None:
        if isinstance(output_path, str):
            output_path = Path(output_path)

    # -log | log transformed values reported as Log XC50, -Log XC50, etc, might not
    # have a pchembl value, but *could* still be used.  If standard_type contains `Log`,
    # the standard_value will be transferred to pchembl_value.
    if bioactivity_type is None:
        # No filter on standard_type - fetch all
        biotypes = None
    elif calculate_pchembl:
        biotypes = []
        for act in bioactivity_type:
            biotypes.extend([f"Log {act}", f"-Log {act}", act])
    else:
        if standard_relation != ["="]:
            logger.error(
                "pchembl_values are only calculated for standard_relation='='. If you want "
                "to use censored data, please set `calculate_pchembl` to True with the flag "
                "--calculate-pchembl."
            )
        biotypes = bioactivity_type

    # get_bioactivities_workflow -> fetch with either webresource or downloader -> (minimally) process bioactivities
    # -> standardization is done here -> curate bioactivity errors (if curate_annotation_errors=True)
    full_df = get_bioactivities_workflow(
        molecule_chembl_ids=molecule_ids or None,
        target_chembl_ids=target_ids or None,
        assay_chembl_ids=assay_ids or None,
        document_chembl_ids=document_ids or None,
        confidence_scores=confidence_scores,
        assay_types=assay_types,
        standard_relation=standard_relation,
        standard_type=biotypes,
        standard_units=standard_units,
        calculate_pchembl=calculate_pchembl,
        curate_annotation_errors=curate_annotation_errors,
        require_document_date=require_doc_date,
        chembl_release=chembl_release,
        version=version,
        backend=backend,
        value_col=value_col,
    )

    # Flag activities without document dates for transparency
    # Note: if require_doc_date=True, these will be hard-filtered in process_bioactivities
    full_df = flag_missing_document_date(full_df)

    full_df = flag_censored_activity_comment(full_df)

    # Convert units if requested
    if enable_unit_conversion:
        logger.info("Converting units to standard formats")

        full_df = convert_permeability_units(  # Convert to 10^-6 cm/s
            full_df,
            value_col=value_col,
            unit_col="standard_units",
        )
        full_df = flag_unit_conversion(full_df)

        full_df = convert_molar_concentration_units(  # Convert to nM
            full_df,
            value_col=value_col,
            unit_col="standard_units",
        )
        full_df = flag_unit_conversion(full_df)

        full_df = convert_mass_concentration_units(  # Convert to ug/mL
            full_df,
            value_col=value_col,
            unit_col="standard_units",
        )
        full_df = flag_unit_conversion(full_df)

        full_df = convert_dose_units(  # Convert to mg/kg
            full_df,
            value_col=value_col,
            unit_col="standard_units",
        )
        full_df = flag_unit_conversion(full_df)

        full_df = convert_time_units(  # Conver to hr
            full_df,
            value_col=value_col,
            unit_col="standard_units",
        )
        full_df = flag_unit_conversion(full_df)

    # Filter on the unchanged source relation.
    if "standard_relation" in full_df.columns and standard_relation is not None:
        excluded_relations = ~full_df["standard_relation"].isin(standard_relation)
        num_excluded = excluded_relations.sum()
        if num_excluded > 0:
            logger.info(
                f"Filtering out {num_excluded} activities with standard_relation not in {standard_relation}. "
                "These activities will be flagged for removal and saved to the _removed_subset file."
            )
            # Flag these activities for removal
            full_df.loc[excluded_relations, DATA_DROPPING_COMMENT] = full_df.loc[
                excluded_relations, DATA_DROPPING_COMMENT
            ].fillna("") + (
                full_df.loc[excluded_relations, DATA_DROPPING_COMMENT]
                .apply(lambda x: "; " if x and str(x).strip() else "")
                .fillna("")
                + f"Standard relation not in selected values {standard_relation}"
            )

    # Filter assays by size
    if min_assay_size is not None or max_assay_size is not None:
        logger.info(
            f"Filtering assays by size: min={min_assay_size}, max={max_assay_size}. "
            "Assays with insufficient size will be flagged for removal."
        )
        full_df = full_df.pipe(flag_min_assay_size, min_assay_size=min_assay_size).pipe(
            flag_max_assay_size, max_assay_size=max_assay_size
        )

    # Filter by minimum assay overlap
    if min_assay_overlap > 0 and not full_df.empty:
        logger.info(f"Filtering assays based on minimum overlap of {min_assay_overlap} compounds.")
        full_df = flag_insufficient_assay_overlap(
            df=full_df,
            min_overlap=min_assay_overlap,
            molecule_col=MOLECULE_ID,
            assay_col=ASSAY_ID,
            target_col=TARGET_ID,
            comment_col=DATA_DROPPING_COMMENT,
        )

    # Columns to remove after standardization
    # Note: standard_value and standard_units are always preserved as multivalue columns
    cols_to_remove_post_standardization = [
        "type",
        "relation",
        "units",
        "value",
        "type",
        # "description",
    ]

    logger.debug(f"All fetched bioactivity types from ChEMBL: {full_df.standard_type.unique().tolist()}")
    logger.debug(f"Filtering for bioactivity types: {biotypes}")

    # drop rows without chemical structures
    no_smiles_mask = full_df.canonical_smiles.isna()
    if no_smiles_mask.any():
        _info = full_df[no_smiles_mask].iloc[:, :6]
        logger.info(f"Dropping rows with missing canonical smiles:\n{_info}")
        full_df = full_df.drop(index=_info.index).reset_index(drop=True)

    stdzer = ChemStandardizer(
        from_smi=True, n_jobs=8, verbose=False, isomeric=chirality, progress=True, chunk_size=1000
    )

    # Filter by bioactivity_type only if it's not None
    if bioactivity_type is not None:
        df = full_df.query("standard_type.isin(@bioactivity_type)")
    else:
        df = full_df.copy()

    # A compound can have many activity rows. Standardization is deterministic, so run
    # the expensive ChEMBL pipeline once per distinct source structure and map it back.
    unique_canonical_smiles = df["canonical_smiles"].drop_duplicates().tolist()
    standardized_by_smiles = dict(zip(unique_canonical_smiles, stdzer(unique_canonical_smiles)))

    df = (
        df
        # standardize the smiles & clean possible solvents & salts from the string
        .pipe(flag_missing_canonical_smiles)
        .assign(standard_smiles=lambda x: x["canonical_smiles"].map(standardized_by_smiles))
        .dropna(subset=["standard_smiles"])  # drop if no structure is found
        .pipe(flag_salt_or_solvent_removal)
        .assign(final_smiles=lambda x: x["standard_smiles"].apply(clean_mixtures))
        .drop(columns="standard_smiles")
        .rename(columns={"final_smiles": "standard_smiles"})
        .drop(columns=[c for c in cols_to_remove_post_standardization if c in full_df.columns])
        .reset_index(drop=True)
        .copy()
    )

    # make sure we don't have Nan, can result from merging pChEMBL-lacking calculated values
    df[DATA_PROCESSING_COMMENT] = df[DATA_PROCESSING_COMMENT].fillna("")

    # Raise error if df is empty after critical processing steps
    if df.empty:
        func_params = signature(get_standardize_and_clean_workflow).parameters
        local_vars = locals()
        # Filter out df from params to avoid large object in error message
        error_params = {
            k: v for k, v in local_vars.items() if k in func_params and k != "full_df" and k != "queried_df"
        }
        raise BioactivitiesNotFoundError(parameters=error_params)

    # find mixtures in the data
    mask = df["standard_smiles"].str.contains(".", regex=False)
    n_mixtures = mask.sum()
    if n_mixtures > 0:
        df = flag_to_remove_mixture_compounds(df)
        logger.info(f"Number of mixtures: {mask.sum()}")

    # Search for undefined stereocenters within the remaining data
    if drop_unassigned_chiral:  # here we have the problem with the "." SMILES
        # Pass SMILES (not pickled RDKit molecules) to workers and process each distinct
        # structure once. Activity datasets commonly contain the same compound many times.
        unique_smiles = df["standard_smiles"].drop_duplicates().tolist()
        logger.debug(
            f"Finding undefined stereocenters in {len(unique_smiles)} distinct SMILES "
            f"({len(df)} activity rows)"
        )
        applier = ParallelApplier(
            find_undefined_stereocenters,
            unique_smiles,
            n_jobs=8,
            backend="loky",
            custom_desc="Find undefined stereocenters",
            logger=logger,
            chunk_size=200,
        )
        undefined_stereo_counts = [len(value) for value in applier()]
        undefined_stereo_by_smiles = dict(zip(unique_smiles, undefined_stereo_counts))

        df = df.assign(
            undefined_stereocenters=lambda x: x["standard_smiles"].map(undefined_stereo_by_smiles)
        ).pipe(flag_undefined_stereochemistry)
        logger.trace(f'Unassigned stereocenters: {df["undefined_stereocenters"].unique().tolist()}')
        undefined_stereo_mask = df["undefined_stereocenters"] > 0
        if undefined_stereo_mask.any():
            logger.info(f"Flagging {undefined_stereo_mask.sum()} rows with undefined stereocenters.")
            logger.debug(df[undefined_stereo_mask].iloc[:, :5])
        if df.empty:
            logger.warning("All data points have been dropped due to undefined stereocenters!!")
            return pd.DataFrame()

    # Strict mutant removal based on assay_description
    if strict_mutant_removal:
        df = flag_strict_mutant_assays(df, strict_mutant_removal=True)
        if df.empty:
            logger.warning(
                "All data points have been dropped after strict mutant removal based on assay_description."
            )
            return pd.DataFrame()

    # for the duplication we try to find the same molecule identifiers (molID, SMILES) and
    # activity outcomes (targetID, organismID, standard_value, standard_relation), but reported
    # in different ChEMBL documents (different papers) so we can keep same-readouts reported
    # by two assays performed in the same paper !
    df = flag_inter_document_duplication(df).sort_values(WORKFLOW_SAVE_SORTED_BY).reset_index(drop=True)

    # This part needs to be removed prior to data aggregation. Here, we have either
    # inorganic compounds (SMILES removed from the salt removal step), mixtures (SMILES with "."), or
    # compounds with missing activity values, which are needed for the statistics
    missing_smiles_patt = r"Missing Standard SMILES|Missing SMILES|Mixture in SMILES"
    only_salt_entry_patt = r"^\.+$"  # if only salts are present, SMILES will be just "."

    # Build the query dynamically based on which columns exist and which value_col is used
    query_parts = [
        "data_dropping_comment.str.contains(@missing_smiles_patt, na=False, regex=True)",
        r"standard_smiles.str.contains(@only_salt_entry_patt, regex=True)",
    ]

    # Only filter by value_col if it exists in the dataframe
    if value_col in df.columns:
        query_parts.append(f"{value_col}.isna()")

    removed_subset = df.query(" | ".join(query_parts)).copy()
    if output_path is not None:
        suffixes = "".join(output_path.suffixes)
        new_name = output_path.stem.split(".")[0] + "_removed_subset" + suffixes
        save_dataframe(removed_subset, output_path.with_name(new_name))
    df = df.drop(index=removed_subset.index)

    if save_not_aggregated and output_path is not None:
        suffixes = "".join(output_path.suffixes)
        new_name = output_path.stem.split(".")[0] + "_not_aggregated" + suffixes
        save_dataframe(df, output_path.with_name(new_name))
    return df


def aggregate_data(
    df,
    chirality: bool,
    metadata_cols: list[str] = [],
    extra_id_cols: list[str] = [],
    extra_multival_cols: list[str] = [],
    aggregate_mutants: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    compound_equality: CompoundEqualityMethod = "connectivity",
    value_col: str = "pchembl_value",
):
    """Aggregate the data obtained from ChEMBL by:
    1) Calculate fingerprints and use those to identify same-structure compounds;
    2) Identify identical arrays from fingerprints and aggregate the data;

    Aggregated data will contain the original data separated by a semicolon and calculate
    the mean, median, standard deviation, median absolute deviation, and value counts
    for the pchembl values.

    Args:

        df: dataframe output from `CompoundMapper.cli.workflow.fetch_standardize_and_clean_workflow`
        chirality: toggle chiral-sensitive fingerprints for identifying same molecules
        extra_id_cols: additional columns to use as identifiers for the aggregation. Passing
            `["assay_chembl_id"]` to this argument, for example, will only aggregate the data
            if the compound is the same and the assay is the same.
        extra_multival_cols: list of extra columns that you'd like to keep as aggregated
            values in the final dataframe. Caveat: these columns will be displayed as (str)
            separated by `|` in the final dataframe. Defaults to [].
        aggregate_mutants: if true, will aggregate data solely based on the target_chembl_id,
            regardless of the mutation flag in ChEMBL. Defaults to False.
        output_path: path to save the aggregated data
        compound_equality: How to identify compounds in the dataset. ``connectivity`` uses
            the first InChIKey block; ``inchi`` and ``inchikey`` use the complete standard
            InChI representation or its hashed key; ``smiles`` uses standardized SMILES;
            and ``mixed_fp`` uses combined ECFP4 and RDKit fingerprints.
        value_col: Column name containing the values to aggregate statistics on.
            Defaults to "pchembl_value". Use "standard_value" for non-pChEMBL data (e.g., % inhibition).

    Returns:
        pd.DataFrame: the aggregated data
    """
    current_extra_id_cols = list(extra_id_cols)  # mutable copy

    identifier_by_smiles = None

    if compound_equality == "mixed_fp":
        # Exact duplicate SMILES necessarily have identical fingerprints.
        unique_smiles = df["standard_smiles"].drop_duplicates().tolist()
        unique_fps = calculate_mixed_FPs(
            unique_smiles, n_jobs=8, morgan_kwargs={"useChirality": chirality}, chunk_size=50
        )
        fp_by_smiles = dict(zip(unique_smiles, unique_fps))
        df = df.assign(id_array=[fp_by_smiles[value] for value in df["standard_smiles"]])
    elif compound_equality in INCHI_IDENTIFIERS:
        if compound_equality == "connectivity":

            def _strip_stereo(smi):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    return smi
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol)

            if chirality:
                logger.warning(
                    "Connectivity-based compound equality merges stereoisomers!!! "
                    "Stripping stereochemistry from standard_smiles to avoid "
                    "retaining an arbitrary enantiomer's SMILES in the output."
                )
            unique_smiles = df["standard_smiles"].drop_duplicates()
            stripped_by_smiles = dict(zip(unique_smiles, unique_smiles.apply(_strip_stereo)))
            df["standard_smiles"] = df["standard_smiles"].map(stripped_by_smiles)

        identifier_by_smiles = _identifier_map(df["standard_smiles"], compound_equality)
        df = df.assign(id_array=df["standard_smiles"].map(identifier_by_smiles))
    elif compound_equality == "smiles":
        df = df.assign(id_array=df["standard_smiles"].values)
    else:
        raise ValueError(
            f"Invalid compound_equality value: {compound_equality}. "
            "Expected 'mixed_fp', 'connectivity', 'inchi', 'inchikey', or 'smiles'."
        )

    # For censored measurements (!=), include relation and value in the compound identifier
    # so they are only aggregated if they have the same value AND relation.
    # NaN standard_relation (e.g., AstraZeneca PPB assays) is treated as non-censored.
    censored_mask = df["standard_relation"].notna() & df["standard_relation"].ne("=")
    has_censored = censored_mask.any()
    if has_censored:
        logger.info(
            "Detected censored measurements (standard_relation != '='). "
            f"These will only be aggregated if they have identical relation AND {value_col}."
        )
        # Round value to some decimal places to avoid floating point precision issues
        if value_col == "pchembl_value":
            rounded_value = df[value_col].round(2).astype(str)
        elif value_col == "standard_value":
            rounded_value = df[value_col].round(4).astype(str)
        df.loc[censored_mask, "id_array"] = (
            df.loc[censored_mask, "id_array"].astype(str)
            + "_"
            + df.loc[censored_mask, "standard_relation"]
            + "_"
            + rounded_value[censored_mask]
        )

    # Treat NaN standard_relation as "=" (exact measurement) for aggregation.
    # process_repeat_mols groups by standard_relation, and pandas drops NaN groups.
    df["standard_relation"] = df["standard_relation"].fillna("=")

    # Here we have a repeat index for compounds across all fetched data. Processing which repeats
    # get aggregated (e.g.: same target ID, same `extra_id_cols`, etc) is done in `process_repeat_mols`.
    repeats_idxs = repeated_indices_from_array_series(df["id_array"])

    if "activity_comment" in df.columns:
        df["activity_comment"] = df["activity_comment"].fillna("")

    include_metadata = [
        "doc_type",
        "doi",
        "journal",
        "year",
        "chembl_release",
        *(["activity_comment"] if "activity_comment" in df.columns else []),
        *extra_multival_cols,
        DATA_DROPPING_COMMENT,
        DATA_PROCESSING_COMMENT,
    ]

    preserve_stereo = chirality or compound_equality in STEREO_SENSITIVE_EQUALITY_METHODS
    final_data = process_repeat_mols(
        df,
        repeats_idxs,
        solve_strat="keep",
        extra_id_cols=current_extra_id_cols,
        chirality=preserve_stereo,
        extra_multival_cols=include_metadata,
        aggregate_mutants=aggregate_mutants,
        value_col=value_col,
    )

    final_data = _finalize_aggregated_output(
        final_data,
        compound_equality,
        current_extra_id_cols,
        identifier_by_smiles=identifier_by_smiles,
    )

    _warn_info_post_aggregation_repeats(
        final_data,
        extra_id_cols=extra_id_cols,
        aggregate_mutants=aggregate_mutants,
        value_col=value_col,
        compound_equality=compound_equality,
    )

    if output_path is not None:
        save_dataframe(final_data, output_path)

    return final_data


def re_aggregate_data(
    df: pd.DataFrame,
    chirality: bool,
    extra_id_cols: list[str] = [],
    extra_multival_cols: list[str] = [],
    aggregate_mutants: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    compound_equality: CompoundEqualityMethod = "connectivity",
) -> pd.DataFrame:
    """Re-aggregate the data obtained from the `aggregate_data` method after dataset
    explosion. Useful for exploring the effect of different `extra_id_cols` and other
    parameters.

    Args:
        df: dataframe output from `aggregate_data`
        chirality: toggle chiral-sensitive fingerprints for identifying same molecules
        extra_id_cols: additional columns to use as identifiers for the aggregation. Passing
            `["assay_chembl_id"]` to this argument, for example, will only aggregate the data
            if the compound is the same and the assay is the same.
        extra_multival_cols: list of extra columns that you'd like to keep as aggregated
            values in the final dataframe. Caveat: these columns will be displayes as (str)
            separated by `|` (pipe) in the final dataframe. Defaults to [].
        aggregate_mutants: if true, will aggregate data solely based on the target_chembl_id,
            regardless of the mutation flag in ChEMBL. Defaults to False.
        output_path: path to save the aggregated data
        compound_equality: How to identify compounds in the dataset. ``connectivity`` uses
            the first InChIKey block; ``inchi`` and ``inchikey`` use the complete standard
            InChI representation or its hashed key; ``smiles`` uses standardized SMILES;
            and ``mixed_fp`` uses combined ECFP4 and RDKit fingerprints.

    Returns:
        pd.DataFrame: the re-aggregated data
    """
    if "processed_smiles" in df.columns:
        df = df.rename(columns={"processed_smiles": "standard_smiles"})
    if "standard_smiles" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'standard_smiles' column.")
    if "smiles" not in df.columns:
        raise ValueError(
            "This method expects the output from CompoundMapper's CLI, which includes a 'smiles' column."
        )

    identifier_by_smiles = None
    if compound_equality == "mixed_fp":
        unique_smiles = df["standard_smiles"].drop_duplicates().tolist()
        unique_fps = calculate_mixed_FPs(unique_smiles, n_jobs=8, morgan_kwargs={"useChirality": chirality})
        fp_by_smiles = dict(zip(unique_smiles, unique_fps))
        id_array = pd.Series([fp_by_smiles[value] for value in df["standard_smiles"]], index=df.index)
    elif compound_equality in INCHI_IDENTIFIERS:
        if compound_equality in df.columns and df[compound_equality].notna().all():
            identifier_by_smiles = dict(zip(df["standard_smiles"], df[compound_equality]))
        else:
            identifier_by_smiles = _identifier_map(df["standard_smiles"], compound_equality)
        id_array = df["standard_smiles"].map(identifier_by_smiles)
    elif compound_equality == "smiles":
        id_array = df["standard_smiles"]
    else:
        raise ValueError(
            f"Invalid compound_equality value: {compound_equality}. "
            "Expected 'mixed_fp', 'connectivity', 'inchi', 'inchikey', or 'smiles'."
        )

    # For censored measurements (!=), include relation and pchembl_value in the compound identifier
    # so they are only aggregated if they have the same value AND relation
    if "standard_relation" in df.columns:
        has_censored = df["standard_relation"].ne("=").any()
        if has_censored:
            logger.info(
                "Detected censored measurements (standard_relation != '='). "
                "These will only be aggregated if they have identical relation AND pchembl_value."
            )
            # Round pchembl_value to 2 decimal places to avoid floating point precision issues
            rounded_pchembl = df["pchembl_value"].round(2).astype(str)
            # For censored measurements, append relation + value to the id_array
            censored_mask = df["standard_relation"] != "="
            id_array = id_array.copy()  # Create a copy to avoid modifying the original
            id_array.loc[censored_mask] = (
                id_array.loc[censored_mask].astype(str)
                + "_"
                + df.loc[censored_mask, "standard_relation"]
                + "_"
                + rounded_pchembl[censored_mask]
            )

    repeats_idxs = repeated_indices_from_array_series(id_array)

    if "activity_comment" in df.columns:
        df["activity_comment"] = df["activity_comment"].fillna("")

    include_metadata = [
        "doc_type",
        "doi",
        "journal",
        "year",
        "chembl_release",
        *(["activity_comment"] if "activity_comment" in df.columns else []),
        *extra_multival_cols,
        DATA_DROPPING_COMMENT,
        DATA_PROCESSING_COMMENT,
    ]

    for col in include_metadata:  # make sure columns exist
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' is required in the DataFrame but is missing. "
                "Please ensure that the DataFrame contains all necessary columns."
            )

    preserve_stereo = chirality or compound_equality in STEREO_SENSITIVE_EQUALITY_METHODS
    final_data = process_repeat_mols(  # recalculate the stats given new conditions
        df,
        repeats_idxs,
        solve_strat="keep",
        extra_id_cols=extra_id_cols,
        chirality=preserve_stereo,
        extra_multival_cols=include_metadata,
        aggregate_mutants=aggregate_mutants,
    )
    final_data = _finalize_aggregated_output(
        final_data,
        compound_equality,
        extra_id_cols,
        identifier_by_smiles=identifier_by_smiles,
    )

    _warn_info_post_aggregation_repeats(
        final_data,
        extra_id_cols=extra_id_cols,
        aggregate_mutants=aggregate_mutants,
        compound_equality=compound_equality,
    )

    if output_path is not None:
        save_dataframe(final_data, output_path)

    return final_data
