"""Module holding functionalities for the ChEMBL API using chembl downloader as the backend."""

import json
import re
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
import pystow
from chembl_downloader import download_extract_sqlite, latest, query
from chembl_downloader.api import _find_sqlite_file

from ...logger import logger
from ..exceptions import BioactivitiesNotFoundError

PYSTOW_PARTS = ["chembl"]
PYSTOW_CONFIG = {"name": "chembl_downloader_config_{version}.json"}


def _get_kwargs_where_clauses(**kwargs):
    """Generate WHERE clauses for SQL queries based on kwargs."""
    where_clauses = []
    for key, value in kwargs.items():
        if isinstance(value, list):  # Handle lists of values (IN clause)
            placeholders = ", ".join([f"'{v}'" for v in value])
            where_clauses.append(f"a.{key} IN ({placeholders})")
        else:  # Handle single values
            where_clauses.append(f"a.{key} = '{value}'")
    return where_clauses


def _get_config_file(version: Optional[Union[int, str]] = None) -> Path:
    version = version if version is not None else latest()
    version = str(version) if isinstance(version, int) else version
    return pystow.join(*(PYSTOW_PARTS), name=PYSTOW_CONFIG["name"].format(version=version))


def check_and_download_chembl_db(
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Check if the ChEMBL database is present. Download and extract it if not. After extraction,
    remove the tarball to free up space. This method is also used to assert the correct downloaded
    ChEMBL version is used across different query functions.

    Args:
        prefix: Optional prefix for an alternative data directory with path components passed as a list of strings.
            If passed, will create a new configuration file under `~/.data/chembl_downloader_config_{version}.json`
            pointing to the new data directory. Defaults to None.
        version: Optional ChEMBL version to download. If not provided, will download the latest
            available version. Defaults to None.

    Returns:
        Path to the ChEMBL SQLite database
    """
    # if present, config file override the default path, unless a prefix is defined
    version = version if version is not None else latest()
    config_file = _get_config_file(version)
    configs = {"prefix": (prefix if prefix is not None else PYSTOW_PARTS), "version": version}

    if config_file.exists():  # only exists if that version was downloaded to custom path before
        logger.info(f"Loaded ChEMBL configuration from:\n\t{config_file}")
        configs = json.loads(config_file.read_text())
        logger.debug(f"configuration:\n{json.dumps(configs, indent=2)}")

    sql_path = _find_sqlite_file(pystow.join(*(configs["prefix"]), f"{configs['version']}"))
    if sql_path is None:
        logger.info(f"Downloading and extracting ChEMBL version {version}...")
        rv = download_extract_sqlite(version=str(version), prefix=(configs["prefix"] or PYSTOW_PARTS))
        tar_path = rv.parents[3] / f"chembl_{version}_sqlite.tar.gz"
        if tar_path.exists():
            logger.info(f"Removing downloaded tarball: {tar_path}")
            tar_path.unlink()
        if prefix is not None:
            config_file.write_text(json.dumps(configs, indent=2))
    else:
        logger.debug(f"Loaded local ChEMBL database at:\n\t{sql_path}")

    return configs


def get_document_table_sql(
    document_chembl_ids: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Get publication details for a list of ChEMBL document IDs using SQL backend.

    Args:
        document_chembl_ids: list of ChEMBL document IDs. Fetch all if None. Defaults to None.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.

    Returns:
        pd.DataFrame: a DataFrame with the publication details.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    # Build the query
    if not document_chembl_ids:
        raise ValueError("No document IDs provided")

    where_clauses = []
    if document_chembl_ids:
        doc_placeholders = ", ".join([f"'{doc_id}'" for doc_id in document_chembl_ids])
        where_clauses.append(f"chembl_id IN ({doc_placeholders})")

    where_clauses.extend(_get_kwargs_where_clauses(**kwargs))
    where_clause = " AND ".join(where_clauses)

    query_str = dedent(f"""\
        SELECT 
            chembl_id AS document_chembl_id,
            doc_type,
            authors,
            doi,
            journal,
            volume,
            year,
            title,
            chembl_release_id AS chembl_release
        FROM docs
        WHERE
            {where_clause}
        """)

    logger.debug(f"Generated SQL query for documents:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        logger.warning(f"No publication details found for document IDs: {document_chembl_ids}")

    return result


def get_compound_table_sql(
    molecule_chembl_ids: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Get information on molecules from ChEMBL using SQL backend.

    Args:
        molecule_chembl_ids: list of molecule ChEMBL IDs.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.

    Returns:
        pd.DataFrame: a DataFrame with molecule information.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    if not molecule_chembl_ids:
        raise ValueError("No molecule IDs provided")

    where_clauses = []

    if molecule_chembl_ids:
        mol_placeholders = ", ".join([f"'{mol_id}'" for mol_id in molecule_chembl_ids])
        where_clauses.append(f"md.chembl_id IN ({mol_placeholders})")

    where_clauses.extend(_get_kwargs_where_clauses(**kwargs))
    where_clause = " AND ".join(where_clauses)

    query_str = dedent(f"""\
        SELECT
            md.chembl_id AS molecule_chembl_id,
            cs.canonical_smiles,
            cs.standard_inchi,
            cs.standard_inchi_key,
            mh.parent_molregno,
            md.chirality,
            md.oral,
            md.prodrug,
            md.max_phase,
            md.therapeutic_flag,
            md.withdrawn_flag,
        FROM
            molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN molecule_hierarchy mh ON md.molregno = mh.molregno
        WHERE
            {where_clause}
        ORDER BY
            md.chembl_id
        """)

    logger.debug(f"Generated SQL query for compounds:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        raise ValueError(f"No information found for molecule IDs: {molecule_chembl_ids}")

    # Process parent molecule relationships if needed
    parent_molregnos = result["parent_molregno"].dropna().unique().tolist()
    if parent_molregnos:
        parent_query = dedent(f"""\
            SELECT
                mh.parent_molregno,
                md.chembl_id AS parent_chembl_id,
                cs.canonical_smiles AS parent_smiles
            FROM molecule_hierarchy mh
            JOIN molecule_dictionary md ON mh.parent_molregno = md.molregno
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE
                mh.parent_molregno IN ({', '.join(map(str, parent_molregnos))})
            """)

        parent_data = query(
            parent_query,
            version=downloader_configs["version"],
            prefix=downloader_configs["prefix"],
        )

        # Merge parent information if available
        if not parent_data.empty:
            result = pd.merge(result, parent_data, on="parent_molregno", how="left")

    # Sort the result to match the order of input IDs (similar to the original function)
    result["idx"] = result["molecule_chembl_id"].apply(
        lambda x: molecule_chembl_ids.index(x) if x in molecule_chembl_ids else len(molecule_chembl_ids)
    )
    result = result.sort_values("idx").drop("idx", axis=1)

    return result


_COMPOUND_LOOKUP_QUERY = dedent("""\
    SELECT
        md.chembl_id AS molecule_chembl_id,
        cs.molregno,
        cs.canonical_smiles,
        cs.standard_inchi_key,
        pmd.chembl_id AS parent_chembl_id,
        pcs.canonical_smiles AS parent_smiles
    FROM compound_structures cs
    JOIN molecule_dictionary md ON cs.molregno = md.molregno
    LEFT JOIN molecule_hierarchy mh ON cs.molregno = mh.molregno
    LEFT JOIN molecule_dictionary pmd ON mh.parent_molregno = pmd.molregno
    LEFT JOIN compound_structures pcs ON mh.parent_molregno = pcs.molregno
    WHERE
        {where_clause}
    """)

CONNECTIVITY_PATTERN = re.compile(r"^[A-Z]{14}$")


def get_compounds_by_inchikey_sql(
    inchi_keys: Optional[List[str]] = None,
    connectivities: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Look up compounds by full standard InChI key and/or by connectivity.

    A full InChI key identifies a single structure including stereochemistry and protonation
    state. A connectivity, the first 14 characters of the key, identifies the molecular skeleton
    and therefore also matches stereoisomers and salt forms of the same compound. Both are
    resolved against the indexed ``standard_inchi_key`` column, so lookups stay in the
    millisecond range even though ``compound_structures`` holds ~2.9M rows.

    Args:
        inchi_keys: list of full standard InChI keys to match exactly. Defaults to None.
        connectivities: list of 14-character connectivities to match as key prefixes.
            Defaults to None.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.

    Returns:
        pd.DataFrame: molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key and the
        parent compound from ``molecule_hierarchy``. Empty if nothing matched.
    """
    if not inchi_keys and not connectivities:
        raise ValueError("Provide at least one of 'inchi_keys' or 'connectivities'.")

    where_clauses = []
    if inchi_keys:
        key_placeholders = ", ".join([f"'{key}'" for key in sorted(set(inchi_keys))])
        where_clauses.append(f"cs.standard_inchi_key IN ({key_placeholders})")

    for connectivity in sorted(set(connectivities or [])):
        if not CONNECTIVITY_PATTERN.match(connectivity):
            raise ValueError(
                f"Invalid connectivity {connectivity!r}: expected 14 uppercase letters, "
                "i.e. the first block of a standard InChI key."
            )
        # GLOB with a trailing wildcard is rewritten by SQLite into a range scan on
        # idx_cmpdstr_stdkey, unlike substr(), which would force a full table scan.
        where_clauses.append(f"cs.standard_inchi_key GLOB '{connectivity}*'")

    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    query_str = _COMPOUND_LOOKUP_QUERY.format(where_clause=" OR\n        ".join(where_clauses))
    logger.debug(f"Generated SQL query for compound lookup:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        logger.warning(
            f"No compounds found for {len(inchi_keys or [])} InChI key(s) and "
            f"{len(connectivities or [])} connectivity value(s)."
        )

    return result


def get_compounds_by_molregno_sql(
    molregnos: Sequence[int],
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Look up compounds by molregno, the internal ChEMBL identifier.

    Used to resolve fingerprint search hits, since the ChEMBL fingerprint index is keyed by
    molregno rather than by ChEMBL ID.

    Args:
        molregnos: molregno values to look up.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.

    Returns:
        pd.DataFrame: molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key and the
        parent compound from ``molecule_hierarchy``. Empty if nothing matched.
    """
    if len(molregnos) == 0:
        raise ValueError("No molregnos provided")

    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    placeholders = ", ".join([str(int(molregno)) for molregno in sorted(set(molregnos))])
    query_str = _COMPOUND_LOOKUP_QUERY.format(where_clause=f"cs.molregno IN ({placeholders})")
    logger.debug(f"Generated SQL query for molregno lookup:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        logger.warning(f"No compounds found for {len(molregnos)} molregno value(s).")

    return result


def get_assay_table_sql(
    assay_chembl_ids: Optional[List[str]] = None,
    confidence_scores: Optional[List[int]] = None,
    assay_types: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Get assay information from ChEMBL using SQL backend.

    Args:
        assay_chembl_ids: list of assay ChEMBL IDs. If None, all assays are fetched. Defaults to None.
        confidence_scores: list of confidence scores to filter the assays. Defaults to None.
        assay_types: list of assay types to filter. Defaults to None.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.
        **kwargs: Additional filtering parameters not used in SQL implementation.

    Returns:
        pd.DataFrame: a DataFrame with assay information.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    if confidence_scores is None:
        confidence_scores = list(range(0, 10))

    assay_placeholders = ", ".join([f"'{aid}'" for aid in assay_chembl_ids]) if assay_chembl_ids else "''"
    confidence_placeholders = ", ".join([f"{score}" for score in confidence_scores])

    where_clauses = []
    if assay_chembl_ids:
        where_clauses.append(f"a.chembl_id IN ({assay_placeholders})")
    where_clauses.append(f"a.confidence_score IN ({confidence_placeholders})")

    if assay_types:
        type_placeholders = ", ".join([f"'{assay_type}'" for assay_type in assay_types])
        where_clauses.append(f"a.assay_type IN ({type_placeholders})")

    where_clauses.extend(_get_kwargs_where_clauses(**kwargs))
    where_clause = " AND ".join(where_clauses)

    query_str = dedent(f"""\
        SELECT
            a.chembl_id AS assay_chembl_id,
            a.description AS assay_description,
            a.relationship_type,
            a.assay_type,
            a.assay_organism,
            a.assay_category,
            a.assay_tax_id,
            a.assay_strain,
            a.assay_tissue,
            a.assay_cell_type,
            a.assay_subcellular_fraction,
            a.bao_format,
            a.confidence_score,
            d.chembl_id AS document_chembl_id,
            a.tid,
            t.chembl_id AS target_chembl_id,
            vs.mutation
        FROM assays a
        LEFT JOIN docs d ON a.doc_id = d.doc_id
        LEFT JOIN target_dictionary t ON a.tid = t.tid
        LEFT JOIN variant_sequences vs ON a.variant_id = vs.variant_id
        WHERE
            {where_clause}
        """)

    logger.debug(f"Generated SQL query for assays:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        activity_kwargs = {
            "assay_chembl_ids": assay_chembl_ids,
            "confidence_scores": confidence_scores,
            "assay_types": assay_types,
        }
        activity_kwargs.update(kwargs)
        raise ValueError(f"No assays found with the parameters: {activity_kwargs}")

    # Fill mutation field with "WT" for null values
    result["mutation"] = result["mutation"].fillna("WT")

    return result


def get_activity_table_sql(
    molecule_chembl_ids: Optional[List[str]] = None,
    target_chembl_ids: Optional[List[str]] = None,
    assay_chembl_ids: Optional[List[str]] = None,
    document_chembl_ids: Optional[List[str]] = None,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, dict]:
    """Get bioactivity data from ChEMBL using SQL backend.

    Args:
        molecule_chembl_ids: list of molecule ChEMBL IDs. Defaults to None.
        target_chembl_ids: list of target ChEMBL IDs. Defaults to None.
        assay_chembl_ids: list of assay ChEMBL IDs. Defaults to None.
        document_chembl_ids: list of document ChEMBL IDs. Defaults to None.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.
        **kwargs: Additional filtering parameters. e.g.: standard_relation=["="]

    Returns:
        Tuple[pd.DataFrame, dict]: a DataFrame with bioactivity data and the parameters used.
    """
    # Download the ChEMBL database if not present
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)
    where_conditions = []  # Store the WHERE conditions

    if molecule_chembl_ids is not None:
        m_placeholders = ", ".join([f"'{id}'" for id in molecule_chembl_ids])
        where_conditions.append(f"md.chembl_id IN ({m_placeholders})")

    if target_chembl_ids is not None:
        t_placeholders = ", ".join([f"'{id}'" for id in target_chembl_ids])
        where_conditions.append(f"td.chembl_id IN ({t_placeholders})")

    if assay_chembl_ids is not None:
        a_placeholders = ", ".join([f"'{id}'" for id in assay_chembl_ids])
        where_conditions.append(f"a.chembl_id IN ({a_placeholders})")

    if document_chembl_ids is not None:
        d_placeholders = ", ".join([f"'{id}'" for id in document_chembl_ids])
        where_conditions.append(f"d.chembl_id IN ({d_placeholders})")

    for field, value in kwargs.items():  # Handle additional kwargs as filters
        # Determine the prefix for the field
        if field in [
            "standard_relation",
            "standard_type",
            "standard_units",
            "standard_value",
            "standard_flag",
        ]:
            prefix = "act"
        elif field in ["assay_type"]:
            prefix = "a"
        else:
            raise ValueError(f"Field '{field}' not supported for filtering")

        if isinstance(value, list):
            placeholders = ", ".join([f"'{v}'" for v in value])
            where_conditions.append(f"{prefix}.{field} IN ({placeholders})")
        else:  # Handle simple equality filters
            where_conditions.append(f"{prefix}.{field} = '{value}'")

    where_conditions.append("act.standard_value IS NOT NULL")
    where_clause = " AND ".join(where_conditions)

    query_str = dedent(f"""\
        SELECT
            act.activity_id,
            a.chembl_id AS assay_chembl_id,
            a.description AS assay_description,
            a.assay_type,
            a.assay_organism,
            a.assay_category,
            a.assay_tax_id,
            a.assay_strain,
            a.assay_tissue,
            a.assay_cell_type,
            a.assay_subcellular_fraction,
            a.bao_format,
            a.variant_id,
            md.chembl_id AS molecule_chembl_id,
            act.standard_flag,
            act.standard_relation,
            act.standard_type,
            act.standard_units,
            act.standard_value,
            act.pchembl_value,
            td.chembl_id AS target_chembl_id,
            td.organism AS target_organism,
            act.data_validity_comment,
            act.potential_duplicate,
            d.chembl_id AS document_chembl_id
        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN docs d ON act.doc_id = d.doc_id
        JOIN target_dictionary td ON a.tid = td.tid
        WHERE
            {where_clause}
        ORDER BY
            md.chembl_id, act.activity_id
        """)

    logger.debug(f"Generated SQL query for activities:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    # Create parameters dictionary for error reporting
    activity_kwargs = {}
    if molecule_chembl_ids is not None:
        activity_kwargs["molecule_chembl_id__in"] = molecule_chembl_ids
    if target_chembl_ids is not None:
        activity_kwargs["target_chembl_id__in"] = target_chembl_ids
    if assay_chembl_ids is not None:
        activity_kwargs["assay_chembl_id__in"] = assay_chembl_ids
    if document_chembl_ids is not None:
        activity_kwargs["document_chembl_id__in"] = document_chembl_ids
    activity_kwargs.update(kwargs)

    if result.empty:
        raise BioactivitiesNotFoundError(parameters=activity_kwargs)

    return result


def get_full_activity_data_sql(
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
) -> pd.DataFrame:
    """Retrieve ChEMBL bioactivity data from any combination of molecule, target, assay, or document IDs.
    Data is retrieved using the ChEMBL downloader. Merges are performed on the SQL query level and a
    DataFrame is returned with the bioactivity data.

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
        additional_fields: Optional list of additional fields to include in the sql query. E.g.:
            ["vs.sequence"], to retrieve the sequence of the variant, if available. Defaults to None.
        prefix: Optional prefix for an alternative data directory. If passed, will create
            a new configuration file under `~/.data/chembl_downloader_config_{version}.json`
            pointing to the new data directory. Defaults to None.
        version: ChEMBL database to be downloaded and used by ChEMBL downloader. If not provided,
            will download the latest available version. Defaults to None.

    Returns:
        pd.DataFrame: a DataFrame with the bioactivity data.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)
    where_conditions_main = []

    if molecule_chembl_ids is not None:
        ids = [molecule_chembl_ids] if isinstance(molecule_chembl_ids, str) else molecule_chembl_ids
        placeholders = ", ".join([f"'{id}'" for id in ids])
        where_conditions_main.append(f"md.chembl_id IN ({placeholders})")

    if target_chembl_ids is not None:
        ids = [target_chembl_ids] if isinstance(target_chembl_ids, str) else target_chembl_ids
        placeholders = ", ".join([f"'{id}'" for id in ids])
        where_conditions_main.append(f"td.chembl_id IN ({placeholders})")

    if document_chembl_ids is not None:
        ids = [document_chembl_ids] if isinstance(document_chembl_ids, str) else document_chembl_ids
        placeholders = ", ".join([f"'{id}'" for id in ids])
        where_conditions_main.append(f"d.chembl_id IN ({placeholders})")

    if assay_chembl_ids is not None:
        ids = [assay_chembl_ids] if isinstance(assay_chembl_ids, str) else assay_chembl_ids
        placeholders = ", ".join([f"'{id}'" for id in ids])
        where_conditions_main.append(f"a.chembl_id IN ({placeholders})")

    where_conditions_main.append("act.standard_value IS NOT NULL")

    # Only filter by standard_relation if explicitly provided
    # When None, include all data including those with NULL standard_relation (e.g., AstraZeneca PPB assays)
    if standard_relation:
        placeholders = ", ".join([f"'{rel}'" for rel in standard_relation])
        where_conditions_main.append(f"act.standard_relation IN ({placeholders})")

    if standard_type:
        placeholders = ", ".join([f"'{stype}'" for stype in standard_type])
        where_conditions_main.append(f"act.standard_type IN ({placeholders})")

    if standard_units:
        placeholders = ", ".join([f"'{unit}'" for unit in standard_units])
        where_conditions_main.append(f"act.standard_units IN ({placeholders})")

    if confidence_scores:
        placeholders = ", ".join([f"{score}" for score in confidence_scores])
        where_conditions_main.append(f"a.confidence_score IN ({placeholders})")

    if assay_types:
        placeholders = ", ".join([f"'{atype}'" for atype in assay_types])
        where_conditions_main.append(f"a.assay_type IN ({placeholders})")

    if chembl_release:
        where_conditions_main.append(
            f"(d.chembl_release_id IS NULL OR d.chembl_release_id <= {chembl_release})"
        )

    base_fields = [
        "act.activity_id",
        "a.chembl_id AS assay_chembl_id",
        "a.description AS assay_description",
        "a.relationship_type",
        "a.assay_type",
        "a.assay_organism",
        "a.assay_category",
        "a.assay_tax_id",
        "a.assay_strain",
        "a.assay_tissue",
        "a.assay_cell_type",
        "a.assay_subcellular_fraction",
        "a.bao_format",
        "a.confidence_score",
        "md.chembl_id AS molecule_chembl_id",
        "md.first_in_class",
        "md.chirality",
        "md.oral",
        "md.prodrug",
        "md.max_phase",
        "md.therapeutic_flag",
        "md.withdrawn_flag",
        "act.standard_flag",
        "act.standard_relation",
        "act.standard_type",
        "act.standard_units",
        "act.standard_value",
        "act.pchembl_value",
        "td.chembl_id AS target_chembl_id",
        "td.organism AS target_organism",
        "cs.canonical_smiles",
        "cs.standard_inchi_key",
        "act.data_validity_comment AS data_validity_comment",
        "act.potential_duplicate",
        "d.chembl_id AS document_chembl_id",
        "d.doc_type",
        "d.authors",
        "d.doi",
        "d.journal",
        "d.volume",
        "d.year",
        "d.title",
        "a.variant_id",
        "vs.mutation",
        "d.chembl_release_id AS chembl_release",
    ]
    all_fields = base_fields + (additional_fields if additional_fields else [])

    from_join_clauses_main = [
        "molecule_dictionary md",
        "JOIN compound_structures cs ON md.molregno = cs.molregno",
        "JOIN activities act ON md.molregno = act.molregno",
        "JOIN docs d ON act.doc_id = d.doc_id",
        "JOIN assays a ON act.assay_id = a.assay_id",
        "LEFT JOIN variant_sequences vs ON a.variant_id = vs.variant_id",
        "JOIN target_dictionary td ON a.tid = td.tid",
    ]

    fields_clause_str = ",\n            ".join(all_fields)
    from_join_clause_main_str = "\n        ".join(from_join_clauses_main)
    where_clause_main_str = (
        " AND\n            ".join(where_conditions_main) if where_conditions_main else "1=1"
    )

    query_str = dedent(f"""\
        SELECT
            {fields_clause_str}
        FROM
            {from_join_clause_main_str}
        WHERE
            {where_clause_main_str}
        ORDER BY
            md.chembl_id, act.activity_id, act.standard_value
    """)

    logger.debug(f"Generated SQL query:\n{query_str}")

    return query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    ).assign(mutation=lambda x: x["mutation"].fillna("WT"))


def get_target_names_sql(
    target_chembl_ids: List[str],
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> dict:
    """Get target names for a list of ChEMBL target IDs using SQL backend.

    Args:
        target_chembl_ids: list of ChEMBL target IDs.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use.

    Returns:
        dict: a dictionary mapping chembl_id to pref_name.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)

    if not target_chembl_ids:
        raise ValueError("No target IDs provided")

    placeholders = ", ".join([f"'{id}'" for id in target_chembl_ids])
    where_clause = f"chembl_id IN ({placeholders})"

    query_str = dedent(f"""\
        SELECT
            chembl_id,
            pref_name
        FROM target_dictionary
        WHERE
            {where_clause}
        """)

    logger.debug(f"Generated SQL query for target names:\n{query_str}")

    result = query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )

    if result.empty:
        logger.warning(f"No targets found for IDs: {target_chembl_ids}")
        return {}

    return dict(zip(result["chembl_id"], result["pref_name"]))


def get_assay_size_sql(
    assay_chembl_ids: Union[list, str],
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """Get the number of distinct molecules for a list of ChEMBL assay IDs.

    Args:
        assay_chembl_ids: list of ChEMBL assay IDs to fetch data for.
        prefix: Optional prefix for an alternative data directory.
        version: ChEMBL database to be downloaded and used by ChEMBL downloader.

    Returns:
        pd.DataFrame: a DataFrame with assay_chembl_id and assay_size.
    """
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)
    ids = [assay_chembl_ids] if isinstance(assay_chembl_ids, str) else assay_chembl_ids
    placeholders = ", ".join([f"'{id}'" for id in ids])
    where_clause = f"a.chembl_id IN ({placeholders})"

    query_str = dedent(f"""\
        SELECT
            a.chembl_id AS assay_chembl_id,
            COUNT(DISTINCT act.molregno) as assay_size
        FROM
            assays a
        JOIN activities act ON a.assay_id = act.assay_id
        WHERE
            {where_clause}
        GROUP BY
            a.chembl_id
        """)

    logger.debug(f"Generated SQL query for assay size:\n{query_str}")

    return query(
        query_str,
        version=downloader_configs["version"],
        prefix=downloader_configs["prefix"],
    )
