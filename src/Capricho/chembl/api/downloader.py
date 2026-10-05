"""Module holding functionalities for the ChEMBL API using chembl downloader as the backend."""

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from collections.abc import Iterator, Sequence

import pandas as pd
import pystow
from chembl_downloader import connect, download_extract_sqlite, latest
from chembl_downloader.api import _find_sqlite_file

from ...logger import logger
from ..exceptions import BioactivitiesNotFoundError

PYSTOW_PARTS = ["chembl"]
PYSTOW_CONFIG = {"name": "chembl_downloader_config_{version}.json"}

# ChEMBL SQLite dumps are named after their release, e.g. chembl_35.db or chembl_24_1.db.
CHEMBL_DB_FILENAME = re.compile(r"^chembl_(\d+(?:_\d+)?)\.db$")

# Every dump states its own release in the `version` table, alongside rows for the other
# resources it embeds (Bioassay Ontology, COCONUT), e.g. name = "ChEMBL_36".
CHEMBL_VERSION_ROW = re.compile(r"^ChEMBL_(\d+(?:[._]\d+)?)$", re.IGNORECASE)

# Tables read by the queries in this module. Used to tell a ChEMBL dump apart from any
# other SQLite file a user might point at.
REQUIRED_TABLES = frozenset(
    {
        "activities",
        "assays",
        "compound_structures",
        "docs",
        "molecule_dictionary",
        "molecule_hierarchy",
        "target_dictionary",
        "variant_sequences",
    }
)


@lru_cache(maxsize=1)
def _latest_version() -> str:
    """Resolve and pin the latest ChEMBL release for the lifetime of this process."""
    return latest()


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


def _get_config_file(version: int | str | None = None) -> Path:
    version = version if version is not None else _latest_version()
    version = str(version) if isinstance(version, int) else version
    return pystow.join(*(PYSTOW_PARTS), name=PYSTOW_CONFIG["name"].format(version=version))


def _version_from_filename(path: Path) -> str | None:
    """Read the ChEMBL release out of a database file name, or None if it doesn't carry one."""
    match = CHEMBL_DB_FILENAME.match(path.name)
    return match.group(1).replace("_", ".") if match else None


def _validate_chembl_db(path: Path) -> None:
    """Check that a file is a readable SQLite database holding the ChEMBL tables.

    Args:
        path: path to the candidate database file.

    Raises:
        ValueError: if the file cannot be read as SQLite or lacks the ChEMBL tables.
    """
    try:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
            tables = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
    except sqlite3.DatabaseError as error:
        raise ValueError(f"{path} is not a readable SQLite database: {error}") from error

    missing = REQUIRED_TABLES - tables
    if missing:
        raise ValueError(
            f"{path} does not look like a ChEMBL database. Missing tables: {', '.join(sorted(missing))}"
        )


def _read_reported_release(path: Path) -> str | None:
    """Read the ChEMBL release a database states for itself, or None if it does not state one.

    Args:
        path: path to the ChEMBL SQLite database.

    Returns:
        The release from the `version` table, e.g. "36", or None when the table is absent or
        holds no ChEMBL row.
    """
    try:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
            names = [row[0] for row in conn.execute("SELECT name FROM version")]
    except sqlite3.DatabaseError:  # no `version` table, as in trimmed or hand-built databases
        return None

    for name in names:
        match = CHEMBL_VERSION_ROW.match(name or "")
        if match:
            return match.group(1).replace("_", ".")
    return None


def _check_release_matches(path: Path, claimed: str) -> None:
    """Check a database against the release it is about to be registered under.

    A file name can be wrong and a `--version` can be a typo, and a release registered
    wrongly would misreport the provenance of every dataset drawn from it. The release the
    database states for itself settles the question.

    Args:
        path: path to the ChEMBL SQLite database.
        claimed: the release the database is about to be registered under.

    Raises:
        ValueError: if the database states a different release than the one claimed.
    """
    reported = _read_reported_release(path)

    if reported is None:
        logger.warning(
            f"{path} does not state its own release, so it is registered as ChEMBL {claimed} on "
            "the strength of its file name alone. Check that this is the release you meant."
        )
        return

    if reported != str(claimed):
        raise ValueError(
            f"{path} states that it is ChEMBL {reported}, not ChEMBL {claimed}. Registering it "
            f"as ChEMBL {claimed} would misreport the provenance of every dataset drawn from it. "
            f"Register it as ChEMBL {reported}, or check that this is the file you meant."
        )

    logger.debug(f"{path} confirms it is ChEMBL {reported}")


def _discover_chembl_dbs(path: Path, version: int | str | None = None) -> dict[str, Path]:
    """Map ChEMBL release to database file for a file or directory the user points at.

    Args:
        path: a `chembl_<version>.db` file, or a directory holding one or more of them.
        version: only keep this release. Defaults to None, keeping every release found.

    Returns:
        Mapping of ChEMBL release to the database file holding it.

    Raises:
        FileNotFoundError: if the path does not exist or holds no matching database.
        ValueError: if the release cannot be read from a file name and none was given.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    if path.is_file():
        file_version = str(version) if version is not None else _version_from_filename(path)
        if file_version is None:
            raise ValueError(
                f"Cannot tell which ChEMBL release {path.name} holds, since the file name does "
                "not follow the chembl_<version>.db convention. Pass the release explicitly, "
                "e.g. --version 35."
            )
        return {file_version: path}

    databases = {}
    for candidate in sorted(path.rglob("chembl_*.db")):
        candidate_version = _version_from_filename(candidate)
        if candidate_version is not None:
            databases[candidate_version] = candidate

    if not databases:
        raise FileNotFoundError(f"No chembl_<version>.db file found under {path}")

    if version is not None:
        wanted = str(version)
        if wanted not in databases:
            raise FileNotFoundError(
                f"No chembl_{wanted}.db found under {path}. "
                f"Releases available there: {', '.join(sorted(databases))}"
            )
        return {wanted: databases[wanted]}

    return databases


def set_chembl_db_path(
    path: str | Path, version: int | str | None = None
) -> dict[str, Path]:
    """Register an already-available ChEMBL SQLite database so queries read it where it lies.

    Writes a configuration file per release under `~/.data/chembl/`, which
    `check_and_download_chembl_db` then honours instead of downloading the release.

    Args:
        path: a `chembl_<version>.db` file, or a directory holding one or more of them.
        version: only register this release. Required when the file name does not carry the
            release. Defaults to None, registering every release found under the path.

    Returns:
        Mapping of the registered ChEMBL releases to their database files.

    Raises:
        ValueError: if a database is not a ChEMBL dump, or states a release other than the one
            it would be registered under.
    """
    databases = _discover_chembl_dbs(Path(path), version=version)

    # Check every database before writing anything, so a bad one cannot leave half the
    # releases of a directory registered.
    for db_version, db_path in databases.items():
        _validate_chembl_db(db_path)
        _check_release_matches(db_path, db_version)

    for db_version, db_path in databases.items():
        configs = {"prefix": PYSTOW_PARTS, "version": db_version, "path": str(db_path)}
        config_file = _get_config_file(db_version)
        config_file.write_text(json.dumps(configs, indent=2))
        logger.info(f"ChEMBL {db_version} will be read from:\n\t{db_path}")
        logger.debug(f"Wrote configuration to:\n\t{config_file}")

    return databases


def unset_chembl_db_path(version: int | str | None = None) -> bool:
    """Forget the database registered for a release, so it is downloaded again when queried.

    Args:
        version: ChEMBL release to forget. Defaults to None, resolving to the latest release.

    Returns:
        Whether a configuration file was removed.
    """
    config_file = _get_config_file(version)
    if not config_file.exists():
        logger.warning(f"No ChEMBL configuration to remove at:\n\t{config_file}")
        return False

    config_file.unlink()
    logger.info(f"Removed ChEMBL configuration:\n\t{config_file}")
    return True


@contextmanager
def connect_chembl(configs: dict) -> Iterator[sqlite3.Connection]:
    """Connect to the ChEMBL database described by a configuration mapping.

    Args:
        configs: mapping as returned by `check_and_download_chembl_db`. A `path` entry points
            at a database registered with `set_chembl_db_path`; otherwise the release is
            resolved from the pystow `prefix` and `version`.

    Yields:
        An open connection to the ChEMBL SQLite database.
    """
    if configs.get("path"):
        with closing(sqlite3.connect(Path(configs["path"]).as_posix())) as conn:
            yield conn
    else:
        with connect(version=configs["version"], prefix=configs["prefix"]) as conn:
            yield conn


def run_query(sql: str, configs: dict, params: Sequence | None = None) -> pd.DataFrame:
    """Run a SQL query against the ChEMBL database described by a configuration mapping.

    Args:
        sql: the SQL query to run.
        configs: mapping as returned by `check_and_download_chembl_db`.
        params: optional values bound to placeholders in ``sql``.

    Returns:
        pd.DataFrame: the query result.
    """
    with connect_chembl(configs) as conn:
        return pd.read_sql(sql, conn, params=params)


def check_and_download_chembl_db(
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
) -> dict:
    """Check if the ChEMBL database is present. Download and extract it if not. After extraction,
    remove the tarball to free up space. This method is also used to assert the correct downloaded
    ChEMBL version is used across different query functions. Nothing is downloaded when the release
    was registered with `set_chembl_db_path`; that database is read where it lies.

    Args:
        prefix: Optional prefix for an alternative data directory with path components passed as a list of strings.
            If passed, will create a new configuration file under `~/.data/chembl_downloader_config_{version}.json`
            pointing to the new data directory. Defaults to None.
        version: Optional ChEMBL version to download. If not provided, will download the latest
            available version. Defaults to None.

    Returns:
        Configuration describing the database to query: the pystow `prefix` and `version`, plus a
        `path` entry when the release was registered from a database already on the system.
    """
    # if present, config file override the default path, unless a prefix is defined
    version = version if version is not None else _latest_version()
    config_file = _get_config_file(version)
    configs = {"prefix": (prefix if prefix is not None else PYSTOW_PARTS), "version": version}

    if config_file.exists():  # only exists if that version was downloaded to custom path before
        logger.debug(f"Loaded ChEMBL configuration from:\n\t{config_file}")
        configs = json.loads(config_file.read_text())
        logger.debug(f"configuration:\n{json.dumps(configs, indent=2)}")

    if configs.get("path"):  # a database the user registered with `set_chembl_db_path`
        db_path = Path(configs["path"])
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChEMBL {configs['version']} was registered at {db_path}, which no longer exists. "
                "Point CAPRICHO at the database again with `capricho download --set-from-path "
                "<path>`, or run `capricho download --unset-path --version "
                f"{configs['version']}` to download the release instead."
            )
        logger.info(
            f"Reading ChEMBL {configs['version']} from the database you registered:\n\t{db_path}"
        )
        logger.debug(f"Registration is held in:\n\t{config_file}")
        return configs

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
        logger.info(f"Reading ChEMBL {configs['version']} from:\n\t{sql_path}")

    return configs


def get_document_table_sql(
    document_chembl_ids: list[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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

    result = run_query(query_str, downloader_configs)

    if result.empty:
        logger.warning(f"No publication details found for document IDs: {document_chembl_ids}")

    return result


def get_compound_table_sql(
    molecule_chembl_ids: list[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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
            md.withdrawn_flag
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

    result = run_query(query_str, downloader_configs)

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

        parent_data = run_query(parent_query, downloader_configs)

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

#: Columns returned by ``_COMPOUND_LOOKUP_QUERY``; keep in step with its SELECT list.
COMPOUND_HIT_COLUMNS = [
    "molecule_chembl_id",
    "molregno",
    "canonical_smiles",
    "standard_inchi_key",
    "parent_chembl_id",
    "parent_smiles",
]

CONNECTIVITY_PATTERN = re.compile(r"^[A-Z]{14}$")
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SQLITE_LOOKUP_BATCH_SIZE = 500


def _batches(values: Sequence, size: int = SQLITE_LOOKUP_BATCH_SIZE):
    """Yield bounded lists so lookups stay below SQLite parameter/expression limits."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def get_compounds_by_inchikey_sql(
    inchi_keys: Sequence[str] | None = None,
    connectivities: Sequence[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
) -> pd.DataFrame:
    """Look up compounds by full standard InChIKey and/or connectivity block.

    Args:
        inchi_keys: full standard InChIKeys to match exactly.
        connectivities: 14-character connectivity blocks to match as key prefixes.
        prefix: optional prefix for an alternative data directory.
        version: optional ChEMBL version to use.

    Returns:
        Compound identifiers, structures, and parent-compound information. The result is empty
        when nothing matches.
    """
    if not inchi_keys and not connectivities:
        raise ValueError("Provide at least one of 'inchi_keys' or 'connectivities'.")

    unique_keys = sorted(set(inchi_keys or []))
    for key in unique_keys:
        if not isinstance(key, str) or not INCHIKEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid standard InChIKey: {key!r}")

    unique_connectivities = sorted(set(connectivities or []))
    for connectivity in unique_connectivities:
        if not isinstance(connectivity, str) or not CONNECTIVITY_PATTERN.fullmatch(connectivity):
            raise ValueError(
                f"Invalid connectivity {connectivity!r}: expected 14 uppercase letters, "
                "i.e. the first block of a standard InChIKey."
            )

    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)
    frames = []
    for batch in _batches(unique_keys):
        where_clause = f"cs.standard_inchi_key IN ({', '.join('?' for _ in batch)})"
        query_str = _COMPOUND_LOOKUP_QUERY.format(where_clause=where_clause)
        logger.debug(f"Generated SQL query for compound lookup:\n{query_str}")
        frames.append(run_query(query_str, downloader_configs, params=batch))

    for batch in _batches(unique_connectivities):
        # GLOB with a trailing wildcard is rewritten by SQLite into a range scan on
        # idx_cmpdstr_stdkey, unlike substr(), which would force a full table scan.
        where_clause = " OR\n        ".join("cs.standard_inchi_key GLOB ?" for _ in batch)
        query_str = _COMPOUND_LOOKUP_QUERY.format(where_clause=where_clause)
        logger.debug(f"Generated SQL query for compound lookup:\n{query_str}")
        frames.append(run_query(query_str, downloader_configs, params=[f"{value}*" for value in batch]))

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="molregno")
    if result.empty:
        logger.warning(
            f"No compounds found for {len(unique_keys)} InChIKey(s) and "
            f"{len(unique_connectivities)} connectivity value(s)."
        )
    return result


def get_compounds_by_molregno_sql(
    molregnos: Sequence[int],
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
) -> pd.DataFrame:
    """Look up compounds by the internal ChEMBL ``molregno`` identifier."""
    if len(molregnos) == 0:
        raise ValueError("No molregnos provided")

    unique_molregnos = sorted({int(molregno) for molregno in molregnos})
    downloader_configs = check_and_download_chembl_db(prefix=prefix, version=version)
    frames = []
    for batch in _batches(unique_molregnos):
        placeholders = ", ".join("?" for _ in batch)
        query_str = _COMPOUND_LOOKUP_QUERY.format(where_clause=f"cs.molregno IN ({placeholders})")
        logger.debug(f"Generated SQL query for molregno lookup:\n{query_str}")
        frames.append(run_query(query_str, downloader_configs, params=batch))

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="molregno")
    if result.empty:
        logger.warning(f"No compounds found for {len(unique_molregnos)} molregno value(s).")
    return result


def get_assay_table_sql(
    assay_chembl_ids: list[str] | None = None,
    confidence_scores: list[int] | None = None,
    assay_types: list[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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

    result = run_query(query_str, downloader_configs)

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
    molecule_chembl_ids: list[str] | None = None,
    target_chembl_ids: list[str] | None = None,
    assay_chembl_ids: list[str] | None = None,
    document_chembl_ids: list[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
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
            act.activity_comment,
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

    result = run_query(query_str, downloader_configs)

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
    molecule_chembl_ids: list | str | None = None,
    target_chembl_ids: list | str | None = None,
    assay_chembl_ids: list | str | None = None,
    document_chembl_ids: list | str | None = None,
    standard_relation: list[str] | None = None,
    standard_type: list[str] | None = None,
    standard_units: list[str] | None = None,
    confidence_scores: list | tuple = (9, 8),
    assay_types: list | tuple = ("B", "F"),
    chembl_release: int | None = None,
    additional_fields: list[str] | None = None,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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

    # docs.chembl_release_id was introduced in ChEMBL 33; earlier releases (e.g. the
    # ChEMBL 32 that Landrum and Riniker used) lack the column, so selecting it errors.
    records_release = int(str(downloader_configs["version"])) >= 33

    if chembl_release:
        if not records_release:
            raise ValueError(
                f"ChEMBL {downloader_configs['version']} does not record which release a "
                "document came from (docs.chembl_release_id was introduced in ChEMBL 33), so "
                "chembl_release cannot be applied. Drop the release filter, or query ChEMBL 33 "
                "or newer."
            )
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
        "act.activity_comment",
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
        # NULL keeps the chembl_release column that downstream aggregation requires.
        "d.chembl_release_id AS chembl_release" if records_release else "NULL AS chembl_release",
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

    return run_query(query_str, downloader_configs).assign(
        mutation=lambda x: x["mutation"].fillna("WT")
    )


def get_target_names_sql(
    target_chembl_ids: list[str],
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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

    result = run_query(query_str, downloader_configs)

    if result.empty:
        logger.warning(f"No targets found for IDs: {target_chembl_ids}")
        return {}

    return dict(zip(result["chembl_id"], result["pref_name"], strict=True))


def get_assay_size_sql(
    assay_chembl_ids: list | str,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
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

    return run_query(query_str, downloader_configs)
