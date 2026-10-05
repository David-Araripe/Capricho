"""Module with methods to find compounds in ChEMBL from SMILES strings.

Three routes are available. :func:`search_by_structure` resolves a SMILES to the ChEMBL entries
describing the same molecule, :func:`search_by_similarity` ranks the whole database by
fingerprint similarity, and :func:`get_and_curate_chembl_compounds` queries the ChEMBL web API.
The first two run against the local database. Once the database and, for similarity searches,
the fingerprint index are cached, passing ``version=`` avoids all network access.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from numbers import Integral

import numpy as np
import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer, InchiHandling
from rdkit import Chem
from rdkit import __version__ as rdkit_version

from .. import __version__ as capricho_version
from ..core.fp_utils import calculate_mixed_FPs
from ..core.smiles_utils import clean_mixtures
from ..core.stats_make import repeated_indices_from_array_series
from ..logger import logger
from .api.downloader import (
    COMPOUND_HIT_COLUMNS,
    _latest_version,
    get_compounds_by_inchikey_sql,
    get_compounds_by_molregno_sql,
)
from .api.fingerprint_index import load_fingerprint_index

MATCH_TYPE_ORDER = {"exact": 0, "connectivity": 1, "no_match": 2}
SIMILARITY_METRICS = frozenset({"tanimoto", "dice", "cosine"})


def get_and_curate_chembl_compounds(
    smiles: list[str], similarity: float, n_threads: int = 5, chirality: bool = True
) -> pd.DataFrame:
    """Get similar compounds from the ChEMBL API using multiple threads and identify
    repeats based on fingerprints. API call is forced to a limit of 5 calls per second
    not to overload ChEMBL servers. The fingerprints used to identify repeats is a combination of
    rdkit and morgan. Compounds identified as repeats will have the smallest `molecule_chembl_id`
    from the identified repeats listed in the `repeats` column.

    Returned fields (other than the ChEMBL fields):

    - querySmiles -> the original SMILES used to query the similar compounds
    - standard_smiles -> the standardized SMILES of the compound
    - repeats -> the smallest `molecule_chembl_id` from the identified repeats (based on fingerprint
      similarity after compound standardization + salt/solvent removal)

    Usage:

    >>> from Capricho.chembl.similarity import get_and_curate_chembl_compounds
    >>> aspirin = 'O=C(C)Oc1ccccc1C(=O)O'
    >>> df = get_and_curate_chembl_compounds([aspirin], similarity=70)

    Args:
        smiles: list of SMILES to find similar molecules to.
        similarity: similarity threshold to use for the search. Value should be between 40 and 100.
        n_threads: Number of threads to use for searching the similar compounds. Defaults to 5.
        chirality: Whether to consider chirality when identifying repeats. Defaults to True.

    Returns:
        pd.DataFrame: a DataFrame with the processed similar molecules.
    """
    from .api.webresource import get_similarity_compound_table

    if not isinstance(smiles, list):
        raise TypeError("The 'smiles' argument must be a list of SMILES.")

    extracted = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for smi in smiles:
            futures.append(executor.submit(get_similarity_compound_table, smi, similarity))
        for future in as_completed(futures):
            extracted.append(future.result())

    similar_compounds = pd.concat(extracted, ignore_index=True)

    if similar_compounds.empty:
        logger.warning(
            "No similar compounds found for the provided SMILES with the given similarity threshold {similarity}."
        )
        return pd.DataFrame()

    stdzer = ChemStandardizer(from_smi=True, n_jobs=1, verbose=False, isomeric=chirality, progress=True)
    df = (
        similar_compounds.assign(standard_smiles=lambda x: stdzer(x["canonical_smiles"]))
        .dropna(subset=["standard_smiles"])  # drop if no structure is found
        .query("standard_smiles.notna()")
        .assign(final_smiles=lambda x: x["standard_smiles"].apply(clean_mixtures))
        .drop(columns="standard_smiles")
        .rename(columns={"final_smiles": "standard_smiles"})
    )

    if df.shape[0] == 1:
        return df
    elif df.shape[0] == 0:
        return pd.DataFrame()

    # Calculate fingerprints and identify repeats
    fps = calculate_mixed_FPs(
        df["standard_smiles"].tolist(), n_jobs=n_threads, morgan_kwargs={"useChirality": chirality}
    )
    df = df.assign(fps=fps).assign(repeats=False)
    repeats_idxs = repeated_indices_from_array_series(df["fps"])
    for repeats in repeats_idxs:
        min_id = min(df.loc[repeats, "molecule_chembl_id"], key=lambda _id: int(_id.lstrip("CHEMBL")))
        df.loc[repeats, "repeats"] = min_id

    df = df.drop(columns=["fps"])

    return df


def _as_smiles_list(smiles: str | Sequence[str]) -> list[str]:
    """Accept a single SMILES or a sequence of them and always return a list."""
    if isinstance(smiles, str):
        return [smiles]
    return list(smiles)


def _parse_query_smiles(smi: str) -> Chem.Mol | None:
    """Keep unusable inputs out of standardization and fingerprint generation."""
    if not isinstance(smi, str):
        logger.warning(f"Structure search requires a SMILES string, got {smi!r}")
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except (RuntimeError, ValueError) as exc:
        logger.warning(f"Could not parse the query SMILES {smi!r}: {exc}")
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        logger.warning(f"Could not parse a non-empty molecule from the query SMILES: {smi!r}")
        return None
    return mol


def _search_provenance(version: int | str | None, **settings) -> dict:
    """Record the resolved release and settings without resolving unused databases."""
    return {
        "capricho_version": capricho_version,
        "chembl_version": str(version) if version is not None else None,
        "rdkit_version": rdkit_version,
        **settings,
    }


def _describe_queries(smiles: list[str], standardize: bool, chirality: bool, n_jobs: int) -> pd.DataFrame:
    """Build the query frame holding, per input SMILES, the structure keys used for lookup.

    Returns a DataFrame with query_smiles, standard_smiles, query_inchikey and connectivity.
    Entries RDKit cannot parse keep a null key and are reported as unmatched downstream, rather
    than being dropped.
    """
    unique_inputs = list(dict.fromkeys(smi for smi in smiles if isinstance(smi, str)))
    parsed = {smi: mol for smi in unique_inputs if (mol := _parse_query_smiles(smi)) is not None}
    if standardize and parsed:
        stdzer = ChemStandardizer(
            from_smi=False, n_jobs=n_jobs, verbose=False, isomeric=chirality, progress=False
        )
        standardized = {
            original: clean_mixtures(smi) if isinstance(smi, str) else None
            for original, smi in zip(parsed, stdzer(list(parsed.values())), strict=True)
        }
    else:
        standardized = {smi: smi for smi in parsed}
    standard_smiles = [standardized.get(smi) if isinstance(smi, str) else None for smi in smiles]

    # Deduplicate before generating keys: repeated SMILES would otherwise cost one InChI
    # conversion each, which dominates the runtime for large query lists.
    unique = list(dict.fromkeys(smi for smi in standard_smiles if isinstance(smi, str)))
    inchikey_writer = InchiHandling(convert_to="inchikey", n_jobs=n_jobs, from_smi=True, progress=False)
    keys = (
        {smi: key for smi, key in zip(unique, inchikey_writer(unique), strict=True) if key} if unique else {}
    )

    query_inchikeys = [keys.get(smi) for smi in standard_smiles]

    for smi, key in zip(smiles, query_inchikeys, strict=True):
        if key is None:
            logger.warning(f"Could not derive an InChI key for the query SMILES: {smi}")

    queries = pd.DataFrame(
        {
            "_query_index": range(len(smiles)),
            "query_smiles": smiles,
            "standard_smiles": standard_smiles,
            "query_inchikey": query_inchikeys,
        }
    )
    return queries.assign(connectivity=queries["query_inchikey"].str[:14])


def search_by_structure(
    smiles: str | Sequence[str],
    standardize: bool = True,
    chirality: bool = True,
    n_jobs: int = 1,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
) -> pd.DataFrame:
    """Find the ChEMBL entries that describe the same compound as each query SMILES.

    Matching happens on the standard InChI key, reported at two levels of strictness in the
    ``match_type`` column:

    - ``exact``: the full InChI key of ``standard_smiles`` matches the stored key. This is
      identity after the selected query normalization, not necessarily identity of the raw input.
    - ``connectivity``: only the first 14 characters match, so the molecular skeleton agrees
      while stereochemistry or isotopic labelling differ. This is the same notion of compound
      equality used by ``capricho get --compound-equality connectivity``.
    - ``no_match``: nothing in ChEMBL matched that query. These rows are kept so that queries
      never disappear silently from the output.

    By default the ChEMBL parent pipeline strips salts, neutralizes compounds, and removes
    isotope labels before deriving the query key. For example, ``[13CH3]CO`` is normalized to
    ``CCO`` and can be an exact match to ordinary ethanol. Set ``standardize=False`` to derive
    keys from the input molecule without those transformations. Standard InChI itself can still
    normalize tautomeric representations. Setting ``chirality=False`` also removes specified
    stereochemistry during standardization.

    Note that salt and co-crystal entries do not share the connectivity of their parent, since
    the counter-ion is part of the skeleton the InChI key is derived from. A salt-stripped query
    therefore matches the parent structure, not the salt entries registered against it. Every hit
    reports its own parent from ``molecule_hierarchy`` in ``parent_chembl_id``.

    Usage:

    >>> from Capricho.chembl.similarity import search_by_structure
    >>> hits = search_by_structure("CC(=O)Oc1ccccc1C(=O)O")
    >>> hits.query("match_type == 'exact'")["molecule_chembl_id"].tolist()
    ['CHEMBL25']

    Args:
        smiles: a single SMILES or a sequence of SMILES to look up.
        standardize: whether to standardize the query with the ChEMBL structure pipeline and
            strip salts and solvents, neutralize compounds, and remove isotope labels before
            deriving keys. Set False for identity-sensitive queries. Defaults to True.
        chirality: whether standardization keeps stereochemistry. Only relevant when
            ``standardize`` is True. Defaults to True.
        n_jobs: number of jobs used for standardization and InChI key generation. Defaults to 1.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use. Defaults to the latest available version.

    Returns:
        pd.DataFrame: one row per query/hit pair, with the query columns (query_index, query_smiles,
        standard_smiles, query_inchikey, connectivity), the ``match_type`` label and the hit
        columns (molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key,
        parent_chembl_id, parent_smiles). ``query_index`` is the zero-based input position,
        including repeated queries. ``attrs['capricho_search']`` records the resolved release,
        software versions and normalization settings; an unused release remains None.
    """
    query_list = _as_smiles_list(smiles)
    output_columns = [
        "query_index",
        "query_smiles",
        "standard_smiles",
        "query_inchikey",
        "connectivity",
        "match_type",
        *COMPOUND_HIT_COLUMNS,
    ]
    if not query_list:
        result = pd.DataFrame(columns=output_columns)
        result.attrs["capricho_search"] = _search_provenance(
            version, search_type="structure", standardize=standardize, chirality=chirality
        )
        return result

    queries = _describe_queries(query_list, standardize, chirality, n_jobs)

    searchable = queries.dropna(subset=["query_inchikey"])
    if searchable.empty:
        logger.warning("None of the query SMILES could be resolved to an InChI key.")
        hits = pd.DataFrame(columns=COMPOUND_HIT_COLUMNS)
    else:
        version = version if version is not None else _latest_version()
        # Only the connectivities are needed: every full key starts with its own connectivity, so
        # an exact-key clause would match a strict subset of what the prefix scan already returns.
        hits = get_compounds_by_inchikey_sql(
            connectivities=searchable["connectivity"].tolist(), prefix=prefix, version=version
        )
    # reindex also gives the expected columns back when the lookup returned no rows at all
    hits = hits.reindex(columns=COMPOUND_HIT_COLUMNS).assign(
        connectivity=lambda x: x["standard_inchi_key"].str[:14]
    )

    result = queries.merge(hits, on="connectivity", how="left")
    result["match_type"] = np.select(
        [
            result["molecule_chembl_id"].isna(),
            result["standard_inchi_key"] == result["query_inchikey"],
        ],
        ["no_match", "exact"],
        default="connectivity",
    )

    # Keep query occurrences in input order, with exact hits before connectivity-only hits.
    result = result.sort_values(
        ["_query_index", "match_type", "molecule_chembl_id"],
        key=lambda col: col.map(MATCH_TYPE_ORDER) if col.name == "match_type" else col,
        kind="stable",
    ).reset_index(drop=True)

    matched = result[result["match_type"] != "no_match"]
    logger.info(
        f"Structure search matched {matched['_query_index'].nunique()}/{len(queries)} queries "
        f"to {len(matched)} ChEMBL compounds."
    )

    result = result.rename(columns={"_query_index": "query_index"})[output_columns]
    result.attrs["capricho_search"] = _search_provenance(
        version, search_type="structure", standardize=standardize, chirality=chirality
    )
    return result


def search_by_similarity(
    smiles: str | Sequence[str],
    threshold: float = 0.7,
    top_k: int | None = None,
    metric: str = "tanimoto",
    n_workers: int = 1,
    in_memory: bool = True,
    prefix: Sequence[str] | None = None,
    version: int | str | None = None,
) -> pd.DataFrame:
    """Rank every compound in ChEMBL by fingerprint similarity to each query SMILES.

    The search runs against the FPSim2 index published by ChEMBL, which holds 2048-bit radius 2
    Morgan fingerprints. Results are exact rather than approximate: FPSim2 uses popcount bounds
    to skip fingerprints that cannot reach the threshold, so no compound above the threshold is
    missed. Queries are fingerprinted straight from the input SMILES with RDKit sanitization,
    matching how the index itself was built, so they are deliberately not run through the
    Capricho standardization pipeline.

    Scores describe fingerprint similarity, not molecular identity: distinct compounds can
    score 1.0. The published Morgan index is not sensitive to stereochemistry. Query fingerprints
    use the installed RDKit; changes since the version that built the index may affect scores.

    Usage:

    >>> from Capricho.chembl.similarity import search_by_similarity
    >>> hits = search_by_similarity("CC(=O)Oc1ccccc1C(=O)O", threshold=0.8)

    Args:
        smiles: a single SMILES or a sequence of SMILES to search with.
        threshold: minimum similarity coefficient a compound must reach to be returned. Must be
            greater than zero. Defaults to 0.7; FPSim2 performs best at 0.7 and above.
        top_k: if given, keep only the k most similar compounds per query. The threshold still
            applies, so fewer than k compounds may be returned. Defaults to None.
        metric: similarity metric, one of "tanimoto", "dice" or "cosine". Defaults to "tanimoto".
        n_workers: workers used to split a single query across cores. In-memory searches use
            threads; on-disk searches use processes. On-disk searches with more than one worker
            require an importable Python entry point, guarded by ``if __name__ == '__main__':``
            on platforms using spawn. Use one worker in notebooks and stdin sessions. Defaults to 1.
        in_memory: whether to load the index into memory (~1 GB) instead of searching from disk.
            Defaults to True.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use. Defaults to the latest available version.

    Returns:
        pd.DataFrame: one row per query/hit pair with query_index, query_smiles, similarity and the hit
        columns (molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key,
        parent_chembl_id, parent_smiles), sorted by descending similarity within each query.
        A query with no hit above the threshold, or one RDKit cannot parse, keeps a single row
        with null hit columns, so every input is accounted for in the output. Drop them with
        ``.dropna(subset=["molecule_chembl_id"])``. ``query_index`` is the zero-based input
        position. Repeated SMILES are searched once, retaining a result set for each occurrence.
        ``attrs['capricho_search']`` records the resolved release, search settings, fingerprint
        parameters and software versions. Save these attributes separately when exporting CSV.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"'threshold' must be greater than 0 and at most 1, got {threshold}")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1):
        raise ValueError(f"'top_k' must be a positive integer, got {top_k}")
    if isinstance(n_workers, bool) or not isinstance(n_workers, Integral) or n_workers < 1:
        raise ValueError(f"'n_workers' must be a positive integer, got {n_workers}")
    if metric not in SIMILARITY_METRICS:
        raise ValueError(f"'metric' must be one of {sorted(SIMILARITY_METRICS)}, got {metric!r}")

    query_list = _as_smiles_list(smiles)
    output_columns = ["query_index", "query_smiles", "similarity", *COMPOUND_HIT_COLUMNS]
    provenance = _search_provenance(
        version, search_type="similarity", threshold=threshold, top_k=top_k, metric=metric
    )
    if not query_list:
        result = pd.DataFrame(columns=output_columns)
        result.attrs["capricho_search"] = provenance
        return result

    unique_inputs = list(dict.fromkeys(smi for smi in query_list if isinstance(smi, str)))
    parsed_queries = {smi: mol for smi in unique_inputs if (mol := _parse_query_smiles(smi)) is not None}

    searches = []
    if parsed_queries:
        version = version if version is not None else _latest_version()
        engine = load_fingerprint_index(prefix=prefix, version=version, in_memory=in_memory)
        try:
            fpsim2_version = package_version("FPSim2")
        except PackageNotFoundError:
            fpsim2_version = None
        provenance.update(
            chembl_version=str(version),
            fpsim2_version=fpsim2_version,
            fp_type=engine.fp_type,
            fp_params=dict(engine.fp_params),
            index_rdkit_version=engine.rdkit_ver,
            index_fpsim2_version=engine.fpsim2_ver,
        )

        # Both choices are fixed for the whole call, so resolve the engine method once and leave
        # the loop to deal with results only.
        method = "top_k" if top_k is not None else "similarity"
        search = getattr(engine, method if in_memory else f"on_disk_{method}")
        search_kwargs = {"threshold": threshold, "metric": metric, "n_workers": n_workers}
        if top_k is not None:
            search_kwargs["k"] = top_k

        for smi, query_mol in parsed_queries.items():
            # Passing the parsed molecule means invalid input is handled above while index, I/O,
            # and worker failures propagate instead of being misreported as an unmatched query.
            found = search(query_mol, **search_kwargs)
            if len(found) == 0:
                logger.warning(f"No compound reached a similarity of {threshold} for the query: {smi}")
                continue
            searches.append(
                pd.DataFrame(found)
                .rename(columns={"mol_id": "molregno", "coeff": "similarity"})
                .assign(query_smiles=smi)
            )

    hits = (
        pd.concat(searches, ignore_index=True)
        if searches
        else pd.DataFrame(columns=["query_smiles", "molregno", "similarity"])
    )
    molregnos = hits["molregno"].unique().tolist()
    compounds = (
        get_compounds_by_molregno_sql(molregnos, prefix=prefix, version=version)
        if molregnos
        else pd.DataFrame(columns=COMPOUND_HIT_COLUMNS)
    )
    hits = hits.merge(compounds.reindex(columns=COMPOUND_HIT_COLUMNS), on="molregno", how="left")

    # Each unique SMILES owns one hit set. Expand it back to every input occurrence and retain
    # the position so callers can join results to their original query records.
    result = (
        pd.DataFrame({"_query_index": range(len(query_list)), "query_smiles": query_list})
        .merge(hits, on="query_smiles", how="left")
        .sort_values(["_query_index", "similarity"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )

    matched = result.dropna(subset=["molecule_chembl_id"])
    logger.info(
        f"Similarity search returned {len(matched)} compounds for "
        f"{matched['_query_index'].nunique()}/{len(query_list)} queries "
        f"at a {metric} threshold of {threshold}."
    )

    result = result.rename(columns={"_query_index": "query_index"})[output_columns]
    result.attrs["capricho_search"] = provenance
    return result
