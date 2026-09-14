"""Module with methods to find compounds in ChEMBL from SMILES strings.

Three routes are available. :func:`search_by_structure` resolves a SMILES to the ChEMBL entries
describing the same molecule, :func:`search_by_similarity` ranks the whole database by
fingerprint similarity, and :func:`get_and_curate_chembl_compounds` queries the ChEMBL web API.
The first two run against the local database; once it is downloaded their only network access is
resolving which ChEMBL release is latest, which passing ``version=`` avoids entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral

import numpy as np
import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer, InchiHandling
from rdkit import Chem

from ..core.fp_utils import calculate_mixed_FPs
from ..core.smiles_utils import clean_mixtures
from ..core.stats_make import repeated_indices_from_array_series
from ..logger import logger
from .api.downloader import (
    COMPOUND_HIT_COLUMNS,
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


def _describe_queries(smiles: list[str], standardize: bool, chirality: bool, n_jobs: int) -> pd.DataFrame:
    """Build the query frame holding, per input SMILES, the structure keys used for lookup.

    Returns a DataFrame with query_smiles, standard_smiles, query_inchikey and connectivity.
    Entries RDKit cannot parse keep a null key and are reported as unmatched downstream, rather
    than being dropped.
    """
    if standardize:
        stdzer = ChemStandardizer(
            from_smi=True, n_jobs=n_jobs, verbose=False, isomeric=chirality, progress=False
        )
        standard_smiles = [clean_mixtures(smi) if isinstance(smi, str) else None for smi in stdzer(smiles)]
    else:
        standard_smiles = list(smiles)

    # Deduplicate before generating keys: repeated SMILES would otherwise cost one InChI
    # conversion each, which dominates the runtime for large query lists.
    unique = list(dict.fromkeys(smi for smi in standard_smiles if isinstance(smi, str)))
    inchikey_writer = InchiHandling(convert_to="inchikey", n_jobs=n_jobs, from_smi=True, progress=False)
    keys = {smi: key for smi, key in zip(unique, inchikey_writer(unique)) if key} if unique else {}

    query_inchikeys = [keys.get(smi) for smi in standard_smiles]

    for smi, key in zip(smiles, query_inchikeys):
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

    - ``exact``: the full InChI key matches, so stereochemistry, isotopes and protonation state
      all agree.
    - ``connectivity``: only the first 14 characters match, so the molecular skeleton agrees
      while stereochemistry or isotopic labelling differ. This is the same notion of compound
      equality used by ``capricho get --compound-equality connectivity``.
    - ``no_match``: nothing in ChEMBL matched that query. These rows are kept so that queries
      never disappear silently from the output.

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
            strip salts and solvents before deriving keys. Recommended, since ChEMBL stores
            standardized structures. Defaults to True.
        chirality: whether standardization keeps stereochemistry. Only relevant when
            ``standardize`` is True. Defaults to True.
        n_jobs: number of jobs used for standardization and InChI key generation. Defaults to 1.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use. Defaults to the latest available version.

    Returns:
        pd.DataFrame: one row per query/hit pair, with the query columns (query_smiles,
        standard_smiles, query_inchikey, connectivity), the ``match_type`` label and the hit
        columns (molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key,
        parent_chembl_id, parent_smiles).
    """
    query_list = _as_smiles_list(smiles)
    output_columns = [
        "query_smiles",
        "standard_smiles",
        "query_inchikey",
        "connectivity",
        "match_type",
    ] + COMPOUND_HIT_COLUMNS
    if not query_list:
        return pd.DataFrame(columns=output_columns)

    queries = _describe_queries(query_list, standardize, chirality, n_jobs)

    searchable = queries.dropna(subset=["query_inchikey"])
    if searchable.empty:
        logger.warning("None of the query SMILES could be resolved to an InChI key.")
        hits = pd.DataFrame(columns=COMPOUND_HIT_COLUMNS)
    else:
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

    return result[output_columns]


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
        n_workers: threads used to split a single query across cores. Defaults to 1.
        in_memory: whether to load the index into memory (~1 GB) instead of searching from disk.
            Defaults to True.
        prefix: Optional prefix for an alternative data directory.
        version: Optional ChEMBL version to use. Defaults to the latest available version.

    Returns:
        pd.DataFrame: one row per query/hit pair with query_smiles, similarity and the hit
        columns (molecule_chembl_id, molregno, canonical_smiles, standard_inchi_key,
        parent_chembl_id, parent_smiles), sorted by descending similarity within each query.
        A query with no hit above the threshold, or one RDKit cannot parse, keeps a single row
        with null hit columns, so every input is accounted for in the output. Drop them with
        ``.dropna(subset=["molecule_chembl_id"])``.
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
    if not query_list:
        return pd.DataFrame(columns=["query_smiles", "similarity"] + COMPOUND_HIT_COLUMNS)

    parsed_queries = []
    for query_index, smi in enumerate(query_list):
        if not isinstance(smi, str):
            logger.warning(f"Similarity search requires a SMILES string, got {smi!r}")
            continue
        try:
            query_mol = Chem.MolFromSmiles(smi)
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"Could not parse the query SMILES {smi!r}: {exc}")
            continue
        if query_mol is None or query_mol.GetNumAtoms() == 0:
            logger.warning(f"Could not parse a non-empty molecule from the query SMILES: {smi!r}")
            continue
        parsed_queries.append((query_index, smi, query_mol))

    searches = []
    if parsed_queries:
        engine = load_fingerprint_index(prefix=prefix, version=version, in_memory=in_memory)

        # Both choices are fixed for the whole call, so resolve the engine method once and leave
        # the loop to deal with results only.
        method = "top_k" if top_k is not None else "similarity"
        search = getattr(engine, method if in_memory else f"on_disk_{method}")
        search_kwargs = {"threshold": threshold, "metric": metric, "n_workers": n_workers}
        if top_k is not None:
            search_kwargs["k"] = top_k

        for query_index, smi, query_mol in parsed_queries:
            # Passing the parsed molecule means invalid input is handled above while index, I/O,
            # and worker failures propagate instead of being misreported as an unmatched query.
            found = search(query_mol, **search_kwargs)
            if len(found) == 0:
                logger.warning(f"No compound reached a similarity of {threshold} for the query: {smi}")
                continue
            searches.append(
                pd.DataFrame(found)
                .rename(columns={"mol_id": "molregno", "coeff": "similarity"})
                .assign(_query_index=query_index, query_smiles=smi)
            )

    hits = (
        pd.concat(searches, ignore_index=True)
        if searches
        else pd.DataFrame(columns=["_query_index", "query_smiles", "molregno", "similarity"])
    )
    molregnos = hits["molregno"].unique().tolist()
    compounds = (
        get_compounds_by_molregno_sql(molregnos, prefix=prefix, version=version)
        if molregnos
        else pd.DataFrame(columns=COMPOUND_HIT_COLUMNS)
    )
    hits = hits.merge(compounds.reindex(columns=COMPOUND_HIT_COLUMNS), on="molregno", how="left")

    # Merge on a per-occurrence identifier, rather than SMILES text, so duplicate queries do not
    # create a Cartesian product. The identifier also preserves the caller's input order.
    result = (
        pd.DataFrame({"_query_index": range(len(query_list)), "query_smiles": query_list})
        .merge(hits, on=["_query_index", "query_smiles"], how="left")
        .sort_values(["_query_index", "similarity"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )

    matched = result.dropna(subset=["molecule_chembl_id"])
    logger.info(
        f"Similarity search returned {len(matched)} compounds for "
        f"{matched['_query_index'].nunique()}/{len(query_list)} queries "
        f"at a {metric} threshold of {threshold}."
    )

    return result[["query_smiles", "similarity"] + COMPOUND_HIT_COLUMNS]
