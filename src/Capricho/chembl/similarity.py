"""Module with methods to find compounds in ChEMBL from SMILES strings.

Three routes are available. :func:`search_by_structure` resolves a SMILES to the ChEMBL entries
describing the same molecule, :func:`search_by_similarity` ranks the whole database by
fingerprint similarity, and :func:`get_and_curate_chembl_compounds` queries the ChEMBL web API.
The first two run against the local database and need no network access once it is downloaded.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from chemFilters.chem.standardizers import ChemStandardizer, InchiHandling

from ..core.fp_utils import calculate_mixed_FPs
from ..core.smiles_utils import clean_mixtures
from ..core.stats_make import repeated_indices_from_array_series
from ..logger import logger
from .api.downloader import get_compounds_by_inchikey_sql, get_compounds_by_molregno_sql
from .api.fingerprint_index import load_fingerprint_index

COMPOUND_HIT_COLUMNS = [
    "molecule_chembl_id",
    "molregno",
    "canonical_smiles",
    "standard_inchi_key",
    "parent_chembl_id",
    "parent_smiles",
]


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
        min_id = sorted(df.loc[repeats, "molecule_chembl_id"], key=lambda _id: int(_id.lstrip("CHEMBL")))[0]
        df.loc[repeats, "repeats"] = min_id

    df = df.drop(columns=["fps"])

    return df


def _as_smiles_list(smiles: Union[str, Sequence[str]]) -> List[str]:
    """Accept a single SMILES or a sequence of them and always return a list."""
    if isinstance(smiles, str):
        return [smiles]
    return list(smiles)


def _describe_queries(smiles: List[str], standardize: bool, chirality: bool, n_jobs: int) -> pd.DataFrame:
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

    valid = [smi for smi in standard_smiles if isinstance(smi, str)]
    inchikey_writer = InchiHandling(convert_to="inchikey", n_jobs=n_jobs, from_smi=True, progress=False)
    keys = dict(zip(valid, inchikey_writer(valid))) if valid else {}

    query_inchikeys = [keys.get(smi) if isinstance(smi, str) else None for smi in standard_smiles]
    query_inchikeys = [key if isinstance(key, str) and key else None for key in query_inchikeys]

    for smi, key in zip(smiles, query_inchikeys):
        if key is None:
            logger.warning(f"Could not derive an InChI key for the query SMILES: {smi}")

    return pd.DataFrame(
        {
            "query_smiles": smiles,
            "standard_smiles": standard_smiles,
            "query_inchikey": query_inchikeys,
            "connectivity": [key[:14] if key else None for key in query_inchikeys],
        }
    )


def search_by_structure(
    smiles: Union[str, Sequence[str]],
    standardize: bool = True,
    chirality: bool = True,
    n_jobs: int = 1,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
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
    queries = _describe_queries(_as_smiles_list(smiles), standardize, chirality, n_jobs)

    searchable = queries.dropna(subset=["query_inchikey"])
    if searchable.empty:
        logger.warning("None of the query SMILES could be resolved to an InChI key.")
        hits = pd.DataFrame(columns=COMPOUND_HIT_COLUMNS + ["connectivity"])
    else:
        hits = get_compounds_by_inchikey_sql(
            inchi_keys=searchable["query_inchikey"].tolist(),
            connectivities=searchable["connectivity"].tolist(),
            prefix=prefix,
            version=version,
        )
        if hits.empty:
            hits = pd.DataFrame(columns=COMPOUND_HIT_COLUMNS + ["connectivity"])
        else:
            hits = hits.assign(connectivity=lambda x: x["standard_inchi_key"].str[:14])

    result = queries.merge(hits, on="connectivity", how="left")
    result["match_type"] = np.where(
        result["molecule_chembl_id"].isna(),
        "no_match",
        np.where(result["standard_inchi_key"] == result["query_inchikey"], "exact", "connectivity"),
    )

    # Order exact hits ahead of the looser connectivity hits, keeping the input order of queries.
    result = (
        result.assign(_rank=lambda x: x["match_type"].map({"exact": 0, "connectivity": 1, "no_match": 2}))
        .sort_values(["_rank", "molecule_chembl_id"], kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    n_matched = result.loc[result["match_type"] != "no_match", "query_smiles"].nunique()
    logger.info(
        f"Structure search matched {n_matched}/{len(queries)} queries to {len(result[result['match_type'] != 'no_match'])} "
        "ChEMBL compounds."
    )

    return result[
        ["query_smiles", "standard_smiles", "query_inchikey", "connectivity", "match_type"]
        + COMPOUND_HIT_COLUMNS
    ]


def search_by_similarity(
    smiles: Union[str, Sequence[str]],
    threshold: float = 0.7,
    top_k: Optional[int] = None,
    metric: str = "tanimoto",
    n_workers: int = 1,
    in_memory: bool = True,
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
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
        threshold: minimum similarity coefficient a compound must reach to be returned.
            Defaults to 0.7; FPSim2 performs best at thresholds of 0.7 and above.
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
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"'threshold' must be between 0 and 1, got {threshold}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"'top_k' must be a positive integer, got {top_k}")

    query_list = _as_smiles_list(smiles)
    engine = load_fingerprint_index(prefix=prefix, version=version, in_memory=in_memory)

    searches = []
    for smi in query_list:
        try:
            if top_k is not None:
                found = (
                    engine.top_k(smi, k=top_k, threshold=threshold, metric=metric, n_workers=n_workers)
                    if in_memory
                    else engine.on_disk_top_k(smi, k=top_k, threshold=threshold, metric=metric)
                )
            else:
                found = (
                    engine.similarity(smi, threshold=threshold, metric=metric, n_workers=n_workers)
                    if in_memory
                    else engine.on_disk_similarity(smi, threshold=threshold, metric=metric)
                )
        except Exception as exc:  # RDKit and FPSim2 raise different errors for unusable queries
            logger.warning(f"Similarity search failed for the query SMILES {smi!r}: {exc}")
            continue

        if len(found) == 0:
            logger.warning(f"No compound reached a similarity of {threshold} for the query: {smi}")
            continue

        searches.append(
            pd.DataFrame(found)
            .rename(columns={"mol_id": "molregno", "coeff": "similarity"})
            .assign(query_smiles=smi)
        )

    if not searches:
        return pd.DataFrame(columns=["query_smiles", "similarity"] + COMPOUND_HIT_COLUMNS)

    found_df = pd.concat(searches, ignore_index=True)
    compounds = get_compounds_by_molregno_sql(
        found_df["molregno"].unique().tolist(), prefix=prefix, version=version
    )

    result = (
        found_df.merge(compounds, on="molregno", how="left")
        .sort_values(["query_smiles", "similarity"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )

    logger.info(
        f"Similarity search returned {len(result)} compounds for {found_df['query_smiles'].nunique()}"
        f"/{len(query_list)} queries at a {metric} threshold of {threshold}."
    )

    return result[["query_smiles", "similarity"] + COMPOUND_HIT_COLUMNS]
