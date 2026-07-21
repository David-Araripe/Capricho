"""Fetch compounds from PubChem, paced and retried so a batch survives a bad moment."""

import time
from functools import wraps

from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

try:
    from pubchempy import Compound, PubChemHTTPError, get_compounds
except ImportError:
    raise ImportError('pubchempy is required for this module. To install: pip install "pubchempy>=1.0.5"')

from ..core.rate_limit import rate_limit

# 1.0.5 is where `Compound.isomeric_smiles` became `.smiles` and `.canonical_smiles` became
# `.connectivity_smiles`. Reading the old names on it only emits a warning, so checking for
# a new one is what keeps an older release from failing later with an AttributeError.
if not hasattr(Compound, "connectivity_smiles"):
    raise ImportError("pubchempy >= 1.0.5 is required for this module. To upgrade: pip install -U pubchempy")

# Codes PubChem answers with when it is momentarily overloaded, so worth retrying.
TRANSIENT_HTTP_CODES = (500, 502, 503, 504)

# One 4 calls/s budget (PubChem allows 5/s) shared by every request this module issues,
# including the extra one hidden behind `Compound.synonyms`.
pubchem_rate_limit = rate_limit(4)


def retry_on_server_error(max_attempts: int = 3, backoff: float = 1.0):
    """Decorator to repeat a PubChem request that was answered with a transient error.

    Args:
        max_attempts: how many times to call the function, the first attempt included.
        backoff: seconds to wait before the first retry, doubling on every further retry.

    Returns:
        Decorator: Function decorator that will retry the decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except PubChemHTTPError as e:
                    if e.code not in TRANSIENT_HTTP_CODES or attempt == max_attempts - 1:
                        raise
                    time_to_wait = backoff * 2**attempt
                    logger.warning(
                        f"PubChem answered with a transient error ({e}). Retrying in "
                        f"{time_to_wait:.1f}s, attempt {attempt + 2} of {max_attempts}"
                    )
                    time.sleep(time_to_wait)

        return wrapper

    return decorator


@retry_on_server_error()
@pubchem_rate_limit
def get_compound_by(cpd_input, input_type: str = "name") -> list[Compound | None]:
    """Get a compound by a specific identifier using pubchempy.

    Args:
        cpd_input: input to fetch
        input_type: type of input. Defaults to "name".

    Raises:
        ValueError: if invalid compound_input is provided

    Returns:
        List[Compound: PubChem Compound object]
    """
    supported_inputs = ["name", "smiles", "sdf", "inchi", "inchikey", "formula"]
    if input_type not in supported_inputs:
        raise ValueError("Invalid compound_input")
    return get_compounds(cpd_input, input_type, list_return="flat")


@retry_on_server_error()
@pubchem_rate_limit
def get_compound_synonyms(compound: Compound) -> list[str] | None:
    """Get the synonyms of a compound, which pubchempy resolves with a separate request.

    Args:
        compound: PubChem Compound object to get the synonyms of

    Returns:
        List[str]: ranked list of the names associated with the compound
    """
    return compound.synonyms


def get_multiple_compounds(cpd_list, input_type: str = "name", n_jobs=4) -> list[list[Compound | None]]:
    """
    Fetch multiple compounds in parallel using joblib.

    Args:
        cpd_list: List of compound identifiers to fetch
        input_type: Type of input (name, smiles, etc.)
        n_jobs: Number of parallel jobs (4 is advised as max here)

    Raises:
        Exception: the error raised for the first compound, if every compound failed.
            Nothing coming through points at PubChem being unreachable rather than at
            the inputs being unknown to it.

    Returns:
        list: List of compound objects, None where the compound could not be fetched
    """
    failures = []

    def fetch_or_report(compound):
        try:
            return get_compound_by(compound, input_type)
        except Exception as e:
            logger.warning(f"Could not fetch {compound!r} from PubChem: {e}")
            failures.append(e)
            return None

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(fetch_or_report)(compound) for compound in tqdm(cpd_list)
    )

    if failures:
        if len(failures) == len(cpd_list):
            raise failures[0]
        logger.error(f"{len(failures)} out of {len(cpd_list)} compounds could not be fetched")

    return results
