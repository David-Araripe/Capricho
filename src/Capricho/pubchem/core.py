from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

try:
    from pubchempy import Compound, get_compounds
except ImportError as exc:
    raise ImportError("pubchempy is required for this module. To install: pip install pubchempy") from exc

from ..core.rate_limit import rate_limit


@rate_limit(4)  # 4 calls per second not to exceed the PubChem API rate limit (5/sec)
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


def get_multiple_compounds(cpd_list, input_type: str = "name", n_jobs=4) -> list[list[Compound | None]]:
    """
    Fetch multiple compounds in parallel using joblib.

    Args:
        cpd_list: List of compound identifiers to fetch
        input_type: Type of input (name, smiles, etc.)
        n_jobs: Number of parallel jobs (4 is advised as max here)

    Returns:
        list: List of compound objects
    """
    results = []

    try:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(get_compound_by)(compound, input_type) for compound in tqdm(cpd_list)
        )

        return results

    except Exception as e:
        logger.error(f"Error in parallel processing: {e!s}")
        return []
