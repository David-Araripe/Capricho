"""Module holding access to the fingerprint index used for ChEMBL similarity searches.

ChEMBL publishes an FPSim2 index alongside every release: 2048-bit, radius 2 Morgan
fingerprints calculated with RDKit over each entry of ``compound_structures``, keyed by
``molregno``. Reusing that file avoids recomputing fingerprints for the ~2.9M structures in
ChEMBL and keeps similarity results identical to the ones served by the ChEMBL web interface.

FPSim2 is an optional dependency; install it with ``pip install capricho[similarity]``.

Note that FPSim2 reports a warning when the RDKit release used to build the index differs from
the installed one, since fingerprint generation is in principle free to change between releases.
The warning is left visible so that the discrepancy is never hidden from the user.
"""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ...logger import logger
from .downloader import check_and_download_chembl_db

if TYPE_CHECKING:
    from FPSim2 import FPSim2Engine

FPSIM2_INSTALL_HINT = (
    "FPSim2 is required to run similarity searches but is not installed. Install it with "
    "`pip install 'capricho[similarity]'` or `pip install FPSim2`."
)


def check_and_download_fingerprint_index(
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
) -> Path:
    """Check if the ChEMBL fingerprint index is present, downloading it if not.

    The index is stored next to the SQLite database, in the same pystow versioned directory.
    Resolving the version goes through :func:`check_and_download_chembl_db` so that the index
    and the database always describe the same ChEMBL release; searches need both, since the
    index is keyed by ``molregno`` and the database resolves those to ChEMBL IDs.

    Args:
        prefix: Optional prefix for an alternative data directory, as a list of path components.
        version: Optional ChEMBL version to use. Defaults to the latest available version.

    Returns:
        Path: path to the local FPSim2 ``.h5`` index.
    """
    # chembl_downloader has no public accessor for the .h5 asset; `_download_helper` builds the
    # release URL and stores the file through pystow, the same way `download_fps` does for .fps.gz.
    from chembl_downloader.api import _download_helper

    configs = check_and_download_chembl_db(prefix=prefix, version=version)

    fp_path = _download_helper(
        suffix=".h5",
        version=configs["version"],
        prefix=configs["prefix"],
        return_version=False,
    )

    if fp_path is None or not Path(fp_path).exists():
        raise FileNotFoundError(
            f"Could not download the ChEMBL fingerprint index for version {configs['version']}. "
            "Check that the release publishes a chembl_<version>.h5 file on the EBI FTP server."
        )

    logger.debug(f"Using ChEMBL fingerprint index at:\n\t{fp_path}")
    return Path(fp_path)


@lru_cache(maxsize=2)
def _load_engine(fp_path: str, in_memory: bool) -> "FPSim2Engine":
    """Load and cache an FPSim2 engine. Cached because loading the index costs a few seconds."""
    try:
        from FPSim2 import FPSim2Engine
    except ImportError as exc:
        raise ImportError(FPSIM2_INSTALL_HINT) from exc

    logger.info(f"Loading fingerprint index ({'in memory' if in_memory else 'on disk'}):\n\t{fp_path}")
    return FPSim2Engine(fp_path, in_memory_fps=in_memory)


def load_fingerprint_index(
    prefix: Optional[Sequence[str]] = None,
    version: Optional[Union[int, str]] = None,
    in_memory: bool = True,
) -> "FPSim2Engine":
    """Load the ChEMBL fingerprint index, downloading it first if needed.

    Repeated calls with the same arguments reuse a cached engine, so loading the index into
    memory only happens once per session.

    Args:
        prefix: Optional prefix for an alternative data directory, as a list of path components.
        version: Optional ChEMBL version to use. Defaults to the latest available version.
        in_memory: Whether to hold the fingerprints in memory (~1 GB for a full ChEMBL release).
            Set to False to search straight from disk, which is considerably slower but works
            when the index does not fit in RAM. Defaults to True.

    Returns:
        FPSim2Engine: the engine to run searches against.
    """
    fp_path = check_and_download_fingerprint_index(prefix=prefix, version=version)
    return _load_engine(str(fp_path), in_memory)
