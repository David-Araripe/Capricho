"""Utility functions to calculate fingerprints & identify molecules to be treated as identical"""

from functools import partial

import numpy as np
from job_tqdflex import ParallelApplier
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def _mol_to_morganFP(mol: Chem.Mol, radius: int = 2, nBits=2048, useChirality=False, **kwargs) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=nBits, includeChirality=useChirality, **kwargs
    )
    return generator.GetFingerprintAsNumPy(mol).reshape(1, -1)


def _mol_to_RDKitFP(mol: Chem.Mol, minPath=1, maxPath=7, nBits=2048, **kwargs) -> np.ndarray:
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=minPath, maxPath=maxPath, fpSize=nBits, **kwargs
    )
    return generator.GetFingerprintAsNumPy(mol).reshape(1, -1)


def smi_to_morganFP(smi, radius: int = 2, nBits=2048, useChirality=False, **kwargs) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.warning(f"Invalid SMILES detected: {smi}")
        return None
    return _mol_to_morganFP(mol, radius=radius, nBits=nBits, useChirality=useChirality, **kwargs)


def smi_to_RDKitFP(smi, minPath=1, maxPath=7, nBits=2048, **kwargs) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.warning(f"Invalid SMILES detected: {smi}")
        return None
    return _mol_to_RDKitFP(mol, minPath=minPath, maxPath=maxPath, nBits=nBits, **kwargs)


def smi_to_mixed_FP(smi, morgan_kwargs: dict, rdkit_kwargs: dict) -> np.ndarray:
    """Calculate both fingerprints after parsing a SMILES string only once."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.warning(f"Invalid SMILES detected: {smi}")
        return None
    morgan_fp = _mol_to_morganFP(mol, **morgan_kwargs)
    rdkit_fp = _mol_to_RDKitFP(mol, **rdkit_kwargs)
    return np.concatenate([morgan_fp, rdkit_fp], axis=1)


def calculate_mixed_FPs(
    smiles: list,
    n_jobs: int = 8,
    morgan_kwargs: dict = None,
    rdkit_kwargs: dict = None,
    return_stacked: bool = False,
    chunk_size: int = 50,
):
    """Outputs a mixed fingerprint used for compound identification. The motivation for this is
    that either of the fingerprints can fail to identify the same compound, but the combination
    is less prone to failure in this regard.

    Args:
        smiles (list): a list of smiles for which to compoute the mixed fingerprint
        n_jobs (int): number of jobs to run the fp calculation in parallel. Defaults to 1.
        morgan_kwargs (dict): keyword arguments for the morgan fingerprints. Defaults to None.
        rdkit_kwargs (dict): keyword arguments for the rdkit path fingerprints. Defaults to None.
        return_stacked (bool): if true, will return the stacked fingerprints instead of a list of
            numpy arrays. Defaults to False.
        chunk_size (int): chunk size to use for the parallel applier. Defaults to 50.

    Returns:
        np.ndarray: a mixed fingerprint for the input smiles
    """
    if morgan_kwargs is None:
        morgan_kwargs = {}
    if rdkit_kwargs is None:
        rdkit_kwargs = {}

    mixed_func = partial(
        smi_to_mixed_FP,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
    )
    applier = ParallelApplier(
        mixed_func,
        smiles,
        n_jobs=n_jobs,
        backend="loky",
        show_progress=True,
        chunk_size=chunk_size,
        custom_desc="Calculating mixed FPs",
    )
    mixed_fps = applier()

    if return_stacked:
        return np.concatenate(mixed_fps).shape
    return mixed_fps
