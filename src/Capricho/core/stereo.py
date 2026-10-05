"""Module for stereochemistry related functions"""


from rdkit import Chem

from ..logger import logger


def find_undefined_stereocenters(_input: Chem.Mol | str) -> list[int]:
    """
    Find atoms that are stereocenters but have undefined chirality.

    Args:
        mol: Input RDKit molecule

    Returns:
        List[int]: List of atom indices that are undefined stereocenters
    """
    if _input is None:
        return []

    if isinstance(_input, str):
        if _input.strip(".") == "":
            return []  # Yes, weird... But if SMILES has only salts, it becomes "." or ".." after removal
        mol = Chem.MolFromSmiles(_input)

    elif isinstance(_input, Chem.Mol):
        mol = _input

    if mol is None:
        return []

    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    except TypeError as e:
        raise TypeError(f"Something wront with input: {_input}") from e
    except RuntimeError as e:
        logger.error(f"Error finding chiral centers in molecule: {_input}. Error: {e}")
        return []

    undefined_stereo = []  # find atoms with undefined stereochemistry
    for atom_idx, _chirality in chiral_centers:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            undefined_stereo.append(atom_idx)

    return undefined_stereo
