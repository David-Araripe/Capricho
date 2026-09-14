"""Utility functions for further processing the standardized smiles by removing mixtures and salts."""

import re

import numpy as np

MIXTURE_REGEX = re.compile(
    r"^("  # start of the string
    r"Na\+?|Cl\-?|Br\-?|K\+?|F\-?|I\-?|Ca\+{2}?|Mg\+{2}?|Zn\+{2}?|OH\-?|"  # Salts (bare)
    r"\[Na\+\]?|\[K\+\]?|\[Cl-\]?|\[Br-\]?|\[I-\]?|"  # Salts (bracketed, monovalent)
    r"\[Zn\+2\]?|\[Ca\+2\]?|\[Mg\+2\]?|\[Li\+\]?|"  # Salts (bracketed, divalent + Li)
    r"CCCC\(=O\)\[O-\]|"  # Butyrate
    r"CCCCC\(=O\)\[O-\]|"  # Pentanoate
    r"O=C\(\[O-\]\)c1ccccc1|"  # Benzoate (RDKit-canonical)
    r"\[O-\]\[Cl\+3\]\(\[O-\]\)\(\[O-\]\)\[O-\]|"  # Perchlorate [O-][Cl+3]([O-])([O-])[O-]
    r"c1ccncc1|"  # Pyridine (RDKit-canonical)
    r"CN\(C\)C=O|"  # N,N-Dimethylformamide (RDKit-canonical)
    r"\[N\]=O|"  # Nitric oxide (RDKit-canonical)
    r"c1ccc\(\[B-\]\(c2ccccc2\)\(c2ccccc2\)c2ccccc2\)cc1|"  # Tetraphenylborate
    r"^O$|^N$"  # Match full strings for O and N
    r")$"  # end of the string
)


def clean_mixtures(smi):
    """Removes mixtures/salts from a SMILES tring that aren't captured by the
    `chembl_structure_pipeline` using MIXTURE_REGEX.

    Args:
        smi: smiles string

    Returns:
        str: smiles string without mixture.
    """
    smiles = np.unique(smi.split(".")).tolist()
    if len(smiles) == 1:
        return smiles[0]
    else:
        no_salt_smiles = [MIXTURE_REGEX.sub("", smi) for smi in smiles]
        no_salt_smiles = [smi for smi in no_salt_smiles if smi]
        if len(no_salt_smiles) == 0:
            return "."  # in this case wil drop the smiles
        return ".".join(no_salt_smiles)
