"""Test function to check for undefined stereocenters in a molecule."""

import unittest

from rdkit import Chem

from Capricho.core.stereo import find_undefined_stereocenters


def has_undefined_stereocenter(input: Chem.Mol | str) -> bool:
    """
    Check if molecule has any stereocenters with undefined chirality.

    Args:
        input: either an RDKit molecule or a SMILES string

    Returns:
        bool: True if molecule has undefined stereocenters, False otherwise
    """
    return len(find_undefined_stereocenters(input)) > 0


class TestStereocenters(unittest.TestCase):
    def test_no_stereocenter(self):
        # Ethane has no stereocenters
        smiles = "CC"
        self.assertFalse(has_undefined_stereocenter(smiles))
        self.assertEqual(find_undefined_stereocenters(smiles), [])

    def test_defined_stereocenter(self):
        # 2-butanol with defined stereochemistry
        smiles = "CC[C@H](C)O"
        self.assertFalse(has_undefined_stereocenter(smiles))
        self.assertEqual(find_undefined_stereocenters(smiles), [])

    def test_undefined_stereocenter(self):
        # 2-butanol without defined stereochemistry
        smiles = "CCC(C)O"
        self.assertTrue(has_undefined_stereocenter(smiles))
        self.assertEqual(len(find_undefined_stereocenters(smiles)), 1)

    def test_multiple_stereocenters(self):
        # 2,3-butanediol with one defined and one undefined stereocenter
        smiles = "C[C@H](O)C(C)O"
        self.assertTrue(has_undefined_stereocenter(smiles))
        undefined = find_undefined_stereocenters(smiles)
        self.assertEqual(len(undefined), 1)

    def test_all_undefined_stereocenters(self):
        # 2,3-butanediol with both stereocenters undefined
        smiles = "CC(O)C(C)O"
        self.assertTrue(has_undefined_stereocenter(smiles))
        undefined = find_undefined_stereocenters(smiles)
        self.assertEqual(len(undefined), 2)

    def test_none_input(self):
        self.assertFalse(has_undefined_stereocenter(None))
        self.assertEqual(find_undefined_stereocenters(None), [])

    def test_aromatic_compound(self):
        # Phenethyl alcohol has no stereocenters
        smiles = "OCCc1ccccc1"
        self.assertFalse(has_undefined_stereocenter(smiles))
        self.assertEqual(find_undefined_stereocenters(smiles), [])

    def test_complex_molecule(self):
        # A more complex molecule with multiple stereocenters
        smiles = "CC(C)[C@H](C(=O)N[C@@H](Cc1ccccc1)C(C)C)NC(=O)C(C)C"
        self.assertFalse(has_undefined_stereocenter(smiles))
        self.assertEqual(find_undefined_stereocenters(smiles), [])
