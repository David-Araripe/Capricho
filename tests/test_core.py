import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Capricho.core import pandas_helper, smiles_utils, stats_make
from Capricho.core.pandas_helper import conflicting_duplicates


class TestPandasHelper(unittest.TestCase):
    def test_format_value(self):
        self.assertEqual(pandas_helper.format_value(3.14159), "3.14")
        self.assertEqual(pandas_helper.format_value(2000), "2000")
        self.assertEqual(pandas_helper.format_value("test"), "test")

    def test_aggr_val_series(self):
        series = pd.Series([1, 2.5, 3, 4.7])
        self.assertEqual(pandas_helper.aggr_val_series(series), "1.00|2.50|3.00|4.70")

    def test_get_mad(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(pandas_helper.get_mad(values), 1.0)
        self.assertTrue(np.isnan(pandas_helper.get_mad([1])))

    def test_apply_func_grpd(self):
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})
        grouped = df.groupby("A")
        result = pandas_helper.apply_func_grpd(grouped, pandas_helper.aggr_val_series, ["A"], "B")
        expected = pd.DataFrame({"A": [1, 2], "B": ["1|2", "3|4"]})
        pd.testing.assert_frame_equal(result, expected)

    def test_assign_stats(self):
        df = pd.DataFrame({"value": ["1|2|3", "4|5|6"]})
        result = pandas_helper.assign_stats(df, sep="|", value_col="value", use_geometric=False)
        self.assertIn("value_mean", result.columns)
        self.assertIn("value_std", result.columns)
        self.assertIn("value_median", result.columns)
        self.assertIn("value_counts", result.columns)

        self.assertEqual(result["value_mean"].iloc[0], 2.0)
        self.assertAlmostEqual(result["value_std"].iloc[0], 0.816497, 6)
        self.assertEqual(result["value_median"].iloc[0], 2.0)
        self.assertEqual(result["value_counts"].iloc[0], 3)

        self.assertEqual(result["value_mean"].iloc[1], 5.0)
        self.assertAlmostEqual(result["value_std"].iloc[1], 0.816497, 6)
        self.assertEqual(result["value_median"].iloc[1], 5.0)
        self.assertEqual(result["value_counts"].iloc[1], 3)

    def test_conflicting_duplicates(self):
        data = {
            "A": [
                1,
                1,
                1,
                2,
                2,
            ],
            "B": [
                "x",
                "x",
                "x",
                "y",
                "y",
            ],
            "C": [
                "p",
                "p",
                "p",
                "q",
                "q",
            ],
            "D": [
                10,  # Same A, B, C (e.g. mol identifiers) but different D (document)
                10,  # Same as above
                20,  # Same as above
                30,  # Same everything -> reported in same document; shouldn't drop
                30,  # Same as above
            ],
            "year": [
                2020,
                2020,
                2021,
                2021,
                2021,
            ],
        }
        df = pd.DataFrame(data).sort_values(by="year")
        mask = conflicting_duplicates(df, key_subset=["A", "B", "C"], diff_subset=["D"])
        expected_flags = [True, True, True, False, False]
        true_idxs = np.where(mask)[0]
        assert np.array_equal(
            true_idxs, np.array([0, 1, 2])
        ), "Expected indices with conflicting duplicates do not match."
        pd.testing.assert_series_equal(mask, pd.Series(expected_flags, index=df.index))


class TestFilterDroppingFlags(unittest.TestCase):
    def test_removes_flagged_rows(self):
        df = pd.DataFrame(
            {
                "data_dropping_comment": ["", "Unit Annotation Error", "Potential Duplicate", ""],
                "value": [1, 2, 3, 4],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, ["Unit Annotation Error"])
        self.assertEqual(len(result), 3)
        self.assertListEqual(result["value"].tolist(), [1, 3, 4])

    def test_removes_multiple_flags(self):
        df = pd.DataFrame(
            {
                "data_dropping_comment": ["", "Unit Annotation Error", "Potential Duplicate", ""],
                "value": [1, 2, 3, 4],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, ["Unit Annotation Error", "Potential Duplicate"])
        self.assertEqual(len(result), 2)
        self.assertListEqual(result["value"].tolist(), [1, 4])

    def test_handles_compound_flags(self):
        """Rows can have multiple flags separated by ' & '."""
        df = pd.DataFrame(
            {
                "data_dropping_comment": ["", "Flag A & Flag B", "Flag C", ""],
                "value": [1, 2, 3, 4],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, ["Flag A"])
        self.assertEqual(len(result), 3)
        self.assertListEqual(result["value"].tolist(), [1, 3, 4])

    def test_handles_nan_in_column(self):
        df = pd.DataFrame(
            {
                "data_dropping_comment": [np.nan, "Unit Annotation Error", np.nan, ""],
                "value": [1, 2, 3, 4],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, ["Unit Annotation Error"])
        self.assertEqual(len(result), 3)
        self.assertListEqual(result["value"].tolist(), [1, 3, 4])

    def test_custom_column_name(self):
        df = pd.DataFrame(
            {
                "dropping_comment": ["", "Bad Flag", ""],
                "value": [1, 2, 3],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, ["Bad Flag"], column="dropping_comment")
        self.assertEqual(len(result), 2)
        self.assertListEqual(result["value"].tolist(), [1, 3])

    def test_empty_flags_returns_unchanged(self):
        df = pd.DataFrame(
            {
                "data_dropping_comment": ["", "some flag", ""],
                "value": [1, 2, 3],
            }
        )
        result = pandas_helper.filter_dropping_flags(df, [])
        pd.testing.assert_frame_equal(result, df)

    def test_missing_column_returns_unchanged(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = pandas_helper.filter_dropping_flags(df, ["some flag"])
        pd.testing.assert_frame_equal(result, df)


class TestSmilesUtils(unittest.TestCase):
    # Canonical (RDKit) SMILES of real drug parents. These are the standardized
    # structures that survive salt/solvent removal; each is used below combined
    # with a real (or, where noted, closest-analog) counter-ion / solvent.
    DICLOFENAC = "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"  # CHEMBL139
    DIPHENHYDRAMINE = "CN(C)CCOC(c1ccccc1)c1ccccc1"  # CHEMBL657
    NAPROXEN = "COc1ccc2cc(C(C)C(=O)O)ccc2c1"  # CHEMBL154
    LOSARTAN = "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"  # CHEMBL226
    RIZATRIPTAN = "CN(C)CCc1c[nH]c2ccc(Cn3ccnc3)cc12"  # CHEMBL1657
    ATORVASTATIN = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"  # CHEMBL1487
    ESOMEPRAZOLE = "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"  # CHEMBL1200983
    DEXTROMETHORPHAN = "COc1ccc2c(c1)CCN(C)C1Cc3ccc(OC)cc3C21"  # CHEMBL52440
    PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"  # CHEMBL112
    NEOSTIGMINE = "CN(C)C(=O)Oc1cccc([N+](C)(C)C)c1"  # CHEMBL54126 (quaternary cation)
    UNDECYLENATE = "C=CCCCCCCCCC(=O)[O-]"  # undecylenic acid anion (zinc undecylenate)
    TETRAETHYLAMMONIUM = "CC[N+](CC)(CC)CC"  # CHEMBL86407 (quaternary cation)

    def test_clean_mixtures_strips_every_salt_and_solvent(self):
        """Every salt/solvent branch of MIXTURE_REGEX is removed, leaving the parent.

        Each input is a real drug parent joined with the counter-ion/solvent as the
        regex targets it. ``description`` names the salt list entry and the real
        example (or the closest analog used when no real ChEMBL salt exists).
        clean_mixtures is a string function that runs AFTER chembl_structure_pipeline
        standardization, so inputs use the standardized/canonical fragment spellings.
        """
        cases = [
            # --- monovalent metal cations, bracketed canonical form (as ChEMBL deposits them) ---
            (self.NAPROXEN + ".[Na+]", self.NAPROXEN, "Na+ | naproxen sodium (CHEMBL1697742)"),
            (self.LOSARTAN + ".[K+]", self.LOSARTAN, "K+ | losartan potassium (CHEMBL1237)"),
            (self.RIZATRIPTAN + ".[Li+]", self.RIZATRIPTAN, "Li+ | closest analog: rizatriptan lithium carboxylate salt"),
            # --- bare (non-bracketed) cation tokens the regex also anchors ---
            ("CC(=O)[O-].Na+", "CC(=O)[O-]", "Na (bare token variant) | sodium acetate"),
            # --- hydrohalide salts: ChEMBL stores the neutral acid fragment (Cl == HCl) ---
            (self.DIPHENHYDRAMINE + ".Cl", self.DIPHENHYDRAMINE, "Cl- (as HCl) | diphenhydramine hydrochloride (CHEMBL1200662)"),
            (self.DIPHENHYDRAMINE + ".[Cl-]", self.DIPHENHYDRAMINE, "Cl- (bracketed chloride) | diphenhydramine HCl standardized"),
            (self.DEXTROMETHORPHAN + ".Br", self.DEXTROMETHORPHAN, "Br- (as HBr) | dextromethorphan hydrobromide (CHEMBL1200736)"),
            (self.NEOSTIGMINE + ".[Br-]", self.NEOSTIGMINE, "Br- (bracketed) | neostigmine bromide (CHEMBL1201231)"),
            (self.PARACETAMOL + ".F", self.PARACETAMOL, "F- (bare token, as HF) | closest analog: fluoride of an organic base"),
            (self.NEOSTIGMINE + ".I", self.NEOSTIGMINE, "I- (bare token, as HI) | quaternary ammonium iodide"),
            (self.NEOSTIGMINE + ".[I-]", self.NEOSTIGMINE, "I- (bracketed) | neostigmine/quaternary iodide"),
            # --- divalent metal cations. Regex only anchors the bare 'Xx++' spelling for
            #     Ca/Mg (no bracketed branch exists), so those use that form; Zn also has a
            #     bracketed branch matching ChEMBL's canonical [Zn+2]. ---
            (self.ATORVASTATIN + ".[Ca+2]", self.ATORVASTATIN, "Ca2+ (canonical [Ca+2]) | atorvastatin calcium (CHEMBL1487)"),
            (self.ESOMEPRAZOLE + ".[Mg+2]", self.ESOMEPRAZOLE, "Mg2+ (canonical [Mg+2]) | esomeprazole magnesium (CHEMBL1213492)"),
            (self.UNDECYLENATE + ".[Zn+2]", self.UNDECYLENATE, "Zn2+ (bracketed [Zn+2]) | zinc undecylenate (CHEMBL1200967)"),
            # --- hydroxide (quaternary ammonium hydroxide) ---
            (self.TETRAETHYLAMMONIUM + ".OH-", self.TETRAETHYLAMMONIUM, "OH- | tetraethylammonium hydroxide (CHEMBL86407 cation)"),
            # --- carboxylate counter-ions ---
            (self.RIZATRIPTAN + ".O=C([O-])c1ccccc1", self.RIZATRIPTAN, "benzoate (canonical) | rizatriptan benzoate (CHEMBL1201090)"),
            (self.PARACETAMOL + ".CCCC(=O)[O-]", self.PARACETAMOL, "butyrate | closest analog: butyrate salt of a base"),
            (self.PARACETAMOL + ".CCCCC(=O)[O-]", self.PARACETAMOL, "pentanoate (valerate) | closest analog: valerate salt of a base"),
            # --- inorganic / solvent / misc. residual fragments ---
            (self.PARACETAMOL + ".[O-][Cl+3]([O-])([O-])[O-]", self.PARACETAMOL, "perchlorate | closest analog: perchlorate of an organic base"),
            (self.PARACETAMOL + ".c1ccncc1", self.PARACETAMOL, "pyridine solvate (canonical)"),
            (self.PARACETAMOL + ".CN(C)C=O", self.PARACETAMOL, "N,N-dimethylformamide solvate (canonical)"),
            (self.PARACETAMOL + ".[N]=O", self.PARACETAMOL, "nitric oxide (canonical) | contrived, no realistic ChEMBL counter-ion example"),
            (self.NEOSTIGMINE + ".c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", self.NEOSTIGMINE, "tetraphenylborate | closest analog: quaternary ammonium tetraphenylborate"),
            (self.PARACETAMOL + ".O", self.PARACETAMOL, "water (hydrate) | paracetamol hemihydrate"),
            (self.PARACETAMOL + ".N", self.PARACETAMOL, "ammonia | ammonia adduct/solvate"),
        ]
        for input_smiles, expected_parent, description in cases:
            with self.subTest(salt=description):
                self.assertEqual(smiles_utils.clean_mixtures(input_smiles), expected_parent)

    def test_clean_mixtures_returns_dot_when_all_fragments_are_salts(self):
        """If every fragment matches the regex, clean_mixtures returns '.' (drop marker)."""
        self.assertEqual(smiles_utils.clean_mixtures("[Na+].[Cl-]"), ".")  # inorganic sodium chloride
        self.assertEqual(smiles_utils.clean_mixtures("O.[Na+]"), ".")  # water + sodium
        self.assertEqual(smiles_utils.clean_mixtures("N.O"), ".")  # ammonia + water

    def test_clean_mixtures_preserves_non_salt_fragments(self):
        """Fragments not in the salt list are left untouched (negative controls).

        clean_mixtures sorts and de-duplicates fragments via np.unique, so multi-fragment
        outputs come back in alphabetical order.
        """
        # ethanol is a solvent NOT in the regex list -> the whole mixture is preserved
        self.assertEqual(smiles_utils.clean_mixtures("CC.CCO"), "CC.CCO")
        # ethylene glycol co-former is not listed -> preserved alongside the drug
        self.assertEqual(
            smiles_utils.clean_mixtures(self.PARACETAMOL + ".OCCO"),
            "CC(=O)Nc1ccc(O)cc1.OCCO",
        )
        # a single-fragment molecule is returned unchanged
        self.assertEqual(smiles_utils.clean_mixtures(self.DICLOFENAC), self.DICLOFENAC)
        # a genuine two-drug combination has no salt fragment -> both kept (sorted)
        self.assertEqual(
            smiles_utils.clean_mixtures(self.NAPROXEN + "." + self.PARACETAMOL),
            "CC(=O)Nc1ccc(O)cc1.COc1ccc2cc(C(C)C(=O)O)ccc2c1",
        )


class TestStatsMake(unittest.TestCase):
    def setUp(self):
        self.testroot = Path(__file__).parent
        self.not_aggr_df = pd.read_csv(self.testroot / "resources/ADORA3_data_not_aggregated.csv")
        self.aggr_df = pd.read_csv(self.testroot / "resources/ADORA3_data.csv")

    # note - test_process_repeat_mols not tested here but indirectly on `test_workflow.py`

    def test_repeated_indices_from_array_series(self):
        series = pd.Series([np.array([1, 2]), np.array([1, 2]), np.array([3, 4])])
        result = stats_make.repeated_indices_from_array_series(series)
        self.assertEqual(result, [[0, 1]])

    def test_process_repeat_mols_with_different_standard_relations(self):
        """Test that compounds with different standard_relation values are not aggregated together"""
        # Create test data with same compound (fingerprint) but different standard_relations
        test_df = pd.DataFrame(
            {
                "standard_smiles": ["CCO", "CCO", "CCO"],
                "pchembl_value": [6.5, 7.0, 8.0],
                "target_chembl_id": ["CHEMBL123", "CHEMBL123", "CHEMBL123"],
                "mutation": ["None", "None", "None"],
                "standard_relation": ["=", "<", "="],  # Different relations
                "molecule_chembl_id": ["MOL1", "MOL1", "MOL1"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY3"],
                "assay_description": ["Test assay 1", "Test assay 2", "Test assay 3"],
                "activity_id": [1, 2, 3],
                "standard_type": ["IC50", "IC50", "IC50"],
                "assay_type": ["B", "B", "B"],
                "confidence_score": [9, 9, 9],
                "target_organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
                "assay_tissue": ["None", "None", "None"],
                "assay_cell_type": ["None", "None", "None"],
                "relationship_type": ["D", "D", "D"],
                "max_phase": ["None", "None", "None"],
                "oral": ["None", "None", "None"],
                "prodrug": ["None", "None", "None"],
                "withdrawn_flag": ["None", "None", "None"],
                "document_chembl_id": ["DOC1", "DOC2", "DOC3"],
                "canonical_smiles": ["CCO", "CCO", "CCO"],
            }
        )

        # All rows have same fingerprint, should be identified as repeats
        repeat_idxs = [[0, 1, 2]]

        # Process the repeats
        result_df = stats_make.process_repeat_mols(
            test_df,
            repeat_idxs,
            solve_strat="keep",
            extra_id_cols=[],
            aggregate_mutants=False,
            chirality=False,
        )

        # Check that we have 2 separate rows (not 1) because standard_relation differs
        # Rows 0 and 2 have '=' so should aggregate together
        # Row 1 has '<' so should remain separate
        equal_rows = result_df[result_df["standard_relation"] == "="]
        less_than_rows = result_df[result_df["standard_relation"] == "<"]

        self.assertEqual(len(equal_rows), 1, "Should have 1 row with standard_relation='='")
        self.assertEqual(len(less_than_rows), 1, "Should have 1 row with standard_relation='<'")

        # The '=' row should have aggregated pchembl values from rows 0 and 2
        # Since pchembl values use geometric mean by default
        equal_row = equal_rows.iloc[0]
        self.assertEqual(equal_row["pchembl_value_counts"], 2)
        from scipy.stats.mstats import gmean

        expected_mean = gmean([6.5, 8.0])
        self.assertAlmostEqual(equal_row["pchembl_value_mean"], expected_mean)

        # The '<' row should have a single pchembl value from row 1
        less_than_row = less_than_rows.iloc[0]
        self.assertEqual(less_than_row["pchembl_value"], "7.00")  # format_value converts to string

    def test_process_repeat_mols_with_custom_value_col(self):
        """Test that process_repeat_mols works with a custom value column (e.g., standard_value)"""
        # Create test data with standard_value instead of pchembl_value
        test_df = pd.DataFrame(
            {
                "standard_smiles": ["CCO", "CCO", "CCC"],
                "standard_value": [50.0, 75.0, 100.0],  # Use standard_value, not pchembl_value
                "standard_units": ["%", "%", "%"],
                "target_chembl_id": ["CHEMBL123", "CHEMBL123", "CHEMBL123"],
                "mutation": ["None", "None", "None"],
                "standard_relation": ["=", "=", "="],
                "molecule_chembl_id": ["MOL1", "MOL1", "MOL2"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY3"],
                "assay_description": ["Test assay 1", "Test assay 2", "Test assay 3"],
                "activity_id": [1, 2, 3],
                "standard_type": ["Inhibition", "Inhibition", "Inhibition"],
                "assay_type": ["A", "A", "A"],
                "confidence_score": [9, 9, 9],
                "target_organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
                "assay_tissue": ["None", "None", "None"],
                "assay_cell_type": ["None", "None", "None"],
                "relationship_type": ["D", "D", "D"],
                "max_phase": ["None", "None", "None"],
                "oral": ["None", "None", "None"],
                "prodrug": ["None", "None", "None"],
                "withdrawn_flag": ["None", "None", "None"],
                "document_chembl_id": ["DOC1", "DOC2", "DOC3"],
                "canonical_smiles": ["CCO", "CCO", "CCC"],
            }
        )

        # Rows 0 and 1 have same SMILES (repeat), row 2 is different
        repeat_idxs = [[0, 1]]

        # Process the repeats with custom value_col
        result_df = stats_make.process_repeat_mols(
            test_df,
            repeat_idxs,
            solve_strat="keep",
            extra_id_cols=["standard_units"],  # Group by units to prevent mixing
            aggregate_mutants=False,
            chirality=False,
            value_col="standard_value",  # Use standard_value instead of pchembl_value
        )

        # Check that we have 2 rows (1 aggregated from CCO, 1 single from CCC)
        self.assertEqual(len(result_df), 2)

        # Check that standard_value statistics columns exist (not pchembl_value)
        self.assertIn("standard_value_mean", result_df.columns)
        self.assertIn("standard_value_std", result_df.columns)
        self.assertIn("standard_value_median", result_df.columns)
        self.assertIn("standard_value_counts", result_df.columns)

        # Check that pchembl_value columns do NOT exist
        self.assertNotIn("pchembl_value_mean", result_df.columns)
        self.assertNotIn("pchembl_value_std", result_df.columns)

        # Check the aggregated row (CCO) has correct statistics
        cco_rows = result_df[result_df["smiles"].str.contains("CCO", na=False)]
        if len(cco_rows) > 0:
            cco_row = cco_rows.iloc[0]
            self.assertEqual(cco_row["standard_value_counts"], 2)
            self.assertAlmostEqual(cco_row["standard_value_mean"], (50.0 + 75.0) / 2)


if __name__ == "__main__":
    unittest.main()
