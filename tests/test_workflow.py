"""Module to test the workflow functions within /cli. To reproduce the same data in the
`resources/` directory, check the check `ADORA3_data_recipe.json`.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Capricho.cli.chembl_data_pipeline import aggregate_data
from Capricho.cli.prepare import clean_data, prepare_multitask_data
from Capricho.core.pandas_helper import assign_shared_identifier_groups


class TestFetchFromChEMBL(unittest.TestCase):
    def setUp(self):
        self.testroot = Path(__file__).parent
        self.aggr_df = pd.read_csv(self.testroot / "resources/ADORA3_data.csv")
        self.aggr_df = assign_shared_identifier_groups(self.aggr_df)
        self.not_aggr_df = pd.read_csv(self.testroot / "resources/ADORA3_data_not_aggregated.csv")

    def test_aggregate_data(self):
        aggr_df = aggregate_data(
            self.not_aggr_df,
            chirality=True,
            metadata_cols=[],
            output_path=None,
            compound_equality="connectivity",
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = Path(tmpdirname) / "ADORA3_data.csv"
            aggr_df.to_csv(outfile, index=False)
            aggr_df = pd.read_csv(outfile)

        aggr_df.sort_values(["activity_id"], inplace=True, ignore_index=True)
        self.aggr_df.sort_values(["activity_id"], inplace=True, ignore_index=True)

        # assert if the columns are the same
        np.testing.assert_array_equal(self.aggr_df.columns, aggr_df.columns)

        # Compare if the bioactivity data is the same
        np.testing.assert_array_equal(aggr_df.pchembl_value.values, self.aggr_df.pchembl_value.values)
        np.testing.assert_allclose(
            aggr_df.pchembl_value_median.values, self.aggr_df.pchembl_value_median.values
        )
        np.testing.assert_allclose(aggr_df.pchembl_value_mean.values, self.aggr_df.pchembl_value_mean.values)
        np.testing.assert_array_equal(
            aggr_df.molecule_chembl_id.values, self.aggr_df.molecule_chembl_id.values
        )

        # Compare if the assays are the same
        np.testing.assert_array_equal(aggr_df.assay_chembl_id.values, self.aggr_df.assay_chembl_id.values)
        np.testing.assert_array_equal(aggr_df.activity_id.values, self.aggr_df.activity_id.values)

    def test_censored_measurement_aggregation(self):
        """Test that censored measurements are only aggregated if they have identical relation AND value."""
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4, 5, 6],
                "molecule_chembl_id": ["CHEMBL1"] * 6,
                "target_chembl_id": ["TARGET1"] * 6,
                "standard_smiles": ["CCO"] * 6,  # Same compound
                "canonical_smiles": ["CCO"] * 6,
                "pchembl_value": [6.0, 6.0, 6.0, 5.0, 7.0, 7.0],
                "standard_relation": ["<", "<", ">", "<", "=", "="],
                "mutation": ["WT"] * 6,
                "assay_chembl_id": ["ASSAY1"] * 6,
                "standard_type": ["IC50"] * 6,
                "assay_description": ["Test assay"] * 6,
                "assay_type": ["B"] * 6,
                "confidence_score": [9] * 6,
                "target_organism": ["Homo sapiens"] * 6,
                "document_chembl_id": ["DOC1"] * 6,
                "assay_tissue": [""] * 6,
                "assay_cell_type": [""] * 6,
                "relationship_type": ["D"] * 6,
                "max_phase": [1] * 6,
                "oral": [False] * 6,
                "prodrug": [False] * 6,
                "withdrawn_flag": [False] * 6,
                "doc_type": ["PUBLICATION"] * 6,
                "doi": ["10.1234/test"] * 6,
                "journal": ["Test Journal"] * 6,
                "year": [2020] * 6,
                "chembl_release": [30] * 6,
                "data_dropping_comment": [""] * 6,
                "data_processing_comment": [""] * 6,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=[],
            output_path=None,
            compound_equality="connectivity",
        )

        # Expected behavior:
        # Row 1,2: pchembl=6.0, relation='<' → should aggregate (same value AND relation)
        # Row 3: pchembl=6.0, relation='>' → separate row (different relation)
        # Row 4: pchembl=5.0, relation='<' → separate row (different value)
        # Row 5,6: pchembl=7.0, relation='=' → should aggregate (both are discrete measurements)

        # Check that we get 4 rows in the output
        self.assertEqual(len(aggr_df), 4, f"Expected 4 rows, got {len(aggr_df)}")

        # Check that rows with '<' and pchembl=6.0 were aggregated
        censored_lt_6 = aggr_df[
            (aggr_df["standard_relation"].str.contains("<", regex=False, na=False))
            & (aggr_df["pchembl_value_mean"].round(1) == 6.0)
        ]
        self.assertEqual(len(censored_lt_6), 1, "Expected 1 aggregated row for '<6.0' measurements")
        self.assertEqual(
            censored_lt_6.iloc[0]["pchembl_value_counts"], 2, "Expected 2 measurements aggregated for '<6.0'"
        )
        # For censored measurements, mean/median should be the unique value (not calculated)
        self.assertEqual(
            censored_lt_6.iloc[0]["pchembl_value_mean"], 6.0, "Expected mean to be 6.0 (unique value)"
        )
        self.assertEqual(
            censored_lt_6.iloc[0]["pchembl_value_median"], 6.0, "Expected median to be 6.0 (unique value)"
        )
        self.assertTrue(
            pd.isna(censored_lt_6.iloc[0]["pchembl_value_std"]), "Expected std to be NaN for censored data"
        )

        # Check that row with '>' and pchembl=6.0 is separate
        censored_gt_6 = aggr_df[
            (aggr_df["standard_relation"].str.contains(">", regex=False, na=False))
            & (aggr_df["pchembl_value_mean"].round(1) == 6.0)
        ]
        self.assertEqual(len(censored_gt_6), 1, "Expected 1 row for '>6.0' measurement")
        self.assertEqual(
            censored_gt_6.iloc[0]["pchembl_value_counts"], 1, "Expected 1 count for single measurement '>6.0'"
        )

        # Check that row with '<' and pchembl=5.0 is separate
        censored_lt_5 = aggr_df[
            (aggr_df["standard_relation"].str.contains("<", regex=False, na=False))
            & (aggr_df["pchembl_value_mean"].round(1) == 5.0)
        ]
        self.assertEqual(len(censored_lt_5), 1, "Expected 1 row for '<5.0' measurement")
        self.assertEqual(
            censored_lt_5.iloc[0]["pchembl_value_counts"], 1, "Expected 1 count for single measurement '<5.0'"
        )

        # Check that discrete measurements with '=' were aggregated normally
        discrete_7 = aggr_df[
            (aggr_df["standard_relation"] == "=") & (aggr_df["pchembl_value_mean"].round(1) == 7.0)
        ]
        self.assertEqual(len(discrete_7), 1, "Expected 1 aggregated row for '=7.0' measurements")
        self.assertEqual(
            discrete_7.iloc[0]["pchembl_value_counts"], 2, "Expected 2 measurements aggregated for '=7.0'"
        )

    def test_aggregate_data_with_standard_value(self):
        """Test that aggregate_data works with standard_value instead of pchembl_value."""
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4],
                "molecule_chembl_id": ["CHEMBL1"] * 4,
                "target_chembl_id": ["TARGET1"] * 4,
                "standard_smiles": ["CCO", "CCO", "CCC", "CCC"],  # 2 compounds
                "canonical_smiles": ["CCO", "CCO", "CCC", "CCC"],
                "standard_value": [50.0, 75.0, 100.0, 80.0],  # % inhibition values
                "standard_units": ["%", "%", "%", "%"],
                "standard_relation": ["=", "=", "=", "="],
                "mutation": ["WT"] * 4,
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY1", "ASSAY2"],
                "standard_type": ["Inhibition"] * 4,
                "assay_description": ["Test assay"] * 4,
                "assay_type": ["A"] * 4,
                "confidence_score": [9] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1"] * 4,
                "assay_tissue": [""] * 4,
                "assay_cell_type": [""] * 4,
                "relationship_type": ["D"] * 4,
                "max_phase": [1] * 4,
                "oral": [False] * 4,
                "prodrug": [False] * 4,
                "withdrawn_flag": [False] * 4,
                "doc_type": ["PUBLICATION"] * 4,
                "doi": ["10.1234/test"] * 4,
                "journal": ["Test Journal"] * 4,
                "year": [2020] * 4,
                "chembl_release": [30] * 4,
                "data_dropping_comment": [""] * 4,
                "data_processing_comment": [""] * 4,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=["standard_units"],  # Group by units to prevent mixing
            output_path=None,
            compound_equality="connectivity",
            value_col="standard_value",  # Use standard_value instead of pchembl_value
        )

        # Check that we have 2 rows (2 compounds aggregated)
        self.assertEqual(len(aggr_df), 2)

        # Check that standard_value statistics columns exist
        self.assertIn("standard_value_mean", aggr_df.columns)
        self.assertIn("standard_value_std", aggr_df.columns)
        self.assertIn("standard_value_median", aggr_df.columns)
        self.assertIn("standard_value_counts", aggr_df.columns)

        # Check that pchembl_value columns do NOT exist
        self.assertNotIn("pchembl_value_mean", aggr_df.columns)
        self.assertNotIn("pchembl_value_std", aggr_df.columns)

        # Check the aggregated values for CCO compound
        cco_rows = aggr_df[aggr_df["smiles"].str.contains("CCO", na=False)]
        if len(cco_rows) > 0:
            cco_row = cco_rows.iloc[0]
            self.assertEqual(cco_row["standard_value_counts"], 2)
            self.assertAlmostEqual(cco_row["standard_value_mean"], (50.0 + 75.0) / 2)

    def test_aggregate_standard_value_without_pchembl_filtering(self):
        """When aggregating on standard_value, datapoints without pchembl should not be filtered.

        This test ensures that when aggregating on standard_value (e.g., % inhibition),
        rows with NaN pchembl_value are NOT filtered out or flagged for removal.
        """
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4],
                "molecule_chembl_id": ["CHEMBL1"] * 4,
                "target_chembl_id": ["TARGET1"] * 4,
                "standard_smiles": ["CCO", "CCO", "CCC", "CCC"],  # 2 compounds
                "canonical_smiles": ["CCO", "CCO", "CCC", "CCC"],
                "standard_value": [50.0, 75.0, 100.0, 80.0],  # % inhibition values
                "standard_units": ["%", "%", "%", "%"],
                "pchembl_value": [np.nan, np.nan, np.nan, np.nan],  # No pchembl for % units
                "standard_relation": ["=", "=", "=", "="],
                "mutation": ["WT"] * 4,
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY1", "ASSAY2"],
                "standard_type": ["Inhibition"] * 4,
                "assay_description": ["Test assay"] * 4,
                "assay_type": ["A"] * 4,
                "confidence_score": [9] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1"] * 4,
                "assay_tissue": [""] * 4,
                "assay_cell_type": [""] * 4,
                "relationship_type": ["D"] * 4,
                "max_phase": [1] * 4,
                "oral": [False] * 4,
                "prodrug": [False] * 4,
                "withdrawn_flag": [False] * 4,
                "doc_type": ["PUBLICATION"] * 4,
                "doi": ["10.1234/test"] * 4,
                "journal": ["Test Journal"] * 4,
                "year": [2020] * 4,
                "chembl_release": [30] * 4,
                "data_dropping_comment": [""] * 4,
                "data_processing_comment": [""] * 4,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=["standard_units"],
            output_path=None,
            compound_equality="connectivity",
            value_col="standard_value",  # Aggregating on standard_value, not pchembl_value
        )

        # Check that we have 2 rows (both compounds should be included, not filtered out)
        self.assertEqual(len(aggr_df), 2, f"Expected 2 compounds, got {len(aggr_df)}")

        # Check that data_dropping_comment does NOT contain "Missing pChEMBL" or similar
        for _, row in aggr_df.iterrows():
            comment = str(row.get("data_dropping_comment", ""))
            self.assertNotIn(
                "pchembl",
                comment.lower(),
                f"Row should not be flagged for missing pchembl when aggregating on standard_value. "
                f"Got comment: {comment}",
            )

        # Check that aggregation actually worked (2 measurements per compound)
        self.assertEqual(
            aggr_df["standard_value_counts"].sum(), 4, "Total count of all measurements should be 4"
        )

    def test_prepare_multitask_data(self):
        """Test that prepare command creates correct activity matrix."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "some_flag", "", ""],
            }
        )

        matrix = prepare_multitask_data(
            df=test_data,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="connectivity",
            smiles_col="smiles",
        )

        # Verify shape: 2 compounds x 2 targets
        self.assertEqual(len(matrix), 2, "Expected 2 compounds in the activity matrix")
        self.assertIn("TARGET1", matrix.columns, "Expected TARGET1 column")
        self.assertIn("TARGET2", matrix.columns, "Expected TARGET2 column")
        self.assertIn("smiles", matrix.columns, "Expected smiles column")

        # Verify values
        conn1_row = matrix[matrix.index == "CONN1"]
        self.assertEqual(len(conn1_row), 1)
        self.assertAlmostEqual(conn1_row["TARGET1"].iloc[0], 6.5)
        self.assertAlmostEqual(conn1_row["TARGET2"].iloc[0], 7.0)
        self.assertEqual(conn1_row["smiles"].iloc[0], "CCO")

        conn2_row = matrix[matrix.index == "CONN2"]
        self.assertEqual(len(conn2_row), 1)
        self.assertAlmostEqual(conn2_row["TARGET1"].iloc[0], 5.5)
        self.assertAlmostEqual(conn2_row["TARGET2"].iloc[0], 8.0)
        self.assertEqual(conn2_row["smiles"].iloc[0], "CCC")

    def test_prepare_multitask_data_with_flags(self):
        """Test that clean_data + prepare_multitask_data correctly filters out flagged data."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "flag_to_remove", "", "another_flag"],
            }
        )

        cleaned = clean_data(test_data, drop_flags=["flag_to_remove", "another_flag"])
        matrix = prepare_multitask_data(
            df=cleaned,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="connectivity",
            smiles_col="smiles",
        )

        # After filtering, we should only have CONN1->TARGET1 and CONN2->TARGET1
        self.assertEqual(len(matrix), 2, "Expected 2 compounds after filtering")

        # Only TARGET1 should exist as a column (TARGET2 was completely filtered out)
        self.assertIn("TARGET1", matrix.columns, "Expected TARGET1 column")
        self.assertNotIn("TARGET2", matrix.columns, "TARGET2 should not exist after filtering")
        self.assertIn("smiles", matrix.columns, "Expected smiles column")

        # Verify the values for TARGET1
        conn1_row = matrix[matrix.index == "CONN1"]
        self.assertAlmostEqual(conn1_row["TARGET1"].iloc[0], 6.5)

        conn2_row = matrix[matrix.index == "CONN2"]
        self.assertAlmostEqual(conn2_row["TARGET1"].iloc[0], 5.5)

    def test_prepare_multitask_data_warns_on_duplicates(self):
        """Test that prepare warns when duplicates exist (e.g., same compound-target but different id_columns)."""
        # Simulate data aggregated with --id-columns assay_tissue
        # Same compound-target pair but different tissues = duplicates when pivoting on target alone
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "assay_tissue": ["Liver", "Kidney", "Liver"],  # Different tissues
                "pchembl_value_mean": [6.5, 7.0, 5.5],
                "data_dropping_comment": ["", "", ""],
            }
        )

        # Capture log output
        import io
        import logging

        from Capricho.logger import logger

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger.add(handler, format="{message}", level="WARNING")

        matrix = prepare_multitask_data(
            df=test_data,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="connectivity",
            smiles_col="smiles",
        )

        # Remove our test handler
        logger.remove()

        # The function should still work (take first value), but there should be a warning
        # CONN1 has 2 values for TARGET1 (Liver=6.5, Kidney=7.0), only one will be kept
        self.assertEqual(len(matrix), 2)  # 2 compounds

        # Check that only one value was kept for CONN1-TARGET1
        conn1_target1 = matrix.loc["CONN1", "TARGET1"]
        self.assertIn(conn1_target1, [6.5, 7.0])  # Either value is acceptable

    def test_prepare_multitask_data_with_id_columns(self):
        """Test that id_columns creates composite task identifiers."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "assay_tissue": ["Liver", "Kidney", "Liver", "Kidney"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "", "", ""],
            }
        )

        matrix = prepare_multitask_data(
            df=test_data,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="connectivity",
            smiles_col="smiles",
            id_columns=["assay_tissue"],
        )

        # With id_columns, we should have 4 unique tasks: TARGET1-Liver, TARGET1-Kidney
        self.assertEqual(len(matrix), 2)  # 2 compounds
        self.assertIn("TARGET1-Liver", matrix.columns)
        self.assertIn("TARGET1-Kidney", matrix.columns)

        # Verify correct values
        self.assertAlmostEqual(matrix.loc["CONN1", "TARGET1-Liver"], 6.5)
        self.assertAlmostEqual(matrix.loc["CONN1", "TARGET1-Kidney"], 7.0)
        self.assertAlmostEqual(matrix.loc["CONN2", "TARGET1-Liver"], 5.5)
        self.assertAlmostEqual(matrix.loc["CONN2", "TARGET1-Kidney"], 8.0)

    def test_prepare_multitask_data_smiles_as_compound_col(self):
        """Test that smiles column is not duplicated when compound_col == smiles_col."""
        test_data = pd.DataFrame(
            {
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "", "", ""],
            }
        )

        matrix = prepare_multitask_data(
            df=test_data,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="smiles",
            smiles_col="smiles",
        )

        # smiles should appear exactly once (as the index), not duplicated as a column
        smiles_occurrences = list(matrix.columns).count("smiles")
        self.assertEqual(
            smiles_occurrences, 0, "smiles should be the index, not a column when compound_col==smiles_col"
        )
        self.assertEqual(matrix.index.name, "smiles")
        self.assertEqual(len(matrix), 2)
        self.assertIn("TARGET1", matrix.columns)
        self.assertIn("TARGET2", matrix.columns)

    def test_aggregate_data_with_standard_units_filter(self):
        """Test that standard_units is preserved and can be used as an id column."""
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4],
                "molecule_chembl_id": ["CHEMBL1"] * 4,
                "target_chembl_id": ["TARGET1"] * 4,
                "standard_smiles": ["CCO", "CCO", "CCO", "CCO"],
                "canonical_smiles": ["CCO", "CCO", "CCO", "CCO"],
                "standard_value": [50.0, 75.0, 100.0, 80.0],
                "standard_units": ["%", "%", "nM", "nM"],  # Different units
                "standard_relation": ["=", "=", "=", "="],
                "mutation": ["WT"] * 4,
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY1", "ASSAY2"],
                "standard_type": ["Inhibition", "Inhibition", "IC50", "IC50"],
                "assay_description": ["Test assay"] * 4,
                "assay_type": ["A"] * 4,
                "confidence_score": [9] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1"] * 4,
                "assay_tissue": [""] * 4,
                "assay_cell_type": [""] * 4,
                "relationship_type": ["D"] * 4,
                "max_phase": [1] * 4,
                "oral": [False] * 4,
                "prodrug": [False] * 4,
                "withdrawn_flag": [False] * 4,
                "doc_type": ["PUBLICATION"] * 4,
                "doi": ["10.1234/test"] * 4,
                "journal": ["Test Journal"] * 4,
                "year": [2020] * 4,
                "chembl_release": [30] * 4,
                "data_dropping_comment": [""] * 4,
                "data_processing_comment": [""] * 4,
            }
        )

        # Aggregate with standard_units as an id column to prevent mixing
        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=["standard_units"],
            output_path=None,
            compound_equality="connectivity",
            value_col="standard_value",
        )

        # Should have 2 rows: one for % units, one for nM units. They share the
        # simpler compound-target identifier and therefore receive one shared group label.
        self.assertEqual(len(aggr_df), 2)
        self.assertEqual(aggr_df["shared_identifier_group"].nunique(), 1)
        self.assertEqual(aggr_df["shared_identifier_group"].value_counts().iloc[0], 2)

        # Check that units are preserved and separate
        pct_row = aggr_df[aggr_df["standard_units"] == "%"]
        nm_row = aggr_df[aggr_df["standard_units"] == "nM"]

        self.assertEqual(len(pct_row), 1)
        self.assertEqual(len(nm_row), 1)

        # Check the aggregated values
        self.assertAlmostEqual(pct_row.iloc[0]["standard_value_mean"], (50.0 + 75.0) / 2)
        self.assertAlmostEqual(nm_row.iloc[0]["standard_value_mean"], (100.0 + 80.0) / 2)

    def test_compound_equality_smiles(self):
        """Test that compound_equality='smiles' uses standardized SMILES for aggregation."""
        # Two compounds with different connectivities but we'll use SMILES equality
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4],
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
                "target_chembl_id": ["TARGET1"] * 4,
                "standard_smiles": ["CCO", "CCO", "CCC", "CCC"],  # 2 unique SMILES
                "canonical_smiles": ["CCO", "CCO", "CCC", "CCC"],
                "pchembl_value": [6.0, 7.0, 5.0, 6.0],
                "standard_relation": ["=", "=", "=", "="],
                "mutation": ["WT"] * 4,
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY1", "ASSAY2"],
                "standard_type": ["IC50"] * 4,
                "assay_description": ["Test assay"] * 4,
                "assay_type": ["B"] * 4,
                "confidence_score": [9] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1", "DOC2", "DOC3", "DOC4"],
                "assay_tissue": [""] * 4,
                "assay_cell_type": [""] * 4,
                "relationship_type": ["D"] * 4,
                "max_phase": [1] * 4,
                "oral": [False] * 4,
                "prodrug": [False] * 4,
                "withdrawn_flag": [False] * 4,
                "doc_type": ["PUBLICATION"] * 4,
                "doi": ["10.1234/test"] * 4,
                "journal": ["Test Journal"] * 4,
                "year": [2020] * 4,
                "chembl_release": [30] * 4,
                "data_dropping_comment": [""] * 4,
                "data_processing_comment": [""] * 4,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=[],
            output_path=None,
            compound_equality="smiles",  # Use SMILES-based equality
        )

        # Should have 2 rows: one for CCO (2 measurements), one for CCC (2 measurements)
        self.assertEqual(len(aggr_df), 2, f"Expected 2 compounds, got {len(aggr_df)}")

        # Check that the aggregation happened correctly based on SMILES
        cco_rows = aggr_df[aggr_df["smiles"].str.contains("CCO", na=False)]
        self.assertEqual(len(cco_rows), 1, "Expected 1 aggregated row for CCO")
        self.assertEqual(cco_rows.iloc[0]["pchembl_value_counts"], 2, "Expected 2 measurements for CCO")

        ccc_rows = aggr_df[aggr_df["smiles"].str.contains("CCC", na=False)]
        self.assertEqual(len(ccc_rows), 1, "Expected 1 aggregated row for CCC")
        self.assertEqual(ccc_rows.iloc[0]["pchembl_value_counts"], 2, "Expected 2 measurements for CCC")

    def test_standard_value_and_units_preserved_in_aggregation(self):
        """Test that standard_value and standard_units are preserved as multivalue columns
        when aggregating on pchembl_value."""
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4],
                "molecule_chembl_id": ["CHEMBL1"] * 4,
                "target_chembl_id": ["TARGET1"] * 4,
                "standard_smiles": ["CCO", "CCO", "CCC", "CCC"],
                "canonical_smiles": ["CCO", "CCO", "CCC", "CCC"],
                "pchembl_value": [6.0, 7.0, 5.0, 6.0],
                "standard_value": [10.0, 1.0, 100.0, 50.0],
                "standard_units": ["nM", "nM", "uM", "uM"],
                "standard_relation": ["=", "=", "=", "="],
                "mutation": ["WT"] * 4,
                "assay_chembl_id": ["ASSAY1", "ASSAY1", "ASSAY2", "ASSAY2"],
                "standard_type": ["Ki", "Ki", "IC50", "IC50"],
                "assay_description": ["Test assay"] * 4,
                "assay_type": ["B"] * 4,
                "confidence_score": [9] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1", "DOC2", "DOC3", "DOC4"],
                "assay_tissue": [""] * 4,
                "assay_cell_type": [""] * 4,
                "relationship_type": ["D"] * 4,
                "max_phase": [1] * 4,
                "oral": [False] * 4,
                "prodrug": [False] * 4,
                "withdrawn_flag": [False] * 4,
                "doc_type": ["PUBLICATION"] * 4,
                "doi": ["10.1234/test"] * 4,
                "journal": ["Test Journal"] * 4,
                "year": [2020] * 4,
                "chembl_release": [30] * 4,
                "data_dropping_comment": [""] * 4,
                "data_processing_comment": [""] * 4,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            output_path=None,
            compound_equality="connectivity",
            value_col="pchembl_value",  # Aggregating on pchembl_value, NOT standard_value
        )

        # Should have 2 rows: one for CCO, one for CCC
        self.assertEqual(len(aggr_df), 2, f"Expected 2 rows, got {len(aggr_df)}")

        # Verify standard_value column exists
        self.assertIn(
            "standard_value",
            aggr_df.columns,
            "standard_value column should be preserved in aggregated output",
        )

        # Verify standard_units column exists
        self.assertIn(
            "standard_units",
            aggr_df.columns,
            "standard_units column should be preserved in aggregated output",
        )

        # Check that the values are preserved (should be pipe-separated multivalue columns)
        cco_row = aggr_df[aggr_df["smiles"].str.contains("CCO", na=False)]
        self.assertEqual(len(cco_row), 1, "Should have one row for CCO compound")

        # standard_value should contain both 10.0 and 1.0
        cco_std_val = cco_row.iloc[0]["standard_value"]
        self.assertIsNotNone(cco_std_val, "standard_value should not be None for CCO")
        # Should be a string with pipe-separated values
        self.assertIn(
            "|",
            str(cco_std_val),
            "standard_value should contain pipe-separated values for multiple measurements",
        )

        # standard_units should contain both nM values
        cco_std_units = cco_row.iloc[0]["standard_units"]
        self.assertIsNotNone(cco_std_units, "standard_units should not be None for CCO")
        self.assertIn(
            "|",
            str(cco_std_units),
            "standard_units should contain pipe-separated values for multiple measurements",
        )

    def test_nan_standard_relation_not_lost_during_aggregation(self):
        """Rows with NaN standard_relation must not be lost during aggregation.

        When standard_relation is NaN (e.g., AstraZeneca PPB assays in ChEMBL),
        those rows should be treated as non-censored and aggregated normally.
        Previously, NaN != "=" evaluated to True, causing NaN rows to enter the
        censored code path where string concatenation with NaN corrupted id_array.
        """
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2, 3, 4, 5],
                "molecule_chembl_id": ["CHEMBL1"] * 3 + ["CHEMBL2"] * 2,
                "target_chembl_id": ["TARGET1"] * 5,
                "standard_smiles": ["CCO"] * 3 + ["CCC"] * 2,
                "canonical_smiles": ["CCO"] * 3 + ["CCC"] * 2,
                "standard_value": [60.0, 65.0, 70.0, 80.0, 85.0],
                "standard_units": ["%"] * 5,
                "standard_relation": [np.nan, np.nan, "=", np.nan, np.nan],
                "mutation": ["WT"] * 5,
                "assay_chembl_id": ["ASSAY1"] * 5,
                "standard_type": ["PPB"] * 5,
                "assay_description": ["Test assay"] * 5,
                "assay_type": ["A"] * 5,
                "confidence_score": [9] * 5,
                "target_organism": ["Homo sapiens"] * 5,
                "document_chembl_id": ["DOC1"] * 5,
                "assay_tissue": [""] * 5,
                "assay_cell_type": [""] * 5,
                "relationship_type": ["D"] * 5,
                "max_phase": [1] * 5,
                "oral": [False] * 5,
                "prodrug": [False] * 5,
                "withdrawn_flag": [False] * 5,
                "doc_type": ["PUBLICATION"] * 5,
                "doi": ["10.1234/test"] * 5,
                "journal": ["Test Journal"] * 5,
                "year": [2020] * 5,
                "chembl_release": [30] * 5,
                "data_dropping_comment": [""] * 5,
                "data_processing_comment": [""] * 5,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,
            metadata_cols=[],
            extra_id_cols=["standard_units"],
            output_path=None,
            compound_equality="connectivity",
            value_col="standard_value",
        )

        # Both compounds must survive aggregation
        self.assertEqual(len(aggr_df), 2, f"Expected 2 compounds, got {len(aggr_df)}")

        # All 5 measurements should be accounted for
        self.assertEqual(
            aggr_df["standard_value_counts"].sum(),
            5,
            "All 5 measurements (including NaN relation rows) should be aggregated",
        )

        # CCO has 3 measurements (2 NaN + 1 "="), mean should be (60+65+70)/3
        cco_rows = aggr_df[aggr_df["smiles"].str.contains("CCO", na=False)]
        self.assertEqual(len(cco_rows), 1, "Should have one aggregated row for CCO")
        self.assertAlmostEqual(cco_rows.iloc[0]["standard_value_mean"], 65.0)
        self.assertEqual(cco_rows.iloc[0]["standard_value_counts"], 3)

        # CCC has 2 measurements (both NaN relation), mean should be (80+85)/2
        ccc_rows = aggr_df[aggr_df["smiles"].str.contains("CCC", na=False)]
        self.assertEqual(len(ccc_rows), 1, "Should have one aggregated row for CCC")
        self.assertAlmostEqual(ccc_rows.iloc[0]["standard_value_mean"], 82.5)
        self.assertEqual(ccc_rows.iloc[0]["standard_value_counts"], 2)

    def test_clean_data_drops_flags(self):
        """Test that clean_data filters out rows with specified quality flags."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "flag_to_remove", "", "another_flag"],
            }
        )

        result = clean_data(
            test_data,
            drop_flags=["flag_to_remove", "another_flag"],
        )

        # After filtering, only rows without flags should remain
        self.assertEqual(len(result), 2)
        self.assertTrue((result["data_dropping_comment"] == "").all())

    def test_clean_data_deduplicates(self):
        """Test that clean_data with deduplicate=True removes duplicate values and recalculates stats."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2"],
                "smiles": ["CCO", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["8.00|8.00|6.92", "5.50"],
                "pchembl_value_mean": [7.64, 5.50],
                "pchembl_value_median": [8.00, 5.50],
                "pchembl_value_std": [0.62, 0.0],
                "pchembl_value_counts": [3, 1],
                "data_dropping_comment": ["", ""],
            }
        )

        result = clean_data(
            test_data,
            deduplicate=True,
            value_col="pchembl_value",
        )

        # CONN1 should have had duplicates removed: "8.00|8.00|6.92" -> "8.00|6.92"
        from scipy.stats import gmean

        conn1 = result[result["connectivity"] == "CONN1"].iloc[0]
        self.assertEqual(conn1["pchembl_value"], "8.00|6.92")
        self.assertEqual(conn1["pchembl_value_counts"], 2)
        # Stats use geometric mean (correct for pchembl_value, which is -log10 scale)
        self.assertAlmostEqual(conn1["pchembl_value_mean"], gmean([8.00, 6.92]), places=3)

        # CONN2 should be unchanged (no duplicates)
        conn2 = result[result["connectivity"] == "CONN2"].iloc[0]
        self.assertEqual(conn2["pchembl_value"], "5.50")
        self.assertEqual(conn2["pchembl_value_counts"], 1)

    def test_clean_data_raises_on_conflicting_resolve_and_drop(self):
        """Test that clean_data raises ValueError when both resolve_annotation_error and drop unit error are set."""
        from Capricho.analysis import DroppingComment

        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "pchembl_value": ["6.0"],
                "pchembl_value_mean": [6.0],
                "data_dropping_comment": [""],
            }
        )

        with self.assertRaises(ValueError, msg="Should raise when both resolve and drop unit error are set"):
            clean_data(
                test_data,
                drop_flags=[DroppingComment.UNIT_ANNOTATION_ERROR.value],
                resolve_annotation_error="first",
            )

    def test_clean_data_composable_with_prepare(self):
        """Test end-to-end: clean_data then prepare_multitask_data."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "smiles": ["CCO", "CCO", "CCC", "CCC"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [6.5, 7.0, 5.5, 8.0],
                "data_dropping_comment": ["", "flag_to_remove", "", ""],
            }
        )

        # Step 1: clean
        cleaned = clean_data(test_data, drop_flags=["flag_to_remove"])

        # Step 2: pivot
        matrix = prepare_multitask_data(
            df=cleaned,
            task_col="target_chembl_id",
            value_col="pchembl_value_mean",
            compound_col="connectivity",
            smiles_col="smiles",
        )

        # CONN1-TARGET2 was flagged and removed, so TARGET2 should only have CONN2
        self.assertEqual(len(matrix), 2)
        self.assertIn("TARGET1", matrix.columns)
        self.assertIn("TARGET2", matrix.columns)
        self.assertTrue(pd.isna(matrix.loc["CONN1", "TARGET2"]))
        self.assertAlmostEqual(matrix.loc["CONN2", "TARGET2"], 8.0)

    def test_connectivity_assigned_when_chirality_false(self):
        """Connectivity must be assigned even when chirality=False strips stereochemistry.

        When chirality=False, process_repeat_mols re-canonicalizes SMILES without
        stereochemistry. The precomputed connectivity mapping (keyed on isomeric
        standard_smiles) must still produce valid connectivity values.
        """
        # Use SMILES with stereochemistry that will be stripped when chirality=False
        test_data = pd.DataFrame(
            {
                "activity_id": [1, 2],
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "target_chembl_id": ["TARGET1"] * 2,
                "standard_smiles": [
                    "N[C@@H](C(=O)O)Cc1ccccc1",  # L-phenylalanine (with stereo)
                    "C[C@H](O)CC",  # (R)-2-butanol (with stereo)
                ],
                "canonical_smiles": [
                    "N[C@@H](C(=O)O)Cc1ccccc1",
                    "C[C@H](O)CC",
                ],
                "standard_value": [50.0, 75.0],
                "standard_units": ["%", "%"],
                "standard_relation": ["=", "="],
                "mutation": ["WT"] * 2,
                "assay_chembl_id": ["ASSAY1"] * 2,
                "standard_type": ["PPB"] * 2,
                "assay_description": ["Test assay"] * 2,
                "assay_type": ["A"] * 2,
                "confidence_score": [9] * 2,
                "target_organism": ["Homo sapiens"] * 2,
                "document_chembl_id": ["DOC1"] * 2,
                "assay_tissue": [""] * 2,
                "assay_cell_type": [""] * 2,
                "relationship_type": ["D"] * 2,
                "max_phase": [1] * 2,
                "oral": [False] * 2,
                "prodrug": [False] * 2,
                "withdrawn_flag": [False] * 2,
                "doc_type": ["PUBLICATION"] * 2,
                "doi": ["10.1234/test"] * 2,
                "journal": ["Test Journal"] * 2,
                "year": [2020] * 2,
                "chembl_release": [30] * 2,
                "data_dropping_comment": [""] * 2,
                "data_processing_comment": [""] * 2,
            }
        )

        aggr_df = aggregate_data(
            test_data,
            chirality=False,  # This strips stereochemistry from smiles
            metadata_cols=[],
            extra_id_cols=["standard_units"],
            output_path=None,
            compound_equality="connectivity",
            value_col="standard_value",
        )

        # All rows must have valid (non-NaN) connectivity
        self.assertTrue(
            aggr_df["connectivity"].notna().all(),
            f"All rows should have connectivity assigned, but got NaN for: "
            f"{aggr_df[aggr_df['connectivity'].isna()]['smiles'].tolist()}",
        )

        # Connectivity values should be 14-char InChI key first layer
        for conn in aggr_df["connectivity"]:
            self.assertEqual(len(conn), 14, f"Connectivity should be 14 chars, got: {conn}")

    def test_measurement_level_flag_filtering_partial(self):
        """Partial flag: 3 measurements, 1 flagged -> 2 remain, stats recalculated."""
        from scipy.stats import gmean

        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.50|7.00"],
                "pchembl_value_mean": [6.5],
                "pchembl_value_median": [6.5],
                "pchembl_value_std": [0.5],
                "pchembl_value_counts": [3],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "data_dropping_comment": ["Unit Error||"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])

        # Row survives (only 1 of 3 measurements flagged)
        self.assertEqual(len(result), 1)
        # Flagged measurement (position 0) removed
        self.assertEqual(result.iloc[0]["pchembl_value"], "6.50|7.00")
        self.assertEqual(result.iloc[0]["activity_id"], "ACT2|ACT3")
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 2)
        # Stats recalculated with geometric mean
        expected_mean = gmean([6.50, 7.00])
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], expected_mean, places=3)

    def test_measurement_level_flag_filtering_all_flagged(self):
        """All measurements flagged -> row removed entirely."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.50|7.00"],
                "pchembl_value_mean": [6.5],
                "pchembl_value_median": [6.5],
                "pchembl_value_std": [0.5],
                "pchembl_value_counts": [3],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "data_dropping_comment": ["Unit Error|Unit Error|Unit Error"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])
        self.assertEqual(len(result), 0)

    def test_measurement_level_flag_filtering_none_flagged(self):
        """No measurements flagged -> row unchanged."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.50|7.00"],
                "pchembl_value_mean": [6.5],
                "pchembl_value_median": [6.5],
                "pchembl_value_std": [0.5],
                "pchembl_value_counts": [3],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "data_dropping_comment": ["||"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value"], "6.00|6.50|7.00")
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 3)

    def test_measurement_level_flag_filtering_single_flagged(self):
        """Single measurement with flag -> row removed (same as old behavior)."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "pchembl_value": ["6.00"],
                "pchembl_value_mean": [6.0],
                "pchembl_value_median": [6.0],
                "pchembl_value_std": [0.0],
                "pchembl_value_counts": [1],
                "activity_id": ["ACT1"],
                "data_dropping_comment": ["Unit Error"],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])
        self.assertEqual(len(result), 0)

    def test_measurement_level_flag_filtering_single_clean(self):
        """Single measurement without flag -> row kept."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "pchembl_value": ["6.00"],
                "pchembl_value_mean": [6.0],
                "pchembl_value_median": [6.0],
                "pchembl_value_std": [0.0],
                "pchembl_value_counts": [1],
                "activity_id": ["ACT1"],
                "data_dropping_comment": [""],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], 6.0)

    def test_measurement_level_flag_filtering_selective(self):
        """Filter only 'Flag A' when 'Flag B' also present at different positions."""
        from scipy.stats import gmean

        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.50|7.00"],
                "pchembl_value_mean": [6.5],
                "pchembl_value_median": [6.5],
                "pchembl_value_std": [0.5],
                "pchembl_value_counts": [3],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "data_dropping_comment": ["Flag A|Flag B|"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, drop_flags=["Flag A"])

        # Only position 0 (Flag A) removed; position 1 (Flag B) and 2 (clean) remain
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value"], "6.50|7.00")
        self.assertEqual(result.iloc[0]["activity_id"], "ACT2|ACT3")
        # Flag B still present in remaining comment
        self.assertIn("Flag B", result.iloc[0]["data_dropping_comment"])

    def test_measurement_level_flag_filtering_compound_comment(self):
        """Comment 'Flag A & Flag B' on one measurement; filtering 'Flag A' removes it."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["TARGET1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.50|7.00"],
                "pchembl_value_mean": [6.5],
                "pchembl_value_median": [6.5],
                "pchembl_value_std": [0.5],
                "pchembl_value_counts": [3],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "data_dropping_comment": ["Flag A & Flag B||"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, drop_flags=["Flag A"])

        self.assertEqual(len(result), 1)
        # Position 0 removed (contains "Flag A" as substring)
        self.assertEqual(result.iloc[0]["pchembl_value"], "6.50|7.00")
        self.assertEqual(result.iloc[0]["activity_id"], "ACT2|ACT3")

    def test_measurement_level_flag_filtering_mixed_rows(self):
        """Mix of aggregated and non-aggregated rows: each handled correctly."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "smiles": ["CCO", "CCC", "CCCC"],
                "target_chembl_id": ["T1", "T1", "T1"],
                "standard_relation": ["=|=|=", "=", "="],
                "pchembl_value": ["6.00|6.50|7.00", "5.50", "8.00"],
                "pchembl_value_mean": [6.5, 5.5, 8.0],
                "pchembl_value_median": [6.5, 5.5, 8.0],
                "pchembl_value_std": [0.5, 0.0, 0.0],
                "pchembl_value_counts": [3, 1, 1],
                "activity_id": ["A1|A2|A3", "A4", "A5"],
                "data_dropping_comment": ["Unit Error||", "Unit Error", ""],
                "data_processing_comment": ["||", "", ""],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error"])

        # CONN1: 1 of 3 flagged -> 2 remain
        # CONN2: single measurement flagged -> row removed
        # CONN3: no flag -> kept
        self.assertEqual(len(result), 2)
        conn1 = result[result["connectivity"] == "CONN1"].iloc[0]
        self.assertEqual(conn1["pchembl_value"], "6.50|7.00")
        self.assertEqual(conn1["pchembl_value_counts"], 2)

        conn3 = result[result["connectivity"] == "CONN3"].iloc[0]
        self.assertAlmostEqual(conn3["pchembl_value_mean"], 8.0)

    def test_measurement_level_dedup_then_flag_filter(self):
        """End-to-end: deduplicate then filter flags at measurement level."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["T1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["6.00|6.00|7.00"],
                "pchembl_value_mean": [6.33],
                "pchembl_value_median": [6.0],
                "pchembl_value_std": [0.58],
                "pchembl_value_counts": [3],
                "activity_id": ["A1|A2|A3"],
                "data_dropping_comment": ["||Unit Error"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, deduplicate=True, drop_flags=["Unit Error"])

        # After dedup: "6.00|6.00|7.00" -> "6.00|7.00", comments "||Unit Error" -> "|Unit Error"
        # After flag filter: position 1 of "6.00|7.00" has "Unit Error" -> removed -> "6.00"
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value"], "6.00")
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 1)
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], 6.0)

    def test_measurement_level_backward_compat(self):
        """Non-aggregated DataFrame: same behavior as old filter_dropping_flags."""
        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "smiles": ["CCO", "CCC", "CCCC"],
                "target_chembl_id": ["T1", "T1", "T1"],
                "pchembl_value_mean": [6.5, 5.5, 8.0],
                "data_dropping_comment": ["Unit Error", "", "Another Flag"],
            }
        )

        result = clean_data(test_data, drop_flags=["Unit Error", "Another Flag"])

        # CONN1 and CONN3 removed, CONN2 kept
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["connectivity"], "CONN2")

    def test_dedup_uses_geometric_stats(self):
        """Deduplication recalculates stats with geometric mean for pchembl_value."""
        from scipy.stats import gmean

        test_data = pd.DataFrame(
            {
                "connectivity": ["CONN1"],
                "smiles": ["CCO"],
                "target_chembl_id": ["T1"],
                "standard_relation": ["=|=|="],
                "pchembl_value": ["8.00|8.00|6.92"],
                "pchembl_value_mean": [7.64],
                "pchembl_value_median": [8.0],
                "pchembl_value_std": [0.62],
                "pchembl_value_counts": [3],
                "activity_id": ["A1|A2|A3"],
                "data_dropping_comment": ["||"],
                "data_processing_comment": ["||"],
            }
        )

        result = clean_data(test_data, deduplicate=True)

        # After dedup: "8.00|8.00|6.92" -> "8.00|6.92"
        self.assertEqual(result.iloc[0]["pchembl_value"], "8.00|6.92")
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 2)

        # Stats should use geometric mean, not arithmetic mean
        expected_gmean = gmean([8.00, 6.92])
        expected_amean = (8.00 + 6.92) / 2
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], expected_gmean, places=3)
        # Sanity check: geometric and arithmetic means differ
        self.assertNotAlmostEqual(expected_gmean, expected_amean, places=2)


if __name__ == "__main__":
    unittest.main()
