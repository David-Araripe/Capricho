"""Tests for analysis module."""

import unittest

import numpy as np
import pandas as pd

from Capricho.analysis import (
    DroppingComment,
    ProcessingComment,
    deaggregate_data,
    explode_assay_comparability,
    extract_assay_threshold,
    get_all_comments,
    normalize_comment_pattern,
)


class TestNormalizeCommentPattern(unittest.TestCase):
    """Tests for normalize_comment_pattern function."""

    def test_normalize_assay_size_min(self):
        """Test that assay size minimum patterns are normalized correctly."""
        self.assertEqual(normalize_comment_pattern("Assay size < 20"), "Assay size <")
        self.assertEqual(normalize_comment_pattern("Assay size < 10"), "Assay size <")
        self.assertEqual(normalize_comment_pattern("Assay size < 50"), "Assay size <")

    def test_normalize_assay_size_max(self):
        """Test that assay size maximum patterns are normalized correctly."""
        self.assertEqual(normalize_comment_pattern("Assay size > 100"), "Assay size >")
        self.assertEqual(normalize_comment_pattern("Assay size > 200"), "Assay size >")
        self.assertEqual(normalize_comment_pattern("Assay size > 50"), "Assay size >")

    def test_normalize_static_comments(self):
        """Test that static comments remain unchanged."""
        static_comments = [
            "Unit Annotation Error",
            "Potential Duplicate",
            "Data Validity Comment Present",
            "Undefined Stereochemistry",
            "Salt/solvent removed",
            "Calculated pChEMBL",
        ]
        for comment in static_comments:
            self.assertEqual(
                normalize_comment_pattern(comment), comment, f"Comment '{comment}' should not be normalized"
            )


class TestExtractAssayThreshold(unittest.TestCase):
    """Tests for extract_assay_threshold function."""

    def test_extract_min_threshold(self):
        """Test extraction of minimum assay size threshold."""
        self.assertEqual(extract_assay_threshold("Assay size < 20"), "20")
        self.assertEqual(extract_assay_threshold("Assay size < 10"), "10")
        self.assertEqual(extract_assay_threshold("Assay size < 50"), "50")

    def test_extract_max_threshold(self):
        """Test extraction of maximum assay size threshold."""
        self.assertEqual(extract_assay_threshold("Assay size > 100"), "100")
        self.assertEqual(extract_assay_threshold("Assay size > 200"), "200")
        self.assertEqual(extract_assay_threshold("Assay size > 500"), "500")

    def test_extract_non_assay_size_comment(self):
        """Test that non-assay-size comments return empty string."""
        self.assertEqual(extract_assay_threshold("Unit Annotation Error"), "")
        self.assertEqual(extract_assay_threshold("Potential Duplicate"), "")
        self.assertEqual(extract_assay_threshold(""), "")


class TestGetAllComments(unittest.TestCase):
    """Tests for get_all_comments function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_all_comments()
        self.assertIsInstance(result, list)

    def test_contains_all_enum_values(self):
        """Test that all enum values are present in the returned list."""
        result = get_all_comments()

        # Check ProcessingComment values
        for comment in ProcessingComment:
            self.assertIn(
                comment.value, result, f"ProcessingComment '{comment.value}' should be in get_all_comments()"
            )

        # Check DroppingComment values
        for comment in DroppingComment:
            self.assertIn(
                comment.value, result, f"DroppingComment '{comment.value}' should be in get_all_comments()"
            )

    def test_no_duplicates(self):
        """Test that there are no duplicate values in the returned list."""
        result = get_all_comments()
        self.assertEqual(len(result), len(set(result)), "get_all_comments() should not contain duplicates")

    def test_assay_size_patterns_present(self):
        """Test that assay size patterns (without thresholds) are present."""
        result = get_all_comments()
        self.assertIn("Assay size <", result)
        self.assertIn("Assay size >", result)


class TestExplodeAssayComparability(unittest.TestCase):
    """Tests for explode_assay_comparability function."""

    def test_explode_basic(self):
        """Test basic explosion of multi-assay data."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "repeat": [0, 1],
                "activity_id": ["ACT1|ACT2|ACT3", "ACT4|ACT5"],
                "assay_chembl_id": ["ASSAY1|ASSAY2|ASSAY3", "ASSAY4|ASSAY5"],
                "pchembl_value": ["6.0|6.5|7.0", "5.5|6.0"],
                "data_processing_comment": ["||", "|"],
                "data_dropping_comment": ["||", "|"],
                "standard_type": ["IC50|IC50|IC50", "Ki|Ki"],
                "canonical_smiles": ["CCCC|CCCC|CCCC", "CCCO|CCCO"],
            }
        )

        result = explode_assay_comparability(df)

        # MOL1 has 3 assays -> 3 choose 2 = 3 pairs
        # MOL2 has 2 assays -> 2 choose 2 = 1 pair
        # Total = 4 rows
        self.assertEqual(len(result), 4)

        # Check that _x and _y columns exist
        self.assertIn("assay_chembl_id_x", result.columns)
        self.assertIn("assay_chembl_id_y", result.columns)
        self.assertIn("pchembl_value_x", result.columns)
        self.assertIn("pchembl_value_y", result.columns)
        self.assertIn("canonical_smiles_x", result.columns)
        self.assertIn("canonical_smiles_y", result.columns)

    def test_explode_filters_same_assay(self):
        """Test that comparisons of same assay are filtered out."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "target_chembl_id": ["TARGET1"],
                "repeat": [0],
                "activity_id": ["ACT1|ACT2"],
                "assay_chembl_id": ["ASSAY1|ASSAY2"],
                "pchembl_value": ["6.0|6.5"],
                "data_processing_comment": ["|"],
                "data_dropping_comment": ["|"],
                "standard_type": ["IC50|IC50"],
                "canonical_smiles": ["CCCC|CCCC"],
            }
        )

        result = explode_assay_comparability(df)

        # Should only have pairs where assay_x != assay_y
        self.assertTrue(
            (result["assay_chembl_id_x"] != result["assay_chembl_id_y"]).all(),
            "All pairs should have different assays",
        )

    def test_explode_combines_comments(self):
        """Test that processing and dropping comments are combined correctly."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "target_chembl_id": ["TARGET1"],
                "repeat": [0],
                "activity_id": ["ACT1|ACT2"],
                "assay_chembl_id": ["ASSAY1|ASSAY2"],
                "pchembl_value": ["6.0|6.5"],
                "data_processing_comment": ["Calculated pChEMBL|Salt/solvent removed"],
                "data_dropping_comment": ["Assay size < 20|"],
                "standard_type": ["IC50|IC50"],
                "canonical_smiles": ["CCCC|CCCC"],
            }
        )

        result = explode_assay_comparability(df)

        # Check that combined comments contain both original comments
        self.assertIn("processing_comment", result.columns)
        self.assertIn("dropping_comment", result.columns)

        # The first row should have both comments combined
        proc_comment = result.iloc[0]["processing_comment"]
        self.assertIn("Calculated pChEMBL", proc_comment)
        self.assertIn("Salt/solvent removed", proc_comment)

    def test_explode_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(
            {
                "connectivity": [],
                "target_chembl_id": [],
                "repeat": [],
                "activity_id": [],
                "assay_chembl_id": [],
                "pchembl_value": [],
                "data_processing_comment": [],
                "data_dropping_comment": [],
                "standard_type": [],
                "canonical_smiles": [],
            }
        )

        result = explode_assay_comparability(df)

        # Should return empty DataFrame with correct columns
        self.assertEqual(len(result), 0)
        self.assertIn("assay_chembl_id_x", result.columns)
        self.assertIn("assay_chembl_id_y", result.columns)

    def test_explode_custom_separator(self):
        """Test that custom separator works correctly."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "target_chembl_id": ["TARGET1"],
                "repeat": [0],
                "activity_id": ["ACT1;ACT2"],
                "assay_chembl_id": ["ASSAY1;ASSAY2"],
                "pchembl_value": ["6.0;6.5"],
                "data_processing_comment": [";"],
                "data_dropping_comment": [";"],
                "standard_type": ["IC50;IC50"],
                "canonical_smiles": ["CCCC;CCCC"],
            }
        )

        result = explode_assay_comparability(df, sep_str=";")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["assay_chembl_id_x"], "ASSAY1")
        self.assertEqual(result.iloc[0]["assay_chembl_id_y"], "ASSAY2")

    def test_explode_with_single_value_columns(self):
        """Test explosion when some columns are single-valued (e.g., from --id-columns).

        When data is aggregated with --id-columns, those columns become single-valued
        rather than pipe-delimited. The function should auto-detect this and handle
        them as single-value columns.
        """
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "repeat": [0, 1],
                "activity_id": ["ACT1|ACT2|ACT3", "ACT4|ACT5"],
                "assay_chembl_id": ["ASSAY1|ASSAY2|ASSAY3", "ASSAY4|ASSAY5"],
                "pchembl_value": ["6.0|6.5|7.0", "5.5|6.0"],
                "data_processing_comment": ["||", "|"],
                "data_dropping_comment": ["||", "|"],
                # standard_type is single-valued (as if aggregated with --id-columns standard_type)
                "standard_type": ["IC50", "Ki"],
                "canonical_smiles": ["CCCC|CCCC|CCCC", "CCCO|CCCO"],
            }
        )

        result = explode_assay_comparability(df)

        # MOL1 has 3 assays -> 3 choose 2 = 3 pairs
        # MOL2 has 2 assays -> 2 choose 2 = 1 pair
        # Total = 4 rows
        self.assertEqual(len(result), 4)

        # Check that standard_type is preserved as single value (not split)
        mol1_rows = result[result["connectivity"] == "MOL1"]
        mol2_rows = result[result["connectivity"] == "MOL2"]

        # standard_type should be in columns as _x suffix since it's treated as single-value
        self.assertIn("standard_type", result.columns)
        self.assertTrue((mol1_rows["standard_type"] == "IC50").all())
        self.assertTrue((mol2_rows["standard_type"] == "Ki").all())

    def test_explode_with_custom_value_column(self):
        """Test explosion with standard_value instead of pchembl_value."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "repeat": [0, 1],
                "activity_id": ["ACT1|ACT2", "ACT3|ACT4"],
                "assay_chembl_id": ["ASSAY1|ASSAY2", "ASSAY3|ASSAY4"],
                "standard_value": ["100.0|200.0", "50.0|75.0"],
                "data_processing_comment": ["|", "|"],
                "data_dropping_comment": ["|", "|"],
                "standard_type": ["Pc|Pc", "Pc|Pc"],
                "canonical_smiles": ["CCCC|CCCC", "CCCO|CCCO"],
            }
        )

        result = explode_assay_comparability(df, value_column="standard_value")

        # MOL1 has 2 assays -> 1 pair, MOL2 has 2 assays -> 1 pair
        # Total = 2 rows
        self.assertEqual(len(result), 2)

        # Check that standard_value_x and standard_value_y columns exist
        self.assertIn("standard_value_x", result.columns)
        self.assertIn("standard_value_y", result.columns)
        # pchembl columns should NOT exist
        self.assertNotIn("pchembl_value_x", result.columns)
        self.assertNotIn("pchembl_value_y", result.columns)

        # Verify values
        mol1_row = result[result["connectivity"] == "MOL1"].iloc[0]
        self.assertEqual(mol1_row["standard_value_x"], "100.0")
        self.assertEqual(mol1_row["standard_value_y"], "200.0")


class TestDeaggregateData(unittest.TestCase):
    """Tests for deaggregate_data function."""

    def test_deaggregate_basic(self):
        """Test basic de-aggregation of pipe-delimited data."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2", "MOL3"],
                "pchembl_value": ["6.0|6.5|7.0", "5.5", "8.0|8.5"],
                "activity_id": ["ACT1|ACT2|ACT3", "ACT4", "ACT5|ACT6"],
                "assay_chembl_id": ["ASSAY1|ASSAY2|ASSAY3", "ASSAY4", "ASSAY5|ASSAY6"],
            }
        )

        result = deaggregate_data(df)

        # MOL1 should split into 3 rows, MOL2 stays 1 row, MOL3 splits into 2 rows
        # Total: 3 + 1 + 2 = 6 rows
        self.assertEqual(len(result), 6)

        # Check that MOL1 data was split correctly
        mol1_data = result[result["connectivity"] == "MOL1"].sort_values("pchembl_value")
        self.assertEqual(len(mol1_data), 3)
        self.assertEqual(mol1_data["pchembl_value"].tolist(), ["6.0", "6.5", "7.0"])
        self.assertEqual(mol1_data["activity_id"].tolist(), ["ACT1", "ACT2", "ACT3"])

        # Check that MOL2 data remained unchanged
        mol2_data = result[result["connectivity"] == "MOL2"]
        self.assertEqual(len(mol2_data), 1)
        self.assertEqual(mol2_data["pchembl_value"].values[0], "5.5")

    def test_deaggregate_no_pipes(self):
        """Test that data without pipes is returned unchanged."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["6.0", "5.5"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = deaggregate_data(df)

        # Should return same data
        self.assertEqual(len(result), 2)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), df)

    def test_deaggregate_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame({"connectivity": [], "pchembl_value": [], "activity_id": []})

        result = deaggregate_data(df)

        # Should return empty DataFrame with same structure
        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), list(df.columns))

    def test_deaggregate_mixed_columns(self):
        """Test de-aggregation with some columns having pipes and others not."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],  # No pipes
                "target_chembl_id": ["TARGET1", "TARGET2"],  # No pipes
                "pchembl_value": ["6.0|6.5", "7.0|7.5|8.0"],  # Has pipes
                "activity_id": ["ACT1|ACT2", "ACT3|ACT4|ACT5"],  # Has pipes
            }
        )

        result = deaggregate_data(df)

        # MOL1: 2 rows, MOL2: 3 rows -> Total: 5 rows
        self.assertEqual(len(result), 5)

        # Check that single-value columns are preserved
        mol1_data = result[result["connectivity"] == "MOL1"]
        self.assertTrue((mol1_data["connectivity"] == "MOL1").all())
        self.assertTrue((mol1_data["target_chembl_id"] == "TARGET1").all())

    def test_deaggregate_with_nan_values(self):
        """Test de-aggregation handles NaN values correctly."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["6.0|6.5", None],
                "activity_id": ["ACT1|ACT2", "ACT3"],
            }
        )

        result = deaggregate_data(df)

        # MOL1 should split into 2 rows, MOL2 stays as 1 (no pipe in activity_id)
        # Total: 2 + 1 = 3 rows
        self.assertEqual(len(result), 3)

        mol1_data = result[result["connectivity"] == "MOL1"]
        self.assertEqual(len(mol1_data), 2)

    def test_deaggregate_custom_separator(self):
        """Test de-aggregation with custom separator."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["6.0;6.5", "7.0"],
                "activity_id": ["ACT1;ACT2", "ACT3"],
            }
        )

        result = deaggregate_data(df, sep_str=";")

        # MOL1 should split into 2 rows, MOL2 stays as 1
        self.assertEqual(len(result), 3)

        mol1_data = result[result["connectivity"] == "MOL1"].sort_values("pchembl_value")
        self.assertEqual(mol1_data["pchembl_value"].tolist(), ["6.0", "6.5"])
        self.assertEqual(mol1_data["activity_id"].tolist(), ["ACT1", "ACT2"])

    def test_deaggregate_preserves_data_types(self):
        """Test that de-aggregation preserves column order and structure."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "target_chembl_id": ["TARGET1"],
                "pchembl_value": ["6.0|6.5"],
                "activity_id": ["ACT1|ACT2"],
            }
        )

        result = deaggregate_data(df)

        # Check that column order is preserved
        self.assertListEqual(list(result.columns), list(df.columns))

        # Check that we have the expected rows
        self.assertEqual(len(result), 2)

    def test_deaggregate_all_rows_aggregated(self):
        """Test de-aggregation when all rows contain pipes."""
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["6.0|6.5", "7.0|7.5"],
                "activity_id": ["ACT1|ACT2", "ACT3|ACT4"],
            }
        )

        result = deaggregate_data(df)

        # Both rows split into 2 each -> Total: 4 rows
        self.assertEqual(len(result), 4)

        # Verify all molecules are present
        self.assertEqual(len(result[result["connectivity"] == "MOL1"]), 2)
        self.assertEqual(len(result[result["connectivity"] == "MOL2"]), 2)


class TestDeduplicateAggregatedValues(unittest.TestCase):
    """Tests for deduplicate_aggregated_values function."""

    def test_deduplicate_identical_values(self):
        """Test that identical pchembl values within a row are deduplicated."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "pchembl_value": ["8.00|8.00|8.00|6.92"],
                "activity_id": ["ACT1|ACT2|ACT3|ACT4"],
                "document_chembl_id": ["DOC1|DOC2|DOC3|DOC4"],
                "data_processing_comment": [
                    "pChEMBL Duplication Across Documents|pChEMBL Duplication Across Documents||"
                ],
            }
        )

        result = deduplicate_aggregated_values(df)

        # Should keep only unique pchembl values: 8.00 and 6.92
        self.assertEqual(len(result), 1)
        pchembl_vals = result.iloc[0]["pchembl_value"].split("|")
        self.assertEqual(len(pchembl_vals), 2)
        self.assertIn("8.00", pchembl_vals)
        self.assertIn("6.92", pchembl_vals)

    def test_deduplicate_keeps_first_occurrence(self):
        """Test that deduplication keeps first occurrence of each value."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "pchembl_value": ["8.00|6.50|8.00"],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "document_chembl_id": ["DOC1|DOC2|DOC3"],
                "data_processing_comment": ["||"],
            }
        )

        result = deduplicate_aggregated_values(df)

        # First 8.00 should be kept (ACT1, DOC1)
        pchembl_vals = result.iloc[0]["pchembl_value"].split("|")
        activity_ids = result.iloc[0]["activity_id"].split("|")
        doc_ids = result.iloc[0]["document_chembl_id"].split("|")

        self.assertEqual(len(pchembl_vals), 2)
        # Check order: 8.00 first, then 6.50
        self.assertEqual(pchembl_vals[0], "8.00")
        self.assertEqual(pchembl_vals[1], "6.50")
        self.assertEqual(activity_ids[0], "ACT1")
        self.assertEqual(doc_ids[0], "DOC1")

    def test_deduplicate_no_duplicates(self):
        """Test that rows without duplicates are unchanged."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "pchembl_value": ["8.00|7.50|6.92"],
                "activity_id": ["ACT1|ACT2|ACT3"],
                "document_chembl_id": ["DOC1|DOC2|DOC3"],
                "data_processing_comment": ["||"],
            }
        )

        result = deduplicate_aggregated_values(df)

        # No duplicates, should remain unchanged
        self.assertEqual(result.iloc[0]["pchembl_value"], "8.00|7.50|6.92")
        self.assertEqual(result.iloc[0]["activity_id"], "ACT1|ACT2|ACT3")

    def test_deduplicate_single_value_rows(self):
        """Test that single-value rows are unchanged."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["8.00", "7.50|7.50"],
                "activity_id": ["ACT1", "ACT2|ACT3"],
                "document_chembl_id": ["DOC1", "DOC2|DOC3"],
                "data_processing_comment": ["", "|"],
            }
        )

        result = deduplicate_aggregated_values(df)

        # MOL1 unchanged (single value)
        mol1 = result[result["connectivity"] == "MOL1"].iloc[0]
        self.assertEqual(mol1["pchembl_value"], "8.00")

        # MOL2 deduplicated (7.50|7.50 -> 7.50)
        mol2 = result[result["connectivity"] == "MOL2"].iloc[0]
        self.assertEqual(mol2["pchembl_value"], "7.50")

    def test_deduplicate_preserves_other_columns(self):
        """Test that columns not involved in deduplication are preserved."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "target_chembl_id": ["TARGET1"],
                "smiles": ["CCCC"],
                "pchembl_value": ["8.00|8.00"],
                "activity_id": ["ACT1|ACT2"],
                "document_chembl_id": ["DOC1|DOC2"],
                "data_processing_comment": ["|"],
                "pchembl_value_mean": [8.0],
                "pchembl_value_counts": [2.0],
            }
        )

        result = deduplicate_aggregated_values(df)

        # Single-value columns should be preserved
        self.assertEqual(result.iloc[0]["connectivity"], "MOL1")
        self.assertEqual(result.iloc[0]["target_chembl_id"], "TARGET1")
        self.assertEqual(result.iloc[0]["smiles"], "CCCC")

    def test_deduplicate_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        from Capricho.analysis import deduplicate_aggregated_values

        df = pd.DataFrame(
            {
                "connectivity": [],
                "pchembl_value": [],
                "activity_id": [],
                "document_chembl_id": [],
                "data_processing_comment": [],
            }
        )

        result = deduplicate_aggregated_values(df)

        self.assertEqual(len(result), 0)


class TestRecalculateAggregatedStats(unittest.TestCase):
    """Tests for recalculate_aggregated_stats function."""

    def test_recalculate_stats_after_dedup(self):
        """Test that stats are recalculated after deduplication."""
        from Capricho.analysis import recalculate_aggregated_stats

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "pchembl_value": ["8.00|6.00"],  # Already deduplicated
                "pchembl_value_mean": [7.33],  # Old mean from 8.00|8.00|6.00
                "pchembl_value_median": [8.0],
                "pchembl_value_std": [1.15],
                "pchembl_value_counts": [3.0],
            }
        )

        result = recalculate_aggregated_stats(df)

        # New mean should be (8.0 + 6.0) / 2 = 7.0
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], 7.0, places=2)
        # New median should be 7.0
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_median"], 7.0, places=2)
        # New count should be 2
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 2)

    def test_recalculate_stats_single_value(self):
        """Test that single values get correct stats."""
        from Capricho.analysis import recalculate_aggregated_stats

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1"],
                "pchembl_value": ["8.00"],
                "pchembl_value_mean": [8.0],
                "pchembl_value_median": [8.0],
                "pchembl_value_std": [0.0],
                "pchembl_value_counts": [1.0],
            }
        )

        result = recalculate_aggregated_stats(df)

        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], 8.0)
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_median"], 8.0)
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 1)

    def test_recalculate_stats_multiple_rows(self):
        """Test recalculation works for multiple rows."""
        from Capricho.analysis import recalculate_aggregated_stats

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL2"],
                "pchembl_value": ["8.00|6.00|7.00", "5.50"],
                "pchembl_value_mean": [7.0, 5.5],
                "pchembl_value_median": [7.0, 5.5],
                "pchembl_value_std": [1.0, 0.0],
                "pchembl_value_counts": [3.0, 1.0],
            }
        )

        result = recalculate_aggregated_stats(df)

        # MOL1: mean of 8, 6, 7 = 7.0
        self.assertAlmostEqual(result.iloc[0]["pchembl_value_mean"], 7.0, places=2)
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 3)

        # MOL2: single value, unchanged
        self.assertAlmostEqual(result.iloc[1]["pchembl_value_mean"], 5.5)
        self.assertEqual(result.iloc[1]["pchembl_value_counts"], 1)

    def test_recalculate_stats_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        from Capricho.analysis import recalculate_aggregated_stats

        df = pd.DataFrame(
            {
                "connectivity": [],
                "pchembl_value": [],
                "pchembl_value_mean": [],
                "pchembl_value_median": [],
                "pchembl_value_std": [],
                "pchembl_value_counts": [],
            }
        )

        result = recalculate_aggregated_stats(df)

        self.assertEqual(len(result), 0)


class TestResolveAnnotationErrors(unittest.TestCase):
    """Tests for resolve_annotation_errors function."""

    def test_resolve_keeps_earliest_document(self):
        """Test that resolution keeps measurement from earliest document."""
        from Capricho.analysis import resolve_annotation_errors

        # Two measurements for same molecule, different assays, differ by 3.0
        # Value 5.0 from 2010, value 8.0 from 2015 - should keep 5.0
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1"],
                "pchembl_value": ["5.00", "8.00"],
                "year": ["2010", "2015"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2"],
                "molecule_chembl_id": ["CHEMBL123", "CHEMBL123"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # Should keep only the 2010 measurement (value=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value"], "5.00")
        self.assertEqual(result.iloc[0]["year"], "2010")

    def test_resolve_6_log_unit_difference(self):
        """Test that 6.0 log unit differences are also detected."""
        from Capricho.analysis import resolve_annotation_errors

        # Differ by 6.0: 4.0 vs 10.0
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1"],
                "pchembl_value": ["4.00", "10.00"],
                "year": ["2012", "2008"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2"],
                "molecule_chembl_id": ["CHEMBL123", "CHEMBL123"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # Should keep 2008 measurement (value=10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value"], "10.00")
        self.assertEqual(result.iloc[0]["year"], "2008")

    def test_resolve_no_error_unchanged(self):
        """Test that measurements without annotation errors are unchanged."""
        from Capricho.analysis import resolve_annotation_errors

        # Two measurements differ by 1.5 (not 3.0 or 6.0) - no error
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1"],
                "pchembl_value": ["5.00", "6.50"],
                "year": ["2010", "2015"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2"],
                "molecule_chembl_id": ["CHEMBL123", "CHEMBL123"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # Both should be kept
        self.assertEqual(len(result), 2)

    def test_resolve_same_assay_ignored(self):
        """Test that pairs from the same assay are not flagged."""
        from Capricho.analysis import resolve_annotation_errors

        # Same assay - even with 3.0 difference, not an annotation error
        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1"],
                "pchembl_value": ["5.00", "8.00"],
                "year": ["2010", "2015"],
                "assay_chembl_id": ["ASSAY1", "ASSAY1"],  # Same assay
                "molecule_chembl_id": ["CHEMBL123", "CHEMBL123"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # Both should be kept (same assay doesn't trigger annotation error)
        self.assertEqual(len(result), 2)

    def test_resolve_multiple_molecules(self):
        """Test resolution works across multiple molecules."""
        from Capricho.analysis import resolve_annotation_errors

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1", "MOL2", "MOL2"],
                "pchembl_value": ["5.00", "8.00", "6.00", "7.00"],
                "year": ["2010", "2015", "2012", "2014"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY3", "ASSAY4"],
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL2", "CHEMBL2"],
                "activity_id": ["ACT1", "ACT2", "ACT3", "ACT4"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # MOL1 has 3.0 diff -> keep earliest (2010, value=5.0)
        # MOL2 has 1.0 diff -> keep both
        self.assertEqual(len(result), 3)
        mol1_vals = result[result["connectivity"] == "MOL1"]["pchembl_value"].tolist()
        mol2_vals = result[result["connectivity"] == "MOL2"]["pchembl_value"].tolist()
        self.assertEqual(mol1_vals, ["5.00"])
        self.assertEqual(sorted(mol2_vals), ["6.00", "7.00"])

    def test_resolve_nan_years_excluded(self):
        """Test that measurements with nan years are handled gracefully."""
        from Capricho.analysis import resolve_annotation_errors

        df = pd.DataFrame(
            {
                "connectivity": ["MOL1", "MOL1"],
                "pchembl_value": ["5.00", "8.00"],
                "year": ["nan", "2015"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2"],
                "molecule_chembl_id": ["CHEMBL123", "CHEMBL123"],
                "activity_id": ["ACT1", "ACT2"],
            }
        )

        result = resolve_annotation_errors(df, strategy="first")

        # Should keep 2015 (the one with valid year) since nan is not comparable
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["year"], "2015")


class TestEnumValues(unittest.TestCase):
    """Tests for enum value correctness."""

    def test_processing_comment_enum_values(self):
        """Test that ProcessingComment enum has expected values."""
        self.assertEqual(ProcessingComment.CALCULATED_PCHEMBL.value, "Calculated pChEMBL")
        self.assertEqual(ProcessingComment.SALT_SOLVENT_REMOVED.value, "Salt/solvent removed")
        self.assertEqual(
            ProcessingComment.PCHEMBL_DUPLICATION_ACROSS_DOCUMENTS.value,
            "pChEMBL Duplication Across Documents",
        )

    def test_dropping_comment_enum_values(self):
        """Test that DroppingComment enum has expected values."""
        self.assertEqual(DroppingComment.DATA_VALIDITY_COMMENT.value, "Data Validity Comment Present")
        self.assertEqual(DroppingComment.POTENTIAL_DUPLICATE.value, "Potential Duplicate")
        self.assertEqual(DroppingComment.UNDEFINED_STEREOCHEMISTRY.value, "Undefined Stereochemistry")
        self.assertEqual(DroppingComment.MUTATION_KEYWORD.value, "Mutation keyword in assay description")
        self.assertEqual(DroppingComment.UNIT_ANNOTATION_ERROR.value, "Unit Annotation Error")

    def test_assay_size_enum_values_are_patterns(self):
        """Test that assay size enum values are patterns (without thresholds)."""
        self.assertEqual(DroppingComment.ASSAY_SIZE_TOO_SMALL.value, "Assay size <")
        self.assertEqual(DroppingComment.ASSAY_SIZE_TOO_LARGE.value, "Assay size >")

    def test_enum_string_inheritance(self):
        """Test that enums inherit from str for easy comparison."""
        self.assertIsInstance(ProcessingComment.CALCULATED_PCHEMBL, str)
        self.assertIsInstance(DroppingComment.POTENTIAL_DUPLICATE, str)


class TestLogScaleTransformation(unittest.TestCase):
    """Tests for log scale transformation with scale factor support."""

    def test_log_transform_with_scale_factor_produces_positive_values(self):
        """Test that log transformation with scale factor produces correct positive values.

        When data is in units of 10^-6 cm/s (stored as coefficients like 5 meaning
        5 × 10^-6 cm/s), the transformation -log10(value * 1e-6) should produce
        positive values around 5-6, not negative values.
        """
        import numpy as np

        from Capricho.analysis import plot_subset

        # Typical Caco-2 permeability values in 10^-6 cm/s units
        df = pd.DataFrame(
            {
                "standard_value_x": [5.0, 10.0, 50.0],  # 5, 10, 50 × 10^-6 cm/s
                "standard_value_y": [4.0, 12.0, 45.0],
            }
        )

        # Expected transformation: -log10(5 * 1e-6) = -log10(5e-6) = 5.3
        # For value=5 with scale_factor=1e-6:
        # -log10(5) + (-log10(1e-6)) = -0.7 + 6 = 5.3
        fig, ax = plot_subset(
            df,
            value_column="standard_value",
            log_transform=True,
            log_scale_factor=1e-6,
        )

        # Get the plotted data from the scatter collection
        scatter = ax.collections[0]
        offsets = scatter.get_offsets()

        # All values should be positive (in the 4-7 range for typical Caco-2 data)
        self.assertTrue((offsets[:, 0] > 0).all(), "X values should be positive after transformation")
        self.assertTrue((offsets[:, 1] > 0).all(), "Y values should be positive after transformation")

        # Verify specific values: -log10(5e-6) ≈ 5.30
        expected_x_first = -np.log10(5.0 * 1e-6)  # ≈ 5.30
        self.assertAlmostEqual(offsets[0, 0], expected_x_first, places=2)

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_log_transform_without_scale_factor_unchanged(self):
        """Test that log transformation without scale factor (default) works as before."""
        import numpy as np

        from Capricho.analysis import plot_subset

        df = pd.DataFrame(
            {
                "standard_value_x": [100.0, 1000.0],
                "standard_value_y": [200.0, 800.0],
            }
        )

        fig, ax = plot_subset(
            df,
            value_column="standard_value",
            log_transform=True,
            # No log_scale_factor specified - should default to 1.0
        )

        scatter = ax.collections[0]
        offsets = scatter.get_offsets()

        # With scale_factor=1.0 and epsilon=1e-9, result is essentially -log10(value)
        # For value=100: -log10(100 + 1e-9) ≈ -2.0
        expected_x_first = -np.log10(100.0 + 1e-9)
        self.assertAlmostEqual(offsets[0, 0], expected_x_first, places=2)

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestFormatTitleWithN(unittest.TestCase):
    """Tests for format_title_with_n function."""

    def test_appends_pair_count(self):
        """Test that the pairwise comparison count is appended to the title."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(format_title_with_n("Cleaned Ki Data", 383), "Cleaned Ki Data (n = 383 pairs)")

    def test_thousands_separator(self):
        """Test that large counts are formatted with a thousands separator."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(
            format_title_with_n("pChEMBL Duplication Across Documents", 3360),
            "pChEMBL Duplication Across Documents (n = 3,360 pairs)",
        )

    def test_single_pair_is_singular(self):
        """Test that a single comparison uses the singular noun."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(format_title_with_n("Salt/solvent removed", 1), "Salt/solvent removed (n = 1 pair)")

    def test_empty_title_reports_count_alone(self):
        """Test that an empty title yields the count without stray whitespace."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(format_title_with_n("", 2), "n = 2 pairs")

    def test_custom_separator_wraps_to_new_line(self):
        """Test that a newline separator keeps long panel titles off their neighbours."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(
            format_title_with_n("7. Insufficient assay overlap (min_overlap=5)", 6879, sep="\n"),
            "7. Insufficient assay overlap (min_overlap=5)\n(n = 6,879 pairs)",
        )

    def test_zero_pairs(self):
        """Test that an empty subset is reported as zero pairs."""
        from Capricho.analysis import format_title_with_n

        self.assertEqual(format_title_with_n("Empty Flag", 0), "Empty Flag (n = 0 pairs)")


class TestFormatMetricsText(unittest.TestCase):
    """Tests for format_metrics_text function."""

    def test_all_metrics_defined(self):
        """Test that defined metrics are rendered on separate lines."""
        from Capricho.analysis import format_metrics_text

        self.assertEqual(
            format_metrics_text(0.78, 0.85, 0.72),
            "$R^2: 0.78$\nSpearman $\\rho: 0.85$\nKendall $\\tau: 0.72$",
        )

    def test_undefined_metrics_are_omitted(self):
        """Test that a panel with undefined correlations renders no annotation.

        Two identical points have zero variance, so every metric is nan; printing
        "nan" would read as a plotting bug.
        """
        from Capricho.analysis import format_metrics_text

        self.assertEqual(format_metrics_text(np.nan, np.nan, np.nan), "")

    def test_partially_defined_metrics(self):
        """Test that defined metrics survive when others are undefined."""
        from Capricho.analysis import format_metrics_text

        self.assertEqual(
            format_metrics_text(np.nan, 0.85, 0.72), "Spearman $\\rho: 0.85$\nKendall $\\tau: 0.72$"
        )

    def test_negative_r2_is_kept(self):
        """Test that a negative R2 is a real value and must not be dropped."""
        from Capricho.analysis import format_metrics_text

        self.assertIn("$R^2: -0.79$", format_metrics_text(-0.79, 0.17, 0.15))


class TestSampleSizeInPlotTitles(unittest.TestCase):
    """Tests that plots report the number of pairwise comparisons in their titles."""

    @staticmethod
    def _exploded_fixture():
        """Build a minimal exploded subset with two distinguishable flags."""
        return pd.DataFrame(
            {
                "pchembl_value_x": ["7.0", "8.0", "6.5", "9.0"],
                "pchembl_value_y": ["7.2", "8.4", "6.1", "9.3"],
                "assay_chembl_id_x": ["CHEMBL1", "CHEMBL1", "CHEMBL3", "CHEMBL3"],
                "assay_chembl_id_y": ["CHEMBL2", "CHEMBL2", "CHEMBL4", "CHEMBL4"],
                "data_dropping_comment_x": [
                    "Data Validity Comment Present",
                    "Data Validity Comment Present",
                    "Undefined Stereochemistry",
                    "Undefined Stereochemistry",
                ],
                "data_dropping_comment_y": [
                    "Data Validity Comment Present",
                    "Data Validity Comment Present",
                    "Undefined Stereochemistry",
                    "Undefined Stereochemistry",
                ],
                "data_processing_comment_x": ["", "", "", ""],
                "data_processing_comment_y": ["", "", "", ""],
                "dropping_comment": [
                    "Data Validity Comment Present",
                    "Data Validity Comment Present",
                    "Undefined Stereochemistry",
                    "Undefined Stereochemistry",
                ],
                "processing_comment": ["", "", "", ""],
            }
        )

    def test_plot_subset_reports_sample_size(self):
        """Test that plot_subset appends the number of plotted pairs to the title."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_subset

        df = pd.DataFrame(
            {
                "pchembl_value_x": [7.0, 8.0, 6.5],
                "pchembl_value_y": [7.2, 8.4, 6.1],
            }
        )

        fig, ax = plot_subset(df, title="Cleaned Ki Data")
        self.assertEqual(ax.get_title(), "Cleaned Ki Data (n = 3 pairs)")
        plt.close(fig)

    def test_plot_subset_sample_size_can_be_disabled(self):
        """Test that show_n=False restores the bare title."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_subset

        df = pd.DataFrame(
            {
                "pchembl_value_x": [7.0, 8.0],
                "pchembl_value_y": [7.2, 8.4],
            }
        )

        fig, ax = plot_subset(df, title="Cleaned Ki Data", show_n=False)
        self.assertEqual(ax.get_title(), "Cleaned Ki Data")
        plt.close(fig)

    def test_multi_panel_reports_per_panel_sample_size(self):
        """Test that each panel title carries its own pair count."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import DroppingComment, plot_multi_panel_comparability

        fig, axs = plot_multi_panel_comparability(
            self._exploded_fixture(),
            [
                DroppingComment.DATA_VALIDITY_COMMENT.value,
                DroppingComment.UNDEFINED_STEREOCHEMISTRY.value,
            ],
            ncols=2,
        )

        titles = [ax.get_title() for ax in np.asarray(axs).flatten() if ax.get_visible()]
        self.assertEqual(titles[0], "1. Data Validity Comment Present\n(n = 2 pairs)")
        self.assertEqual(titles[1], "2. Undefined Stereochemistry\n(n = 2 pairs)")
        plt.close(fig)

    def test_scatter_drawn_above_reference_lines(self):
        """Test that points are drawn above the reference lines.

        Points lying exactly on y=x (e.g. cross-document duplicates) were hidden
        beneath the identity line, making populated panels look empty.
        """
        import matplotlib.pyplot as plt

        from Capricho.analysis import (
            DroppingComment,
            plot_multi_panel_comparability,
            plot_subset,
        )

        df = pd.DataFrame({"pchembl_value_x": [7.57, 8.0], "pchembl_value_y": [7.57, 8.0]})
        fig, ax = plot_subset(df, title="On identity line")
        line_zorders = [line.get_zorder() for line in ax.lines]
        self.assertTrue(
            all(ax.collections[0].get_zorder() > z for z in line_zorders),
            "Scatter must sit above every reference line",
        )
        plt.close(fig)

        fig, axs = plot_multi_panel_comparability(
            self._exploded_fixture(),
            [DroppingComment.DATA_VALIDITY_COMMENT.value],
            ncols=1,
        )
        panel = np.asarray(axs).flatten()[0]
        self.assertTrue(
            all(panel.collections[0].get_zorder() > line.get_zorder() for line in panel.lines),
            "Panel scatter must sit above every reference line",
        )
        plt.close(fig)

    def test_multi_panel_uses_saturated_colors(self):
        """Test that panels use tab10 colors, which have no washed-out light variants.

        tab20 alternates dark/light pairs, so every even-numbered panel was rendered
        in a light tint that reviewers found illegible.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        from Capricho.analysis import DroppingComment, plot_multi_panel_comparability

        fig, axs = plot_multi_panel_comparability(
            self._exploded_fixture(),
            [
                DroppingComment.DATA_VALIDITY_COMMENT.value,
                DroppingComment.UNDEFINED_STEREOCHEMISTRY.value,
            ],
            ncols=2,
        )

        tab10 = [tuple(c) for c in colormaps["tab10"].colors]
        axs_flat = np.asarray(axs).flatten()
        for idx in range(2):
            facecolor = tuple(axs_flat[idx].collections[0].get_facecolor()[0][:3])
            self.assertEqual(facecolor, tab10[idx], f"Panel {idx + 1} should use tab10 color {idx}")
        plt.close(fig)


class TestPlotCrossAssayCoverage(unittest.TestCase):
    """Tests that the coverage plot adapts to the number of datasets it is given."""

    @staticmethod
    def _coverage_fixture(n_datasets=2):
        """Coverage table with one low-overlap and one high-overlap dataset per repeat."""
        import pandas as pd

        rows = []
        for idx in range(n_datasets):
            overlap = 41.2 if idx % 2 else 5.6
            assay_overlap = 80.0 if idx % 2 else 7.5
            rows.append(
                {
                    "dataset": f"d{idx}",
                    "comparable_datapoints": 1234,
                    "aggregated_datapoints": 2995,
                    "datapoint_overlap_pct": overlap,
                    "assays_with_overlap": 80,
                    "represented_assays": 100,
                    "assay_overlap_pct": assay_overlap,
                }
            )
        return pd.DataFrame(rows)

    def test_draws_one_panel_per_denominator(self):
        """Test that the two overlap denominators get one panel each."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_cross_assay_coverage

        fig, axes = plot_cross_assay_coverage(self._coverage_fixture())

        self.assertEqual(len(axes), 2)
        # A full-width track plus the filled share, for each dataset.
        for ax in axes:
            self.assertEqual(len(ax.patches), 4)
        plt.close(fig)

    def test_labels_rename_the_axis_ticks(self):
        """Test that dataset names are shown, and mapped through `labels` when given."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_cross_assay_coverage

        fig, axes = plot_cross_assay_coverage(self._coverage_fixture(), labels={"d0": r"IC$_{50}$"})

        self.assertEqual([t.get_text() for t in axes[0].get_yticklabels()], [r"IC$_{50}$", "d1"])
        plt.close(fig)

    def test_height_grows_with_the_number_of_datasets(self):
        """Test that 14 datasets are not squeezed into the height used for 2."""
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_cross_assay_coverage

        fig_small, _ = plot_cross_assay_coverage(self._coverage_fixture(2))
        fig_large, _ = plot_cross_assay_coverage(self._coverage_fixture(14))

        small_height = fig_small.get_size_inches()[1]
        large_height = fig_large.get_size_inches()[1]
        self.assertGreater(large_height, small_height)
        # Row pitch must stay wide enough for the two-line in-bar label.
        self.assertGreater((large_height - small_height) / 12, 0.45)
        plt.close(fig_small)
        plt.close(fig_large)

    def test_color_can_be_set_per_dataset(self):
        """Test that a dict of colors is applied row by row, as case 3 does per direction."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        from Capricho.analysis import plot_cross_assay_coverage

        fig, axes = plot_cross_assay_coverage(
            self._coverage_fixture(), color={"d0": "darkorange", "d1": "purple"}
        )

        # The first two patches are the tracks; the filled bars follow in row order.
        filled = axes[0].patches[2:]
        self.assertEqual(filled[0].get_facecolor()[:3], mcolors.to_rgb("darkorange"))
        self.assertEqual(filled[1].get_facecolor()[:3], mcolors.to_rgb("purple"))
        plt.close(fig)

    def test_empty_coverage_is_an_error(self):
        """Test that an empty table is refused rather than drawn as a blank figure."""
        import pandas as pd

        from Capricho.analysis import plot_cross_assay_coverage

        with self.assertRaisesRegex(ValueError, "no rows"):
            plot_cross_assay_coverage(pd.DataFrame(columns=["dataset"]))

    def test_plots_what_summarize_curation_reports(self):
        """Test the whole seam, so the plot cannot drift from the table that feeds it.

        The fixture above hand-builds the wide table; only this test proves the columns
        `coverage_table` actually produces are the ones the panels read.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from Capricho.analysis import plot_cross_assay_coverage
        from Capricho.flag_report import coverage_table, summarize_curation

        aggregated = pd.DataFrame(
            {
                "assay_chembl_id": ["CHEMBL1|CHEMBL2", "CHEMBL1", "CHEMBL3"],
                "data_dropping_comment": ["", "", ""],
            }
        )
        coverage = coverage_table(summarize_curation("A2B", aggregated=aggregated))

        fig, axes = plot_cross_assay_coverage(coverage)

        # 1 of 3 datapoints is comparable, and 2 of 3 assay IDs overlap.
        self.assertIn("(1/3)", axes[0].texts[0].get_text())
        self.assertIn("(2/3)", axes[1].texts[0].get_text())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
