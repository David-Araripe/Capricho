import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from Capricho.core.binarization import (
    _max_confidence_score,
    binarize_aggregated_data,
    invert_relation_for_pchembl,
)
from Capricho.core.default_fields import DATA_DROPPING_COMMENT


class TestInvertRelationForPchembl(unittest.TestCase):
    def test_invert_less_than(self):
        self.assertEqual(invert_relation_for_pchembl("<"), ">")

    def test_invert_greater_than(self):
        self.assertEqual(invert_relation_for_pchembl(">"), "<")

    def test_invert_equals(self):
        self.assertEqual(invert_relation_for_pchembl("="), "=")

    def test_invert_less_than_equals(self):
        self.assertEqual(invert_relation_for_pchembl("<="), ">=")

    def test_invert_greater_than_equals(self):
        self.assertEqual(invert_relation_for_pchembl(">="), "<=")

    def test_invalid_relation(self):
        with self.assertRaises(ValueError):
            invert_relation_for_pchembl("invalid")

    def test_invert_approximately_equal(self):
        self.assertEqual(invert_relation_for_pchembl("~"), "~")

    def test_invert_much_greater_than(self):
        self.assertEqual(invert_relation_for_pchembl(">>"), "<<")

    def test_invert_much_less_than(self):
        self.assertEqual(invert_relation_for_pchembl("<<"), ">>")


class TestBinarizeAggregatedData(unittest.TestCase):
    def setUp(self):
        self.threshold = 6.0

    def test_basic_discrete_binarization(self):
        """Test basic binarization with only '=' relations"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 6.0],
                "standard_relation": ["=", "=", "="],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Check binary labels
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 >= 6.0
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # 5.0 < 6.0
        self.assertEqual(result.loc[2, "activity_binary"], 1)  # 6.0 >= 6.0 (edge case)

    def test_censored_active_binarization(self):
        """Test binarization with '<' (censored active) relations"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 6.0],
                "standard_relation": ["<", "<", "<"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # '<' means compound is MORE active (lower concentration)
        # If pchembl >= threshold, definitely active
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 >= 6.0 → active
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # 5.0 < 6.0 → might be inactive
        self.assertEqual(result.loc[2, "activity_binary"], 1)  # 6.0 >= 6.0 → active

    def test_censored_inactive_binarization(self):
        """Test binarization with '>' (censored inactive) relations"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 6.0],
                "standard_relation": [">", ">", ">"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # '>' means compound is LESS active (higher concentration)
        # If pchembl <= threshold, definitely inactive
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 > 6.0 → might be active
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # 5.0 <= 6.0 → inactive
        self.assertEqual(result.loc[2, "activity_binary"], 0)  # 6.0 <= 6.0 → inactive

    def test_mixed_relations_with_agreement(self):
        """Test mixed discrete and censored measurements that agree"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 8.0],
                "standard_relation": ["<", "="],  # censored active + discrete
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Both should be active
        self.assertEqual(result.loc[0, "activity_binary"], 1)
        self.assertEqual(result.loc[1, "activity_binary"], 1)

        # Should not have conflict flag
        if DATA_DROPPING_COMMENT in result.columns:
            self.assertTrue(
                result[DATA_DROPPING_COMMENT].isna().all()
                or not result[DATA_DROPPING_COMMENT].str.contains("Non-agreeing").any()
            )

    def test_mixed_relations_with_disagreement(self):
        """Test mixed discrete and censored measurements that disagree"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["<", "="],  # censored says active, discrete says inactive
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Should have conflict flag for both rows
        self.assertIn(DATA_DROPPING_COMMENT, result.columns)
        self.assertTrue(result[DATA_DROPPING_COMMENT].str.contains("Non-agreeing", na=False).all())

    def test_nan_values(self):
        """Test handling of NaN values in pchembl column"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, np.nan, 5.0],
                "standard_relation": ["=", "=", "="],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Check that NaN is preserved
        self.assertEqual(result.loc[0, "activity_binary"], 1)
        self.assertTrue(pd.isna(result.loc[1, "activity_binary"]))
        self.assertEqual(result.loc[2, "activity_binary"], 0)

    def test_missing_standard_relation_column(self):
        """Test that function handles missing standard_relation column"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Should assume '=' for all rows
        self.assertEqual(result.loc[0, "activity_binary"], 1)
        self.assertEqual(result.loc[1, "activity_binary"], 0)

    def test_custom_column_names(self):
        """Test using custom column names"""
        df = pd.DataFrame(
            {
                "my_compound_id": ["CONN1", "CONN2"],
                "my_target_id": ["TARGET1", "TARGET1"],
                "my_value": [7.0, 5.0],
                "my_relation": ["=", "="],
            }
        )

        result = binarize_aggregated_data(
            df,
            threshold=self.threshold,
            value_column="my_value",
            compound_id_col="my_compound_id",
            target_id_col="my_target_id",
            relation_col="my_relation",
            output_binary_col="my_binary",
        )

        self.assertIn("my_binary", result.columns)
        self.assertEqual(result.loc[0, "my_binary"], 1)
        self.assertEqual(result.loc[1, "my_binary"], 0)

    def test_different_threshold(self):
        """Test with non-default threshold"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 6.5, 6.0],
                "standard_relation": ["=", "=", "="],
            }
        )

        result = binarize_aggregated_data(df, threshold=6.5, compound_id_col="connectivity")

        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 >= 6.5
        self.assertEqual(result.loc[1, "activity_binary"], 1)  # 6.5 >= 6.5
        self.assertEqual(result.loc[2, "activity_binary"], 0)  # 6.0 < 6.5

    def test_median_column(self):
        """Test using median instead of mean"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "pchembl_value_median": [6.5, 5.5],
                "standard_relation": ["=", "="],
            }
        )

        result = binarize_aggregated_data(
            df, threshold=self.threshold, value_column="pchembl_value_median", compound_id_col="connectivity"
        )

        # Using median values now
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 6.5 >= 6.0
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # 5.5 < 6.0

    def test_multiple_compound_target_pairs(self):
        """Test with multiple compound-target pairs"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET2", "TARGET1", "TARGET2"],
                "pchembl_value_mean": [7.0, 5.0, 6.5, 5.5],
                "standard_relation": ["=", "=", "=", "="],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Each compound-target pair should be binarized independently
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # CONN1-TARGET1
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # CONN1-TARGET2
        self.assertEqual(result.loc[2, "activity_binary"], 1)  # CONN2-TARGET1
        self.assertEqual(result.loc[3, "activity_binary"], 0)  # CONN2-TARGET2

    def test_complex_mixed_relations(self):
        """Test complex scenario with multiple relations for same compound-target"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [5.5, 6.5, 7.5],
                "standard_relation": ["<", "=", ">"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # < at 5.5: might be inactive (below threshold)
        self.assertEqual(result.loc[0, "activity_binary"], 0)
        # = at 6.5: active (above threshold)
        self.assertEqual(result.loc[1, "activity_binary"], 1)
        # > at 7.5: might be active (above threshold)
        self.assertEqual(result.loc[2, "activity_binary"], 1)

    def test_approximately_equal_binarization(self):
        """Test binarization with '~' (approximately equal) relation"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3", "CONN4"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 6.5, 6.0, 5.5],
                "standard_relation": ["~", "~", "~", "~"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # ~ at 7.0: lower bound = 6.5 >= 6.0 → active
        self.assertEqual(result.loc[0, "activity_binary"], 1)
        # ~ at 6.5: lower bound = 6.0 >= 6.0 → active (edge case)
        self.assertEqual(result.loc[1, "activity_binary"], 1)
        # ~ at 6.0: lower bound = 5.5 < 6.0 → inactive
        self.assertEqual(result.loc[2, "activity_binary"], 0)
        # ~ at 5.5: lower bound = 5.0 < 6.0 → inactive
        self.assertEqual(result.loc[3, "activity_binary"], 0)

    def test_much_greater_than_binarization(self):
        """Test binarization with '>>' (much greater than) relation"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 6.0, 5.0],
                "standard_relation": [">>", ">>", ">>"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # >> means compound is MUCH LESS active (concentration >> reported)
        # Treat like '>': if pchembl <= threshold → inactive
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 > 6.0 → might be active
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # 6.0 <= 6.0 → inactive
        self.assertEqual(result.loc[2, "activity_binary"], 0)  # 5.0 < 6.0 → inactive

    def test_much_less_than_binarization(self):
        """Test binarization with '<<' (much less than) relation"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN2", "CONN3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 6.0, 5.0],
                "standard_relation": ["<<", "<<", "<<"],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # << means compound is MUCH MORE active (concentration << reported)
        # Treat like '<': if pchembl >= threshold → active
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # 7.0 >= 6.0 → active
        self.assertEqual(result.loc[1, "activity_binary"], 1)  # 6.0 >= 6.0 → active (edge case)
        self.assertEqual(result.loc[2, "activity_binary"], 0)  # 5.0 < 6.0 → might be inactive

    def test_mixed_with_approximate(self):
        """Test mixed '~' and '=' measurements for same compound-target"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [6.8, 7.0],
                "standard_relation": ["~", "="],
            }
        )

        result = binarize_aggregated_data(df, threshold=self.threshold, compound_id_col="connectivity")

        # Both should be active
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # ~ at 6.8: lower bound 6.3
        self.assertEqual(result.loc[1, "activity_binary"], 1)  # = at 7.0

    def test_different_mutants_not_flagged_when_compare_across_mutants_false(self):
        """Test that different mutants are NOT considered as disagreements when compare_across_mutants=False"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "mutation": ["Wild-type", "V600E"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", "="],
            }
        )

        result = binarize_aggregated_data(
            df, threshold=self.threshold, compare_across_mutants=False, compound_id_col="connectivity"
        )

        # Both should be binarized
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # Active on wild-type
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # Inactive on mutant

        # Should NOT have conflict flag because they are different mutants
        if DATA_DROPPING_COMMENT in result.columns:
            self.assertTrue(
                result[DATA_DROPPING_COMMENT].isna().all()
                or not result[DATA_DROPPING_COMMENT].str.contains("Non-agreeing", na=False).any()
            )

    def test_different_mutants_flagged_when_compare_across_mutants_true(self):
        """Test that different mutants ARE considered as disagreements when compare_across_mutants=True"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "mutation": ["Wild-type", "V600E"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", "="],
            }
        )

        result = binarize_aggregated_data(
            df, threshold=self.threshold, compare_across_mutants=True, compound_id_col="connectivity"
        )

        # Both should be binarized
        self.assertEqual(result.loc[0, "activity_binary"], 1)  # Active on wild-type
        self.assertEqual(result.loc[1, "activity_binary"], 0)  # Inactive on mutant

        # SHOULD have conflict flag because we're aggregating across mutants
        self.assertIn(DATA_DROPPING_COMMENT, result.columns)
        self.assertTrue(result[DATA_DROPPING_COMMENT].str.contains("Non-agreeing", na=False).all())

    def test_same_mutant_with_disagreement_always_flagged(self):
        """Test that disagreements on the same mutant are always flagged regardless of compare_across_mutants"""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "mutation": ["Wild-type", "Wild-type"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["<", "="],  # Censored says active, discrete says inactive
            }
        )

        # Test with compare_across_mutants=False
        result_false = binarize_aggregated_data(
            df, threshold=self.threshold, compare_across_mutants=False, compound_id_col="connectivity"
        )
        self.assertIn(DATA_DROPPING_COMMENT, result_false.columns)
        self.assertTrue(result_false[DATA_DROPPING_COMMENT].str.contains("Non-agreeing", na=False).all())

        # Test with compare_across_mutants=True (should have same result)
        result_true = binarize_aggregated_data(
            df, threshold=self.threshold, compare_across_mutants=True, compound_id_col="connectivity"
        )
        self.assertIn(DATA_DROPPING_COMMENT, result_true.columns)
        self.assertTrue(result_true[DATA_DROPPING_COMMENT].str.contains("Non-agreeing", na=False).all())


class TestMaxConfidenceScore(unittest.TestCase):
    def test_single_value(self):
        self.assertEqual(_max_confidence_score("9"), 9)

    def test_pipe_separated(self):
        self.assertEqual(_max_confidence_score("8|9"), 9)

    def test_nan_returns_zero(self):
        self.assertEqual(_max_confidence_score(np.nan), 0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(_max_confidence_score(""), 0)

    def test_multiple_values(self):
        self.assertEqual(_max_confidence_score("4|7|9|6"), 9)


class TestConflictResolution(unittest.TestCase):
    """Tests for conflict resolution strategies in binarize_aggregated_data."""

    def _make_conflict_df(self):
        """Helper: two rows for same compound-target that will conflict at threshold=6.0.

        Row 0: = at 7.0 → active
        Row 1: > at 5.0 → inactive (censored: pchembl <= threshold → inactive)
        """
        return pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", ">"],
                "confidence_score": ["9", "7"],
            }
        )

    def test_no_resolution_preserves_all_rows(self):
        """Default (no conflict_resolution) keeps all rows, just flags."""
        df = self._make_conflict_df()
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution=None, compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 2)

    def test_drop_removes_all_conflicting_rows(self):
        """strategy='drop' removes all rows for conflicting compound-target pairs."""
        df = self._make_conflict_df()
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="drop", compound_id_col="connectivity"
        )
        # Both rows for CONN1-TARGET1 should be dropped
        self.assertEqual(len(result), 0)

    def test_relation_keeps_equal_drops_censored(self):
        """strategy='relation' keeps '=' rows and drops censored rows."""
        df = self._make_conflict_df()
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["standard_relation"], "=")

    def test_relation_fallback_to_drop_when_no_equal(self):
        """strategy='relation' drops all if no '=' rows exist."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["<", ">"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 0)

    def test_confidence_keeps_highest_score(self):
        """strategy='confidence' keeps row with highest confidence_score."""
        df = self._make_conflict_df()  # scores "9" vs "7"
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="confidence", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["confidence_score"], "9")

    def test_confidence_fallback_on_tie(self):
        """strategy='confidence' drops all on tie."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", ">"],
                "confidence_score": ["9", "9"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="confidence", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 0)

    def test_confidence_missing_column_raises_error(self):
        """strategy='confidence' raises ValueError when confidence_score column is missing."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", ">"],
            }
        )
        with self.assertRaises(ValueError, msg="confidence_score"):
            binarize_aggregated_data(
                df, threshold=6.0, conflict_resolution="confidence", compound_id_col="connectivity"
            )

    def test_majority_keeps_majority_label(self):
        """strategy='majority' keeps rows matching the majority binary label."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 7.5, 5.0],
                "standard_relation": ["=", "=", ">"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        # 2 active vs 1 inactive → keep active rows
        self.assertEqual(len(result), 2)
        self.assertTrue((result["activity_binary"] == 1).all())

    def test_majority_tie_falls_back_to_drop(self):
        """strategy='majority' drops all on tie (row-based, no counts column)."""
        df = self._make_conflict_df()  # 1 active vs 1 inactive → tie
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 0)

    def test_majority_weights_by_measurement_count(self):
        """strategy='majority' uses pchembl_value_counts to weight votes."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "pchembl_value_counts": [174, 1],
                "standard_relation": ["=", ">"],
            }
        )
        # Row-based: 1 active vs 1 inactive → tie → drop all
        # Measurement-weighted: 174 active vs 1 inactive → active wins
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["activity_binary"], 1)
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 174)

    def test_majority_measurement_weighted_tie(self):
        """strategy='majority' drops all when measurement counts tie."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "pchembl_value_counts": [10, 10],
                "standard_relation": ["=", ">"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 0)

    def test_invalid_strategy_raises_error(self):
        """Unknown strategy name raises ValueError."""
        df = self._make_conflict_df()
        with self.assertRaises(ValueError, msg="unknown_strategy"):
            binarize_aggregated_data(
                df, threshold=6.0, conflict_resolution="unknown_strategy", compound_id_col="connectivity"
            )

    def test_non_conflicting_pairs_unaffected(self):
        """Conflict resolution only affects conflicting pairs; others stay intact."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 8.0],
                "standard_relation": ["=", ">", "="],
                "confidence_score": ["9", "7", "8"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="drop", compound_id_col="connectivity"
        )
        # CONN1-TARGET1 is conflicting → dropped; CONN2-TARGET1 is not → kept
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["connectivity"], "CONN2")


class TestMeasurementLevelMajority(unittest.TestCase):
    """Tests for measurement-level voting in majority conflict resolution.

    When the raw pchembl_value column (pipe-separated individual measurements) is present,
    the majority strategy should classify each individual measurement against the threshold
    instead of using row-level counts.
    """

    def test_majority_measurement_level_basic(self):
        """Individual measurements within a row each get a vote, not just the row mean."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["5.50|7.00", "5.00"],
                "pchembl_value_mean": [6.25, 5.0],
                "standard_relation": ["=", "="],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        # Row 1 mean=6.25 → active. Row 2 mean=5.0 → inactive. They conflict.
        # Measurement-level: 5.5(inactive) + 7.0(active) from Row 1, 5.0(inactive) from Row 2
        # 1 active vs 2 inactive → inactive wins → drop Row 1, keep Row 2
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["activity_binary"], 0)

    def test_majority_measurement_level_tie_drops_all(self):
        """Equal measurement-level counts → tie → drop all."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["7.00", "5.00"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", "="],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        # 1 active (7.0) vs 1 inactive (5.0) → tie → drop all
        self.assertEqual(len(result), 0)

    def test_majority_measurement_level_censored_votes(self):
        """Each identical value in a censored row counts as a separate vote."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["7.00", "5.00|5.00|5.00"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", ">"],
            }
        )
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        # = at 7.0: 1 active vote
        # > at 5.0|5.0|5.0: each 5.0 with > → 5.0 <= 6.0 → 3 inactive votes
        # 1 active vs 3 inactive → inactive wins → keep Row 2
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["activity_binary"], 0)


class TestPostResolutionDeduplication(unittest.TestCase):
    """Tests for deduplication and measurement filtering after conflict resolution.

    When a resolution strategy is active and the raw pchembl_value column is present,
    compound-target groups should be merged into one row with disagreeing individual
    measurements filtered out.
    """

    def test_dedup_merges_agreeing_rows_into_one(self):
        """Two rows that both resolve to active should be merged into one row."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value": ["7.00|7.50", "8.00", "5.00"],
                "pchembl_value_mean": [7.25, 8.0, 5.0],
                "standard_relation": ["=", "=", ">"],
            }
        )
        # Row 0 (=, active), Row 1 (=, active), Row 2 (>, inactive) → conflict
        # Majority: 3 active (7.0, 7.5, 8.0) vs 1 inactive (5.0) → active wins
        # After resolution: Rows 0 and 1 survive. Dedup merges them into 1 row.
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        conn1_rows = result[result["connectivity"] == "CONN1"]
        self.assertEqual(len(conn1_rows), 1)
        self.assertEqual(conn1_rows.iloc[0]["activity_binary"], 1)

    def test_dedup_filters_disagreeing_measurements(self):
        """Within a kept row, individual measurements below threshold should be filtered."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["5.50|6.50|7.00", "4.50"],
                "pchembl_value_mean": [6.3, 4.5],
                "standard_relation": ["=", ">"],
            }
        )
        # Row 0: mean=6.3 → active. Individual: 5.5(inactive), 6.5(active), 7.0(active)
        # Row 1: mean=4.5 → inactive. Individual: 4.5 with > → inactive
        # Measurement-level: 2 active vs 2 inactive → tie → drop all
        # Use 'relation' strategy instead: keeps "=" row, drops ">" row
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        # After dedup: the "=" row has 5.5 filtered out (inactive at threshold 6.0)
        raw_values = result.iloc[0]["pchembl_value"]
        self.assertNotIn("5.50", str(raw_values))
        self.assertIn("6.50", str(raw_values))
        self.assertIn("7.00", str(raw_values))

    def test_dedup_aligns_all_pipe_columns(self):
        """When filtering measurements by position, all pipe-separated columns are aligned."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["5.50|7.00", "4.50"],
                "pchembl_value_mean": [6.25, 4.5],
                "assay_chembl_id": ["ASSAY_A|ASSAY_B", "ASSAY_C"],
                "standard_relation": ["=", ">"],
            }
        )
        # 'relation' strategy: keep "=" row, drop ">" row
        # Within "=" row: 5.5(inactive), 7.0(active) → filter 5.5, keep 7.0
        # assay_chembl_id should also filter: drop ASSAY_A (pos 0), keep ASSAY_B (pos 1)
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        assays = str(result.iloc[0]["assay_chembl_id"])
        self.assertNotIn("ASSAY_A", assays)
        self.assertIn("ASSAY_B", assays)

    def test_dedup_recalculates_stats(self):
        """Stats (mean, counts) should be recalculated from kept measurements only."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["7.00|8.00", "5.00"],
                "pchembl_value_mean": [7.5, 5.0],
                "pchembl_value_counts": [2, 1],
                "standard_relation": ["=", ">"],
            }
        )
        # 'relation' strategy: keep "=" row (active), drop ">" row
        # After dedup: both 7.0 and 8.0 agree (active), counts should be 2
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["pchembl_value_counts"], 2)

    def test_dedup_standard_relation_becomes_pipe_separated(self):
        """After merging rows with different relations, standard_relation is pipe-separated."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value": ["7.00", "8.00", "4.50"],
                "pchembl_value_mean": [7.0, 8.0, 4.5],
                "standard_relation": ["=", "<", ">"],
            }
        )
        # Row 0: = at 7.0 → active. Row 1: < at 8.0 → active. Row 2: > at 4.5 → inactive.
        # Majority: 2 active vs 1 inactive → active wins
        # After dedup: Rows 0 and 1 merged, standard_relation should be "=|<"
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="majority", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        relation = str(result.iloc[0]["standard_relation"])
        self.assertIn("=", relation)
        self.assertIn("<", relation)

    def test_dedup_single_row_still_filtered(self):
        """A single remaining row after resolution should still have disagreeing measurements filtered."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": ["5.50|7.00|8.00", "4.50"],
                "pchembl_value_mean": [6.8, 4.5],
                "standard_relation": ["=", ">"],
            }
        )
        # 'relation' strategy: keep "=" row, drop ">" row. One row remains.
        # Within that row: 5.5(inactive), 7.0(active), 8.0(active)
        # Filter 5.5, keep 7.0 and 8.0
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="relation", compound_id_col="connectivity"
        )
        self.assertEqual(len(result), 1)
        raw_values = str(result.iloc[0]["pchembl_value"])
        self.assertNotIn("5.50", raw_values)
        self.assertIn("7.00", raw_values)
        self.assertIn("8.00", raw_values)

    def test_dedup_non_conflicting_pairs_unchanged(self):
        """Groups that had no conflicts should not be modified by deduplication."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value": ["7.00", "5.00", "8.00"],
                "pchembl_value_mean": [7.0, 5.0, 8.0],
                "standard_relation": ["=", ">", "="],
            }
        )
        # CONN1-TARGET1 conflicts (active vs inactive). CONN2-TARGET1 has 1 row, no conflict.
        result = binarize_aggregated_data(
            df, threshold=6.0, conflict_resolution="drop", compound_id_col="connectivity"
        )
        # CONN1 rows dropped (conflict). CONN2 row untouched.
        conn2_rows = result[result["connectivity"] == "CONN2"]
        self.assertEqual(len(conn2_rows), 1)
        # The raw value should be preserved as-is (single value, no filtering needed)
        self.assertEqual(str(conn2_rows.iloc[0]["pchembl_value"]), "8.00")


class TestConflictReportEnhancements(unittest.TestCase):
    """Tests for severity, recommendation, and summary stats in conflict reports."""

    def _binarize_with_report(self, df, threshold=6.0, conflict_resolution=None):
        """Helper that binarizes and returns parsed conflict report JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_path = f.name

        binarize_aggregated_data(
            df,
            threshold=threshold,
            conflict_report_path=report_path,
            conflict_resolution=conflict_resolution,
            compound_id_col="connectivity",
        )

        with open(report_path) as f:
            return json.load(f)

    def test_severity_classification_low(self):
        """Spread < 1.0 → severity 'low'."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [6.3, 5.8],
                "standard_relation": ["=", ">"],
            }
        )
        report = self._binarize_with_report(df)
        conflict = report["conflicts"][0]
        self.assertEqual(conflict["severity"]["classification"], "low")
        self.assertAlmostEqual(conflict["severity"]["measurement_spread"], 0.5, places=1)

    def test_severity_classification_medium(self):
        """Spread between 1.0 and 2.0 → severity 'medium'."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.5],
                "standard_relation": ["=", ">"],
            }
        )
        report = self._binarize_with_report(df)
        conflict = report["conflicts"][0]
        self.assertEqual(conflict["severity"]["classification"], "medium")

    def test_severity_classification_high(self):
        """Spread > 2.0 → severity 'high'."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [8.5, 5.0],
                "standard_relation": ["=", ">"],
            }
        )
        report = self._binarize_with_report(df)
        conflict = report["conflicts"][0]
        self.assertEqual(conflict["severity"]["classification"], "high")
        self.assertAlmostEqual(conflict["severity"]["measurement_spread"], 3.5, places=1)

    def test_severity_max_distance_from_threshold(self):
        """max_distance_from_threshold is correct."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [8.0, 5.0],
                "standard_relation": ["=", ">"],
            }
        )
        report = self._binarize_with_report(df, threshold=6.0)
        severity = report["conflicts"][0]["severity"]
        # max(|8.0-6.0|, |5.0-6.0|) = max(2.0, 1.0) = 2.0
        self.assertAlmostEqual(severity["max_distance_from_threshold"], 2.0, places=1)

    def test_recommendation_exact_vs_censored(self):
        """When conflict has exact vs censored, recommendation mentions exact reliability."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", ">"],
            }
        )
        report = self._binarize_with_report(df)
        conflict = report["conflicts"][0]
        self.assertIn("exact", conflict["recommendation"].lower())
        self.assertIn("reliable", conflict["recommendation"].lower())

    def test_recommendation_same_type(self):
        """When all measurements have same relation type, recommend manual review."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0],
                "standard_relation": ["=", "="],
            }
        )
        report = self._binarize_with_report(df)
        conflict = report["conflicts"][0]
        self.assertIn("manual review", conflict["recommendation"].lower())

    def test_summary_stats_include_mcc_and_counts(self):
        """Summary includes total_rows, active_count, inactive_count, mcc."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 8.0],
                "standard_relation": ["=", ">", "="],
            }
        )
        report = self._binarize_with_report(df)
        summary = report["summary"]
        self.assertIn("total_rows", summary)
        self.assertIn("active_count", summary)
        self.assertIn("inactive_count", summary)
        self.assertIn("mcc", summary)
        self.assertEqual(summary["total_rows"], 3)

    def test_conflict_patterns_counting(self):
        """Summary has conflict_patterns with exact_vs_censored and censored_vs_censored counts."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN2", "CONN2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 5.0, 7.0, 5.0],
                "standard_relation": ["=", ">", "<", ">"],
            }
        )
        report = self._binarize_with_report(df)
        patterns = report["summary"]["conflict_patterns"]
        self.assertIn("exact_vs_censored", patterns)
        self.assertIn("censored_vs_censored", patterns)
        # CONN1: = vs > → exact_vs_censored; CONN2: < vs > → censored_vs_censored
        self.assertEqual(patterns["exact_vs_censored"], 1)
        self.assertEqual(patterns["censored_vs_censored"], 1)

    def test_resolution_info_in_conflict_report(self):
        """When a strategy is active, each conflict has resolution info and summary has resolution_summary."""
        df = pd.DataFrame(
            {
                "connectivity": ["CONN1", "CONN1", "CONN1"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value_mean": [7.0, 7.5, 5.0],
                "standard_relation": ["=", "=", ">"],
            }
        )
        report = self._binarize_with_report(df, conflict_resolution="majority")
        # Check conflict-level resolution info
        conflict = report["conflicts"][0]
        self.assertIn("resolution", conflict)
        self.assertEqual(conflict["resolution"]["strategy"], "majority")
        self.assertIn("rows_kept", conflict["resolution"])
        self.assertIn("rows_dropped", conflict["resolution"])

        # Check summary-level resolution_summary
        summary = report["summary"]
        self.assertIn("resolution_summary", summary)
        self.assertEqual(summary["resolution_summary"]["strategy"], "majority")


if __name__ == "__main__":
    unittest.main()
