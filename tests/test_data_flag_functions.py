"""Tests for data_flag_functions module."""

import unittest

import pandas as pd

from Capricho.analysis import DroppingComment, get_all_comments
from Capricho.chembl.data_flag_functions import (
    REVIEW_ACTIVITY_COMMENTS,
    flag_censored_activity_comment,
    flag_incompatible_units,
    flag_insufficient_assay_overlap,
    flag_inter_document_duplication,
    flag_missing_document_date,
)
from Capricho.core.default_fields import DATA_DROPPING_COMMENT

FLAG = DroppingComment.ACTIVITY_COMMENT_REVIEW.value


class TestFlagActivityCommentReview(unittest.TestCase):
    """The source relation is preserved and the comment/relation combination is flagged."""

    def _row(self, **overrides):
        row = {
            "molecule_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "standard_value": 10000.0,
            "pchembl_value": 5.0,
            "standard_relation": "=",
            "activity_comment": "Not Active",
            "data_dropping_comment": None,
            "data_processing_comment": None,
        }
        row.update(overrides)
        return pd.DataFrame({k: [v] for k, v in row.items()})

    def assert_flagged(self, result, idx=0):
        self.assertIn(FLAG, str(result.loc[idx, DATA_DROPPING_COMMENT]))

    def assert_not_flagged(self, result, idx=0):
        comment = result.loc[idx, DATA_DROPPING_COMMENT]
        self.assertNotIn(FLAG, "" if pd.isna(comment) else str(comment))

    def test_recognized_phrases_are_flagged_without_modifying_source_data(self):
        phrases = [*REVIEW_ACTIVITY_COMMENTS, "INCONCLUSIVE", "InAcTiVe at 10 uM"]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = flag_censored_activity_comment(
                    self._row(activity_comment=phrase, standard_value=12345.0)
                )

                self.assertEqual(result.loc[0, "standard_relation"], "=")
                self.assertEqual(result.loc[0, "standard_value"], 12345.0)
                self.assertTrue(pd.isna(result.loc[0, "data_processing_comment"]))
                self.assert_flagged(result)

    def test_flagging_is_independent_of_the_endpoint(self):
        for standard_type in ["IC50", "Ki", "Potency", "Inhibition", "Residual activity", None]:
            with self.subTest(standard_type=standard_type):
                df = self._row(standard_type=standard_type)
                if standard_type is None:
                    df = df.drop(columns=["standard_type"])

                result = flag_censored_activity_comment(df)

                self.assertEqual(result.loc[0, "standard_relation"], "=")
                self.assert_flagged(result)

    def test_unrecognized_comments_are_not_flagged(self):
        comments = [
            None,
            "",
            "Active",
            "Highly active",
            "Compound bound to the standard reference",
            "Not determinedly conclusive",
            "Inactivestate mutant",
            "ND",
            "N.D.",
            "nd at 10 uM",
            "Ligand efficiency reported",
            "Standard compound",
            "Bound to target",
            "No inhibition found",
            "Interpretation: Specific Binding",
            "See Activity_Supp For Individual Animal Data",
        ]
        for comment in comments:
            with self.subTest(comment=comment):
                result = flag_censored_activity_comment(self._row(activity_comment=comment))

                self.assertEqual(result.loc[0, "standard_relation"], "=")
                self.assert_not_flagged(result)

    def test_all_null_non_string_comment_column_is_accepted(self):
        df = self._row(activity_comment=float("nan"))
        self.assertTrue(pd.api.types.is_float_dtype(df["activity_comment"]))

        result = flag_censored_activity_comment(df)

        self.assertTrue(result["activity_comment"].isna().all())
        self.assert_not_flagged(result)

    def test_non_exact_relations_are_not_flagged(self):
        """Rows already carrying a bound are not an exact-relation conflict."""
        for relation in ["<", ">", "<=", ">="]:
            with self.subTest(relation=relation):
                result = flag_censored_activity_comment(self._row(standard_relation=relation))

                self.assertEqual(result.loc[0, "standard_relation"], relation)
                self.assert_not_flagged(result)

    def test_existing_dropping_comment_is_appended_to(self):
        """An existing quality flag is preserved rather than overwritten."""
        result = flag_censored_activity_comment(self._row(data_dropping_comment="Potential Duplicate"))

        comment = str(result.loc[0, DATA_DROPPING_COMMENT])
        self.assertIn("Potential Duplicate", comment)
        self.assertIn(FLAG, comment)

    def test_flag_is_registered_as_a_dropping_comment(self):
        """The flag is a first-class quality flag, so summaries and prepare can see it."""
        self.assertIn(FLAG, get_all_comments())

    def test_missing_required_columns_leave_the_frame_unchanged(self):
        for column in ["activity_comment", "standard_relation"]:
            with self.subTest(column=column):
                df = self._row().drop(columns=[column])

                result = flag_censored_activity_comment(df)

                self.assertEqual(result.to_dict(), df.to_dict())


class TestActivityCommentReviewRealChEMBLRows(unittest.TestCase):
    """Regression fixture of traceable ChEMBL 36 activities."""

    FIXTURE = [
        # activity_id, assay_chembl_id, standard_type, relation, value, comment
        (1230176, "CHEMBL815031", "Inhibition", "=", None, "Not Active"),
        (1233640, "CHEMBL816326", "Inhibition", "=", None, "Not Active"),
        (3607895, "CHEMBL1614478", "IC50", "=", 11403.0, "Not Active"),
        (3607918, "CHEMBL1614478", "IC50", "=", 20942.0, "Not Active"),
        (3630349, "CHEMBL1614542", "EC50", "=", 64450.0, "Not Active"),
        (3630376, "CHEMBL1614542", "EC50", "=", 10780.0, "Not Active"),
        (5538426, "CHEMBL1738125", "Ki", "=", 10000.0, "inactive"),
        (10915394, "CHEMBL2040657", "Inhibition", "=", None, "inactive"),
        (14249048, "CHEMBL3215220", "Ki", "=", 21100.0, "inactive"),
    ]

    UNRELATED = [
        (900000001, "CHEMBL999901", "IC50", "=", 50.0, "Compound bound to the standard reference"),
        (900000002, "CHEMBL999902", "IC50", "=", 75.0, "Active"),
    ]

    def build_df(self, rows):
        return pd.DataFrame(
            {
                "activity_id": [r[0] for r in rows],
                "assay_chembl_id": [r[1] for r in rows],
                "standard_type": [r[2] for r in rows],
                "standard_relation": [r[3] for r in rows],
                "standard_value": [r[4] for r in rows],
                "activity_comment": [r[5] for r in rows],
                "data_dropping_comment": [None] * len(rows),
            }
        )

    def test_recognized_comments_are_flagged_without_changing_relations_or_values(self):
        result = flag_censored_activity_comment(self.build_df(self.FIXTURE))

        self.assertListEqual(list(result["standard_relation"]), ["="] * len(self.FIXTURE))
        for _, row in result.iterrows():
            with self.subTest(activity_id=row["activity_id"]):
                self.assertIn(FLAG, str(row[DATA_DROPPING_COMMENT]))

        same_assay = result[result["assay_chembl_id"] == "CHEMBL1614478"]
        self.assertListEqual(sorted(same_assay["standard_value"]), [11403.0, 20942.0])

    def test_unrelated_free_text_rows_are_unchanged(self):
        """Rows whose comment says nothing about inactivity stay unflagged."""
        df = self.build_df(self.UNRELATED)

        result = flag_censored_activity_comment(df)

        self.assertListEqual(list(result["standard_relation"]), ["=", "="])
        for _, row in result.iterrows():
            with self.subTest(activity_id=row["activity_id"]):
                comment = row[DATA_DROPPING_COMMENT]
                self.assertNotIn(FLAG, "" if pd.isna(comment) else str(comment))


class TestFlagInterDocumentDuplication(unittest.TestCase):
    """Validates inter-document duplication detection for discrete measurements."""

    def test_only_flags_discrete_measurements(self):
        """Test that only discrete measurements (standard_relation='=') are flagged as duplicates."""
        test_data = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1"] * 6,
                "standard_smiles": ["CCO"] * 6,
                "canonical_smiles": ["CCO"] * 6,
                "pchembl_value": [6.0, 6.0, 6.0, 6.0, 5.0, 5.0],
                "standard_relation": ["=", "=", "<", "<", "=", "="],
                "target_chembl_id": ["TARGET1"] * 6,
                "mutation": ["WT"] * 6,
                "target_organism": ["Homo sapiens"] * 6,
                "document_chembl_id": ["DOC1", "DOC2", "DOC3", "DOC4", "DOC5", "DOC6"],
                "data_processing_comment": [""] * 6,
            }
        )

        result = flag_inter_document_duplication(test_data)

        # Only rows 0 and 1 (discrete measurements with pchembl=6.0, relation='=') should be flagged
        # Rows 2 and 3 (censored measurements with pchembl=6.0, relation='<') should NOT be flagged
        # Rows 4 and 5 (discrete measurements with pchembl=5.0, relation='=') should be flagged
        flagged_mask = result["data_processing_comment"].str.contains(
            "pChEMBL Duplication Across Documents", na=False
        )

        # Check that rows 0, 1, 4, 5 are flagged (discrete measurements)
        self.assertTrue(flagged_mask.iloc[0], "Row 0 should be flagged (discrete, pchembl=6.0)")
        self.assertTrue(flagged_mask.iloc[1], "Row 1 should be flagged (discrete, pchembl=6.0)")
        self.assertTrue(flagged_mask.iloc[4], "Row 4 should be flagged (discrete, pchembl=5.0)")
        self.assertTrue(flagged_mask.iloc[5], "Row 5 should be flagged (discrete, pchembl=5.0)")

        # Check that rows 2 and 3 are NOT flagged (censored measurements)
        self.assertFalse(flagged_mask.iloc[2], "Row 2 should NOT be flagged (censored, relation='<')")
        self.assertFalse(flagged_mask.iloc[3], "Row 3 should NOT be flagged (censored, relation='<')")

    def test_no_discrete_measurements(self):
        """Test that when all measurements are censored, nothing is flagged."""
        test_data = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1"] * 4,
                "standard_smiles": ["CCO"] * 4,
                "canonical_smiles": ["CCO"] * 4,
                "pchembl_value": [6.0, 6.0, 5.0, 5.0],
                "standard_relation": ["<", "<", ">", ">"],
                "target_chembl_id": ["TARGET1"] * 4,
                "mutation": ["WT"] * 4,
                "target_organism": ["Homo sapiens"] * 4,
                "document_chembl_id": ["DOC1", "DOC2", "DOC3", "DOC4"],
                "data_processing_comment": [""] * 4,
            }
        )

        result = flag_inter_document_duplication(test_data)

        # No rows should be flagged since all are censored
        flagged_mask = result["data_processing_comment"].str.contains(
            "pChEMBL Duplication Across Documents", na=False
        )
        self.assertFalse(flagged_mask.any(), "No censored measurements should be flagged")

    def test_missing_standard_relation_column(self):
        """Test that function handles missing standard_relation column gracefully."""
        test_data = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1"] * 2,
                "standard_smiles": ["CCO"] * 2,
                "canonical_smiles": ["CCO"] * 2,
                "pchembl_value": [6.0, 6.0],
                "target_chembl_id": ["TARGET1"] * 2,
                "mutation": ["WT"] * 2,
                "target_organism": ["Homo sapiens"] * 2,
                "document_chembl_id": ["DOC1", "DOC2"],
                "data_processing_comment": [""] * 2,
            }
        )

        result = flag_inter_document_duplication(test_data)

        # Function should return dataframe unchanged
        self.assertEqual(len(result), 2)
        self.assertTrue(result["data_processing_comment"].str.strip().eq("").all())


class TestFlagMissingDocumentDate(unittest.TestCase):
    """Tests for flag_missing_document_date function."""

    def test_flag_missing_year(self):
        """Test that activities with missing year are flagged in processing comment."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "year": [2020, None, 2021],
                "pchembl_value": [6.0, 6.5, 7.0],
                "data_dropping_comment": [None, None, None],
            }
        )

        result = flag_missing_document_date(df)

        # Check that only row 1 (with None year) is flagged
        self.assertFalse(
            "Missing document date" in str(result.loc[0, "data_dropping_comment"]),
            "Row 0 should NOT be flagged (has year)",
        )
        self.assertTrue(
            "Missing document date" in str(result.loc[1, "data_dropping_comment"]),
            "Row 1 should be flagged (missing year)",
        )
        self.assertFalse(
            "Missing document date" in str(result.loc[2, "data_dropping_comment"]),
            "Row 2 should NOT be flagged (has year)",
        )

    def test_all_have_year(self):
        """Test that when all activities have year, nothing is flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "year": [2019, 2020, 2021],
                "pchembl_value": [6.0, 6.5, 7.0],
                "data_processing_comment": [None, None, None],
            }
        )

        result = flag_missing_document_date(df)

        # No rows should have "Missing document date" in processing comment
        for idx in range(len(result)):
            comment = result.loc[idx, "data_processing_comment"]
            self.assertFalse(
                comment and "Missing document date" in str(comment),
                f"Row {idx} should NOT be flagged (has year)",
            )

    def test_all_missing_year(self):
        """Test that when all activities lack year, all are flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "year": [None, None, None],
                "pchembl_value": [6.0, 6.5, 7.0],
                "data_dropping_comment": [None, None, None],
            }
        )

        result = flag_missing_document_date(df)

        # All rows should be flagged
        for idx in range(len(result)):
            self.assertTrue(
                "Missing document date" in str(result.loc[idx, "data_dropping_comment"]),
                f"Row {idx} should be flagged (missing year)",
            )

    def test_missing_year_column(self):
        """Test that function handles missing year column gracefully."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "pchembl_value": [6.0, 6.5],
                "data_processing_comment": [None, None],
            }
        )

        result = flag_missing_document_date(df)

        # Should return DataFrame unchanged
        self.assertEqual(len(result), 2)
        self.assertTrue("year" not in result.columns)

    def test_preserves_existing_comments(self):
        """Test that existing processing comments are preserved when flagging."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "year": [None, None],
                "pchembl_value": [6.0, 6.5],
                "data_dropping_comment": ["Existing comment", None],
            }
        )

        result = flag_missing_document_date(df)

        # Check that existing comment is preserved and new flag is added
        self.assertIn("Existing comment", result.loc[0, "data_dropping_comment"])
        self.assertIn("Missing document date", result.loc[0, "data_dropping_comment"])
        self.assertIn("Missing document date", result.loc[1, "data_dropping_comment"])


class TestFlagIncompatibleUnits(unittest.TestCase):
    """Tests for flag_incompatible_units function."""

    def test_flag_incompatible_units(self):
        """Test that activities with non-convertible units are flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
                "standard_units": ["nM", "%", "µM", "ug.mL-1"],
                "standard_value": [100.0, 50.0, 10.0, 5.0],
                "pchembl_value": [7.0, None, 8.0, None],
                "data_dropping_comment": [None, None, None, None],
            }
        )

        result = flag_incompatible_units(df)

        # Check that rows with "%" and "ug.mL-1" are flagged, but "nM" and "µM" are not
        self.assertFalse(
            "Incompatible units" in str(result.loc[0, "data_dropping_comment"]),
            "Row 0 (nM) should NOT be flagged",
        )
        self.assertTrue(
            "Incompatible units" in str(result.loc[1, "data_dropping_comment"]),
            "Row 1 (%) should be flagged",
        )
        self.assertFalse(
            "Incompatible units" in str(result.loc[2, "data_dropping_comment"]),
            "Row 2 (µM) should NOT be flagged",
        )
        self.assertTrue(
            "Incompatible units" in str(result.loc[3, "data_dropping_comment"]),
            "Row 3 (L) should be flagged",
        )

    def test_all_compatible_units(self):
        """Test that when all units are compatible, nothing is flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "standard_units": ["nM", "uM", "mM"],
                "standard_value": [100.0, 10.0, 1.0],
                "pchembl_value": [7.0, 8.0, 6.0],
                "data_dropping_comment": [None, None, None],
            }
        )

        result = flag_incompatible_units(df)

        # No rows should be flagged
        for idx in range(len(result)):
            comment = result.loc[idx, "data_dropping_comment"]
            self.assertFalse(
                comment and "Incompatible units" in str(comment),
                f"Row {idx} should NOT be flagged (compatible unit)",
            )

    def test_null_units_not_flagged(self):
        """Test that null/NA units are not flagged (they're handled separately)."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "standard_units": [None, "nM"],
                "standard_value": [7.5, 100.0],
                "pchembl_value": [7.5, 7.0],
                "data_dropping_comment": [None, None],
            }
        )

        result = flag_incompatible_units(df)

        # Null units should not be flagged
        self.assertFalse(
            result.loc[0, "data_dropping_comment"]
            and "Incompatible units" in str(result.loc[0, "data_dropping_comment"]),
            "Row 0 (null unit) should NOT be flagged",
        )

    def test_missing_standard_units_column(self):
        """Test that function handles missing standard_units column gracefully."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1"],
                "pchembl_value": [6.0],
                "data_dropping_comment": [None],
            }
        )

        result = flag_incompatible_units(df)

        # Should return DataFrame unchanged
        self.assertEqual(len(result), 1)


class TestFlagInsufficientAssayOverlap(unittest.TestCase):
    """Tests for flag_insufficient_assay_overlap function with and without metadata matching."""

    def test_basic_overlap_filtering(self):
        """Test that assays without sufficient overlap are flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["MOL1", "MOL2", "MOL3", "MOL4", "MOL5"],
                "assay_chembl_id": ["ASSAY1", "ASSAY1", "ASSAY2", "ASSAY2", "ASSAY2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "document_chembl_id": ["DOC1", "DOC1", "DOC1", "DOC1", "DOC1"],
                "pchembl_value": [6.0, 6.5, 7.0, 7.5, 8.0],
                "data_dropping_comment": [None, None, None, None, None],
            }
        )

        # ASSAY1 has MOL1, MOL2 (2 compounds)
        # ASSAY2 has MOL3, MOL4, MOL5 (3 compounds)
        # Overlap between them: 0 compounds
        # With min_overlap=2, both assays should be flagged

        result = flag_insufficient_assay_overlap(df, min_overlap=2)

        # All activities should be flagged since no assay pair meets min_overlap
        flagged_mask = result["data_dropping_comment"].str.contains("Insufficient assay overlap", na=False)
        self.assertTrue(flagged_mask.all(), "All activities should be flagged when overlap is insufficient")

    def test_sufficient_overlap_not_flagged(self):
        """Test that assays with sufficient overlap are not flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["MOL1", "MOL2", "MOL1", "MOL2", "MOL3"],
                "assay_chembl_id": ["ASSAY1", "ASSAY1", "ASSAY2", "ASSAY2", "ASSAY2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "pchembl_value": [6.0, 6.5, 6.1, 6.6, 7.0],
                "data_dropping_comment": [None, None, None, None, None],
            }
        )

        # ASSAY1 has MOL1, MOL2
        # ASSAY2 has MOL1, MOL2, MOL3
        # Overlap: 2 compounds (MOL1, MOL2)
        # With min_overlap=2, both assays should NOT be flagged

        result = flag_insufficient_assay_overlap(df, min_overlap=2)

        # No activities should be flagged
        flagged_mask = result["data_dropping_comment"].str.contains("Insufficient assay overlap", na=False)
        self.assertFalse(flagged_mask.any(), "No activities should be flagged when overlap is sufficient")

    def test_multiple_targets_handled_independently(self):
        """Test that overlap checking is done independently for each target."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["MOL1", "MOL2", "MOL1", "MOL2"],
                "assay_chembl_id": ["ASSAY1", "ASSAY1", "ASSAY2", "ASSAY2"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET2", "TARGET2"],  # Different targets
                "pchembl_value": [6.0, 6.5, 7.0, 7.5],
                "data_dropping_comment": [None, None, None, None],
            }
        )

        # TARGET1: ASSAY1 has MOL1, MOL2 (only one assay, can't form pair)
        # TARGET2: ASSAY2 has MOL1, MOL2 (only one assay, can't form pair)
        # With min_overlap=2, nothing should be flagged (single assays per target)

        result = flag_insufficient_assay_overlap(df, min_overlap=2)

        # No activities should be flagged (no pairs to compare)
        flagged_mask = result["data_dropping_comment"].str.contains("Insufficient assay overlap", na=False)
        self.assertFalse(
            flagged_mask.any(), "No activities should be flagged when each target has only one assay"
        )

    def test_min_overlap_zero_skips_filtering(self):
        """Test that min_overlap=0 skips the filtering entirely."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["MOL1", "MOL2"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2"],
                "target_chembl_id": ["TARGET1", "TARGET1"],
                "pchembl_value": [6.0, 6.5],
                "data_dropping_comment": [None, None],
            }
        )

        result = flag_insufficient_assay_overlap(df, min_overlap=0)

        # No activities should be flagged
        flagged_mask = result["data_dropping_comment"].str.contains("Insufficient assay overlap", na=False)
        self.assertFalse(flagged_mask.any(), "min_overlap=0 should skip all filtering")

    def test_skips_size_flagged_assays_goldilocks(self):
        """Test that assays already flagged for size issues are skipped in overlap checking (goldilocks approach)."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["MOL1", "MOL2", "MOL3", "MOL4", "MOL5", "MOL6"],
                "assay_chembl_id": ["ASSAY1", "ASSAY1", "ASSAY2", "ASSAY2", "ASSAY3", "ASSAY3"],
                "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1", "TARGET1", "TARGET1", "TARGET1"],
                "document_chembl_id": ["DOC1", "DOC1", "DOC2", "DOC2", "DOC3", "DOC3"],
                "pchembl_value": [6.0, 6.5, 6.0, 7.0, 6.0, 8.0],
                "data_dropping_comment": [
                    "Assay size < 20",
                    "Assay size < 20",
                    None,
                    None,
                    None,
                    None,
                ],
            }
        )

        # ASSAY1 is already flagged for size (has MOL1, MOL2)
        # ASSAY2 has MOL3, MOL4
        # ASSAY3 has MOL5, MOL6
        # ASSAY1 shares MOL1 with no one, but it's already size-flagged so should be skipped
        # ASSAY2 and ASSAY3 have no overlap, so both should be flagged for insufficient overlap

        result = flag_insufficient_assay_overlap(df, min_overlap=1)

        # ASSAY1 should still only have size flag, not overlap flag
        assay1_comments = result[result["assay_chembl_id"] == "ASSAY1"]["data_dropping_comment"]
        for comment in assay1_comments:
            self.assertIn("Assay size < 20", comment)
            self.assertNotIn("Insufficient assay overlap", comment)

        # ASSAY2 and ASSAY3 should have overlap flag (they don't overlap with each other or ASSAY1)
        assay2_comments = result[result["assay_chembl_id"] == "ASSAY2"]["data_dropping_comment"]
        for comment in assay2_comments:
            self.assertIn("Insufficient assay overlap", comment)

        assay3_comments = result[result["assay_chembl_id"] == "ASSAY3"]["data_dropping_comment"]
        for comment in assay3_comments:
            self.assertIn("Insufficient assay overlap", comment)


class TestFlagZeroValues(unittest.TestCase):
    """Tests for flag_zero_values function."""

    def test_flag_zero_standard_value(self):
        """Test that rows with standard_value of 0 are flagged."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "standard_value": [0.0, 5.5, 10.0],
                "data_dropping_comment": [None, None, None],
            }
        )

        from Capricho.chembl.data_flag_functions import flag_zero_values

        result = flag_zero_values(df)

        # First row should be flagged
        self.assertIn("Zero Value", str(result.loc[0, "data_dropping_comment"]))
        # Other rows should not be flagged
        self.assertTrue(pd.isna(result.loc[1, "data_dropping_comment"]) or
                       "Zero Value" not in str(result.loc[1, "data_dropping_comment"]))
        self.assertTrue(pd.isna(result.loc[2, "data_dropping_comment"]) or
                       "Zero Value" not in str(result.loc[2, "data_dropping_comment"]))

    def test_flag_zero_with_small_values(self):
        """Test that only exact zeros are flagged, not small values."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "standard_value": [0.0, 0.001, 0.01],
                "data_dropping_comment": [None, None, None],
            }
        )

        from Capricho.chembl.data_flag_functions import flag_zero_values

        result = flag_zero_values(df)

        # Only first row (exact zero) should be flagged
        self.assertIn("Zero Value", str(result.loc[0, "data_dropping_comment"]))
        self.assertTrue(pd.isna(result.loc[1, "data_dropping_comment"]) or
                       "Zero Value" not in str(result.loc[1, "data_dropping_comment"]))
        self.assertTrue(pd.isna(result.loc[2, "data_dropping_comment"]) or
                       "Zero Value" not in str(result.loc[2, "data_dropping_comment"]))

    def test_flag_zero_handles_nan(self):
        """Test that NaN values are not flagged as zeros."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "standard_value": [None, 0.0],
                "data_dropping_comment": [None, None],
            }
        )

        from Capricho.chembl.data_flag_functions import flag_zero_values

        result = flag_zero_values(df)

        # First row (NaN) should not be flagged
        self.assertTrue(pd.isna(result.loc[0, "data_dropping_comment"]) or
                       "Zero Value" not in str(result.loc[0, "data_dropping_comment"]))
        # Second row (zero) should be flagged
        self.assertIn("Zero Value", str(result.loc[1, "data_dropping_comment"]))


if __name__ == "__main__":
    unittest.main()
