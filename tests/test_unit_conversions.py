"""Tests for unit conversion functionality.

ABOUTME: Tests for converting ChEMBL bioactivity measurement units to standardized formats.
ABOUTME: Covers permeability, molar concentration, mass concentration, dose, and time conversions.
"""

import unittest

import pandas as pd

from Capricho.chembl.data_flag_functions import flag_unit_conversion
from Capricho.chembl.unit_conversions import (
    convert_dose_units,
    convert_mass_concentration_units,
    convert_molar_concentration_units,
    convert_permeability_units,
    convert_time_units,
    is_unit_annotation_error_diff,
)


class TestIsUnitAnnotationErrorDiff(unittest.TestCase):
    """The unit-annotation-error predicate flags exact multiples of 3 log units (nM<->uM<->mM<->M)."""

    def test_exact_multiples_of_three_are_flagged(self):
        for diff in (3.0, 6.0, 9.0, 12.0):
            self.assertTrue(bool(is_unit_annotation_error_diff(diff)), f"{diff} should flag")

    def test_zero_and_non_multiples_are_not_flagged(self):
        for diff in (0.0, 1.5, 4.5, 5.0, 7.0):
            self.assertFalse(bool(is_unit_annotation_error_diff(diff)), f"{diff} should not flag")

    def test_floating_point_tolerance(self):
        self.assertTrue(bool(is_unit_annotation_error_diff(2.9999999998)))  # ~3.0 within tolerance
        self.assertFalse(bool(is_unit_annotation_error_diff(3.01)))  # clearly off a multiple

    def test_vectorized_input(self):
        mask = is_unit_annotation_error_diff(pd.Series([3.0, 4.5, 6.0, 9.0]))
        self.assertEqual(list(mask), [True, False, True, True])


class TestConvertPermeabilityUnits(unittest.TestCase):
    """Tests for convert_permeability_units function."""

    def test_convert_cm_per_s(self):
        """Test conversion from cm/s to 10^-6 cm/s."""
        df = pd.DataFrame({"standard_value": [1.0, 2.5], "standard_units": ["cm/s", "cm/s"]})

        result = convert_permeability_units(df)

        # 1 cm/s = 1,000,000 * 10^-6 cm/s
        self.assertEqual(result.loc[0, "standard_value"], 1e6)
        self.assertEqual(result.loc[1, "standard_value"], 2.5e6)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")
        self.assertIn("conversion_factor", result.columns)

    def test_convert_nm_per_s(self):
        """Test conversion from nm/s to 10^-6 cm/s."""
        df = pd.DataFrame({"standard_value": [100.0, 50.0], "standard_units": ["nm/s", "nm/s"]})

        result = convert_permeability_units(df)

        # 1 nm/s = 0.1 * 10^-6 cm/s
        self.assertEqual(result.loc[0, "standard_value"], 10.0)
        self.assertEqual(result.loc[1, "standard_value"], 5.0)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")

    def test_already_in_target_unit(self):
        """Test that values already in target unit (10^-6 cm/s) remain unchanged."""
        df = pd.DataFrame({"standard_value": [5.0, 10.0], "standard_units": ["10^-6 cm/s", "10^-6 cm/s"]})

        result = convert_permeability_units(df)

        # Factor is 1.0, so values should remain the same
        self.assertEqual(result.loc[0, "standard_value"], 5.0)
        self.assertEqual(result.loc[1, "standard_value"], 10.0)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")

    def test_unknown_units_unchanged(self):
        """Test that unknown units remain unchanged."""
        df = pd.DataFrame({"standard_value": [5.0, 10.0], "standard_units": ["unknown", "weird_unit"]})

        result = convert_permeability_units(df)

        # Unknown units should not be converted
        self.assertEqual(result.loc[0, "standard_value"], 5.0)
        self.assertEqual(result.loc[1, "standard_value"], 10.0)
        self.assertEqual(result.loc[0, "standard_units"], "unknown")
        self.assertEqual(result.loc[1, "standard_units"], "weird_unit")
        # conversion_factor should be NaN for unknown units
        self.assertTrue(pd.isna(result.loc[0, "conversion_factor"]))
        self.assertTrue(pd.isna(result.loc[1, "conversion_factor"]))

    def test_mixed_units(self):
        """Test conversion with mixed known and unknown units."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 100.0, 5.0, 10.0], "standard_units": ["cm/s", "nm/s", "unknown", "10^-6 cm/s"]}
        )

        result = convert_permeability_units(df)

        # Check conversions
        self.assertEqual(result.loc[0, "standard_value"], 1e6)  # cm/s converted
        self.assertEqual(result.loc[1, "standard_value"], 10.0)  # nm/s converted
        self.assertEqual(result.loc[2, "standard_value"], 5.0)  # unknown unchanged
        self.assertEqual(result.loc[3, "standard_value"], 10.0)  # already in target unit

        # Check units
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[2, "standard_units"], "unknown")
        self.assertEqual(result.loc[3, "standard_units"], "10^-6 cm/s")

    def test_missing_values(self):
        """Test that rows with missing values are not converted."""
        df = pd.DataFrame({"standard_value": [1.0, None, 5.0], "standard_units": ["cm/s", "cm/s", None]})

        result = convert_permeability_units(df)

        # First row should be converted
        self.assertEqual(result.loc[0, "standard_value"], 1e6)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertFalse(pd.isna(result.loc[0, "conversion_factor"]))

        # Second row has missing value, should not be converted
        # Value stays NaN, unit stays unchanged
        self.assertTrue(pd.isna(result.loc[1, "standard_value"]))
        self.assertEqual(result.loc[1, "standard_units"], "cm/s")
        # conversion_factor is set (known unit) but conversion doesn't happen due to missing value
        self.assertFalse(pd.isna(result.loc[1, "conversion_factor"]))

        # Third row has missing units, should not be converted
        self.assertEqual(result.loc[2, "standard_value"], 5.0)
        self.assertTrue(pd.isna(result.loc[2, "standard_units"]))
        self.assertTrue(pd.isna(result.loc[2, "conversion_factor"]))

    def test_additional_permeability_unit_variants(self):
        """Test conversion of permeability units found in MDCK-MDR1 ChEMBL data."""
        df = pd.DataFrame(
            {
                "standard_value": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                "standard_units": [
                    "1E-6 cm/s",       # equivalent to 10^-6 cm/s → factor 1.0
                    "10'-5cm/s",       # 10^-5 cm/s = 10 × 10^-6 cm/s → factor 10.0
                    "10'-5 cm/s",      # same with space
                    "10^-5 cm/s",      # caret notation
                    "10^-5cm/s",       # caret no space
                    "10-6 cm/s",       # missing exponent marker, 10^-6 → factor 1.0
                    "10'-8m/s",        # 10^-8 m/s = 10^-6 cm/s → factor 1.0
                    "10'-9meter/s",    # 10^-9 m/s = 10^-7 cm/s → factor 0.1
                ],
            }
        )

        result = convert_permeability_units(df)

        # All should be converted to 10^-6 cm/s
        for i in range(len(result)):
            self.assertEqual(result.loc[i, "standard_units"], "10^-6 cm/s", f"Row {i} unit not converted")

        # Check correct conversion factors
        self.assertEqual(result.loc[0, "standard_value"], 5.0)    # 1E-6: factor 1.0
        self.assertEqual(result.loc[1, "standard_value"], 50.0)   # 10'-5cm/s: factor 10.0
        self.assertEqual(result.loc[2, "standard_value"], 50.0)   # 10'-5 cm/s: factor 10.0
        self.assertEqual(result.loc[3, "standard_value"], 50.0)   # 10^-5 cm/s: factor 10.0
        self.assertEqual(result.loc[4, "standard_value"], 50.0)   # 10^-5cm/s: factor 10.0
        self.assertEqual(result.loc[5, "standard_value"], 5.0)    # 10-6 cm/s: factor 1.0
        self.assertEqual(result.loc[6, "standard_value"], 5.0)    # 10'-8m/s: factor 1.0
        self.assertAlmostEqual(result.loc[7, "standard_value"], 0.5, places=6)  # 10'-9meter/s: factor 0.1

    def test_case_insensitive_units(self):
        """Test that unit matching is case-insensitive."""
        df = pd.DataFrame({"standard_value": [1.0, 1.0], "standard_units": ["CM/S", "Cm/s"]})

        result = convert_permeability_units(df)

        # Both should be converted despite different case
        self.assertEqual(result.loc[0, "standard_value"], 1e6)
        self.assertEqual(result.loc[1, "standard_value"], 1e6)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")

    def test_custom_value_column(self):
        """Test conversion with custom value column name."""
        df = pd.DataFrame({"my_value": [1.0, 100.0], "standard_units": ["cm/s", "nm/s"]})

        result = convert_permeability_units(df, value_col="my_value")

        self.assertEqual(result.loc[0, "my_value"], 1e6)
        self.assertEqual(result.loc[1, "my_value"], 10.0)
        self.assertEqual(result.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "standard_units"], "10^-6 cm/s")

    def test_custom_unit_column(self):
        """Test conversion with custom unit column name."""
        df = pd.DataFrame({"standard_value": [1.0, 100.0], "my_units": ["cm/s", "nm/s"]})

        result = convert_permeability_units(df, unit_col="my_units")

        self.assertEqual(result.loc[0, "standard_value"], 1e6)
        self.assertEqual(result.loc[1, "standard_value"], 10.0)
        self.assertEqual(result.loc[0, "my_units"], "10^-6 cm/s")
        self.assertEqual(result.loc[1, "my_units"], "10^-6 cm/s")

    def test_missing_columns(self):
        """Test that function handles missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        # Should return unchanged dataframe if columns are missing
        result = convert_permeability_units(df)
        pd.testing.assert_frame_equal(result, df)


class TestFlagUnitConversion(unittest.TestCase):
    """Tests for flag_unit_conversion function."""

    def test_flag_converted_units(self):
        """Test that converted units are flagged in data_processing_comment."""
        df = pd.DataFrame(
            {
                "standard_value": [1e6, 10.0],
                "standard_units": ["10^-6 cm/s", "10^-6 cm/s"],
                "conversion_factor": [1e6, 0.1],
                "original_unit": ["cm/s", "nm/s"],
                "data_processing_comment": [None, None],
            }
        )

        result = flag_unit_conversion(df)

        # Check that both rows are flagged with dynamic comments
        self.assertIn("Unit converted to 10^-6 cm/s from cm/s", str(result.loc[0, "data_processing_comment"]))
        self.assertIn("Unit converted to 10^-6 cm/s from nm/s", str(result.loc[1, "data_processing_comment"]))
        # Check that conversion_factor and original_unit columns are removed
        self.assertNotIn("conversion_factor", result.columns)
        self.assertNotIn("original_unit", result.columns)

    def test_no_conversion_factor_column(self):
        """Test that function handles missing conversion_factor column gracefully."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 2.0], "standard_units": ["cm/s", "nm/s"], "data_processing_comment": [None, None]}
        )

        result = flag_unit_conversion(df)

        # Should return unchanged dataframe
        pd.testing.assert_frame_equal(result, df)

    def test_mixed_conversion_factors(self):
        """Test flagging with some NaN conversion factors."""
        df = pd.DataFrame(
            {
                "standard_value": [1e6, 5.0, 10.0],
                "standard_units": ["10^-6 cm/s", "unknown", "10^-6 cm/s"],
                "conversion_factor": [1e6, None, 0.1],
                "original_unit": ["cm/s", None, "nm/s"],
                "data_processing_comment": [None, None, None],
            }
        )

        result = flag_unit_conversion(df)

        # Check that only converted rows are flagged
        self.assertIn("Unit converted to 10^-6 cm/s from cm/s", str(result.loc[0, "data_processing_comment"]))
        self.assertTrue(
            pd.isna(result.loc[1, "data_processing_comment"])
            or result.loc[1, "data_processing_comment"] in [None, "", pd.NA]
        )
        self.assertIn("Unit converted to 10^-6 cm/s from nm/s", str(result.loc[2, "data_processing_comment"]))
        # Check that conversion_factor and original_unit columns are removed
        self.assertNotIn("conversion_factor", result.columns)
        self.assertNotIn("original_unit", result.columns)

    def test_append_to_existing_comment(self):
        """Test that flagging appends to existing data_processing_comment."""
        df = pd.DataFrame(
            {
                "standard_value": [1e6],
                "standard_units": ["10^-6 cm/s"],
                "conversion_factor": [1e6],
                "original_unit": ["cm/s"],
                "data_processing_comment": ["Existing comment"],
            }
        )

        result = flag_unit_conversion(df)

        # Check that new comment is appended to existing one
        comment = result.loc[0, "data_processing_comment"]
        self.assertIn("Existing comment", str(comment))
        self.assertIn("Unit converted to 10^-6 cm/s from cm/s", str(comment))

    def test_missing_original_unit_logs_warning(self):
        """Test that missing original_unit column logs a warning and skips flagging."""
        df = pd.DataFrame(
            {
                "standard_value": [1e6],
                "standard_units": ["10^-6 cm/s"],
                "conversion_factor": [1e6],
                "data_processing_comment": [None],
            }
        )

        result = flag_unit_conversion(df)

        # Should not add any comment since original_unit is missing
        self.assertTrue(
            pd.isna(result.loc[0, "data_processing_comment"])
            or result.loc[0, "data_processing_comment"] in [None, "", pd.NA]
        )
        # conversion_factor should still be removed
        self.assertNotIn("conversion_factor", result.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full conversion and flagging workflow."""

    def test_full_workflow(self):
        """Test the full workflow: convert then flag."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "standard_value": [1.0, 100.0, 5.0],
                "standard_units": ["cm/s", "nm/s", "unknown"],
                "data_processing_comment": [None, None, None],
            }
        )

        # Convert units
        df = convert_permeability_units(df)
        # Flag conversions
        df = flag_unit_conversion(df)

        # Check conversions
        self.assertEqual(df.loc[0, "standard_value"], 1e6)
        self.assertEqual(df.loc[1, "standard_value"], 10.0)
        self.assertEqual(df.loc[2, "standard_value"], 5.0)

        # Check units
        self.assertEqual(df.loc[0, "standard_units"], "10^-6 cm/s")
        self.assertEqual(df.loc[1, "standard_units"], "10^-6 cm/s")
        self.assertEqual(df.loc[2, "standard_units"], "unknown")

        # Check flags
        self.assertIn("Unit converted", str(df.loc[0, "data_processing_comment"]))
        self.assertIn("Unit converted", str(df.loc[1, "data_processing_comment"]))
        self.assertTrue(
            pd.isna(df.loc[2, "data_processing_comment"])
            or df.loc[2, "data_processing_comment"] in [None, "", pd.NA]
        )

        # Check that conversion_factor column was removed
        self.assertNotIn("conversion_factor", df.columns)


class TestConvertMolarConcentrationUnits(unittest.TestCase):
    """Tests for convert_molar_concentration_units function. Target: nM."""

    def test_convert_um_to_nm(self):
        """Test conversion from uM to nM."""
        df = pd.DataFrame({"standard_value": [1.0, 2.5], "standard_units": ["uM", "uM"]})

        result = convert_molar_concentration_units(df)

        # 1 uM = 1000 nM
        self.assertEqual(result.loc[0, "standard_value"], 1000.0)
        self.assertEqual(result.loc[1, "standard_value"], 2500.0)
        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")
        self.assertIn("conversion_factor", result.columns)

    def test_convert_micromolar_variants(self):
        """Test conversion from various micromolar representations."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 1.0, 1.0], "standard_units": ["uM", "µM", "um"]}
        )

        result = convert_molar_concentration_units(df)

        # All should be converted to 1000 nM
        self.assertEqual(result.loc[0, "standard_value"], 1000.0)
        self.assertEqual(result.loc[1, "standard_value"], 1000.0)
        self.assertEqual(result.loc[2, "standard_value"], 1000.0)

    def test_convert_mm_to_nm(self):
        """Test conversion from mM to nM."""
        df = pd.DataFrame({"standard_value": [1.0, 0.5], "standard_units": ["mM", "mM"]})

        result = convert_molar_concentration_units(df)

        # 1 mM = 1,000,000 nM
        self.assertEqual(result.loc[0, "standard_value"], 1e6)
        self.assertEqual(result.loc[1, "standard_value"], 5e5)
        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")

    def test_convert_pm_to_nm(self):
        """Test conversion from pM to nM."""
        df = pd.DataFrame({"standard_value": [1000.0, 500.0], "standard_units": ["pM", "pM"]})

        result = convert_molar_concentration_units(df)

        # 1 pM = 0.001 nM
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 0.5)
        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")

    def test_convert_m_to_nm(self):
        """Test conversion from M to nM."""
        df = pd.DataFrame({"standard_value": [1e-9, 1e-8], "standard_units": ["M", "M"]})

        result = convert_molar_concentration_units(df)

        # 1 M = 1e9 nM
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 10.0)
        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")

    def test_nm_unchanged(self):
        """Test that nM values remain unchanged (factor 1.0)."""
        df = pd.DataFrame({"standard_value": [100.0, 500.0], "standard_units": ["nM", "nM"]})

        result = convert_molar_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 100.0)
        self.assertEqual(result.loc[1, "standard_value"], 500.0)
        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")

    def test_case_insensitive(self):
        """Test that unit matching is case-insensitive."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 1.0, 1.0], "standard_units": ["UM", "um", "Um"]}
        )

        result = convert_molar_concentration_units(df)

        # All should be converted to 1000 nM
        for i in range(3):
            self.assertEqual(result.loc[i, "standard_value"], 1000.0)
            self.assertEqual(result.loc[i, "standard_units"], "nM")

    def test_mixed_units(self):
        """Test conversion with mixed known and unknown units."""
        df = pd.DataFrame(
            {
                "standard_value": [1.0, 1.0, 5.0, 100.0],
                "standard_units": ["uM", "mM", "unknown", "nM"],
            }
        )

        result = convert_molar_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1000.0)  # uM -> nM
        self.assertEqual(result.loc[1, "standard_value"], 1e6)  # mM -> nM
        self.assertEqual(result.loc[2, "standard_value"], 5.0)  # unknown unchanged
        self.assertEqual(result.loc[3, "standard_value"], 100.0)  # nM unchanged

        self.assertEqual(result.loc[0, "standard_units"], "nM")
        self.assertEqual(result.loc[1, "standard_units"], "nM")
        self.assertEqual(result.loc[2, "standard_units"], "unknown")
        self.assertEqual(result.loc[3, "standard_units"], "nM")

    def test_missing_values(self):
        """Test that rows with missing values are not converted."""
        df = pd.DataFrame(
            {"standard_value": [1.0, None, 5.0], "standard_units": ["uM", "uM", None]}
        )

        result = convert_molar_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1000.0)
        self.assertTrue(pd.isna(result.loc[1, "standard_value"]))
        self.assertEqual(result.loc[2, "standard_value"], 5.0)

    def test_missing_columns(self):
        """Test that function handles missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = convert_molar_concentration_units(df)
        pd.testing.assert_frame_equal(result, df)

    def test_tracks_original_unit(self):
        """Test that original_unit column is added for tracking."""
        df = pd.DataFrame({"standard_value": [1.0], "standard_units": ["uM"]})

        result = convert_molar_concentration_units(df)

        self.assertIn("original_unit", result.columns)
        self.assertEqual(result.loc[0, "original_unit"], "uM")


class TestConvertMassConcentrationUnits(unittest.TestCase):
    """Tests for convert_mass_concentration_units function. Target: ug/mL."""

    def test_convert_ng_ml_to_ug_ml(self):
        """Test conversion from ng/ml to ug/mL."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, 500.0], "standard_units": ["ng/ml", "ng ml-1"]}
        )

        result = convert_mass_concentration_units(df)

        # 1 ng/ml = 0.001 ug/mL
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 0.5)
        self.assertEqual(result.loc[0, "standard_units"], "ug/mL")
        self.assertEqual(result.loc[1, "standard_units"], "ug/mL")

    def test_convert_mg_ml_to_ug_ml(self):
        """Test conversion from mg/ml to ug/mL."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 0.5], "standard_units": ["mg/ml", "mg ml-1"]}
        )

        result = convert_mass_concentration_units(df)

        # 1 mg/ml = 1000 ug/mL
        self.assertEqual(result.loc[0, "standard_value"], 1000.0)
        self.assertEqual(result.loc[1, "standard_value"], 500.0)
        self.assertEqual(result.loc[0, "standard_units"], "ug/mL")
        self.assertEqual(result.loc[1, "standard_units"], "ug/mL")

    def test_convert_mg_l_to_ug_ml(self):
        """Test conversion from mg/L to ug/mL."""
        df = pd.DataFrame({"standard_value": [1.0, 2.0], "standard_units": ["mg/L", "mg/l"]})

        result = convert_mass_concentration_units(df)

        # 1 mg/L = 1 ug/mL (same)
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 2.0)
        self.assertEqual(result.loc[0, "standard_units"], "ug/mL")
        self.assertEqual(result.loc[1, "standard_units"], "ug/mL")

    def test_convert_pg_ml_to_ug_ml(self):
        """Test conversion from pg/ml to ug/mL."""
        df = pd.DataFrame(
            {"standard_value": [1e6, 5e5], "standard_units": ["pg/ml", "pg ml-1"]}
        )

        result = convert_mass_concentration_units(df)

        # 1 pg/ml = 1e-6 ug/mL
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 0.5)
        self.assertEqual(result.loc[0, "standard_units"], "ug/mL")
        self.assertEqual(result.loc[1, "standard_units"], "ug/mL")

    def test_ug_ml_variants_unchanged(self):
        """Test that ug/mL and variants remain unchanged (factor 1.0)."""
        df = pd.DataFrame(
            {"standard_value": [10.0, 20.0, 30.0], "standard_units": ["ug/mL", "ug.mL-1", "ug ml-1"]}
        )

        result = convert_mass_concentration_units(df)

        for i in range(3):
            self.assertEqual(result.loc[i, "standard_units"], "ug/mL")
        # Values should stay the same (factor 1.0)
        self.assertEqual(result.loc[0, "standard_value"], 10.0)
        self.assertEqual(result.loc[1, "standard_value"], 20.0)
        self.assertEqual(result.loc[2, "standard_value"], 30.0)

    def test_case_insensitive(self):
        """Test that unit matching is case-insensitive."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, 1000.0], "standard_units": ["NG/ML", "Ng/mL"]}
        )

        result = convert_mass_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 1.0)

    def test_mixed_units(self):
        """Test conversion with mixed known and unknown units."""
        df = pd.DataFrame(
            {
                "standard_value": [1000.0, 1.0, 5.0, 10.0],
                "standard_units": ["ng/ml", "mg/ml", "unknown", "ug/mL"],
            }
        )

        result = convert_mass_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)  # ng/ml -> ug/mL
        self.assertEqual(result.loc[1, "standard_value"], 1000.0)  # mg/ml -> ug/mL
        self.assertEqual(result.loc[2, "standard_value"], 5.0)  # unknown unchanged
        self.assertEqual(result.loc[3, "standard_value"], 10.0)  # ug/mL unchanged

        self.assertEqual(result.loc[2, "standard_units"], "unknown")
        for i in [0, 1, 3]:
            self.assertEqual(result.loc[i, "standard_units"], "ug/mL")

    def test_missing_values(self):
        """Test that rows with missing values are not converted."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, None, 5.0], "standard_units": ["ng/ml", "ng/ml", None]}
        )

        result = convert_mass_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertTrue(pd.isna(result.loc[1, "standard_value"]))
        self.assertEqual(result.loc[2, "standard_value"], 5.0)

    def test_missing_columns(self):
        """Test that function handles missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = convert_mass_concentration_units(df)
        pd.testing.assert_frame_equal(result, df)


class TestConvertDoseUnits(unittest.TestCase):
    """Tests for convert_dose_units function. Target: mg/kg."""

    def test_convert_ug_kg_to_mg_kg(self):
        """Test conversion from ug/kg to mg/kg."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, 500.0], "standard_units": ["ug/kg", "ug.kg-1"]}
        )

        result = convert_dose_units(df)

        # 1 ug/kg = 0.001 mg/kg
        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 0.5)
        self.assertEqual(result.loc[0, "standard_units"], "mg/kg")
        self.assertEqual(result.loc[1, "standard_units"], "mg/kg")

    def test_mg_kg_variants_unchanged(self):
        """Test that mg/kg and variants remain unchanged (factor 1.0)."""
        df = pd.DataFrame(
            {"standard_value": [10.0, 20.0, 30.0], "standard_units": ["mg/kg", "mg.kg-1", "mg kg-1"]}
        )

        result = convert_dose_units(df)

        for i in range(3):
            self.assertEqual(result.loc[i, "standard_units"], "mg/kg")
        self.assertEqual(result.loc[0, "standard_value"], 10.0)
        self.assertEqual(result.loc[1, "standard_value"], 20.0)
        self.assertEqual(result.loc[2, "standard_value"], 30.0)

    def test_case_insensitive(self):
        """Test that unit matching is case-insensitive."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, 1000.0], "standard_units": ["UG/KG", "Ug.Kg-1"]}
        )

        result = convert_dose_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 1.0)

    def test_mixed_units(self):
        """Test conversion with mixed known and unknown units."""
        df = pd.DataFrame(
            {
                "standard_value": [1000.0, 5.0, 10.0],
                "standard_units": ["ug/kg", "unknown", "mg/kg"],
            }
        )

        result = convert_dose_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)  # ug/kg -> mg/kg
        self.assertEqual(result.loc[1, "standard_value"], 5.0)  # unknown unchanged
        self.assertEqual(result.loc[2, "standard_value"], 10.0)  # mg/kg unchanged

        self.assertEqual(result.loc[1, "standard_units"], "unknown")
        self.assertEqual(result.loc[0, "standard_units"], "mg/kg")
        self.assertEqual(result.loc[2, "standard_units"], "mg/kg")

    def test_missing_values(self):
        """Test that rows with missing values are not converted."""
        df = pd.DataFrame(
            {"standard_value": [1000.0, None, 5.0], "standard_units": ["ug/kg", "ug/kg", None]}
        )

        result = convert_dose_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertTrue(pd.isna(result.loc[1, "standard_value"]))
        self.assertEqual(result.loc[2, "standard_value"], 5.0)

    def test_missing_columns(self):
        """Test that function handles missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = convert_dose_units(df)
        pd.testing.assert_frame_equal(result, df)


class TestConvertTimeUnits(unittest.TestCase):
    """Tests for convert_time_units function. Target: hr (hours)."""

    def test_convert_min_to_hr(self):
        """Test conversion from min to hr."""
        df = pd.DataFrame({"standard_value": [60.0, 120.0], "standard_units": ["min", "min"]})

        result = convert_time_units(df)

        # 1 min = 1/60 hr
        self.assertAlmostEqual(result.loc[0, "standard_value"], 1.0, places=6)
        self.assertAlmostEqual(result.loc[1, "standard_value"], 2.0, places=6)
        self.assertEqual(result.loc[0, "standard_units"], "hr")
        self.assertEqual(result.loc[1, "standard_units"], "hr")

    def test_convert_s_to_hr(self):
        """Test conversion from s to hr."""
        df = pd.DataFrame({"standard_value": [3600.0, 7200.0], "standard_units": ["s", "s"]})

        result = convert_time_units(df)

        # 1 s = 1/3600 hr
        self.assertAlmostEqual(result.loc[0, "standard_value"], 1.0, places=6)
        self.assertAlmostEqual(result.loc[1, "standard_value"], 2.0, places=6)
        self.assertEqual(result.loc[0, "standard_units"], "hr")
        self.assertEqual(result.loc[1, "standard_units"], "hr")

    def test_convert_ms_to_hr(self):
        """Test conversion from ms to hr."""
        df = pd.DataFrame(
            {"standard_value": [3600000.0, 7200000.0], "standard_units": ["ms", "ms"]}
        )

        result = convert_time_units(df)

        # 1 ms = 1/3,600,000 hr
        self.assertAlmostEqual(result.loc[0, "standard_value"], 1.0, places=4)
        self.assertAlmostEqual(result.loc[1, "standard_value"], 2.0, places=4)
        self.assertEqual(result.loc[0, "standard_units"], "hr")
        self.assertEqual(result.loc[1, "standard_units"], "hr")

    def test_convert_day_to_hr(self):
        """Test conversion from day to hr."""
        df = pd.DataFrame({"standard_value": [1.0, 2.0], "standard_units": ["day", "day"]})

        result = convert_time_units(df)

        # 1 day = 24 hr
        self.assertEqual(result.loc[0, "standard_value"], 24.0)
        self.assertEqual(result.loc[1, "standard_value"], 48.0)
        self.assertEqual(result.loc[0, "standard_units"], "hr")
        self.assertEqual(result.loc[1, "standard_units"], "hr")

    def test_hr_unchanged(self):
        """Test that hr values remain unchanged (factor 1.0)."""
        df = pd.DataFrame({"standard_value": [1.0, 2.5], "standard_units": ["hr", "hr"]})

        result = convert_time_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1.0)
        self.assertEqual(result.loc[1, "standard_value"], 2.5)
        self.assertEqual(result.loc[0, "standard_units"], "hr")
        self.assertEqual(result.loc[1, "standard_units"], "hr")

    def test_case_insensitive(self):
        """Test that unit matching is case-insensitive."""
        df = pd.DataFrame(
            {"standard_value": [60.0, 60.0, 60.0], "standard_units": ["MIN", "Min", "min"]}
        )

        result = convert_time_units(df)

        for i in range(3):
            self.assertAlmostEqual(result.loc[i, "standard_value"], 1.0, places=6)
            self.assertEqual(result.loc[i, "standard_units"], "hr")

    def test_mixed_units(self):
        """Test conversion with mixed known and unknown units."""
        df = pd.DataFrame(
            {
                "standard_value": [60.0, 24.0, 5.0, 1.0],
                "standard_units": ["min", "day", "unknown", "hr"],
            }
        )

        result = convert_time_units(df)

        self.assertAlmostEqual(result.loc[0, "standard_value"], 1.0, places=6)  # min -> hr
        self.assertEqual(result.loc[1, "standard_value"], 576.0)  # day -> hr
        self.assertEqual(result.loc[2, "standard_value"], 5.0)  # unknown unchanged
        self.assertEqual(result.loc[3, "standard_value"], 1.0)  # hr unchanged

        self.assertEqual(result.loc[2, "standard_units"], "unknown")
        for i in [0, 1, 3]:
            self.assertEqual(result.loc[i, "standard_units"], "hr")

    def test_missing_values(self):
        """Test that rows with missing values are not converted."""
        df = pd.DataFrame(
            {"standard_value": [60.0, None, 5.0], "standard_units": ["min", "min", None]}
        )

        result = convert_time_units(df)

        self.assertAlmostEqual(result.loc[0, "standard_value"], 1.0, places=6)
        self.assertTrue(pd.isna(result.loc[1, "standard_value"]))
        self.assertEqual(result.loc[2, "standard_value"], 5.0)

    def test_missing_columns(self):
        """Test that function handles missing columns gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = convert_time_units(df)
        pd.testing.assert_frame_equal(result, df)


class TestFlagUnitConversionEnhanced(unittest.TestCase):
    """Additional tests for flag_unit_conversion with multiple conversion types."""

    def test_flag_with_original_unit_column(self):
        """Test that flagging includes original unit information when available."""
        df = pd.DataFrame(
            {
                "standard_value": [1000.0, 1.0],
                "standard_units": ["nM", "ug/mL"],
                "conversion_factor": [1000.0, 0.001],
                "original_unit": ["uM", "ng/ml"],
                "data_processing_comment": [None, None],
            }
        )

        result = flag_unit_conversion(df)

        # Should include both the target unit and original unit in the comment
        self.assertIn("Unit converted", str(result.loc[0, "data_processing_comment"]))
        self.assertIn("Unit converted", str(result.loc[1, "data_processing_comment"]))
        # original_unit column should be removed
        self.assertNotIn("original_unit", result.columns)
        self.assertNotIn("conversion_factor", result.columns)


class TestIntegrationEnhanced(unittest.TestCase):
    """Integration tests for the full conversion and flagging workflow with multiple converters."""

    def test_molar_concentration_workflow(self):
        """Test full workflow for molar concentration conversions."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
                "standard_value": [1.0, 1000.0, 5.0, 100.0],
                "standard_units": ["uM", "pM", "unknown", "nM"],
                "data_processing_comment": [None, None, None, None],
            }
        )

        # Convert units
        df = convert_molar_concentration_units(df)
        # Flag conversions
        df = flag_unit_conversion(df)

        # Check conversions
        self.assertEqual(df.loc[0, "standard_value"], 1000.0)  # uM -> nM
        self.assertEqual(df.loc[1, "standard_value"], 1.0)  # pM -> nM
        self.assertEqual(df.loc[2, "standard_value"], 5.0)  # unchanged
        self.assertEqual(df.loc[3, "standard_value"], 100.0)  # nM unchanged

        # Check flags
        self.assertIn("Unit converted", str(df.loc[0, "data_processing_comment"]))
        self.assertIn("Unit converted", str(df.loc[1, "data_processing_comment"]))
        self.assertTrue(
            pd.isna(df.loc[2, "data_processing_comment"])
            or df.loc[2, "data_processing_comment"] in [None, "", pd.NA]
        )

    def test_time_units_workflow(self):
        """Test full workflow for time unit conversions."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "standard_value": [60.0, 1.0, 5.0],
                "standard_units": ["min", "day", "unknown"],
                "data_processing_comment": [None, None, None],
            }
        )

        # Convert units
        df = convert_time_units(df)
        # Flag conversions
        df = flag_unit_conversion(df)

        # Check conversions
        self.assertAlmostEqual(df.loc[0, "standard_value"], 1.0, places=6)  # min -> hr
        self.assertEqual(df.loc[1, "standard_value"], 24.0)  # day -> hr
        self.assertEqual(df.loc[2, "standard_value"], 5.0)  # unchanged

        # Check units
        self.assertEqual(df.loc[0, "standard_units"], "hr")
        self.assertEqual(df.loc[1, "standard_units"], "hr")
        self.assertEqual(df.loc[2, "standard_units"], "unknown")

    def test_sequential_conversions_different_types(self):
        """Test that applying different converters sequentially doesn't interfere."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
                "standard_value": [1.0, 60.0],
                "standard_units": ["uM", "min"],  # Different unit types
                "data_processing_comment": [None, None],
            }
        )

        # Apply molar conversion (should only affect row 0)
        df = convert_molar_concentration_units(df)
        self.assertEqual(df.loc[0, "standard_value"], 1000.0)
        self.assertEqual(df.loc[1, "standard_value"], 60.0)

        # Apply time conversion (should only affect row 1)
        df = convert_time_units(df)
        self.assertEqual(df.loc[0, "standard_value"], 1000.0)  # Still molar
        self.assertAlmostEqual(df.loc[1, "standard_value"], 1.0, places=6)  # Now hours


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for unit conversion functions."""

    def test_whitespace_in_units(self):
        """Test that units with leading/trailing whitespace are handled."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 1.0], "standard_units": [" uM ", "  uM"]}
        )

        result = convert_molar_concentration_units(df)

        self.assertEqual(result.loc[0, "standard_value"], 1000.0)
        self.assertEqual(result.loc[1, "standard_value"], 1000.0)

    def test_empty_dataframe(self):
        """Test that empty dataframes are handled gracefully."""
        df = pd.DataFrame({"standard_value": [], "standard_units": []})

        result = convert_molar_concentration_units(df)

        self.assertTrue(result.empty)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved after conversion."""
        df = pd.DataFrame(
            {"standard_value": [1.0, 2.0], "standard_units": ["uM", "uM"]}
        )
        original_dtype = df["standard_value"].dtype

        result = convert_molar_concentration_units(df)

        self.assertEqual(result["standard_value"].dtype, original_dtype)

    def test_very_small_values(self):
        """Test conversion of very small values."""
        df = pd.DataFrame({"standard_value": [1e-15], "standard_units": ["M"]})

        result = convert_molar_concentration_units(df)

        # 1e-15 M = 1e-6 nM
        self.assertAlmostEqual(result.loc[0, "standard_value"], 1e-6, places=12)

    def test_very_large_values(self):
        """Test conversion of very large values."""
        df = pd.DataFrame({"standard_value": [1e12], "standard_units": ["pM"]})

        result = convert_molar_concentration_units(df)

        # 1e12 pM = 1e9 nM
        self.assertEqual(result.loc[0, "standard_value"], 1e9)


if __name__ == "__main__":
    unittest.main()
