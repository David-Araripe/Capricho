"""Functions for converting ChEMBL bioactivity units to standardized formats.

Provides unit conversion functions for ChEMBL bioactivity data standardization.
Supports permeability, molar concentration, mass concentration, dose, and time units.
"""

import numpy as np
import pandas as pd

from ..logger import logger

# Metric-prefix step in log10 space: activities that differ by an exact integer multiple of this
# (nM<->uM = 3, nM<->mM = 6, nM<->M = 9, ...) are the classic unit-conversion/annotation error.
UNIT_ERROR_LOG_STEP = 3.0


def is_unit_annotation_error_diff(abs_diff, tol: float = 1e-9):
    """Boolean mask of pChEMBL differences that look like a unit-annotation error.

    A difference that is an exact positive integer multiple of 3 log units corresponds to a
    metric-prefix confusion (nM<->uM = 3, nM<->mM = 6, nM<->M = 9, ...), a frequent unit
    conversion error in deposited data. Accepts a scalar or array-like of absolute differences
    and returns a numpy boolean of the same shape.
    """
    abs_diff = np.abs(np.asarray(abs_diff, dtype=float))
    nearest_multiple = np.round(abs_diff / UNIT_ERROR_LOG_STEP) * UNIT_ERROR_LOG_STEP
    return (nearest_multiple > 0) & np.isclose(abs_diff, nearest_multiple, rtol=tol, atol=tol)


def convert_permeability_units(
    df: pd.DataFrame,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
) -> pd.DataFrame:
    """Convert permeability data to standardized unit: 10^-6 cm/s.

    This function converts various permeability measurement units found in ChEMBL
    (cm/s, nm/s, ucm/s, etc.) to a single standard unit (10^-6 cm/s) to enable
    aggregation across assays with different reporting conventions.

    Rows that cannot be converted (unknown units or missing values) are left unchanged.
    All conversions are logged and can be tracked via the data_processing_comment column
    using the flag_unit_conversion() function.

    Args:
        df: DataFrame containing bioactivity data.
        value_col: Column name containing numeric values to convert. Defaults to 'standard_value'.
        unit_col: Column name containing unit strings. Defaults to 'standard_units'.

    Returns:
        pd.DataFrame: DataFrame with converted values and standardized units where applicable.
            Adds 'conversion_factor' and 'original_unit' columns for transparency.

    Examples:
        >>> df = pd.DataFrame({
        ...     'standard_value': [1.0, 100.0, 5.0],
        ...     'standard_units': ['cm/s', 'nm/s', 'unknown']
        ... })
        >>> converted = convert_permeability_units(df)
        >>> converted['standard_value'].tolist()
        [1000000.0, 10.0, 5.0]  # cm/s and nm/s converted, unknown unchanged
    """
    if value_col not in df.columns:
        logger.warning(f"Column '{value_col}' not found. Skipping permeability unit conversion.")
        return df

    if unit_col not in df.columns:
        logger.warning(f"Column '{unit_col}' not found. Skipping permeability unit conversion.")
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Normalize unit strings: lowercase and strip whitespace
    df["temp_unit"] = df[unit_col].astype(str).str.lower().str.strip()

    # Define conversion factors to target unit: 10^-6 cm/s
    # Formula: New_Value = Old_Value * Factor
    conversions = {
        # Base cm/s (1 cm/s = 1,000,000 * 10^-6 cm/s)
        "cm s-1": 1e6,
        "cm/s": 1e6,
        # Nanometers (1 nm/s = 0.1 * 10^-6 cm/s)
        "nm s-1": 0.1,
        "nm/s": 0.1,
        # Already in target unit or variations
        "10'-6 cm/s": 1.0,
        "10^-6 cm/s": 1.0,
        "ucm/s": 1.0,  # micro-cm/s
        # Typos/formats indicating value is already scaled to 10^-6
        "cm/s * 10e6": 1.0,
        "10'6cm/s": 1.0,
        "10^6cm/s": 1.0,
        # 10^-7 cm/s (smaller than target by factor of 10)
        "10'-7 cm/s": 0.1,
        # 10^-5 cm/s variants (10x larger than target)
        "10'-5cm/s": 10.0,
        "10'-5 cm/s": 10.0,
        "10^-5 cm/s": 10.0,
        "10^-5cm/s": 10.0,
        # Scientific notation variants for 10^-6 cm/s
        "1e-6 cm/s": 1.0,
        "10-6 cm/s": 1.0,
        # Meter-based units (1 m = 100 cm)
        # 10^-8 m/s = 10^-8 * 100 cm/s = 10^-6 cm/s
        "10'-8m/s": 1.0,
        # 10^-9 m/s = 10^-9 * 100 cm/s = 10^-7 cm/s = 0.1 * 10^-6 cm/s
        "10'-9meter/s": 0.1,
        # Time conversion: cm/min to cm/s then to 10^-6 cm/s
        # 10^-5 cm/min = (10^-5 / 60) cm/s * 10^6 = 10/60
        "10^-5 cm/min": 10 / 60,
        # Very small unit: 10^-6 nm/s
        "10'-6nanometer/s": 1e-7,
    }

    # Map conversion factors
    df["conversion_factor"] = df["temp_unit"].map(conversions)

    # Store original unit for transparency before conversion
    df["original_unit"] = df[unit_col].where(df["conversion_factor"].notna())

    # Identify rows that can be converted:
    # - Must have a known conversion factor
    # - Must have a numeric value
    mask_convertible = df["conversion_factor"].notna() & pd.to_numeric(df[value_col], errors="coerce").notna()

    num_to_convert = mask_convertible.sum()
    num_unconvertible = (~mask_convertible).sum()

    if num_to_convert > 0:
        logger.info(
            f"Converting {num_to_convert} permeability measurements to standard unit (10^-6 cm/s). "
            f"{num_unconvertible} measurements cannot be converted (unknown units or missing values)."
        )

        # Apply conversion to convertible rows
        # Preserve the original dtype to avoid FutureWarning
        original_dtype = df[value_col].dtype
        converted_values = (
            pd.to_numeric(df.loc[mask_convertible, value_col]) * df.loc[mask_convertible, "conversion_factor"]
        )
        df.loc[mask_convertible, value_col] = converted_values.astype(original_dtype)

        # Update unit label for converted rows
        target_label = "10^-6 cm/s"
        df.loc[mask_convertible, unit_col] = target_label

        # Log summary of conversions by original unit
        conversion_summary = (
            df[mask_convertible].groupby("temp_unit")["conversion_factor"].agg(["first", "count"])
        )
        logger.debug("Conversion summary by original unit:")
        for unit, row in conversion_summary.iterrows():
            logger.debug(f"  {unit}: {row['count']} measurements (factor: {row['first']})")
    else:
        logger.info("No permeability measurements found that can be converted to standard unit.")

    # Clean up temporary column
    df.drop(columns=["temp_unit"], inplace=True)

    # Keep conversion_factor for flagging - will be used by flag_unit_conversion()
    return df


def convert_molar_concentration_units(
    df: pd.DataFrame,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
) -> pd.DataFrame:
    """Convert molar concentration data to standardized unit: nM (nanomolar).

    This function converts various molar concentration units found in ChEMBL
    (uM, µM, mM, pM, M) to a single standard unit (nM) to enable aggregation
    across assays with different reporting conventions. This is particularly
    useful for IC50, EC50, Ki, Kd measurements that are not covered by pChEMBL.

    Rows that cannot be converted (unknown units or missing values) are left unchanged.
    All conversions are logged and can be tracked via the data_processing_comment column
    using the flag_unit_conversion() function.

    Args:
        df: DataFrame containing bioactivity data.
        value_col: Column name containing numeric values to convert. Defaults to 'standard_value'.
        unit_col: Column name containing unit strings. Defaults to 'standard_units'.

    Returns:
        pd.DataFrame: DataFrame with converted values and standardized units where applicable.
            Adds 'conversion_factor' and 'original_unit' columns for transparency.

    Examples:
        >>> df = pd.DataFrame({
        ...     'standard_value': [1.0, 1000.0, 5.0],
        ...     'standard_units': ['uM', 'pM', 'unknown']
        ... })
        >>> converted = convert_molar_concentration_units(df)
        >>> converted['standard_value'].tolist()
        [1000.0, 1.0, 5.0]  # uM and pM converted, unknown unchanged
    """
    if value_col not in df.columns:
        logger.warning(f"Column '{value_col}' not found. Skipping molar concentration unit conversion.")
        return df

    if unit_col not in df.columns:
        logger.warning(f"Column '{unit_col}' not found. Skipping molar concentration unit conversion.")
        return df

    df = df.copy()

    # Normalize unit strings: lowercase and strip whitespace
    df["temp_unit"] = df[unit_col].astype(str).str.lower().str.strip()

    # Define conversion factors to target unit: nM
    # Formula: New_Value = Old_Value * Factor
    conversions = {
        # Nanomolar (target unit)
        "nm": 1.0,
        # Micromolar variants (1 µM = 1000 nM)
        "um": 1000.0,
        "µm": 1000.0,
        # Millimolar (1 mM = 1,000,000 nM)
        "mm": 1e6,
        # Picomolar (1 pM = 0.001 nM)
        "pm": 0.001,
        # Molar (1 M = 1,000,000,000 nM)
        "m": 1e9,
    }

    # Map conversion factors
    df["conversion_factor"] = df["temp_unit"].map(conversions)

    # Store original unit for transparency before conversion
    df["original_unit"] = df[unit_col].where(df["conversion_factor"].notna())

    # Identify rows that can be converted
    mask_convertible = df["conversion_factor"].notna() & pd.to_numeric(df[value_col], errors="coerce").notna()

    num_to_convert = mask_convertible.sum()
    num_unconvertible = (~mask_convertible).sum()

    if num_to_convert > 0:
        logger.info(
            f"Converting {num_to_convert} molar concentration measurements to standard unit (nM). "
            f"{num_unconvertible} measurements cannot be converted (unknown units or missing values)."
        )

        # Apply conversion to convertible rows
        original_dtype = df[value_col].dtype
        converted_values = (
            pd.to_numeric(df.loc[mask_convertible, value_col]) * df.loc[mask_convertible, "conversion_factor"]
        )
        df.loc[mask_convertible, value_col] = converted_values.astype(original_dtype)

        # Update unit label for converted rows
        target_label = "nM"
        df.loc[mask_convertible, unit_col] = target_label

        # Log summary of conversions by original unit
        conversion_summary = (
            df[mask_convertible].groupby("temp_unit")["conversion_factor"].agg(["first", "count"])
        )
        logger.debug("Conversion summary by original unit:")
        for unit, row in conversion_summary.iterrows():
            logger.debug(f"  {unit}: {row['count']} measurements (factor: {row['first']})")
    else:
        logger.info("No molar concentration measurements found that can be converted to standard unit.")

    # Clean up temporary column
    df.drop(columns=["temp_unit"], inplace=True)

    return df


def convert_mass_concentration_units(
    df: pd.DataFrame,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
) -> pd.DataFrame:
    """Convert mass concentration data to standardized unit: ug/mL (micrograms per milliliter).

    This function converts various mass concentration units found in ChEMBL
    (ng/ml, mg/ml, mg/L, pg/ml, etc.) to a single standard unit (ug/mL) to enable
    aggregation across assays with different reporting conventions. This is useful
    for MIC, solubility, and clinical chemistry measurements.

    Rows that cannot be converted (unknown units or missing values) are left unchanged.
    All conversions are logged and can be tracked via the data_processing_comment column
    using the flag_unit_conversion() function.

    Args:
        df: DataFrame containing bioactivity data.
        value_col: Column name containing numeric values to convert. Defaults to 'standard_value'.
        unit_col: Column name containing unit strings. Defaults to 'standard_units'.

    Returns:
        pd.DataFrame: DataFrame with converted values and standardized units where applicable.
            Adds 'conversion_factor' and 'original_unit' columns for transparency.

    Examples:
        >>> df = pd.DataFrame({
        ...     'standard_value': [1000.0, 1.0, 5.0],
        ...     'standard_units': ['ng/ml', 'mg/ml', 'unknown']
        ... })
        >>> converted = convert_mass_concentration_units(df)
        >>> converted['standard_value'].tolist()
        [1.0, 1000.0, 5.0]  # ng/ml and mg/ml converted, unknown unchanged
    """
    if value_col not in df.columns:
        logger.warning(f"Column '{value_col}' not found. Skipping mass concentration unit conversion.")
        return df

    if unit_col not in df.columns:
        logger.warning(f"Column '{unit_col}' not found. Skipping mass concentration unit conversion.")
        return df

    df = df.copy()

    # Normalize unit strings: lowercase and strip whitespace
    df["temp_unit"] = df[unit_col].astype(str).str.lower().str.strip()

    # Define conversion factors to target unit: ug/mL
    # Formula: New_Value = Old_Value * Factor
    conversions = {
        # Micrograms per milliliter variants (target unit)
        "ug/ml": 1.0,
        "ug.ml-1": 1.0,
        "ug ml-1": 1.0,
        # Nanograms per milliliter (1 ng/ml = 0.001 ug/mL)
        "ng/ml": 0.001,
        "ng ml-1": 0.001,
        "ng.ml-1": 0.001,
        # Milligrams per milliliter (1 mg/ml = 1000 ug/mL)
        "mg/ml": 1000.0,
        "mg ml-1": 1000.0,
        "mg.ml-1": 1000.0,
        # Milligrams per liter (1 mg/L = 1 ug/mL)
        "mg/l": 1.0,
        "mg l-1": 1.0,
        "mg.l-1": 1.0,
        # Picograms per milliliter (1 pg/ml = 1e-6 ug/mL)
        "pg/ml": 1e-6,
        "pg ml-1": 1e-6,
        "pg.ml-1": 1e-6,
    }

    # Map conversion factors
    df["conversion_factor"] = df["temp_unit"].map(conversions)

    # Store original unit for transparency before conversion
    df["original_unit"] = df[unit_col].where(df["conversion_factor"].notna())

    # Identify rows that can be converted
    mask_convertible = df["conversion_factor"].notna() & pd.to_numeric(df[value_col], errors="coerce").notna()

    num_to_convert = mask_convertible.sum()
    num_unconvertible = (~mask_convertible).sum()

    if num_to_convert > 0:
        logger.info(
            f"Converting {num_to_convert} mass concentration measurements to standard unit (ug/mL). "
            f"{num_unconvertible} measurements cannot be converted (unknown units or missing values)."
        )

        # Apply conversion to convertible rows
        original_dtype = df[value_col].dtype
        converted_values = (
            pd.to_numeric(df.loc[mask_convertible, value_col]) * df.loc[mask_convertible, "conversion_factor"]
        )
        df.loc[mask_convertible, value_col] = converted_values.astype(original_dtype)

        # Update unit label for converted rows
        target_label = "ug/mL"
        df.loc[mask_convertible, unit_col] = target_label

        # Log summary of conversions by original unit
        conversion_summary = (
            df[mask_convertible].groupby("temp_unit")["conversion_factor"].agg(["first", "count"])
        )
        logger.debug("Conversion summary by original unit:")
        for unit, row in conversion_summary.iterrows():
            logger.debug(f"  {unit}: {row['count']} measurements (factor: {row['first']})")
    else:
        logger.info("No mass concentration measurements found that can be converted to standard unit.")

    # Clean up temporary column
    df.drop(columns=["temp_unit"], inplace=True)

    return df


def convert_dose_units(
    df: pd.DataFrame,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
) -> pd.DataFrame:
    """Convert dose data to standardized unit: mg/kg (milligrams per kilogram body weight).

    This function converts various dose units found in ChEMBL
    (ug/kg, ug.kg-1, mg.kg-1, etc.) to a single standard unit (mg/kg) to enable
    aggregation across in vivo studies with different reporting conventions.

    Note: Molar dose units (umol/kg, nM kg-1) are not converted as they require
    molecular weight information.

    Rows that cannot be converted (unknown units or missing values) are left unchanged.
    All conversions are logged and can be tracked via the data_processing_comment column
    using the flag_unit_conversion() function.

    Args:
        df: DataFrame containing bioactivity data.
        value_col: Column name containing numeric values to convert. Defaults to 'standard_value'.
        unit_col: Column name containing unit strings. Defaults to 'standard_units'.

    Returns:
        pd.DataFrame: DataFrame with converted values and standardized units where applicable.
            Adds 'conversion_factor' and 'original_unit' columns for transparency.

    Examples:
        >>> df = pd.DataFrame({
        ...     'standard_value': [1000.0, 10.0, 5.0],
        ...     'standard_units': ['ug/kg', 'mg/kg', 'unknown']
        ... })
        >>> converted = convert_dose_units(df)
        >>> converted['standard_value'].tolist()
        [1.0, 10.0, 5.0]  # ug/kg converted, mg/kg unchanged, unknown unchanged
    """
    if value_col not in df.columns:
        logger.warning(f"Column '{value_col}' not found. Skipping dose unit conversion.")
        return df

    if unit_col not in df.columns:
        logger.warning(f"Column '{unit_col}' not found. Skipping dose unit conversion.")
        return df

    df = df.copy()

    # Normalize unit strings: lowercase and strip whitespace
    df["temp_unit"] = df[unit_col].astype(str).str.lower().str.strip()

    # Define conversion factors to target unit: mg/kg
    # Formula: New_Value = Old_Value * Factor
    conversions = {
        # Milligrams per kilogram variants (target unit)
        "mg/kg": 1.0,
        "mg.kg-1": 1.0,
        "mg kg-1": 1.0,
        # Micrograms per kilogram (1 ug/kg = 0.001 mg/kg)
        "ug/kg": 0.001,
        "ug.kg-1": 0.001,
        "ug kg-1": 0.001,
    }

    # Map conversion factors
    df["conversion_factor"] = df["temp_unit"].map(conversions)

    # Store original unit for transparency before conversion
    df["original_unit"] = df[unit_col].where(df["conversion_factor"].notna())

    # Identify rows that can be converted
    mask_convertible = df["conversion_factor"].notna() & pd.to_numeric(df[value_col], errors="coerce").notna()

    num_to_convert = mask_convertible.sum()
    num_unconvertible = (~mask_convertible).sum()

    if num_to_convert > 0:
        logger.info(
            f"Converting {num_to_convert} dose measurements to standard unit (mg/kg). "
            f"{num_unconvertible} measurements cannot be converted (unknown units or missing values)."
        )

        # Apply conversion to convertible rows
        original_dtype = df[value_col].dtype
        converted_values = (
            pd.to_numeric(df.loc[mask_convertible, value_col]) * df.loc[mask_convertible, "conversion_factor"]
        )
        df.loc[mask_convertible, value_col] = converted_values.astype(original_dtype)

        # Update unit label for converted rows
        target_label = "mg/kg"
        df.loc[mask_convertible, unit_col] = target_label

        # Log summary of conversions by original unit
        conversion_summary = (
            df[mask_convertible].groupby("temp_unit")["conversion_factor"].agg(["first", "count"])
        )
        logger.debug("Conversion summary by original unit:")
        for unit, row in conversion_summary.iterrows():
            logger.debug(f"  {unit}: {row['count']} measurements (factor: {row['first']})")
    else:
        logger.info("No dose measurements found that can be converted to standard unit.")

    # Clean up temporary column
    df.drop(columns=["temp_unit"], inplace=True)

    return df


def convert_time_units(
    df: pd.DataFrame,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
) -> pd.DataFrame:
    """Convert time data to standardized unit: hr (hours).

    This function converts various time units found in ChEMBL
    (min, s, ms, day) to a single standard unit (hr) to enable aggregation
    across studies with different reporting conventions. This is useful for
    half-life, duration of action, and onset time measurements.

    Rows that cannot be converted (unknown units or missing values) are left unchanged.
    All conversions are logged and can be tracked via the data_processing_comment column
    using the flag_unit_conversion() function.

    Args:
        df: DataFrame containing bioactivity data.
        value_col: Column name containing numeric values to convert. Defaults to 'standard_value'.
        unit_col: Column name containing unit strings. Defaults to 'standard_units'.

    Returns:
        pd.DataFrame: DataFrame with converted values and standardized units where applicable.
            Adds 'conversion_factor' and 'original_unit' columns for transparency.

    Examples:
        >>> df = pd.DataFrame({
        ...     'standard_value': [60.0, 1.0, 5.0],
        ...     'standard_units': ['min', 'day', 'unknown']
        ... })
        >>> converted = convert_time_units(df)
        >>> converted['standard_value'].tolist()
        [1.0, 24.0, 5.0]  # min and day converted, unknown unchanged
    """
    if value_col not in df.columns:
        logger.warning(f"Column '{value_col}' not found. Skipping time unit conversion.")
        return df

    if unit_col not in df.columns:
        logger.warning(f"Column '{unit_col}' not found. Skipping time unit conversion.")
        return df

    df = df.copy()

    # Normalize unit strings: lowercase and strip whitespace
    df["temp_unit"] = df[unit_col].astype(str).str.lower().str.strip()

    # Define conversion factors to target unit: hr
    # Formula: New_Value = Old_Value * Factor
    conversions = {
        # Hours (target unit)
        "hr": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "h": 1.0,
        # Minutes (1 min = 1/60 hr)
        "min": 1.0 / 60.0,
        "minute": 1.0 / 60.0,
        "minutes": 1.0 / 60.0,
        # Seconds (1 s = 1/3600 hr)
        "s": 1.0 / 3600.0,
        "sec": 1.0 / 3600.0,
        "second": 1.0 / 3600.0,
        "seconds": 1.0 / 3600.0,
        # Milliseconds (1 ms = 1/3,600,000 hr)
        "ms": 1.0 / 3600000.0,
        "millisecond": 1.0 / 3600000.0,
        "milliseconds": 1.0 / 3600000.0,
        # Days (1 day = 24 hr)
        "day": 24.0,
        "days": 24.0,
        "d": 24.0,
    }

    # Map conversion factors
    df["conversion_factor"] = df["temp_unit"].map(conversions)

    # Store original unit for transparency before conversion
    df["original_unit"] = df[unit_col].where(df["conversion_factor"].notna())

    # Identify rows that can be converted
    mask_convertible = df["conversion_factor"].notna() & pd.to_numeric(df[value_col], errors="coerce").notna()

    num_to_convert = mask_convertible.sum()
    num_unconvertible = (~mask_convertible).sum()

    if num_to_convert > 0:
        logger.info(
            f"Converting {num_to_convert} time measurements to standard unit (hr). "
            f"{num_unconvertible} measurements cannot be converted (unknown units or missing values)."
        )

        # Apply conversion to convertible rows
        original_dtype = df[value_col].dtype
        converted_values = (
            pd.to_numeric(df.loc[mask_convertible, value_col]) * df.loc[mask_convertible, "conversion_factor"]
        )
        df.loc[mask_convertible, value_col] = converted_values.astype(original_dtype)

        # Update unit label for converted rows
        target_label = "hr"
        df.loc[mask_convertible, unit_col] = target_label

        # Log summary of conversions by original unit
        conversion_summary = (
            df[mask_convertible].groupby("temp_unit")["conversion_factor"].agg(["first", "count"])
        )
        logger.debug("Conversion summary by original unit:")
        for unit, row in conversion_summary.iterrows():
            logger.debug(f"  {unit}: {row['count']} measurements (factor: {row['first']})")
    else:
        logger.info("No time measurements found that can be converted to standard unit.")

    # Clean up temporary column
    df.drop(columns=["temp_unit"], inplace=True)

    return df
