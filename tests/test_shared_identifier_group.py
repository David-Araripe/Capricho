import io

import pandas as pd

from Capricho.cli.chembl_data_pipeline import _warn_info_post_aggregation_repeats
from Capricho.cli.prepare import clean_data
from Capricho.core.pandas_helper import assign_shared_identifier_groups, save_dataframe
from Capricho.logger import logger


def test_assign_shared_identifier_groups_labels_complete_groups_deterministically():
    data = pd.DataFrame(
        {
            "connectivity": ["ZZZ", "AAA", "UNIQUE", "AAA", "ZZZ", "AAA"],
            "target_chembl_id": ["TARGET2", "TARGET1", "TARGET1", "TARGET1", "TARGET2", "TARGET1"],
            "smiles": ["CC", "CCC", "CO", "CCC", "CC", "CCC"],
        }
    )

    result = assign_shared_identifier_groups(data)

    assert str(result["shared_identifier_group"].dtype) == "Int64"
    assert result.columns.get_loc("shared_identifier_group") == result.columns.get_loc("smiles") + 1
    assert result.loc[result["connectivity"] == "AAA", "shared_identifier_group"].eq(1).all()
    assert result.loc[result["connectivity"] == "ZZZ", "shared_identifier_group"].eq(2).all()
    assert result.loc[result["connectivity"] == "UNIQUE", "shared_identifier_group"].isna().all()
    assert result["shared_identifier_group"].value_counts().to_dict() == {1: 3, 2: 2}


def test_assign_shared_identifier_groups_replaces_stale_labels_and_saves(tmp_path):
    data = pd.DataFrame(
        {
            "connectivity": ["AAA", "BBB"],
            "target_chembl_id": ["TARGET1", "TARGET1"],
            "smiles": ["CC", "CCC"],
            "shared_identifier_group": [7, 7],
        }
    )

    result = assign_shared_identifier_groups(data)
    output_path = tmp_path / "annotated.csv"
    save_dataframe(result, output_path)
    saved = pd.read_csv(output_path)

    assert result["shared_identifier_group"].isna().all()
    assert saved["shared_identifier_group"].isna().all()


def test_post_aggregation_message_explains_mutation_split_and_shows_both_rows():
    data = pd.DataFrame(
        {
            "connectivity": ["SAMECONNECTIVITY", "SAMECONNECTIVITY", "UNIQUE"],
            "target_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL1"],
            "mutation": ["WT", "L858R", "WT"],
            "smiles": ["CCO", "CCO", "CCC"],
            "standard_relation": ["=", "=", "="],
            "molecule_chembl_id": ["CHEMBL10", "CHEMBL10", "CHEMBL11"],
            "assay_chembl_id": ["CHEMBL_A", "CHEMBL_B", "CHEMBL_C"],
            "pchembl_value_mean": [6.0, 8.0, 7.0],
        }
    )
    data = assign_shared_identifier_groups(data)
    stream = io.StringIO()
    sink = logger.add(stream, format="{level}: {message}", level="INFO")
    try:
        _warn_info_post_aggregation_repeats(data, extra_id_cols=[], aggregate_mutants=False)
    finally:
        logger.remove(sink)

    message = stream.getvalue()
    assert (
        "found 2 separate activity rows with 1 repeated `connectivity` + `target_chembl_id` combination"
        in message
    )
    assert "varying fields: `mutation`" in message
    assert "valid distinct readouts" in message
    assert "downstream tasks use target only" in message
    assert "--id-columns mutation" in message
    assert "labels rows sharing a combination (NaN otherwise)" in message
    assert "collision" not in message
    assert "shared_identifier_group" in message
    assert "WT" in message
    assert "L858R" in message
    assert "molecule_chembl_id" not in message
    assert "assay_chembl_id" not in message


def test_post_aggregation_message_describes_admet_id_columns():
    data = pd.DataFrame(
        {
            "connectivity": ["COMPOUND1", "COMPOUND1", "COMPOUND2", "COMPOUND2"],
            "target_chembl_id": ["CHEMBL1"] * 4,
            "standard_units": ["10^-6 cm/s", "10'-6/cm", "10^-6 cm/s", "10^-6 cm/s"],
            "assay_cell_type": ["MDCK", "MDCK", "MDCK", "MDCK-MDR1"],
            "smiles": ["CCO", "CCO", "CCC", "CCC"],
            "standard_relation": ["="] * 4,
            "standard_value_mean": [40.0, 40.0, 4.37, 3.64],
        }
    )
    data = assign_shared_identifier_groups(data)
    stream = io.StringIO()
    sink = logger.add(stream, format="{message}", level="INFO")
    try:
        _warn_info_post_aggregation_repeats(
            data,
            extra_id_cols=["standard_units", "assay_cell_type"],
            aggregate_mutants=True,
            value_col="standard_value",
        )
    finally:
        logger.remove(sink)

    message = stream.getvalue()
    assert "4 separate activity rows with 2 repeated" in message
    assert "varying fields: `standard_units`, `assay_cell_type`" in message
    assert "--id-columns standard_units,assay_cell_type" in message
    assert "mutation" not in message


def test_clean_data_uses_selected_compound_identifier():
    data = pd.DataFrame(
        {
            "connectivity": ["SAME", "SAME"],
            "inchikey": ["KEY1", "KEY2"],
            "target_chembl_id": ["TARGET1", "TARGET1"],
            "smiles": ["C[C@H](O)Cl", "C[C@@H](O)Cl"],
            "pchembl_value": [6.0, 7.0],
            "data_dropping_comment": ["", ""],
        }
    )

    cleaned = clean_data(data, compound_col="inchikey")

    assert cleaned["shared_identifier_group"].isna().all()


def test_clean_data_recalculates_groups_after_removing_a_row():
    data = pd.DataFrame(
        {
            "connectivity": ["AAA", "AAA", "BBB"],
            "target_chembl_id": ["TARGET1", "TARGET1", "TARGET1"],
            "smiles": ["CC", "CC", "CCC"],
            "pchembl_value": [6.0, 7.0, 5.0],
            "data_dropping_comment": ["", "remove", ""],
        }
    )
    data = assign_shared_identifier_groups(data)
    assert data["shared_identifier_group"].notna().sum() == 2

    cleaned = clean_data(data, drop_flags=["remove"])

    assert len(cleaned) == 2
    assert cleaned["shared_identifier_group"].isna().all()
