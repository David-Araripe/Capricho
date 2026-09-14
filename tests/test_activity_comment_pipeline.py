"""Integration tests for the activity-comment review flag."""

import pandas as pd

from Capricho.analysis import DroppingComment
from Capricho.chembl.data_flag_functions import flag_censored_activity_comment
from Capricho.cli.chembl_data_pipeline import aggregate_data
from Capricho.cli.prepare import clean_data
from Capricho.core.default_fields import DATA_DROPPING_COMMENT, DATA_PROCESSING_COMMENT
from Capricho.flag_report import summarize_flags

FLAG = DroppingComment.ACTIVITY_COMMENT_REVIEW.value


def build_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activity_id": [1, 2],
            "molecule_chembl_id": ["CHEMBL25"] * 2,
            "target_chembl_id": ["CHEMBL205"] * 2,
            "standard_smiles": ["CC(=O)Oc1ccccc1C(=O)O"] * 2,
            "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"] * 2,
            "pchembl_value": [4.94, 4.68],
            "standard_relation": ["=", "="],
            "standard_value": [11403.0, 20942.0],
            "activity_comment": ["Not Active", None],
            "mutation": ["WT"] * 2,
            "assay_chembl_id": ["ASSAY1"] * 2,
            "standard_type": ["IC50"] * 2,
            "assay_description": ["Test assay"] * 2,
            "assay_type": ["B"] * 2,
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
            DATA_DROPPING_COMMENT: [""] * 2,
            DATA_PROCESSING_COMMENT: [""] * 2,
        }
    )


def aggregated_frame() -> pd.DataFrame:
    return aggregate_data(
        flag_censored_activity_comment(build_measurements()),
        chirality=False,
        compound_equality="connectivity",
    )


def test_flag_and_source_annotation_survive_aggregation():
    result = aggregated_frame()

    assert "Not Active" in "".join(result["activity_comment"].astype(str))
    assert FLAG in "".join(result[DATA_DROPPING_COMMENT].astype(str))
    relations = "".join(result["standard_relation"].astype(str)).replace("|", "")
    assert relations and set(relations) == {"="}
    assert "standard_relation" not in "".join(result[DATA_PROCESSING_COMMENT].astype(str))


def test_null_activity_comments_aggregate_cleanly():
    measurements = build_measurements()
    measurements["activity_comment"] = None

    result = aggregate_data(measurements, chirality=False, compound_equality="connectivity")

    assert len(result) == 1
    assert FLAG not in "".join(result[DATA_DROPPING_COMMENT].astype(str))


def test_activity_comment_remains_optional_for_older_input():
    measurements = build_measurements().drop(columns=["activity_comment"])

    result = aggregate_data(measurements, chirality=False, compound_equality="connectivity")

    assert len(result) == 1
    assert "activity_comment" not in result.columns


def test_flag_is_reported_at_measurement_level():
    summary = summarize_flags(flag_censored_activity_comment(build_measurements()))
    row = summary[summary["flag"] == FLAG].iloc[0]

    assert row["n"] == 1
    assert row["pct"] == 50.0


def test_removal_is_opt_in():
    aggregated = aggregated_frame()

    assert "Not Active" in "".join(clean_data(aggregated)["activity_comment"].astype(str))
    assert "Not Active" not in "".join(
        clean_data(aggregated, drop_flags=[FLAG])["activity_comment"].astype(str)
    )
