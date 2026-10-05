"""Regression checks for Python modernization without changing curation semantics."""

from typing import get_type_hints

import pandas as pd
import pytest

from Capricho.chembl.data_flag_functions import flag_inter_document_duplication
from Capricho.cli import chembl_data_pipeline
from Capricho.core.pandas_helper import add_comment


class FetchIntercepted(Exception):
    """Stop before curation so tests can inspect the backend's filter arguments."""


@pytest.mark.parametrize(
    "filters", [{}, {"confidence_scores": None, "standard_relation": None, "assay_types": None}]
)
def test_default_filters_are_fresh_and_explicit_none_disables_filtering(monkeypatch, filters):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        raise FetchIntercepted

    monkeypatch.setattr(chembl_data_pipeline, "get_bioactivities_workflow", fetch)
    with pytest.raises(FetchIntercepted):
        chembl_data_pipeline.get_standardize_and_clean_workflow(**filters)

    expected = filters or {
        "confidence_scores": [7, 8, 9],
        "standard_relation": ["="],
        "assay_types": ["B", "F"],
    }
    for name, value in expected.items():
        assert calls[0][name] == value
        if value is not None:
            calls[0][name].append("mutated")

    with pytest.raises(FetchIntercepted):
        chembl_data_pipeline.get_standardize_and_clean_workflow(**filters)
    for name, value in expected.items():
        assert calls[1][name] == value


def test_duplicate_document_default_and_explicit_none_remain_distinct():
    source = pd.DataFrame(
        {
            "molecule_chembl_id": ["MOL1", "MOL1"],
            "standard_relation": ["=", "="],
            "document_chembl_id": ["DOC1", "DOC1"],
        }
    )
    default = flag_inter_document_duplication(source.copy(), key_subset=["molecule_chembl_id"])
    assert (
        "data_processing_comment" not in default or default["data_processing_comment"].fillna("").eq("").all()
    )

    without_document_check = flag_inter_document_duplication(
        source.copy(), key_subset=["molecule_chembl_id"], diff_subset=None
    )
    assert (
        without_document_check["data_processing_comment"]
        .str.contains("pChEMBL Duplication Across Documents")
        .all()
    )


def test_identifier_mapping_rejects_truncated_conversion_results(monkeypatch):
    monkeypatch.setattr(chembl_data_pipeline, "_convert_smiles_to_identifier", lambda *args: ["one-key"])
    with pytest.raises(ValueError, match="zip"):
        chembl_data_pipeline._identifier_map(pd.Series(["CCO", "CCN"]), "inchikey")


def test_comment_callable_annotation_can_be_resolved():
    assert get_type_hints(add_comment)["criteria_func"] is not None
