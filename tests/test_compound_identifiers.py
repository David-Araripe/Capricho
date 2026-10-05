import io
import warnings

import pandas as pd
import pytest

from Capricho.analysis import deaggregate_data
from Capricho.cli import chembl_data_pipeline
from Capricho.cli.chembl_data_pipeline import (
    _warn_info_post_aggregation_repeats,
    aggregate_data,
    re_aggregate_data,
)
from Capricho.cli.prepare import prepare_multitask_data
from Capricho.logger import logger


def _stereoisomer_source():
    source = (
        pd.read_csv("tests/resources/ADORA3_data_not_aggregated.csv")
        .iloc[[0, 0]]
        .copy()
        .reset_index(drop=True)
    )
    source["standard_smiles"] = ["C[C@H](O)Cl", "C[C@@H](O)Cl"]
    source["canonical_smiles"] = source["standard_smiles"]
    source["activity_id"] = [900001, 900002]
    source["molecule_chembl_id"] = ["MOL1", "MOL2"]
    source["assay_chembl_id"] = ["ASSAY1", "ASSAY2"]
    source["pchembl_value"] = [6.0, 7.0]
    return source


@pytest.mark.parametrize("identifier", ["inchi", "inchikey"])
def test_aggregate_data_supports_full_inchi_identifiers(identifier):
    result = aggregate_data(_stereoisomer_source(), chirality=False, compound_equality=identifier)

    assert len(result) == 2
    assert result.columns[:2].tolist() == ["connectivity", identifier]
    assert result[identifier].nunique() == 2
    assert result["connectivity"].nunique() == 1
    assert result["shared_identifier_group"].isna().all()
    if identifier == "inchi":
        assert result[identifier].str.startswith("InChI=1S/").all()
    else:
        assert result[identifier].str.split("-").str.len().eq(3).all()
        assert result[identifier].str[:14].eq(result["connectivity"]).all()

    matrix = prepare_multitask_data(
        result,
        task_col="target_chembl_id",
        value_col="pchembl_value_mean",
        compound_col=identifier,
        smiles_col="smiles",
    )
    assert matrix.index.name == identifier
    assert len(matrix) == 2


def test_connectivity_equality_merges_stereoisomers():
    result = aggregate_data(_stereoisomer_source(), chirality=False, compound_equality="connectivity")

    assert len(result) == 1


@pytest.mark.parametrize("identifier", ["connectivity", "inchi", "inchikey"])
def test_identifier_is_calculated_once_per_distinct_smiles(monkeypatch, identifier):
    source = _stereoisomer_source()
    source["standard_smiles"] = "CCO"
    source["canonical_smiles"] = "CCO"
    calls = []
    convert = chembl_data_pipeline._convert_smiles_to_identifier

    def recording_convert(smiles, identifier, **kwargs):
        calls.append((list(smiles), identifier))
        return convert(smiles, identifier, **kwargs)

    monkeypatch.setattr(chembl_data_pipeline, "_convert_smiles_to_identifier", recording_convert)

    result = aggregate_data(source, chirality=False, compound_equality=identifier)

    assert len(result) == 1
    assert calls == [(["CCO"], identifier)]
    assert result["connectivity"].iloc[0] == "LFQSCWFLJHTTHZ"


def test_shared_identifier_message_uses_selected_inchi_column():
    data = pd.DataFrame(
        {
            "connectivity": ["SAME", "SAME"],
            "inchi": ["InChI=1S/example", "InChI=1S/example"],
            "target_chembl_id": ["CHEMBL1", "CHEMBL1"],
            "mutation": ["WT", "MUTANT"],
            "smiles": ["CCO", "CCO"],
            "standard_relation": ["=", "="],
            "pchembl_value_mean": [6.0, 7.0],
        }
    )
    stream = io.StringIO()
    sink = logger.add(stream, format="{message}", level="INFO")
    try:
        _warn_info_post_aggregation_repeats(
            data,
            extra_id_cols=[],
            aggregate_mutants=False,
            compound_equality="inchi",
        )
    finally:
        logger.remove(sink)

    message = stream.getvalue()
    assert "repeated `inchi` + `target_chembl_id` combination" in message
    assert "repeated `connectivity` + `target_chembl_id`" not in message
    assert "shared_identifier_group" in message
    assert "--compound-col inchi" in message


def test_reaggregate_data_retains_selected_inchikey():
    source = pd.read_csv("tests/resources/ADORA3_data_not_aggregated.csv").head(2)
    aggregated = aggregate_data(source, chirality=False, compound_equality="inchikey")

    result = re_aggregate_data(
        deaggregate_data(aggregated),
        chirality=False,
        compound_equality="inchikey",
    )

    assert result.columns[:2].tolist() == ["connectivity", "inchikey"]
    assert result["inchikey"].notna().all()


def test_cli_enums_expose_full_inchi_identifiers():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The 'is_flag'.*")
        from Capricho.cli.main import CompoundEquality, CompoundIdColumn

    assert {"inchi", "inchikey"}.issubset(member.value for member in CompoundEquality)
    assert {"inchi", "inchikey"}.issubset(member.value for member in CompoundIdColumn)


@pytest.fixture
def fetched_stereoisomers(monkeypatch):
    # Start before standardization: aggregation-only tests cannot detect stereo loss
    # in the fetched canonical_smiles -> standard_smiles conversion (issue #21).
    # Controlled measurements for a simple enantiomer pair in the same assay.
    # No precomputed standard_smiles: the real standardizer must preserve the stereo.
    source = pd.DataFrame(
        {
            "activity_id": [900001, 900002],
            "molecule_chembl_id": ["MOL1", "MOL2"],
            "canonical_smiles": [
                "C[C@H](O)Cl",
                "C[C@@H](O)Cl",
            ],
            "pchembl_value": [6.0, 7.0],
            "standard_value": [1000.0, 100.0],
            "standard_type": "IC50",
            "standard_units": "nM",
            "standard_relation": "=",
            "assay_chembl_id": "ASSAY1",
            "target_chembl_id": "CHEMBL240",
            "target_organism": "Homo sapiens",
            "mutation": "WT",
            "document_chembl_id": "DOC1",
            "doc_type": "PUBLICATION",
            "doi": "",
            "journal": "",
            "year": 2024,
            "chembl_release": 37,
            "data_dropping_comment": "",
            "data_processing_comment": "",
        }
    )
    monkeypatch.setattr(chembl_data_pipeline, "get_bioactivities_workflow", lambda **kwargs: source.copy())
    standardizer = chembl_data_pipeline.ChemStandardizer

    def serial_standardizer(**kwargs):
        return standardizer(**{**kwargs, "n_jobs": 1, "progress": False})

    monkeypatch.setattr(chembl_data_pipeline, "ChemStandardizer", serial_standardizer)
    return source


@pytest.mark.parametrize("identifier", ["inchi", "inchikey", "smiles", "connectivity", "mixed_fp"])
@pytest.mark.parametrize("flags", [[], ["--no-chirality"], ["--chirality"]])
def test_cli_stereo_policy_before_aggregation(fetched_stereoisomers, tmp_path, identifier, flags):
    from typer.testing import CliRunner

    from Capricho.cli.main import app

    output = tmp_path / "stereoisomers.csv"
    result = CliRunner().invoke(
        app,
        [
            "get",
            "--molecule-ids",
            ",".join(fetched_stereoisomers["molecule_chembl_id"]),
            "--bioactivity-type",
            "IC50",
            "--chembl-version",
            "37",
            "--compound-equality",
            identifier,
            "--skip-recipe",
            "-o",
            str(output),
            *flags,
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    preserves_stereo = identifier in {"inchi", "inchikey", "smiles"} or "--chirality" in flags
    standardized = pd.read_csv(tmp_path / "stereoisomers_not_aggregated.csv")
    assert len(standardized) == 2
    assert dict(zip(standardized["molecule_chembl_id"], standardized["pchembl_value"])) == {
        "MOL1": 6.0,
        "MOL2": 7.0,
    }
    assert standardized["standard_smiles"].str.contains("@").eq(preserves_stereo).all()
    assert standardized["standard_smiles"].nunique() == (2 if preserves_stereo else 1)

    separates_stereo = identifier != "connectivity" and preserves_stereo
    aggregated = pd.read_csv(output)
    assert len(aggregated) == (2 if separates_stereo else 1)
    assert aggregated["connectivity"].nunique() == 1
    assert aggregated["smiles"].str.contains("@").eq(separates_stereo).all()
    if separates_stereo:
        assert dict(zip(aggregated["molecule_chembl_id"], aggregated["pchembl_value_mean"])) == {
            "MOL1": 6.0,
            "MOL2": 7.0,
        }
        assert aggregated["pchembl_value_counts"].eq(1).all()
    else:
        merged = aggregated.iloc[0]
        assert set(merged["molecule_chembl_id"].split("|")) == {"MOL1", "MOL2"}
        assert set(merged["activity_id"].split("|")) == {"900001", "900002"}
        assert merged["pchembl_value_counts"] == 2
        # Pin the current geometric-mean aggregation policy, not just the row count.
        assert merged["pchembl_value_mean"] == pytest.approx((6.0 * 7.0) ** 0.5)
    if identifier in {"inchi", "inchikey", "smiles"}:
        assert aggregated[identifier].nunique() == 2
    if identifier == "inchi":
        assert aggregated[identifier].str.contains("/t").all()
    elif identifier == "inchikey":
        assert aggregated[identifier].str.split("-").str[1].nunique() == 2
