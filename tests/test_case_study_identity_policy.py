"""Keep historical case-study grouping independent of changing CLI defaults."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize(
    "filename,fetch_count,chirality_flag",
    [
        ("case-1-transparent_quality_control_cross-assay_aggr.ipynb", 1, "-chiral"),
        ("case-2-cyp-inhibition-mt-dataset.ipynb", 2, "--chirality"),
        ("case-3-caco2-permeability-w-unit-standardization.ipynb", 8, "--no-chirality"),
    ],
)
def test_case_study_identity_and_stereo_policies_are_explicit(filename, fetch_count, chirality_flag):
    notebook = json.loads((NOTEBOOKS / filename).read_text())
    fetches = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if re.search(r"capricho get\s+-", source):
            fetches.append(source)
            assert "--compound-equality connectivity" in source
            assert re.search(rf"{re.escape(chirality_flag)}(?:\s|\")", source)
        if re.search(r"capricho prepare\s+-", source):
            assert "--compound-col connectivity" in source
        if cell["cell_type"] == "code" and "clean_data(" in source:
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "clean_data"
                ):
                    identity = next((kw.value for kw in node.keywords if kw.arg == "compound_col"), None)
                    assert isinstance(identity, ast.Constant) and identity.value == "connectivity"
    assert len(fetches) == fetch_count


def test_case_study_benchmarks_pin_original_identity_and_chirality():
    script = (NOTEBOOKS / "performance-profiling" / "benchmark_case_studies.sh").read_text()
    commands = script.split('"${CAPRICHO[@]}" get \\\n')[1:]
    assert len(commands) == 5
    for index, command in enumerate(commands):
        command = command.split("\ndone", 1)[0]
        assert "--compound-equality connectivity" in command
        if index < 3:
            assert "--chirality" in command
            assert "--no-chirality" not in command
        else:
            assert "--no-chirality" in command
