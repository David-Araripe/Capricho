# API Reference

CAPRICHO is primarily designed as a command-line tool, but its core functionality is also available programmatically through Python APIs.

## CLI Module

The main entry point for CAPRICHO commands.

```{eval-rst}
.. automodule:: Capricho.cli.main
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Pipeline

Core data processing workflow that orchestrates fetch → standardize → clean → aggregate.

```{eval-rst}
.. automodule:: Capricho.cli.chembl_data_pipeline
   :members: get_standardize_and_clean_workflow, aggregate_data, re_aggregate_data
   :undoc-members:
   :show-inheritance:
```

## Data Preparation

Filter data based on quality flags and transform to activity matrices for ML.

```{eval-rst}
.. automodule:: Capricho.cli.prepare
   :members: prepare_multitask_data
   :undoc-members:
   :show-inheritance:
```

## ChEMBL Processing

Bioactivity data processing and pChEMBL value calculation.

```{eval-rst}
.. automodule:: Capricho.chembl.processing
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Quality Flags

Functions that flag (rather than remove) problematic data entries.

```{eval-rst}
.. automodule:: Capricho.chembl.data_flag_functions
   :members:
   :undoc-members:
   :show-inheritance:
```

(structure-search)=
## Structure Search

Find compounds in ChEMBL starting from a SMILES string, either by structure identity or by
fingerprint similarity across the whole database. Similarity search requires the optional
`similarity` extra (see [Installation](../installation.md)).

These searches are available through Python; there is no CLI search command. Pass an explicit
ChEMBL release to reuse cached data without checking the latest release:

```python
from Capricho.chembl.similarity import search_by_similarity, search_by_structure

identity = search_by_structure(["CCO", "[13CH3]CO", ""], standardize=False, version=37)
hits = search_by_similarity(["CCO", "CCO"], threshold=0.7, top_k=10, version=37)
query_records = hits.groupby("query_index")  # retains repeated input occurrences
provenance = hits.attrs["capricho_search"]
```

With default standardization, identity means an InChIKey match **after** the ChEMBL parent
pipeline strips salts, neutralizes compounds and removes isotope labels. Use
`standardize=False` to preserve those details in the query key. Invalid and empty inputs retain
an unmatched row. A similarity score of 1.0 is a fingerprint match and does not establish
molecular identity or stereochemical equality.

Search results keep the zero-based input position in `query_index`. Their
`attrs["capricho_search"]` dictionary records the resolved ChEMBL release and software versions;
similarity searches also record fingerprint parameters and the versions used to build the
index. DataFrame attributes are not saved in CSV, so save provenance separately:

```python
import json
from pathlib import Path

hits.to_csv("similarity_hits.csv", index=False)
Path("similarity_hits.provenance.json").write_text(json.dumps(provenance, indent=2))
```

On-disk searches with `n_workers > 1` use multiprocessing. Use `n_workers=1` in notebooks and
stdin sessions. For multiple on-disk workers on platforms using spawn, run from a Python file
with an importable entry point and a main guard:

```python
from Capricho.chembl.similarity import search_by_similarity

if __name__ == "__main__":
    hits = search_by_similarity("CCO", top_k=10, in_memory=False, n_workers=2, version=37)
```

```{eval-rst}
.. automodule:: Capricho.chembl.similarity
   :members: search_by_structure, search_by_similarity, get_and_curate_chembl_compounds
   :undoc-members:
   :show-inheritance:
```

## Analysis Tools

Tools for data quality analysis and comparability studies.

```{eval-rst}
.. automodule:: Capricho.analysis
   :members: explode_assay_comparability, plot_multi_panel_comparability, plot_subset, get_all_comments
   :undoc-members:
   :show-inheritance:
```

## Core Utilities

### Statistical Aggregation

```{eval-rst}
.. automodule:: Capricho.core.stats_make
   :members:
   :undoc-members:
   :show-inheritance:
```

### DataFrame Helpers

```{eval-rst}
.. automodule:: Capricho.core.pandas_helper
   :members: save_dataframe, add_comment
   :undoc-members:
   :show-inheritance:
```

### Binarization

```{eval-rst}
.. automodule:: Capricho.core.binarization
   :members:
   :undoc-members:
   :show-inheritance:
```

## Backends

### Local SQL Backend

```{eval-rst}
.. automodule:: Capricho.chembl.api.downloader
   :members:
   :undoc-members:
   :show-inheritance:
```

### Web API Backend

```{eval-rst}
.. automodule:: Capricho.chembl.api.webresource
   :members:
   :undoc-members:
   :show-inheritance:
```

### Fingerprint Index

```{eval-rst}
.. automodule:: Capricho.chembl.api.fingerprint_index
   :members:
   :undoc-members:
   :show-inheritance:
```

---

*API documentation is generated from source docstrings. See the [CLI Reference](../cli-reference.md)
for functionality exposed as commands; local structure and similarity searches use the Python API.*
