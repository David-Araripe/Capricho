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

## Structure Search

Find compounds in ChEMBL starting from a SMILES string, either by structure identity or by
fingerprint similarity across the whole database. Similarity search requires the optional
`similarity` extra (see [Installation](../installation.md)).

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

*Note: API documentation is automatically generated from docstrings in the source code. For the most comprehensive and up-to-date information, refer to the [CLI Reference](../cli-reference.md) which exposes all functionality.*
