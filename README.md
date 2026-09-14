<div align="center">
  <img src="https://raw.githubusercontent.com/David-Araripe/CompoundMapper/master/logo.svg" alt="CAPRICHO logo" width=240>
  <p><strong>The ChEMBL data curator that flags issues instead of silently dropping them.</strong></p>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat-square&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](https://opensource.org/licenses/MIT)
[![GH Actions](https://github.com/David-Araripe/Capricho/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/David-Araripe/Capricho/actions)

</div>

> Inspired in the Portuguese word "*capricho*" [🔊](https://ipa-reader.com/?text=ka%CB%88p%C9%BEi.%CA%83u&voice=Ricardo). Doing someting *with capricho* means doing it *meticulously*, *with care* and *attention to detail*.

CAPRICHO (**C**hEMBL **A**ggregation **P**ackage with **R**obust **I**nspection and **C**uration **H**andling **O**ptions) is a Python package that streamlines fetching, curating, and aggregating ChEMBL data into a machine learning-ready format for drug discovery in a flexible and reproducible manner. Instead of making opiniated decisions on the source data, CAPRICHO curates it based on several quality control filters that can be chosen by the user. Its guiding principle is to never silently drop data. Entries that don't meet the criteria are marked, allowing the user to analyze how each curation step affects the comparability of assay readouts for the same compound.

## 🎯 Goals

The development of CAPRICHO is guided by two core principles:
- **Transparency Above All**: Data curation should never be a black box. Removed data points should be saved to be scrutinized by the user and the original data should be always preserved to ensure data integrity.
- **Flexibility by Design**: Every modeling project is unique. Aggregation should be stratified by ChEMBL metadata columns, aggregating repeated compound measurements only within the scope you define as comparable.

## ✨ Features:

- Data retrieval by any ChEMBL identifier (molecule IDs, target IDs, assay IDs, or document IDs)
- ADMET data curation support with unit conversion and non-pChEMBL aggregation
- Quality control through data flagging — never silently drops data
- Customizable filtering options with max curation standards introduced by [Landrum & Riniker (2024)](https://doi.org/10.1021/acs.jcim.4c00049)
- Configurable data aggregation options
- Binary classification support with censored data handling
- Save a fetching and processing recipe for reproducibility
- Command-line interface for easy use

## ⚙️ Installation

With pip, into the environment you work in — the `capricho` command comes with it:

```shell
python -m pip install capricho
```

With [uv](https://docs.astral.sh/uv/), either install it as a standalone command, available anywhere:

```shell
uv tool install capricho
```

or add it to a project, where it is importable as a library too and runs as `uv run capricho`:

```shell
uv add capricho
```

Check the installation with `capricho -h`. For the development version, swap `capricho` for `git+https://github.com/David-Araripe/Capricho.git` in any of the commands above.

## 🚀 Quick Start

### Basic Usage
```bash
# Download ChEMBL database
capricho download

# ...or use a ChEMBL SQLite dump you already have, read where it lies
capricho download --set-from-path /path/to/chembl_35.db

# Get bioactivity data for EGFR
capricho get --target-ids CHEMBL203 --output-path egfr_data.csv

# Get high-confidence data for multiple targets
capricho get --target-ids CHEMBL203,CHEMBL204 --confidence-scores 8,9 --output-path results.csv
```

### Tab Completion

Our CLI supports tab completion for commands and options. To enable it, run the following command in your terminal:

```bash
capricho --install-completion
```

### Key Features
- **Five main commands**: `download`, `explore`, `get`, `prepare`, `binarize`
- **Flexible filtering**: By confidence, assay type, bioactivity type
- **Transparent processing**: All filtering steps are logged and flagged
- **Reproducible workflows**: Automatic recipe generation
- **Multiple backends**: Local SQL or web API
- **Binary classification support**: Convert continuous activity values to binary labels

## 📖 Documentation

For comprehensive documentation including detailed CLI options, advanced usage, tutorials, and API reference, visit our [full documentation](https://capricho.readthedocs.io/en/latest/).

**Quick Links:**
- [Installation Guide](https://capricho.readthedocs.io/en/latest/installation.html)
- [Quick Start](https://capricho.readthedocs.io/en/latest/quickstart.html)
- [CLI Reference](https://capricho.readthedocs.io/en/latest/cli-reference.html)
- [Key Concepts](https://capricho.readthedocs.io/en/latest/concepts.html)
- [ADMET Data Guide](https://capricho.readthedocs.io/en/latest/guides/admet-data.html)
- [API Reference](https://capricho.readthedocs.io/en/latest/api/index.html)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.