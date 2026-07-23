# Installation

CAPRICHO requires Python 3.10 or later.

## From PyPI (Recommended)

Install the latest stable version from PyPI using pip:

```bash
pip install capricho
```

Or using uv (faster):

```bash
uv pip install capricho
```

## Similarity Search (Optional)

Searching the whole ChEMBL database by fingerprint similarity relies on
[FPSim2](https://github.com/chembl/FPSim2), which is not installed by default:

```bash
pip install "capricho[similarity]"
```

The extra is optional because FPSim2 ships compiled wheels and pulls in PyTables, which are only
needed for {func}`Capricho.chembl.similarity.search_by_similarity`. Looking a compound up by
structure with {func}`Capricho.chembl.similarity.search_by_structure` works without it.

On Python 3.10, pip resolves to FPSim2 0.7.3, the last release supporting that interpreter;
Python 3.11 and later get the current release.

## From GitHub (Development Version)

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/David-Araripe/Capricho.git
```

Or with uv:

```bash
uv pip install git+https://github.com/David-Araripe/Capricho.git
```

## Development Installation

For development purposes, clone the repository and install in editable mode:

```bash
git clone https://github.com/David-Araripe/Capricho.git
cd Capricho
pip install -e ".[dev]"
```

This installs CAPRICHO with development dependencies (linting, formatting) and documentation dependencies.

## Verification

Verify the installation by running:

```bash
capricho --help
```

You should see the main help message with available commands.

## Tab Completion (Optional)

Enable tab completion for command line interface:

```bash
capricho --install-completion
```

This makes it easier to discover available commands and options as you type.
