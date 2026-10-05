# Installation

CAPRICHO requires Python 3.10 or later. Python 3.10–3.14 are tested in CI.

## With pip

Install into the environment you work in, be it a conda/mamba environment or a
virtual environment:

```bash
python -m pip install capricho
```

The `capricho` command comes with it, and is available whenever that environment
is active.

## With uv

[uv](https://docs.astral.sh/uv/) does not install into an activated environment
by default, so `uv pip install` alone can leave `capricho` off your `PATH` (see
uv's [guide to environments](https://docs.astral.sh/uv/pip/environments/)).
Install it as a standalone command instead, available anywhere:

```bash
uv tool install capricho
```

or add it to a project, where it is importable as a library too and runs as
`uv run capricho`:

```bash
uv add capricho
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

The first similarity search downloads the selected release's SQLite database and FPSim2
`.h5` index if they are not cached. Registering an existing SQLite database avoids its download,
but the fingerprint index must still be cached or downloaded separately. Subsequent searches
can run offline when both files are present and `version=` is explicit. Recent releases publish
the index; older releases may not provide one.

The default loads fingerprints into memory (roughly 1 GB). Use `in_memory=False` to search
from disk. Keep `n_workers=1` in notebooks; multiple on-disk workers require an importable,
guarded Python entry point. See {ref}`Structure Search <structure-search>` for examples,
normalization behavior and result provenance.

## From GitHub (Development Version)

Swap `capricho` for the repository URL in any of the commands above:

```bash
python -m pip install git+https://github.com/David-Araripe/Capricho.git
uv tool install git+https://github.com/David-Araripe/Capricho.git
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
