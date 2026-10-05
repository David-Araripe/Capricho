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
