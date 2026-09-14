"""Module containing the command line interface to get data from ChEMBL"""

import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import typer
from typing_extensions import Annotated

from .. import __version__
from ..logger import logger, setup_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from chembl_downloader import latest  # noqa: F401

    from ..chembl.api.downloader import check_and_download_chembl_db  # noqa: F401
    from ..chembl.api.sql_explorer import explorer_main  # noqa: F401
    from .chembl_data_pipeline import (  # noqa: F401
        aggregate_data,
        get_standardize_and_clean_workflow,
    )

DEFAULTS = {
    "molecule_ids": [],
    "target_ids": [],
    "assay_ids": [],
    "document_ids": [],
    "calculate_pchembl": False,
    "output_path": "chembl_data.csv",
    "confidence_scores": [7, 8, 9],
    "bioactivity_type": ["Potency", "Kd", "Ki", "IC50", "AC50", "EC50"],
    "chirality": False,
    "drop_unassigned_chiral": False,
    "curate_annotation_errors": True,
    "standard_relation": ["="],
    "standard_units": None,
    "assay_types": ["B", "F"],
    "log_level": "info",
    "chembl_release": None,
    "metadata_columns": [],
    "id_columns": [],
    "skip_not_aggregated": False,
    "aggregate_mutants": False,
    "skip_recipe": False,
    "chembl_backend": "downloader",
    "chembl_version": None,
    "require_doc_date": False,
    "max_assay_size": None,
    "min_assay_size": None,
    "min_assay_overlap": 0,
    "strict_mutant_removal": False,
    "compound_equality": "connectivity",
    "value_column": "pchembl_value",
}

DEFAULT_FALSE_ARGS = [
    "calculate_pchembl",
    "chirality",
    "drop_unassigned_chiral",
    "skip_not_aggregated",
    "aggregate_mutants",
    "skip_recipe",
    "require_doc_date",
    "strict_mutant_removal",
]

DEFAULT_TRUE_ARGS = [
    "curate_annotation_errors",
]


app = typer.Typer(
    name="CAPRICHO",
    help="A ChEMBL data curator that flags questionable entries instead of silently dropping them.",
    no_args_is_help=True,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["--help", "-h"], "max_content_width": 88},
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


def csv_string(value: Optional[str]) -> List[str]:
    """Parses a comma-separated string into a list of strings."""
    logger.debug(f"Parsing CSV string type= {type(value)}, value= {value}")
    if isinstance(value, list):
        return value
    elif value is None:
        raise ValueError("Value cannot be None")
    return [item.strip() for item in value.split(",")]


def csv_intergers(value: Optional[str]) -> List[int]:
    """Parses a comma-separated string into a list of integers."""
    logger.debug(f"Parsing CSV int type={type(value)}, value={value}")
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, int)]
    elif isinstance(value, int):
        return [value]
    elif value is None:
        raise ValueError("Value cannot be None")
    return [int(item.strip()) for item in value.split(",") if item.strip().isdigit()]


class LogLevel(str, Enum):
    trace = "trace"
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


class ChemblBackend(str, Enum):
    downloader = "downloader"
    webresource = "webresource"


class CompoundEquality(str, Enum):
    mixed_fp = "mixed_fp"
    connectivity = "connectivity"
    inchi = "inchi"
    inchikey = "inchikey"
    smiles = "smiles"


class CompoundIdColumn(str, Enum):
    connectivity = "connectivity"
    inchi = "inchi"
    inchikey = "inchikey"
    smiles = "smiles"


class ValueColumn(str, Enum):
    pchembl_value = "pchembl_value"
    standard_value = "standard_value"


class ConflictResolution(str, Enum):
    drop = "drop"
    relation = "relation"
    confidence = "confidence"
    majority = "majority"


class ExploreFormat(str, Enum):
    markdown = "markdown"
    csv = "csv"


@app.callback()
def main(
    ctx: typer.Context,  # noqa: F821
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "-log",
            "--log-level",
            help="Set the logging level.",
            case_sensitive=False,
        ),
    ] = DEFAULTS.get("log_level", "info"),
):
    """
    Manage the ChEMBL data-wrangling workflow.
    """
    ctx.obj = {}
    logger.info(f"Log level set to {log_level.value.upper()}")
    setup_logger(level=log_level.upper())


@app.command()
def download(
    ctx: typer.Context,  # noqa: F821
    version: Annotated[
        Optional[int],
        typer.Option("--version", "-v", help="ChEMBL version to download. Defaults to the latest."),
    ] = None,
    prefix: Annotated[
        Optional[str],
        typer.Option(
            "--prefix",
            "-p",
            help="Custom pystow storage path. Defaults to None, saving to ~/.data/chembl/.",
        ),
    ] = None,
    set_from_path: Annotated[
        Optional[Path],
        typer.Option(
            "--set-from-path",
            help=(
                "Use a ChEMBL SQLite database you already have instead of downloading one. "
                "Takes a chembl_<version>.db file or a directory holding one or more of them, "
                "and registers every release found unless --version narrows it down."
            ),
        ),
    ] = None,
    unset_path: Annotated[
        bool,
        typer.Option(
            "--unset-path",
            help="Forget the database registered with --set-from-path for the given version.",
        ),
    ] = False,
):
    """Download ChEMBL SQL database using chembl_downloader.

    Alternatively, point CAPRICHO at a database already on the system with --set-from-path;
    every later command reads that file in place, with no download.
    """
    from ..chembl.api.downloader import (
        check_and_download_chembl_db,
        set_chembl_db_path,
        unset_chembl_db_path,
    )

    if set_from_path is not None and unset_path:
        raise typer.BadParameter("--set-from-path and --unset-path cannot be combined.")

    if set_from_path is not None:
        if prefix is not None:
            raise typer.BadParameter(
                "--prefix sets where CAPRICHO downloads to, so it does not apply to a database "
                "registered with --set-from-path."
            )
        databases = set_chembl_db_path(set_from_path, version=version)
        logger.info(f"Registered ChEMBL release(s): {', '.join(sorted(databases))}")
        raise typer.Exit()

    if unset_path:
        unset_chembl_db_path(version=version)
        raise typer.Exit()

    logger.info(f"Starting ChEMBL download command for version: {version or 'latest'}")
    check_and_download_chembl_db(prefix=prefix.split("/") if prefix else None, version=version)
    raise typer.Exit()


@app.command()
def explore(
    ctx: typer.Context,  # noqa: F821
    version: Annotated[
        Optional[int], typer.Option("--version", "-v", help="ChEMBL version to use. Defaults to the latest.")
    ] = None,
    list_tables: Annotated[
        bool, typer.Option("--list-tables", "-list", help="List all tables within the SQL database and exit.")
    ] = False,
    table: Annotated[Optional[str], typer.Option("--table", "-t", help="Explore a specific table.")] = None,
    search_column: Annotated[
        Optional[str],
        typer.Option(
            "--search-column", "-search", help="Search for tables containing a column name pattern."
        ),
    ] = None,
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="Run a custom SQL query.")] = None,
    fmt: Annotated[
        ExploreFormat,
        typer.Option("--format", "-f", help="Console output format for tables."),
    ] = ExploreFormat.markdown,
    output_path: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Save primary result DataFrame to file (format from extension)."),
    ] = None,
    colorize: Annotated[
        bool,
        typer.Option("--colorize/--no-colorize", help="ANSI color cycling on console table rows."),
    ] = False,
):
    """
    Explore the downloaded ChEMBL SQL database.

    For a visual inspection of the latest ChEMBL schema, see: https://www.ebi.ac.uk/chembl/db_schema
    """
    from ..chembl.api.sql_explorer import explorer_main

    logger.info("Starting ChEMBL explore command.")
    explorer_main(
        version=version,
        list_tables=list_tables,
        table=table,
        search_column=search_column,
        query=query,
        fmt=fmt.value,
        output_path=output_path,
        colorize=colorize,
    )
    raise typer.Exit()


@app.command(name="get", no_args_is_help=True)
def get_data(
    ctx: typer.Context,
    # --- Input ID Arguments ---
    molecule_ids: Annotated[
        Optional[str],
        typer.Option(
            "-mids",
            "--molecule-ids",
            parser=csv_string,
            help="ChEMBL molecule IDs, comma-separated.",
            show_default=True,
            metavar="ID,ID,...",
        ),
    ] = DEFAULTS["molecule_ids"],
    target_ids: Annotated[
        Optional[str],
        typer.Option(
            "-tids",
            "--target-ids",
            parser=csv_string,
            help="ChEMBL target IDs, comma-separated.",
            show_default=True,
            metavar="ID,ID,...",
        ),
    ] = DEFAULTS["target_ids"],
    assay_ids: Annotated[
        Optional[str],
        typer.Option(
            "-asids",
            "--assay-ids",
            parser=csv_string,
            help="ChEMBL assay IDs, comma-separated.",
            show_default=True,
            metavar="ID,ID,...",
        ),
    ] = DEFAULTS["assay_ids"],
    document_ids: Annotated[
        Optional[str],
        typer.Option(
            "-dids",
            "--document-ids",
            parser=csv_string,
            help="ChEMBL document IDs, comma-separated.",
            show_default=True,
            metavar="ID,ID,...",
        ),
    ] = DEFAULTS["document_ids"],
    # --- Configuration Arguments ---
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-path",
            help="Path to save the output files.",
        ),
    ] = DEFAULTS["output_path"],
    confidence_scores: Annotated[
        str,
        typer.Option(
            "-c",
            "--confidence-scores",
            help="Confidence scores to filter, comma-separated.",
            parser=csv_intergers,
            metavar="SCORES[1-9]",
            show_default=True,
        ),
    ] = DEFAULTS["confidence_scores"],
    bioactivity_type: Annotated[
        Optional[str],
        typer.Option(
            "-biotype",
            "--bioactivity-type",
            parser=csv_string,
            help="Bioactivity types to filter, comma-separated. If not specified, fetches all types.",
            metavar="Ki,Kd,...",
        ),
    ] = None,
    standard_relation: Annotated[
        str,
        typer.Option(
            "-rel",
            "--standard-relation",
            parser=csv_string,
            help="Filter by standard relation, comma-separated.",
            metavar="=,>,<",
        ),
    ] = DEFAULTS["standard_relation"],
    standard_units: Annotated[
        Optional[str],
        typer.Option(
            "-units",
            "--standard-units",
            parser=csv_string,
            help="Filter by standard units, comma-separated. E.g., '%' for percent inhibition.",
            metavar="nM,uM,µM,mM",
        ),
    ] = None,
    assay_types: Annotated[
        str,
        typer.Option(
            "-at",
            "--assay-types",
            parser=csv_string,
            help="Assay types (B, F, A, T, P), comma-separated.",
            metavar="B,F",
        ),
    ] = DEFAULTS["assay_types"],
    chembl_release: Annotated[
        Optional[int],
        typer.Option(
            "-cr",
            "--chembl-release",
            help="Only fetch data reported **up to** a certain ChEMBL release (e.g., 34).",
            metavar="int",
        ),
    ] = DEFAULTS["chembl_release"],
    chembl_version: Annotated[
        Optional[str],
        typer.Option(
            "-v",
            "--chembl-version",
            help="ChEMBL version used by _chembl_downloader_.",
            metavar="str",
        ),
    ] = DEFAULTS["chembl_version"],
    chembl_backend: Annotated[
        ChemblBackend,
        typer.Option(
            "-back",
            "--chembl-backend",
            help="Backend to use for ChEMBL interaction.",
        ),
    ] = DEFAULTS["chembl_backend"],
    compound_equality: Annotated[
        CompoundEquality,
        typer.Option(
            "-cpd-eq",
            "--compound-equality",
            help="Method used to identify equivalent compounds during aggregation.",
        ),
    ] = DEFAULTS["compound_equality"],
    value_column: Annotated[
        ValueColumn,
        typer.Option(
            "-vcol",
            "--value-column",
            help="Column holding the experimental measurement to summarize (mean/median/std). "
            "Use 'standard_value' for non-pChEMBL data (e.g., % inhibition).",
        ),
    ] = DEFAULTS["value_column"],
    # --- Metadata & Aggregation ---
    metadata_columns: Annotated[
        str,
        typer.Option(
            "-mcols",
            "--metadata-columns",
            parser=csv_string,
            help="Extra metadata columns to keep, comma-separated.",
            show_default=False,
            metavar="col1,col2,...",
        ),
    ] = DEFAULTS["metadata_columns"],
    id_columns: Annotated[
        str,
        typer.Option(
            "-idcols",
            "--id-columns",
            parser=csv_string,
            help="Additional columns to append to the aggregation key (compound + task), comma-separated. "
            "E.g.: 'assay_chembl_id' keeps measurements from different assays separate.",
            show_default=False,
            metavar="col1,col2,...",
        ),
    ] = DEFAULTS["id_columns"],
    # --- Numeric Filter Arguments ---
    max_assay_size: Annotated[
        Optional[int],
        typer.Option(
            "-maxas",
            "--max-assay-size",
            help="Max number of compounds in an assay.",
            metavar="int",
        ),
    ] = DEFAULTS["max_assay_size"],
    min_assay_size: Annotated[
        Optional[int],
        typer.Option(
            "-minas",
            "--min-assay-size",
            help="Min number of compounds in an assay.",
            metavar="int",
        ),
    ] = DEFAULTS["min_assay_size"],
    min_assay_overlap: Annotated[
        int,
        typer.Option(
            "-maso",
            "--min-assay-overlap",
            help="Min overlapping compounds between assays.",
            metavar="int",
        ),
    ] = DEFAULTS["min_assay_overlap"],
    # --- Boolean Flags ---
    calculate_pchembl: Annotated[
        bool,
        typer.Option(
            "-calc/-no-calc",
            "--calculate-pchembl/--no-calculate-pchembl",
            help="Calculate pChEMBL values if not reported.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["calculate_pchembl"],
    chirality: Annotated[
        bool,
        typer.Option(
            "-chiral/-no-chiral",
            "--chirality/--no-chirality",
            help="Consider chirality during fingerprint calculation.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["chirality"],
    drop_unassigned_chiral: Annotated[
        bool,
        typer.Option(
            "-duchi/-dont-duchi",
            "--drop-unassigned-chiral/--dont-drop-unassigned-chiral",
            help="Drop entries with unassigned chiral centers.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["drop_unassigned_chiral"],
    curate_annotation_errors: Annotated[
        bool,
        typer.Option(
            "-cure/-dont-cure",
            "--curate-annotation-errors/--dont-curate-annotation-errors",
            help="Apply curation for pChEMBL annotation errors.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["curate_annotation_errors"],
    skip_not_aggregated: Annotated[
        bool,
        typer.Option(
            "-skip-agg/-dont-skip-agg",
            "--skip-not-aggregated/--dont-skip-not-aggregated",
            help="Skip saving pre-aggregation data.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["skip_not_aggregated"],
    aggregate_mutants: Annotated[
        bool,
        typer.Option(
            "-mutagg/-dont-mutagg",
            "--aggregate-mutants/--dont-aggregate-mutants",
            help="Aggregate data on targets regardless of mutation.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["aggregate_mutants"],
    skip_recipe: Annotated[
        bool,
        typer.Option(
            "-rec/-dont-rec",
            "--skip-recipe/--dont-skip-recipe",
            help="Skip saving the JSON recipe file.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["skip_recipe"],
    require_doc_date: Annotated[
        bool,
        typer.Option(
            "-reqdoc/-dont-reqdoc",
            "--require-doc-date/--dont-require-doc-date",
            help="Filter out bioactivities without a document date.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["require_doc_date"],
    strict_mutant_removal: Annotated[
        bool,
        typer.Option(
            "-smr/-no-smr",
            "--strict-mutant-removal/--no-strict-mutant-removal",
            help="Flag assays with mutant-related keywords in assay_description for removal.",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS["strict_mutant_removal"],
    convert_units: Annotated[
        bool,
        typer.Option(
            "-conu/-no-conu",
            "--convert-units/--no-convert-units",
            help="Convert units to standard formats: permeability (10^-6 cm/s), "
            "molar concentration (nM), mass concentration (ug/mL), dose (mg/kg), time (hr).",
            is_flag=True,
            metavar="bool",
        ),
    ] = DEFAULTS.get("convert_units", False),
):
    """
    Filter, download, and process bioactivity data from ChEMBL.
    """
    import numpy as np
    from chembl_downloader import latest

    from .chembl_data_pipeline import (
        _log_pipeline_summary,
        aggregate_data,
        get_standardize_and_clean_workflow,
    )

    if not output_path.parent.exists():
        output_path.mkdir()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")

    configs = {k: v for k, v in ctx.params.items() if v is not None}
    logger.debug(f"Command line arguments:\n{configs}")

    valid_suffixes = [".csv", ".tsv", ".parquet"]
    if not np.intersect1d(valid_suffixes, output_path.suffixes).shape[0] > 0:
        logger.error(
            "Output file must have a valid suffix: .csv, .tsv, or .parquet. "
            f"Provided suffix: {output_path.suffixes}"
        )
        raise typer.Exit(code=1)

    if chirality and not drop_unassigned_chiral:
        logger.warning(
            "Consider passing the `--drop-unassigned-chiral` flag when using the `--chirality` flag. "
            "For more information on why this could be problematic, see the link:\n"
            "https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00934-w#:~:text="
            "Some%20duplicates%20were,for%20kinetic%20solubility."
        )

    pre_agg_df = get_standardize_and_clean_workflow(
        molecule_ids=molecule_ids or [],
        target_ids=target_ids or [],
        assay_ids=assay_ids or [],
        document_ids=document_ids or [],
        chirality=chirality,
        calculate_pchembl=calculate_pchembl,
        output_path=output_path,
        confidence_scores=confidence_scores,
        bioactivity_type=bioactivity_type,
        standard_relation=standard_relation,
        standard_units=standard_units,
        assay_types=assay_types,
        chembl_release=chembl_release,
        save_not_aggregated=(not skip_not_aggregated),
        drop_unassigned_chiral=drop_unassigned_chiral,
        curate_annotation_errors=curate_annotation_errors,
        version=chembl_version,
        backend=chembl_backend.value,
        require_doc_date=require_doc_date,
        min_assay_size=min_assay_size,
        max_assay_size=max_assay_size,
        min_assay_overlap=min_assay_overlap,
        strict_mutant_removal=strict_mutant_removal,
        value_col=value_column.value,
        enable_unit_conversion=convert_units,
    )
    pre_agg_count = len(pre_agg_df)

    df = aggregate_data(
        df=pre_agg_df,
        chirality=chirality,
        extra_multival_cols=metadata_columns,
        extra_id_cols=id_columns,
        aggregate_mutants=aggregate_mutants,
        output_path=output_path,
        compound_equality=compound_equality.value,
        value_col=value_column.value,
    )

    _log_pipeline_summary(pre_agg_df, aggregated_df=df, pre_aggregation_count=pre_agg_count)

    if not skip_recipe:
        output_name = output_path.stem.split(".")[0]
        recipe_path = output_path.parent / f"{output_name}_recipe.json"

        command_vals = []
        configs = {k: v for k, v in ctx.params.items() if v is not None}
        for k, v in configs.items():
            save_k = k.replace("_", "-")  # same format as the command line
            if k in ["chembl_release", "chembl_version"]:
                if v is not None:
                    command_vals.append((f"--{save_k} {v}"))
                else:  # safe to assume latest version; if None, chembl_downloader gets latest
                    command_vals.append(f"--{save_k} {latest()}")
            elif isinstance(v, Path) or isinstance(v, Enum):
                configs[k] = str(v)  # Convert Path and Enum to string for JSON serialization
                if DEFAULTS[k] is not v:
                    command_vals.append(f"--{save_k} {str(v)}")
            elif isinstance(v, bool):
                if k in DEFAULT_FALSE_ARGS and v is not False:
                    command_vals.append((f"--{save_k}"))
                elif k in DEFAULT_TRUE_ARGS and v is not True:
                    command_vals.append(f"--dont-{save_k}")
            elif isinstance(v, list):
                if DEFAULTS[k] != v:
                    command_vals.append(f"--{save_k} {','.join([str(i) for i in v])}")
            elif isinstance(v, str):
                if DEFAULTS[k] != v:
                    command_vals.append(f"--{save_k} {v}")
            elif isinstance(v, int):
                if DEFAULTS[k] != v:
                    command_vals.append(f"--{save_k} {v}")
        command = "capricho get " + " ".join(command_vals)
        configs = {k: configs[k] for k in DEFAULTS if k in configs}
        configs = {"command": command, "capricho version": __version__, **configs}

        with open(recipe_path, "w") as f:
            json.dump(configs, f, indent=2)

        logger.info(f"Recipe saved to {recipe_path}")

    logger.info(f"Successfully processed and saved data to {output_path}")
    return df


@app.command(name="binarize", no_args_is_help=True)
def binarize_data(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input-path",
            help="Path to aggregated data file (CSV, TSV, or Parquet).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-path",
            help="Path to save the binarized output file.",
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option(
            "-t",
            "--threshold",
            help="Activity threshold for binarization (pchembl scale). Default 6.0 = 1 µM.",
            metavar="float",
        ),
    ] = 6.0,
    value_column: Annotated[
        str,
        typer.Option(
            "-vcol",
            "--value-column",
            help="Column to use for binarization (e.g., pchembl_value_mean, pchembl_value_median).",
            metavar="str",
        ),
    ] = "pchembl_value_mean",
    compound_id_col: Annotated[
        CompoundIdColumn,
        typer.Option(
            "-cid",
            "--compound-id-col",
            help="Column used as the compound identifier.",
        ),
    ] = CompoundIdColumn.connectivity,
    target_id_col: Annotated[
        str,
        typer.Option(
            "-tid",
            "--target-id-col",
            help="Column name for target identifiers.",
            metavar="str",
        ),
    ] = "target_chembl_id",
    relation_col: Annotated[
        str,
        typer.Option(
            "-rel",
            "--relation-col",
            help="Column name for standard_relation values.",
            metavar="str",
        ),
    ] = "standard_relation",
    output_binary_col: Annotated[
        str,
        typer.Option(
            "-bcol",
            "--binary-col",
            help="Name for the output binary column.",
            metavar="str",
        ),
    ] = "activity_binary",
    compare_across_mutants: Annotated[
        bool,
        typer.Option(
            "-cmp-mut/-dont-cmp-mut",
            "--compare-across-mutants/--dont-compare-across-mutants",
            help="If True, measurements on different mutants are compared for conflicts. Default: False (different mutants are separate compound-target pairs).",
            is_flag=True,
            metavar="bool",
        ),
    ] = False,
    conflict_report_path: Annotated[
        Path | None,
        typer.Option(
            "-rp",
            "--conflict-report-path",
            help="Optional path to save detailed conflict report as JSON for interactive analysis.",
            metavar="path",
        ),
    ] = None,
    conflict_resolution: Annotated[
        ConflictResolution | None,
        typer.Option(
            "-cr",
            "--conflict-resolution",
            help="Strategy for resolving binarization conflicts: "
            "'drop' removes all conflicting rows, "
            "'relation' keeps exact (=) and drops censored, "
            "'confidence' keeps highest confidence_score, "
            "'majority' keeps majority binary label vote.",
        ),
    ] = None,
):
    """
    Binarize aggregated bioactivity data based on activity threshold.

    This command converts continuous pchembl values to binary labels (0=inactive, 1=active)
    while properly handling censored measurements (< and >) and validating agreement between
    discrete (=) and censored measurements for the same compound-target pair.

    The output file also contains a new relation column with signs adjusted for -log scale.
    E.g.: for threshold of 6.0, ``IC50`` with ``pchembl_value`` 6.0 and ``pchembl_relation`` >

    - -log[IC50 concentration] > 6.0;
    - IC50 concentration < 1 µM;
    - active (1).

    Example:
        capricho binarize -i aggregated_data.csv -o binarized_data.csv -t 6.5
    """
    import pandas as pd

    from ..core.binarization import binarize_aggregated_data
    from ..core.pandas_helper import save_dataframe

    logger.info(f"Loading aggregated data from {input_path}")

    if input_path.suffix in [".csv", ".csv.gz"]:
        df = pd.read_csv(input_path)
    elif input_path.suffix in [".tsv", ".tsv.gz"]:
        df = pd.read_csv(input_path, sep="\t")
    elif input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}. Use .csv, .tsv, .gz, or .parquet")
        raise typer.Exit(code=1)

    logger.info(f"Loaded {len(df)} rows from {input_path}")
    logger.info(f"Binarizing data with threshold={threshold} using column '{value_column}'")

    binarized_df = binarize_aggregated_data(
        df=df,
        threshold=threshold,
        value_column=value_column,
        compound_id_col=compound_id_col.value,
        target_id_col=target_id_col,
        relation_col=relation_col,
        output_binary_col=output_binary_col,
        compare_across_mutants=compare_across_mutants,
        conflict_report_path=conflict_report_path,
        conflict_resolution=conflict_resolution.value if conflict_resolution else None,
    )

    # Ensure out/dir exists, add proper suffix for saving and Save the binarized data.
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(input_path.suffix)
    save_dataframe(binarized_df, output_path)

    # Log number of actives/inactives
    n_actives = binarized_df[output_binary_col].sum()
    n_inactives = len(binarized_df) - n_actives
    logger.info(
        f"{n_actives}/{len(binarized_df)} actives ({n_actives / len(binarized_df) * 100:.2f}%); "
        f"{n_inactives}/{len(binarized_df)} inactives ({n_inactives / len(binarized_df) * 100:.2f}%)."
    )

    logger.info(f"Binarization complete. Output saved to {output_path}")
    return binarized_df


@app.command(name="prepare", no_args_is_help=True)
def prepare_data(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input-path",
            help="Path to aggregated data file (CSV, TSV, or Parquet).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-path",
            help="Path to save the activity matrix.",
        ),
    ],
    task_col: Annotated[
        str,
        typer.Option(
            "--task-col",
            help="Column to use as task identifier (e.g., target_chembl_id).",
            metavar="str",
        ),
    ] = "target_chembl_id",
    value_column: Annotated[
        ValueColumn,
        typer.Option(
            "-vcol",
            "--value-column",
            help="Column holding the experimental measurement, as passed to 'capricho get --value-column'. "
            "Statistics are read from '{value_column}_mean'.",
        ),
    ] = ValueColumn.pchembl_value,
    compound_col: Annotated[
        CompoundIdColumn,
        typer.Option(
            "--compound-col",
            help="Column used as the compound identifier.",
        ),
    ] = CompoundIdColumn.connectivity,
    smiles_col: Annotated[
        str,
        typer.Option(
            "--smiles-col",
            help="Column containing SMILES strings.",
            metavar="str",
        ),
    ] = "smiles",
    remove_flags: Annotated[
        Optional[str],
        typer.Option(
            "--remove-flags",
            parser=csv_string,
            help="Quality flags to remove, comma-separated. Rows with these flags in data_dropping_comment will be filtered out.",
            metavar="flag1,flag2,...",
        ),
    ] = None,
    id_columns: Annotated[
        Optional[str],
        typer.Option(
            "--id-columns",
            parser=csv_string,
            help="Additional columns to combine with task_col for composite task identifiers. "
            "Use the same columns passed to 'capricho get --id-columns' during aggregation.",
            metavar="col1,col2,...",
        ),
    ] = None,
    # Tab-completable drop flags
    drop_undefined_stereo: Annotated[
        bool,
        typer.Option(
            "--drop-undefined-stereo/--keep-undefined-stereo",
            help="Drop entries with undefined stereochemistry.",
            is_flag=True,
        ),
    ] = False,
    drop_potential_duplicate: Annotated[
        bool,
        typer.Option(
            "--drop-potential-duplicate/--keep-potential-duplicate",
            help="Drop entries flagged as potential duplicates.",
            is_flag=True,
        ),
    ] = False,
    drop_data_validity: Annotated[
        bool,
        typer.Option(
            "--drop-data-validity/--keep-data-validity",
            help="Drop entries with data validity comments.",
            is_flag=True,
        ),
    ] = False,
    drop_unit_error: Annotated[
        bool,
        typer.Option(
            "--drop-unit-error/--keep-unit-error",
            help="Drop entries with unit annotation errors.",
            is_flag=True,
        ),
    ] = False,
    drop_mixture: Annotated[
        bool,
        typer.Option(
            "--drop-mixture/--keep-mixture",
            help="Drop entries containing mixtures in SMILES.",
            is_flag=True,
        ),
    ] = False,
    drop_activity_comment: Annotated[
        bool,
        typer.Option(
            "--drop-activity-comment/--keep-activity-comment",
            help="Drop entries whose activity_comment reports inactivity while the source "
            "standard_relation is '='.",
            is_flag=True,
        ),
    ] = False,
    drop_assay_size: Annotated[
        bool,
        typer.Option(
            "--drop-assay-size/--keep-assay-size",
            help="Drop entries outside assay size bounds (both too small and too large).",
            is_flag=True,
        ),
    ] = False,
    drop_insufficient_overlap: Annotated[
        bool,
        typer.Option(
            "--drop-insufficient-overlap/--keep-insufficient-overlap",
            help="Drop entries from assays with insufficient overlap.",
            is_flag=True,
        ),
    ] = False,
    # Deduplication and recalculation
    deduplicate: Annotated[
        bool,
        typer.Option(
            "--deduplicate/--no-deduplicate",
            help="Remove duplicate pChEMBL values within aggregated rows and recalculate statistics.",
            is_flag=True,
        ),
    ] = False,
    # Annotation error resolution
    resolve_annotation_error: Annotated[
        Optional[str],
        typer.Option(
            "--resolve-annotation-error",
            help="Resolve unit annotation errors (3.0 or 6.0 log unit differences) by keeping "
            "measurement from earliest document. Use 'first' to enable.",
            metavar="first",
        ),
    ] = None,
    # Plot output
    plot_path: Annotated[
        Optional[Path],
        typer.Option(
            "--plot",
            help="Path to save comparability plot (e.g., comparability.png). If not provided, no plot is generated.",
        ),
    ] = None,
):
    """Transform aggregated bioactivity data into multitask format (activity matrix).

    This command pivots aggregated data to create an activity matrix where rows are
    compounds and columns are tasks (e.g., targets). This format is suitable for
    multitask machine learning models.

    The command supports tab-completable flags for common data quality filters,
    as well as a --deduplicate option to remove duplicate pChEMBL values and
    recalculate statistics.

    Example:
        capricho prepare -i aggregated_data.csv -o activity_matrix.csv
        capricho prepare -i data.csv -o matrix.csv --drop-undefined-stereo --drop-unit-error
        capricho prepare -i data.csv -o matrix.csv --deduplicate --plot comparability.png
    """
    import pandas as pd

    from ..analysis import (
        DroppingComment,
        explode_assay_comparability,
        get_all_comments,
        plot_multi_panel_comparability,
    )
    from ..core.pandas_helper import save_dataframe
    from .prepare import clean_data, prepare_multitask_data

    logger.info(f"Loading aggregated data from {input_path}")

    if input_path.suffix in [".csv", ".csv.gz"]:
        df = pd.read_csv(input_path)
    elif input_path.suffix in [".tsv", ".tsv.gz"]:
        df = pd.read_csv(input_path, sep="\t")
    elif input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}. Use .csv, .tsv, .gz, or .parquet")
        raise typer.Exit(code=1)

    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Build list of flags to remove from tab-completable options
    flags_to_remove = list(remove_flags) if remove_flags else []

    if drop_undefined_stereo:
        flags_to_remove.append(DroppingComment.UNDEFINED_STEREOCHEMISTRY.value)
    if drop_potential_duplicate:
        flags_to_remove.append(DroppingComment.POTENTIAL_DUPLICATE.value)
    if drop_data_validity:
        flags_to_remove.append(DroppingComment.DATA_VALIDITY_COMMENT.value)
    if drop_unit_error:
        flags_to_remove.append(DroppingComment.UNIT_ANNOTATION_ERROR.value)
    if drop_mixture:
        flags_to_remove.append(DroppingComment.MIXTURE_IN_SMILES.value)
    if drop_activity_comment:
        flags_to_remove.append(DroppingComment.ACTIVITY_COMMENT_REVIEW.value)
    if drop_assay_size:
        flags_to_remove.append(DroppingComment.ASSAY_SIZE_TOO_SMALL.value)
        flags_to_remove.append(DroppingComment.ASSAY_SIZE_TOO_LARGE.value)
    if drop_insufficient_overlap:
        flags_to_remove.append(DroppingComment.INSUFFICIENT_ASSAY_OVERLAP.value)
        flags_to_remove.append(DroppingComment.INSUFFICIENT_ASSAY_OVERLAP_WITH_METADATA.value)

    value_col = value_column.value

    # Clean data: deduplicate, resolve annotation errors, filter flags
    df = clean_data(
        df,
        deduplicate=deduplicate,
        value_col=value_col,
        resolve_annotation_error=resolve_annotation_error,
        drop_flags=flags_to_remove if flags_to_remove else None,
        compound_col=compound_col.value,
    )

    # Use mean column for the activity matrix
    value_col_mean = f"{value_column.value}_mean"

    activity_matrix = prepare_multitask_data(
        df=df,
        task_col=task_col,
        value_col=value_col_mean,
        compound_col=compound_col.value,
        smiles_col=smiles_col,
        id_columns=id_columns,
    )

    # Ensure output directory exists and add proper suffix
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(input_path.suffix)

    # Save the activity matrix
    save_dataframe(activity_matrix, output_path)
    logger.info(f"Activity matrix saved to {output_path}")
    logger.info(
        f"Matrix dimensions: {len(activity_matrix)} compounds x "
        f"{len(activity_matrix.columns)-1} tasks (plus smiles column)"
    )

    # Save prepared data (before pivoting)
    prepared_path = output_path.with_name(
        output_path.stem.replace("_matrix", "") + "_prepared" + output_path.suffix
    )
    save_dataframe(df, prepared_path)
    logger.info(f"Prepared data saved to {prepared_path}")

    # Generate comparability plots if requested
    if plot_path is not None:
        logger.info("Generating comparability plots...")
        # Get rows with aggregated values for comparability analysis
        subset = df[df[value_col].astype(str).str.contains("|", regex=False, na=False)].copy()
        if len(subset) > 0:
            subset = subset.assign(repeat=range(len(subset)))
            all_comments = get_all_comments()
            exploded_subset = explode_assay_comparability(subset, value_column=value_col)

            if len(exploded_subset) > 0:
                import matplotlib.pyplot as plt

                from ..analysis import plot_subset
                from ..core.pandas_helper import filter_dropping_flags

                if flags_to_remove:
                    cleaned_subset = filter_dropping_flags(
                        exploded_subset, flags_to_remove, column="dropping_comment"
                    )
                else:
                    cleaned_subset = exploded_subset

                # Plot 1: Cleaned data comparability (single panel)
                if len(cleaned_subset) > 0:
                    fig_clean, ax_clean = plot_subset(
                        cleaned_subset,
                        title="Cleaned Data Comparability",
                        value_column=value_col,
                    )
                    clean_plot_path = plot_path.with_name(plot_path.stem + "_cleaned" + plot_path.suffix)
                    fig_clean.savefig(clean_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig_clean)
                    logger.info(f"Cleaned comparability plot saved to {clean_plot_path}")
                else:
                    logger.warning("No data remaining after filtering for cleaned plot.")

                # Plot 2: Multi-panel showing remaining flags
                fig_multi, axs = plot_multi_panel_comparability(
                    exploded_subset,
                    all_comments,
                    title="Remaining Flags in Prepared Data",
                    figsize=(20, 8),
                    ncols=5,
                    value_column=value_col,
                )
                multi_plot_path = plot_path.with_name(plot_path.stem + "_flags" + plot_path.suffix)
                fig_multi.savefig(multi_plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig_multi)
                logger.info(f"Flags comparability plot saved to {multi_plot_path}")
            else:
                logger.warning("No pairwise comparisons available for plotting.")
        else:
            logger.warning("No aggregated data found for comparability plot.")

    return activity_matrix


if __name__ == "__main__":
    app()
