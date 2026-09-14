"""Explore the ChEMBL database schema without loading the entire database."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ...core.pandas_helper import save_dataframe
from ...core.table_format import format_dataframe
from .downloader import check_and_download_chembl_db, connect_chembl


def get_tables(conn):
    """Get a list of all tables in the database."""
    query = """
    SELECT name FROM sqlite_master
    WHERE type='table'
    ORDER BY name;
    """
    return pd.read_sql(query, conn)


def get_table_info(conn, table_name):
    """Get detailed information about a specific table."""
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql(query, conn)


def get_table_counts(conn, table_name):
    """Get the row count for a specific table."""
    query = f"SELECT COUNT(*) as count FROM {table_name};"
    return pd.read_sql(query, conn)["count"][0]


def get_sample_data(conn, table_name, limit=5):
    """Get sample data from a table."""
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    return pd.read_sql(query, conn)


def get_foreign_keys(conn, table_name):
    """Get foreign key information for a table."""
    query = f"PRAGMA foreign_key_list({table_name});"
    return pd.read_sql(query, conn)


def find_related_tables(conn, table_name):
    """Find tables that reference this table or are referenced by this table."""
    # Get all tables
    tables = get_tables(conn)["name"].tolist()

    # Find relationships
    relationships = []

    # Look for foreign keys in the current table
    fks_in_table = get_foreign_keys(conn, table_name)
    for _, row in fks_in_table.iterrows():
        relationships.append(
            {
                "type": "outgoing",
                "table": table_name,
                "column": row["from"],
                "references_table": row["table"],
                "references_column": row["to"],
            }
        )

    # Look for tables that reference this table
    for other_table in tables:
        if other_table == table_name:
            continue

        fks = get_foreign_keys(conn, other_table)
        for _, row in fks.iterrows():
            if row["table"] == table_name:
                relationships.append(
                    {
                        "type": "incoming",
                        "table": other_table,
                        "column": row["from"],
                        "references_table": table_name,
                        "references_column": row["to"],
                    }
                )

    return relationships


def search_tables_for_column(conn, column_pattern):
    """Search for tables that have columns matching the given pattern."""
    tables = get_tables(conn)["name"].tolist()
    results = []

    for table in tables:
        columns = get_table_info(conn, table)
        matches = columns[columns["name"].str.contains(column_pattern, case=False)]

        if not matches.empty:
            for _, match in matches.iterrows():
                results.append({"table": table, "column": match["name"], "type": match["type"]})

    return results


def list_all_tables(conn):
    """List all tables with their row counts."""
    tables = get_tables(conn)

    results = []
    for table_name in tables["name"]:
        try:
            count = get_table_counts(conn, table_name)
            results.append({"table": table_name, "rows": count})
        except Exception as e:
            results.append({"table": table_name, "rows": f"Error: {str(e)}"})

    return pd.DataFrame(results).sort_values(by="rows", ascending=False)


def explore_table(conn, table_name):
    """Explore a specific table and return structured data.

    Returns:
        dict with keys: table_name, count, columns (DataFrame),
        relationships (DataFrame or None), sample (DataFrame).
    """
    columns = get_table_info(conn, table_name)
    count = get_table_counts(conn, table_name)
    sample = get_sample_data(conn, table_name)
    relationships_list = find_related_tables(conn, table_name)
    rel_df = pd.DataFrame(relationships_list) if relationships_list else None

    return {
        "table_name": table_name,
        "count": count,
        "columns": columns,
        "relationships": rel_df,
        "sample": sample,
    }


def explorer_main(
    version: str | int | None = None,
    list_tables: bool = False,
    table: str = None,
    search_column: str = None,
    query: str = None,
    fmt: str = "markdown",
    output_path: Optional[Path] = None,
    colorize: bool = False,
):
    """Main function for the ChEMBL schema explorer. This function is called by the CLI script.
    For a visual overview of the lastest ChEMBL database schema, consider checking the official
    ChEMBL schema diagram: https://www.ebi.ac.uk/chembl/db_schema

    Args:
        version: ChEMBL version to use. Defaults to None (latest).
        list_tables: list all tables in the database and exit. Defaults to False.
        table: explore a specific table. Defaults to None.
        search_column: search for columns containing a specific pattern. Defaults to None.
        query: execute a custom SQL query. Defaults to None.
        fmt: console output format (markdown, csv, plain, grid). Defaults to "markdown".
        output_path: optional file path to save the primary DataFrame. Defaults to None.
        colorize: apply ANSI color cycling to console rows. Defaults to False.
    """
    primary_df = None
    configs = check_and_download_chembl_db(version=version)
    with connect_chembl(configs) as conn:
        if list_tables:
            tables_df = list_all_tables(conn)
            primary_df = tables_df
            print("\nTABLES IN CHEMBL DATABASE:")
            print(format_dataframe(tables_df, fmt, colorize))

        elif table:
            try:
                info = explore_table(conn, table)
                primary_df = info["columns"]

                print(f"\n{'='*80}")
                print(f"TABLE: {info['table_name']} ({info['count']:,} rows)")
                print(f"{'='*80}")

                print("\nCOLUMNS:")
                print(format_dataframe(info["columns"], fmt, colorize))

                if info["relationships"] is not None:
                    print("\nRELATIONSHIPS:")
                    print(format_dataframe(info["relationships"], fmt, colorize))

                print("\nSAMPLE DATA:")
                print(format_dataframe(info["sample"], fmt, colorize))
            except Exception as e:
                print(f"Error exploring table {table}: {str(e)}")

        elif search_column:
            results = search_tables_for_column(conn, search_column)
            if results:
                results_df = pd.DataFrame(results)
                primary_df = results_df
                print(f"\nTables containing columns matching '{search_column}':")
                print(format_dataframe(results_df, fmt, colorize))
            else:
                print(f"No columns found matching '{search_column}'")

        elif query:
            try:
                result = pd.read_sql(query, conn)
                primary_df = result
                print("\nQuery Result:")
                print(format_dataframe(result, fmt, colorize))
            except Exception as e:
                print(f"Error executing query: {str(e)}")

        else:
            # Default: show database overview
            print("\nChEMBL DATABASE OVERVIEW")
            print("=" * 80)

            tables_df = list_all_tables(conn)
            top_20 = tables_df.head(20)
            primary_df = top_20

            print("\nTop 20 tables by row count:")
            print(format_dataframe(top_20, fmt, colorize))

            print("\nSuggested tables to explore:")
            suggestions = [
                "molecule_dictionary - Main table for chemical compounds",
                "compound_structures - Chemical structures (SMILES, InChI)",
                "activities - Bioactivity measurements",
                "assays - Assay information",
                "target_dictionary - Biological targets",
                "docs - Document/publication information",
            ]

            for suggestion in suggestions:
                print(f"  • {suggestion}")

            print("\nTo explore a specific table:")
            print("  capricho explore --table <table_name>")
            print("\nTo list all tables:")
            print("  capricho explore --list-tables")
            print("\nTo search for columns:")
            print("  capricho explore --search-column <pattern>")

    if output_path is not None and primary_df is not None:
        save_dataframe(primary_df, output_path)
