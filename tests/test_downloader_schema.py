"""Tests for ChEMBL schema differences across releases.

docs.chembl_release_id only exists from ChEMBL 33 onwards. Older releases must still be
queryable, notably ChEMBL 32, which Landrum and Riniker used and which case study 1
reproduces. These tests build real SQLite databases with each schema and run the real
queries against them.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

# ChEMBL tables and columns touched by get_full_activity_data_sql, trimmed to what the
# query selects and joins on.
SCHEMA = {
    "molecule_dictionary": [
        "molregno INTEGER",
        "chembl_id TEXT",
        "first_in_class INTEGER",
        "chirality INTEGER",
        "oral INTEGER",
        "prodrug INTEGER",
        "max_phase INTEGER",
        "therapeutic_flag INTEGER",
        "withdrawn_flag INTEGER",
    ],
    "compound_structures": ["molregno INTEGER", "canonical_smiles TEXT", "standard_inchi_key TEXT"],
    "molecule_hierarchy": ["molregno INTEGER", "parent_molregno INTEGER"],
    "activities": [
        "activity_id INTEGER",
        "molregno INTEGER",
        "assay_id INTEGER",
        "doc_id INTEGER",
        "standard_flag INTEGER",
        "standard_relation TEXT",
        "standard_type TEXT",
        "standard_units TEXT",
        "standard_value REAL",
        "pchembl_value REAL",
        "data_validity_comment TEXT",
        "activity_comment TEXT",
        "potential_duplicate INTEGER",
    ],
    "assays": [
        "assay_id INTEGER",
        "chembl_id TEXT",
        "description TEXT",
        "relationship_type TEXT",
        "assay_type TEXT",
        "assay_organism TEXT",
        "assay_category TEXT",
        "assay_tax_id INTEGER",
        "assay_strain TEXT",
        "assay_tissue TEXT",
        "assay_cell_type TEXT",
        "assay_subcellular_fraction TEXT",
        "bao_format TEXT",
        "confidence_score INTEGER",
        "variant_id INTEGER",
        "tid INTEGER",
    ],
    "variant_sequences": ["variant_id INTEGER", "mutation TEXT"],
    "target_dictionary": ["tid INTEGER", "chembl_id TEXT", "organism TEXT"],
}

DOCS_BASE = [
    "doc_id INTEGER",
    "chembl_id TEXT",
    "doc_type TEXT",
    "authors TEXT",
    "doi TEXT",
    "journal TEXT",
    "volume TEXT",
    "year INTEGER",
    "title TEXT",
]


def build_chembl_db(path: Path, with_release_column: bool) -> None:
    """Create a minimal ChEMBL-shaped SQLite database holding one activity."""
    con = sqlite3.connect(path)
    for table, columns in SCHEMA.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(columns)})")

    docs_columns = DOCS_BASE + (["chembl_release_id INTEGER"] if with_release_column else [])
    con.execute(f"CREATE TABLE docs ({', '.join(docs_columns)})")

    con.execute(
        "INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL25', 0, 0, 0, 0, 4, 0, 0)"
    )
    con.execute(
        "INSERT INTO compound_structures VALUES "
        "(1, 'CC(=O)Oc1ccccc1C(=O)O', 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N')"
    )
    con.execute("INSERT INTO molecule_hierarchy VALUES (1, 1)")
    con.execute(
        "INSERT INTO activities VALUES "
        "(10, 1, 100, 1000, 1, '=', 'Ki', 'nM', 5.0, 8.3, NULL, 'Not Active', 0)"
    )
    con.execute(
        "INSERT INTO assays VALUES (100, 'CHEMBL1', 'test assay', 'D', 'B', 'Homo sapiens', "
        "NULL, 9606, NULL, NULL, NULL, NULL, 'BAO_0000357', 9, NULL, 500)"
    )
    con.execute("INSERT INTO target_dictionary VALUES (500, 'CHEMBL205', 'Homo sapiens')")

    docs_values = "(1000, 'CHEMBL_DOC1', 'PUBLICATION', 'Smith J', '10.1/x', 'J Med Chem', '50', 2010, 'A title'"
    docs_values += ", 30)" if with_release_column else ")"
    con.execute(f"INSERT INTO docs VALUES {docs_values}")

    con.commit()
    con.close()


class ChemblSchemaTestCase(unittest.TestCase):
    """Base case pointing pystow at a temporary ChEMBL database."""

    with_release_column = True
    version = "36"

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("PYSTOW_HOME")
        os.environ["PYSTOW_HOME"] = self.tmpdir.name

        db_dir = Path(self.tmpdir.name) / "chembl" / self.version
        db_dir.mkdir(parents=True)
        build_chembl_db(db_dir / f"chembl_{self.version}.db", self.with_release_column)

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("PYSTOW_HOME", None)
        else:
            os.environ["PYSTOW_HOME"] = self._old_home
        self.tmpdir.cleanup()


class TestChembl32Schema(ChemblSchemaTestCase):
    """ChEMBL 32 has no docs.chembl_release_id column."""

    with_release_column = False
    version = "32"

    def test_query_succeeds_without_release_column(self):
        """Test that activity data can be fetched from a release predating chembl_release_id."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version=self.version)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["molecule_chembl_id"], "CHEMBL25")

    def test_release_column_present_but_null(self):
        """Test that chembl_release is still emitted, since downstream requires the column."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version=self.version)

        self.assertIn("chembl_release", df.columns)
        self.assertTrue(df["chembl_release"].isna().all())

    def test_release_filter_raises_clear_error(self):
        """Test that filtering by release fails loudly when the release is not recorded.

        Silently ignoring the filter would return data from every release while the user
        believes it was restricted.
        """
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        with self.assertRaises(ValueError) as ctx:
            get_full_activity_data_sql(
                target_chembl_ids=["CHEMBL205"], chembl_release=30, version=self.version
            )
        self.assertIn("chembl_release", str(ctx.exception).lower())


class TestChembl33Boundary(ChemblSchemaTestCase):
    """ChEMBL 33 is the first release with docs.chembl_release_id (release notes).

    This pins the version cutoff: 33 must be treated as having the column and 32 (below)
    must not, so a future off-by-one edit to the threshold is caught here.
    """

    version = "33"

    def test_first_supported_release_reads_the_column(self):
        """Test that the boundary release uses the real column rather than NULL."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version=self.version)

        self.assertEqual(df.iloc[0]["chembl_release"], 30)


class TestChembl36Schema(ChemblSchemaTestCase):
    """ChEMBL 36 records docs.chembl_release_id (the base-class defaults)."""

    def test_release_value_is_returned(self):
        """Test that the release is read through when the column exists."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version=self.version)

        self.assertEqual(df.iloc[0]["chembl_release"], 30)

    def test_release_filter_applies(self):
        """Test that the release filter still selects rows at or below the threshold."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql

        kept = get_full_activity_data_sql(
            target_chembl_ids=["CHEMBL205"], chembl_release=30, version=self.version
        )
        self.assertEqual(len(kept), 1)

        dropped = get_full_activity_data_sql(
            target_chembl_ids=["CHEMBL205"], chembl_release=29, version=self.version
        )
        self.assertEqual(len(dropped), 0)


class TestActivityCommentRetrieval(ChemblSchemaTestCase):
    def test_sql_retrieval_exposes_the_comment_to_flagging(self):
        from Capricho.analysis import DroppingComment
        from Capricho.chembl.api.downloader import get_full_activity_data_sql
        from Capricho.chembl.data_flag_functions import flag_censored_activity_comment
        from Capricho.core.default_fields import DATA_DROPPING_COMMENT

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version=self.version)
        result = flag_censored_activity_comment(df)

        self.assertEqual(result.iloc[0]["activity_comment"], "Not Active")
        self.assertEqual(result.iloc[0]["standard_relation"], "=")
        self.assertIn(
            DroppingComment.ACTIVITY_COMMENT_REVIEW.value,
            str(result.iloc[0][DATA_DROPPING_COMMENT]),
        )


class TestWebresourceActivityCommentRequested(unittest.TestCase):
    def test_only_selection_includes_activity_comment(self):
        from unittest import mock

        # The upstream module fetches the API schema on import, before our
        # query mock can take effect. Replace it before importing webresource.
        import sys
        from types import ModuleType

        client = mock.Mock()
        client_module = ModuleType("chembl_webresource_client.new_client")
        client_module.new_client = client
        with mock.patch.dict(sys.modules, {client_module.__name__: client_module}):
            import Capricho.chembl.api.webresource as webresource

        query = mock.Mock()
        query.only.return_value = [
            {
                "activity_id": 10,
                "molecule_chembl_id": "CHEMBL25",
                "standard_relation": "=",
                "activity_comment": "Not Active",
            }
        ]

        with mock.patch.object(webresource, "new_client", client):
            client.activity.filter.return_value = query
            df, _ = webresource.get_activity_table(target_chembl_ids=["CHEMBL205"])

        self.assertIn("activity_comment", query.only.call_args.args)
        self.assertEqual(df.iloc[0]["activity_comment"], "Not Active")


if __name__ == "__main__":
    unittest.main()
