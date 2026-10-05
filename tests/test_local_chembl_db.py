"""Tests for reading a ChEMBL release that is already on the system.

Users who keep their own ChEMBL dumps should not have to re-download tens of gigabytes
into the pystow tree. `capricho download --set-from-path` registers a database where it
lies, and every query then reads it in place. These tests build real SQLite databases and
run the real queries against them, with pystow pointed at an empty temporary directory so
that any accidental fallback to downloading would fail loudly.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from test_downloader_schema import build_chembl_db


def add_version_table(path: Path, reported_version: str) -> None:
    """Give a database the `version` table real ChEMBL dumps use to state their release.

    Real dumps list the embedded resources first and the ChEMBL release among them, so the
    ChEMBL row is deliberately not the first one here.
    """
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE version (name TEXT, creation_date TEXT, comments TEXT)")
    con.execute("INSERT INTO version VALUES ('Bioassay Ontology 2.0', NULL, 'BAO version')")
    con.execute("INSERT INTO version VALUES ('COCONUT 2025-07', NULL, 'COCONUT version')")
    con.execute(
        "INSERT INTO version VALUES "
        f"('ChEMBL_{reported_version}', '2025-01-01 00:00:00.000000', "
        f"'ChEMBL Release {reported_version}')"
    )
    con.commit()
    con.close()


class LocalDatabaseTestCase(unittest.TestCase):
    """Base case with an empty pystow home and a directory of ChEMBL databases outside it."""

    def setUp(self):
        self.pystow_dir = tempfile.TemporaryDirectory()
        self.store_dir = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("PYSTOW_HOME")
        os.environ["PYSTOW_HOME"] = self.pystow_dir.name

        # resolved, since registering stores an absolute path and macOS symlinks /var
        self.store = Path(self.store_dir.name).resolve()

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("PYSTOW_HOME", None)
        else:
            os.environ["PYSTOW_HOME"] = self._old_home
        self.pystow_dir.cleanup()
        self.store_dir.cleanup()

    def make_db(
        self,
        version: str,
        with_release_column: bool = True,
        reports: str | None = "",
        name: str | None = None,
    ) -> Path:
        """Write a ChEMBL-shaped database for a release into the external store.

        Args:
            version: the release the file name announces.
            with_release_column: whether docs carries chembl_release_id.
            reports: the release the database states for itself in its `version` table.
                Defaults to matching `version`; None writes no `version` table at all.
            name: override the file name, to build a database whose name lies about its release.
        """
        path = self.store / (name or f"chembl_{version}.db")
        build_chembl_db(path, with_release_column)
        if reports is not None:
            add_version_table(path, reports or version)
        return path

    def config_file(self, version: str) -> Path:
        return Path(self.pystow_dir.name) / "chembl" / f"chembl_downloader_config_{version}.json"


class TestRegisterDatabase(LocalDatabaseTestCase):
    """Registering a database that already exists on the system."""

    def test_registers_release_read_from_the_file_name(self):
        """Test that the release is taken from chembl_<version>.db, so --version is optional."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        db_path = self.make_db("36")

        registered = set_chembl_db_path(db_path)

        self.assertEqual(registered, {"36": db_path})
        configs = json.loads(self.config_file("36").read_text())
        self.assertEqual(configs["path"], str(db_path))
        self.assertEqual(configs["version"], "36")

    def test_registers_every_release_in_a_directory(self):
        """Test that a directory of dumps is registered in full, as users keep several releases."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        paths = {version: self.make_db(version) for version in ("34", "35", "36")}

        registered = set_chembl_db_path(self.store)

        self.assertEqual(registered, paths)
        for version, path in paths.items():
            self.assertEqual(json.loads(self.config_file(version).read_text())["path"], str(path))

    def test_version_narrows_a_directory_to_one_release(self):
        """Test that --version picks a single release out of a directory holding several."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        self.make_db("34")
        wanted = self.make_db("35")

        registered = set_chembl_db_path(self.store, version=35)

        self.assertEqual(registered, {"35": wanted})
        self.assertFalse(self.config_file("34").exists())

    def test_version_is_required_for_an_unconventional_file_name(self):
        """Test that a file name without a release is rejected rather than guessed at."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        renamed = self.store / "my_chembl_dump.db"
        build_chembl_db(renamed, True)

        with self.assertRaises(ValueError) as ctx:
            set_chembl_db_path(renamed)
        self.assertIn("--version", str(ctx.exception))

        self.assertEqual(set_chembl_db_path(renamed, version=36), {"36": renamed})

    def test_directory_without_a_dump_is_rejected(self):
        """Test that pointing at a directory holding no ChEMBL dump fails immediately."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        with self.assertRaises(FileNotFoundError):
            set_chembl_db_path(self.store)

    def test_missing_release_in_a_directory_lists_what_is_there(self):
        """Test that asking for an absent release reports the releases that are available."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        self.make_db("34")

        with self.assertRaises(FileNotFoundError) as ctx:
            set_chembl_db_path(self.store, version=35)
        self.assertIn("34", str(ctx.exception))

    def test_non_chembl_database_is_rejected(self):
        """Test that a SQLite file that is not a ChEMBL dump is refused when registering.

        Registering silently would defer the failure to a confusing SQL error much later.
        """
        from Capricho.chembl.api.downloader import set_chembl_db_path

        path = self.store / "chembl_36.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE something_else (id INTEGER)")
        con.commit()
        con.close()

        with self.assertRaises(ValueError) as ctx:
            set_chembl_db_path(path)
        self.assertIn("molecule_dictionary", str(ctx.exception))
        self.assertFalse(self.config_file("36").exists())


class TestReleaseIsVerified(LocalDatabaseTestCase):
    """The release a database is registered under must be the release it actually holds.

    A mislabelled registration is the dangerous kind of mistake: it does not fail, it
    silently misreports the provenance of every dataset drawn from that database, and the
    schema differences between releases then change results rather than raising.
    """

    def test_wrong_version_flag_is_refused(self):
        """Test that registering a ChEMBL 36 dump as ChEMBL 37 is refused, not taken on trust."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        db_path = self.make_db("36")

        with self.assertRaises(ValueError) as ctx:
            set_chembl_db_path(db_path, version=37)

        message = str(ctx.exception)
        self.assertIn("36", message)
        self.assertIn("37", message)
        self.assertFalse(self.config_file("37").exists())

    def test_misleading_file_name_is_refused(self):
        """Test that a renamed dump is caught, since the file name is not evidence of anything."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        renamed = self.make_db("36", reports="36", name="chembl_32.db")

        with self.assertRaises(ValueError) as ctx:
            set_chembl_db_path(renamed)

        self.assertIn("36", str(ctx.exception))
        self.assertFalse(self.config_file("32").exists())

    def test_one_bad_database_registers_none_of_a_directory(self):
        """Test that a directory is registered all or not at all, leaving no half-applied state."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        self.make_db("35")
        self.make_db("36", reports="34", name="chembl_36.db")

        with self.assertRaises(ValueError):
            set_chembl_db_path(self.store)

        self.assertFalse(self.config_file("35").exists())
        self.assertFalse(self.config_file("36").exists())

    def test_matching_release_registers_quietly(self):
        """Test that a database stating the release it is registered under is accepted."""
        from Capricho.chembl.api.downloader import set_chembl_db_path

        db_path = self.make_db("36")

        self.assertEqual(set_chembl_db_path(db_path, version=36), {"36": db_path})

    def test_database_without_a_version_table_is_registered_with_a_warning(self):
        """Test that a dump stating no release is still usable, but says the name was trusted."""
        from Capricho.chembl.api.downloader import set_chembl_db_path
        from Capricho.logger import logger

        db_path = self.make_db("36", reports=None)

        messages = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            registered = set_chembl_db_path(db_path)
        finally:
            logger.remove(handler_id)

        self.assertEqual(registered, {"36": db_path})
        self.assertTrue(
            any("file name" in message for message in messages),
            f"registering on the file name alone was not warned about: {messages}",
        )


class TestQueryRegisteredDatabase(LocalDatabaseTestCase):
    """Querying against a registered database instead of the pystow tree."""

    def test_activity_query_reads_the_registered_file(self):
        """Test that data is served from the registered path with nothing in the pystow tree."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql, set_chembl_db_path

        set_chembl_db_path(self.make_db("36"))

        df = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version="36")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["molecule_chembl_id"], "CHEMBL25")
        self.assertFalse((Path(self.pystow_dir.name) / "chembl" / "36").exists())

    def test_each_registered_release_serves_its_own_data(self):
        """Test that releases stay separate, so --version selects among registered databases."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql, set_chembl_db_path

        self.make_db("32", with_release_column=False)
        self.make_db("36")
        set_chembl_db_path(self.store)

        # docs.chembl_release_id only exists from ChEMBL 33 on, so the two dumps differ in schema.
        released = get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version="32")
        self.assertTrue(released["chembl_release"].isna().all())
        self.assertEqual(
            get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version="36").iloc[0][
                "chembl_release"
            ],
            30,
        )

    def test_the_database_being_read_is_reported(self):
        """Test that a normal run says which database it reads, so the source is never a guess."""
        from Capricho.chembl.api.downloader import get_full_activity_data_sql, set_chembl_db_path
        from Capricho.logger import logger

        db_path = self.make_db("36")
        set_chembl_db_path(db_path)

        messages = []
        handler_id = logger.add(messages.append, level="INFO")
        try:
            get_full_activity_data_sql(target_chembl_ids=["CHEMBL205"], version="36")
        finally:
            logger.remove(handler_id)

        self.assertTrue(
            any(str(db_path) in message for message in messages),
            f"the database path was not reported at INFO level: {messages}",
        )

    def test_moved_database_raises_an_actionable_error(self):
        """Test that a registered path that has since gone away says how to fix the registration."""
        from Capricho.chembl.api.downloader import check_and_download_chembl_db, set_chembl_db_path

        db_path = self.make_db("36")
        set_chembl_db_path(db_path)
        db_path.unlink()

        with self.assertRaises(FileNotFoundError) as ctx:
            check_and_download_chembl_db(version="36")
        self.assertIn("--set-from-path", str(ctx.exception))

    def test_unset_forgets_the_registration(self):
        """Test that unsetting drops the configuration, restoring the download behaviour."""
        from Capricho.chembl.api.downloader import set_chembl_db_path, unset_chembl_db_path

        set_chembl_db_path(self.make_db("36"))

        self.assertTrue(unset_chembl_db_path(version="36"))
        self.assertFalse(self.config_file("36").exists())
        self.assertFalse(unset_chembl_db_path(version="36"))


class TestDownloadCommand(LocalDatabaseTestCase):
    """The `capricho download` options that reach the registration functions."""

    def test_set_from_path_registers_the_database(self):
        """Test that the CLI option registers a release the same way the API does."""
        from typer.testing import CliRunner

        from Capricho.cli.main import app

        result = CliRunner().invoke(app, ["download", "--set-from-path", str(self.make_db("36"))])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(self.config_file("36").exists())

    def test_prefix_and_set_from_path_are_mutually_exclusive(self):
        """Test that combining a download location with a registered path is refused.

        The two describe different things, and honouring one silently would surprise the user.
        """
        from typer.testing import CliRunner

        from Capricho.cli.main import app

        result = CliRunner().invoke(
            app, ["download", "--set-from-path", str(self.make_db("36")), "--prefix", "elsewhere"]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(self.config_file("36").exists())

    def test_unset_path_removes_the_registration(self):
        """Test that the CLI can undo a registration without hand-editing files under ~/.data."""
        from typer.testing import CliRunner

        from Capricho.chembl.api.downloader import set_chembl_db_path
        from Capricho.cli.main import app

        set_chembl_db_path(self.make_db("36"))

        result = CliRunner().invoke(app, ["download", "--unset-path", "--version", "36"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(self.config_file("36").exists())


if __name__ == "__main__":
    unittest.main()
