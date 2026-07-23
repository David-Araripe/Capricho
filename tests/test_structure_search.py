"""Tests for looking up compounds in ChEMBL from SMILES strings.

Tests are split into three groups. The first needs neither the ChEMBL database nor FPSim2 and
covers query preparation and input validation. The second builds a small fingerprint index with
real molecules to check the engine contract the search code relies on. The third runs against a
locally downloaded ChEMBL database and is skipped when one is not available.
"""

import tempfile
import unittest
from pathlib import Path

from Capricho.chembl.api.downloader import (
    get_compounds_by_inchikey_sql,
    get_compounds_by_molregno_sql,
)
from Capricho.chembl.similarity import (
    _as_smiles_list,
    _describe_queries,
    search_by_similarity,
    search_by_structure,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
ASPIRIN_CONNECTIVITY = "BSYNRYMUTXBXSQ"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"

try:
    import FPSim2  # noqa: F401

    HAS_FPSIM2 = True
except ImportError:
    HAS_FPSIM2 = False


def find_local_chembl_version():
    """Return a locally downloaded ChEMBL version, or None. Never triggers a download."""
    import pystow
    from chembl_downloader.api import _find_sqlite_file

    root = pystow.join("chembl")
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and _find_sqlite_file(path) is not None:
            return path.name
    return None


LOCAL_CHEMBL_VERSION = find_local_chembl_version()
HAS_CHEMBL_DB = LOCAL_CHEMBL_VERSION is not None


class TestQueryPreparation(unittest.TestCase):
    """Deriving the structure keys used to look a query up in ChEMBL."""

    def test_as_smiles_list_accepts_single_string(self):
        self.assertEqual(_as_smiles_list(ASPIRIN), [ASPIRIN])

    def test_as_smiles_list_preserves_sequences(self):
        self.assertEqual(_as_smiles_list([ASPIRIN, IBUPROFEN]), [ASPIRIN, IBUPROFEN])

    def test_derives_inchikey_and_connectivity(self):
        queries = _describe_queries([ASPIRIN], standardize=True, chirality=True, n_jobs=1)
        self.assertEqual(queries.loc[0, "query_inchikey"], ASPIRIN_KEY)
        self.assertEqual(queries.loc[0, "connectivity"], ASPIRIN_CONNECTIVITY)

    def test_standardization_strips_salts_to_the_parent_key(self):
        """A salt form should resolve to the same key as the parent structure."""
        queries = _describe_queries([f"{ASPIRIN}.[Na+]"], standardize=True, chirality=True, n_jobs=1)
        self.assertEqual(queries.loc[0, "query_inchikey"], ASPIRIN_KEY)

    def test_unstandardized_query_keeps_the_input_smiles(self):
        queries = _describe_queries([ASPIRIN], standardize=False, chirality=True, n_jobs=1)
        self.assertEqual(queries.loc[0, "standard_smiles"], ASPIRIN)
        self.assertEqual(queries.loc[0, "query_inchikey"], ASPIRIN_KEY)

    def test_unparseable_smiles_is_kept_with_null_keys(self):
        """Queries RDKit cannot parse must survive as rows rather than being dropped."""
        queries = _describe_queries(
            [ASPIRIN, "this-is-not-a-smiles"], standardize=True, chirality=True, n_jobs=1
        )
        self.assertEqual(len(queries), 2)
        self.assertIsNone(queries.loc[1, "query_inchikey"])
        self.assertIsNone(queries.loc[1, "connectivity"])


class TestLookupValidation(unittest.TestCase):
    """Input validation happens before any database access, so bad input fails fast."""

    def test_inchikey_lookup_requires_a_query(self):
        with self.assertRaises(ValueError):
            get_compounds_by_inchikey_sql()

    def test_malformed_connectivity_is_rejected(self):
        for bad in ["TOOSHORT", "bsynrymutxbxsq", "BSYNRYMUTXBXSQ*", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]:
            with self.subTest(connectivity=bad), self.assertRaises(ValueError):
                get_compounds_by_inchikey_sql(connectivities=[bad])

    def test_molregno_lookup_requires_a_query(self):
        with self.assertRaises(ValueError):
            get_compounds_by_molregno_sql([])

    def test_similarity_rejects_out_of_range_threshold(self):
        for bad in [-0.1, 1.5]:
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                search_by_similarity(ASPIRIN, threshold=bad)

    def test_similarity_rejects_non_positive_top_k(self):
        with self.assertRaises(ValueError):
            search_by_similarity(ASPIRIN, top_k=0)

    def test_structure_search_short_circuits_when_no_query_resolves(self):
        """With nothing to look up the database is never touched, but rows are still returned."""
        hits = search_by_structure(["bad!!", "also-bad!!"])
        self.assertEqual(hits["match_type"].tolist(), ["no_match", "no_match"])
        self.assertIn("molecule_chembl_id", hits.columns)
        self.assertTrue(hits["molecule_chembl_id"].isna().all())


@unittest.skipUnless(HAS_FPSIM2, "FPSim2 is not installed")
class TestFingerprintIndexContract(unittest.TestCase):
    """Check the FPSim2 behaviour the search code depends on, using a small real index."""

    @classmethod
    def setUpClass(cls):
        from FPSim2.io import create_db_file

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.index_path = str(Path(cls._tmpdir.name) / "test_index.h5")
        # molregno-like integer ids, as used by the index ChEMBL publishes
        cls.molecules = [
            [ASPIRIN, 1],
            ["CC(=O)Oc1ccccc1C(=O)[O-]", 2],
            ["c1ccccc1C(=O)O", 3],
            [IBUPROFEN, 4],
            ["CCO", 5],
        ]
        create_db_file(
            mols_source=cls.molecules,
            filename=cls.index_path,
            mol_format="smiles",
            fp_type="Morgan",
            fp_params={"radius": 2, "fpSize": 2048},
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        from FPSim2 import FPSim2Engine

        self.engine = FPSim2Engine(self.index_path)

    def test_results_expose_mol_id_and_coeff(self):
        """The search code renames these two fields, so their names are part of the contract."""
        results = self.engine.similarity(ASPIRIN, threshold=0.1, n_workers=1)
        self.assertEqual(set(results.dtype.names), {"mol_id", "coeff"})

    def test_query_finds_itself_with_perfect_similarity(self):
        results = self.engine.similarity(ASPIRIN, threshold=0.1, n_workers=1)
        best = max(results, key=lambda row: row["coeff"])
        self.assertEqual(best["mol_id"], 1)
        self.assertAlmostEqual(float(best["coeff"]), 1.0, places=5)

    def test_threshold_excludes_dissimilar_compounds(self):
        """Ethanol shares almost nothing with aspirin and must fall below a high threshold."""
        found = {int(row["mol_id"]) for row in self.engine.similarity(ASPIRIN, threshold=0.6)}
        self.assertIn(1, found)
        self.assertNotIn(5, found)

    def test_top_k_limits_the_number_of_hits(self):
        results = self.engine.top_k(ASPIRIN, k=2, threshold=0.1, n_workers=1)
        self.assertEqual(len(results), 2)

    def test_results_are_ordered_by_descending_similarity(self):
        coeffs = [float(row["coeff"]) for row in self.engine.similarity(ASPIRIN, threshold=0.1)]
        self.assertEqual(coeffs, sorted(coeffs, reverse=True))


@unittest.skipUnless(HAS_CHEMBL_DB, "No ChEMBL database has been downloaded locally")
class TestStructureSearchAgainstChembl(unittest.TestCase):
    """End-to-end lookups against a real, locally downloaded ChEMBL database."""

    @classmethod
    def setUpClass(cls):
        cls.version = LOCAL_CHEMBL_VERSION

    def test_aspirin_resolves_to_chembl25(self):
        hits = search_by_structure(ASPIRIN, version=self.version)
        exact = hits[hits["match_type"] == "exact"]
        self.assertEqual(exact["molecule_chembl_id"].tolist(), ["CHEMBL25"])
        self.assertEqual(exact["standard_inchi_key"].tolist(), [ASPIRIN_KEY])

    def test_racemate_also_matches_its_enantiomers_by_connectivity(self):
        """Racemic ibuprofen is an exact hit; the single enantiomers match on connectivity."""
        hits = search_by_structure(IBUPROFEN, version=self.version)
        exact = hits[hits["match_type"] == "exact"]
        connectivity = hits[hits["match_type"] == "connectivity"]
        self.assertEqual(exact["molecule_chembl_id"].tolist(), ["CHEMBL521"])
        self.assertIn("CHEMBL175", connectivity["molecule_chembl_id"].tolist())
        # every connectivity hit shares the skeleton but differs in the full key
        self.assertTrue((connectivity["standard_inchi_key"].str[:14] == "HEFNNWSXXWATRW").all())
        self.assertNotIn(ASPIRIN_KEY, connectivity["standard_inchi_key"].tolist())

    def test_unmatched_query_is_reported_rather_than_dropped(self):
        hits = search_by_structure([ASPIRIN, "this-is-not-a-smiles"], version=self.version)
        unmatched = hits[hits["match_type"] == "no_match"]
        self.assertEqual(unmatched["query_smiles"].tolist(), ["this-is-not-a-smiles"])
        self.assertTrue(unmatched["molecule_chembl_id"].isna().all())

    def test_every_query_appears_in_the_output(self):
        queries = [ASPIRIN, IBUPROFEN, "this-is-not-a-smiles"]
        hits = search_by_structure(queries, version=self.version)
        self.assertEqual(set(hits["query_smiles"]), set(queries))

    def test_hits_report_their_parent_compound(self):
        hits = search_by_structure(ASPIRIN, version=self.version)
        exact = hits[hits["match_type"] == "exact"]
        self.assertEqual(exact["parent_chembl_id"].tolist(), ["CHEMBL25"])

    def test_molregno_lookup_round_trips(self):
        hits = search_by_structure(ASPIRIN, version=self.version)
        molregno = int(hits.loc[hits["match_type"] == "exact", "molregno"].iloc[0])
        compounds = get_compounds_by_molregno_sql([molregno], version=self.version)
        self.assertEqual(compounds["molecule_chembl_id"].tolist(), ["CHEMBL25"])


@unittest.skipUnless(HAS_CHEMBL_DB and HAS_FPSIM2, "Needs FPSim2 and a local ChEMBL database")
class TestSimilaritySearchAgainstChembl(unittest.TestCase):
    """End-to-end similarity searches against the fingerprint index ChEMBL publishes."""

    @classmethod
    def setUpClass(cls):
        from Capricho.chembl.api.fingerprint_index import (
            check_and_download_fingerprint_index,
        )

        cls.version = LOCAL_CHEMBL_VERSION
        try:
            check_and_download_fingerprint_index(version=cls.version)
        except Exception as exc:  # network or a release without a published index
            raise unittest.SkipTest(f"Fingerprint index unavailable: {exc}")

    def test_query_finds_itself_at_perfect_similarity(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.9, version=self.version)
        perfect = hits[hits["similarity"] == 1.0]["molecule_chembl_id"].tolist()
        self.assertIn("CHEMBL25", perfect)

    def test_salt_forms_are_found_even_though_their_connectivity_differs(self):
        """Fingerprints of a salt still contain the parent's bits, unlike its InChI key."""
        hits = search_by_similarity(ASPIRIN, threshold=0.9, version=self.version)
        parents = hits[hits["similarity"] == 1.0]["parent_chembl_id"].tolist()
        self.assertGreater(len(parents), 1)
        self.assertIn("CHEMBL25", parents)

    def test_top_k_limits_the_number_of_hits(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.5, top_k=3, version=self.version)
        self.assertEqual(len(hits), 3)

    def test_higher_threshold_returns_a_subset(self):
        loose = search_by_similarity(ASPIRIN, threshold=0.6, version=self.version)
        strict = search_by_similarity(ASPIRIN, threshold=0.9, version=self.version)
        self.assertLessEqual(len(strict), len(loose))
        self.assertTrue(set(strict["molecule_chembl_id"]).issubset(set(loose["molecule_chembl_id"])))

    def test_results_are_sorted_by_descending_similarity(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.6, version=self.version)
        self.assertEqual(hits["similarity"].tolist(), sorted(hits["similarity"], reverse=True))

    def test_unusable_query_yields_an_empty_frame_without_raising(self):
        hits = search_by_similarity("this-is-not-a-smiles", version=self.version)
        self.assertTrue(hits.empty)
        self.assertIn("molecule_chembl_id", hits.columns)


if __name__ == "__main__":
    unittest.main()
