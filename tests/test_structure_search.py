"""Tests for looking up compounds in ChEMBL from SMILES strings.

Tests are split into three groups. The first needs neither the ChEMBL database nor FPSim2 and
covers query preparation and input validation. The second builds a small fingerprint index with
real molecules to check the engine contract the search code relies on. The third runs against a
locally downloaded ChEMBL database and is skipped when one is not available.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from Capricho.chembl.api.downloader import (
    COMPOUND_HIT_COLUMNS,
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
        self.assertTrue(pd.isna(queries.loc[1, "query_inchikey"]))
        self.assertTrue(pd.isna(queries.loc[1, "connectivity"]))

    def test_empty_and_missing_inputs_do_not_abort_valid_queries(self):
        inputs = [ASPIRIN, "", "   ", None, "this-is-not-a-smiles"]
        for standardize in [True, False]:
            with self.subTest(standardize=standardize):
                queries = _describe_queries(inputs, standardize, chirality=True, n_jobs=1)
                self.assertEqual(len(queries), len(inputs))
                self.assertEqual(queries.loc[0, "query_inchikey"], ASPIRIN_KEY)
                self.assertTrue(queries.loc[1:, "query_inchikey"].isna().all())

    def test_repeated_queries_are_standardized_once(self):
        from chemFilters.chem.standardizers import ChemStandardizer

        standardizer = Mock(wraps=ChemStandardizer(from_smi=False, n_jobs=1, progress=False))
        with patch("Capricho.chembl.similarity.ChemStandardizer", return_value=standardizer):
            queries = _describe_queries([ASPIRIN] * 3, True, True, 1)
        standardizer.assert_called_once()
        self.assertEqual(len(standardizer.call_args.args[0]), 1)
        self.assertEqual(queries["query_inchikey"].tolist(), [ASPIRIN_KEY] * 3)

    def test_isotope_labels_are_removed_only_when_standardizing(self):
        normalized = _describe_queries(["[13CH3]CO"], True, True, 1)
        raw = _describe_queries(["[13CH3]CO"], False, True, 1)
        self.assertEqual(normalized.loc[0, "standard_smiles"], "CCO")
        self.assertEqual(normalized.loc[0, "query_inchikey"], "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        self.assertEqual(raw.loc[0, "query_inchikey"], "LFQSCWFLJHTTHZ-OUBTZVSYSA-N")

    def test_protonation_is_preserved_without_standardization(self):
        normalized = _describe_queries(["[nH+]1ccccc1"], True, True, 1)
        raw = _describe_queries(["[nH+]1ccccc1"], False, True, 1)
        self.assertEqual(normalized.loc[0, "standard_smiles"], "c1ccncc1")
        self.assertNotEqual(normalized.loc[0, "query_inchikey"], raw.loc[0, "query_inchikey"])

    def test_standardization_preserves_specified_stereo_by_default(self):
        for inputs in [["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"], ["C/C=C/C", "C/C=C\\C"]]:
            with self.subTest(inputs=inputs):
                queries = _describe_queries(inputs, True, True, 1)
                self.assertEqual(queries["query_inchikey"].nunique(), 2)
                self.assertEqual(queries["connectivity"].nunique(), 1)
                achiral = _describe_queries(inputs, True, False, 1)
                self.assertEqual(achiral["query_inchikey"].nunique(), 1)

    def test_truncated_standardizer_results_are_rejected(self):
        standardizer = Mock(return_value=["CCO"])
        with (
            patch("Capricho.chembl.similarity.ChemStandardizer", return_value=standardizer),
            self.assertRaisesRegex(ValueError, "zip"),
        ):
            _describe_queries(["CCO", "CCN"], True, True, 1)

    def test_truncated_inchikey_results_are_rejected(self):
        writer = Mock(return_value=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"])
        with (
            patch("Capricho.chembl.similarity.InchiHandling", return_value=writer),
            self.assertRaisesRegex(ValueError, "zip"),
        ):
            _describe_queries(["CCO", "CCN"], False, True, 1)


class TestLookupValidation(unittest.TestCase):
    """Input validation happens before any database access, so bad input fails fast."""

    def test_inchikey_lookup_requires_a_query(self):
        with self.assertRaises(ValueError):
            get_compounds_by_inchikey_sql()

    def test_malformed_connectivity_is_rejected(self):
        for bad in ["TOOSHORT", "bsynrymutxbxsq", "BSYNRYMUTXBXSQ*", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]:
            with self.subTest(connectivity=bad), self.assertRaises(ValueError):
                get_compounds_by_inchikey_sql(connectivities=[bad])

    def test_malformed_inchikey_is_rejected(self):
        for bad in ["TOOSHORT", "BSYNRYMUTXBXSQ", "BSYNRYMUTXBXSQ-UHFFFAOYSA-*", "' OR 1=1 --"]:
            with self.subTest(inchikey=bad), self.assertRaises(ValueError):
                get_compounds_by_inchikey_sql(inchi_keys=[bad])

    def test_molregno_lookup_requires_a_query(self):
        with self.assertRaises(ValueError):
            get_compounds_by_molregno_sql([])

    def test_similarity_rejects_out_of_range_threshold(self):
        for bad in [-0.1, 0, 1.5]:
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                search_by_similarity(ASPIRIN, threshold=bad)

    def test_similarity_rejects_invalid_top_k(self):
        for bad in [0, 1.5, True]:
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                search_by_similarity(ASPIRIN, top_k=bad)

    def test_similarity_rejects_invalid_worker_count(self):
        for bad in [0, -1, 1.5, True]:
            with self.subTest(n_workers=bad), self.assertRaises(ValueError):
                search_by_similarity(ASPIRIN, n_workers=bad)

    def test_empty_similarity_query_does_not_load_the_index(self):
        with patch("Capricho.chembl.similarity.load_fingerprint_index") as load_index:
            result = search_by_similarity([])
        load_index.assert_not_called()
        self.assertTrue(result.empty)
        self.assertIn("molecule_chembl_id", result.columns)

    def test_invalid_similarity_query_does_not_load_the_index(self):
        for query in ["this-is-not-a-smiles", "", "   "]:
            with (
                self.subTest(query=query),
                patch("Capricho.chembl.similarity.load_fingerprint_index") as load_index,
            ):
                result = search_by_similarity(query)
            load_index.assert_not_called()
            self.assertEqual(result["query_smiles"].tolist(), [query])
            self.assertTrue(result["molecule_chembl_id"].isna().all())

    def test_empty_structure_query_does_not_access_the_database(self):
        with patch("Capricho.chembl.similarity.get_compounds_by_inchikey_sql") as lookup:
            result = search_by_structure([])
        lookup.assert_not_called()
        self.assertTrue(result.empty)
        self.assertIn("match_type", result.columns)

    def test_large_molregno_lookups_are_batched(self):
        empty = pd.DataFrame(columns=COMPOUND_HIT_COLUMNS)
        with (
            patch("Capricho.chembl.api.downloader.check_and_download_chembl_db", return_value={}),
            patch("Capricho.chembl.api.downloader.run_query", return_value=empty) as run_query,
        ):
            result = get_compounds_by_molregno_sql(range(1001))
        self.assertEqual(run_query.call_count, 3)
        self.assertTrue(result.empty)

    def test_structure_search_short_circuits_when_no_query_resolves(self):
        """With nothing to look up the database is never touched, but rows are still returned."""
        hits = search_by_structure(["bad!!", "also-bad!!"])
        self.assertEqual(hits["match_type"].tolist(), ["no_match", "no_match"])
        self.assertIn("molecule_chembl_id", hits.columns)
        self.assertTrue(hits["molecule_chembl_id"].isna().all())

    def test_empty_structure_smiles_never_resolves_or_downloads_a_database(self):
        with (
            patch("Capricho.chembl.similarity._latest_version") as latest,
            patch("Capricho.chembl.similarity.get_compounds_by_inchikey_sql") as lookup,
        ):
            hits = search_by_structure(["", "   ", None])
        latest.assert_not_called()
        lookup.assert_not_called()
        self.assertEqual(hits["match_type"].tolist(), ["no_match"] * 3)
        self.assertEqual(hits["query_index"].tolist(), [0, 1, 2])
        self.assertIsNone(hits.attrs["capricho_search"]["chembl_version"])


class TestSimilarityResultAssembly(unittest.TestCase):
    """Result assembly preserves one result set per input query occurrence."""

    @staticmethod
    def _engine(found=None, error=None):
        engine = Mock()
        engine.fp_type = "Morgan"
        engine.fp_params = {"radius": 2, "fpSize": 2048}
        engine.rdkit_ver = "2022.09.4"
        engine.fpsim2_ver = "0.7.3"
        if error is not None:
            engine.similarity.side_effect = error
        else:
            engine.similarity.return_value = (
                found
                if found is not None
                else np.array([(1, 0.9)], dtype=[("mol_id", "<i8"), ("coeff", "<f8")])
            )
        return engine

    @staticmethod
    def _compounds():
        return pd.DataFrame(
            [
                {
                    "molecule_chembl_id": "CHEMBL1",
                    "molregno": 1,
                    "canonical_smiles": "C",
                    "standard_inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                    "parent_chembl_id": "CHEMBL1",
                    "parent_smiles": "C",
                }
            ]
        )

    def test_duplicate_queries_do_not_multiply_hits(self):
        engine = self._engine()
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
            patch(
                "Capricho.chembl.similarity.get_compounds_by_molregno_sql",
                return_value=self._compounds(),
            ),
        ):
            result = search_by_similarity(["CC", "CC"], version=37)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["query_smiles"].tolist(), ["CC", "CC"])
        self.assertEqual(result["query_index"].tolist(), [0, 1])
        engine.similarity.assert_called_once()

    def test_query_order_is_preserved(self):
        engine = self._engine()
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
            patch(
                "Capricho.chembl.similarity.get_compounds_by_molregno_sql",
                return_value=self._compounds(),
            ),
        ):
            result = search_by_similarity(["N", "C"], version=37)

        self.assertEqual(result["query_smiles"].tolist(), ["N", "C"])

    def test_operational_search_errors_are_not_reported_as_no_match(self):
        engine = self._engine(error=OSError("index read failed"))
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
            self.assertRaisesRegex(OSError, "index read failed"),
        ):
            search_by_similarity(ASPIRIN, version=37)

    def test_resolved_release_and_index_provenance_are_retained(self):
        engine = self._engine()
        with (
            patch("Capricho.chembl.similarity._latest_version", return_value="37") as latest,
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine) as load,
            patch(
                "Capricho.chembl.similarity.get_compounds_by_molregno_sql", return_value=self._compounds()
            ) as lookup,
        ):
            result = search_by_similarity("C", prefix=["custom"])
        latest.assert_called_once()
        load.assert_called_once_with(prefix=["custom"], version="37", in_memory=True)
        lookup.assert_called_once_with([1], prefix=["custom"], version="37")
        provenance = result.attrs["capricho_search"]
        self.assertEqual(provenance["chembl_version"], "37")
        self.assertEqual(provenance["fp_params"], {"radius": 2, "fpSize": 2048})
        self.assertEqual(provenance["index_rdkit_version"], "2022.09.4")
        self.assertEqual(provenance["index_fpsim2_version"], "0.7.3")
        self.assertIn("rdkit_version", provenance)
        self.assertIn("capricho_version", provenance)
        self.assertEqual(json.loads(json.dumps(provenance)), provenance)

    def test_mixed_invalid_inputs_retain_their_query_positions(self):
        engine = self._engine()
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
            patch("Capricho.chembl.similarity.get_compounds_by_molregno_sql", return_value=self._compounds()),
        ):
            result = search_by_similarity(["C", "", None, "bad!!", "C"], version=37)
        self.assertEqual(result["query_index"].tolist(), list(range(5)))
        self.assertEqual(result["molecule_chembl_id"].notna().tolist(), [True, False, False, False, True])
        engine.similarity.assert_called_once()

    def test_structure_results_keep_normalization_provenance_and_input_positions(self):
        with patch(
            "Capricho.chembl.similarity.get_compounds_by_inchikey_sql", return_value=self._compounds()
        ):
            result = search_by_structure(["C", "", "C"], standardize=False, version=37)
        self.assertEqual(result["query_index"].tolist(), [0, 1, 2])
        self.assertEqual(result["match_type"].tolist(), ["exact", "no_match", "exact"])
        self.assertEqual(result.attrs["capricho_search"]["chembl_version"], "37")
        self.assertFalse(result.attrs["capricho_search"]["standardize"])


class TestFingerprintIndexLoading(unittest.TestCase):
    """Optional dependency checks happen before any large download."""

    def test_missing_fpsim2_fails_before_checking_downloads(self):
        from Capricho.chembl.api import fingerprint_index

        with (
            patch.object(fingerprint_index, "_require_fpsim2", side_effect=ImportError("install FPSim2")),
            patch.object(fingerprint_index, "check_and_download_fingerprint_index") as download,
            self.assertRaisesRegex(ImportError, "install FPSim2"),
        ):
            fingerprint_index.load_fingerprint_index()
        download.assert_not_called()

    def tearDown(self):
        from Capricho.chembl.api.fingerprint_index import clear_fingerprint_index_cache

        clear_fingerprint_index_cache()

    def test_registered_database_config_selects_the_index_release_and_prefix(self):
        from Capricho.chembl.api import fingerprint_index

        configs = {"version": "36", "prefix": ["registered"], "path": "/data/chembl_36.db"}
        with (
            patch.object(fingerprint_index, "check_and_download_chembl_db", return_value=configs) as database,
            patch("chembl_downloader.api._download_helper", return_value=Path("index.h5")) as download,
        ):
            path = fingerprint_index.check_and_download_fingerprint_index(prefix=["requested"], version=36)
        database.assert_called_once_with(prefix=["requested"], version=36)
        download.assert_called_once_with(
            suffix=".h5", version="36", prefix=["registered"], return_version=False
        )
        self.assertEqual(path, Path("index.h5"))

    def test_unavailable_release_index_has_an_actionable_error(self):
        from Capricho.chembl.api import fingerprint_index

        with (
            patch.object(
                fingerprint_index,
                "check_and_download_chembl_db",
                return_value={"version": "24", "prefix": ["chembl"]},
            ),
            patch.object(
                fingerprint_index, "_download_fingerprint_asset", side_effect=ValueError("unavailable")
            ),
            self.assertRaisesRegex(FileNotFoundError, "version 24"),
        ):
            fingerprint_index.check_and_download_fingerprint_index(version=24)

    def test_missing_private_downloader_api_does_not_break_structure_only_imports(self):
        from importlib import reload

        import chembl_downloader.api as downloader_api

        from Capricho.chembl.api import fingerprint_index

        helper = downloader_api._download_helper
        try:
            del downloader_api._download_helper
            reload(fingerprint_index)
            with self.assertRaisesRegex(ImportError, "upgrade chembl_downloader"):
                fingerprint_index._download_fingerprint_asset({"version": "37", "prefix": ["chembl"]})
            self.assertEqual(search_by_structure([])["query_smiles"].tolist(), [])
        finally:
            downloader_api._download_helper = helper

    def test_engine_cache_is_keyed_by_path_and_memory_mode_and_can_be_cleared(self):
        from Capricho.chembl.api import fingerprint_index

        factory = Mock(side_effect=lambda *args, **kwargs: object())
        with (
            patch.object(fingerprint_index, "_require_fpsim2", return_value=factory),
            patch.object(
                fingerprint_index, "check_and_download_fingerprint_index", return_value=Path("index.h5")
            ),
        ):
            first = fingerprint_index.load_fingerprint_index(version=37)
            self.assertIs(first, fingerprint_index.load_fingerprint_index(version=37))
            self.assertEqual(factory.call_count, 1)
            disk = fingerprint_index.load_fingerprint_index(version=37, in_memory=False)
            self.assertIsNot(first, disk)
            self.assertEqual(factory.call_count, 2)
            fingerprint_index.clear_fingerprint_index_cache()
            self.assertIsNot(disk, fingerprint_index.load_fingerprint_index(version=37, in_memory=False))
            self.assertEqual(factory.call_count, 3)
            factory.assert_called_with("index.h5", in_memory_fps=False)
            with patch.object(
                fingerprint_index, "check_and_download_fingerprint_index", return_value=Path("other.h5")
            ):
                fingerprint_index.load_fingerprint_index(version=36, in_memory=False)
            factory.assert_called_with("other.h5", in_memory_fps=False)


@unittest.skipUnless(HAS_FPSIM2, "FPSim2 is not installed")
class TestFingerprintIndexContract(unittest.TestCase):
    """Check the FPSim2 behaviour the search code depends on, using a small real index."""

    @classmethod
    def setUpClass(cls):
        from FPSim2.io import create_db_file

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.index_path = str(Path(cls._tmpdir.name) / "test_index.h5")
        # molregno-like integer ids, as used by the index ChEMBL publishes
        molecules = [
            [ASPIRIN, 1],
            ["CC(=O)Oc1ccccc1C(=O)[O-]", 2],
            ["c1ccccc1C(=O)O", 3],
            [IBUPROFEN, 4],
            ["CCO", 5],
        ]
        cls.smiles = [row[0] for row in molecules]
        create_db_file(
            mols_source=molecules,
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

    @staticmethod
    def _metadata(molregnos, **kwargs):
        return pd.DataFrame(
            [
                {
                    "molecule_chembl_id": f"CHEMBL{molregno}",
                    "molregno": molregno,
                    "canonical_smiles": None,
                    "standard_inchi_key": None,
                    "parent_chembl_id": None,
                    "parent_smiles": None,
                }
                for molregno in molregnos
            ]
        )

    def test_capricho_wrapper_uses_the_real_in_memory_engine(self):
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=self.engine),
            patch("Capricho.chembl.similarity.get_compounds_by_molregno_sql", side_effect=self._metadata),
        ):
            hits = search_by_similarity(ASPIRIN, threshold=0.1, top_k=2, version=37)

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits.iloc[0]["molregno"], 1)
        self.assertAlmostEqual(hits.iloc[0]["similarity"], 1.0, places=5)

    def test_capricho_wrapper_supports_on_disk_searches(self):
        from FPSim2 import FPSim2Engine

        engine = FPSim2Engine(self.index_path, in_memory_fps=False)
        with (
            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
            patch("Capricho.chembl.similarity.get_compounds_by_molregno_sql", side_effect=self._metadata),
        ):
            all_hits = search_by_similarity(ASPIRIN, threshold=0.6, in_memory=False, version=37)
            top_hits = search_by_similarity(ASPIRIN, threshold=0.1, top_k=2, in_memory=False, version=37)

        self.assertIn(1, all_hits["molregno"].tolist())
        self.assertEqual(len(top_hits), 2)
        self.assertEqual(top_hits.iloc[0]["molregno"], 1)

    def test_metrics_workers_and_top_k_match_independent_rdkit_scores(self):
        from FPSim2 import FPSim2Engine

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in self.smiles]
        references = {
            "tanimoto": DataStructs.TanimotoSimilarity,
            "dice": DataStructs.DiceSimilarity,
            "cosine": DataStructs.CosineSimilarity,
        }
        for in_memory in [True, False]:
            engine = FPSim2Engine(self.index_path, in_memory_fps=in_memory)
            for workers in [1, 2]:
                for metric, reference in references.items():
                    expected = {
                        i + 1: reference(fps[0], fp)
                        for i, fp in enumerate(fps)
                        if reference(fps[0], fp) >= 0.2
                    }
                    for top_k in [None, 2]:
                        with (
                            self.subTest(in_memory=in_memory, workers=workers, metric=metric, top_k=top_k),
                            patch("Capricho.chembl.similarity.load_fingerprint_index", return_value=engine),
                            patch(
                                "Capricho.chembl.similarity.get_compounds_by_molregno_sql",
                                side_effect=self._metadata,
                            ),
                        ):
                            hits = search_by_similarity(
                                ASPIRIN,
                                threshold=0.2,
                                top_k=top_k,
                                metric=metric,
                                n_workers=workers,
                                in_memory=in_memory,
                                version=37,
                            )
                            actual = dict(zip(hits["molregno"], hits["similarity"], strict=True))
                            if top_k is None:
                                self.assertEqual(set(actual), set(expected))
                            else:
                                expected_scores = sorted(expected.values(), reverse=True)[:top_k]
                                self.assertEqual(len(hits), len(expected_scores))
                                np.testing.assert_allclose(hits["similarity"], expected_scores, rtol=1e-6)
                            for molregno, coefficient in actual.items():
                                self.assertAlmostEqual(coefficient, expected[molregno], places=6)


@unittest.skipUnless(HAS_CHEMBL_DB, "No ChEMBL database has been downloaded locally")
class TestStructureSearchAgainstChembl(unittest.TestCase):
    """End-to-end lookups against a real, locally downloaded ChEMBL database."""

    def test_aspirin_resolves_to_chembl25(self):
        hits = search_by_structure(ASPIRIN, version=LOCAL_CHEMBL_VERSION)
        exact = hits[hits["match_type"] == "exact"]
        self.assertEqual(exact["molecule_chembl_id"].tolist(), ["CHEMBL25"])
        self.assertEqual(exact["standard_inchi_key"].tolist(), [ASPIRIN_KEY])

    def test_racemate_also_matches_its_enantiomers_by_connectivity(self):
        """Racemic ibuprofen is an exact hit; the single enantiomers match on connectivity."""
        hits = search_by_structure(IBUPROFEN, version=LOCAL_CHEMBL_VERSION)
        exact = hits[hits["match_type"] == "exact"]
        connectivity = hits[hits["match_type"] == "connectivity"]
        self.assertEqual(exact["molecule_chembl_id"].tolist(), ["CHEMBL521"])
        self.assertIn("CHEMBL175", connectivity["molecule_chembl_id"].tolist())
        # every connectivity hit shares the skeleton but differs in the full key
        self.assertTrue((connectivity["standard_inchi_key"].str[:14] == "HEFNNWSXXWATRW").all())
        self.assertNotIn(ASPIRIN_KEY, connectivity["standard_inchi_key"].tolist())

    def test_unmatched_query_is_reported_rather_than_dropped(self):
        hits = search_by_structure([ASPIRIN, "this-is-not-a-smiles"], version=LOCAL_CHEMBL_VERSION)
        unmatched = hits[hits["match_type"] == "no_match"]
        self.assertEqual(unmatched["query_smiles"].tolist(), ["this-is-not-a-smiles"])
        self.assertTrue(unmatched["molecule_chembl_id"].isna().all())

    def test_every_query_appears_in_the_output(self):
        queries = [ASPIRIN, IBUPROFEN, "this-is-not-a-smiles"]
        hits = search_by_structure(queries, version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(set(hits["query_smiles"]), set(queries))

    def test_query_order_is_preserved(self):
        queries = [IBUPROFEN, ASPIRIN]
        hits = search_by_structure(queries, version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(hits["query_smiles"].drop_duplicates().tolist(), queries)

    def test_hits_report_their_parent_compound(self):
        hits = search_by_structure(ASPIRIN, version=LOCAL_CHEMBL_VERSION)
        exact = hits[hits["match_type"] == "exact"]
        self.assertEqual(exact["parent_chembl_id"].tolist(), ["CHEMBL25"])

    def test_molregno_lookup_round_trips(self):
        hits = search_by_structure(ASPIRIN, version=LOCAL_CHEMBL_VERSION)
        molregno = int(hits.loc[hits["match_type"] == "exact", "molregno"].iloc[0])
        compounds = get_compounds_by_molregno_sql([molregno], version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(compounds["molecule_chembl_id"].tolist(), ["CHEMBL25"])


@unittest.skipUnless(HAS_CHEMBL_DB and HAS_FPSIM2, "Needs FPSim2 and a local ChEMBL database")
class TestSimilaritySearchAgainstChembl(unittest.TestCase):
    """End-to-end similarity searches against the fingerprint index ChEMBL publishes."""

    @classmethod
    def setUpClass(cls):
        from Capricho.chembl.api.fingerprint_index import (
            check_and_download_fingerprint_index,
        )

        check_and_download_fingerprint_index(version=LOCAL_CHEMBL_VERSION)

    def test_query_finds_itself_at_perfect_similarity(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.9, version=LOCAL_CHEMBL_VERSION)
        perfect = hits[hits["similarity"] == 1.0]["molecule_chembl_id"].tolist()
        self.assertIn("CHEMBL25", perfect)

    def test_salt_forms_are_found_even_though_their_connectivity_differs(self):
        """Fingerprints of a salt still contain the parent's bits, unlike its InChI key."""
        hits = search_by_similarity(ASPIRIN, threshold=0.9, version=LOCAL_CHEMBL_VERSION)
        parents = hits[hits["similarity"] == 1.0]["parent_chembl_id"].tolist()
        self.assertGreater(len(parents), 1)
        self.assertIn("CHEMBL25", parents)

    def test_top_k_limits_the_number_of_hits(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.5, top_k=3, version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(len(hits), 3)

    def test_higher_threshold_returns_a_subset(self):
        loose = search_by_similarity(ASPIRIN, threshold=0.6, version=LOCAL_CHEMBL_VERSION)
        strict = search_by_similarity(ASPIRIN, threshold=0.9, version=LOCAL_CHEMBL_VERSION)
        self.assertLessEqual(len(strict), len(loose))
        self.assertTrue(set(strict["molecule_chembl_id"]).issubset(set(loose["molecule_chembl_id"])))

    def test_results_are_sorted_by_descending_similarity(self):
        hits = search_by_similarity(ASPIRIN, threshold=0.6, version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(hits["similarity"].tolist(), sorted(hits["similarity"], reverse=True))

    def test_unusable_query_is_reported_rather_than_dropped(self):
        """Like search_by_structure, a query that yields nothing still owns a row."""
        hits = search_by_similarity("this-is-not-a-smiles", version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(hits["query_smiles"].tolist(), ["this-is-not-a-smiles"])
        self.assertTrue(hits["molecule_chembl_id"].isna().all())

    def test_every_query_appears_in_the_output(self):
        queries = [ASPIRIN, "this-is-not-a-smiles"]
        hits = search_by_similarity(queries, threshold=0.9, version=LOCAL_CHEMBL_VERSION)
        self.assertEqual(set(hits["query_smiles"]), set(queries))


if __name__ == "__main__":
    unittest.main()
