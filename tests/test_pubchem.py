"""Tests for the PubChem layer: shared rate limit budget, retries and batch resilience.

Only the transport is faked; the rate limiter, retry policy and curation are real code.
"""

import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pubchempy import PubChemHTTPError

from Capricho.logger import setup_logger
from Capricho.pubchem import core
from Capricho.pubchem.api import get_and_curate_multiple_compounds_result

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

CURATED_COLUMNS = {
    "name",
    "smiles",
    "pubchem_cid",
    "inchi",
    "inchikey",
    "isomeric_smiles",
    "canonical_smiles",
    "iupac_name",
    "synonyms",
}


class FakeCompound:
    """Stand-in for a pubchempy Compound exposing the fields the curation reads."""

    def __init__(self, cid, smiles=ASPIRIN_SMILES):
        self.cid = cid
        self.smiles = smiles
        self.connectivity_smiles = smiles
        self.inchi = f"InChI=1S/fake/{cid}"
        self.inchikey = f"FAKEINCHIKEY{cid}"
        self.iupac_name = f"fake compound {cid}"
        self.synonyms = [f"synonym of {cid}"]


def bad_gateway_error():
    return PubChemHTTPError(502, "Bad Gateway", [])


class PubChemTestCase(unittest.TestCase):
    """Captures log output and records the backoff waits instead of sleeping them."""

    def setUp(self):
        self.log_capture = StringIO()
        setup_logger(level="INFO", _sink=self.log_capture)
        self.addCleanup(setup_logger)

        self.waits = []
        time_patch = patch.object(core, "time", SimpleNamespace(sleep=self.waits.append))
        time_patch.start()
        self.addCleanup(time_patch.stop)


class TestSharedRateLimitBudget(unittest.TestCase):
    """Fetching synonyms costs an extra request, which must share the compound budget."""

    def test_synonym_requests_count_against_the_compound_budget(self):
        num_calls = 8
        max_per_second = 4  # the budget `Capricho.pubchem.core` configures

        with patch.object(core, "get_compounds", return_value=[FakeCompound(2244)]):
            start_time = time.perf_counter()
            for i in range(num_calls // 2):
                compound = core.get_compound_by(f"compound {i}")[0]
                core.get_compound_synonyms(compound)
            total_time = time.perf_counter() - start_time

        # Were the synonym requests outside the limiter, only half of the calls would be
        # paced and the batch would finish in about half the time
        minimum_time = (num_calls - 1) / max_per_second
        self.assertGreaterEqual(
            total_time,
            minimum_time - 0.02,
            f"{num_calls} requests took {total_time:.2f}s, faster than the {max_per_second}/s budget",
        )


class TestTransientErrorRetry(PubChemTestCase):
    """Requests answered with a 5xx are repeated before the failure is passed on."""

    def test_request_succeeding_after_a_transient_error(self):
        compound = FakeCompound(2244)
        attempts = [bad_gateway_error(), bad_gateway_error(), [compound]]

        def answer(*args, **kwargs):
            outcome = attempts.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(core, "get_compounds", side_effect=answer):
            result = core.get_compound_by("aspirin")

        self.assertEqual(result, [compound])
        self.assertEqual(self.waits, [1.0, 2.0], "Backoff should double between retries")
        self.assertIn("Bad Gateway", self.log_capture.getvalue())
        self.assertIn("Retrying in", self.log_capture.getvalue())

    def test_transient_error_outliving_the_retries_keeps_its_own_type(self):
        def answer(*args, **kwargs):
            raise bad_gateway_error()

        with patch.object(core, "get_compounds", side_effect=answer):
            with self.assertRaises(PubChemHTTPError) as raised:
                core.get_compound_by("aspirin")

        self.assertEqual(raised.exception.code, 502)
        self.assertEqual(len(self.waits), 2, "Three attempts should be spaced by two waits")

    def test_error_that_is_not_transient_is_not_retried(self):
        def answer(*args, **kwargs):
            raise PubChemHTTPError(400, "PUGREST.BadRequest", [])

        with patch.object(core, "get_compounds", side_effect=answer):
            with self.assertRaises(PubChemHTTPError) as raised:
                core.get_compound_by("aspirin")

        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(self.waits, [], "A bad request should be passed on without retrying")


class TestBatchResilience(PubChemTestCase):
    """One compound PubChem cannot answer for must not cost the whole batch."""

    @staticmethod
    def answer_except_for(failing_input):
        """Return a `get_compounds` stand-in that only fails for the given inputs."""

        def answer(identifier, *args, **kwargs):
            if identifier in failing_input:
                raise bad_gateway_error()
            return [FakeCompound(hash(identifier) % 10000)]

        return answer

    def test_failing_compound_keeps_the_rest_of_the_batch(self):
        cpd_list = ["aspirin", "unreachable", "ibuprofen"]

        with patch.object(core, "get_compounds", side_effect=self.answer_except_for({"unreachable"})):
            with redirect_stderr(StringIO()):  # the curation draws progress bars on stderr
                df = get_and_curate_multiple_compounds_result(cpd_list, input_type="name")

        self.assertEqual(len(df), len(cpd_list), "Every input should still be represented")
        self.assertEqual(set(df.columns), CURATED_COLUMNS)
        self.assertEqual(df["name"].tolist(), cpd_list)

        failed_row = df[df["name"] == "unreachable"].iloc[0]
        self.assertTrue(np.isnan(failed_row["pubchem_cid"]), "The failed compound should be empty")
        self.assertEqual(df["pubchem_cid"].notna().sum(), 2, "The other compounds should be filled")

        log_output = self.log_capture.getvalue()
        self.assertIn("Could not fetch 'unreachable' from PubChem", log_output)
        self.assertIn("Bad Gateway", log_output)
        self.assertIn("1 out of 3 compounds could not be fetched", log_output)

    def test_unreachable_pubchem_is_reported_as_such(self):
        cpd_list = ["aspirin", "ibuprofen"]

        with patch.object(core, "get_compounds", side_effect=self.answer_except_for(set(cpd_list))):
            with redirect_stderr(StringIO()):
                with self.assertRaises(PubChemHTTPError) as raised:
                    get_and_curate_multiple_compounds_result(cpd_list, input_type="name")

        self.assertEqual(raised.exception.code, 502)
        self.assertIn("Bad Gateway", str(raised.exception))
        self.assertIn("Could not fetch 'aspirin' from PubChem", self.log_capture.getvalue())

    def test_synonyms_can_be_left_out(self):
        cpd_list = ["aspirin"]

        with patch.object(core, "get_compounds", side_effect=self.answer_except_for(set())):
            with redirect_stderr(StringIO()):
                fetched = get_and_curate_multiple_compounds_result(cpd_list, input_type="name")
                skipped = get_and_curate_multiple_compounds_result(
                    cpd_list, input_type="name", fetch_synonyms=False
                )

        self.assertEqual(set(skipped.columns), CURATED_COLUMNS, "The column contract should hold")
        self.assertTrue(fetched["synonyms"].notna().all())
        self.assertTrue(skipped["synonyms"].isna().all())


if __name__ == "__main__":
    unittest.main()
