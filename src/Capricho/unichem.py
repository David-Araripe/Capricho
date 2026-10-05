
from abc import ABC
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry

from .logger import logger


class BaseUniChem(ABC):
    """Base class for UniChem API access"""

    def __init__(
        self,
        pooling_interval=3,
        total_retries=5,
        backoff_factor=0.25,
        base_url="https://www.ebi.ac.uk/unichem/api/v1/",
        endpoint="compounds",
        cpd_id="inchikey",
    ) -> None:
        self.base_url = base_url
        self._POOLING_INTERVAL = pooling_interval
        self._check_params(endpoint, cpd_id)
        self.retries = self._setup_retries(total_retries, backoff_factor)
        self.session = requests.Session()
        self._setup_session()
        self.sources = self.load_sources(urljoin(base_url, "sources"))

    def load_sources(self, url) -> dict:
        res = self.session.get(url, data="")
        res.raise_for_status()
        return res.json()

    def _check_params(self, endpoint, cpd_id) -> None:
        if endpoint not in self.available_endpoints():
            raise ValueError(f"Endpoint {endpoint} not in available sources: {self.available_endpoints()}")
        if cpd_id not in self.available_ids():
            raise ValueError(
                f"Compound identifier {cpd_id} not in available identifiers: {self.available_ids()}"
            )

    def available_endpoints(self) -> list:
        """Return a list of available sources. For the full list: https://www.ebi.ac.uk/unichem/api/docs#/.
        Note: legacy endpoints are not included in this list."""
        return ["compounds", "connectivity", "images", "sources"]

    def available_ids(self) -> list:
        """Available compound identifiers for the given source"""
        return ["uci", "inchi", "inchikey", "sourceID"]

    @property
    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _setup_session(self) -> None:
        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))

    def _setup_retries(self, total_retries, backoff_factor) -> None:
        return Retry(
            total=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )

    def _setup_url(self, base_url, endpoint) -> str:
        return urljoin(base_url, endpoint)


class UniChem(BaseUniChem):
    def __init__(
        self,
        pooling_interval=3,
        total_retries=5,
        backoff_factor=0.25,
        base_url="https://www.ebi.ac.uk/unichem/api/v1/",
        endpoint="compounds",
        cpd_id="inchikey",
    ) -> None:
        super().__init__(pooling_interval, total_retries, backoff_factor, base_url, endpoint, cpd_id)
        self.last_request = None  # for debugging purposes

    def get_compound(self, compound, id_type="inchikey", source_id=None) -> dict:  # TODO
        """Uses the UniChem API to retrieve compounds based on a given compound's
        connectivities. Input can be either any of `self.available_ids()`.

        Args:
            compound: Compound representation to search. Must be the kind specified in
                the type parameter. Ignored when type is sourceID.
            id_type: The kind of compound representation for the molecule to search.
            sourceID: Unique identifier from the source database. It will only be
                consider when the type of search is sourceID.

        Raises:
            ValueError: If the response is not ok.

        Returns:
            dict: A dictionary with the response from the API.
        """
        # TODO; this is not done yet. Need to parse the results...
        url = self._setup_url(self.base_url, "compounds")
        params = {"compound": compound, "type": id_type}
        if source_id is not None:
            params.update({"sourceID": source_id})
        res = self.session.post(url, json=params, headers=self._headers)
        res.raise_for_status()
        if res.ok:
            return res.json()
        else:
            raise ValueError(f"Error: {res.status_code}:\n{res.text}")

    def get_connectivity(
        self,
        compound,
        id_type="inchikey",
        source_id: int | None = None,
        search_components: bool = True,
        save_searched: bool = False,
    ):
        """Uses the UniChem API to retrieve compounds based on a given compound's
        connectivities. Input can be either any of `self.available_ids()`.

        Args:
            compound: Compound representation to search. Must be the kind specified in
                the type parameter. Ignored when type is sourceID.
            id_type: The kind of compound representation for the molecule to search.
            sourceID: Unique identifier from the source database. It will only be
                consider when the type of search is sourceID.
            searchComponents: Indicates wether to use the individual components of the given mixture
                (if it applies) and match connectivity on the mixtures and its components (true) or to
                match connectivity for the whole mixture.
            save_searched: If True, will save the information on the last request to the API. Values
                [`searchedCompound`, `totalCompounds`, `totalSources`] will be kept in the
                self.last_request attribute.


        Raises:
            ValueError: If the response is not ok.

        Returns:
            dict: A dictionary with the response from the API.
        """
        url = self._setup_url(self.base_url, "connectivity")
        params = {
            "compound": compound,
            "type": id_type,
            "searchComponents": search_components,
        }
        if source_id is not None:
            params.update({"sourceID": source_id})
            logger.info(
                "Defining a source ID might not be supported yet. "
                "To verify, test the API at: https://www.ebi.ac.uk/unichem/api/docs#"
            )
        res = self.session.post(url, json=params, headers=self._headers)
        res.raise_for_status()
        parsed = []
        if res.ok:
            response = res.json()
            if "sources" not in response.keys():
                logger.warning(f"No compounds found for {compound}")
                return
            if len(response["sources"]) == 1:
                logger.warning(f"No compounds found for {compound}")
                return
            searched_cpd = response["searchedCompound"]
            total_cpds = response["totalCompounds"]
            total_srcs = response["totalSources"]
            if save_searched:
                self.last_request = {
                    "searchedCompound": searched_cpd,
                    "totalCompounds": total_cpds,
                    "totalSources": total_srcs,
                }
            logger.info(
                f"Found {total_cpds} compounds in {total_srcs} sources. "
                "Filtering for source ID: {source_id} as source."
            )
            for r in response["sources"]:
                comparison = r.pop("comparison")
                comparison = {f"comparison_{k}": v for k, v in comparison.items()}
                parsed.append({**r, **comparison})
        else:
            raise ValueError(f"Error: {res.status_code}:\n{res.text}")
        return parsed
