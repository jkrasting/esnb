import json
import logging
from pathlib import Path

import intake
import intake_esm
from esnb import sites
from esnb.core.mdtf import MDTFCaseSettings

from . import html, util

logger = logging.getLogger(__name__)


def infer_case_source(source):
    """
    Infers the type of a given case source and returns its mode.

    Parameters
    ----------
    source : str or int
        The source to be inferred. This can be a string representing a Dora ID,
        a URL, a local file path, or a project-level Dora ID, or an integer
        representing a Dora ID.

    Returns
    -------
    mode : str
        The inferred mode of the source. Possible values include:
        - 'dora_id': Dora ID (numeric or project-level)
        - 'dora_url': URL pointing to Dora
        - 'intake_url': URL suggesting an intake catalog
        - 'path': Local file path
        - 'pp_dir': Directory containing post-processing (raises NotImplementedError)
        - 'intake_path': Local JSON file assumed to be an intake catalog
        - 'mdtf_settings': Local file assumed to be an MDTF settings file

    Raises
    ------
    ValueError
        If the source type is unsupported.
    FileNotFoundError
        If the provided path does not exist.
    NotImplementedError
        If the supplied path is a directory (future support planned).
    """
    if isinstance(source, str):
        if source.isnumeric():
            logger.debug(f"Found source string with numeric Dora ID - {source}")
            mode = "dora_id"
        elif source.startswith("http") or source.startswith("https"):
            if "dora.gfdl" in source:
                logger.debug(f"Found source url pointing to Dora - {source}")
                mode = "dora_url"
            else:
                logger.debug(f"Found source url suggesting intake catalog - {source}")
                mode = "intake_url"
        elif "-" in source:
            if source.split("-")[1].isnumeric():
                logger.debug(
                    f"Found source string with project-level dora ID - {source}"
                )
                mode = "dora_id"
            else:
                mode = "path"
        else:
            mode = "path"
    elif isinstance(source, int):
        logger.debug(f"Found source integer suggesting dora ID - {source}")
        mode = "dora_id"
    else:
        raise ValueError(
            "Unsupported source type. Must be path or url to"
            + " intake_esm catalog, MDTF settings file, or DORA ID"
        )

    if mode == "path":
        logger.debug(f"Assuming source is a local file path - {source}")
        filepath = Path(source)
        if not filepath.exists():
            logger.error(f"Path {filepath} does not exist")
            raise FileNotFoundError(f"Path {filepath} does not exist")
        if filepath.is_dir():
            mode = "pp_dir"
            logger.debug(
                "Supplied path appears to be a directory, possibly containing post-processing"
            )
            err = "The supplied path is a directory. In the future, support will be added to generate a catalog."
            logger.error(err)
            raise NotImplementedError(err)
        else:
            try:
                with open(filepath, "r") as f:
                    json.load(f)
                logger.debug(
                    "Source appears to be a JSON file, assuming intake catalog"
                )
                mode = "intake_path"
            except json.JSONDecodeError:
                logger.debug("Source is not a JSON file, assuming MDTF settings file")
                mode = "mdtf_settings"

    return mode


def open_intake_catalog(source, mode):
    """
    Opens an intake catalog from a given source using the specified mode.

    Parameters
    ----------
    source : str
        The path or URL to the intake catalog to be opened.
    mode : str
        The mode specifying how to open the catalog. Must be either
        "intake_url" to fetch from a URL or "intake_path" to open from a
        local file.

    Returns
    -------
    catalog : intake.ESMDataStore
        The opened intake catalog object.

    Raises
    ------
    RuntimeError
        If an unrecognized mode is provided.

    Notes
    -----
    Requires the `intake` package and a properly configured logger.
    """
    if mode == "intake_url":
        logger.info(f"Fetching intake catalog from url: {source}")
        catalog = intake.open_esm_datastore(source)

    elif mode == "intake_path":
        logger.info(f"Opening intake catalog from file: {source}")
        catalog = intake.open_esm_datastore(source)

    else:
        err = f"Encountered unrecognized source mode: {mode}"
        loggger.error(err)  # noqa
        raise RuntimeError(err)

    return catalog


def open_intake_catalog_dora(source, mode):
    """
    Opens an intake ESM datastore catalog from a specified source and mode.

    Parameters
    ----------
    source : str
        The source identifier. If `mode` is "dora_url", this should be the full
        URL to the intake catalog. If `mode` is "dora_id", this should be the
        identifier used to construct the catalog URL.
    mode : str
        The mode specifying how to interpret `source`. Must be either "dora_url"
        to use `source` as a direct URL, or "dora_id" to construct the URL from
        a known pattern.

    Returns
    -------
    catalog : intake.ESMDataStore
        The opened intake ESM datastore catalog.

    Raises
    ------
    RuntimeError
        If an unrecognized `mode` is provided.

    Notes
    -----
    Logs the process of fetching the catalog and checks network availability to
    the Dora service.
    """
    if mode == "dora_url":
        url = source
    elif mode == "dora_id":
        url = f"https://{sites.gfdl.dora_hostname}/api/intake/{source}.json"
    else:
        err = f"Encountered unrecognized source mode: {mode}"
        loggger.error(err)  # noqa
        raise RuntimeError(err)

    logger.info(f"Fetching intake catalog from url: {url}")
    if not sites.gfdl.dora:
        logger.critical("Network route to dora is unavailble. Check connection.")
    catalog = intake.open_esm_datastore(url)

    return catalog


class CaseExperiment2(MDTFCaseSettings):
    """
    CaseExperiment2 is a class for managing and validating a single experiment case
    from various sources, such as MDTF settings files, intake catalogs, or DORA
    catalogs. It loads the case, sets up the catalog, and processes metadata such
    as the time range.


    Attributes
        The original source provided for the case.
    mode : str
        The inferred mode of the source (e.g., "mdtf_settings", "intake", "dora").
    catalog : object
        The loaded catalog object, which may be an intake ESM datastore or similar.
    name : str
        The name of the case associated with this instance.
    mdtf_settings : dict, optional
        The MDTF settings dictionary, present if the source is an MDTF settings file.


    - Only single-case MDTF settings files are supported; use `CaseGroup` for
      multiple cases.
    """

    def __init__(self, source, verbose=True):
        """
        Initialize a CaseExperiment2 instance by loading and validating the provided
        source, which may be an MDTF settings file, an intake catalog, or a DORA
        catalog. Sets up the catalog and case name, and processes the catalog's time
        range if applicable.

        Parameters
        ----------
        source : str or Path
            Path to the MDTF settings file, intake catalog, or DORA catalog.
        verbose : bool, optional
            If True, enables verbose logging output. Default is True.

        Raises
        ------
        ValueError
            If the MDTF settings file contains zero or multiple cases.
        RuntimeError
            If the source mode is unrecognized.

        Notes
        -----
        - For MDTF settings files, only single-case files are supported; use the
          `CaseGroup` class for multiple cases.
        - The catalog's `time_range` column is converted to a tuple of datetime
          objects if the catalog is an intake ESM datastore.
        """
        self.source = source
        self.mode = infer_case_source(self.source)

        # Read the MDTF settings case file
        if self.mode == "mdtf_settings":
            logger.info("Loading MDTF Settings File")
            self.load_mdtf_settings_file(source)
            if len(self.mdtf_settings["case_list"]) == 0:
                raise ValueError("No cases found in MDTF settings file")
            elif len(self.mdtf_settings["case_list"]) > 1:
                raise ValueError(
                    "Multiple cases found in MDTF settings file. "
                    + "Please initialize using the `CaseGroup` class."
                )
            self.name = list(self.mdtf_settings["case_list"].keys())[0]

            catalog_file = Path(self.catalog)
            logger.debug(
                f"Loading intake catalog from path specified in MDTF settings file: {str(catalog_file)}"
            )
            if catalog_file.exists():
                self.catalog = open_intake_catalog(str(catalog_file), "intake_path")
            else:
                logger.warning(
                    f"MDTF-specified intake catalog path does not exist: {str(catalog_file)}"
                )

        elif "intake" in self.mode or "dora" in self.mode:
            if "intake" in self.mode:
                self.catalog = open_intake_catalog(self.source, self.mode)
            elif "dora" in self.mode:
                self.catalog = open_intake_catalog_dora(self.source, self.mode)
            self.name = self.catalog.__dict__["esmcat"].__dict__["id"]

        else:
            err = f"Encountered unrecognized source mode: {self.mode}"
            loggger.error(err)  # noqa
            raise RuntimeError(err)

        # Convert catalog `time_range` to tuple of datetime objects
        if isinstance(self.catalog, intake_esm.core.esm_datastore):
            logger.debug(
                f"Converting time range in {self.name} catalog to datetime object"
            )
            self.catalog.df["time_range"] = self.catalog.df["time_range"].apply(
                util.process_time_string
            )

    def __str__(self):
        """
        Returns a string representation of the object.

        Returns
        -------
        str
            The name of the object as a string.
        """
        return str(self.name)

    def __repr__(self):
        """
        Return a string representation of the CaseExperiment2 object.

        Returns
        -------
        str
            A string in the format 'CaseExperiment2(<case_name>)', where
            <case_name> is the name of the case associated with this instance.
        """
        return f"{self.__class__.__name__}({self.name})"

    def _repr_html_(self):
        """
        Generate an HTML representation of the CaseExperiment2 object for
        display in Jupyter notebooks.

        Returns
        -------
        str
            An HTML string containing a summary table of the object's main
            attributes, including the source type, catalog, and (if present)
            the MDTF settings in a collapsible section.

        Notes
        -----
        This method is intended for use in interactive environments such as
        Jupyter notebooks, where the HTML output will be rendered for easier
        inspection of the object's state.
        """
        result = html.gen_html_sub()
        # Table Header
        result += f"<h3>{self.__class__.__name__}  --  {self.name}</h3>"
        result += "<table class='cool-class-table'>"

        # Iterate over attributes, handling the dictionary separately
        result += f"<tr><td><strong>Source Type</strong></td><td>{self.mode}</td></tr>"
        result += f"<tr><td><strong>catalog</strong></td><td>{str(self.catalog).replace('<', '').replace('>', '')}</td></tr>"

        if hasattr(self, "mdtf_settings"):
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>View MDTF Settings</summary>"
            result += "<div><table>"
            for d_key in sorted(self.mdtf_settings.keys()):
                d_value = self.mdtf_settings[d_key]
                result += f"<tr><td>{d_key}</td><td>{d_value}</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        result += "</table>"

        return result

    def __hash__(self):
        return hash((self.name, self.source))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
