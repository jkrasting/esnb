import copy
import logging
from pathlib import Path

import intake_esm

from esnb.core.mdtf import MDTFCaseSettings
from esnb.sites.gfdl import open_intake_catalog_dora

from . import html, util
from .util_case import infer_case_source
from .util_catalog import open_intake_catalog

logger = logging.getLogger(__name__)


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

    def __init__(self, source, name=None, verbose=True):
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
        self.name = name

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
            self.name = (
                list(self.mdtf_settings["case_list"].keys())[0]
                if self.name is None
                else self.name
            )

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
            self.name = (
                self.catalog.__dict__["esmcat"].__dict__["id"]
                if self.name is None
                else self.name
            )

        elif self.mode == "catalog_object":
            self.catalog = self.source
            self.source = "Intake ESM Catalog Object"
            self.name = (
                self.catalog.__dict__["esmcat"].__dict__["id"]
                if self.name is None
                else self.name
            )

        else:
            err = f"Encountered unrecognized source mode: {self.mode}"
            logger.error(err)
            raise RuntimeError(err)

        self.name = "Generic Case" if self.name is None else self.name

        # Convert catalog `time_range` to tuple of datetime objects
        if isinstance(self.catalog, intake_esm.core.esm_datastore):
            logger.debug(
                f"Converting time range in {self.name} catalog to datetime object"
            )
            self.catalog.df["time_range"] = self.catalog.df["time_range"].apply(
                util.process_time_string
            )

        # Try to eep a copy of the original catalog in case its needed later
        try:
            self._source_catalog = copy.deepcopy(self.catalog)
        except Exception as exc:
            logger.debug(str(exc))
            logger.debug("Unable to deep copy source catalog. Not an immediate issue.")

    def files(self, **kwargs):
        if len(kwargs) > 0:
            result = self.catalog.search(**kwargs).df["path"]
        else:
            result = self.catalog.df["path"]
        return sorted(list(result))

    def query(self, param):
        return sorted(list(set(self.catalog.df[param])))

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
