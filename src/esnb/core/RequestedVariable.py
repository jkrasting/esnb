from . import html
from .VirtualDataset import resolve_dataset_refs


class RequestedVariable:
    """
    Represents a variable requested for analysis, including metadata and search options.

    Parameters
    ----------
    varname : str
        Name of the variable as used in the analysis script.
    preferred_realm : str or list of str, optional
        Preferred realm(s) for the variable (e.g., 'atmos', 'ocean').
    path_variable : str, optional
        Name of the variable as it appears in the file path.
    scalar_coordinates : list or None, optional
        List of scalar coordinates associated with the variable.
    standard_name : str, optional
        Standardized name for the variable.
    source_varname : str, optional
        Name of the variable in the source dataset.
    units : str, optional
        Units of the variable.
    preferred_chunkfreq : list of str or str, optional
        Preferred chunking frequencies, e.g., ["5yr", "2yr", "1yr", "20yr", "unknown"].
    preferred_grid_label : list of str or str, optional
        Preferred grid label, e.g., ["gn", "gr", "unknown"].
    frequency : str, optional
        Frequency of the data (e.g., 'mon' for monthly). Default is "mon".
    ppkind : str, optional
        Kind of post-processing (e.g., 'ts' for time series). Default is "ts".
    dimensions : list or None, optional
        List of dimensions for the variable.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    varname : str
        Name of the variable.
    preferred_realm : list of str or None
        Preferred realm(s) for the variable.
    path_variable : str or None
        Name of the variable in the file path.
    scalar_coordinates : list or None
        Scalar coordinates for the variable.
    standard_name : str or None
        Standardized name for the variable.
    source_varname : str or None
        Name of the variable in the source dataset.
    units : str or None
        Units of the variable.
    preferred_chunkfreq : list of str or None
        Preferred chunking frequencies.
    preferred_grid_label : list of str or None
        Preferred grid.
    frequency : str
        Frequency of the data.
    ppkind : str
        Kind of post-processing.
    dimensions : list or None
        Dimensions of the variable.
    catalog : None
        Placeholder for catalog information.

    Methods
    -------
    to_dict()
        Returns a dictionary representation of the variable's metadata.
    search_options
        Returns a dictionary of search options for querying datasets.
    _repr_html_()
        Returns an HTML representation for display in Jupyter notebooks.
    __repr__()
        Returns a string representation for debugging.
    __str__()
        Returns the variable name as a string.
    """

    def __init__(
        self,
        varname,
        preferred_realm=None,
        path_variable=None,
        scalar_coordinates=None,
        standard_name=None,
        source_varname=None,
        units=None,
        preferred_chunkfreq=["5yr", "2yr", "1yr", "20yr", "unknown"],
        preferred_grid_label=["gn", "gr", "unknown"],
        frequency="mon",
        ppkind="ts",
        dimensions=None,
        **kwargs,
    ):
        """
        Initialize a RequestedVariable instance.

        Parameters
        ----------
        varname : str
            Name of the variable used in the analysis script.
        preferred_realm : str or list of str, optional
            Preferred realm(s) for the variable (e.g., 'atmos', 'ocean').
        path_variable : str, optional
            Path variable name, if different from `varname`.
        scalar_coordinates : list or None, optional
            List of scalar coordinates associated with the variable.
        standard_name : str, optional
            Standardized name for the variable.
        source_varname : str, optional
            Source variable name, if different from `varname`.
        units : str, optional
            Units of the variable.
        preferred_chunkfreq : list of str or str, optional
            Preferred chunking frequencies, by default ["5yr", "2yr", "1yr", "20yr", "unknown"].
        frequency : str, optional
            Frequency of the variable (e.g., 'mon' for monthly), by default "mon".
        ppkind : str, optional
            Post-processing kind, by default "ts".
        dimensions : list or None, optional
            List of dimensions for the variable.
        **kwargs
            Additional keyword arguments.

        Attributes
        ----------
        path_variable : str
            Path variable name.
        varname : str
            Variable name.
        preferred_realm : list of str or None
            Preferred realm(s) as a list.
        scalar_coordinates : list or None
            Scalar coordinates.
        standard_name : str or None
            Standardized name.
        source_varname : str or None
            Source variable name.
        units : str or None
            Units of the variable.
        preferred_chunkfreq : list of str or None
            Preferred chunking frequencies as a list.
        preferred_grid_label : list of str or str, optional
            Preferred grid label, e.g., ["gn", "gr", "unknown"].
        frequency : str
            Frequency of the variable.
        ppkind : str
            Post-processing kind.
        dimensions : list or None
            Dimensions of the variable.
        catalog : None
            Placeholder for catalog information.
        """
        # Variable name used in the analysis script
        self.path_variable = path_variable
        self.varname = varname
        self.preferred_realm = preferred_realm
        self.scalar_coordinates = scalar_coordinates
        self.standard_name = standard_name
        self.source_varname = source_varname
        self.units = units
        self.preferred_chunkfreq = preferred_chunkfreq
        self.preferred_grid_label = preferred_grid_label
        self.frequency = frequency
        self.ppkind = "ts"
        self.dimensions = dimensions
        self.catalog = None

        if self.preferred_realm is not None:
            self.preferred_realm = (
                [self.preferred_realm]
                if not isinstance(self.preferred_realm, list)
                else self.preferred_realm
            )

        if self.preferred_chunkfreq is not None:
            self.preferred_chunkfreq = (
                [self.preferred_chunkfreq]
                if not isinstance(self.preferred_chunkfreq, list)
                else self.preferred_chunkfreq
            )

        if self.preferred_grid_label is not None:
            self.preferred_grid_label = (
                [self.preferred_grid_label]
                if not isinstance(self.preferred_grid_label, list)
                else self.preferred_grid_label
            )

    def to_dict(self):
        """
        Convert the RequestedVariable instance to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the RequestedVariable instance,
            containing the following keys: 'varname', 'preferred_realm',
            'standard_name', 'source_varname', 'units', 'preferred_chunkfreq',
            'preferred_grid_label', 'frequency', 'ppkind', and 'dimensions'.
        """
        return {
            "varname": self.varname,
            "preferred_realm": self.preferred_realm,
            "standard_name": self.standard_name,
            "source_varname": self.source_varname,
            "units": self.source_varname,
            "preferred_chunkfreq": self.preferred_chunkfreq,
            "preferred_grid_label": self.preferred_grid_label,
            "frequency": self.frequency,
            "ppkind": self.ppkind,
            "dimensions": self.dimensions,
        }

    @property
    def datasets(self):
        return resolve_dataset_refs(self._datasets)

    @property
    def search_options(self):
        """
        Constructs a dictionary of search options for the requested variable.

        The dictionary includes the variable name (from `source_varname` if available,
        otherwise `varname`), and optionally includes frequency, kind, preferred realm,
        and preferred chunk frequency if these attributes are not None.

        Returns
        -------
        dict
            A dictionary containing search options for the requested variable.
            Keys may include 'var', 'freq', 'kind', 'preferred_realm', and
            'preferred_chunkfreq', depending on which attributes are set.
        """
        result = {}
        result["var"] = (
            self.source_varname if self.source_varname is not None else self.varname
        )
        if self.frequency is not None:
            result["freq"] = self.frequency
        if self.ppkind is not None:
            result["kind"] = self.ppkind
        if self.preferred_realm is not None:
            result["preferred_realm"] = self.preferred_realm
        if self.preferred_chunkfreq is not None:
            result["preferred_chunkfreq"] = self.preferred_chunkfreq
        return result

    def _repr_html_(self, title=True):
        """
        Generate an HTML representation of the object for Jupyter display.

        Returns
        -------
        str
            An HTML string containing a table representation of the object's
            attributes. Non-None attributes are displayed in the main table,
            while attributes with None values are grouped under an expandable
            "Inactive Settings" section.

        Notes
        -----
        This method is intended for use in Jupyter notebooks, allowing for
        rich HTML display of the object's state. The table includes the class
        name and variable name as a header.
        """
        # Table Header
        if title:
            result = html.gen_html_sub()
            result += f"<h3>{self.__class__.__name__}  --  {self.varname}</h3>"
            result += "<table class='cool-class-table'>"
        else:
            result = ""

        inactive = {}
        display_keys = [
            "frequency",
            "ppkind",
            "preferred_chunkfreq",
            "preferred_grid_label",
            "preferred_realm",
            "varname",
        ]
        for k in sorted(self.__dict__.keys()):
            val = self.__dict__[k]
            if val is not None:
                if k in display_keys:
                    result += f"<tr><td><strong>{k}</strong></td><td>{val}</td></tr>"
            else:
                inactive[k] = val

        if len(inactive) > 0:
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>Inactive Settings</summary>"
            result += "<div><table>"
            for d_key in sorted(inactive.keys()):
                d_value = inactive[d_key]
                result += f"<tr><td>{d_key}</td><td>{d_value}</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        if hasattr(self, "datasets"):
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>Xarray Datasets</summary>"
            result += "<div><table>"
            for group in self.datasets.keys():
                result += "<tr><td colspan='2'>"
                result += "<details>"
                result += f"<summary>({group.name})</summary>"
                result += "<div><table>"
                result += f"<tr><td>{self.datasets[group]._repr_html_()}</td></tr>"
                result += "</table></div>"
                result += "</details>"
                result += "</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        if title:
            result += "</table>"

        return result

    def __repr__(self):
        """
        Return a string representation of the RequestedVariable instance.

        Returns
        -------
        str
            A string describing the RequestedVariable, including its variable
            name.
        """
        reprstr = f"RequestedVariable {self.varname}"
        return reprstr

    def __str__(self):
        """
        Return the variable name as a string.

        Returns
        -------
        str
            The variable name (`varname`) of the RequestedVariable instance.
        """
        return str(self.varname)

    def __hash__(self):
        hashables = []
        acceptable_keys = ["varname", "source_varname", "frequency"]
        for k in sorted(list(self.__dict__.keys())):
            if k in acceptable_keys:
                v = self.__dict__[k]
                v = tuple(v) if isinstance(v, list) else v
                hashables.append(v)
        return hash(tuple(hashables))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
