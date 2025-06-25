import logging
from pathlib import Path
import warnings

import esnb
from esnb.core.util2 import case_time_filter, flatten_list, initialize_cases_from_source

from . import html
from .util2 import merge_intake_catalogs

logger = logging.getLogger(__name__)


def infer_casegroup_name(group):
    names = [x.name for x in group.cases]
    return str(" + ").join(names)


def test_infer_casegroup_name():
    group1 = esnb.CaseGroup2(esnb.datasources.test_catalog_esm4_ctrl)
    group2 = esnb.CaseGroup2(
        [
            esnb.datasources.test_catalog_esm4_hist,
            esnb.datasources.test_catalog_esm4_futr,
        ]
    )
    assert infer_casegroup_name(group2) == "ESM4_historical + ESM4_SSP5-8.5"
    assert infer_casegroup_name(group1) == "ESM4_control"


def shorten_string(source, convert_path=False):
    """
    Shortens a string or file path for display purposes.

    If `convert_path` is True, treats the input as a file path, extracts the file
    name (without extension), and shortens it if necessary. Otherwise, shortens
    the input string directly.

    Parameters
    ----------
    source : str or Path-like
        The string or file path to be shortened.
    convert_path : bool, optional
        If True, treats `source` as a file path and processes the file name.
        Default is False.

    Returns
    -------
    str
        The shortened string or file name, with the original file extension
        preserved if `convert_path` is True.

    Examples
    --------
    >>> shorten_string("averylongfilenameexample.txt", convert_path=True)
    'averylongf...lename.txt'
    >>> shorten_string("averylongstringwithoutpath", convert_path=False)
    'averylongs...houtpath'
    """
    source = str(source)
    if convert_path:
        x = Path(source)
        fname = x.name
        suffix = x.suffix
        fname = fname.replace(suffix, "")
    else:
        fname = source
        suffix = ""

    if len(fname) > 23:
        fname = f"{fname[0:10]}...{fname[-10:]}"

    return f"{fname}{suffix}"


def test_shorten_string():
    source = "/Users/krasting/notebook-template/tests/test_data/ESM4_hist_this_is_something_new.json"
    assert shorten_string(source) == "/Users/kra...g_new.json"

    source = "/Users/krasting/notebook-template/tests/test_data/ESM4_hist_this_is_something_new.json"
    assert shorten_string(source, convert_path=True) == "ESM4_hist_...ething_new.json"

    source = "odiv-514"
    assert shorten_string(source) == "odiv-514"


def filter_catalog(catalog, variable):
    """
    Filter a catalog of variables based on specified criteria and preferences.

    This function searches the provided catalog for entries matching the variable's
    name, cell methods, and frequency. If multiple realms or chunk frequencies are
    found, it further filters the catalog using the variable's preferred realms and
    chunk frequencies.

    Parameters
    ----------
    catalog : object
        The catalog object to be searched and filtered. Must support the `search`
        method and have a `df` attribute (DataFrame-like) and `nunique` method.
    variable : object
        An object containing variable metadata. Must have the attributes:
        `varname`, `ppkind`, `frequency`, `preferred_realm`, and
        `preferred_chunkfreq`.

    Returns
    -------
    _cat : object
        The filtered catalog object, narrowed down according to the variable's
        preferences for realm and chunk frequency.
    """
    _cat = catalog.search(
        variable_id=variable.varname,
        cell_methods=variable.ppkind,
        frequency=variable.frequency,
    )

    # see if one realm exists:
    nrealm = int(_cat.nunique()["realm"])
    if nrealm > 1:
        realms = list(set(list(_cat.df["realm"])))
        preferred_realms = variable.preferred_realm
        if (
            "ocean_month" in preferred_realms
            and "ocean_monthly" not in preferred_realms
        ):
            preferred_realms.append("ocean_monthly")
        elif (
            "ocean_monthly" in preferred_realms
            and "ocean_month" not in preferred_realms
        ):
            preferred_realms.append("ocean_month")
        else:
            preferred_realms = preferred_realms
        preferred_realms = [x for x in preferred_realms if x in realms]
        if len(preferred_realms) >= 1:
            realm = preferred_realms[0]
        else:
            realm = ""
        _cat = _cat.search(realm=realm)

    # see if one time frequency exists:
    nchunk_freq = int(_cat.nunique()["chunk_freq"])
    if nchunk_freq > 1:
        chunk_freqs = list(set(list(_cat.df["chunk_freq"])))
        preferred_chunkfreqs = variable.preferred_chunkfreq
        preferred_chunkfreqs = [x for x in preferred_chunkfreqs if x in chunk_freqs]
        if len(preferred_chunkfreqs) >= 1:
            chunk_freq = preferred_chunkfreqs[0]
        else:
            chunk_freq = ""
        _cat = _cat.search(chunk_freqs=chunk_freq)

    return _cat


class CaseGroup2:
    """
    CaseGroup2(source, name=None, description=None, concat_dim=None, date_range=None, **kwargs)

    A group of case objects with shared metadata and catalog management.

    This class initializes and manages a group of case objects, allowing for
    metadata assignment, catalog resolution, and filtering by date range. It
    provides methods to access file paths, resolve catalogs based on variables,
    and display object summaries in both string and HTML formats.


    Attributes
    source : object or list
        The original source(s) for the case group.
    name : str
        The name of the case group.
    description : str
        Description of the case group.
    date_range : tuple or list
        The date range used to filter cases.
    concat_dim : str
        The dimension along which to concatenate cases.
    is_resolved : bool
        Indicates whether the group's catalogs have been resolved.
    is_loaded : bool
        Indicates whether the group's data has been loaded.
    cases : list
        The list of case objects managed by the group.

    - The class assumes the existence of helper functions such as
      `initialize_cases_from_source`, `flatten_list`, `case_time_filter`,
      `filter_catalog`, `merge_intake_catalogs`, and `shorten_string`.
    - Designed for use in data catalog and case management workflows.
    """

    def __init__(
        self,
        source,
        concat_dim=None,
        name=None,
        date_range=None,
        description=None,
        **kwargs,
    ):
        """
        Initialize a CaseGroup2 object.

        Parameters
        ----------
        source : object or list of objects
            The source(s) from which to initialize case objects. Can be a single
            source or a list of sources.
        name : str, optional
            The name of the case group. Defaults to None.
        description : str, optional
            A description of the case group. Defaults to None.
        concat_dim : str, optional
            The dimension along which to concatenate cases, if applicable.
            Defaults to None.
        date_range : tuple or list, optional
            The date range to filter cases. Should be a tuple or list specifying
            the start and end dates. Defaults to None.

        Notes
        -----
        Initializes metadata and loads cases from the provided source(s). If a
        date range is specified, filters the cases accordingly.
        """

        if "verbose" in kwargs.keys():
            warnings.warn(
                "`verbose` is no longer supported, remove this keyword argument. Future versions will fail if it is used.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.source = source
        self.name = name
        self.description = description
        self.date_range = date_range
        self.concat_dim = concat_dim
        self.is_resolved = False
        self.is_loaded = False

        # metadata
        self.description = "" if self.description is None else str(self.description)

        logger.info(f"Initializing CaseGroup object(s) from source: {source}")
        source = [source] if not isinstance(source, list) else source
        self.cases = initialize_cases_from_source(source)

        if date_range is not None:
            logger.info(
                f"Filtering individual case catalogs by date range: {date_range}"
            )
            _ = [case_time_filter(x, date_range) for x in flatten_list(self.cases)]

        self.name = infer_casegroup_name(self) if self.name is None else self.name

    @property
    def files(self):
        """
        Returns a sorted list of file paths from all cases in the group.

        This method flattens the list of cases, extracts the 'path' column from each
        case's catalog DataFrame, and returns a sorted list of all file paths.

        Returns
        -------
        list of str
            Sorted list of file paths from all cases in the group.
        """
        caselist = [x for x in flatten_list(self.cases)]
        return sorted(flatten_list([list(x.catalog.df["path"]) for x in caselist]))

    def resolve(self, varlist):
        """
        Resolve the catalogs for each case in the group based on the provided variables.

        This method flattens the list of cases, extracts their catalogs, and applies
        the `filter_catalog` function to each catalog for every variable in `varlist`.
        The filtered catalogs are then merged using `merge_intake_catalogs`, and the
        resulting catalogs are assigned back to each case. Marks the group as resolved.

        Parameters
        ----------
        varlist : list
            A list of variables to filter the catalogs with.

        Returns
        -------
        None

        Notes
        -----
        - Assumes `flatten_list`, `filter_catalog`, and `merge_intake_catalogs` are
          defined elsewhere.
        - Updates the `catalog` attribute of each case in the group.
        - Sets `self.is_resolved` to True upon completion.
        """
        caselist = flatten_list(self.cases)
        catalogs = [x.catalog for x in caselist]
        results = []
        for var in varlist:
            results.append([filter_catalog(x, var) for x in catalogs])
        zipped = list(zip(*results))
        merged = [merge_intake_catalogs(list(x)) for x in zipped]
        for n, case in enumerate(caselist):
            case.catalog = merged[n]
        self.is_resolved = True

    def __str__(self):
        """
        Returns the string representation of the object.

        Returns
        -------
        str
            The name of the object.
        """
        return self.name

    def __repr__(self):
        """
        Return a string representation of the CaseGroup object.

        The representation includes the group's name, a shortened description,
        and the status of its resolution and loading.

        Returns
        -------
        str
            A string summarizing the CaseGroup's name, description, and status.
        """
        name = self.name
        description = shorten_string(self.description)
        res = ""
        res = (
            res
            + f"CaseGroup {name} <{description}>  resolved={self.is_resolved}  loaded={self.is_loaded}"
        )
        return res

    def _repr_html_(self):
        """
        Generate an HTML representation of the object for Jupyter display.

        This method constructs an HTML summary of the object's key attributes,
        including its name, description, source, date range, concatenation
        dimension, resolution and loading status, and cases. Boolean attributes
        are color-coded for clarity.

        Returns
        -------
        str
            An HTML string representing the object, suitable for display in
            Jupyter notebooks.
        """
        result = html.gen_html_sub()

        def color_logical(var):
            if var is True:
                result = f"<span style='color: green;'>{var}</span>"
            else:
                result = f"<span style='color: red;'>{var}</span>"
            return result

        result += f"<h3>{self.__class__.__name__}  --  {self.name}</h3><i>{self.description}</i>"
        result += "<table class='cool-class-table'>"
        result += f"<tr><td><strong>source</strong></td><td>{self.source}</td></tr>"
        result += (
            f"<tr><td><strong>date_range</strong></td><td>{self.date_range}</td></tr>"
        )
        result += (
            f"<tr><td><strong>concat_dim</strong></td><td>{self.concat_dim}</td></tr>"
        )
        result += f"<tr><td><strong>is_resolved</strong></td><td>{color_logical(self.is_resolved)}</td></tr>"
        result += f"<tr><td><strong>is_loaded</strong></td><td>{color_logical(self.is_loaded)}</td></tr>"
        result += f"<tr><td><strong>cases</strong></td><td>{self.cases}</td></tr>"
        result += "</table>"

        return result

    def __hash__(self):
        hashables = []
        acceptable_keys = ["cases", "name", "source"]
        for k in sorted(list(self.__dict__.keys())):
            if k in acceptable_keys:
                v = self.__dict__[k]
                v = tuple(v) if isinstance(v, list) else v
                hashables.append(v)
        return hash(tuple(hashables))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
