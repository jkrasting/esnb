import copy
import datetime as dt
import logging
import time

import pandas as pd
from esnb.core.CaseExperiment2 import CaseExperiment2
from esnb.core.util import is_overlapping, process_time_string

logger = logging.getLogger("__name__")


def case_time_filter(case, date_range):
    """
    Filters the cases in the catalog based on overlap with a given date range.

    Parameters
    ----------
    case : object
        An object containing a catalog with a DataFrame `df` and an associated
        `esmcat` attribute. The DataFrame must have a "time_range" column.
    date_range : list or tuple of str
        A sequence of two date strings representing the start and end of the
        desired time range.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the rows whose "time_range" overlaps with
        the specified `date_range`.

    Raises
    ------
    AssertionError
        If `date_range` does not contain exactly two elements.

    Notes
    -----
    This function modifies `case.catalog.esmcat._df` in place to reflect the
    filtered DataFrame.
    """
    assert len(date_range) == 2
    trange = xr_date_range_to_datetime(date_range)
    df = case.catalog.df
    non_matching_times = []
    for index, row in df.iterrows():
        if not is_overlapping(trange, row["time_range"]):
            non_matching_times.append(index)
    df = df.drop(non_matching_times)
    case.catalog.esmcat._df = df
    return df


def check_schema_equivalence(esmcat1, esmcat2, keys=None):
    """
    Check if two ESM catalog objects have equivalent schema definitions for
    specified keys.

    Parameters
    ----------
    esmcat1 : object
        The first ESM catalog object, expected to have an `esmcat` attribute
        with a dictionary-like interface.
    esmcat2 : object
        The second ESM catalog object, expected to have an `esmcat` attribute
        with a dictionary-like interface.
    keys : list of str, optional
        List of keys to compare within the `esmcat` dictionary. If None,
        defaults to ["attributes", "aggregation_control"].

    Returns
    -------
    bool
        True if the specified keys in both catalog objects are equivalent,
        False otherwise.

    Notes
    -----
    This function compares the lower-level definitions of the provided ESM
    catalog objects by accessing their internal `esmcat` dictionaries.
    """

    # esmcat keys:  ['esmcat_version',
    #                'attributes',
    #                'assets',
    #                'aggregation_control',
    #                'id',
    #                'catalog_dict',
    #                'catalog_file',
    #                'description',
    #                'title',
    #                'last_updated']

    # Default set of keys to check for equivalence
    keys = ["attributes", "aggregation_control"] if keys is None else keys

    # Access each catalog's `esmcat` dictionary
    cat1 = esmcat1.__dict__["esmcat"].__dict__
    cat2 = esmcat2.__dict__["esmcat"].__dict__

    return all([cat1[k] == cat2[k] for k in keys])


def flatten_list(nested_list):
    """
    Recursively flattens a nested list into a single list of elements.

    Parameters
    ----------
    nested_list : list
        A list which may contain other lists as elements, at any level of nesting.

    Returns
    -------
    flat_list : list
        A flat list containing all the elements from the nested list, with all
        levels of nesting removed.

    Examples
    --------
    >>> flatten_list([1, [2, [3, 4], 5], 6])
    [1, 2, 3, 4, 5, 6]
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursive call
        else:
            flat_list.append(item)
    return flat_list


def initialize_cases_from_source(source):
    """
    Initializes case or experiment groups from a nested source list.

    Parameters
    ----------
    source : list
        A list containing case/experiment definitions. Each element can be
        either a single case/experiment or a list of cases/experiments. Only
        two levels of nesting are supported.

    Returns
    -------
    groups : list
        A list of initialized `CaseExperiment2` objects or lists of
        `CaseExperiment2` objects, corresponding to the structure of the
        input `source`.

    Raises
    ------
    ValueError
        If `source` is not a list.
    NotImplementedError
        If more than two levels of case aggregation are provided.

    Notes
    -----
    Each case/experiment is wrapped in a `CaseExperiment2` instance. If a
    sublist is encountered, each of its elements is also wrapped, and the
    sublist is appended to the result.
    """
    if not isinstance(source, list):
        err = "Sources provided to `initialize_cases_from_source` must be a list"
        logger.error(err)
        raise ValueError(err)

    groups = []
    for x in source:
        if not isinstance(x, list):
            logging.debug(f"Setting up individual case/experiment: {x}")
            groups.append(CaseExperiment2(x))
        else:
            subgroup = []
            for i in x:
                if isinstance(i, list):
                    err = "Only two levels of case aggregation are supported."
                    logging.error(err)
                    raise NotImplementedError(err)
                else:
                    logging.debug(f"Setting up individual case/experiment: {i}")
                    subgroup.append(CaseExperiment2(i))
            groups.append(subgroup)

    return groups


def merge_intake_catalogs(catalogs, id=None, description=None, title=None, **kwargs):
    """
    Merge multiple intake catalogs with equivalent schemas into a single catalog.

    Parameters
    ----------
    catalogs : list or object
        A list of intake catalog objects to merge, or a single catalog object.
    id : str, optional
        Identifier for the merged catalog. If None, the original id is retained.
    description : str, optional
        Description for the merged catalog. If None, the original description is
        retained.
    title : str, optional
        Title for the merged catalog. If None, the original title is retained.
    **kwargs
        Additional keyword arguments passed to `pandas.merge` when merging the
        underlying dataframes. If not provided, defaults to `how="outer"`.

    Returns
    -------
    result : object
        The merged intake catalog object with updated metadata and combined
        dataframe.

    Raises
    ------
    AssertionError
        If the provided catalogs do not have equivalent schemas.

    Notes
    -----
    All catalogs must have the same schema to be merged successfully. The
    function merges the underlying dataframes of the catalogs and updates the
    metadata as specified.
    """
    # catalogs is a list of catalogs
    catalogs = [catalogs] if not isinstance(catalogs, list) else catalogs
    if len(catalogs) == 1:
        result = catalogs[0]
    else:
        # test that catalogs are equivalent
        equivalence = [check_schema_equivalence(catalogs[0], x) for x in catalogs]
        assert all(equivalence), "All catalogs must have the same schema to merge"

        # obtain the underlying dataframes
        dfs = [x.df for x in catalogs]

        # merge subsequent catalogs onto the first
        merged_df = copy.deepcopy(dfs[0])

        for df in dfs[1:]:
            # Use pandas.merge() function and pass options, if provided
            kwargs = {"how": "outer"} if kwargs == {} else kwargs
            merged_df = pd.merge(merged_df, df, **kwargs)

        result = copy.deepcopy(catalogs[0])
        result = update_intake_dataframe(result, merged_df)
        result = reset_catalog_metadata(
            result, id=id, description=description, title=title
        )

    return result


def reset_catalog_metadata(cat, id=None, description=None, title=None):
    """
    Reset and update the metadata attributes of a catalog object.

    Parameters
    ----------
    cat : object
        The catalog object whose metadata will be reset and updated.
    id : str or None, optional
        The new identifier for the catalog. If None, an empty string is used.
    description : str or None, optional
        The new description for the catalog. If None, an empty string is used.
    title : str or None, optional
        The new title for the catalog. If None, an empty string is used.

    Returns
    -------
    object
        The catalog object with updated metadata attributes.

    Notes
    -----
    This function directly modifies the internal dictionaries of the catalog and
    its associated `esmcat` attribute. The `updated` and `last_updated` fields
    are set to the current time.
    """
    id = "" if id is None else str(id)
    description = "" if description is None else str(description)
    title = "" if title is None else str(title)
    cat.__dict__["_captured_init_args"] = None
    cat.__dict__["updated"] = time.time()
    cat.__dict__["esmcat"].__dict__["last_updated"] = dt.datetime.fromtimestamp(
        cat.__dict__["updated"], dt.UTC
    )
    cat.__dict__["esmcat"].__dict__["catalog_file"] = None
    cat.__dict__["esmcat"].__dict__["id"] = id
    cat.__dict__["esmcat"].__dict__["description"] = description
    cat.__dict__["esmcat"].__dict__["title"] = title
    return cat


def update_intake_dataframe(cat, df, reset_index=True):
    """
    Updates the internal dataframe of an intake catalog object.

    Parameters
    ----------
    cat : object
        The intake catalog object whose internal dataframe will be updated. It is
        expected to have an `esmcat` attribute with a `_df` attribute.
    df : pandas.DataFrame
        The new dataframe to assign to the catalog's internal dataframe.
    reset_index : bool, optional
        If True (default), resets the index of the dataframe before assignment.

    Returns
    -------
    object
        The updated intake catalog object with the new dataframe assigned.
    """
    if reset_index:
        df = df.reset_index()
    cat.esmcat._df = df
    return cat


def xr_date_range_to_datetime(date_range):
    """
    Converts a list of date strings into a processed datetime string.

    Each date in the input list is expected to be in the format 'YYYY-MM-DD'.
    The function zero-pads the year, month, and day components, concatenates
    them, joins the resulting strings with a hyphen, and then processes the
    final string using the `process_time_string` function.

    Parameters
    ----------
    date_range : list of str
        List of date strings, each in the format 'YYYY-MM-DD'.

    Returns
    -------
    str
        A processed datetime string obtained after formatting and joining the
        input dates, and applying `process_time_string`.

    Notes
    -----
    The function assumes that `process_time_string` is defined elsewhere in
    the codebase.
    """
    _date_range = []
    for x in date_range:
        x = x.split("-")
        x = str(x[0]).zfill(4) + str(x[1]).zfill(2) + str(x[2].zfill(2))
        _date_range.append(x)
    _date_range = str("-").join(_date_range)
    _date_range = process_time_string(_date_range)
    return _date_range
