import copy
import datetime as dt
import logging
import time

import intake
import pandas as pd

import esnb

logger = logging.getLogger(__name__)

__all__ = [
    "check_schema_equivalence",
    "convert_pangeo_catalog_to_catalogbuilder",
    "merge_intake_catalogs",
    "open_intake_catalog",
    "reset_catalog_metadata",
    "update_intake_dataframe",
]

pangeo_column_mapping = {
    "activity_id": "activity_id",
    "institution_id": "institution_id",
    "source_id": "source_id",
    "experiment_id": "experiment_id",
    "table_id": "realm",
    "member_id": "member_id",
    "grid_label": "grid_label",
    "variable_id": "variable_id",
    "zstore": "path",
}

pangeo_freq_mapping = {
    "3hr": "3hr",
    "6hrLev": "6hr",
    "6hrPlev": "6hr",
    "6hrPlevPt": "6hr",
    "AERday": "day",
    "AERhr": "1hr",
    "AERmon": "mon",
    "AERmonZ": "mon",
    "Aclim": "clim",
    "Amon": "mon",
    "CF3hr": "3hr",
    "CFday": "day",
    "CFmon": "mon",
    "E1hrClimMon": "climmon",
    "E3hr": "3hr",
    "Eclim": "clim",
    "Eday": "day",
    "EdayZ": "day",
    "Efx": "fx",
    "Emon": "mon",
    "EmonZ": "mon",
    "Eyr": "yearly",
    "IfxGre": "fx",
    "ImonGre": "mon",
    "LImon": "mon",
    "Lmon": "mon",
    "Oclim": "clim",
    "Oday": "day",
    "Odec": "dec",
    "Ofx": "fx",
    "Omon": "mon",
    "Oyr": "yearly",
    "SIclim": "clim",
    "SIday": "day",
    "SImon": "mon",
    "day": "day",
    "fx": "fx",
}


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


def convert_pangeo_catalog_to_catalogbuilder(
    cat=None, column_mapping=pangeo_column_mapping, freq_mapping=pangeo_freq_mapping
):
    if cat is None:
        cat = intake.open_esm_datastore(esnb.datasources.cmip6_pangeo)

    df = copy.deepcopy(cat.df)
    df = df[column_mapping.keys()].rename(columns=column_mapping)
    df.insert(4, "frequency", "unknown")
    df.insert(6, "table_id", "unknown")
    df.insert(10, "time_range", "00010101-99991231")
    df.insert(11, "chunk_freq", "unknown")
    df.insert(12, "platform", "unknown")
    df.insert(13, "dimensions", "unknown")
    df.insert(14, "cell_methods", "ts")
    df.insert(15, "standard_name", "unknown")

    # infer frequency from realm
    df["frequency"] = df["realm"].map(freq_mapping)

    # TODO: infer time range from filename (default range for now)

    # load blank catalog as a template and insert the modified dataframe
    cat = intake.open_esm_datastore(esnb.datasources.blank_catalog)
    cat = update_intake_dataframe(cat, df)
    name = "pangeo-cmip6"
    cat = reset_catalog_metadata(cat, id=name, description=name, title=name)

    return cat


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
        df = df.reset_index(drop=True)
    cat.esmcat._df = df
    return cat
