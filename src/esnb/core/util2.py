import datetime
import io
import json
import logging
import random
import os
import re
import string
import tempfile
from pathlib import Path

import xarray as xr

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


def clean_string(input_string):
    res = re.sub(r"[^a-zA-Z0-9\s]", "", input_string)
    res = res.replace(" ", "_")
    res = re.sub(r"_+", "_", res)
    return res


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


def generate_tempdir_path(name=None):
    name = "" if name is None else clean_string(name) + "_"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rand_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    custom_name = f"{name}{date_str}_{rand_str}"
    base_temp_dir = tempfile.gettempdir()
    full_path = Path(base_temp_dir) / Path(custom_name)
    return full_path


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def gemini(figs, prompt=None, model_name="gemini-2.5-pro", jupyter=True):
    import google.generativeai as genai
    import PIL.Image

    if "GEMINI_API_KEY" not in os.environ.keys():
        raise RuntimeError("'GEMINI_API_KEY' must be set to use this feature.")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
    image_data_list = []
    figs = [figs] if not isinstance(figs, list) else figs
    for fig in figs:
        image_data_list.append(PIL.Image.open(io.BytesIO(fig_to_bytes(fig))))

    prompt_parts = []
    for i, img_pil in enumerate(image_data_list):
        prompt_parts.append(f"Here is Figure {i + 1}:")
        prompt_parts.append(img_pil)
        if i < len(image_data_list) - 1:
            prompt_parts.append("\n")

    if prompt is None:
        prompt = (
            "\n\nBased on these figures, please provide a comprehensive "
            + "analysis of the trends, outliers, and any notable observations. "
            + "Compare and contrast the data presented in each figure. "
            + "What insights can be drawn from this visualization?"
        )

    prompt_parts.append(prompt)

    response = model.generate_content(prompt_parts)

    result = f"Prompt: {prompt} \n\n" + response.text

    if jupyter:
        from IPython.display import display, Markdown

        display(Markdown(result))
    else:
        return result


def get_nesting_depth(lst):
    if not isinstance(lst, list):
        return 0
    elif not lst:
        return 1
    else:
        return 1 + max(get_nesting_depth(item) for item in lst)


def infer_source_data_file_types(flist):
    flist = [flist] if not isinstance(flist, list) else flist
    flist = [Path(x) for x in flist]
    prefixes = [x.as_posix().split(":")[0] + ":" for x in flist if ":" in str(x)]
    prefixes = list(set(prefixes))

    if len(prefixes) > 1:
        raise ValueError(
            "Multiple source data file types detected. Unsure how to proceed."
        )
    elif len(prefixes) == 1:
        if prefixes[0] == "gs:":
            file_type = "google_cloud"
        else:
            raise ValueError(f"Unrecognized storage type: {prefixes[0]}")
    elif len(prefixes) == 0:
        file_type = "unix_file"
    else:
        raise RuntimeError("Unable to determine source file type")

    return file_type


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


def read_json(name):
    """
    Reads a JSON file, removes lines containing '//' comments, and returns the
    parsed JSON object.

    Parameters
    ----------
    name : str
        The path to the JSON file to be read.

    Returns
    -------
    dict or list
        The parsed JSON object from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON after comment removal.

    Notes
    -----
    Lines containing '//' anywhere are excluded before parsing as JSON.
    """
    with open(name, "r") as f:
        lines = [line.strip() for line in f]
    lines = [x for x in lines if "//" not in x]
    json_str = "".join(lines)
    return json.loads(json_str)


def reset_encoding(xobj, attrs=None):
    """Function to reset encoding attributes on an xarray object

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    attrs : list, optional
        Attributes to reset, by default None

    Returns
    -------
    xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Xarray object without encoding attributes
    """

    attrs = ["chunks", "preferred_chunks"] if attrs is None else attrs

    if isinstance(xobj, xr.DataArray):
        for attr in attrs:
            xobj.encoding.pop(attr, None)

    elif isinstance(xobj, xr.Dataset):
        for attr in attrs:
            xobj.encoding.pop(attr, None)
            for var in xobj.variables:
                xobj[var].encoding.pop(attr, None)

    else:
        raise ValueError("xobj must be an xarray Dataset or DataArray")

    return xobj


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
