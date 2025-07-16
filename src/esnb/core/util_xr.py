import logging

import fsspec
import xarray as xr

from esnb.core.util2 import get_nesting_depth, infer_source_data_file_types

logger = logging.getLogger(__name__)


def open_paths(files, varname=None):
    file_type = infer_source_data_file_types(files)
    logger.debug(f"Found {file_type} files: {files}")

    if file_type == "unix_file":
        logger.info(f"Opening local files in xarray: {files}")
        _ds = open_xr(files)
    elif file_type == "google_cloud":
        logger.info(f"Opening Google Cloud stores in xarray: {files}")
        _ds = open_gcs(files)
    else:
        raise ValueError(f"There is no rule yet to open file type: {file_type}")

    if varname is not None:
        ds = xr.Dataset()

        ds[varname] = _ds[varname]

        if "z_i" in _ds.keys():
            logger.debug("Found `z_i` in dataset; associating it as a coordinate.")
            ds["z_i"] = _ds["z_i"]

        ds.attrs = dict(_ds.attrs)
    else:
        ds = _ds

    return ds


def open_gcs(files):
    mappers = [fsspec.get_mapper(x) for x in files]
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = [
        xr.open_zarr(x, decode_times=time_coder, decode_timedelta=True) for x in mappers
    ]
    ds = xr.merge(ds, compat="override")
    return ds


def open_xr(files, xr_merge_opts=None):
    xr_merge_opts = (
        {"coords": "minimal", "compat": "override"}
        if xr_merge_opts is None
        else xr_merge_opts
    )

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(
        files,
        decode_times=time_coder,
        decode_timedelta=True,
        chunks={},
        **xr_merge_opts,
    )
    return ds


def open_var_from_group(group, varname):
    concat_dim = group.concat_dim
    concat_dim = [concat_dim] if not isinstance(concat_dim, list) else concat_dim

    expected_nest_level = get_nesting_depth(group.cases)
    nest_level = len(concat_dim)
    if nest_level != expected_nest_level:
        logger.debug(
            f"Expecting concat_dim to have {expected_nest_level} entries but found {nest_level}"
        )

    ncases = len(group.cases)
    logger.debug(f"Found {ncases} CaseExperiment objects in this group")

    group_ds = []
    for n, case in enumerate(group.cases):
        case_elements = [case] if not isinstance(case, list) else case
        nelements = len(case_elements)
        logger.debug(f"This case has {nelements} elements: {case_elements}")
        case_elements = [
            open_paths(x.files(variable_id=varname), varname=varname)
            for x in case_elements
        ]
        if nelements > 1:
            cdim = concat_dim[1]
            logger.info(f"Concatenating datasets along dimension: {cdim}")
            ds = xr.concat(case_elements, cdim)
        else:
            ds = case_elements[0]
        group_ds.append(ds)

    if len(group_ds) > 1:
        cdim = concat_dim[0]
        logger.info(f"Concatenating datasets along dimension: {cdim}")
        ds = xr.concat(group_ds, cdim)
    else:
        ds = group_ds[0]

    return ds
