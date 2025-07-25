"""Module for computing area-weighted statistics"""

import numpy as np
import xarray as xr

__all__ = ["corr", "cov", "xr_stats_2d"]

# Xarray, Pandas, and Numpy do not have a multidimensional function
# to compute a weighted pattern correlation. Define internal functions
# for weighted covariance and correlation


def corr(arr1, arr2, area):
    """Calculates area-weighted correlation

    Parameters
    ----------
    arr1 : numpy.ndarray
        First array
    arr2 : numpy.ndarray
        Second array
    area : numpy.ndarray
        Cell area field

    Returns
    -------
    numpy.float64
    """
    return cov(arr1, arr2, area) / np.sqrt(
        cov(arr1, arr1, area) * cov(arr2, arr2, area)
    )


def cov(arr1, arr2, area):
    """Calculates area-weighted covariance

    Parameters
    ----------
    arr1 : numpy.ndarray
        First array
    arr2 : numpy.ndarray
        Second array
    area : numpy.ndarray
        Cell area field

    Returns
    -------
        numpy.float64

    """
    return np.sum(
        area
        * (arr1 - np.average(arr1, weights=area))
        * (arr2 - np.average(arr2, weights=area))
    ) / np.sum(area)


def xr_stats_2d(arr1, arr2, area, fmt="list"):
    """Calculates basic area-weighted statistics for two DataArrays

    Parameters
    ----------
    arr1: xarray.DataArray
        First input array
    arr2: xarray.DataArray
        Second input array
    area: xarray.DataArray
        Cell area field
    fmt: str
        Define output format "list" or "dict", default="list"

    Returns
    -------
        List or dict of bias, rmse, and pattern correlation

    """

    # load arrays into memory to prevent any warnings later
    arr1.load()
    arr2.load()
    area.load()
    # the two arrays may have different valid data masks
    # get the union of masks from both arrays
    mask = xr.where(arr1.isnull(), 0.0, 1.0) * xr.where(arr2.isnull(), 0.0, 1.0)
    # fill all NaNs with zeros and apply the unified mask to all arrays
    _arr1 = arr1.fillna(0.0) * mask
    _arr2 = arr2.fillna(0.0) * mask
    _area = area.fillna(0.0) * mask
    # calculate difference of arrays
    diff = _arr1 - _arr2
    # calculate area-weighted bias
    bias = diff.weighted(_area).mean()
    # rmse
    rse = np.sqrt(diff**2)
    rmse = rse.weighted(_area).mean()

    # convert to numpy arrays
    _arr1 = _arr1.values
    _arr2 = _arr2.values
    _area = _area.values
    # weighted pattern correlation
    rsquared = corr(_arr1, _arr2, _area)
    result = [float(bias.values), float(rmse.values), float(rsquared)]
    if fmt == "dict":
        result = dict(zip(["bias", "rmse", "rsquared"], result))
    return result
