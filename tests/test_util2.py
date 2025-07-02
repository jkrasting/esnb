import copy
import datetime as dt
import pathlib

import intake_esm

import esnb
from esnb import CaseExperiment2
from esnb.core.util2 import (
    case_time_filter,
    generate_tempdir_path,
    infer_source_data_file_types,
    initialize_cases_from_source,
    xr_date_range_to_datetime,
)

# TODO: Add tests for `read_json`, `reset_encoding`

source1 = esnb.datasources.test_catalog_gfdl_uda
source2 = esnb.datasources.test_mdtf_settings
source3 = esnb.datasources.test_catalog_esm4_hist

cat1 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_ctrl)
cat2 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_hist)
cat3 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_futr)
cat4 = intake_esm.esm_datastore(esnb.datasources.test_catalog_gfdl_uda)
cat5 = intake_esm.esm_datastore(esnb.datasources.cmip6_pangeo)

test_paths_1 = [
    "gs://cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/Amon/tas/gn/v20190308/",
    "gs://cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/Omon/tos/gr/v20190308/",
    "gs://cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/Omon/zos/gr/v20190308/",
]

test_paths_2 = [
    "/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/atmos_cmip/ts/monthly/5yr/atmos_cmip.198001-198412.tas.nc",
    "/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/atmos_cmip/ts/monthly/5yr/atmos_cmip.198501-198912.tas.nc",
    "/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/atmos_cmip/ts/monthly/5yr/atmos_cmip.199001-199412.tas.nc",
]


def test_case_time_filter():
    _source3 = copy.deepcopy(source3)
    case = CaseExperiment2(_source3)
    date_range = ("0041-01-01", "0060-12-31")
    n_times_start = int(case.catalog.nunique()["time_range"])
    _ = case_time_filter(case, date_range)
    n_times_end = int(case.catalog.nunique()["time_range"])
    print(n_times_start, n_times_end)
    assert n_times_end < n_times_start


def test_initialize_cases_from_source():
    _source1 = copy.deepcopy(source1)
    _source2 = copy.deepcopy(source2)
    source = [_source1, [_source2, _source2]]
    groups = initialize_cases_from_source(source)
    assert isinstance(groups, list)
    assert isinstance(groups[1], list)
    assert all(
        isinstance(x, esnb.core.CaseExperiment2.CaseExperiment2)
        for x in groups[1] + [groups[0]]
    )


def test_generate_tempdir_path_1():
    assert isinstance(generate_tempdir_path(), pathlib.Path)


def test_generate_tempdir_path_2():
    assert "abc123" in str(generate_tempdir_path("abc123"))


def test_xr_date_range_to_datetime():
    date_range = ("0041-01-01", "0060-12-31")
    assert xr_date_range_to_datetime(date_range) == (
        dt.datetime(41, 1, 1),
        dt.datetime(60, 12, 31),
    )


def test_infer_source_data_file_types_1():
    assert infer_source_data_file_types(test_paths_1) == "google_cloud"


def test_infer_source_data_file_types_2():
    assert infer_source_data_file_types(test_paths_2) == "unix_file"


# test_infer_source_data_file_types_1()
# test_infer_source_data_file_types_2()
