import copy
import datetime as dt

import intake
import intake_esm

import esnb
from esnb.core.util_catalog import (
    check_schema_equivalence,
    convert_pangeo_catalog_to_catalogbuilder,
    merge_intake_catalogs,
    open_intake_catalog,
    reset_catalog_metadata,
    update_intake_dataframe,
)

dora_url = "https://dora.gfdl.noaa.gov/api/intake/odiv-1.json"
dora_id = "odiv-1"
dora_id_2 = 895
intake_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
mdtf_settings = esnb.datasources.test_mdtf_settings
intake_path = esnb.datasources.test_catalog_gfdl_uda


source1 = esnb.datasources.test_catalog_gfdl_uda
source2 = esnb.datasources.test_mdtf_settings
source3 = esnb.datasources.test_catalog_esm4_hist

cat1 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_ctrl)
cat2 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_hist)
cat3 = intake_esm.esm_datastore(esnb.datasources.test_catalog_esm4_futr)
cat4 = intake_esm.esm_datastore(esnb.datasources.test_catalog_gfdl_uda)
cat5 = intake_esm.esm_datastore(esnb.datasources.cmip6_pangeo)


def test_check_schema_equivalence_1():
    _cat4 = copy.deepcopy(cat4)
    assert check_schema_equivalence(_cat4, _cat4)


def test_check_schema_equivalence_2():
    _cat4 = copy.deepcopy(cat4)
    _cat5 = copy.deepcopy(cat5)
    assert not check_schema_equivalence(_cat4, _cat5)


def test_convert_pangeo_catalog_to_catalogbuilder_1():
    result = convert_pangeo_catalog_to_catalogbuilder()
    assert len(result.df.columns) == 17


def test_convert_pangeo_catalog_to_catalogbuilder_2():
    x = intake.open_esm_datastore(esnb.datasources.cmip6_pangeo)
    filtered = x.search(institution_id="NOAA-GFDL")
    result = convert_pangeo_catalog_to_catalogbuilder(filtered)
    assert len(result.df.columns) == 17
    assert len(set(result.df["source_id"])) == 6


def test_merge_intake_catalogs_1():
    _cat1 = copy.deepcopy(cat1)
    catalogs = _cat1
    merge_intake_catalogs(catalogs)


def test_merge_intake_catalogs_2():
    _cat2 = copy.deepcopy(cat2)
    _cat3 = copy.deepcopy(cat3)
    catalogs = [_cat2, _cat3]
    merge_intake_catalogs(catalogs)


def test_merge_intake_catalogs_3():
    _cat1 = copy.deepcopy(cat1)
    _cat2 = copy.deepcopy(cat2)
    _cat3 = copy.deepcopy(cat3)
    catalogs = [_cat1, _cat2, _cat3]
    merge_intake_catalogs(catalogs, id="merged catalog")


def test_open_intake_from_path():
    result = open_intake_catalog(intake_path, "intake_path")
    assert isinstance(result, intake_esm.core.esm_datastore)


def test_open_intake_from_url():
    result = open_intake_catalog(intake_url, "intake_url")
    assert isinstance(result, intake_esm.core.esm_datastore)


def test_reset_catalog_metadata():
    _cat = copy.deepcopy(cat4)
    original_name = str(_cat.esmcat.id)
    original_time = dt.datetime(*tuple(_cat.esmcat.last_updated.timetuple())[0:7])
    print(original_name, original_time)
    new_id = "new catalog"
    newcat = reset_catalog_metadata(_cat, id=new_id)
    new_name = str(newcat.esmcat.id)
    new_time = dt.datetime(*tuple(newcat.esmcat.last_updated.timetuple())[0:7])
    assert new_time > original_time
    assert new_name != original_name
    assert new_name == new_id


def test_update_intake_dataframe():
    _cat = copy.deepcopy(cat4)
    df = _cat.df
    df = df[:123]
    update_intake_dataframe(_cat, df)
