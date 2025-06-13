import pytest
import intake_esm

from esnb.sites import gfdl

from esnb.core.CaseExperiment2 import (
    CaseExperiment2,
    infer_case_source,
    open_intake_catalog,
    open_intake_catalog_dora,
)

dora_url = "https://dora.gfdl.noaa.gov/api/intake/odiv-1.json"
dora_id = "odiv-1"
dora_id_2 = 895
intake_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
mdtf_settings = "tests/test_data/input_timeslice_test.yml"
intake_path = "tests/test_data/intake-uda-cmip.json"


def test_infer_case_source_dora_url():
    assert infer_case_source(dora_url) == "dora_url"


def test_infer_case_source_dora_id_1():
    assert infer_case_source(dora_id) == "dora_id"


def test_infer_case_source_dora_id_2():
    assert infer_case_source(dora_id_2) == "dora_id"


def test_infer_case_source_intake_url():
    assert infer_case_source(intake_url) == "intake_url"


def test_infer_case_source_mdtf_settings():
    assert infer_case_source(mdtf_settings) == "mdtf_settings"


def test_infer_case_source_intake_path():
    assert infer_case_source(intake_path) == "intake_path"


def test_open_intake_from_path():
    result = open_intake_catalog(intake_path, "intake_path")
    assert isinstance(result, intake_esm.core.esm_datastore)


def test_open_intake_from_url():
    result = open_intake_catalog(intake_url, "intake_url")
    assert isinstance(result, intake_esm.core.esm_datastore)


@pytest.mark.skipif(gfdl.dora is False, reason="GFDL Dora is not accessible")
def test_open_intake_from_dora_id_1():
    result = open_intake_catalog_dora(dora_id, "dora_id")
    assert isinstance(result, intake_esm.core.esm_datastore)


@pytest.mark.skipif(gfdl.dora is False, reason="GFDL Dora is not accessible")
def test_open_intake_from_dora_id_2():
    result = open_intake_catalog_dora(dora_id_2, "dora_id")
    assert isinstance(result, intake_esm.core.esm_datastore)


@pytest.mark.skipif(gfdl.dora is False, reason="GFDL Dora is not accessible")
def test_open_intake_from_dora_url():
    result = open_intake_catalog_dora(dora_url, "dora_url")
    assert isinstance(result, intake_esm.core.esm_datastore)
