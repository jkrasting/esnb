import pytest
import intake_esm

from esnb.core.CaseExperiment2 import CaseExperiment2, open_intake_catalog, open_intake_catalog_dora

dora_url = "https://dora.gfdl.noaa.gov/api/intake/odiv-1.json"
dora_id = "odiv-1"
dora_id_2 = 895
intake_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
mdtf_settings = "/home/jpk/pkgs/notebook-template/tests/test_data/input_timeslice_test.yml"
intake_path = "/home/jpk/pkgs/notebook-template/tests/test_data/intake-uda-cmip.json"

def test_infer_case_source():
    assert CaseExperiment2(dora_url).mode == "dora_url"
    assert CaseExperiment2(dora_id).mode == "dora_id"
    assert CaseExperiment2(dora_id_2).mode == "dora_id"
    assert CaseExperiment2(intake_url).mode == "intake_url"
    assert CaseExperiment2(mdtf_settings).mode == "mdtf_settings"
    assert CaseExperiment2(intake_path).mode == "intake_path"

def test_open_intake_from_path():
    result = open_intake_catalog(intake_path,"intake_path")
    assert isinstance(result, intake_esm.core.esm_datastore)

def test_open_intake_from_url():
    result = open_intake_catalog(intake_url,"intake_url")
    assert isinstance(result, intake_esm.core.esm_datastore)

def test_open_intake_from_dora_id_1():
    result = open_intake_catalog_dora(dora_id,"dora_id")
    assert isinstance(result, intake_esm.core.esm_datastore)

def test_open_intake_from_dora_id_2():
    result = open_intake_catalog_dora(dora_id_2,"dora_id")
    assert isinstance(result, intake_esm.core.esm_datastore)

def test_open_intake_from_dora_url():
    result = open_intake_catalog_dora(dora_url,"dora_url")
    assert isinstance(result, intake_esm.core.esm_datastore)

