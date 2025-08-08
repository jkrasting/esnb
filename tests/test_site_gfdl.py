import intake_esm
import pytest
import tempfile
import shutil
import os

from esnb.sites import gfdl
from esnb.sites.gfdl import open_intake_catalog_dora, infer_is_gfdl_ppdir, infer_gfdl_expname

dora_url = "https://dora.gfdl.noaa.gov/api/intake/odiv-1.json"
dora_id = "odiv-1"
dora_id_2 = 895


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


def test_infer_gfdl_expname_1():
    pathpp = "/test/case/CM4_piControl_C/gfdl.ncrc4-intel16-prod-openmp/pp"
    assert infer_gfdl_expname(pathpp) == "CM4_piControl_C"

def test_infer_gfdl_expname_2():
    pathpp = "/test/case/CM4_piControl_C/arbitraty_string/pp"
    assert infer_gfdl_expname(pathpp) == "fre_experiment"


def test_infer_is_gfdl_ppdir_1():
    assert not infer_is_gfdl_ppdir("/home/jpk")


def test_infer_is_gfdl_ppdir_2():
    assert not infer_is_gfdl_ppdir(123)

def test_infer_is_gfdl_ppdir_3():
    temp_dir = tempfile.mkdtemp()
    test_path = temp_dir + "/pp"
    os.makedirs(test_path, exist_ok=True)
    successful = True
    try:
        assert infer_is_gfdl_ppdir(test_path)
    except Exception as exc:
        successful = False
        exception = exc
        pass
    shutil.rmtree(temp_dir)
    if not successful:
        raise exception

