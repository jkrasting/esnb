import intake_esm
import pytest

from esnb.sites import gfdl
from esnb.sites.gfdl import open_intake_catalog_dora

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
