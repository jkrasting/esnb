"""
This module defines file paths and URLs for test data catalogs, MDTF settings,
and the Pangeo CMIP6 catalog used in the esnb.datasources package.
"""

from importlib_resources import files

test_data_root = str(files("esnb.data"))

# -- test data catlogs
test_catalog_esm4_ctrl = str(files("esnb.data") / "ESM4_ctrl.json")
test_catalog_esm4_hist = str(files("esnb.data") / "ESM4_hist.json")
test_catalog_esm4_futr = str(files("esnb.data") / "ESM4_futr.json")
test_catalog_gfdl_uda = str(files("esnb.data") / "intake-uda-cmip.json")

# -- test mdtf settings
test_mdtf_settings = str(files("esnb.data") / "input_timeslice_test.yml")
test_mdtf_pod_settings = str(files("esnb.data") / "settings.jsonc")

# -- blank catalog
blank_catalog = str(files("esnb.data") / "blank_catalog.json")

# -- pangeo catalog
cmip6_pangeo = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
