import pytest
from esnb.core import mdtf


def test_MDTFCaseSettings():
    settings_file = "tests/test_data/input_timeslice_test.yml"
    settings = mdtf.MDTFCaseSettings
    settings.load_mdtf_settings_file(settings, settings_file)


def test_MDTFCaseSettings_invalid_file():
    with pytest.raises(FileNotFoundError):
        x = mdtf.MDTFCaseSettings
        x = x.load_mdtf_settings_file(x, "non_existent_file.yml")
