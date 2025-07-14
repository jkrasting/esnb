import esnb
import pytest
from esnb.core import mdtf
from esnb.core.util_mdtf import mdtf_settings_template_dict


def test_MDTFCaseSettings():
    settings_file = esnb.datasources.test_mdtf_settings
    settings = mdtf.MDTFCaseSettings
    settings.load_mdtf_settings_file(settings, settings_file)


def test_MDTFCaseSettings_invalid_file():
    with pytest.raises(FileNotFoundError):
        x = mdtf.MDTFCaseSettings
        x = x.load_mdtf_settings_file(x, "non_existent_file.yml")


def test_mdtf_settings_template_dict_1():
    result = mdtf_settings_template_dict()
    assert len(result) == 18


def test_mdtf_settings_template_dict_2():
    result = mdtf_settings_template_dict(foo="bar", startdate="18501231000000")
    assert len(result) == 19
