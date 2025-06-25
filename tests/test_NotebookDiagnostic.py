import os

import esnb
from esnb import NotebookDiagnostic

pod_settings_file = esnb.datasources.test_mdtf_pod_settings


def test_init_from_func():
    diag = NotebookDiagnostic("diag name")
    assert "settings" in diag.settings.keys()
    assert "varlist" in diag.settings.keys()
    assert "dimensions" in diag.settings.keys()
    assert "diag_vars" in diag.settings.keys()


def test_init_from_file():
    diag = NotebookDiagnostic(pod_settings_file)
    assert "settings" in diag.settings.keys()
    assert "varlist" in diag.settings.keys()
    assert "dimensions" in diag.settings.keys()
    assert "diag_vars" in diag.settings.keys()
    assert diag.varlist is not None
    assert diag.dimensions is not None


def test_settings_dump():
    diag = NotebookDiagnostic(pod_settings_file)
    diag.dump("dumped_settings.json")
    assert os.path.exists("dumped_settings.json")
    # TODO - check something in the file
    os.remove("dumped_settings.json")


def test_user_opts_1():
    user_options = {"option1": "apple", "option2": False}
    diag = NotebookDiagnostic("diag name", **user_options)
    assert diag.diag_vars == user_options
