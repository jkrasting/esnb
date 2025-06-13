"""Module with MDTF bindings"""

import logging
from pathlib import Path

import yaml
from esnb.core.util import missing_dict_keys

logger = logging.getLogger(__name__)


def gen_mdtf_settings_file_stub():
    return {
        "pod_list": [],
        "case_list": {
            "casename": {
                "model": None,
                "convention": None,
                "startdate": "YYYYMMDD000000",
                "enddate": "YYYYMMDD000000",
            }
        },
        "DATA_CATALOG": None,
        "OBS_DATA_ROOT": None,
        "WORK_DIR": None,
        "OUTPUT_DIR": None,
        "conda_root": None,
        "conda_env_root": None,
        "micromamba_exe": None,
        "large_file": None,
        "save_ps": None,
        "save_pp_data": None,
        "translate_data": None,
        "make_variab_tar": None,
        "overwrite": None,
        "make_multicase_figure_html": None,
        "run_pp": None,
        "user_pp_scripts": None,
    }


def standardize_mdtf_case_settings(settings):
    target = gen_mdtf_settings_file_stub()
    for k in target.keys():
        if k in settings.keys():
            target[k] = settings[k]
    return target


class MDTFCaseSettings:
    """
    Class to handle MDTF case settings and metadata.
    This class is designed to load and manage MDTF settings files, which
    contain metadata about cases, data catalogs, and other relevant information.
    """

    def load_mdtf_settings_file(self, settings_file):
        """
        Load MDTF settings from a YAML file.

        Parameters
        ----------
        settings_file : str or Path
            Path to the MDTF settings YAML file.

        Raises
        ------
        FileNotFoundError
            If the settings file does not exist.
        ValueError
            If required keys are missing in the settings file.
        """
        settings_file = Path(settings_file)
        if not settings_file.exists():
            raise FileNotFoundError(
                f"MDTF settings file does not exist: {settings_file}"
            )
        self.source = str(settings_file)

        with open(settings_file, "r") as f:
            _settings = yaml.safe_load(f)

        missing_keys = missing_dict_keys(_settings, ["DATA_CATALOG", "case_list"])
        if len(missing_keys) > 0:
            raise ValueError(
                f"Encountered missing fields {missing_keys} in MDTF settings file {self.source}"
            )

        # set special attribute
        self.catalog = _settings["DATA_CATALOG"]
        self.mdtf_settings = _settings

        assert len(self.catalog) > 0, (
            f"`DATA_CATALOG` is empty in MDTF settings file {self.source}"
        )

    def write_mdtf_settings_file(self, filename="case_settings.yml", fmt="yaml"):
        _settings = getattr(self, "mdtf_settings", None)
        _settings = gen_mdtf_settings_file_stub() if _settings is None else _settings
        _settings = standardize_mdtf_case_settings(_settings)
        write_dict(_settings, filename, fmt)
