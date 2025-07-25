"""Module with mdtf helper functions"""


def mdtf_settings_template_dict(**kwargs):
    settings = {}
    settings["pod_list"] = []
    settings["case_list"] = {
        "case_name": "case_name",
        "model": "CMIP",
        "convention": "CMIP",
        "startdate": "00010101000000",
        "enddate": "99990101000000",
    }
    settings["DATA_CATALOG"] = ""
    settings["OBS_DATA_ROOT"] = ""
    settings["WORK_DIR"] = ""
    settings["OUTPUT_DIR"] = ""
    settings["conda_root"] = ""
    settings["conda_env_root"] = ""
    settings["micromamba_exe"] = ""
    settings["large_file"] = False
    settings["save_ps"] = False
    settings["save_pp_data"] = True
    settings["translate_data"] = True
    settings["make_variab_tar"] = False
    settings["overwrite"] = True
    settings["make_multicase_figure_html"] = False
    settings["run_pp"] = True
    settings["user_pp_scripts"] = {}

    required_keys = list(settings.keys())
    for key in required_keys:
        if isinstance(settings[key], dict):
            subkeys = list(settings[key].keys())
            for subkey in subkeys:
                result = kwargs.pop(subkey, "null result")
                if result != "null result":
                    settings[key][subkey] = result
        else:
            result = kwargs.pop(key, "null result")
            if result != "null result":
                settings[key] = result

    leftover_keys = list(kwargs.keys())
    if len(leftover_keys) > 0:
        for key in leftover_keys:
            settings[key] = kwargs.pop(key, "")

    return settings
