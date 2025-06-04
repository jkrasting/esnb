import datetime
import os
import re
import subprocess
import tempfile
import warnings

import intake_esm
import json
import pandas as pd
import xarray as xr
import yaml

from . import util

try:
    import doralite
    import momgrid as mg
except:
    pass


def json_init(name):
    with open(name, "r") as f:
        lines = [line.strip() for line in f]
    lines = [x for x in lines if "//" not in x]
    json_str = "".join(lines)
    return json.loads(json_str)


class NotebookDiagnostic:
    def __init__(
        self,
        name,
        description=None,
        dimensions=None,
        variables=None,
        varlist=None,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.dimensions = dimensions
        self.variables = variables
        self.varlist = varlist

        init_settings = {}

        # initialze empty default settings
        settings_keys = [
            "driver",
            "long_name",
            "convention",
            "description",
            "pod_env_vars",
            "runtime_requirements",
        ]

        for key in settings_keys:
            if key in kwargs.keys():
                init_settings[key] = kwargs.pop(key)
            else:
                init_settings[key] = None

        assert isinstance(name, str), "String or valid path must be supplied"

        # load an MDTF-compatible jsonc settings file
        if os.path.exists(name):
            loaded_file = json_init(name)
            settings = loaded_file["settings"]

            self.dimensions = (
                self.dimensions
                if self.dimensions is not None
                else loaded_file["dimensions"]
            )
            self.varlist = (
                self.varlist if self.varlist is not None else loaded_file["varlist"]
            )

            for key in settings.keys():
                if key in init_settings.keys():
                    if init_settings[key] is not None:
                        settings[key] = init_settings.pop(key)
                    else:
                        _ = init_settings.pop(key)

            settings = {**settings, **init_settings}
            settings_keys = list(set(settings_keys + list(settings.keys())))

        # case where a diagnostic is initalized directly
        else:
            if variables is not None:
                if not isinstance(variables, list):
                    variables = [variables]

            settings = init_settings

        # make long_name and description identical
        if self.description is not None:
            settings["long_name"] = self.description
        else:
            self.description = settings["long_name"]

        self.__dict__ = {**self.__dict__, **settings}

        # set the user defined options from whatever is left oever
        self.diag_vars = kwargs

        # stash the settings keys
        self._settings_keys = settings_keys

        # initialize an empty groups attribute
        self.groups = []

    @property
    def metrics(self):
        dimensions = {"json_structure": ["region", "model", "metric"]}
        results = {"Global": {group.name: group.metrics for group in self.groups}}
        metrics = {
            "DIMENSIONS": dimensions,
            "RESULTS": results,
        }
        return metrics

    def write_metrics(self, filename=None):
        print(json.dumps(self.metrics, indent=2))
        filename = (
            util.clean_string(self.name) + ".json" if filename is None else filename
        )
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nOutput written to: {filename}")

    @property
    def settings(self):
        result = {"settings": {}}
        for key in self._settings_keys:
            result["settings"][key] = self.__dict__[key]
        result["varlist"] = self.varlist
        result["dimensions"] = self.dimensions
        result["diag_vars"] = self.diag_vars
        return result

    @property
    def files(self):
        all_files = []
        for group in self.groups:
            for case in group.cases:
                all_files = all_files + case.catalog.files
        return sorted(all_files)

    @property
    def dsets(self):
        return [x.ds for x in self.groups]

    def dump(self, filename="settings.json", type="json"):
        if type == "json":
            filename = f"{filename}"
            with open(filename, "w") as f:
                json.dump(self.settings, f, indent=2)

    def dmget(self, status=False):
        _ = [x.dmget(status=status) for x in self.groups]

    def load(self):
        _ = [x.load() for x in self.groups]

    def resolve(self, groups=None):
        groups = [] if groups is None else groups
        groups = [groups] if not isinstance(groups, list) else groups
        self.groups = groups
        _ = [x.resolve_datasets(self) for x in self.groups]

    def __repr__(self):
        return f"NotebookDiagnostic {self.name}"
