import json
import logging
import os

logger = logging.getLogger(__name__)

from . import html, util
from .RequestedVariable import RequestedVariable


def json_init(name):
    with open(name, "r") as f:
        lines = [line.strip() for line in f]
    lines = [x for x in lines if "//" not in x]
    json_str = "".join(lines)
    return json.loads(json_str)


class NotebookDiagnostic:
    def __init__(
        self,
        source,
        name=None,
        description=None,
        dimensions=None,
        variables=None,
        varlist=None,
        **kwargs,
    ):
        logger.info(f"Initalizing NotebookDiagnostic object from {source}")
        self.source = source
        self.description = description
        self.name = name
        self.dimensions = dimensions
        self.variables = variables
        self.varlist = varlist

        self.name = self.source if self.name is None else self.name

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

        assert isinstance(source, str), "String or valid path must be supplied"

        # load an MDTF-compatible jsonc settings file
        if os.path.exists(source):
            logger.info(f"Reading MDTF settings file from: {source}")
            loaded_file = json_init(source)
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

            self.variables = [
                RequestedVariable(k, **v) for k, v in self.varlist.items()
            ]

        # case where a diagnostic is initalized directly
        else:
            if variables is not None:
                if not isinstance(variables, list):
                    variables = [variables]

            settings = init_settings

        # make long_name and description identical
        if self.description is not None:
            settings["long_name"] = self.description
            settings["description"] = self.description
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

    def _repr_html_(self):
        result = html.gen_html_sub()
        # Table Header
        result += f"<h3>{self.name}</h3><i>{self.description}</i>"
        result += "<table class='cool-class-table'>"

        result += f"<tr><td><strong>name</strong></td><td>{self.name}</td></tr>"
        result += (
            f"<tr><td><strong>description</strong></td><td>{self.description}</td></tr>"
        )

        _vars = str(", ").join([x.varname for x in self.variables])
        result += f"<tr><td><strong>variables</strong></td><td>{_vars}</td></tr>"

        result += f"<tr><td><strong>groups</strong></td><td>{self.groups}</td></tr>"

        if len(self.diag_vars) > 0:
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>User-defined diag_vars</summary>"
            result += "<div><table>"
            for d_key in sorted(self.diag_vars.keys()):
                d_value = self.diag_vars[d_key]
                result += f"<tr><td>{d_key}</td><td>{d_value}</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        if len(self.settings) > 0:
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>MDTF Settings</summary>"
            result += "<div><table>"
            for d_key in sorted(self.settings.keys()):
                if d_key != "settings":
                    d_value = self.settings[d_key]
                    result += f"<tr><td>{d_key}</td><td>{d_value}</td></tr>"
                else:
                    for k in sorted(self.settings["settings"].keys()):
                        v = self.settings["settings"][k]
                        result += f"<tr><td>{k}</td><td>{v}</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        result += "</table>"

        return result
