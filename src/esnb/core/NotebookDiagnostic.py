import json
import logging
import os

import fsspec
import xarray as xr

from esnb.sites import gfdl

from . import html, util
from .RequestedVariable import RequestedVariable
from .util2 import flatten_list, infer_source_data_file_types, read_json
from .VirtualDataset import VirtualDataset

# import warnings


logger = logging.getLogger(__name__)


class NotebookDiagnostic:
    """
    Class for managing and representing notebook diagnostics, including
    settings, variables, groups, and metrics.

    This class can be initialized from a JSON settings file or directly from
    provided arguments. It supports serialization, metrics reporting, and
    HTML representation for use in Jupyter notebooks.

    Parameters
    ----------
    source : str
        Path to the settings file or a string identifier.
    name : str, optional
        Name of the diagnostic.
    description : str, optional
        Description of the diagnostic.
    dimensions : dict, optional
        Dimensions associated with the diagnostic.
    variables : list, optional
        List of variables for the diagnostic.
    varlist : dict, optional
        Dictionary of variable definitions.
    **kwargs
        Additional keyword arguments for settings and user-defined options.

    Attributes
    ----------
    source : str
        Source path or identifier.
    name : str
        Name of the diagnostic.
    description : str
        Description of the diagnostic.
    dimensions : dict
        Dimensions of the diagnostic.
    variables : list
        List of RequestedVariable objects.
    varlist : dict
        Dictionary of variable definitions.
    diag_vars : dict
        User-defined diagnostic variables.
    groups : list
        List of diagnostic groups.
    _settings_keys : list
        List of settings keys.
    """

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
        """
        Initialize a NotebookDiagnostic object from a settings file or arguments.

        Parameters
        ----------
        source : str
            Path to the settings file or a string identifier.
        name : str, optional
            Name of the diagnostic.
        description : str, optional
            Description of the diagnostic.
        dimensions : dict, optional
            Dimensions associated with the diagnostic.
        variables : list, optional
            List of variables for the diagnostic.
        varlist : dict, optional
            Dictionary of variable definitions.
        **kwargs
            Additional keyword arguments for settings and user-defined options.
        """
        logger.info(f"Initalizing NotebookDiagnostic object from {source}")
        self.source = source
        self.description = description
        self.name = name
        self.dimensions = dimensions
        self.variables = variables
        self.varlist = varlist

        self.name = self.source if self.name is None else self.name

        init_settings = {}

        # Needed for tracked list
        self._observers = {}

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
            loaded_file = read_json(source)
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
        """
        Return a dictionary containing diagnostic metrics and dimensions.

        Returns
        -------
        dict
            Dictionary with 'DIMENSIONS' and 'RESULTS' keys representing
            metric dimensions and results.
        """
        dimensions = {"json_structure": ["region", "model", "metric"]}
        results = {"Global": {group.name: group.metrics for group in self.groups}}
        metrics = {
            "DIMENSIONS": dimensions,
            "RESULTS": results,
        }
        return metrics

    def write_metrics(self, filename=None):
        """
        Write diagnostic metrics to a JSON file.

        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses a cleaned version of the diagnostic
            name with '.json' extension.
        """
        print(json.dumps(self.metrics, indent=2))
        filename = (
            util.clean_string(self.name) + ".json" if filename is None else filename
        )
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nOutput written to: {filename}")

    @property
    def settings(self):
        """
        Return a dictionary of diagnostic settings and metadata.

        Returns
        -------
        dict
            Dictionary containing settings, varlist, dimensions, and diag_vars.
        """
        result = {"settings": {}}
        for key in self._settings_keys:
            result["settings"][key] = self.__dict__[key]
        result["varlist"] = self.varlist
        result["dimensions"] = self.dimensions
        result["diag_vars"] = self.diag_vars
        return result

    @property
    def files(self):
        """
        Return a sorted list of all files from all cases in all groups.

        Returns
        -------
        list
            Sorted list of file paths from all cases in all groups.
        """
        if hasattr(self.groups[0], "resolve_datasets"):
            # warnings.warn("Legacy CaseGroup object found.  Make sure you are using the latest version of ESNB.", DeprecationWarning, stacklevel=2)
            all_files = []
            for group in self.groups:
                for case in group.cases:
                    all_files = all_files + case.catalog.files
            return sorted(all_files)
        else:
            return sorted(flatten_list([x.files for x in self.groups]))

    @property
    def dsets(self):
        """
        Return a list of datasets from all groups.

        Returns
        -------
        list
            List of datasets from each group.
        """
        return [x.ds for x in self.groups]

    def dump(self, filename="settings.json", type="json"):
        """
        Dump diagnostic settings to a file in the specified format.

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'settings.json'.
        type : str, optional
            Output format. Currently only 'json' is supported.
        """
        if type == "json":
            filename = f"{filename}"
            with open(filename, "w") as f:
                json.dump(self.settings, f, indent=2)

    def dmget(self, status=False):
        """
        Call the dmget method for all groups.

        Parameters
        ----------
        status : bool, optional
            Status flag to pass to each group's dmget method.
        """
        if hasattr(self.groups[0], "dmget"):
            # warnings.warn("Legacy CaseGroup object found.  Make sure you are using the latest version of ESNB.", DeprecationWarning, stacklevel=2)
            _ = [x.dmget(status=status) for x in self.groups]
        else:
            gfdl.call_dmget(self.files, status=status)

    def load(self, site="gfdl"):
        """
        Load all groups by calling their load method.
        """
        if hasattr(self.groups[0], "dmget"):
            _ = [x.load() for x in self.groups]
        else:
            self.loader(site=site)

    def loader(self, site="gfdl"):
        diag = self
        groups = diag.groups
        variables = diag.variables

        xr_merge_opts = {"coords": "minimal", "compat": "override"}

        def _open_xr(files, varname=None):
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            _ds = xr.open_mfdataset(
                files, decode_times=time_coder, decode_timedelta=True, **xr_merge_opts
            )
            if varname is not None:
                ds = xr.Dataset()
                ds[varname] = _ds[varname]
                ds.attrs = dict(_ds.attrs)
            else:
                ds = _ds
            return ds

        def _open_gcs(files, varname=None):
            mappers = [fsspec.get_mapper(x) for x in files]
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            _ds = [
                xr.open_zarr(x, decode_times=time_coder, decode_timedelta=True)
                for x in mappers
            ]
            _ds = xr.merge(_ds, compat="override")

            if varname is not None:
                ds = xr.Dataset()
                ds[varname] = _ds[varname]
                ds.attrs = dict(_ds.attrs)
            else:
                ds = _ds

            return ds

        if site == "gfdl":
            gfdl.call_dmget(diag.files)

        # dictionary of datasets by var then group
        all_datasets = []
        counter = 0
        ds_by_var = {}
        for var in variables:
            ds_by_var[var] = {}
            for group in groups:
                concat_dim = getattr(group, "concat_dim", None)
                # print(var, group, concat_dim)
                ncases = len(group.cases)
                if ncases > 1:
                    assert concat_dim is not None, (
                        f"Multiple cases discovered in group {group} but no concat_dim found"
                    )

                files = []
                for case in group.cases:
                    # print(f"  - {case}")
                    files.append(
                        list(case.catalog.search(variable_id=var.varname).df["path"])
                    )

                # TODO implement infer_source_data_file_types()
                file_type = infer_source_data_file_types(flatten_list(files))

                if file_type == "unix_file":
                    dsets = [_open_xr(x, var.varname) for x in files]
                elif file_type == "google_cloud":
                    dsets = [_open_gcs(x, var.varname) for x in files]
                else:
                    raise ValueError(
                        f"There is no rule yet to open file type: {file_type}"
                    )

                if len(dsets) > 1:
                    ds = xr.concat(dsets, concat_dim)
                else:
                    ds = dsets[0]

                # Select date range
                tcoord = "time"
                ds = ds.sel({tcoord: slice(*group.date_range)})

                ds = VirtualDataset(ds)
                all_datasets.append(ds)
                ds_by_var[var][group] = ds
                counter = counter + 1

        # dictionary of datasets by group then var
        ds_by_group = {}
        for var in ds_by_var.keys():
            for group in ds_by_var[var].keys():
                if group not in ds_by_group.keys():
                    ds_by_group[group] = {}
                ds_by_group[group][var] = ds_by_var[var][group]

        # assign datasets back to their group and variable objects
        for group in ds_by_group.keys():
            group._datasets = ds_by_group[group]

        for var in ds_by_var.keys():
            var._datasets = ds_by_var[var]

        # set group loaded status
        for group in ds_by_group.keys():
            group.is_loaded = True

        # set top-level datasets
        self._datasets = all_datasets

    def open(self, site="gfdl"):
        self.load(site=site)

    @property
    def datasets(self):
        return [x.dataset for x in self._datasets]

    def access_dataset(self, id=0):
        return self.datasets[id]

    def resolve(self, groups=None):
        """
        Resolve datasets for the provided groups and assign them to the
        diagnostic.

        Parameters
        ----------
        groups : list or None, optional
            List of groups to resolve. If None, uses an empty list.
        """
        groups = [] if groups is None else groups
        groups = [groups] if not isinstance(groups, list) else groups
        self.groups = groups
        if hasattr(self.groups[0], "resolve_datasets"):
            # warnings.warn("Legacy CaseGroup object found.  Make sure you are using the latest version of ESNB.", DeprecationWarning, stacklevel=2)
            _ = [x.resolve_datasets(self) for x in self.groups]
        else:
            _ = [x.resolve(self.variables) for x in self.groups]

    def __repr__(self):
        """
        Return a string representation of the NotebookDiagnostic object.

        Returns
        -------
        str
            String representation.
        """
        return f"NotebookDiagnostic {self.name}"

    def _repr_html_(self):
        """
        Return an HTML representation of the NotebookDiagnostic for Jupyter
        display.

        Returns
        -------
        str
            HTML string representing the diagnostic.
        """
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
        _grps = str("<br>").join([x.name for x in self.groups])
        result += f"<tr><td><strong>groups</strong></td><td>{_grps}</td></tr>"

        # result += "<tr><td colspan='2'>"
        # result += "<details>"
        # result += "<summary>Group Details</summary>"
        # result += "<div><table>"
        # for grp in self.groups:
        #     result += f"<tr>{grp._repr_html_(title=False)}</tr>"
        # result += "</table></div>"
        # result += "</details>"
        # result += "</td></tr>"

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

            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>Variable Details</summary>"
            result += "<div><table>"
            for var in self.variables:
                result += f"<tr>{var._repr_html_(title=False)}</tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>CaseGroup Details</summary>"
            result += "<div><table>"
            for group in self.groups:
                result += f"<tr>{group._repr_html_(title=False)}</tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"
            result += "</table>"

        result += "</table>"

        return result
