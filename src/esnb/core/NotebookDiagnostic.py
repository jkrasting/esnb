import json
import logging
import os
import shutil
from pathlib import Path

import xarray as xr

from esnb.sites import gfdl

from . import html, util
from .CaseGroup2 import CaseGroup2
from .RequestedVariable import RequestedVariable
from .util2 import flatten_list, generate_tempdir_path, read_json, reset_encoding
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
        workdir=None,
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
        workdir : str, optional
            Path to temporary working directory
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
        self.workdir = workdir

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

        # initialize workdir
        if self.workdir is None:
            self.workdir = generate_tempdir_path(self.name)
        else:
            logger.info(f"Diagnostic workdir is set to: {self.workdir}")

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

    def load(self, site="gfdl", dmget=False, use_cache=False, cache_format="zarr"):
        """
        Load all groups by calling their load method.
        """
        if hasattr(self.groups[0], "dmget"):
            _ = [x.load() for x in self.groups]
        else:
            self.loader(
                site=site, dmget=dmget, use_cache=use_cache, cache_format=cache_format
            )

    def loader(self, site="gfdl", dmget=False, use_cache=False, cache_format="zarr"):
        diag = self
        groups = diag.groups
        variables = diag.variables

        if site == "gfdl" and dmget:
            gfdl.call_dmget(diag.files)

        # dictionary of datasets by var then group
        all_datasets = []
        counter = 0
        ds_by_var = {}
        for var in variables:
            ds_by_var[var] = {}
            for group in groups:
                logger.info(f"Opening `{var.varname}` datasets for group: {group.name}")
                workdir = self.workdir
                _date_range = str("_").join(list(group.date_range))
                cached_file_name = f"{group.name}_{var.varname}_{_date_range}"
                cached_file_name = Path(f"{cached_file_name}.{cache_format}")
                cached_file_name = workdir / cached_file_name
                logger.debug(f"Checking for cached file: {cached_file_name}")

                if use_cache and cached_file_name.exists():
                    logger.info(f"Opening cached dataset: {cached_file_name}")
                    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                    if cache_format == "zarr":
                        ds = xr.open_zarr(
                            cached_file_name,
                            decode_times=time_coder,
                            decode_timedelta=True,
                        )
                    else:
                        raise ValueError(
                            f"Trying to open unsupported cache type: {cache_format}"
                        )
                else:
                    ds = group.open_var(var.varname)

                    tcoord = "time"
                    logger.info(
                        f"Subsetting time range {group.date_range}: {group.name}"
                    )
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

    def open(
        self,
        site="gfdl",
        dmget=False,
        use_cache=False,
        cache_format="zarr",
        statics=False,
    ):
        self.load(
            site=site, dmget=dmget, use_cache=use_cache, cache_format=cache_format
        )
        if statics:
            logger.info("Loading dictionary of static files")
            try:
                _ = [x.open_statics() for x in self.groups]
            except Exception:
                logger.warning("Unable to load static files")

    def write_cache(
        self, workdir=None, output_format="zarr", overwrite=False, chunks=None
    ):
        write_cached_datasets(
            self,
            workdir=workdir,
            output_format=output_format,
            overwrite=overwrite,
            chunks=chunks,
        )

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
        esnb_case_data = os.environ.get("ESNB_CASE_DATA", None)
        esnb_case_file = os.environ.get("ESNB_CASE_FILE", None)

        logger.debug(f"Case override settings: ESNB_CASE_DATA={esnb_case_data}")
        logger.debug(f"Case override settings: ESNB_CASE_FILE={esnb_case_file}")

        if esnb_case_data is not None:
            logger.info("Converting case override data to dict")
            logger.info(
                "This feature is not fully implemented; falling back to original groups"
            )
            groups = groups
        elif esnb_case_file is not None:
            logger.info(f"Reading case override settings from file: {esnb_case_file}")
            if not os.path.exists(esnb_case_file):
                raise FileNotFoundError(f"File does not exist: {esnb_case_file}")
            groups = [CaseGroup2(esnb_case_file)]

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
        result += f"<tr><td><strong>workdir</strong></td><td>{self.workdir}</td></tr>"

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


def write_cached_datasets(
    diag, workdir=None, output_format="zarr", overwrite=False, chunks=None
):
    if workdir is None:
        workdir = diag.workdir

    workdir = Path(workdir)
    if not workdir.exists():
        logger.info(f"workdir does not exist, creating: {workdir}")
        os.makedirs(workdir)

    for group in diag.groups:
        for variable in group.datasets.keys():
            _date_range = str("_").join(list(group.date_range))
            output_name = f"{group.name}_{variable.varname}_{_date_range}"
            ds = group.datasets[variable]

            if output_format == "zarr":
                output_name = Path(f"{output_name}.{output_format}")
                output_name = workdir / output_name

                if output_name.exists() and overwrite:
                    logger.info(f"Found existing zarr and deleting: {output_name}")
                    shutil.rmtree(output_name)

                if not output_name.exists():
                    dsout = ds
                    dsout[variable.varname] = reset_encoding(dsout[variable.varname])

                    if chunks is not None:
                        logger.info(
                            f"Resetting chunks and applying new chunks: {chunks}"
                        )
                        chunksizes = chunks
                    else:
                        logger.info("Using automatic chunks")
                        chunksizes = "auto"

                    dsout = dsout.chunk(chunksizes)
                    chunksizes = dsout.chunksizes
                    logger.info(f"Output chunksizes are: {dict(chunksizes)}")

                    logger.info(f"Writing zarr file: {output_name}")
                    dsout.to_zarr(output_name)

                else:
                    logger.info(f"Found existing zarr -- doing nothing: {output_name}")
