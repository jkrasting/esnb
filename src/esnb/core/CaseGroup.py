import os
import tempfile

import xarray as xr

from esnb.sites import gfdl

from . import util
from .CaseExperiment import CaseExperiment


class CaseGroup:
    def __init__(
        self,
        locations,
        concat_dim=None,
        name=None,
        date_range=None,
        catalog=None,
        source="dora",
        verbose=True,
    ):
        self.locations = [locations] if not isinstance(locations, list) else locations
        if len(self.locations) > 1:
            assert concat_dim is not None, (
                "You must supply and existing or new dimension for concatenation"
            )
        self.concat_dim = "time" if concat_dim is None else concat_dim
        self.date_range = date_range
        self.source = source
        self.ds = None
        self.is_resolved = False
        self.is_loaded = False
        self.variables = []
        self.verbose = verbose

        self.cases = [
            CaseExperiment(
                x,
                date_range=date_range,
                catalog=catalog,
                source=source,
                verbose=verbose,
            )
            for x in self.locations
        ]

        self.original_catalog = self.cases[0].original_catalog

        if name is None:
            if len(self.cases) == 1:
                self.name = self.cases[0].name
            elif len(self.cases) == 0:
                self.name = " *EMPTY* "
            else:
                self.name = "Multi-Case Group"
        else:
            self.name = name

        self.metrics = {}

    def add_metric(self, name, keyval):
        assert isinstance(name, str), "metric group name must be a string"
        assert isinstance(keyval, tuple), "metric must be a (key, value) tuple"
        assert len(keyval) == 2
        key, value = keyval
        if name in self.metrics.keys():
            self.metrics[name] = {**self.metrics[name], key: value}
        else:
            self.metrics[name] = {key: value}

    def resolve_datasets(self, diag, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        variables = diag.variables
        for case in self.cases:
            if verbose:
                print(f"Resolving required vars for {case.name}")
            subcatalogs = []
            for var in variables:
                subcat = case.catalog.find(**var.search_options)
                subcatalogs.append(subcat)
            if len(subcatalogs) > 1:
                catalog = subcatalogs[0].merge(subcatalogs[1:])
            else:
                catalog = subcatalogs[0]
            case.catalog = catalog
            # Loop over realms for static files
            realms = list(set(list(case.catalog.df.realm)))
            statics = self.original_catalog.search(
                frequency="fx", table_id="fx", realm=realms
            )
            case.static_catalog = statics
        self.variables = [str(x) for x in diag.variables]
        self.is_resolved = True

    @property
    def ds_by_var(self):
        vardict = {k: None for k in self.variables}
        for var in vardict.keys():
            for ds in self.ds:
                if var in ds.keys():
                    vardict[var] = ds
        return vardict

    def dmget(self, status=False, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        gfdl.call_dmget(self.files, verbose=verbose)

    def open(self, exact_times=True, consolidate=True):
        return load(self, exact_times=exact_times, consolidate=consolidate)

    def load(self, exact_times=True, consolidate=True):
        assert self.is_resolved is True, "Call .resolve_datasets() before loading"
        realms = sum([x.catalog.realms for x in self.cases], [])
        realms = list(set(realms))
        ds_by_realm = {}
        for realm in realms:
            subcats = [case.catalog.search(realm=realm) for case in self.cases]
            dsets = [x.to_xarray() for x in subcats]
            if len(dsets) > 1:
                _ds = xr.concat(dsets, self.concat_dim)
            else:
                _ds = dsets[0]
            if exact_times:
                # TODO - add a check for the time range here
                if self.date_range is not None:
                    dates = util.xr_date_range_format(self.date_range)
                else:
                    dates = (None, None)
                _ds = _ds.sel(time=slice(*dates))
            ds_by_realm[realm] = _ds
        if consolidate:
            self.ds = util.consolidate_datasets(ds_by_realm)
        else:
            self.ds = ds_by_realm
        self.is_loaded = True

    def dump(self, dir=None, fname=None, type="netcdf"):
        assert self.is_loaded is True, "Call .load() before dumping to file"
        assert isinstance(self.ds, list), (
            "Datasets must be consolidated before dumping to file"
        )
        if dir is None:
            dir = tempfile.mkdtemp(dir=os.getcwd())
        assert os.path.isdir(dir)
        updated_ds = []
        if fname is None:
            name = self.name
        for ds in self.ds:
            t0 = ds.time.values[0].isoformat()
            t1 = ds.time.values[-1].isoformat()
            dsvars = list(set(self.variables) & set(list(ds.keys())))
            dsvars = str(" ").join(dsvars)
            fname = clean_string(f"{name} {t0} {t1} {dsvars}")
            resolved_path = f"{dir}/{fname}"
            if type == "netcdf":
                resolved_path = f"{dir}/{fname}.nc"
                ds.to_netcdf(resolved_path)
                updated_ds.append(resolved_path)
            elif type == "zarr":
                ds.to_zarr(resolved_path)
                updated_ds.append(resolved_path)
            else:
                raise ValueError(f"Unsupported type: {type}")
        self.ds = updated_ds

    @property
    def catalog(self):
        assert self.is_resolved, (
            "Datasets must be resolved first. Call .resolve_datasets()"
        )
        if len(self.cases) > 0:
            catalogs = [x.catalog for x in self.cases]
            static_catalogs = [x.static_catalog for x in self.cases]
            if len(catalogs) == 1:
                result = catalogs[0]
            else:
                result = catalogs[0].merge(catalogs[1:])
        return result

    @property
    def static_catalog(self):
        assert self.is_resolved, (
            "Datasets must be resolved first. Call .resolve_datasets()"
        )
        if len(self.cases) > 0:
            catalogs = [x.static_catalog for x in self.cases]
            if len(catalogs) == 1:
                result = catalogs[0]
            else:
                result = catalogs[0].merge(catalogs[1:])
        return result

    def load_statics(self):
        return xr.open_mfdataset(self.static_catalog.info("path"), decode_times=False)

    @property
    def files(self):
        return sorted(self.catalog.info("path"))

    def __repr__(self):
        nloc = len(self.locations)
        name = self.name
        res = ""
        res = (
            res
            + f"CaseGroup <{name}>  n_sources={nloc}  resolved={self.is_resolved}  loaded={self.is_loaded}"
        )
        if len(self.cases) >= 1:
            for case in self.cases:
                res = res + f"\n  * {str(case)}"
        return res
