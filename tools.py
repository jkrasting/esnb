import datetime
import os
import re
import subprocess
import tempfile
import warnings

import doralite
import intake_esm
import momgrid as mg
import pandas as pd
import xarray as xr
import yaml


class RequestedVariable:
    def __init__(
        self,
        varname,
        preferred_realm=None,
        standard_name=None,
        source_varname=None,
        units=None,
        preferred_chunkfreq=["5yr", "2yr", "1yr", "20yr"],
        freq="mon",
        ppkind="ts",
        dimensions=None,
    ):
        # Variable name used in the analysis script
        self.varname = varname
        self.preferred_realm = preferred_realm
        self.standard_name = standard_name
        self.source_varname = source_varname
        self.units = None
        self.preferred_chunkfreq = preferred_chunkfreq
        self.freq = freq
        self.ppkind = "ts"
        self.dimensions = dimensions
        self.catalog = None

    def to_dict(self):
        return {
            "varname": self.varname,
            "preferred_realm": self.preferred_realm,
            "standard_name": self.standard_name,
            "source_varname": self.source_varname,
            "units": self.source_varname,
            "preferred_chunkfreq": self.preferred_chunkfreq,
            "freq": self.freq,
            "ppkind": self.ppkind,
            "dimensions": self.dimensions,
        }

    @property
    def search_options(self):
        result = {}
        result["var"] = (
            self.source_varname if self.source_varname is not None else self.varname
        )
        if self.freq is not None:
            result["freq"] = self.freq
        if self.ppkind is not None:
            result["kind"] = self.ppkind
        if self.preferred_realm is not None:
            result["preferred_realm"] = self.preferred_realm
        if self.preferred_chunkfreq is not None:
            result["preferred_chunkfreq"] = self.preferred_chunkfreq
        return result

    def __repr__(self):
        reprstr = f"RequestedVariable {self.varname}"
        return reprstr

    def __str__(self):
        return str(self.varname)


class CaseExperiment:
    def __init__(
        self,
        location,
        name=None,
        date_range=None,
        catalog=None,
        source="dora",
        verbose=False,
    ):
        self.name = name
        self.location = location
        self.date_range = date_range
        self.source = source
        self.catalog = catalog

        if source == "dora":
            if verbose:
                print(f"{location}: Fetching metadata from Dora")
            self.metadata = doralite.dora_metadata(location)
            if self.name is None:
                self.name = self.metadata["expName"]
            if self.catalog is None:
                if verbose:
                    print(f"{location}: Loading intake catalog from Dora")
                self.catalog = load_dora_catalog(location)
                self.catalog = self.catalog.datetime()
        else:
            self.metadata = None

        if date_range is not None:
            self.date_range = xr_date_range_format(date_range)
            self.filter_date_range(self.date_range)

    @property
    def has_catalog(self):
        return self.catalog is not None

    def filter_date_range(self, date_range):
        assert self.has_catalog, (
            "Date range functionality only works when data catalog is loaded."
        )
        self.catalog = self.catalog.tsel(trange=tuple(self.date_range))

    def __repr__(self):
        name = "<empty>" if self.name is None else self.name
        date_range = "<unlimited>" if self.date_range is None else self.date_range
        return (
            f"CaseExperiment {name}: catalog={self.has_catalog} date_range={date_range}"
        )


class NotebookDiagnostic:
    def __init__(self, name, description=None, variables=None, **kwargs):
        self.name = name
        self.description = "" if description is None else description

        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]

        self.variables = variables
        self.diag_vars = kwargs

    def to_yaml(self):
        output = {}
        output["long_name"] = self.name
        output["description"] = self.description

        vardict = {}
        for v in self.variables:
            _vardict = v.to_dict()
            _ = _vardict.pop("varname", None)
            vardict[str(v)] = _vardict

        output["varlist"] = vardict

        if len(self.diag_vars) > 0:
            output["diag_vars"] = self.diag_vars

        return yaml.dump(
            output,
            Dumper=NoAliasDumper,
            indent=2,
            sort_keys=True,
            default_flow_style=False,
        )

    def __repr__(self):
        return f"NotebookDiagnostic {self.name}"


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
        self.concat_dim = concat_dim
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

        if name is None:
            if len(self.cases) == 1:
                self.name = self.cases[0].name
            elif len(self.cases) == 0:
                self.name = " *EMPTY* "
            else:
                self.name = "Multi-Case Group"
        else:
            self.name = name

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
        self.variables = [str(x) for x in diag.variables]
        self.is_resolved = True

    def dmget(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        call_dmget(self.files, verbose=verbose)

    def load(self, exact_times=True, consolidate=True):
        assert self.is_resolved is True, "Call .resolve_datasets() before loading"
        realms = sum([x.catalog.realms for x in self.cases], [])
        realms = list(set(realms))
        ds_by_realm = {}
        for realm in realms:
            subcats = [case.catalog.search(realm=realm) for case in self.cases]
            dsets = [x.to_xarray() for x in subcats]
            if len(dsets) > 1:
                _ds = xr.concat(dsets, "time")
            else:
                _ds = dsets[0]
            if exact_times:
                if self.date_range is not None:
                    dates = xr_date_range_format(self.date_range)
                else:
                    dates = (None, None)
                _ds = _ds.sel(time=slice(*dates))
            ds_by_realm[realm] = _ds
        if consolidate:
            self.ds = consolidate_datasets(ds_by_realm)
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
            if len(catalogs) == 1:
                result = catalogs[0]
            else:
                result = catalogs[0].merge(catalogs[1:])
        return result

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


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def clean_string(input_string):
    res = re.sub(r"[^a-zA-Z0-9\s]", "", input_string)
    res = res.replace(" ", "_")
    res = re.sub(r"_+", "_", res)
    return res


def xr_date_range_format(date_range):
    date_range = list(date_range)
    date_range = [str(x) for x in date_range]
    predicate = ["-01-01", "-12-31"]
    for x, tstr in enumerate(date_range):
        if len(tstr) <= 4:
            date_range[x] = date_range[x].zfill(4) + predicate[x]
    return date_range


def call_dmget(files, verbose=False):
    files = [files] if not isinstance(files, list) else files
    totalfiles = len(files)
    result = subprocess.run(["dmls", "-l"] + files, capture_output=True, text=True)
    result = result.stdout.splitlines()
    result = [x.split(" ")[-5:] for x in result]
    result = [(x[-1], int(x[0])) for x in result if x[-2] == "(OFL)"]

    if len(result) == 0:
        if verbose:
            print("dmget: All files are online")
    else:
        numfiles = len(result)
        paths, sizes = zip(*result)
        totalsize = round(sum(sizes) / 1024 / 1024, 1)
        if verbose:
            print(
                f"dmget: Dmgetting {numfiles} of {totalfiles} files requested ({totalsize} MB)"
            )
        cmd = ["dmget"] + paths
        _ = subprocess.check_output(cmd)


def case_groups_catalogs(case_groups, diag_settings):
    grp_catalogs = []
    date_ranges = []
    for k, v in case_groups.items():
        idnums = v["idnums"].replace(" ", "").split(",")
        catalogs = [load_dora_catalog(x) for x in idnums]
        if len(catalogs) > 1:
            catalog = catalogs[0].merge(catalogs[1:])
        else:
            catalog = catalogs[0]
        grp_catalogs.append(catalog)
        date_ranges.append(v["date_range"])

    for n, cat in enumerate(grp_catalogs):
        subcats = []
        for k, v in diag_settings["varlist"].items():
            subcat = cat.find(
                var=k,
                kind=v["ppkind"],
                preferred_chunkfreq=v["preferred_chunkfreq"],
                freq=v["freq"],
                preferred_realm=v["preferred_realm"],
            )
            subcats.append(subcat)
        if len(subcats) > 1:
            catalog = subcats[0].merge(subcats[1:])
        else:
            catalog = subcats[0]
        grp_catalogs[n] = catalog

    for n, v in enumerate(date_ranges):
        grp_catalogs[n] = reindex_catalog(grp_catalogs[n].find(trange=tuple(v)))

    return grp_catalogs


def consolidate_datasets(dset_dict):
    all_dsets = [v for _, v in dset_dict.items()]
    consolidated = []
    consolidated.append(all_dsets.pop(0))
    while len(all_dsets) > 0:
        n_consolidated = len(consolidated)
        candidate = all_dsets[0]
        for x in range(0, n_consolidated):
            merged = False
            try:
                _ds = xr.merge([consolidated[x], candidate], compat="no_conflicts")
                merged = True
            except:
                continue

            if merged:
                consolidated[x] = _ds
                _ = all_dsets.pop(0)
                break

        if len(all_dsets) > 0:
            consolidated.append(all_dsets.pop(0))
    assert len(all_dsets) == 0, "Consolidation failed -- some datasets left over"
    return consolidated


def copy_catalog(cat):
    _source = cat.source_catalog()
    _source["df"] = cat.df.copy()
    return Dora_datastore(_source)


def df_to_cat(df, label=""):
    for key in [
        "source_id",
        "experiment_id",
        "frequency",
        "table_id",
        "grid_label",
        "realm",
        "member_id",
        "chunk_freq",
    ]:
        df[key] = df[key].fillna("unknown")

    esmcat_memory = {
        "esmcat": {  # <== Metadata only here
            "esmcat_version": "0.0.1",
            "attributes": [
                {"column_name": "activity_id", "vocabulary": "", "required": False},
                {"column_name": "institution_id", "vocabulary": "", "required": False},
                {"column_name": "source_id", "vocabulary": "", "required": False},
                {"column_name": "experiment_id", "vocabulary": "", "required": True},
                {
                    "column_name": "frequency",
                    "vocabulary": "https://raw.githubusercontent.com/NOAA-GFDL/CMIP6_CVs/master/CMIP6_frequency.json",
                    "required": True,
                },
                {"column_name": "realm", "vocabulary": "", "required": True},
                {"column_name": "table_id", "vocabulary": "", "required": False},
                {"column_name": "member_id", "vocabulary": "", "required": False},
                {"column_name": "grid_label", "vocabulary": "", "required": False},
                {"column_name": "variable_id", "vocabulary": "", "required": True},
                {"column_name": "time_range", "vocabulary": "", "required": True},
                {"column_name": "chunk_freq", "vocabulary": "", "required": False},
                {"column_name": "platform", "vocabulary": "", "required": False},
                {"column_name": "target", "vocabulary": "", "required": False},
                {
                    "column_name": "cell_methods",
                    "vocabulary": "",
                    "required": False,
                },  # Adjusted from "enhanced" -> False
                {"column_name": "path", "vocabulary": "", "required": True},
                {
                    "column_name": "dimensions",
                    "vocabulary": "",
                    "required": False,
                },  # Adjusted from "enhanced" -> False
                {"column_name": "version_id", "vocabulary": "", "required": False},
                {
                    "column_name": "standard_name",
                    "vocabulary": "",
                    "required": False,
                },  # Adjusted from "enhanced" -> False
            ],
            "assets": {
                "column_name": "path",
                "format": "netcdf",
                "format_column_name": None,
            },
            "aggregation_control": {
                "variable_column_name": "variable_id",
                "groupby_attrs": [
                    "source_id",
                    "experiment_id",
                    "frequency",
                    "table_id",
                    "grid_label",
                    "realm",
                    "member_id",
                    "chunk_freq",
                ],
                "aggregations": [
                    {"type": "union", "attribute_name": "variable_id", "options": {}},
                    {
                        "type": "join_existing",
                        "attribute_name": "time_range",
                        "options": {
                            "dim": "time",
                            "coords": "minimal",
                            "compat": "override",
                        },
                    },
                ],
            },
            "id": label,
            "description": label,
            "title": label,
            "last_updated": datetime.datetime.now().isoformat(),
            "catalog_file": "dummy.csv",
        },
        "df": df,  # <== Your loaded DataFrame
    }

    return intake_esm.esm_datastore(esmcat_memory)


def infer_av_files(cat, subcat):
    avlist = [
        "ann",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]
    for var in subcat.vars:
        _subcat = cat.search(variable_id=var)
        for realm in _subcat.realms:
            varentry = _subcat.search(realm=realm).df.iloc[0]
            df = cat.search(variable_id=avlist).df
            df = df[df["path"].str.contains(f"/{realm}/")]
            for k in [
                "source_id",
                "experiment_id",
                "frequency",
                "realm",
                "variable_id",
            ]:
                df[k] = varentry[k]
            df["cell_methods"] = "av"
            df["standard_name"] = varentry["standard_name"]
            df["chunk_freq"] = df["chunk_freq"].str.replace("monthly_", "", regex=False)
            df["chunk_freq"] = df["chunk_freq"].str.replace("annual_", "", regex=False)
            df = df.reindex()
            _subcat = _subcat.merge(df_to_cat(df))
    return _subcat


def is_overlapping(period_a, period_b):
    start_a, end_a = period_a
    start_b, end_b = period_b
    if start_b is None or end_b is None:
        res = False
    else:
        res = start_a < end_b and end_a > start_b
    return res


def load_dora_catalog(idnum, **kwargs):
    return Dora_datastore(
        doralite.catalog(idnum).__dict__["_captured_init_args"][0], **kwargs
    )


def process_time_string(tstring):
    if isinstance(tstring, tuple):
        try:
            for x in tstring:
                assert isinstance(x, datetime.datetime) or x is None
            timetup = tstring
        except:
            timetup = (None, None)
    else:
        try:
            tstring = str(tstring)
            timetup = [x.ljust(8, "0") for x in tstring.split("-")]
            timetup = [[x[0:4], x[4:6], x[6:8]] for x in timetup]
            timetup[0][0] = int(timetup[0][0])
            timetup[0][1] = 1 if timetup[0][1] == "00" else int(timetup[0][1])
            timetup[0][2] = 1 if timetup[0][2] == "00" else int(timetup[0][2])
            timetup[1][0] = int(timetup[1][0])
            timetup[1][1] = 12 if timetup[1][1] == "00" else int(timetup[1][1])
            timetup[1][2] = 31 if timetup[1][2] == "00" else int(timetup[1][2])
            timetup = [tuple(x) for x in timetup]
            timetup = tuple([datetime.datetime(*x) for x in timetup])
        except:
            timetup = (None, None)

    return timetup


def reindex_catalog(cat):
    _source = cat.source_catalog()
    df = cat.df.copy()
    df = df.drop_duplicates("path")
    df = df.reset_index()
    _source["df"] = df
    return Dora_datastore(_source)


class Dora_datastore(intake_esm.core.esm_datastore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def source_catalog(self):
        return self.__dict__["_captured_init_args"][0]

    def find(
        self,
        var=None,
        freq=None,
        kind=None,
        trange=None,
        infer_av=False,
        preferred_realm=None,
        preferred_chunkfreq=None,
    ):
        res = copy_catalog(self)

        if var is not None:
            res = copy_catalog(res)
            res = res.search(variable_id=var)

        if infer_av is True:
            res = copy_catalog(res)
            res = infer_av_files(self, res)

        if freq is not None:
            res = copy_catalog(res)
            res = res.search(frequency=freq)

        if kind is not None:
            assert kind in ["av", "ts", "both"], "kind must be 'av, 'ts', or 'both'"
        else:
            kind = "both"
        kind = ["av", "ts"] if kind == "both" else [kind]
        res = copy_catalog(res)
        res = res.search(cell_methods=kind)

        if trange is not None:
            res = copy_catalog(res)
            res = res.datetime()
            res = res.tsel(trange)

        if preferred_realm is not None:
            preferred_realm = (
                [preferred_realm]
                if not isinstance(preferred_realm, list)
                else preferred_realm
            )
            if "ocean_month" in preferred_realm:
                preferred_realm = list(set(preferred_realm + ["ocean_monthly"]))
            if "ocean_monthly" in preferred_realm:
                preferred_realm = list(set(preferred_realm + ["ocean_month"]))
            _realm = " "
            for x in preferred_realm:
                if x in res.realms:
                    _realm = x
                    break
            res = copy_catalog(res)
            res = res.search(realm=_realm)
            if _realm == " ":
                warnings.warn(
                    f"None of the preferred realms were found: {preferred_realm}"
                )

        if preferred_chunkfreq is not None:
            preferred_chunkfreq = (
                [preferred_chunkfreq]
                if not isinstance(preferred_chunkfreq, list)
                else preferred_chunkfreq
            )
            _chunk_freq = " "
            for x in preferred_chunkfreq:
                if x in res.chunk_freqs:
                    _chunk_freq = x
                    break
            res = copy_catalog(res)
            res = res.search(chunk_freq=_chunk_freq)
            if _chunk_freq == " ":
                warnings.warn(
                    f"None of the preferred chunk frequencies were found: {preferred_chunkfreq}"
                )

        return res

    def tsel(self, trange):
        res = copy_catalog(self)
        _source = res.source_catalog()
        df = res.df.copy()
        trange = list(trange)
        trange = [x.split("-") for x in trange]
        trange[0] = datetime.datetime(*tuple([int(x) for x in trange[0]]))
        trange[1] = datetime.datetime(*tuple([int(x) for x in trange[1]]))
        trange = tuple(trange)
        non_matching_times = []
        for index, row in df.iterrows():
            if not is_overlapping(trange, row["time_range"]):
                non_matching_times.append(index)
        df = df.drop(non_matching_times)
        _source["df"] = df
        return Dora_datastore(_source)
        return res

    def datetime(self):
        _source = self.source_catalog()
        df = self.df.copy()
        df["time_range"] = df["time_range"].apply(process_time_string)
        _source["df"] = df
        return Dora_datastore(_source)

    def merge(self, catalogs):
        _source = self.source_catalog()
        if iter(catalogs):
            if isinstance(catalogs, intake_esm.core.esm_datastore):
                catalogs = [catalogs]
            elif isinstance(catalogs, Dora_datastore):
                catalogs = [catalogs]
            else:
                catalogs = list(catalogs)
        else:
            raise ValueError("input must be an iterable object")
        catalogs = [self] + catalogs
        _ids = [x.__dict__["_captured_init_args"][0]["esmcat"]["id"] for x in catalogs]
        _dfs = [x.df for x in catalogs]
        label = _ids[0] if all(x == _ids[0] for x in _ids) else ""
        _source["df"] = pd.concat(_dfs)
        _source["id"] = label
        _source["description"] = label
        _source["title"] = label
        return Dora_datastore(_source)

    def info(self, attr):
        return sorted(list(set(list(self.df[attr]))))

    def to_xarray(self, dmget=False):
        assert len(self.df) > 0, "No datasets to open."

        try:
            assert not len(self.realms) > 1
        except:
            raise ValueError(
                f"More than one realm is present in the catalog. Filter the catalog further. {self.realms}"
            )

        try:
            assert not len(self.chunk_freqs) > 1
        except:
            raise ValueError(
                f"More than one chunk frequency is present in the catalog. Filter the catalog further. {self.chunk_freqs}"
            )

        _paths = sorted(self.df["path"].tolist())
        if dmget is True:
            call_dmget(_paths)

        ds = xr.open_mfdataset(_paths, use_cftime=True)

        alltimes = sorted([t for x in list(self.df["time_range"].values) for t in x])
        ds.attrs["time_range"] = f"{alltimes[0].isoformat()},{alltimes[-1].isoformat()}"

        return ds

    def to_momgrid(self, dmget=False, to_xarray=True):
        res = mg.Gridset(self.to_xarray(dmget=dmget))
        if to_xarray:
            res = res.data
        return res

    @property
    def realms(self):
        return self.info("realm")

    @property
    def vars(self):
        return self.info("variable_id")

    @property
    def chunk_freqs(self):
        return self.info("chunk_freq")


class Case:
    def __init__(
        self, idnums=None, vars=None, trange=None, chunks=None, freq=None, ppkind=None
    ):
        self.idnums = [idnums] if not isinstance(idnums, list) else idnums
        assert isinstance(vars, dict), "vars must be a dictionary"
        self.vars = vars
        self.preferred_chunkfreq = chunks
        self.freq = freq
        self.ppkind = ppkind

        assert isinstance(trange, tuple), "trange must be a tuple"
        trange = list(trange)
        trange = [str(x) for x in trange]
        predicate = ["-01-01", "-12-31"]
        for x, tstr in enumerate(trange):
            if len(tstr) <= 4:
                trange[x] = trange[x].zfill(4) + predicate[x]
        self.trange = tuple(trange)

        metadata = {k: doralite.dora_metadata(k) for k in self.idnums}
        for x in metadata.keys():
            metadata[x]["catalog"] = load_dora_catalog(metadata[x]["master_id"])
        self.metadata = metadata
        self.expname = str(",").join([v["expName"] for k, v in self.metadata.items()])

        # merge multiple catalogs if necessary
        catalogs = [v["catalog"] for _, v in self.metadata.items()]
        for x, catalog in enumerate(catalogs):
            _filtered = []
            for varname, preferred_realm in self.vars.items():
                _cat = catalog.find(
                    var=varname,
                    preferred_realm=preferred_realm,
                    preferred_chunkfreq=self.preferred_chunkfreq,
                    freq=self.freq,
                    kind=self.ppkind,
                )
                _filtered.append(_cat)
            catalogs[x] = _filtered[0].merge(_filtered)

        if len(catalogs) > 1:
            catalog = catalogs[0].merge(catalogs)
        else:
            catalog = catalogs[0]

        catalog = catalog.find(trange=self.trange, infer_av=False)
        self.catalog = reindex_catalog(catalog)

    def __repr__(self):
        varnames = str(",").join(self.vars.keys())
        return f"Case object {self.expname}: {varnames} {self.trange}"
