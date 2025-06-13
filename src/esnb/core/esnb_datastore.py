import datetime
import warnings

import intake_esm
import pandas as pd
import xarray as xr

try:
    import momgrid as mg
except:
    pass

from . import util


class esnb_datastore(intake_esm.core.esm_datastore):
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
        res = util.copy_catalog(self)

        if var is not None:
            res = util.copy_catalog(res)
            res = res.search(variable_id=var)

        if infer_av is True:
            res = util.copy_catalog(res)
            res = infer_av_files(self, res)

        if freq is not None:
            res = util.copy_catalog(res)
            res = res.search(frequency=freq)

        if kind is not None:
            assert kind in ["av", "ts", "both"], "kind must be 'av, 'ts', or 'both'"
        else:
            kind = "both"
        kind = ["av", "ts"] if kind == "both" else [kind]
        res = util.copy_catalog(res)
        res = res.search(cell_methods=kind)

        if trange is not None:
            res = util.copy_catalog(res)
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
            res = util.copy_catalog(res)
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
            res = util.copy_catalog(res)
            res = res.search(chunk_freq=_chunk_freq)
            if _chunk_freq == " ":
                warnings.warn(
                    f"None of the preferred chunk frequencies were found: {preferred_chunkfreq}"
                )

        return res

    def tsel(self, trange):
        res = util.copy_catalog(self)
        _source = res.source_catalog()
        df = res.df.copy()
        trange = list(trange)
        trange = [x.split("-") for x in trange]
        trange[0] = datetime.datetime(*tuple([int(x) for x in trange[0]]))
        trange[1] = datetime.datetime(*tuple([int(x) for x in trange[1]]))
        trange = tuple(trange)
        non_matching_times = []
        for index, row in df.iterrows():
            if not util.is_overlapping(trange, row["time_range"]):
                non_matching_times.append(index)
        df = df.drop(non_matching_times)
        _source["df"] = df
        return esnb_datastore(_source)
        return res

    def datetime(self):
        _source = self.source_catalog()
        df = self.df.copy()
        df["time_range"] = df["time_range"].apply(util.process_time_string)
        _source["df"] = df
        return esnb_datastore(_source)

    def merge(self, catalogs):
        _source = self.source_catalog()
        if iter(catalogs):
            if isinstance(catalogs, intake_esm.core.esm_datastore):
                catalogs = [catalogs]
            elif isinstance(catalogs, esnb_datastore):
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
        return esnb_datastore(_source)

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

        ds = xr.open_mfdataset(
            _paths,
            use_cftime=True,
            decode_timedelta=True,
            coords="minimal",
            compat="override",
        )

        alltimes = sorted([t for x in list(self.df["time_range"].values) for t in x])
        ds.attrs["time_range"] = f"{alltimes[0].isoformat()},{alltimes[-1].isoformat()}"

        return ds

    def to_momgrid(self, dmget=False, to_xarray=True):
        res = mg.Gridset(self.to_xarray(dmget=dmget))
        if to_xarray:
            res = res.data
        return res

    @property
    def files(self):
        return self.info("path")

    @property
    def realms(self):
        return self.info("realm")

    @property
    def vars(self):
        return self.info("variable_id")

    @property
    def chunk_freqs(self):
        return self.info("chunk_freq")
