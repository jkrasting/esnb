import yaml

try:
    import doralite
except:
    pass


from esnb.core.util import xr_date_range_format
from esnb.sites import gfdl

from . import util


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

        # TODO - make conformant to MDTF keys where possible
        if source == "dora":
            if verbose:
                print(f"{location}: Fetching metadata from Dora")
            self.metadata = doralite.dora_metadata(location)
            if self.name is None:
                self.name = self.metadata["expName"]
            if self.catalog is None:
                if verbose:
                    print(f"{location}: Loading intake catalog from Dora")
                self.catalog = gfdl.load_dora_catalog(location)
                self.catalog = self.catalog.datetime()
                self.original_catalog = util.copy_catalog(self.catalog)

        elif source == "mdtf":
            self.location = location
            self.name = name
            self.date_range = date_range
            self.catalog = location
            # assert os.path.exists(location), "MDTF Case input file is not accessible"
            # with open(location, "r") as f:
            #    input_data = yaml.safe_load(f)
            #    self.settings = input_data
            #    self.__dict__ = {**self.__dict__, **input_data}

        else:
            self.metadata = None

        if date_range is not None:
            self.date_range = xr_date_range_format(date_range)
            self.filter_date_range(self.date_range)

    def dump(self, filename=None, type="yaml"):
        if filename is not None:
            with open(filename, "w") as f:
                yaml.dump(
                    self.settings,
                    f,
                    Dumper=NoAliasDumper,
                    indent=2,
                    sort_keys=True,
                    default_flow_style=False,
                )
        else:
            return yaml.dump(
                self.settings,
                Dumper=NoAliasDumper,
                indent=2,
                sort_keys=True,
                default_flow_style=False,
            )

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
