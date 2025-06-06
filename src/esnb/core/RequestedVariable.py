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

try:
    import doralite
    import momgrid as mg
except:
    pass


class RequestedVariable:
    def __init__(
        self,
        varname,
        preferred_realm=None,
        path_variable=None,
        scalar_coordinates=None,
        standard_name=None,
        source_varname=None,
        units=None,
        preferred_chunkfreq=["5yr", "2yr", "1yr", "20yr"],
        frequency="mon",
        ppkind="ts",
        dimensions=None,
    ):
        # Variable name used in the analysis script
        self.path_variable = path_variable
        self.varname = varname
        self.preferred_realm = preferred_realm
        self.scalar_coordinates= scalar_coordinates
        self.standard_name = standard_name
        self.source_varname = source_varname
        self.units = units
        self.preferred_chunkfreq = preferred_chunkfreq
        self.frequency = frequency
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
            "frequency": self.frequency,
            "ppkind": self.ppkind,
            "dimensions": self.dimensions,
        }

    @property
    def search_options(self):
        result = {}
        result["var"] = (
            self.source_varname if self.source_varname is not None else self.varname
        )
        if self.frequency is not None:
            result["freq"] = self.frequency
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
