from . import html


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
        **kwargs,
    ):
        # Variable name used in the analysis script
        self.path_variable = path_variable
        self.varname = varname
        self.preferred_realm = preferred_realm
        self.scalar_coordinates = scalar_coordinates
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

    def _repr_html_(self):
        result = html.gen_html_sub()
        # Table Header
        result += f"<h3>{self.__class__.__name__}  --  {self.varname}</h3>"
        result += "<table class='cool-class-table'>"

        inactive = {}
        for k in sorted(self.__dict__.keys()):
            val = self.__dict__[k]
            if val is not None:
                result += f"<tr><td><strong>{k}</strong></td><td>{val}</td></tr>"
            else:
                inactive[k] = val

        if len(inactive) > 0:
            result += "<tr><td colspan='2'>"
            result += "<details>"
            result += "<summary>Inactive Settings</summary>"
            result += "<div><table>"
            for d_key in sorted(inactive.keys()):
                d_value = inactive[d_key]
                result += f"<tr><td>{d_key}</td><td>{d_value}</td></tr>"
            result += "</table></div>"
            result += "</details>"
            result += "</td></tr>"

        return result

    def __repr__(self):
        reprstr = f"RequestedVariable {self.varname}"
        return reprstr

    def __str__(self):
        return str(self.varname)
