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

from esnb.core.esnb_datastore import esnb_datastore


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


def load_dora_catalog(idnum, **kwargs):
    return esnb_datastore(
        doralite.catalog(idnum).__dict__["_captured_init_args"][0], **kwargs
    )
