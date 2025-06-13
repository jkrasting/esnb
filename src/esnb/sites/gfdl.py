import os
import socket
import subprocess

import pandas as pd
import xarray as xr

try:
    import doralite
except:
    pass

import logging

logger = logging.getLogger(__name__)

from esnb.core.esnb_datastore import esnb_datastore


def is_host_reachable(host, port=80, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False


dora_hostname = os.environ.get("ESNB_GFDL_DORA_HOSTNAME", "dora.gfdl.noaa.gov")
dora = is_host_reachable(dora_hostname, port=443)


def call_dmget(files, status=False, verbose=True):
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
        if status is False:
            cmd = ["dmget"] + list(paths)
            _ = subprocess.check_output(cmd)


def load_dora_catalog(idnum, **kwargs):
    return esnb_datastore(
        doralite.catalog(idnum).__dict__["_captured_init_args"][0], **kwargs
    )
