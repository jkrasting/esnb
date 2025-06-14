"""
This module provides utilities for interacting with the GFDL DORA system,
including checking host reachability, managing file staging with dmget,
and loading DORA catalogs via the esnb_datastore interface.
"""

import logging
import os
import socket
import subprocess

from esnb.core.esnb_datastore import esnb_datastore

__all__ = ["dora", "is_host_reachable", "call_dmget", "load_dora_catalog"]

try:
    import doralite
except Exception:
    pass


logger = logging.getLogger(__name__)


def is_host_reachable(host, port=80, timeout=1):
    """
    Check if a host is reachable on a specified port within a timeout period.

    Parameters
    ----------
    host : str
        The hostname or IP address to check for reachability.
    port : int, optional
        The port number to attempt the connection on (default is 80).
    timeout : float, optional
        The maximum time in seconds to wait for a connection (default is 1).

    Returns
    -------
    bool
        True if the host is reachable on the specified port within the timeout,
        False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False


def call_dmget(files, status=False, verbose=True):
    """
    Checks the online status of files and retrieves them from mass storage if needed.

    Uses the `dmls` command to check which files are offline and, if necessary,
    calls `dmget` to stage them online. Prints status messages if `verbose` is True.

    Parameters
    ----------
    files : str or list of str
        Path or list of paths to files to check and potentially stage online.
    status : bool, optional
        If True, only checks the status without staging files online. Default is
        False.
    verbose : bool, optional
        If True, prints status messages to stdout. Default is True.

    Returns
    -------
    None

    Notes
    -----
    Requires the `dmls` and `dmget` commands to be available in the system path.
    """
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
    """
    Load a Dora catalog using the provided identifier number.

    Parameters
    ----------
    idnum : int or str
        The identifier number for the Dora catalog to load.
    **kwargs
        Additional keyword arguments passed to `esnb_datastore`.

    Returns
    -------
    object
        The result of loading the Dora catalog via `esnb_datastore`.

    Notes
    -----
    This function retrieves the initialization arguments from the DoraLite
    catalog object and passes them to `esnb_datastore` along with any
    additional keyword arguments.
    """
    return esnb_datastore(
        doralite.catalog(idnum).__dict__["_captured_init_args"][0], **kwargs
    )


dora_hostname = os.environ.get("ESNB_GFDL_DORA_HOSTNAME", "dora.gfdl.noaa.gov")
dora = is_host_reachable(dora_hostname, port=443)
