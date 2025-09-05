"""
This module provides utilities for interacting with the GFDL DORA system,
including checking host reachability, managing file staging with dmget,
and loading DORA catalogs via the esnb_datastore interface.
"""

import getpass
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import intake
import pandas as pd
import requests
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from esnb.core.esnb_datastore import esnb_datastore

__all__ = [
    "dora",
    "generate_gfdl_intake_catalog",
    "is_host_reachable",
    "infer_gfdl_expname",
    "infer_is_gfdl_ppdir",
    "call_dmget",
    "load_dora_catalog",
    "open_intake_catalog_dora",
    "slurm_stub",
]

try:
    import doralite
except Exception:
    pass


logger = logging.getLogger(__name__)


def default_slurm_account(user=None):
    user = getpass.getuser() if user is None else str(user)
    cmd = ["sacctmgr", "show", "user", user, "format=defaultaccount", "-nP"]
    default_account = subprocess.check_output(cmd, text=True).strip()
    return default_account


def dask_cluster_ppan(highmem=False, **kwargs):
    if "account" not in kwargs.keys():
        default_account = default_slurm_account()
        logger.info(f"Setting SLURM account to {default_account}")
        kwargs["account"] = default_account

    if "dashboard_address" in kwargs.keys():
        kwargs["scheduler_options"] = {"dashboard_address": kwargs["dashboard_address"]}
        del kwargs["dashboard_address"]

    if "jobs" in kwargs.keys():
        jobs = kwargs["jobs"]
        del kwargs["jobs"]
    else:
        jobs = None

    if "wait" in kwargs.keys():
        wait = kwargs["wait"]
        del kwargs["wait"]
    else:
        wait = True

    options = {
        "queue": "batch",
        "cores": 8,
        "processes": 2,
        "walltime": "01:00:00",
        "local_directory": "$TMPDIR",
        "death_timeout": 120,
        "job_name": "esnb-dask",
        "memory": "16GB",
    }

    if highmem:
        options["job_extra_directives"] = ["--exclude=pp[008-010],pp[013-075]"]

    options = {**options, **kwargs}

    try:
        cluster = SLURMCluster(**options)
        client = Client(cluster)
        logger.info(
            f"Successfully started SLURM dask cluster: {cluster.dashboard_link}"
        )

    except Exception as exc:
        logger.warning(f"Unable to start SLURMCluster: {exc}")
        cluster = None
        client = None

    if (cluster is not None) and (jobs is not None):
        try:
            cluster.scale(jobs=int(jobs))
            logger.info(f"Scaling SLURM dask cluster to {jobs} jobs.")

            if wait:
                logger.info("Waiting for a worker to come online ...")
                client.wait_for_workers(n_workers=1)

        except Exception as exc:
            logger.warning(f"Unable to scale the SLURM cluster: {exc}")
            cluster.close()

            if client is not None:
                client.close()

            cluster = None
            client = None

    return (cluster, client)


def generate_gfdl_intake_catalog(pathpp, fre_cli=None):
    logger.info(f"Generating intake catalog for: {pathpp}")
    current_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created tempdir: {temp_dir}")
    os.chdir(temp_dir)

    fre_cli = (
        "/home/fms/local/opt/fre/test/bin/fre" if fre_cli is None else str(fre_cli)
    )
    fre_cli = Path(fre_cli)
    logger.debug(f"Using FRE CLI path: {fre_cli}")
    assert fre_cli.exists()

    command = f"{fre_cli} catalog build {pathpp} catalog"
    logger.debug(f"Running: {command}")
    subprocess.run(
        command.split(" "), stderr=subprocess.DEVNULL, check=True, capture_output=False
    )

    assert Path("catalog.json").exists()
    assert Path("catalog.csv").exists()

    logger.debug("Removing extraneous NaNs from generated catalog")
    df = pd.read_csv("catalog.csv")
    for col in df.columns:
        df[col] = df[col].fillna("unknown")
    df.to_csv("catalog.csv", index=False)

    logger.debug("Loading csv into intake catalog object")
    catalog = intake.open_esm_datastore("catalog.json")

    # Immediately force collection of any lazy operations
    try:
        # Method 1: Access df property which should trigger collection
        if hasattr(catalog, "df"):
            df_len = len(catalog.df)
            print(f"Catalog collected: {df_len} entries")

        # Method 2: If there are specific lazy frames, collect them
        if hasattr(catalog, "esmcat") and hasattr(catalog.esmcat, "_frames"):
            frames = catalog.esmcat._frames
            if hasattr(frames, "lf") and frames.lf is not None:
                # Force immediate collection
                frames.pl_df = frames.lf.collect()
                if not hasattr(frames, "df") or frames.df is None:
                    frames.df = frames.pl_df.to_pandas(use_pyarrow_extension_array=True)

    except Exception as exc:
        logger.error("Unable to access generated intake catalog")
        raise exc

    logger.debug(f"Removing tempdir: {temp_dir}")
    os.chdir(current_dir)
    shutil.rmtree(temp_dir)

    return catalog


def infer_gfdl_expname(pathpp):
    # split path into a list:
    pathpp = pathpp.split("/")
    # find the element that includes a FRE target:
    fre_targets = ["prod", "repro", "debug"]
    index = [
        idx
        for idx, s in enumerate(pathpp)
        if any(target in s for target in fre_targets)
    ]
    # infer that the experiment immediately precedes the target-platform part of the path
    expname = pathpp[index[-1] - 1] if len(index) > 0 else "fre_experiment"
    return expname


def infer_is_gfdl_ppdir(location):
    location = str(location) if not isinstance(location, str) else location
    location = Path(location)
    return (location.is_dir()) and ("/pp" in str(location))


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


def convert_to_momgrid(diag, positive_longitudes=False, fatal_on_error=True):
    import momgrid

    for n, ds in enumerate(diag._datasets):
        try:
            if "yT" in ds.keys():
                ds.rename({"yT": "yh"})
            if "xT" in ds.keys():
                ds.rename({"xT": "xh"})
            _ds = momgrid.Gridset(ds.dataset)
            if positive_longitudes:
                logger.debug("Converting geolon to postive definite values")
                _ds.data["geolon"] = _ds.data["geolon"] % 360
            _ds.data.attrs["model"] = _ds.model
            logger.debug(f"Replacing existing dataset [{n}] with momgrid version")
            ds.replace(_ds.data)
        except Exception as exc:
            logger.warning(f"Unable to convert dataset [{n}] with momgrid: {exc}")
            if fatal_on_error: 
                raise exc


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


def open_intake_catalog_dora(source, mode):
    """
    Opens an intake ESM datastore catalog from a specified source and mode.

    Parameters
    ----------
    source : str
        The source identifier. If `mode` is "dora_url", this should be the full
        URL to the intake catalog. If `mode` is "dora_id", this should be the
        identifier used to construct the catalog URL.
    mode : str
        The mode specifying how to interpret `source`. Must be either "dora_url"
        to use `source` as a direct URL, or "dora_id" to construct the URL from
        a known pattern.

    Returns
    -------
    catalog : intake.ESMDataStore
        The opened intake ESM datastore catalog.

    Raises
    ------
    RuntimeError
        If an unrecognized `mode` is provided.

    Notes
    -----
    Logs the process of fetching the catalog and checks network availability to
    the Dora service.
    """
    if mode == "dora_url":
        url = source
    elif mode == "dora_id":
        url = f"https://{dora_hostname}/api/intake/{source}.json"
    else:
        err = f"Encountered unrecognized source mode: {mode}"
        loggger.error(err)  # noqa
        raise RuntimeError(err)

    logger.info(f"Fetching intake catalog from url: {url}")
    if not dora:
        logger.critical("Network route to dora is unavailble. Check connection.")

    try:
        catalog = intake.open_esm_datastore(url)
    except Exception as exc:
        url = url.replace("api/intake", "api/intake/catalog")
        url = url.replace(".json", ".csv.gz")
        response = requests.get(url)
        if response.status_code == 404:
            msg = (
                "Intake catalog does not exist or was not generated. "
                + f"Wait 6 hours or see a Dora admin. URL:{url}"
            )
            raise ValueError(msg)
        else:
            raise exc

    return catalog


def slurm_stub(jobname=None, time=None, outputdir=None):
    homedir = Path(os.environ["HOME"])

    jobname = "esnb-job" if jobname is None else f"esnb-{jobname}"
    time = "01:00:00" if time is None else str(time)
    outputdir = homedir if outputdir is None else Path(outputdir)

    if not outputdir.exists():
        os.makedirs(outputdir, exist_ok=True)

    output = []
    output.append(f"#SBATCH --job-name={jobname}")
    output.append(f"#SBATCH --time={time}")
    output.append(f"#SBATCH --output={str(outputdir / jobname) + '.out'}")
    output = str("\n").join(output)
    return output


dora_hostname = os.environ.get("ESNB_GFDL_DORA_HOSTNAME", "dora.gfdl.noaa.gov")
dora = is_host_reachable(dora_hostname, port=443)
site = True if ".noaa.gov" in socket.getfqdn() else False
