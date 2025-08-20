import inspect
import logging
import os

from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def _local_cluster(**kwargs):
    sig = inspect.signature(LocalCluster)
    params = sig.parameters
    valid_args = [name for name, param in params.items()]

    ignored_args = []
    for k, v in kwargs.items():
        if k not in valid_args:
            ignored_args.append(k)

    if len(ignored_args) > 0:
        for x in ignored_args:
            del kwargs[x]
        logger.warning(f"Ignoring options for LocalCluster: {ignored_args}")

    cluster = LocalCluster(**kwargs)
    client = Client(cluster)
    logger.info(f"Initializing local dask cluster: {cluster.dashboard_link}")
    return (cluster, client)


def init_dask_cluster(site="local", **kwargs):
    if "portdash" in kwargs.keys():
        kwargs["dashboard_address"] = f":{kwargs['portdash']}"
        del kwargs["portdash"]

    if site == "local":
        cluster, client = _local_cluster(**kwargs)
    elif site == "gfdl_ppan":
        from esnb.sites.gfdl import dask_cluster_ppan

        cluster, client = dask_cluster_ppan(**kwargs)

        if cluster is None:
            logger.warning("An error occured; Falling back to dask LocalCluster")
            cluster, client = _local_cluster(**kwargs)

    else:
        raise ValueError(f"Unrecognized Dask Site: {site}")

    return (cluster, client)


def init_dask_cluster_test(**kwargs):
    options = {
        "site": "gfdl_ppan",
        "walltime": "24:00:00",
        "highmem": True,
        "memory": "48GB",
        "portdash": os.getuid() + 6047,
    }

    options = {**options, **kwargs}

    return init_dask_cluster(**options)
