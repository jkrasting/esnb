import json
import logging
from pathlib import Path

import intake_esm
import intake_esm.core

logger = logging.getLogger(__name__)

__all__ = ["infer_case_source"]


def infer_case_source(source):
    """
    Infers the type of a given case source and returns its mode.

    Parameters
    ----------
    source : str or int
        The source to be inferred. This can be a string representing a Dora ID,
        a URL, a local file path, or a project-level Dora ID, an integer
        representing a Dora ID, or an intake_esm catalog object.

    Returns
    -------
    mode : str
        The inferred mode of the source. Possible values include:
        - 'dictionary': Dict-object with MDTF settings key conventions
        - 'dora_id': Dora ID (numeric or project-level)
        - 'dora_url': URL pointing to Dora
        - 'intake_url': URL suggesting an intake catalog
        - 'catalog_object': Raw intake esm catalog object
        - 'path': Local file path
        - 'pp_dir': Directory containing post-processing (raises NotImplementedError)
        - 'intake_path': Local JSON file assumed to be an intake catalog
        - 'mdtf_settings': Local file assumed to be an MDTF settings file

    Raises
    ------
    ValueError
        If the source type is unsupported.
    FileNotFoundError
        If the provided path does not exist.
    NotImplementedError
        If the supplied path is a directory (future support planned).
    """
    if isinstance(source, dict):
        logger.debug("Found source dictionary -- assuming MDTF settings convention")
        mode = "dictionary"

    elif isinstance(source, str):
        if source.isnumeric():
            logger.debug(f"Found source string with numeric Dora ID - {source}")
            mode = "dora_id"
        elif source.startswith("http") or source.startswith("https"):
            if "dora.gfdl" in source:
                logger.debug(f"Found source url pointing to Dora - {source}")
                mode = "dora_url"
            else:
                logger.debug(f"Found source url suggesting intake catalog - {source}")
                mode = "intake_url"
        elif "-" in source:
            if source.split("-")[1].isnumeric():
                logger.debug(
                    f"Found source string with project-level dora ID - {source}"
                )
                mode = "dora_id"
            else:
                mode = "path"
        else:
            mode = "path"
    elif isinstance(source, int):
        logger.debug(f"Found source integer suggesting dora ID - {source}")
        mode = "dora_id"
    elif isinstance(source, intake_esm.core.esm_datastore):
        mode = "catalog_object"
    else:
        source_type = type(source)
        raise ValueError(
            f"Unsupported source type: {source_type}. Must be path or url to"
            + " intake_esm catalog, MDTF settings file, or DORA ID"
        )

    if mode == "path":
        logger.debug(f"Assuming source is a local file path - {source}")
        filepath = Path(source)
        if not filepath.exists():
            logger.error(f"Path {filepath} does not exist")
            raise FileNotFoundError(f"Path {filepath} does not exist")
        if filepath.is_dir():
            mode = "pp_dir"
            logger.debug(
                "Supplied path appears to be a directory, possibly containing post-processing"
            )
            err = "The supplied path is a directory. In the future, support will be added to generate a catalog."
            logger.error(err)
            raise NotImplementedError(err)
        else:
            try:
                with open(filepath, "r") as f:
                    json.load(f)
                logger.debug(
                    "Source appears to be a JSON file, assuming intake catalog"
                )
                mode = "intake_path"
            except json.JSONDecodeError:
                logger.debug("Source is not a JSON file, assuming MDTF settings file")
                mode = "mdtf_settings"

    return mode
