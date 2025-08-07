import importlib.metadata as ilm
import logging
import os
import sys

from . import core, datasources, engine, nbtools, sites
from .core.CaseExperiment import CaseExperiment
from .core.CaseExperiment2 import CaseExperiment2
from .core.CaseGroup import CaseGroup
from .core.CaseGroup2 import CaseGroup2
from .core.esnb_datastore import esnb_datastore
from .core.NotebookDiagnostic import NotebookDiagnostic
from .core.RequestedVariable import RequestedVariable

__all__ = [
    "core",
    "datasources",
    "engine",
    "sites",
    "CaseExperiment",
    "CaseExperiment2",
    "CaseGroup",
    "CaseGroup2",
    "esnb_datastore",
    "NotebookDiagnostic",
    "RequestedVariable",
    "nbtools",
]

msg = ilm.metadata("esnb")

__name__ = msg["Name"]
__version__ = msg["Version"]
__description__ = msg["Summary"]
__requires__ = msg["Requires-Dist"]
__requires_python__ = msg["Requires-Python"]


logger = logging.getLogger(__name__)


# ANSI color codes
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname}{RESET}"
        record.msg = f"{log_color}{record.msg}{RESET}"
        return super().format(record)


# Create logger
log_level = os.environ.get("ESNB_LOG_LEVEL", "WARNING")
if log_level == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif log_level == "INFO":
    logger.setLevel(logging.INFO)
elif log_level == "WARNING":
    logger.setLevel(logging.WARNING)
elif log_level == "ERROR":
    logger.setLevel(logging.ERROR)
elif log_level == "CRITICAL":
    logger.setLevel(logging.CRITICAL)
else:
    raise ValueError(f"Unrecognized logging level: {log_level}")

logger.handlers = []  # Clear existing handlers

# StreamHandler to stdout
handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.WARNING)

# Include module and line number in format string
# format_str = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
format_str = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
formatter = ColorFormatter(format_str)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

# test_data_root = str(
#    Path(importlib.util.find_spec("esnb").origin).parent.parent.parent
#    / "tests/test_data"
# )
