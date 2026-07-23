"""
This is a basic doctest demonstrating that the package and pydra can both be successfully
imported.

>>> import pydra.compose.monai
"""

try:
    from ._version import __version__
except ImportError:
    raise RuntimeError(
        "Pydra package 'pydra-compose-monai' has not been installed, please use "
        "`pip install -e <path-to-repo>` to install development version"
    )

from .builder import define
from .fields import arg, out
from .spec_parser import spec_fragment
from .task import MonaiTask as Task, MonaiOutputs as Outputs

__all__ = ["arg", "out", "define", "spec_fragment", "Task", "Outputs", "__version__"]
