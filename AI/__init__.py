"""
``AI``
==============

This package defines source code for AI
"""

from . import (cpi_model)

from .cpi_model import *

__all__ = []
__all__.extend(cpi_model.__all__.copy())  # type: ignore
