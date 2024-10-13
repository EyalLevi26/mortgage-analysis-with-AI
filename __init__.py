"""
``mortgage``
==============

This package defines source code for mortgage analysis
"""

from . import (main,
               payback_methods,
               format_obj, AI)

from .main import *
from .payback_methods import *
from .format_obj import *
from .AI import *

__all__ = []
__all__.extend(main.__all__.copy())  # type: ignore
__all__.extend(payback_methods.__all__.copy())  # type: ignore
__all__.extend(format_obj.__all__.copy())  # type: ignore
__all__.extend(AI.__all__.copy())  # type: ignore