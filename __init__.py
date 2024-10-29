"""
``mortgage``
==============

This package defines source code for mortgage analysis
"""

from . import (main,
               utills,
               plot_utils,
               my_argparser,
               payback_methods,
               mortgage_toolkit, AI)

from .main import *
from .utills import *
from .plot_utils import *
from .my_argparser import *
from .payback_methods import *
from .mortgage_toolkit import *
from .AI import *

__all__ = []
__all__.extend(main.__all__.copy())  # type: ignore
__all__.extend(utills.__all__.copy())  # type: ignore
__all__.extend(plot_utils.__all__.copy())  # type: ignore
__all__.extend(my_argparser.__all__.copy())  # type: ignore
__all__.extend(payback_methods.__all__.copy())  # type: ignore
__all__.extend(mortgage_toolkit.__all__.copy())  # type: ignore
__all__.extend(AI.__all__.copy())  # type: ignore
