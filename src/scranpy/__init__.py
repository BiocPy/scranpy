import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .aggregation import *
from .batch_correction import *
from .clustering import *
from .dimensionality_reduction import *
from .feature_selection import *
from .feature_set_enrichment import *
from .marker_detection import *
from .nearest_neighbors import *
from .normalization import *
from .quality_control import *
from .analyze import *
