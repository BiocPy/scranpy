from .build_neighbor_index import NeighborIndex
from .find_nearest_neighbors import NeighborResults

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def is_neighbor_class(x: Any) -> bool:
    """Checks whether `x` is an expected nearest neighbor input.

    Args:
        x (Any): Any object.

    Returns:
        bool: True if `x` is supported.
    """
    return (
        isinstance(x, NeighborIndex)
        or isinstance(x, NeighborResults)
        or isinstance(x, ndarray)
    )
