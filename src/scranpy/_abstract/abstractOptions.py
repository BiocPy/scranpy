from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class AbstractStepOptions:
    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        pass

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        pass

    def set_seed(self, seed: int = 42):
        """Set seed for RNG.

        Args:
            seed (int, optional): seed for RNG. Defaults to 42.
        """
        pass

    def set_block(self, block: Optional[Sequence] = None):
        """Set block assignments for each cell.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        pass

    def set_subset(self, subset: Optional[Mapping] = None):
        """Set subsets.

        Args:
            subset (Mapping, optional): Set subsets. Defaults to None.
        """
        pass
