from dataclasses import dataclass
from typing import Optional, Sequence

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class ChooseHvgArgs:
    """Arguments to choose highly variable genes.

    Attributes:
        number (int): Number of HVGs to pick. Defaults to 2500.
        verbose (bool): display logs? Defaults to False.
    """

    number: int = 2500
    verbose: bool = False


@dataclass
class ModelGeneVariancesArgs:
    """Arguments to model gene variances.

    Attributes:
        block (Optional[Sequence], optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    span: float = 0.3
    num_threads: int = 1
    verbose: bool = False
