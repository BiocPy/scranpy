from dataclasses import dataclass
from typing import Literal, Optional, Sequence

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class RunPcaArgs:
    """Arguments for principal components analysis.

    Attributes:
        rank (int): Number of top PC's to compute.
        subset (Mapping, optional): Array specifying which features should be
            retained (e.g., HVGs). This may contain integer indices or booleans.
            Defaults to None, then all features are retained.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        scale (bool, optional): Whether to scale each feature to unit variance.
            Defaults to False.
        block_method (Literal["none", "project", "regress"], optional): How to adjust
            the PCA for the blocking factor.
            - `"regress"` will regress out the factor, effectively performing a PCA on
                the residuals. This only makes sense in limited cases, e.g., inter-block
                differences are linear and the composition of each block is the same.
            - `"project"` will compute the rotation vectors from the residuals but will
                project the cells onto the PC space. This focuses the PCA on
                within-block variance while avoiding any assumptions about the
                nature of the inter-block differences.
            - `"none"` will ignore any blocking factor, i.e., as if `block = null`. Any
                inter-block differences will both contribute to the determination of
                the rotation vectors and also be preserved in the PC space.
                This option is only used if `block` is not `null`.
            Defaults to "project".
        block_weights (bool, optional): Whether to weight each block so that it
            contributes the same number of effective observations to the covariance
            matrix. Defaults to True.
        num_threads (int, optional):  Number of threads to use. Defaults to 1.
        verbose (bool): display logs? Defaults to False.

    Raises:
        ValueError: If `block_method` is not an expected value.
    """

    rank: int = 50
    subset: Optional[Sequence] = None
    block: Optional[Sequence] = None
    scale: bool = False
    block_method: Literal["none", "project", "regress"] = "project"
    block_weights: bool = True
    num_threads: int = 1
    verbose: bool = False

    def __post_init__(self):
        if self.block_method not in ["none", "project", "regress"]:
            raise ValueError(
                '\'block_method\' must be one of "none", "project", "regress"'
                f"provided {self.block_method}"
            )


@dataclass
class InitializeTsneArgs:
    """Arguments to initialize t-SNE.

    Attributes:
        perplexity (int, optional): Perplexity to use when computing neighbor
            probabilities. Defaults to 30.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        seed (int, optional): Seed to use for RNG. Defaults to 42.
        verbose (bool): display logs? Defaults to False.
    """

    perplexity: int = 30
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False


@dataclass
class RunTsneArgs:
    """Arguments to compute t-SNE embeddings.

    Attributes:
        max_iterations (int, optional): Maximum number of iterations. Defaults to 500.
        initialize_tsne (InitializeTsneArgs): Arguments specified by `initialize_tsne`
            function.
        verbose (bool): display logs? Defaults to False.
    """

    max_iterations: int = 500
    initialize_tsne: InitializeTsneArgs = InitializeTsneArgs()
    verbose: bool = False


@dataclass
class InitializeUmapArgs:
    """Arguments to initialize UMAP algorithm.

    Arguments:
        min_dist (float, optional): Minimum distance between points. Defaults to 0.1.
        num_neighbors (int, optional): Number of neighbors to use in the UMAP algorithm.
            Ignored if `input` is a `NeighborResults` object. Defaults to 15.
        num_epochs (int, optional): Number of epochs to run. Defaults to 500.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        seed (int, optional): Seed to use for RNG. Defaults to 42.
        verbose (bool): display logs? Defaults to False.
    """

    min_dist: float = 0.1
    num_neighbors: int = 15
    num_epochs: int = 500
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False


@dataclass
class RunUmapArgs:
    """Arguments to compute UMAP embeddings.

    Attributes:
        initialize_umap (InitializeUmapArgs): Arguments specified by `initialize_umap`
            function.
        verbose (bool): display logs? Defaults to False.
    """

    initialize_umap: InitializeUmapArgs = InitializeUmapArgs()
    verbose: bool = False
