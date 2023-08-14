from dataclasses import dataclass, field
from typing import Optional, Sequence

from .._abstract import AbstractOptions
from ..types import validate_object_type
from .run_pca import PcaResult, RunPcaOptions
from .run_tsne import RunTsneOptions, TsneEmbedding
from .run_umap import RunUmapOptions, UmapEmbedding

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class DimensionalityReductionOptions(AbstractOptions):
    """Arguments to run the dimensionality reduction step.

    Attributes:
        run_pca (RunPcaOptions): Arguments to run PCA
            :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.
        run_tsne (RunTsneOptions): Arguments to run t-SNE
            :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.
        run_umap (RunUmapOptions): Arguments to run UMAP
            :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.
    """

    run_pca: RunPcaOptions = field(default_factory=RunPcaOptions)
    run_tsne: RunTsneOptions = field(default_factory=RunTsneOptions)
    run_umap: RunUmapOptions = field(default_factory=RunUmapOptions)

    def __post_init__(self):
        validate_object_type(self.run_pca, RunPcaOptions)
        validate_object_type(self.run_tsne, RunTsneOptions)
        validate_object_type(self.run_umap, RunUmapOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.run_pca.num_threads = num_threads
        self.run_tsne.initialize_tsne.num_threads = num_threads
        self.run_umap.initialize_umap.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.run_pca.verbose = verbose
        self.run_tsne.verbose = verbose
        self.run_tsne.initialize_tsne.verbose = verbose

        self.run_umap.verbose = verbose
        self.run_umap.initialize_umap.verbose = verbose

    def set_seed(self, seed: int = 42):
        """Set seed for RNG.

        Args:
            seed (int, optional): seed for RNG. Defaults to 42.
        """
        self.run_tsne.initialize_tsne.seed = seed
        self.run_umap.initialize_umap.seed = seed

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.run_pca.block = block


@dataclass
class DimensionalityReductionResults:
    """Results of the dimensionality reduction step.

    Attributes:
        pca (PcaResult, optional): Result of
            :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.
        tsne (TsneEmbedding, optional): Result of
            :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.
        umap (UmapEmbedding, optional): Result of
            :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.
    """

    pca: Optional[PcaResult] = None
    tsne: Optional[TsneEmbedding] = None
    umap: Optional[UmapEmbedding] = None
