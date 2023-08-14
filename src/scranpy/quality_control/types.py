from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from biocframe import BiocFrame

from ..types import validate_object_type
from .rna import PerCellRnaQcMetricsOptions, SuggestRnaQcFiltersOptions, CreateRnaQcFilterOptions 

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class RnaQualityControlOptions:
    """Optional arguments to perform quality control on RNA data.

    Attributes:
        per_cell_rna_qc_metrics (PerCellRnaQcMetricsOptions): Optional arguments to pass to
            (:py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`).
        suggest_rna_qc_filters (SuggestRnaQcFiltersOptions): Optional arguments to pass to
            :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
        create_rna_qc_filter (CreateRnaQcFilterOptions): Optional arguments to pass to
            (:py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`).
        mito_subset (Union[str, bool], optional): subset mitochondrial genes.
    """

    per_cell_rna_qc_metrics: PerCellRnaQcMetricsOptions = field(
        default_factory=PerCellRnaQcMetricsOptions
    )
    suggest_rna_qc_filters: SuggestRnaQcFiltersOptions = field(
        default_factory=SuggestRnaQcFiltersOptions
    )
    create_rna_qc_filter: CreateRnaQcFilterOptions = field(
        default_factory=CreateRnaQcFilterOptions
    )
    mito_subset: Optional[Union[str, int]] = None
    custom_thresholds: Optional[BiocFrame] = None

    def __post_init__(self):
        validate_object_type(self.per_cell_rna_qc_metrics, PerCellRnaQcMetricsOptions)
        validate_object_type(self.suggest_rna_qc_filters, SuggestRnaQcFiltersOptions)
        validate_object_type(self.create_rna_qc_filter, CreateRnaQcFilterOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.per_cell_rna_qc_metrics.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.per_cell_rna_qc_metrics.verbose = verbose
        self.suggest_rna_qc_filters.verbose = verbose
        self.create_rna_qc_filter.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.suggest_rna_qc_filters.block = block
        self.create_rna_qc_filter.block = block

    def set_subset(self, subset: Optional[Mapping] = None):
        """Set subsets.

        Args:
            subset (Mapping, optional): Set subsets. Defaults to None.
        """
        self.per_cell_rna_qc_metrics.subsets = subset


@dataclass
class RnaQualityControlResults:
    """Results of RNA QC step.

    Attributes:
        per_cell_rna_qc_metrics (BiocFrame, optional): Result of
            :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`.
        suggest_rna_qc_filters (BiocFrame, optional): Result of
            :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
        create_rna_qc_filter (np.ndarray, optional): Result of create qc filter
            (:py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`)
        subsets (Mapping, optional): Subsets.
        filtered_cells (np.ndarray, optional): Result of
            :py:meth:`~scranpy.quality_control.filter_cells.filter_cells`.
    """

    qc_metrics: Optional[BiocFrame] = None
    qc_filter: Optional[np.ndarray] = None
    qc_thresholds: Optional[BiocFrame] = None
    subsets: Optional[Mapping] = None
    filtered_cells: Optional[np.ndarray] = None
