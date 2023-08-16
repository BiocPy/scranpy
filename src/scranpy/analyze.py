from typing import Sequence, Union
from functools import singledispatch, singledispatchmethod

from .analyze_live import AnalyzeOptions, AnalyzeResults, __analyze
from .types import is_matrix_expected_type

from singlecellexperiment import SingleCellExperiment
from biocframe import BiocFrame

@singledispatch
def analyze(
    matrix, features: Sequence[str], options: AnalyzeOptions = AnalyzeOptions()
) -> AnalyzeResults:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Arguments:
        matrix (Any): "Count" matrix.
        features (Sequence[str]): Features information for the rows of the matrix.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        options (AnalyzeOptions): Optional analysis parameters.

    Raises:
        NotImplementedError: If ``matrix`` is not an expected type.

    Returns:
        AnalyzeResults: Results from all steps of the scran workflow.
    """
    if is_matrix_expected_type(matrix):
        return __analyze(matrix, features=features, options=options)
    else:
        raise NotImplementedError(
            f"'Analyze' is not supported for objects of class: `{type(matrix)}`"
        )


@analyze.register
def analyze_sce(
    matrix: SingleCellExperiment,
    features: Union[Sequence[str], str],
    assay: str = "counts",
    options: AnalyzeOptions = AnalyzeOptions(),
) -> AnalyzeResults:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Arguments:
        matrix (SingleCellExperiment): A
            :py:class:`singlecellexperiment.SingleCellExperiment` object.
        features (Union[Sequence[str], str]): Features for the rows of
            the matrix.
        block (Union[Sequence, str], optional): Block assignments for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        assay (str): Assay matrix to use for analysis. Defaults to "counts".
        options (AnalyzeOptions): Optional analysis parameters.

    Raises:
        ValueError: If SCE does not contain a ``assay`` matrix.

    Returns:
        AnalyzeResults: Results from various steps.
    """
    if assay not in matrix.assayNames:
        raise ValueError(f"SCE does not contain a '{assay}' matrix.")

    if isinstance(features, str):
        if isinstance(matrix.rowData, BiocFrame):
            features = matrix.rowData.column(features)
        else:
            features = matrix.rowData[features]

    if isinstance(options.block, str):
        if isinstance(matrix.rowData, BiocFrame):
            block = matrix.colData.column(options.block)
        else:
            block = matrix.colData[block]

    return __analyze(matrix.assay("counts"), features=features, options=options)
