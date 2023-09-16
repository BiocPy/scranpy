from typing import Sequence, Union, Optional
from functools import singledispatch

from .AnalyzeOptions import AnalyzeOptions
from .AnalyzeResults import AnalyzeResults
from .live_analyze import live_analyze
from .dry_analyze import dry_analyze

from singlecellexperiment import SingleCellExperiment
from biocframe import BiocFrame
from .._utils import MatrixTypes


@singledispatch
def analyze(
    rna_matrix: Optional[MatrixTypes],
    adt_matrix: Optional[MatrixTypes] = None,
    crispr_matrix: Optional[MatrixTypes] = None,
    options: AnalyzeOptions = AnalyzeOptions(),
    dry_run: bool = False,
) -> Union[AnalyzeResults, str]:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster

    Arguments:
        rna_matrix (MatrixTypes, optional): Count matrix for RNA data.

        adt_matrix (MatrixTypes, optional): Count matrix for the ADT data.

        crispr_matrix (MatrixTypes, optional): Count matrix for the CRISPR data.

        options (AnalyzeOptions, optional): Optional analysis parameters.

        dry_run (bool, optional): Whether to perform a dry run.

    Raises:
        NotImplementedError: If ``matrix`` is not an expected type.

    Returns:
        If ``dry_run = False``, a :py:class:`~scranpy.analyze.AnalyzeResults.AnalyzeResults` object is returned
        containing... well, the analysis results, obviously.

        If ``dry_run = True``, a string is returned containing all the steps required to perform the analysis.
    """
    if dry_run:
        return dry_analyze(options)
    else:
        return live_analyze(
            rna_matrix=rna_matrix, 
            adt_matrix=adt_matrix, 
            crispr_matrix=crispr_matrix, 
            options=options,
        )


@analyze.register
def analyze_sce(
    matrix: SingleCellExperiment,
    features: Union[Sequence[str], str],
    assay: str = "counts",
    options: AnalyzeOptions = AnalyzeOptions(),
    dry_run: bool = False,
) -> Union[AnalyzeResults, str]:
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
        If ``dry_run = False``, a :py:class:`~scranpy.analyze.AnalyzeResults.AnalyzeResults` object is returned
        containing... well, the analysis results, obviously.

        If ``dry_run = True``, a string is returned containing all the steps required to perform the analysis.
    """
    if assay not in matrix.assayNames:
        raise ValueError(f"SCE does not contain a '{assay}' matrix.")

    if isinstance(features, str):
        if isinstance(matrix.row_data, BiocFrame):
            features = matrix.row_data.column(features)
        else:
            features = matrix.row_data[features]

    return analyze(
        matrix.assay("counts"), features=features, options=options, dry_run=dry_run
    )
