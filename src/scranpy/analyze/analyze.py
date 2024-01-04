from typing import Optional, Union

from singlecellexperiment import SingleCellExperiment
from summarizedexperiment import SummarizedExperiment

from .AnalyzeOptions import AnalyzeOptions
from .AnalyzeResults import AnalyzeResults
from .dry_analyze import dry_analyze
from .live_analyze import live_analyze
from .update import update


def analyze(
    rna_matrix,
    adt_matrix=None,
    crispr_matrix=None,
    options: AnalyzeOptions = AnalyzeOptions(),
    dry_run: bool = False,
) -> Union[AnalyzeResults, str]:
    """Run a routine analysis workflow for single-cell datasets. This supports
    RNA-seq, ADT and CRISPR modalities for the same cells. Steps include:

    - Removal of low-quality cells
    - Normalization and log-transformation
    - Variance modelling across genes
    - PCA on highly variable genes
    - Combined embeddings across modalities
    - Batch correction with MNN
    - Graph-based clustering
    - Dimensionality reductions for visualization
    - Marker detection for each cluster

    Arguments:
        rna_matrix:
            Count matrix for RNA data.
            Alternatively None if no RNA data is available.

        adt_matrix:
            Count matrix for the ADT data.
            Alternatively None if no ADT data is available.

        crispr_matrix:
            Count matrix for the CRISPR data.
            Alternatively None if no CRISPR data is available.

        options:
            Optional analysis parameters.

        dry_run:
            Whether to perform a dry run.

    Raises:
        NotImplementedError:
            If ``matrix`` is not an expected type.

    Returns:
        If ``dry_run = False``, a :py:class:`~scranpy.analyze.AnalyzeResults.AnalyzeResults` object is returned
        containing... well, the analysis results, obviously.

        If ``dry_run = True``, a string is returned containing all the steps required to perform the analysis.
    """
    if dry_run:
        return dry_analyze(
            rna_matrix=rna_matrix,
            adt_matrix=adt_matrix,
            crispr_matrix=crispr_matrix,
            options=options,
        )
    else:
        return live_analyze(
            rna_matrix=rna_matrix,
            adt_matrix=adt_matrix,
            crispr_matrix=crispr_matrix,
            options=options,
        )


def analyze_se(
    rna_se: Optional[SummarizedExperiment],
    adt_se: Optional[SummarizedExperiment] = None,
    crispr_se: Optional[SummarizedExperiment] = None,
    assay_type: Union[str, int] = 0,
    options: AnalyzeOptions = AnalyzeOptions(),
    dry_run: bool = False,
) -> Union[AnalyzeResults, str]:
    """Convenience wrapper around :py:meth:`~scranpy.analyze.analyze.analyze` for
    :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` inputs.

    Arguments:
        rna_se:
            SummarizedExperiment containing RNA data.
            Alternatively None if no RNA data is available.

        adt_se:
            SummarizedExperiment containing ADT data.
            Alternatively None if no ADT data is available.

        crispr_se:
            SummarizedExperiment containing CRISPR data.
            Alternatively None if no CRISPR data is available.

        assay_type:
            Assay containing the count data in each SummarizedExperiment.

        options:
            Optional analysis parameters.

        dry_run:
            Whether to perform a dry run.

    Raises:
        NotImplementedError:
            If ``matrix`` is not an expected type.

    Returns:
        If ``dry_run = False``, a :py:class:`~scranpy.analyze.AnalyzeResults.AnalyzeResults` object is returned
        containing... well, the analysis results, obviously.

        If ``dry_run = True``, a string is returned containing all the steps required to perform the analysis.
    """

    def exfil(se):
        if se is not None:
            return se.assay(assay_type), se.row_names
        else:
            return None, None

    rna_matrix, rna_features = exfil(rna_se)
    adt_matrix, adt_features = exfil(adt_se)
    crispr_matrix, crispr_features = exfil(crispr_se)

    return analyze(
        rna_matrix,
        adt_matrix=adt_matrix,
        crispr_matrix=crispr_matrix,
        options=update(
            options,
            miscellaneous_options=update(
                options.miscellaneous_options,
                rna_feature_names=rna_features,
                adt_feature_names=adt_features,
                crispr_feature_names=crispr_features,
            ),
        ),
        dry_run=dry_run,
    )


def analyze_sce(
    sce: SingleCellExperiment,
    rna_exp: Optional[Union[str, int]] = "",
    adt_exp: Optional[Union[str, int]] = None,
    crispr_exp: Optional[Union[str, int]] = None,
    assay_type: str = "counts",
    options: AnalyzeOptions = AnalyzeOptions(),
    dry_run: bool = False,
) -> Union[AnalyzeResults, str]:
    """Convenience wrapper around :py:meth:`~scranpy.analyze.analyze.analyze` for
    :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` inputs.

    Arguments:
        sce:
            A :py:class:`singlecellexperiment.SingleCellExperiment` object,
            possibly with data from other modalities in its alternative experiments.

        rna_exp:
            String or index specifying the alternative experiment containing the RNA data.
            An empty string is assumed to refer to the main experiment.
            If None, we assume that no RNA data is available.

        adt_exp:
            String or index specifying the alternative experiment containing the ADT data.
            An empty string is assumed to refer to the main experiment.
            If None, we assume that no RNA data is available.

        crispr_exp:
            String or index specifying the alternative experiment containing the CRISPR data.
            An empty string is assumed to refer to the main experiment.
            If None, we assume that no RNA data is available.

        assay_type:
            Assay containing the count data in each SummarizedExperiment.

        options:
            Optional analysis parameters.

        dry_run:
            Whether to perform a dry run.

    Raises:
        ValueError:
            If SCE does not contain a ``assay`` matrix.

    Returns:
        If ``dry_run = False``, a :py:class:`~scranpy.analyze.AnalyzeResults.AnalyzeResults` object is returned
        containing... well, the analysis results, obviously.

        If ``dry_run = True``, a string is returned containing all the steps required to perform the analysis.
    """

    def exfil(sce, exp):
        if exp is None:
            return None
        elif exp == "":
            return sce
        else:
            return sce.alternative_experiments[exp]

    return analyze_se(
        rna_se=exfil(sce, rna_exp),
        adt_se=exfil(sce, adt_exp),
        crispr_se=exfil(sce, crispr_exp),
        assay_type=assay_type,
        options=options,
        dry_run=dry_run,
    )
