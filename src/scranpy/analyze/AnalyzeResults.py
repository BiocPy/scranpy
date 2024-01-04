from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Mapping, Optional, Sequence

from biocframe import BiocFrame
from delayedarray import DelayedArray
from igraph import Graph
from numpy import array, log, log1p, ndarray
from singlecellexperiment import SingleCellExperiment

from .. import batch_correction as correct
from .. import dimensionality_reduction as dimred


@dataclass
class AnalyzeResults:
    """Results across all analyis steps from :py:meth:`~scranpy.analyze.analyze.analyze`.

    Attributes:
        rna_quality_control_metrics:
            Output of :py:meth:`~scranpy.quality_control.per_cell_rna_qc_metrics.per_cell_rna_qc_metrics`.

        rna_quality_control_thresholds:
            Output of :py:meth:`~scranpy.quality_control.suggest_rna_qc_filters.suggest_rna_qc_filters`.

        rna_quality_control_filter:
            Output of :py:meth:`~scranpy.quality_control.create_rna_qc_filter.create_rna_qc_filter`.

        adt_quality_control_metrics:
            Output of :py:meth:`~scranpy.quality_control.per_cell_adt_qc_metrics.per_cell_adt_qc_metrics`.

        adt_quality_control_thresholds:
            Output of :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.

        adt_quality_control_filter:
            Output of :py:meth:`~scranpy.quality_control.create_adt_qc_filter.create_adt_qc_filter`.

        crispr_quality_control_metrics:
            Output of :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`.

        crispr_quality_control_thresholds:
            Output of :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.

        crispr_quality_control_filter:
            Output of :py:meth:`~scranpy.quality_control.create_crispr_qc_filter.create_crispr_qc_filter`.

        quality_control_retained:
            Array of length equal to the number of cells in the dataset before quality filtering,
            indicating whether each cell should be retained.

        rna_size_factors:
            Array of length equal to the number of cells in the dataset after quality filtering,
            containing the size factor from the RNA data for each cell.

        adt_size_factors:
            Array of length equal to the number of cells in the dataset after quality filtering,
            containing the size factor from the ADT data for each cell.

        crispr_size_factors:
            Array of length equal to the number of cells in the dataset after quality filtering,
            containing the size factor from the CRISPR data for each cell.

        gene_variances:
            Output of :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

        hvgs:
            Output of :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

        rna_pca:
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca` on the RNA data.

        adt_pca:
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca` on the ADT data.

        crispr_pca:
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca` on the CRISPR data.

        combined_pcs:
            Output of :py:meth:`~scranpy.dimensionality_reduction.combine_embeddings.combine_embeddings`
            on the principal components for multiple modalities.

        mnn:
            Output of :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.

        tsne:
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        umap:
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        snn_graph:
            Output of :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        clusters:
            List of length equal to the number of cells in the (filtered) dataset,
            containing the cluster assignment for each cell.

        rna_markers:
            Output of :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            on the RNA data.

        adt_markers:
            Output of :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            on the ADT data.

        crispr_markers:
            Output of :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            on the CRISPR data.
    """

    rna_quality_control_metrics: Optional[BiocFrame] = None

    rna_quality_control_thresholds: Optional[BiocFrame] = None

    rna_quality_control_filter: Optional[ndarray] = None

    adt_quality_control_metrics: Optional[BiocFrame] = None

    adt_quality_control_thresholds: Optional[BiocFrame] = None

    adt_quality_control_filter: Optional[ndarray] = None

    crispr_quality_control_metrics: Optional[BiocFrame] = None

    crispr_quality_control_thresholds: Optional[BiocFrame] = None

    crispr_quality_control_filter: Optional[ndarray] = None

    quality_control_retained: Optional[ndarray] = None

    rna_size_factors: Optional[ndarray] = None

    adt_size_factors: Optional[ndarray] = None

    crispr_size_factors: Optional[ndarray] = None

    gene_variances: Optional[BiocFrame] = None

    hvgs: Optional[ndarray] = None

    rna_pca: Optional[dimred.PcaResult] = None

    adt_pca: Optional[dimred.PcaResult] = None

    crispr_pca: Optional[dimred.PcaResult] = None

    combined_pcs: Optional[ndarray] = None

    mnn: Optional[correct.MnnCorrectResult] = None

    tsne: Optional[dimred.TsneEmbedding] = None

    umap: Optional[dimred.UmapEmbedding] = None

    snn_graph: Optional[Graph] = None

    clusters: Optional[list] = None

    rna_markers: Optional[Mapping] = None

    adt_markers: Optional[Mapping] = None

    crispr_markers: Optional[Mapping] = None

    def __to_sce(
        self,
        rna_matrix: Optional[Any],
        rna_features: Optional[Sequence[str]],
        adt_matrix: Optional[Any],
        adt_features: Optional[Sequence[str]],
        crispr_matrix: Optional[Any],
        crispr_features: Optional[Sequence[str]],
    ) -> SingleCellExperiment:
        keep = self.quality_control_retained.tolist()
        main_sce = None

        if rna_matrix is not None:
            y = DelayedArray(rna_matrix)
            filtered = y[:, keep]
            normalized = log1p(filtered / self.rna_size_factors) / log(2)
            rna_sce = SingleCellExperiment(
                assays={"counts": filtered, "logcounts": normalized}
            )
            rna_sce.row_names = rna_features
            rna_sce.column_data = self.rna_quality_control_metrics[keep, :]
            rna_sce.column_data["size_factors"] = self.rna_size_factors
            rna_sce.row_data = self.gene_variances
            rna_sce.reduced_dims = {"pca": self.rna_pca.principal_components}
            main_sce = rna_sce
            main_sce.main_experiment_name = "rna"

        if adt_matrix is not None:
            y = DelayedArray(adt_matrix)
            filtered = y[:, keep]
            normalized = log1p(filtered / self.adt_size_factors) / log(2)
            adt_sce = SingleCellExperiment(
                assays={"counts": filtered, "logcounts": normalized}
            )
            adt_sce.row_names = adt_features
            adt_sce.column_data = self.adt_quality_control_metrics[keep, :]
            adt_sce.column_data["size_factors"] = self.adt_size_factors
            adt_sce.reduced_dims = {"pca": self.adt_pca.principal_components}
            if main_sce is None:
                main_sce = adt_sce
                main_sce.main_experiment_name = "adt"
            else:
                if main_sce.alternative_experiments is None:
                    main_sce.alternative_experiments = {}
                main_sce.alternative_experiments["adt"] = adt_sce

        if crispr_matrix is not None:
            y = DelayedArray(crispr_matrix)
            filtered = y[:, keep]
            normalized = log1p(filtered / self.crispr_size_factors) / log(2)
            crispr_sce = SingleCellExperiment(
                assays={"counts": filtered, "logcounts": normalized}
            )
            crispr_sce.row_names = crispr_features
            crispr_sce.column_data = self.crispr_quality_control_metrics[keep, :]
            crispr_sce.column_data["size_factors"] = self.crispr_size_factors
            crispr_sce.reduced_dims = {"pca": self.crispr_pca.principal_components}
            if main_sce is None:
                main_sce = crispr_sce
                main_sce.main_experiment_name = "crispr"
            else:
                if main_sce.alternative_experiments is None:
                    main_sce.alternative_experiments = {}
                main_sce.alternative_experiments["crispr"] = crispr_sce

        main_sce.col_data["clusters"] = self.clusters

        if self.combined_pcs is not None:
            main_sce.reduced_dims["combined"] = self.combined_pcs
        if self.mnn is not None:
            main_sce.reduced_dims["mnn"] = self.mnn.corrected

        main_sce.reduced_dims["tsne"] = array([self.tsne.x, self.tsne.y]).T
        main_sce.reduced_dims["umap"] = (array([self.umap.x, self.umap.y]).T,)

        return main_sce

    @singledispatchmethod
    def to_sce(
        self,
        rna_matrix: Optional[Any],
        rna_features: Optional[Sequence[str]],
        adt_matrix: Optional[Any] = None,
        adt_features: Optional[Sequence[str]] = None,
        crispr_matrix: Optional[Any] = None,
        crispr_features: Optional[Sequence[str]] = None,
    ) -> SingleCellExperiment:
        """Save results as a :py:class:`singlecellexperiment.SingleCellExperiment`.

        Args:
            x: 
                Input object. usually a matrix of raw counts.
            
            assay: 
                Assay name for the matrix.
                Defaults to "counts".

        Returns:
            An SCE with the results.
        """
        return self.__to_sce(
            rna_matrix=rna_matrix,
            rna_features=rna_features,
            adt_matrix=adt_matrix,
            adt_features=adt_features,
            crispr_matrix=crispr_matrix,
            crispr_features=crispr_features,
        )

    @to_sce.register
    def _(
        self,
        x: SingleCellExperiment,
        assay: str = "counts",
        include_gene_data: bool = False,
    ) -> SingleCellExperiment:
        if assay not in x.assay_names:
            raise ValueError(f"SCE does not contain a '{assay}' matrix.")

        mat = x.assay(assay)
        return self.__to_sce(mat, assay, include_gene_data)
