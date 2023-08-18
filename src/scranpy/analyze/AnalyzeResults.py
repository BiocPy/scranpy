from dataclasses import dataclass, field
from functools import singledispatch, singledispatchmethod

from typing import Optional, Mapping
from singlecellexperiment import SingleCellExperiment
from biocframe import BiocFrame
from numpy import ndarray, array
from igraph import Graph
from mattress import TatamiNumericPointer

from .. import clustering as clust
from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import quality_control as qc
from ..types import MatrixTypes

@dataclass
class AnalyzeResults:
    """Results across all analyis steps from :py:meth:`~scranpy.analyze.analyze.analyze`.

    Attributes:
        rna_quality_control_metrics (BiocFrame, optional): 
            Output of :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`.

        rna_quality_control_thresholds (BiocFrame, optional): 
            Output of :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.

        rna_quality_control_filter (ndarray, optional): 
            Output of :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

        size_factors (ndarray, optional):
            Array of length equal to the number of cells in the dataset (usually after quality filtering),
            containing the size factor for each cell.

        gene_variances (BiocFrame, optional):
            Output of :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

        hvgs (ndarray, optional):
            Output of :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

        pca_output (PcaResult, optional):
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

        tsne_output (TsneEmbedding, optional):
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        umap_output (UmapEmbedding, optional):
            Output of :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.
    
        snn_graph (Graph, optional):
            Output of :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        clusters (List, optional):
            List of length equal to the number of cells in the (filtered) dataset, 
            containing the cluster assignment for each cell.
            
        markers (Mapping, optional):
            Output of :py:meth:`~scranpy.marker_detection.score_markers.score_markers`.
    """

    rna_quality_control_subsets: Optional[dict] = None

    rna_quality_control_metrics: Optional[BiocFrame] = None

    rna_quality_control_thresholds: Optional[BiocFrame] = None

    rna_quality_control_filter: Optional[ndarray] = None

    size_factors: Optional[ndarray] = None

    gene_variances: Optional[BiocFrame] = None

    hvgs: Optional[ndarray] = None

    pca_output: Optional[dimred.PcaResult] = None

    tsne_output: Optional[dimred.TsneEmbedding] = None

    umap_output: Optional[dimred.UmapEmbedding] = None

    snn_graph: Optional[Graph] = None

    clusters: Optional[list] = None

    markers: Optional[Mapping] = None

    def __to_sce(self, x: MatrixTypes, assay: str, include_gene_data: bool = False):
        if isinstance(x, TatamiNumericPointer):
            raise ValueError("`TatamiNumericPointer` is not yet supported (for 'x')")

        keep = [not y for y in self.rna_quality_control_filter.tolist()]

        # TODO: need to add logcounts
        sce = SingleCellExperiment(assays={assay: x[:, keep]})

        sce.colData = self.rna_quality_control_metrics[keep,:]
        sce.colData["clusters"] = self.clusters

        sce.reducedDims = {
            "pca": self.pca_out.principal_components,
            "tsne": array(
                [
                    self.tsne_out.x,
                    self.tsne_out.y,
                ]
            ).T,
            "umap": array(
                [
                    self.umap_out.x,
                    self.umap_out.y,
                ]
            ).T,
        }

        if include_gene_data is True:
            sce.rowData = self.gene_variances

        return sce

    @singledispatchmethod
    def to_sce(
        self, x, assay: str = "counts", include_gene_data: bool = False
    ) -> SingleCellExperiment:
        """Save results as a :py:class:`singlecellexperiment.SingleCellExperiment`.

        Args:
            x: Input object. usually a matrix of raw counts.
            assay (str, optional): assay name for the matrix.
                Defaults to "counts".
            include_gene_data (bool, optional): Whether to include gene variances.
                Defaults to False.

        Returns:
            SingleCellExperiment: An SCE with the results.
        """
        return self.__to_sce(x, assay, include_gene_data)

    @to_sce.register
    def _(
        self,
        x: SingleCellExperiment,
        assay: str = "counts",
        include_gene_data: bool = False,
    ) -> SingleCellExperiment:
        if assay not in x.assayNames:
            raise ValueError(f"SCE does not contain a '{assay}' matrix.")

        mat = x.assay(assay)
        return self.__to_sce(mat, assay, include_gene_data)

