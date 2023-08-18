from singlecellexperiment import SingleCellExperiment
from biocframe import BiocFrame

@dataclass
class AnalyzeResults:
    """Class to manage results across all analyis steps."""

    quality_control: qc.RnaQualityControlResults = field(
        default_factory=qc.RnaQualityControlResults
    )
    normalization: norm.NormalizationResults = field(
        default_factory=norm.NormalizationResults
    )
    feature_selection: feat.FeatureSelectionResults = field(
        default_factory=feat.FeatureSelectionResults
    )
    dimensionality_reduction: dimred.DimensionalityReductionResults = field(
        default_factory=dimred.DimensionalityReductionResults
    )
    clustering: clust.ClusteringResults = field(default_factory=clust.ClusteringResults)
    marker_detection: mark.MarkerDetectionResults = field(
        default_factory=mark.MarkerDetectionResults
    )
    nearest_neighbors: nn.NearestNeighborsResults = field(
        default_factory=nn.NearestNeighborsResults
    )

    def __to_sce(self, x: MatrixTypes, assay: str, include_gene_data: bool = False):
        if isinstance(x, TatamiNumericPointer):
            raise ValueError("`TatamiNumericPointer` is not yet supported (for 'x')")

        keep = [not y for y in self.quality_control.qc_filter.tolist()]

        # TODO: need to add logcounts
        sce = SingleCellExperiment(assays={assay: x[:, keep]})

        sce.colData = self.quality_control.qc_metrics[keep,:]
        sce.colData["clusters"] = self.clustering.clusters

        sce.reducedDims = {
            "pca": self.dimensionality_reduction.pca.principal_components,
            "tsne": array(
                [
                    self.dimensionality_reduction.tsne.x,
                    self.dimensionality_reduction.tsne.y,
                ]
            ).T,
            "umap": array(
                [
                    self.dimensionality_reduction.umap.x,
                    self.dimensionality_reduction.umap.y,
                ]
            ).T,
        }

        if include_gene_data is True:
            sce.rowData = self.feature_selection.gene_variances

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

