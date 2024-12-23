from typing import Optional

import numpy

from . import lib_scranpy as lib


def choose_highly_variable_genes(
    stats: numpy.ndarray,
    top: int = 4000,
    larger: bool = True,
    keep_ties: bool = True,
    bound: Optional[float] = None
) -> numpy.ndarray:
    """Choose highly variable genes (HVGs), typically based on a
    variance-related statistic.

    Args:
        stats:
            Array of variances (or a related statistic) across all genes,
            Typically the residuals from
            :py:func:`~model_gene_variances.model_gene_variances` used here.

        top:
            Number of top genes to retain. Note that the actual number of
            retained genes may not be equal to ``top``, depending on the
            other options.

        larger:
            Whether larger values of ``stats`` represent more variable genes.
            If true, HVGs are defined from the largest values of ``stats``.

        keep_ties: 
            Whether to keep ties at the ``top``-th most variable gene. This
            avoids arbitrary breaking of tied values.

        bound:
            The lower bound (if ``larger = True``) or upper bound (otherwise)
            to be applied to ``stats``. Genes are not considered to be HVGs if
            they do not pass this bound, even if they are within the ``top``
            genes. Ignored if ``None``.

    Returns:
        Array containing the indices of genes in ``stats`` that are
        considered to be highly variable.
    """
    local_s = numpy.array(stats, dtype=numpy.float64, copy=None)
    return lib.choose_highly_variable_genes(
        local_s,
        min(top, len(local_s)), # protect against top=Inf when casting to 'int' in pybind11.
        larger,
        keep_ties,
        bound
    )
