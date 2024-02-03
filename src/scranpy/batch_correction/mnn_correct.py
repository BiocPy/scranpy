from dataclasses import dataclass
from typing import Optional, Sequence

from numpy import float64, int32, ndarray, zeros

from .. import _cpphelpers as lib
from .._utils import factorize


@dataclass
class MnnCorrectOptions:
    """Options to pass to :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.

    Attributes:
        k:
            Number of neighbors for detecting mutual nearest neighbors.

        approximate:
            Whether to perform an approximate nearest neighbor search.

        order:
            Ordering of batches to correct. The first
            entry is used as the initial reference, and all subsequent batches
            are merged and added to the reference in the specified order.
            This should contain all unique levels in the ``batch`` argument
            supplied to :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.
            If None, an appropriate ordering is automatically determined.

        reference_policy:
            Policy to use for choosing the initial reference
            batch. This can be one of "max-rss" (maximum residual sum of squares
            within the batch, which is the default), "max-variance" (maximum
            variance within the batch), "max-size" (maximum number of cells),
            or "input" (using the supplied order of levels in ``batch``).
            Only used if ``order`` is not supplied.

        num_mads:
            Number of median absolute deviations, used to define
            the threshold for outliers when computing the center of mass for
            each cell involved in a MNN pair. Larger values reduce kissing but
            may incorporate inappropriately distant subpopulations in a cell's center of mass.

        mass_cap:
            Cap on the number of observations used to compute the
            center of mass for each MNN-involved observation.  The dataset is
            effectively downsampled to `c` observations for this calculation, which
            improves speed at the cost of some precision.

        num_threads:
            Number of threads to use for the various MNN calculations.
    """

    k: int = 15
    approximate: bool = True
    order: Optional[Sequence] = None
    reference_policy: str = "max-rss"
    num_mads: int = 3
    mass_cap: int = -1
    num_threads: int = 1


@dataclass
class MnnCorrectResult:
    """Results from :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.

    Attributes:
        corrected:
            Matrix of corrected coordinates for each cell (row) and dimension (column).
            Rows and columns should be in the same order as the input ``x`` in
            :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.

        merge_order:
            Order of batches used for merging.
            The first batch is used as the initial reference.
            The length of this list is equal to the number of batches.

        num_pairs:
            Number of MNN pairs detected at each merge step.
            This has length one less than the number of batches.
    """

    corrected: Optional[ndarray] = None

    merge_order: Optional[list] = None

    num_pairs: Optional[ndarray] = None


def mnn_correct(
    x: ndarray, batch: Sequence, options: MnnCorrectOptions = MnnCorrectOptions()
) -> MnnCorrectResult:
    """Identify mutual nearest neighbors (MNNs) to correct batch effects in a low-dimensional embedding.

    Args:
        x:
            Numeric matrix where rows are cells and columns are dimensions,
            typically generated from :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

        batch:
            Sequence of length equal to the number of cells (i.e., rows of ``x``),
            specifying the batch for each cell.

        options:
            Optional parameters.

    Returns:
        The corrected coordinates for each cell, along with some diagnostics
        about the MNNs involved.
    """
    ndim = x.shape[1]
    ncells = x.shape[0]

    batch_levels, batch_indices = factorize(batch)
    if len(batch_indices) != ncells:
        raise ValueError("length of 'batch' should be equal to number of rows of 'x'")
    nbatch = len(batch_levels)

    order_info = None
    order_offset = 0
    if options.order is not None:
        mapping = {}
        for i, lev in enumerate(batch_levels):
            mapping[lev] = i

        if len(options.order) != len(batch_levels):
            raise ValueError(
                "length of 'options.order' should be equal to the number of batches"
            )

        order_info = zeros((nbatch,), dtype=int32)
        for i, o in enumerate(options.order):
            if o not in mapping:
                raise ValueError(
                    "'options.order' should contain the same values as 'batch'"
                )
            curo = mapping[o]
            if curo < 0:
                raise ValueError("'options.order' should not contain duplicate values")
            order_info[i] = curo
            mapping[o] = -1

        order_offset = order_info.ctypes.data

    corrected_output = zeros((ncells, ndim), dtype=float64)
    merge_order_output = zeros((nbatch,), dtype=int32)
    num_pairs_output = zeros((nbatch - 1,), dtype=int32)

    lib.mnn_correct(
        ndim=ndim,
        ncells=ncells,
        x=x,
        nbatches=nbatch,
        batch=batch_indices,
        k=options.k,
        nmads=options.num_mads,
        nthreads=options.num_threads,
        mass_cap=options.mass_cap,
        use_order=order_info is not None,
        order=order_offset,
        ref_policy=options.reference_policy.encode("UTF-8"),
        approximate=options.approximate,
        corrected_output=corrected_output,
        merge_order_output=merge_order_output,
        num_pairs_output=num_pairs_output,
    )

    return MnnCorrectResult(
        corrected=corrected_output,
        merge_order=[batch_levels[i] for i in merge_order_output],
        num_pairs=num_pairs_output,
    )
