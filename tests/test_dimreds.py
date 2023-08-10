from scranpy.dimensionality_reduction import (
    InitializeTsneArgs,
    InitializeUmapArgs,
    RunTsneArgs,
    RunUmapArgs,
    TsneEmbedding,
    UmapEmbedding,
    run_tsne,
    run_umap,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_run_tsne(mock_data):
    y = mock_data.pcs
    out = run_tsne(input=y)

    assert isinstance(out, TsneEmbedding)
    assert out.x.shape[0] == out.y.shape[0]
    assert out.x.shape[0] == y.shape[1]

    # Same results with multiple threads.
    outp = run_tsne(
        input=y, options=RunTsneArgs(initialize_tsne=InitializeTsneArgs(num_threads=3))
    )
    assert (out.x == outp.x).all()
    assert (out.y == outp.y).all()


def test_run_umap(mock_data):
    y = mock_data.pcs
    out = run_umap(input=y)

    assert isinstance(out, UmapEmbedding)
    assert out.x.shape[0] == out.y.shape[0]
    assert out.x.shape[0] == y.shape[1]

    # Same results with multiple threads.
    outp = run_umap(
        input=y, options=RunUmapArgs(initialize_umap=InitializeUmapArgs(num_threads=3))
    )
    assert (out.x == outp.x).all()
    assert (out.y == outp.y).all()
