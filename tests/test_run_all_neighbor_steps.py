import numpy
import scranpy
import knncolle


def test_run_all_neighbor_steps_basic():
    x = numpy.random.randn(10, 1000)
    res = scranpy.run_all_neighbor_steps(x, num_threads=2)

    umap_ref = scranpy.run_umap(x)
    assert (umap_ref == res.run_umap).all()
    umap_single = scranpy.run_all_neighbor_steps(x, run_tsne_options=None, cluster_graph_options=None)
    assert (umap_ref == umap_single.run_umap).all()

    tsne_ref = scranpy.run_tsne(x)
    assert (tsne_ref == res.run_tsne).all()
    tsne_single = scranpy.run_all_neighbor_steps(x, run_umap_options=None, cluster_graph_options=None)
    assert (tsne_ref == tsne_single.run_tsne).all()

    graph = scranpy.build_snn_graph(x)
    clustering = scranpy.cluster_graph(graph)
    assert (clustering.membership == res.cluster_graph.membership).all()
    cluster_single = scranpy.run_all_neighbor_steps(x, run_tsne_options=None, run_umap_options=None)
    assert (clustering.membership == cluster_single.cluster_graph.membership).all()


def test_run_all_neighbor_steps_collapsed():
    x = numpy.random.randn(10, 1000)
    nnp = knncolle.VptreeParameters()
    res = scranpy.run_all_neighbor_steps(x, nn_parameters=nnp, num_threads=2)
    collapsed = scranpy.run_all_neighbor_steps(x, collapse_search=True, nn_parameters=nnp, num_threads=2)
    assert (res.cluster_graph.membership == collapsed.cluster_graph.membership).all()
    assert (res.run_tsne == collapsed.run_tsne).all()
    assert (res.run_umap == collapsed.run_umap).all()
