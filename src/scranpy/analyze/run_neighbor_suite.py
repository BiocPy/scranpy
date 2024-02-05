from copy import copy
from typing import Callable, Tuple

from igraph import Graph

from .. import clustering as clust
from .. import dimensionality_reduction as dimred
from .. import nearest_neighbors as nn


def _unserialize_neighbors_before_run(f, serialized, opt):
    nnres = nn.NeighborResults.unserialize(serialized)
    return f(nnres, opt)


def run_neighbor_suite(
    principal_components,
    build_neighbor_index_options: nn.BuildNeighborIndexOptions = nn.BuildNeighborIndexOptions(),
    find_nearest_neighbors_options: nn.FindNearestNeighborsOptions = nn.FindNearestNeighborsOptions(),
    run_umap_options: dimred.RunUmapOptions = dimred.RunUmapOptions(),
    run_tsne_options: dimred.RunTsneOptions = dimred.RunTsneOptions(),
    build_snn_graph_options: clust.BuildSnnGraphOptions = clust.BuildSnnGraphOptions(),
    num_threads: int = 1,
) -> Tuple[Callable, Callable, Graph, int]:
    """Run the suite of nearest neighbor methods together. This builds the index once and re-uses it for all methods.
    Given enough threads, it also runs all post-neighbor-detection functions in parallel, as none of them depend on each
    other.

    Args:
        principal_components:
            Matrix of principal components where rows are cells and columns are PCs.
            Thi is usually produced by :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

        build_neighbor_index_options:
            Optional arguments to pass to
            :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

        find_nearest_neighbors_options:
            Optional arguments to pass to
            :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

        run_umap_options:
            Optional arguments to pass to :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        run_tsne_options:
            Optional arguments to pass to :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        build_snn_graph_options:
            Optional arguments to pass to :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        num_threads:
            Number of threads to use for the parallel execution of UMAP, t-SNE and SNN graph construction.
            This overrides the specified number of threads in ``run_umap``, ``run_tsne`` and ``build_snn_graph``.

    Returns:
        A tuple containing, in order:
        - A function that takes no arguments and returns a tuple containing the t-SNE and UMAP coordinates.
        - The shared nearest neighbor graph from :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.
        - The number of remaining threads.

        The idea is that the number of remaining threads can be used to perform tasks on the main thread
        (e.g., clustering, marker detection) while the t-SNE and UMAP are still being computed;
        once all tasks on the main thread have completed, the first function can be called to obtain the coordinates.
    """

    index = nn.build_neighbor_index(
        principal_components,
        options=build_neighbor_index_options,
    )

    tsne_nn = dimred.tsne_perplexity_to_neighbors(
        run_tsne_options.initialize_tsne.perplexity
    )
    umap_nn = run_umap_options.initialize_umap.num_neighbors
    snn_nn = build_snn_graph_options.num_neighbors

    nn_dict = {}
    for k in set([umap_nn, tsne_nn, snn_nn]):
        nn_dict[k] = nn.find_nearest_neighbors(
            index,
            k=k,
            options=find_nearest_neighbors_options,
        )

    serialized_dict = {}
    for k in set([umap_nn, tsne_nn]):
        serialized_dict[k] = nn_dict[k].serialize()

    # Attempting to evenly distribute threads across the tasks. t-SNE and UMAP
    # are run on separate processes while the SNN graph construction is kept on
    # the main thread because we'll need the output for marker detection.
    threads_per_task = max(1, int(num_threads / 3))
    max_workers = min(2, num_threads)
    _tasks = []

    run_tsne_copy = copy(run_tsne_options)
    run_tsne_copy.set_threads(threads_per_task)

    run_umap_copy = copy(run_umap_options)
    run_umap_copy.set_threads(threads_per_task)

    if max_workers > 1:
        import multiprocessing as mp
        import platform
        from concurrent.futures import ProcessPoolExecutor, wait

        pp = platform.platform()
        extra_args = {}
        if "macos" in pp.lower():
            extra_args["mp_context"] = mp.get_context("fork")

        executor = ProcessPoolExecutor(max_workers=max_workers, **extra_args)

        _tasks.append(
            executor.submit(
                _unserialize_neighbors_before_run,
                dimred.run_tsne,
                serialized_dict[tsne_nn],
                run_tsne_copy,
            )
        )

        _tasks.append(
            executor.submit(
                _unserialize_neighbors_before_run,
                dimred.run_umap,
                serialized_dict[umap_nn],
                run_umap_copy,
            )
        )

        def retrieve():
            wait(_tasks)
            executor.shutdown()

        def get_tsne():
            retrieve()
            return _tasks[0].result()

        def get_umap():
            retrieve()
            return _tasks[1].result()

    else:
        _tasks.append(
            _unserialize_neighbors_before_run(
                dimred.run_tsne,
                serialized_dict[tsne_nn],
                run_tsne_copy,
            )
        )

        _tasks.append(
            _unserialize_neighbors_before_run(
                dimred.run_umap,
                serialized_dict[umap_nn],
                run_umap_copy,
            )
        )

        def get_tsne():
            return _tasks[0]

        def get_umap():
            return _tasks[1]

    build_snn_graph_copy = copy(build_snn_graph_options)
    remaining_threads = max(1, num_threads - threads_per_task * 2)
    build_snn_graph_copy.set_threads(remaining_threads)
    graph = clust.build_snn_graph(nn_dict[snn_nn], options=build_snn_graph_copy)

    return get_tsne, get_umap, graph, remaining_threads
