from ..nearest_neighbors import build_neighbor_index, find_nearest_neighbors, NeighborResults, NeighborIndex
from ..cpphelpers import lib
import igraph as ig
import ctypes as ct
import numpy as np
from copy import deepcopy

def build_snn_graph(x, num_neighbors = 10, approximate = True, weight_scheme = "ranked", num_threads = 1):
    graph = None
    scheme = weight_scheme.encode("UTF-8")

    if not isinstance(x, NeighborResults):
        if not isinstance(x, NeighborIndex):
            x = build_neighbor_index(x, approximate = approximate)
        built = lib.build_snn_graph_from_nn_index(x.ptr, num_neighbors, scheme, num_threads)
    else:
        built = lib.build_snn_graph_from_nn_results(x.ptr, scheme, num_threads)

    try:
        nedges = lib.fetch_snn_graph_edges(built)
        idx_pointer = ct.cast(lib.fetch_snn_graph_indices(built), ct.POINTER(ct.c_int))
        idx_array = np.ctypeslib.as_array(idx_pointer, shape=(nedges* 2,))
        w_pointer = ct.cast(lib.fetch_snn_graph_weights(built), ct.POINTER(ct.c_double))
        w_array = np.ctypeslib.as_array(w_pointer, shape=(nedges,))

        edge_list = []
        for i in range(nedges):
            edge_list.append((idx_array[2 * i], idx_array[2* i + 1]))

        nc = x.num_cells()
        graph = ig.Graph(n = nc, edges = edge_list)
        graph.es["weight"] = deepcopy(w_array)

    finally:
        lib.free_snn_graph(built)

    return graph
