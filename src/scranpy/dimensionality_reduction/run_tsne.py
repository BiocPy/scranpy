from ..cpphelpers import lib
import numpy as np
from ..nearest_neighbors import find_nearest_neighbors, build_neighbor_index, NeighborResults, NeighborIndex
import copy

class TsneStatus:
    def __init__(self, ptr, coordinates):
        self.ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        lib.free_tsne_status(self.ptr)

    def num_cells(self):
        return lib.fetch_tsne_status_nobs(self.ptr)

    def iteration(self):
        return lib.fetch_tsne_status_iteration(self.ptr)

    def clone(self):
        return TsneStatus(lib.clone_tsne_status(self.ptr))

    def run(self, iteration):
        lib.run_tsne(self.ptr, iteration, self.coordinates.ctypes.data)

    def extract(self):
        # TODO: document whether the dict values here are just views.
        return { "x": self.coordinates[:,0], "y": self.coordinates[:,1] }

def initialize_tsne(x, perplexity = 30, num_threads = 1, seed = 42):
    if not isinstance(x, NeighborResults):
        k = lib.perplexity_to_k(perplexity)
        if not isinstance(x, NeighborIndex):
            x = build_neighbor_index(x)
        x = find_nearest_neighbors(x, k = k, num_threads = num_threads)

    ptr = lib.initialize_tsne(x.ptr, perplexity, num_threads)
    coords = np.ndarray((x.num_cells(), 2), dtype=np.float64, order="C")
    lib.randomize_tsne_start(coords.shape[1], coords.ctypes.data, seed)

    return TsneStatus(ptr, coords)

# TODO: use *kwargs to just pass on arguments to initialize_tsne?
def run_tsne(x, perplexity = 30, num_threads = 1, seed = 42, max_iterations = 500):
    status = initialize_tsne(x, perplexity = perplexity, num_threads = num_threads, seed = seed)
    status.run(max_iterations)

    output = status.extract()
    output["x"] = copy.deepcopy(output["x"]) # is this really necessary?
    output["y"] = copy.deepcopy(output["y"])

    return output
