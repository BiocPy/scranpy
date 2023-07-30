from ..cpphelpers import lib
import numpy as np
from ..nearest_neighbors import find_nearest_neighbors, build_neighbor_index, NeighborResults, NeighborIndex
import copy

class UmapStatus:
    def __init__(self, ptr, coordinates):
        self.ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        lib.free_umap_status(self.ptr)

    def num_cells(self):
        return lib.fetch_umap_status_nobs(self.ptr)

    def epoch(self):
        return lib.fetch_umap_status_epoch(self.ptr)

    def num_epochs(self):
        return lib.fetch_umap_status_num_epochs(self.ptr)

    def clone(self):
        cloned = copy.deepcopy(self.coordinates)
        return UmapStatus(lib.clone_umap_status(self.ptr, cloned.ctypes.data), cloned)

    def run(self, epoch_limit = None):
        if epoch_limit is None:
            epoch_limit = 0; # i.e., until the end.
        lib.run_umap(self.ptr, epoch_limit)

    def extract(self):
        # TODO: document whether the dict values here are just views.
        return { "x": self.coordinates[:,0], "y": self.coordinates[:,1] }

def initialize_umap(x, min_dist = 0.1, num_neighbors = 15, num_threads = 1, num_epochs = 500, seed = 42):
    if not isinstance(x, NeighborResults):
        if not isinstance(x, NeighborIndex):
            x = build_neighbor_index(x)
        x = find_nearest_neighbors(x, k = num_neighbors, num_threads = num_threads)

    coords = np.ndarray((x.num_cells(), 2), dtype=np.float64, order="C")
    ptr = lib.initialize_umap(x.ptr, num_epochs, min_dist, coords.ctypes.data, num_threads)

    return UmapStatus(ptr, coords)

# TODO: use *kwargs to just pass on arguments to initialize_umap?
def run_umap(x, min_dist = 0.1, num_neighbors = 15, num_threads = 1, num_epochs = 500, seed = 42):
    status = initialize_umap(x, min_dist = min_dist, num_neighbors = num_neighbors, num_epochs = num_epochs, num_threads = num_threads, seed = seed)
    status.run()

    output = status.extract()
    output["x"] = copy.deepcopy(output["x"]) # is this really necessary?
    output["y"] = copy.deepcopy(output["y"])

    return output
