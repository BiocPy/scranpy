from ..cpphelpers import lib
import numpy as np

class NeighborResults:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        lib.free_neighbor_results(self.ptr)

    def num_cells(self):
        return lib.fetch_neighbor_results_nobs(self.ptr)

    def num_neighbors(self):
        return lib.fetch_neighbor_results_k(self.ptr)

    def get(self, i):
        k = lib.fetch_neighbor_results_k(self.ptr)
        out_d = np.ndarray((k,), dtype=np.float64)
        out_i = np.ndarray((k,), dtype=np.int32)
        lib.fetch_neighbor_results_single(self.ptr, i, out_i.ctypes.data, out_d.ctypes.data)
        return { "index": out_i, "distance": out_d }

    def serialize(self):
        nobs = lib.fetch_neighbor_results_nobs(self.ptr)
        k = lib.fetch_neighbor_results_k(self.ptr)
        out_i = np.ndarray((k, nobs), dtype=np.int32)
        out_d = np.ndarray((k, nobs), dtype=np.float64)
        lib.serialize_neighbor_results(self.ptr, out_i.ctypes.data, out_d.ctypes.data)
        return { "index": out_i, "distance": out_d }
    
    @classmethod
    def unserialize(cls, content):
        idx = content["index"]
        dist = content["distance"]
        ptr = lib.unserialize_neighbor_results(idx.shape[0], idx.shape[1], idx.ctypes.data, dist.ctypes.data)
        return cls(ptr)

def find_nearest_neighbors(idx, k, approximate = True, num_threads = 1):
    ptr = lib.find_nearest_neighbors(idx.ptr, k, num_threads)
    return NeighborResults(ptr)
