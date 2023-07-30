from ..cpphelpers import lib

class NeighborIndex:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        lib.free_neighbor_index(ptr)

    def num_cells(self):
        return lib.fetch_neighbor_index_nobs(self.ptr)

    def num_dimensions(self):
        return lib.fetch_neighbor_index_ndim(self.ptr)

def build_neighbor_index(x, approximate = True):
    ptr = lib.build_neighbor_index(x.shape[0], x.shape[1], x.ctypes.data, approximate)
    return NeighborIndex(ptr)
