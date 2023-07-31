import numpy as np

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

x = np.random.rand(1000, 100)

# PCs in rows, cells in dimensions
pcs = np.random.rand(50, 1000)  

block_levels = ["A", "B", "C"]
block = []
for i in range(x.shape[1]):
    block.append(block_levels[i % len(block_levels)])
