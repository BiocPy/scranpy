from numpy import bool_, float64, int32, ndarray, uint8, uintp, zeros

def create_pointer_array(arrs):
    num = len(arrs)
    output = ndarray((num,), dtype=uintp)

    if isinstance(arrs, list):
        for i in range(num):
            output[i] = arrs[i].ctypes.data
    else:
        i = 0
        for k in arrs:
            output[i] = arrs[k].ctypes.data
            i += 1

    return output

