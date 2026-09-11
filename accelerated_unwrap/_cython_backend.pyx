# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

def apply_q(double[:, ::1] p, double[:, ::1] wx, double[:, ::1] wy):
    cdef Py_ssize_t i, j, rows=p.shape[0], cols=p.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((rows, cols), dtype=np.float64)
    cdef double[:, ::1] o = out
    for i in range(rows):
        for j in range(cols):
            if j < cols-1: o[i,j] += wx[i,j]*(p[i,j+1]-p[i,j])
            if j > 0: o[i,j] -= wx[i,j-1]*(p[i,j]-p[i,j-1])
            if i < rows-1: o[i,j] += wy[i,j]*(p[i+1,j]-p[i,j])
            if i > 0: o[i,j] -= wy[i-1,j]*(p[i,j]-p[i-1,j])
    return out
