from numba import prange, njit
import numpy as np

@njit
def seq_i(i,seq):
    '''
    Given sequence of letters and index of sequence of coef. r_i for counting lambda.
    Returns coef. r_i. (https://en.wikipedia.org/wiki/Lyapunov_fractal#Algorithm)
    '''
    return seq[i%(len(seq))]

@njit
def f(p, seq, N):
    '''
    Calculating lambda like in formula from wiki,
    but calculating x_i while calculating sum.
    Parametrs:
        p - coor-s of point (x,y,z)
        seq - sequence of letters 
        N - number of iterations
    Returns:
        lambda for point p
    '''
    s = 0 
    prev = 0.5 # x_(i-1)
    curr = 0   # x_(i)
    for i in range(1,N):
        r_curr = p[seq_i(i,seq)]
        r_prev = p[seq_i(i-1,seq)]
        curr = r_prev * prev * (1 - prev)
        if(abs(r_curr*(1 - 2*curr))!=0):
            s += np.log(abs(r_curr*(1 - 2*curr)))
        prev = curr
    return s/N

@njit(cache=False, parallel=True)
def fractal(x0, x1, y0, y1, z, h, w, seq, N):
    '''
    Filling net[w,h] with lambdas.
    Parametrs:
        x0, x1, y0, y1 - bounds of net
        z - we will picture slice on that z coord
        h, w - size of net
        N - number of iterations
    Returns:
        net[w,h] of lambdas
    '''
    value = np.zeros((h,w), dtype=np.double)
    for j in prange(w):
        for i in range(h):
            value[j,i] = f((x0 + i*((x1-x0)/h), y0 + j*((y1-y0)/w), z), seq, N)
    return value