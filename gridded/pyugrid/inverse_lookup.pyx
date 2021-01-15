from cython.view cimport array as cvarray


def lookup(int[:] a, int[:] b, int k=2, int fill_value=-999):
    """Look up the elements of b in a

    This function takes two arrays, the reference `b` and the target `a`. For
    every element in `b`, this method looks up where we can find it in`a`.

    :params a: The target array
    :type a: 1D array of type int (i.e. np.int32)

    :params b: The reference array
    :type b: 1D array of type int (i.e. np.int32)

    :params k: The maximal number of for one single element of `b` in `a`
    :type k: int

    :params fill_value: Values in `a` that should be ignored. Commonly, some
                        value like -999 or len(b)
    :type fill_value: int

    :returns: A 2D array of type int with shape (len(b), k)
    """
    cdef int Na = len(a), Nb = len(b)
    cdef int[:, :] ret = cvarray(shape=(Nb, k), itemsize=sizeof(int), format="i")

    cdef int[:] counter =  cvarray(shape=(Nb,), itemsize=sizeof(int), format="i")
    cdef int ia, ib, count

    counter[:] = 0
    ret[:, :] = Na

    for ia in xrange(Na):
        ib = a[ia]
        if ib != fill_value:
            count = counter[ib]
            ret[ib, count] = ia
            counter[ib] = counter[ib] + 1
    return ret
