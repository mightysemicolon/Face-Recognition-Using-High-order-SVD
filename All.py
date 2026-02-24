import numpy as np

def fold(matrix, shape, mode):

    m, n, d = shape
    q, r = matrix.shape
    tensor = np.empty(shape)

    if mode == 1:

        slice = range(0, r + 1, d)
        for i in range(n):
            b, e = slice[i], slice[i + 1]
            tensor[:, i, :] = matrix[:, b : e]
    
    elif mode == 2:

        slice = range(0, r + 1, m)
        for i in range(d):
            b, e = slice[i], slice[i + 1]
            tensor[:, :, i] = matrix[:, b : e].T
    
    else:
        
        slice = range(0, r + 1, n)
        for i in range(m):
            b, e = slice[i], slice[i + 1]
            tensor[i, :, :] = matrix[:, b : e].T
    
    return tensor


def unfold(tensor, mode):

    m, n, d = tensor.shape

    if mode == 1:

        return np.concatenate([tensor[:, i, :] for i in range(n)], axis=1)
    
    elif mode == 2:

        return np.concatenate([tensor[:, :, i].T for i in range(d)], axis=1)
    
    else:

        return np.concatenate([tensor[i, :, :].T for i in range(m)], axis=1)

def Mul_1(matrix, tensor):

    q, r = matrix.shape
    m, n, d = tensor.shape # unfold (m, n * d)
    res = np.empty((q, n * d))

    res = np.concatenate([np.dot(matrix, tensor[:, i, :]) for i in range(n)], axis=1)
    return fold(res, (q, n, d), 1)

def Mul_2(matrix, tensor):

    q, r = matrix.shape
    m, n, d = tensor.shape 
    res = np.empty((q, m * d))

    res = np.concatenate([np.dot(matrix, tensor[:, :, i].T) for i in range(d)], axis=1)
    return fold(res, (m, q, d), 2)

def Mul_3(matrix, tensor):

    q, r = matrix.shape
    m, n, d = tensor.shape
    res = np.empty((q, m * n))

    res = np.concatenate([np.dot(matrix, tensor[i, :, :].T) for i in range(m)], axis=1)
    return fold(res, (m, n, q), 3)