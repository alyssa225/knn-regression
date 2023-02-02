import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    assert X.shape[1] == Y.shape[1]
    edist = np.ndarray((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            e = np.ndarray((X.shape[1],))
            for k in range(X.shape[1]):
                e[k] = (X[i,k]-Y[j,k])**2
            edist[i,j] = np.sqrt(np.sum(e))
    return edist



def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    assert X.shape[1] == Y.shape[1]
    mdist = np.ndarray((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            m = np.ndarray((X.shape[1],))
            for k in range(X.shape[1]):
                m[k] = abs(X[i,k]-Y[j,k])
            mdist[i,j] = np.sum(m)
    return mdist


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    assert X.shape[1] == Y.shape[1]
    cdist = np.ndarray((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            c = np.ndarray((X.shape[1],3))
            for k in range(X.shape[1]):
                c[k,0] = X[i,k]*Y[j,k]
                c[k,1] = X[i,k]**2
                c[k,2] = Y[j,k]**2
            cdist[i,j] = 1-(np.sum(c[:,0])/(np.sqrt(np.sum(c[:,1]))*np.sqrt(np.sum(c[:,2]))))
    return cdist
