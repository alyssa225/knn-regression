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
            edist[i,j] = np.sqrt(np.sum((X[i,:]-Y[j,:])**2))
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
            mdist[i,j] = np.sum(abs(X[i,:]-Y[j,:]))  
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
            cdist[i,j] = 1-(np.sum(np.matmul(X[i,:],Y[j,:]))/(np.sqrt(np.sum(X[i,:]**2))*np.sqrt(np.sum(Y[j,:]**2))+0.000000001))
    return cdist
