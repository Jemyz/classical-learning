import random
import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt


def create_vectors(dim, no_vectors):
    vectors = []
    for i in range(no_vectors):
        vector = []
        for j in range(dim):
            vector.append(random.random())
        vectors.append(vector)
    return vectors


if __name__ == '__main__':
    dim = 2
    no_vectors = 10000
    vectors = np.array(create_vectors(dim, no_vectors))
    space_diagonal_vector = np.ones(dim)

    dist = np.linalg.norm(vectors, axis=1)
    dist_mean = dist.mean()

    angles = np.array([])
    for vector in vectors:
        cosang = np.dot(vector, space_diagonal_vector)
        sinang = la.norm(np.cross(vector, space_diagonal_vector))
        angles = np.append(angles, math.degrees(np.arctan2(sinang, cosang)))

    angles_mean = angles.mean()

    print("expected value of L = %s" % dist_mean)
    print("expected value of alpha = %s" % angles_mean)

    plt.hist(dist)
    plt.title("Histogram of Distances")
    plt.show()

    plt.hist(angles)
    plt.title("Histogram of Angles")
    plt.show()

    vectors = vectors[np.random.choice(vectors.shape[0], 2000,replace=False)]

    from scipy.spatial import distance_matrix

    distance_matrix = distance_matrix(vectors, vectors)
    distance_matrix_mean = distance_matrix.mean()

    print("expected value of E = %s" % distance_matrix_mean)

    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    plt.title('Distance Matrix')
    plt.show()
