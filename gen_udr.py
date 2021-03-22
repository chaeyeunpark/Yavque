import sys
import numpy as np

def test_simple(edges):
    for idx, edge in enumerate(edges):
        v1, v2 = edge
        if v1 == v2: #self-connection
            return False
        if v1 > v2:
            edges[idx] = (v2, v1) # reordering
    return len(edges) == len(set(edges))

        

if __name__ == '__main__':
    n = int(sys.argv[1])
    p = int(sys.argv[2])

    assert n*p % 2 == 0

    vertices = list(range(n))

    is_simple = False
    while not is_simple:
        vertices_set = vertices*p

        np.random.shuffle(vertices_set)
        edges = [(vertices_set[2*i], vertices_set[2*i+1]) for i in range(n*p//2)]
        is_simple = test_simple(edges)

    adjacent_matrix = np.zeros((n, n), dtype=np.int32)
    for v1, v2 in edges:
        adjacent_matrix[v1,v2] = 1
        adjacent_matrix[v2,v1] = 1
    
    np.save("test.npy", adjacent_matrix)
