import networkx as nx
import numpy as np

def compute_incidence_matrices_B(G, k_max=2):
    """
    Computes the incidence matrices B1 and B2 for a given NetworkX graph G with maximum simplex dimension k_max=2.
    
    Parameters:
    - G (networkx.Graph): A NetworkX graph with nodes labeled from 1 to N.
    - k_max (int): Maximum simplex dimension (default is 2).
    
    Returns:
    - B1 (numpy.ndarray): Vertex-to-edge incidence matrix of size N x M.
    - B2 (numpy.ndarray): Edge-to-triangle incidence matrix of size M x T.
    
    Raises:
    - ValueError: If k_max is not between 1 and 2.
    """
    if k_max < 1 or k_max > 2:
        raise ValueError("k_max must be between 1 and 2")
    
    # Number of vertices
    N = G.number_of_nodes()
    
    # Assign unique indices to edges, sorted in ascending order
    edge_list = sorted([tuple(sorted(edge)) for edge in G.edges()]) # the tail is always smaller than the head
    M = len(edge_list)
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
    
    # Construct B1: N x M
    B1 = np.zeros((N, M), dtype=int)
    for j, (u, v) in enumerate(edge_list):
        B1[u-1, j] = -1  # Tail of the edge
        B1[v-1, j] = 1   # Head of the edge
    
    # Find all triangles (cliques of size 3)
    triangles = [tuple(sorted(clique)) for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    T = len(triangles)
    triangle_to_idx = {triangle: idx for idx, triangle in enumerate(triangles)}
    
    # Construct B2: M x T
    if T > 0:
        B2 = np.zeros((M, T), dtype=int)
        for t_idx, (v1, v2, v3) in enumerate(triangles): # t_idx is the index of the triangle
            # Define the edges of the triangle based on sorted vertex order
            e1 = (v1, v2)
            e2 = (v2, v3)
            e3 = (v1, v3)
            
            # Retrieve edge indices
            j1 = edge_to_idx.get(e1)
            j2 = edge_to_idx.get(e2)
            j3 = edge_to_idx.get(e3)
            
            # Assign entries in B2
            if j1 is not None:
                B2[j1, t_idx] = 1
            if j2 is not None:
                B2[j2, t_idx] = 1
            if j3 is not None:
                B2[j3, t_idx] = -1
    else:
        # If no triangles, return an empty matrix with appropriate dimensions
        B2 = np.zeros((M, 0), dtype=int)
    
    return B1, B2

def compute_hodge_laplacian_decomp(B1, B2):
    L1 = B1.T @ B1 + B2 @ B2.T
    # Get the eigenvalues and eigenvectors of L1
    eigenvalues, eigenvectors = np.linalg.eig(L1)
    return eigenvectors
