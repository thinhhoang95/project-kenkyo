import networkx as nx
import numpy as np

def get_edge_flow(G, traversed_edge_list):
    """
    Computes the edge-flow vector based on traversed edges and the reference orientation.

    Parameters:
    - G (networkx.Graph): A NetworkX graph with nodes labeled from 1 to N.
    - traversed_edge_list (List[Tuple[int, int]]): 
        A list of tuples representing traversed edges with direction, 
        where each tuple is (u, v) indicating traversal from node u to node v.

    Returns:
    - f (numpy.ndarray): A 1D NumPy array of size M (number of edges in G) representing the edge-flow vector.
        f[i] = 1 if the i-th edge is traversed in the reference orientation,
               -1 if traversed in the opposite direction,
               0 otherwise.

    Raises:
    - ValueError: If a traversed edge does not exist in the graph G.
    """
    # Number of vertices
    N = G.number_of_nodes()
    
    # Assign unique indices to edges, sorted in ascending order
    edge_list = sorted([tuple(sorted(edge)) for edge in G.edges()])  # (u, v) with u < v
    M = len(edge_list)
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
    
    # Initialize the edge-flow vector
    f = np.zeros(M, dtype=int)
    
    # Process each traversed edge
    for traversal in traversed_edge_list:
        if not isinstance(traversal, (list, tuple)) or len(traversal) != 2:
            raise ValueError(f"Each traversed edge must be a tuple of two nodes. Invalid traversal: {traversal}")
        
        u, v = traversal
        a, b = sorted((u, v))  # Reference orientation
        
        edge = (a, b)
        edge_idx = edge_to_idx.get(edge)
        
        if edge_idx is None:
            raise ValueError(f"Traversed edge ({u}, {v}) does not exist in the graph G.")
        
        if (u, v) == edge:
            f[edge_idx] += 1  # Traversed in reference orientation
        else:
            f[edge_idx] -= 1  # Traversed in opposite direction
    
    return f
