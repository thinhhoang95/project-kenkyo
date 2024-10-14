import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt

def delaunay_partition_graph(points, plot=False):
    """
    Performs Delaunay triangulation on a set of 2D points and generates a partition graph.

    Parameters:
    - points (array-like): An array of shape (n_points, 2) representing the 2D coordinates.
    - plot (bool): If True, plots the points and the Delaunay triangulation.

    Returns:
    - G (networkx.Graph): A graph where nodes are point indices and edges represent Delaunay edges.
    """
    # Convert input to NumPy array
    points = np.array(points)
    
    if points.shape[1] != 2:
        raise ValueError("Only 2D points are supported.")

    # Perform Delaunay triangulation
    delaunay = Delaunay(points)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with point coordinates as attributes
    for idx, (x, y) in enumerate(points):
        G.add_node(idx, pos=(x, y))
    
    # Add edges based on simplices (triangles)
    for simplex in delaunay.simplices:
        edges = [(simplex[i], simplex[j]) for i in range(3) for j in range(i+1, 3)]
        G.add_edges_from(edges)

    if plot:
        plot_delaunay(delaunay, points)
    
    return G, delaunay

def plot_delaunay(delaunay, points):
    plt.figure(figsize=(8, 6))
    plt.triplot(points[:,0], points[:,1], delaunay.simplices, color='gray', linestyle='--')
    plt.plot(points[:,0], points[:,1], 'o', color='blue')
    for idx, (x, y) in enumerate(points):
        plt.text(x, y, str(idx), color="red", fontsize=8, alpha=0.8)
    plt.title('Delaunay Triangulation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def plot_delaunay_with_flow(delaunay, points, flows, G = None):
    
    plt.figure(figsize=(8, 6))
    plt.triplot(points[:,0], points[:,1], delaunay.simplices, color='gray', linestyle='--', linewidth=0.5)
    plt.plot(points[:,0], points[:,1], 'x', color='blue', markersize=2)
    for idx, (x, y) in enumerate(points):
        plt.text(x, y, str(idx), color="red", fontsize=8, alpha=0.8)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(flows)))
    for flow, color in zip(flows, colors):
        for u, v in flow:
            plt.arrow(points[u][0], points[u][1], points[v][0] - points[u][0], points[v][1] - points[u][1],
                  head_width=0.02, head_length=0.03, fc=color, ec=color)
        # Check if the edge is in the graph
        if G is not None:
            for u, v in flow:
                if not G.has_edge(u, v):
                    raise ValueError(f"Edge {u} - {v} is not in the graph")
    plt.title('Flows on Delaunay Triangulation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()