"""
Network Size Handler for Game Theory and Optimisation Assignment
Course: EBC4188 - Game Theory and Optimisation
Authors: Efstratios Gkoltsios and Gera Hooijer
Date: 14 October 2025

This module handles different visualization and output modes based on network size.
"""

import numpy as np
import scipy.sparse as sp
from datetime import datetime
import os


def ensure_output_directory():
    """
    Create output directory structure if it doesn't exist.
    
    Returns:
    - str: Path to the output directory
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def handle_network_output(graph, predecessors, breakage, n, s, t):
    """
    Handle output based on network size:
    - N <= 50: Detailed visualization
    - 50 < N <= 200: Simplified visualization  
    - N > 200: CSR file generation only
    
    Parameters:
    - graph: NetworkX DiGraph - The road network graph
    - predecessors: list - Predecessor list from Dijkstra's algorithm  
    - breakage: list - Breakage percentages at each node
    - n: int - Number of nodes in the graph
    - s: int - Starting node
    - t: int - Target node
    """
    from viz import draw_detailed_graph, draw_large_graph
    
    if n > 200:
        # For very large graphs, generate CSR file instead of visualization
        generate_csr_file(graph, breakage, n, s, t)
        generate_path_file(predecessors, s, t, n)
    elif n > 50:
        # For large graphs, use simplified visualization
        draw_large_graph(graph, predecessors, n, s, t)
        generate_path_file(predecessors, s, t, n)
    else:
        # For small graphs, use detailed visualization
        draw_detailed_graph(graph, predecessors, breakage, n, s, t)


def generate_csr_file(graph, breakage, n, s, t):
    """
    Generate a CSR (Compressed Sparse Row) file containing the breakage matrix for very large networks.
    
    Parameters:
    - graph: NetworkX DiGraph - The road network graph
    - breakage: list - Breakage percentages at each node
    - n: int - Number of nodes in the graph
    - s: int - Starting node
    - t: int - Target node
    """
    # Ensure output directory exists
    output_dir = ensure_output_directory()
    
    # Create adjacency matrix with breakage weights
    adjacency_matrix = np.zeros((n, n))
    
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        adjacency_matrix[u, v] = weight
    
    # Convert to sparse CSR format
    csr_matrix = sp.csr_matrix(adjacency_matrix)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"breakage_matrix_{n}nodes_{timestamp}.npz")
    
    # Save the CSR matrix and metadata
    np.savez_compressed(filename, 
                       data=csr_matrix.data,
                       indices=csr_matrix.indices, 
                       indptr=csr_matrix.indptr,
                       shape=csr_matrix.shape,
                       start_node=s,
                       target_node=t,
                       breakage_values=breakage)
    
    print(f"CSR file saved as: {filename}")


def generate_path_file(predecessors, s, t, n):
    """
    Generate a text file containing the shortest path nodes.
    
    Parameters:
    - predecessors: list - Predecessor list from Dijkstra's algorithm
    - s: int - Starting node
    - t: int - Target node  
    - n: int - Number of nodes in the graph
    """
    # Ensure output directory exists
    output_dir = ensure_output_directory()
    
    # Find the shortest path from s to t
    path = []
    current = t
    while current != -1 and current != s:
        path.append(current)
        current = predecessors[current]
    if current == s:
        path.append(s)
    path.reverse()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"shortest_path_{n}nodes_{timestamp}.txt")
    
    # Write path to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Shortest Path Analysis\n")
        f.write(f"Network Size: {n} nodes\n")
        f.write(f"Start Node: {s}\n")
        f.write(f"Target Node: {t}\n")
        f.write(f"Path Length: {len(path)} nodes\n")
        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Shortest Path:\n")
        f.write(", ".join(map(str, path)))
    
    print(f"Path file saved as: {filename}")