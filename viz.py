"""
Visualization module for Game Theory and Optimisation Assignment
Course: EBC4188 - Game Theory and Optimisation
Authors: Efstratios Gkoltsios and Gera Hooijer
Date: 13 October 2025

This module contains visualization functions for displaying the road network
and shortest path results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import math


def draw_graph(graph, predecessors, breakage, n, s=None, t=None):
    """
    This function draws a weighted, directed graph and its corresponding shortest path tree.
    
    Parameters:
    - graph: NetworkX DiGraph - The road network graph
    - predecessors: list - Predecessor list from Dijkstra's algorithm  
    - breakage: list - Breakage percentages at each node
    - n: int - Number of nodes in the graph
    - s: int - Starting node (optional, for highlighting)
    - t: int - Target node (optional, for highlighting)
    """
    from network_handler import handle_network_output
    handle_network_output(graph, predecessors, breakage, n, s, t)


def draw_detailed_graph(graph, predecessors, breakage, n, s=None, t=None):
    """
    Draw detailed graph with breakage percentages for smaller networks (N <= 50).
    """
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    plt.figure(figsize=(12, 10))

    # Create grid layout for city-like visualization
    grid_width = int(math.sqrt(n * 1.2))
    grid_height = int(n / grid_width) + 1
    while grid_width * grid_height < n:
        grid_width += 1
    
    # Create position dictionary for grid layout
    pos = {}
    node_id = 0
    for r in range(grid_height):
        for c in range(grid_width):
            if node_id >= n:
                break
            pos[node_id] = (c, grid_height - r - 1)  # Invert y-axis for better visualization
            node_id += 1
        if node_id >= n:
            break

    # Draw all nodes
    node_colors = []
    for node in graph.nodes():
        if node == s:
            node_colors.append('lightgreen')  # Start node
        elif node == t:
            node_colors.append('lightcoral')  # Target node
        else:
            node_colors.append('lightgray')   # Regular nodes
    
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    # Draw all edges
    nx.draw_networkx_edges(graph, pos, width=1, edge_color='gray', arrows=True,
                           arrowstyle='->', arrowsize=15, alpha=0.6)
    
    # Add breakage values below nodes
    for u, (x, y) in pos.items():
        if breakage[u] < 100.0:  # Node is reachable
            breakage_percent = 100 - breakage[u]  # Calculate actual breakage percentage
            plt.text(x, y - 0.25, f"{round(breakage_percent, 1)}%", size=10, ha='center', color='red')
        else:  # Node is unreachable
            plt.text(x, y - 0.25, "N/A", size=10, ha='center', color='gray')

    # Highlight nodes and edges in the shortest path tree
    edges_in_tree = []
    for u in graph.nodes():
        for v in graph.nodes():
            if u == predecessors[v]:
                edges_in_tree.append((u, v))

    nx.draw_networkx_edges(graph, pos, edgelist=edges_in_tree, width=3, edge_color='coral', arrows=True,
                           arrowstyle='->', arrowsize=20)
    
    # Add edge labels (weights) - only show a subset to avoid cluttering
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    filtered_edge_labels = {}
    for (u, v), weight in edge_labels.items():
        if (u, v) in edges_in_tree:  # Only show weights for shortest path edges
            filtered_edge_labels[(u, v)] = f"{weight:.2f}"
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=filtered_edge_labels, font_size=8)
    
    # Create title with more information
    title = "City Road Network - Egg Transport Optimization\n(Orange edges show optimal path)"
    if s is not None and t is not None:
        title += f"\nFrom node {s} (green) to node {t} (red)"
    
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def draw_large_graph(graph, predecessors, n, s=None, t=None):
    """
    Draw simplified graph for large networks (N > 50) - shows grid layout with all edges and optimal path.
    """
    plt.figure(figsize=(16, 12))
    
    # Create grid layout for city-like visualization (same as detailed graph)
    grid_width = int(math.sqrt(n * 1.2))
    grid_height = int(n / grid_width) + 1
    while grid_width * grid_height < n:
        grid_width += 1
    
    # Create position dictionary for grid layout
    pos = {}
    node_id = 0
    for r in range(grid_height):
        for c in range(grid_width):
            if node_id >= n:
                break
            pos[node_id] = (c, grid_height - r - 1)  # Invert y-axis for better visualization
            node_id += 1
        if node_id >= n:
            break
    
    # Draw all edges (grid lines) first
    nx.draw_networkx_edges(graph, pos, width=0.5, edge_color='lightgray', 
                          arrows=False, alpha=0.6)
    
    # Draw all nodes but smaller and lighter for large graphs
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='white', 
                          edgecolors='gray', linewidths=0.5)
    
    # Find the actual shortest path from s to t
    path = []
    current = t
    while current != -1 and current != s:
        path.append(current)
        current = predecessors[current]
    if current == s:
        path.append(s)
    path.reverse()
    
    # Draw path edges (the optimal route)
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=5, 
                          edge_color='red', arrows=True, arrowstyle='->', arrowsize=25)
    
    # Draw only start and end nodes with special colors
    if s is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[s], node_size=400, 
                              node_color='green', edgecolors='black', linewidths=2)
    if t is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[t], node_size=400, 
                              node_color='red', edgecolors='black', linewidths=2)
    
    # Add labels for all nodes (just numbers)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
    
    title = f"Large City Grid Network ({n} nodes) - Optimal Path Highlighted"
    if s is not None and t is not None:
        title += f"\nFrom node {s} to node {t}"
    
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def draw_simple_graph(graph, title="Road Network"):
    """
    Draw a simple version of the graph without shortest path highlighting.
    
    Parameters:
    - graph: NetworkX DiGraph - The road network graph
    - title: str - Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Use spring layout for simple visualization
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=600, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, width=1, edge_color='gray', arrows=True,
                           arrowstyle='->', arrowsize=15, alpha=0.7)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    filtered_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=filtered_labels, font_size=8)
    
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def print_results(n, s, t, breakage, shortest_path, distance):
    """
    Print results of the shortest path calculation.
    
    Parameters:
    - n: int - Number of nodes
    - s: int - Starting node
    - t: int - Target node  
    - breakage: list - Breakage percentages at each node
    - shortest_path: list - The shortest path from s to t
    - distance: list - Distance values from Dijkstra's algorithm
    """
    print(f"\nResults for {n}-node city network:")
    
    # For very large networks (N > 200), don't show the path route in terminal
    if n <= 200:
        print(f"Shortest path: {' → '.join(map(str, shortest_path))}")
    else:
        print(f"Shortest path: [Path saved to file - too long to display]")
    
    print(f"Total breakage: {round((100 - breakage[t]), 2)}%")
    print(f"Path length: {len(shortest_path)} nodes")
    print(f"Survival rate: {round(breakage[t], 2)}%")