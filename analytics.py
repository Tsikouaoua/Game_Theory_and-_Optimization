"""
Simple Analytics Module for Game Theory and Optimisation Assignment
Course: EBC4188 - Game Theory and Optimisation
Authors: Efstratios Gkoltsios and Gera Hooijer
Date: 14 October 2025

Simple comparison between Dijkstra and Bellman-Ford algorithms.
"""

import time
import os
from datetime import datetime


def bellman_ford(graph, s):
    """
    Simple Bellman-Ford algorithm implementation.
    """
    n = graph.number_of_nodes()
    distance = [float('inf')] * n
    distance[s] = 0
    
    # Relax edges n-1 times
    for _ in range(n - 1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
    
    return distance


def simple_comparison(graph_alt, dijkstra_func, s, t, n, output_dir):
    """
    Simple comparison between Dijkstra and Bellman-Ford.
    """
    print("\nRunning simple algorithm comparison...")
    
    # Time Dijkstra
    start_time = time.time()
    dijk_distance, _ = dijkstra_func(graph_alt, s)
    dijk_time = time.time() - start_time
    
    # Time Bellman-Ford
    start_time = time.time()
    bell_distance = bellman_ford(graph_alt, s)
    bell_time = time.time() - start_time
    
    # Generate simple report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"simple_comparison_{n}nodes_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("SIMPLE ALGORITHM COMPARISON\n")
        f.write("=" * 40 + "\n")
        f.write(f"Network Size: {n} nodes\n")
        f.write(f"From node {s} to node {t}\n")
        f.write(f"\n")
        f.write(f"Dijkstra Time: {dijk_time:.4f} seconds\n")
        f.write(f"Bellman-Ford Time: {bell_time:.4f} seconds\n")
        f.write(f"Speed Difference: {bell_time/dijk_time:.1f}x slower\n")
        f.write(f"\n")
        f.write(f"Dijkstra Distance: {dijk_distance[t]:.4f}\n")
        f.write(f"Bellman-Ford Distance: {bell_distance[t]:.4f}\n")
        f.write(f"Same Result: {abs(dijk_distance[t] - bell_distance[t]) < 0.0001}\n")
    
    print(f"Comparison saved to: {filename}")
    print(f"Bellman-Ford was {bell_time/dijk_time:.1f}x slower than Dijkstra")
    
    return filename