"""
Programming Assignment: Shortest Path exercise 11
Course: EBC4188 - Game Theory and Optimisation
Authors: Efstratios Gkoltsios and Gera Hooijer
Date: 13 October 2025

Description:
This program generates a random weighted road network using NetworkX. The weights of
each edge represents the breakage of eggs on a given road. The program calculates the
path from starting node s, to any node in the graph, with the minimum total breakage.
It outputs a visualization of the road network, the minimal total breakage path from s
to t, and the total breakage occurred on this path.

"""
import math
import networkx as nx
import matplotlib.pyplot as plt
import random
from graph_gen import gen_city_grid, make_connected
from viz import draw_graph, print_results

""""
This function generates a random city-like grid graph, with n nodes and edge weights w_ij = [0,1). Next, it 
generates an altered graph, in which the weights are altered by -log(1 - w_ij). Such that the graph can
be used as an input for Dijkstra's algorithm.
It returns graph, graph_altered and n.
"""
def graph_generator(n=25, seed=42):
    # Generate city grid adjacency list
    adj_list = gen_city_grid(n, seed=seed)
    adj_list = make_connected(adj_list)  # Ensure the graph is connected
    
    # Create directed graph from adjacency list
    graph = nx.DiGraph()
    
    # Add nodes
    for i in range(n):
        graph.add_node(i)
    
    # Add edges with random weights between 0 and 0.04 (1/5 of previous range, 1/25 of original)
    random.seed(seed)
    for u in range(n):
        for v in adj_list[u]:
            if u < v:  # Add edge in both directions for undirected behavior, but avoid duplicates
                weight_uv = random.uniform(0.0, 0.0396)  # Further reduced to 1/5: 0.198/5 ≈ 0.0396
                weight_vu = random.uniform(0.0, 0.0396)  # This will give very small breakage %
                graph.add_edge(u, v, weight=weight_uv)
                graph.add_edge(v, u, weight=weight_vu)

    # Create new graph, with altered weights, such that we can solve it as an SP-problem
    graph_alt = nx.DiGraph()
    graph_alt.add_nodes_from(graph.nodes())
    
    for u, v in graph.edges():
        w = graph[u][v]['weight']
        graph_alt.add_edge(u, v, weight=-math.log(1.0 - w))

    n = graph.number_of_nodes()
    return graph, graph_alt, n

""""
This function runs the Dijkstra's algorithm on a graph G, to find the shortest path
from starting node s to any other node.
It takes as input a directed graph G, with positive weights, and a starting node s. And outputs a 
list containing the distances from s to node i and a list containing the predecessor in the shortest
path tree of node i. 
"""
def dijkstra(graph, s):         #Input is graph G=(V,A) and starting vertex s, n # of nodes
    #create list for the distance values of all nodes, and set to infinity except for s
    n = graph.number_of_nodes()
    distance = [1e7]*n
    distance[s] = 0   #Set start vertex to 0, but fix that it is an integer, so s is integer?

    nodes_visited = [False] * n             #set that states True if a node is added to the tree, False otherwise
    predecessor = [-1] * n      #set that contains the predecessor of every node in the tree

    for count in graph.nodes():
        #find the neighbouring vertex with the shortest distance
        min_index, min_dist = shortest_distance(graph, nodes_visited, distance)

        #add the node with the shortest distance to the set
        nodes_visited[min_index] = True

        #update the distances
        for u in range(n):
            if nodes_visited[u] == False and u in graph.successors(min_index) and distance[u] > min_dist + graph[min_index][u]['weight'] :
                distance[u] = min_dist + graph[min_index][u]['weight']
                predecessor[u] = min_index

    return distance, predecessor

"""" 
This function finds the adjacent node, not yet included in the shortest path tree, with the minimum 
distance to the tree.
It takes as input a weighted, directed graph, a list of nodes already visited in the shortest path
tree, and a list of distances between the nodes in the graph. It outputs the index of the node with minimum
distance, and the distance itself.
"""
def shortest_distance (graph, nodes_visited, distance):       #find the shortest distance to next node
    min_dist = 1e7
    min_index = -1                                              #indexes of the nodes start at 0, so we initialize at -1
    for u  in graph.nodes():
        if nodes_visited[u] == False and distance[u] < min_dist :            #node is not yet selected, and it's current distance is smaller than the minimum distance
            min_dist = distance[u]
            min_index = u

    return min_index, min_dist

""""
This function finds the path from start node s, to any node t.
It takes as input a weighted, directed graph, a list of predecessors generated by the Dijkstra's algorithm and
a starting node s and end note t. It outputs a list of the path from s to t. 
"""
def find_path(graph, predecessors, s, t):
    path =[t]
    while path[0] != s:
        for u in graph.nodes():
            if predecessors[path[0]] == u:
                path.insert(0, u)

    return path

""""
This function first generates a random city-like directed graph, representing a road network. On each road, represented 
by an edge, a truck transporting eggs incurs a certain amount of breakage, represented by the edge weights 
w_ij = [0,1). It then alters the weights of the graphs, and uses this as input for Dijkstra's algorithm, to find
the path of minimum total breakage. It outputs a drawing of the roadnetwork, the shortest path tree on the network and
the total breakage at each node.
"""
def main():
    # Get number of nodes from user input
    while True:
        try:
            n = int(input("Enter the number of nodes (N) for the city graph: "))
            if n < 2:
                print("Please enter a number greater than 1.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get start and target nodes from user (optional)
    while True:
        try:
            s = int(input(f"Enter the starting node (0 to {n-1}), or press Enter for default (0): ") or "0")
            if s < 0 or s >= n:
                print(f"Starting node must be between 0 and {n-1}.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            t = int(input(f"Enter the target node (0 to {n-1}), or press Enter for default ({n-1}): ") or str(n-1))
            if t < 0 or t >= n:
                print(f"Target node must be between 0 and {n-1}.")
                continue
            if t == s:
                print("Target node must be different from starting node.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"\nGenerating city graph with {n} nodes...")
    print(f"Start node: {s}, Target node: {t}")
    print("-" * 50)
    
    # Generate the city graph
    graph, graph_alt, n = graph_generator(n=n, seed=42)
    
    # Run Dijkstra's algorithm
    distance, predecessors = dijkstra(graph_alt, s)

    # Find the shortest path from s to t
    shortest_path_tree = find_path(graph, predecessors, s, t)

    # Calculate the breakage at each node (percentage of eggs remaining)
    breakage = [100.0]*n    # Initialize the percentage of eggs at 100%
    breakage[s] = 100.0     # Starting node has 100% eggs
    
    # Calculate breakage for all reachable nodes using the shortest path tree
    # We need to process nodes in order of their distance to ensure proper calculation
    nodes_with_distance = [(distance[u], u) for u in range(n) if distance[u] < 1e7]
    nodes_with_distance.sort()  # Sort by distance to process in correct order
    
    for dist, u in nodes_with_distance:
        if predecessors[u] != -1:  # Node u is reachable and has a predecessor
            pred = predecessors[u]
            # Calculate remaining eggs after traveling from predecessor to u
            breakage[u] = breakage[pred] * (1 - graph[pred][u]['weight'])

    # Draw the graph with the shortest path highlighted
    draw_graph(graph, predecessors, breakage, n, s, t)

    # Print results using the new formatted output
    print_results(n, s, t, breakage, shortest_path_tree, distance)
    
    # Ask user if they want to run analytics comparison
    print("\n" + "="*50)
    run_analytics = input("Do you want to compare Dijkstra vs Bellman-Ford? (y/n): ").lower().strip()
    
    if run_analytics in ['y', 'yes']:
        from analytics import simple_comparison
        from network_handler import ensure_output_directory
        
        output_dir = ensure_output_directory()
        simple_comparison(graph_alt, dijkstra, s, t, n, output_dir)
    else:
        print("Analytics skipped.")
    
    # Ask user if they want to run K-routes analysis
    print("\n" + "="*50)
    run_kroutes = input("Do you want to run K-routes analysis (visiting multiple nodes)? (y/n): ").lower().strip()
    
    if run_kroutes in ['y', 'yes']:
        from kroutes import run_kroutes_analysis
        
        # Get required nodes from user
        print(f"\nEnter nodes you want to visit (including start {s} and end {t}):")
        print("Example: 0,5,10,15,24 (comma-separated)")
        required_input = input("Required nodes: ").strip()
        
        try:
            required_nodes = [int(x.strip()) for x in required_input.split(',')]
            if s not in required_nodes:
                required_nodes.insert(0, s)
            if t not in required_nodes:
                required_nodes.append(t)
            
            # Important: pass the original probability graph to K-Routes.
            # K-Routes converts weights internally using -ln(1-p). Using graph_alt would double-transform.
            run_kroutes_analysis(graph, required_nodes, s, t)
            
        except ValueError:
            print("Invalid input format. Please use comma-separated numbers.")
    else:
        print("K-routes analysis skipped.")

"Runs the main function, to solve the Shortest Path exercise 11"
if __name__ == "__main__":
    main()








