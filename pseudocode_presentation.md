# Game Theory and Optimization - Egg Transport Network Analysis
# Pseudocode for Presentation

## 1. OVERALL SYSTEM ARCHITECTURE

```
MAIN ALGORITHM: Egg Transport Network Optimization
Input: Network size N, start node S, end node T, required nodes K
Output: Optimal routes with minimum egg breakage

1. Generate city grid graph with breakage probabilities
2. Transform probabilities to cost weights using -log(1-p)
3. Choose algorithm based on requirements:
   - Single destination: Dijkstra's algorithm (primary) or Bellman-Ford (comparison)
   - Multiple required nodes: Use K-Routes (Held-Karp TSP)
4. Compare algorithm performance Bellman-Ford vs Dijkstra
5. Save comprehensive analysis to output files
```

## 2. GRAPH GENERATION ALGORITHM

```
FUNCTION generate_city_network(N, seed)
    // Generate realistic city grid topology

    adj_list ← create_grid_adjacency_list(N)
    adj_list ← ensure_connected(adj_list)  // Make graph connected

    graph ← empty directed graph
    FOR each node u in 0..N-1:
        FOR each neighbor v in adj_list[u]:
            IF u < v:  // Avoid duplicate edges
                weight_uv ← random(0.0, 0.04)  // Breakage probability
                weight_vu ← random(0.0, 0.04)
                graph.add_edge(u→v, weight_uv)
                graph.add_edge(v→u, weight_vu)

    // Transform to cost graph for shortest path algorithms
    cost_graph ← empty directed graph
    FOR each edge (u,v) with weight w:
        cost ← -log(1 - w)  // Convert probability to cost
        cost_graph.add_edge(u→v, cost)

    RETURN graph, cost_graph, N
```

## 3. DIJKSTRA'S SHORTEST PATH ALGORITHM

```
FUNCTION dijkstra_shortest_path(graph, start_node)
    // Find minimum breakage paths from start to all nodes

    N ← number of nodes in graph
    distance[N] ← infinity  // Distance from start to each node
    predecessor[N] ← -1     // Previous node in optimal path
    visited[N] ← false      // Whether node is processed

    distance[start_node] ← 0

    FOR iteration ← 0 to N-1:
        // Find unvisited node with minimum distance
        min_node ← find_minimum_distance_node(distance, visited)

        IF min_node == null: BREAK
        visited[min_node] ← true

        // Update distances to neighbors
        FOR each neighbor v of min_node:
            edge_weight ← graph[min_node][v]['weight']
            new_distance ← distance[min_node] + edge_weight

            IF new_distance < distance[v]:
                distance[v] ← new_distance
                predecessor[v] ← min_node

    RETURN distance, predecessor
```

## 3.5. BELLMAN-FORD SHORTEST PATH ALGORITHM

```
FUNCTION bellman_ford_shortest_path(graph, start_node)
    // Alternative shortest path algorithm that handles negative edges
    // Complexity: O(V × E) where V = vertices, E = edges
    // Used for comparison with Dijkstra's algorithm

    N ← number of nodes in graph
    distance[N] ← infinity  // Distance from start to each node
    distance[start_node] ← 0

    // Relax all edges |V|-1 times
    FOR iteration ← 1 to N-1:
        FOR each edge (u→v) in graph:
            edge_weight ← graph[u][v]['weight']
            IF distance[u] + edge_weight < distance[v]:
                distance[v] ← distance[u] + edge_weight

    // Check for negative cycles (optional, for validation)
    FOR each edge (u→v) in graph:
        edge_weight ← graph[u][v]['weight']
        IF distance[u] + edge_weight < distance[v]:
            RETURN ERROR("Graph contains negative cycle")

    RETURN distance
```

## 4. K-ROUTES ALGORITHM (TSP-STYLE ROUTING)

```
FUNCTION k_routes_analysis(graph, required_nodes, start, end)
    // Find optimal route visiting all required nodes (TSP-style)

    // Step 1: Convert NetworkX graph to adjacency format
    adj_list, cost_dict ← convert_graph_format(graph)

    // Step 2: Compute all-pairs shortest paths between required nodes
    terminals ← required_nodes
    distance_matrix, path_matrix ← pairwise_shortest_paths(adj_list, cost_dict, terminals)

    // Step 3: Solve TSP using Held-Karp dynamic programming
    total_cost, visit_order ← held_karp_tsp(distance_matrix, start_index, end_index)

    // Step 4: Reconstruct complete path through network
    full_path ← stitch_paths(visit_order, terminals, path_matrix)

    // Step 5: Calculate survival/breakage statistics
    survival_rate ← exp(-total_cost)
    breakage_rate ← 1 - survival_rate

    RETURN visit_order, full_path, survival_rate, breakage_rate
```

## 5. HELD-KARP TSP ALGORITHM (Core of K-Routes)

```
FUNCTION held_karp_tsp(distance_matrix, start_idx, end_idx)
    // Solve Traveling Salesman Problem using dynamic programming
    // Complexity: O(2^K * K^2) where K = number of required nodes

    K ← length of distance_matrix
    IF K > 8: 
        RAISE ERROR("Too many required nodes: {K}. Held-Karp algorithm is only efficient for k ≤ 8 nodes due to O(2^k) complexity. Consider using a heuristic approach for larger problems.")
        // NOTE: No heuristic implementation exists yet

    // Initialize DP table: dp[mask][node] = min cost to visit subset
    dp[2^K][K] ← infinity
    parent[2^K][K] ← -1

    // Base case: start node alone
    start_mask ← 1 << start_idx
    dp[start_mask][start_idx] ← 0

    // Fill DP table
    FOR each subset_mask from 0 to (2^K - 1):
        IF subset doesn't include start: CONTINUE

        FOR each current_node in subset:
            IF dp[subset_mask][current_node] is infinity: CONTINUE

            // Try extending to unvisited nodes
            FOR each next_node not in subset:
                new_mask ← subset_mask | (1 << next_node)
                new_cost ← dp[subset_mask][current_node] + distance_matrix[current_node][next_node]

                IF new_cost < dp[new_mask][next_node]:
                    dp[new_mask][next_node] ← new_cost
                    parent[new_mask][next_node] ← current_node

    // Find optimal ending
    full_mask ← (1 << K) - 1
    IF end_idx specified:
        best_cost ← dp[full_mask][end_idx]
        best_end ← end_idx
    ELSE:
        best_end ← argmin over j: dp[full_mask][j]
        best_cost ← dp[full_mask][best_end]

    // Backtrack to find visit order
    visit_order ← reconstruct_path(parent, full_mask, best_end)

    RETURN best_cost, visit_order
```

## 6. VISUALIZATION AND OUTPUT SYSTEM

```
FUNCTION visualize_results(graph, path, breakage, network_size)
    // Adaptive visualization based on network size

    IF network_size ≤ 50:
        // Detailed visualization
        draw_full_graph(graph)
        highlight_path(path)
        display_breakage_percentages()
        save_high_quality_image()

    ELSE IF network_size ≤ 200:
        // Simplified visualization
        draw_path_only(path)
        show_basic_statistics()
        save_standard_image()

    ELSE:
        // Large network handling
        save_sparse_matrix_format()
        save_path_coordinates_only()
        generate_summary_statistics()
```

## 7. ALGORITHM SELECTION LOGIC

```
FUNCTION select_algorithm(network_size, num_required_nodes, compare_algorithms)
    // Choose appropriate algorithm based on problem constraints

    K ← num_required_nodes

    IF K == 2:  // Simple start→end
        IF compare_algorithms:
            RETURN "Compare Dijkstra vs Bellman-Ford"
        ELSE:
            RETURN "Dijkstra Shortest Path"

    ELSE IF K ≤ 8:  // Small TSP feasible
        RETURN "K-Routes (Held-Karp Exact)"

    ELSE:  // Large TSP - NOT IMPLEMENTED
        RETURN "ERROR: K > 8 not supported (would require heuristic implementation)"

    // Network size considerations
    IF network_size > 10000:
        limit_visualization ← true
        use_memory_efficient_storage ← true
```

## 8. PERFORMANCE ANALYSIS SYSTEM

```
FUNCTION analyze_performance(results, network_sizes)
    // Compare algorithms across different network scales

    FOR each network_size in [100, 200, 1000, 10000]:
        time_dijkstra ← measure_execution_time(dijkstra_algorithm)
        time_bellman ← measure_execution_time(bellman_ford_algorithm)
        time_kroutes ← measure_execution_time(kroutes_algorithm)

        speedup_ratio ← time_bellman / time_dijkstra  // Typically 10-100x slower
        memory_usage ← track_peak_memory()
        path_quality ← calculate_breakage_reduction()

        save_comparison_results(network_size, time_dijkstra, time_bellman,
                               memory_usage, path_quality)

    // Key insights:
    // - Dijkstra: O((V+E)log V) with binary heap
    // - Bellman-Ford: O(V×E) - handles negative edges but slower
    // - K-Routes: O(2^K × K^2) - exponential in required nodes
    // - Current limitation: K-Routes only works for K ≤ 8
```

## 9. KEY MATHEMATICAL TRANSFORMATIONS

```
BREAKAGE PROBABILITY TO COST CONVERSION:
   Input: p = breakage probability (0 ≤ p < 1)
   Cost: c = -log(1 - p)

   Why? Converts multiplication of probabilities to addition of costs
   P(breakage) = 1 - ∏(1 - p_i) along path
   Cost minimization: minimize ∑(-log(1 - p_i)) = minimize -log(∏(1 - p_i))

SURVIVAL CALCULATION:
   Total cost C = ∑ c_i along path
   Survival probability S = e^(-C)
   Breakage probability B = 1 - S

TSP COMPLEXITY:
   Held-Karp: O(2^K × K^2)
   Practical limit: K ≤ 8 for real-time computation
   Current Implementation: Raises error for K > 8
   No heuristic fallback implemented yet
```

## 10. COMPLETE WORKFLOW SUMMARY

```
EGG TRANSPORT OPTIMIZATION WORKFLOW:

1. INPUT PHASE
   - Network size N
   - Start node S, End node T
   - Required intermediate nodes K

2. GRAPH GENERATION
   - Create city grid topology
   - Assign breakage probabilities to edges
   - Convert to cost graph

3. ALGORITHM SELECTION
   - Single destination: Dijkstra
   - Multiple destinations: K-Routes TSP (K ≤ 8 only)

4. COMPUTATION
   - Find optimal path(s)
   - Calculate survival/breakage statistics
   - Validate connectivity

5. OUTPUT & VISUALIZATION
   - Save results to files
   - Generate appropriate visualizations
   - Display performance metrics

6. ANALYSIS
   - Compare Dijkstra time with Bellman-Ford
   - Analyze algorithm scalability
   - Report optimization effectiveness
```

## 11. CURRENT LIMITATIONS & FUTURE WORK

```
IMPLEMENTATION STATUS:
✅ Dijkstra's algorithm - Fully implemented
✅ Bellman-Ford algorithm - Implemented for comparison
✅ K-Routes (Held-Karp) - Implemented for K ≤ 8 only
❌ K-Routes for K > 8 - Not implemented (requires heuristics)

FUTURE ENHANCEMENTS NEEDED:
1. Heuristic TSP solvers for K > 8:
   - Nearest Neighbor
   - Christofides algorithm
   - Genetic algorithms
   - Simulated annealing

2. Approximation guarantees:
   - Performance bounds for heuristics
   - Quality metrics vs optimal solution

3. Scalability improvements:
   - Parallel processing for large networks
   - Memory-efficient data structures
   - Incremental computation approaches
```