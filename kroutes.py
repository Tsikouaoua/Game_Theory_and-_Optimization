"""
K-Routes Module for Game Theory and Optimisation Assignment
Course: EBC4188 - Game Theory and Optimisation
Authors: Efstratios Gkoltsios and Gera Hooijer
Date: 14 October 2025

This module implements TSP-style routing using Held-Karp algorithm to find
the optimal route visiting all required nodes with minimum breakage.
"""

import math
import heapq
from typing import List, Dict, Tuple, Optional

Adj = List[set]
Cost = Dict[Tuple[int,int], float]

# ----- plain Dijkstra (or swap in your bidirectional version) -----
def dijkstra(adj: Adj, s: int, t: int, cost: Cost):
    n = len(adj); INF=float('inf')
    dist=[INF]*n; parent=[-1]*n; dist[s]=0.0
    pq=[(0.0,s)]
    while pq:
        d,u = heapq.heappop(pq)
        if d!=dist[u]: continue
        if u==t: break
        for v in adj[u]:
            # Use only the forward-edge cost; do not fall back to reverse direction.
            w = cost.get((u,v), INF)  # -ln(1-p)
            nd = d + w
            if nd < dist[v]:
                dist[v]=nd; parent[v]=u; heapq.heappush(pq,(nd,v))
    if dist[t]==INF: return INF, []
    path=[]; u=t
    while u!=-1: path.append(u); u=parent[u]
    path.reverse(); return dist[t], path

# ----- step 1: all-pairs on the terminals -----
def pairwise_shortest(adj: Adj, cost: Cost, terminals: List[int]):
    k = len(terminals)
    D = [[math.inf]*k for _ in range(k)]
    P: Dict[Tuple[int,int], List[int]] = {}
    for i, a in enumerate(terminals):
        for j, b in enumerate(terminals):
            if i == j: 
                D[i][j] = 0.0
                P[(i,j)] = [a]
            else:
                # Compute both directions independently (supports directed graphs).
                d, p = dijkstra(adj, a, b, cost)   # replace with bidirectional if you like
                D[i][j] = d
                P[(i,j)] = p
                if math.isinf(d):
                    raise ValueError(f"Unreachable terminals: {a} -> {b}")
    return D, P

# ----- step 2: Held–Karp for PATH (start fixed; end fixed or free) -----
def held_karp_path(D, start_idx: int, end_idx: Optional[int] = None):
    """
    D: k x k matrix of shortest-path distances between terminals.
    Returns (best_cost, order_indices). If end_idx is None, best end is chosen.
    
    Note: This algorithm has O(2^k * k^2) complexity. For k > 8, it becomes very slow.
    """
    k = len(D)
    
    # Check for computational feasibility
    if k > 8:
        raise ValueError(f"Too many required nodes: {k}. Held-Karp algorithm is only efficient for k ≤ 8 nodes due to O(2^k) complexity. Consider using a heuristic approach for larger problems.")
    
    FULL = (1 << k)
    INF = float('inf')
    # Only consider subsets that include start
    dp = [[INF]*k for _ in range(FULL)]
    parent = [[-1]*k for _ in range(FULL)]
    start_mask = 1 << start_idx
    dp[start_mask][start_idx] = 0.0

    for mask in range(FULL):
        if not (mask & start_mask): 
            continue
        for j in range(k):
            if not (mask & (1 << j)): 
                continue
            if dp[mask][j] == INF: 
                continue
            # extend to next node m not yet visited
            nxt_mask_candidates = (~mask) & (FULL - 1)
            m = 0
            while nxt_mask_candidates:
                lb = nxt_mask_candidates & -nxt_mask_candidates
                m = (lb.bit_length() - 1)
                nxt_mask_candidates ^= lb
                nd = dp[mask][j] + D[j][m]
                new_mask = mask | (1 << m)
                if nd < dp[new_mask][m]:
                    dp[new_mask][m] = nd
                    parent[new_mask][m] = j

    full = FULL - 1
    # choose end
    if end_idx is None:
        best_j = min(range(k), key=lambda j: dp[full][j])
    else:
        best_j = end_idx
    best_cost = dp[full][best_j]
    if math.isinf(best_cost):
        raise ValueError("No Hamiltonian path found over terminals (check connectivity).")

    # backtrack order indices
    order = []
    mask = full; j = best_j
    while j != -1:
        order.append(j)
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return best_cost, order

# ----- step 3: stitch the actual graph path -----
def stitch_full_path(order_idx: List[int], terminals: List[int], P_paths: Dict[Tuple[int,int], List[int]]):
    full = []
    for a_idx, b_idx in zip(order_idx, order_idx[1:]):
        seg = P_paths[(a_idx, b_idx)]
        if not full: 
            full.extend(seg)
        else:
            full.extend(seg[1:])   # avoid duplicate junction
    return full

# ======== Public helper you can call ========
def best_route_visiting_all(graph, required: List[int],
                            start_id: int, end_id: Optional[int] = None):
    """
    Find the best route visiting all required nodes using TSP-style algorithm.
    
    Parameters:
    - graph: NetworkX graph
    - required: list of node IDs you must visit (include start_id, and include end_id if fixed end).
    - start_id: starting node
    - end_id: ending node (optional)
    
    Returns: results dictionary with path information
    """
    try:
        # Convert NetworkX graph to adjacency list format
        adj, cost = convert_networkx_to_adj_cost(graph)
        
        # terminals ordered as given
        terminals = required[:]
        start_idx = terminals.index(start_id)
        end_idx = None if end_id is None else terminals.index(end_id)

        D, P = pairwise_shortest(adj, cost, terminals)
        total_cost, order_idx = held_karp_path(D, start_idx, end_idx=end_idx)
        visit_order = [terminals[i] for i in order_idx]
        full_path = stitch_full_path(order_idx, terminals, P)
        
        survival = math.exp(-total_cost)
        breakage = 1 - survival
        
        # Format highlighted path
        highlighted_path = format_path_with_highlights(full_path, required)
        
        # Create results dictionary
        results = {
            'visit_order': visit_order,
            'full_path': full_path,
            'highlighted_path': highlighted_path,
            'total_cost': total_cost,
            'survival_rate': survival,
            'breakage_rate': breakage,
            'path_length': len(full_path),
            'required_nodes': required
        }
        
        # Print results
        print(f"\nK-ROUTES RESULTS:")
        print(f"Visit order: {visit_order}")
        print(f"Full path: {highlighted_path}")
        print(f"Total breakage: {breakage*100:.2f}%")
        print(f"Survival rate: {survival*100:.2f}%")
        print(f"Path length: {len(full_path)} nodes")
        
        return results
        
    except Exception as e:
        print(f"Error in K-routes analysis: {e}")
        return None


def convert_networkx_to_adj_cost(graph):
    """
    Convert NetworkX graph to adjacency list and cost dictionary format.
    
    Parameters:
    - graph: NetworkX Graph (directed or undirected). If directed, edges are kept directed.
    
    Returns:
    - adj: List of sets (adjacency list)
    - cost: Dict of edge costs (transformed weights)
    """
    n = graph.number_of_nodes()
    adj = [set() for _ in range(n)]
    cost = {}
    directed = getattr(graph, "is_directed", lambda: False)()
    
    for u, v, data in graph.edges(data=True):
        # Clamp and transform probability to cost: -ln(1-p)
        weight = data['weight']
        weight = max(0.001, min(0.999, weight))  # Keep in range [0.001, 0.999]
        edge_cost = -math.log(1 - weight)
        
        if directed:
            # Preserve directionality
            adj[u].add(v)
            cost[(u, v)] = edge_cost
        else:
            # For undirected graphs, add both directions with the same cost
            adj[u].add(v)
            adj[v].add(u)
            cost[(u, v)] = edge_cost
            cost[(v, u)] = edge_cost
    
    return adj, cost


def format_path_with_highlights(full_path, required_nodes):
    """
    Format path string with required nodes highlighted using quotes.
    
    Parameters:
    - full_path: List of nodes in the complete path
    - required_nodes: List of nodes that should be highlighted
    
    Returns:
    - formatted_path: String with highlighted nodes
    """
    required_set = set(required_nodes)
    formatted_nodes = []
    
    for node in full_path:
        if node in required_set:
            formatted_nodes.append(f"'{node}'")
        else:
            formatted_nodes.append(str(node))
    
    return " → ".join(formatted_nodes)


def run_kroutes_analysis(graph, required_nodes, start_node, end_node):
    """
    Run K-routes analysis and display results using the implemented algorithm
    """
    print(f"\nRunning K-routes analysis...")
    print(f"Required nodes to visit: {required_nodes}")
    print(f"Start: {start_node}, End: {end_node}")
    
    try:
        results = best_route_visiting_all(graph, required_nodes, start_node, end_node)
        
        if results:
            print(f"\nK-routes Analysis Completed Successfully!")
            print(f"="*50)
            # Results are already printed by the function
        else:
            print("K-routes analysis failed.")
            
    except Exception as e:
        print(f"Error in K-routes analysis: {e}")
        print("This might happen with disconnected graphs or invalid node selections.")


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create a simple test graph
    G = nx.Graph()
    edges = [(0, 1, 2), (1, 2, 3), (2, 3, 1), (0, 3, 5), (1, 3, 2)]
    G.add_weighted_edges_from(edges)
    
    required = [0, 2, 3]
    start = 0
    end = 3
    
    print("Test K-routes with simple graph:")
    run_kroutes_analysis(G, required, start, end)