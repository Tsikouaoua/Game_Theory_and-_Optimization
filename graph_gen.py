import random
from collections import deque
from typing import List, Set

Adj = List[Set[int]]


def gen_city_grid(n: int, avg_deg: float = 3.2, seed: int | None = None) -> Adj:
    """Generate a city-like grid graph with some diagonal shortcuts.

    This keeps the behavior used by `main.py`: create a roughly rectangular
    grid of `n` nodes, connect 4-neighbors, add occasional diagonal edges,
    """
    if seed is not None:
        random.seed(seed)

    import math

    # Choose grid dimensions close to square but slightly wider
    grid_width = int(math.sqrt(n * 1.2))
    grid_height = int(n / grid_width) + 1
    while grid_width * grid_height < n:
        grid_width += 1

    adj: Adj = [set() for _ in range(n)]

    # map node id <-> grid position
    node_to_pos = {}
    pos_to_node = {}
    node_id = 0
    for r in range(grid_height):
        for c in range(grid_width):
            if node_id >= n:
                break
            node_to_pos[node_id] = (r, c)
            pos_to_node[(r, c)] = node_id
            node_id += 1
        if node_id >= n:
            break

    # 4-neighborhood
    for u, (r, c) in node_to_pos.items():
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pos_to_node:
                v = pos_to_node[(nr, nc)]
                adj[u].add(v); adj[v].add(u)

    # diagonals (sparse)
    for u, (r, c) in node_to_pos.items():
        for nr, nc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (nr, nc) in pos_to_node and random.random() < 0.20:
                v = pos_to_node[(nr, nc)]
                adj[u].add(v); adj[v].add(u)
    return adj


def _components(adj: Adj):
    n = len(adj)
    seen = [False] * n
    comps = []
    for s in range(n):
        if not seen[s]:
            q = deque([s]); seen[s] = True; comp = [s]
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True; q.append(w); comp.append(w)
            comps.append(comp)
    return comps


def make_connected(adj: Adj) -> Adj:
    """Connect components by adding a random edge between successive components."""
    comps = _components(adj)
    for i in range(len(comps) - 1):
        u = random.choice(comps[i]); v = random.choice(comps[i + 1])
        adj[u].add(v); adj[v].add(u)
    return adj