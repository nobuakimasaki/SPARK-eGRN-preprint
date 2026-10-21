"""
Helper functions related to the graph G, where the graph is now an adjacency matrix.

Convention:
- A[i, j] == 1 means there is a directed edge i -> j.
- A is a (p x p) numpy array of 0/1 (or bool) values.
- No self-edges are allowed (i != j) for move proposals.

Optional:
- If you have node names, pass `nodes` as a list of names (length p).
  Then edges can be returned as (name_i, name_j) instead of (i, j).
"""
import numpy as np
from copy import deepcopy
from collections import Counter

def extract_edges(A):
    """Return list of (i, j) such that A[i, j] == 1 and i != j."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    ii, jj = np.where(A != 0)
    return [(int(i), int(j)) for i, j in zip(ii, jj) if i != j]

def has_path(A, start, target):
    """Is there a directed path start -> ... -> target? (indices)"""
    A = np.asarray(A)
    p = A.shape[0]
    if A.ndim != 2 or p != A.shape[1]:
        raise ValueError("A must be square.")
    if not (0 <= start < p and 0 <= target < p):
        raise IndexError("start/target out of bounds.")
    if start == target:
        return True

    stack = [start]
    visited = np.zeros(p, dtype=bool)

    while stack:
        u = stack.pop()
        if u == target:
            return True
        if visited[u]:
            continue
        visited[u] = True

        nbrs = np.flatnonzero(A[u] != 0)
        for v in nbrs:
            v = int(v)
            if not visited[v]:
                stack.append(v)

    return False

def would_create_cycle(A, parent, child):
    """
    If we add edge parent -> child, does it create a cycle?
    A cycle occurs if there is already a path child -> ... -> parent.
    """
    return has_path(A, child, parent)

def has_cycle(A):
    """Detect cycle in directed graph adjacency matrix A using DFS coloring."""
    A = np.asarray(A)
    p = A.shape[0]
    if A.ndim != 2 or p != A.shape[1]:
        raise ValueError("A must be square.")

    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.full(p, WHITE, dtype=np.int8)

    def dfs(u):
        color[u] = GRAY
        for v in np.flatnonzero(A[u] != 0):
            v = int(v)
            if v == u:
                return True  # self-loop
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for u in range(p):
        if color[u] == WHITE and dfs(u):
            return True
    return False

def _graph_move_candidates(A, allowed_mask=None):
    """
    Enumerate all valid birth, death, and swap candidates for a single MCMC
    graph move starting from adjacency matrix A.

    Birth  : add an edge (i,j) that does not exist and would not create a cycle.
    Death  : remove an existing edge (i,j).
    Swap   : reverse an existing edge (i,j) -> (j,i) without creating a cycle.

    Parameters
    ----------
    A : np.ndarray, shape (p, p)
        Current binary adjacency matrix (DAG).
    allowed_mask : np.ndarray of bool, shape (p, p), or None
        allowed_mask[i, j] is True iff a directed edge i -> j is permitted by
        the prior (i.e. not forbidden; src_idx[i, j] != -1) with i != j.  Births
        are restricted to allowed entries.  Because the prior is directional
        (rows -> columns), reversing an allowed edge may land on a forbidden
        entry, so a swap (i,j) -> (j,i) is only offered when the reversed edge
        (j,i) is itself allowed.  Pass None to allow every off-diagonal edge.

    Returns
    -------
    birth_candidates : list of (int, int)
    death_candidates : list of (int, int)
    swap_candidates  : list of (int, int)
    move_types       : list of str
        Subset of ["birth", "death", "swap"] — only types with at least one
        candidate are included.
    """
    A = np.asarray(A)
    p = A.shape[0]
    if A.ndim != 2 or p != A.shape[1]:
        raise ValueError("A must be square.")

    if allowed_mask is not None:
        allowed_mask = np.asarray(allowed_mask, dtype=bool)
        if allowed_mask.shape != A.shape:
            raise ValueError("allowed_mask must have the same shape as A.")

    death_candidates = extract_edges(A)
    edge_set = set(death_candidates)

    birth_candidates = []
    if allowed_mask is not None:
        # Only enumerate edges the prior permits — this skips the ~half of
        # cross-set pairs that are forbidden (src_idx == -1) and would be
        # rejected on sight by the -inf graph prior.
        ii, jj = np.where(allowed_mask)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if (i, j) not in edge_set and not would_create_cycle(A, i, j):
                birth_candidates.append((i, j))
    else:
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if (i, j) in edge_set:
                    continue
                if not would_create_cycle(A, i, j):
                    birth_candidates.append((i, j))

    swap_candidates = []
    for (i, j) in death_candidates:
        if (j, i) in edge_set:
            continue
        # Directional prior: only swap if the reversed edge is itself allowed.
        if allowed_mask is not None and not allowed_mask[j, i]:
            continue
        tmpA = A.copy()
        tmpA[i, j] = 0
        tmpA[j, i] = 1
        if not has_cycle(tmpA):
            swap_candidates.append((i, j))

    move_types = []
    if birth_candidates:
        move_types.append("birth")
    if death_candidates:
        move_types.append("death")
    if swap_candidates:
        move_types.append("swap")

    return birth_candidates, death_candidates, swap_candidates, move_types

def _graph_move_prob(A, move_type, edge, allowed_mask=None):
    """
    Compute q(A -> A') for a specific move (move_type, edge).

    Candidates are enumerated by _graph_move_candidates(A, allowed_mask), so the
    same prior restriction applies to both the forward and backward proposals.

    Proposal mechanism:
        - choose move_type uniformly among available move_types
        - choose edge uniformly among candidates of that type
    """
    birth_candidates, death_candidates, swap_candidates, move_types = \
        _graph_move_candidates(A, allowed_mask)

    if move_type not in move_types:
        return 0.0

    if move_type == "birth":
        cand = birth_candidates
    elif move_type == "death":
        cand = death_candidates
    elif move_type == "swap":
        cand = swap_candidates
    else:
        return 0.0

    if edge not in cand:
        return 0.0

    return (1.0 / len(move_types)) * (1.0 / len(cand))

def propose_graph_move_with_probs(
    A,
    rng,
    allowed_mask=None,
):
    """
    Propose a new adjacency matrix A_new from A by birth/death/swap.
    Also returns log proposal probs for MH: log q_fwd, log q_bwd.

    If allowed_mask is given, births (and the reversed edge of a swap) are
    restricted to entries the prior permits (src_idx != -1); see
    _graph_move_candidates.
    """
    A = np.asarray(A)
    A_new = A.copy()

    birth_candidates, death_candidates, swap_candidates, move_types = \
        _graph_move_candidates(A, allowed_mask=allowed_mask)

    if not move_types:
        return A_new, "none", {}, 0.0, 0.0

    move_type = rng.choice(move_types)

    if move_type == "birth":
        parent, child = birth_candidates[int(rng.integers(len(birth_candidates)))]
        A_new[parent, child] = 1
        move_detail = {"parent": parent, "child": child}
        edge_forward = (parent, child)
        edge_backward = (parent, child)  # reverse is death

    elif move_type == "death":
        parent, child = death_candidates[int(rng.integers(len(death_candidates)))]
        A_new[parent, child] = 0
        move_detail = {"parent": parent, "child": child}
        edge_forward = (parent, child)
        edge_backward = (parent, child)  # reverse is birth

    elif move_type == "swap":
        parent, child = swap_candidates[int(rng.integers(len(swap_candidates)))]
        A_new[parent, child] = 0
        A_new[child, parent] = 1
        move_detail = {
            "old_parent": parent,
            "old_child": child,
            "new_parent": child,
            "new_child": parent,
        }
        edge_forward = (parent, child)
        edge_backward = (child, parent)  # reverse is swap

    else:
        return A_new, "none", {}, 0.0, 0.0

    # Forward proposal prob
    q_fwd = _graph_move_prob(A, move_type, edge_forward, allowed_mask=allowed_mask)

    # Determine backward move type
    if move_type == "birth":
        back_type = "death"
    elif move_type == "death":
        back_type = "birth"
    else:
        back_type = "swap"

    # Backward proposal prob
    q_bwd = _graph_move_prob(A_new, back_type, edge_backward, allowed_mask=allowed_mask)

    log_q_fwd = -np.inf if q_fwd == 0.0 else float(np.log(q_fwd))
    log_q_bwd = -np.inf if q_bwd == 0.0 else float(np.log(q_bwd))

    return A_new, move_type, move_detail, log_q_fwd, log_q_bwd

def edges_to_adj(edges, p, values=None):
    """
    Reconstruct a dense (p, p) matrix from an edge list.

    Inverse of the sparse sample format written by mcmc.SampleStore.record:
    each graph sample is an (E, 2) integer array whose rows are
    (parent, child) pairs, and the matching W sample is an (E,) float array
    of weights on those edges.

    Parameters
    ----------
    edges : np.ndarray, shape (E, 2)
        Edge list; row k is the (parent, child) of edge k.
    p : int
        Number of nodes.
    values : np.ndarray, shape (E,), or None
        If None, return the int8 adjacency matrix (1 on each edge).
        Otherwise return a float64 matrix with values[k] at edges[k].

    Returns
    -------
    np.ndarray, shape (p, p)
        Adjacency matrix (values=None) or dense weight matrix.
    """
    edges = np.asarray(edges)
    if values is None:
        A = np.zeros((p, p), dtype=np.int8)
        if edges.size:
            A[edges[:, 0], edges[:, 1]] = 1
        return A
    M = np.zeros((p, p), dtype=np.float64)
    if edges.size:
        M[edges[:, 0], edges[:, 1]] = values
    return M

def canonical_graph(A):
    """Hashable canonical key: sorted tuple of (i, j) edges (A[i, j]==1, i!=j)."""
    return tuple(sorted(extract_edges(A)))

def graph_posterior_counts(A_samples):
    """
    Count how often each distinct DAG structure appears in a list of samples.

    Parameters
    ----------
    A_samples : list of np.ndarray
        Posterior samples of adjacency matrices (e.g. from the cold chain).

    Returns
    -------
    Counter
        Maps canonical_graph(A) -> frequency.  Most-visited graphs are
        accessible via .most_common().
    """
    keys = [canonical_graph(A) for A in A_samples]
    return Counter(keys)