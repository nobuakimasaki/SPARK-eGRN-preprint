"""
graph_fast.py — drop-in accelerated replacement for the hot graph-move
functions in graph.py.

Change #1 from the speed review: the original `propose_graph_move_with_probs`
enumerated birth/death/swap candidates by running a pure-Python DFS
(`would_create_cycle` -> `has_path`) for *every* allowed non-edge, and it did
this three times per proposal (once to propose, once for the forward proposal
probability, once for the backward). With ~11k allowed edges and 302 nodes that
is the dominant cost of the whole sampler.

This module keeps the *exact same proposal distribution* (uniform over available
move types, then uniform over candidates of that type, with the same row-major
candidate ordering) but:

  1. Computes an all-pairs reachability matrix R once per graph, in Numba, so
     "does adding i->j create a cycle?" becomes the vectorized test R[j, i].
     Birth enumeration is then a single boolean-matrix expression with no
     Python loop and no per-edge DFS.
  2. Runs cycle detection for swaps in Numba (`_has_cycle`) instead of pure
     Python, over the small set of existing edges only.
  3. Enumerates candidates twice per proposal (current graph + proposed graph)
     instead of three times, reusing the forward enumeration for both the
     sample and the forward proposal probability.

Public API (`propose_graph_move_with_probs`, `has_cycle`, `extract_edges`, ...)
matches graph.py so it can be swapped in by changing a single import.
"""
import numpy as np
from numba import njit

# Re-export the unchanged, non-hot helpers so this module is API-compatible.
from graph import (
    extract_edges,
    has_path,
    would_create_cycle,
    canonical_graph,
    graph_posterior_counts,
    edges_to_adj,
)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=False)
def _reachability(A):
    """
    Boolean all-pairs reachability over directed graph A (int8, n x n).

    R[s, t] == True iff there is a directed path s -> ... -> t of length >= 1.
    The diagonal is left False (a DAG has no self-path), which is exactly what
    the birth cycle-test needs.
    """
    n = A.shape[0]
    R = np.zeros((n, n), dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    for s in range(n):
        top = 0
        # Seed the DFS with the direct successors of s.
        for v in range(n):
            if A[s, v] != 0 and not R[s, v]:
                R[s, v] = True
                stack[top] = v
                top += 1
        while top > 0:
            top -= 1
            u = stack[top]
            for v in range(n):
                if A[u, v] != 0 and not R[s, v]:
                    R[s, v] = True
                    stack[top] = v
                    top += 1
    return R


@njit(cache=False)
def _has_cycle(A):
    """Iterative DFS-coloring cycle detection for directed graph A (int8)."""
    n = A.shape[0]
    color = np.zeros(n, dtype=np.int8)     # 0=white, 1=gray, 2=black
    it = np.zeros(n, dtype=np.int64)       # per-node neighbor cursor
    stack = np.empty(n, dtype=np.int64)

    for start in range(n):
        if color[start] != 0:
            continue
        color[start] = 1
        it[start] = 0
        stack[0] = start
        top = 1
        while top > 0:
            u = stack[top - 1]
            k = it[u]
            v = -1
            while k < n:
                if A[u, k] != 0:
                    v = k
                    break
                k += 1
            it[u] = k + 1
            if v == -1:
                color[u] = 2
                top -= 1
            else:
                if v == u:
                    return True
                if color[v] == 1:
                    return True
                if color[v] == 0:
                    color[v] = 1
                    it[v] = 0
                    stack[top] = v
                    top += 1
    return False


@njit(cache=False)
def _reach2(A, R):
    """
    R2[i, j] = OR_m ( A[i, m] != 0 AND R[m, j] ): nodes reachable from i by a
    directed path of length >= 2, given the >=1-step reachability R.

    Used for the swap cycle-test. Reversing edge i->j (remove i->j, add j->i)
    creates a cycle iff i still reaches j by some *other* path, i.e. a path of
    length >= 2. In a DAG that is exactly R2[i, j]: any successor m of i cannot
    reach i (that would be a cycle i->m->...->i), so an m->...->j path never
    passes through i and therefore never uses the edge i->j. Hence R2 computed
    on the full A answers the alternate-path question without removing the edge.

    Cost is O(sum_i outdeg(i) * n) = O(E * n) -- cheap for sparse DAGs and far
    below the O(E) full cycle-detection copies it replaces.
    """
    n = A.shape[0]
    R2 = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for m in range(n):
            if A[i, m] != 0:
                Rm = R[m]
                R2i = R2[i]
                for j in range(n):
                    if Rm[j]:
                        R2i[j] = True
    return R2


def has_cycle(A):
    """Numba-accelerated drop-in for graph.has_cycle."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    return bool(_has_cycle(np.ascontiguousarray(A, dtype=np.int8)))


# ---------------------------------------------------------------------------
# Candidate enumeration (same sets & ordering as graph._graph_move_candidates)
# ---------------------------------------------------------------------------

def _kth_true(mask, k, n):
    """
    Return the (row, col) of the k-th True entry of the (n, n) boolean `mask`
    in C (row-major) order. This is exactly the k-th element the old code would
    have produced with `np.where(mask)` + `list(zip(...))`, so sampling by index
    picks the same candidate edge.
    """
    flat_idx = int(np.flatnonzero(mask.ravel())[k])
    return divmod(flat_idx, n)


def _graph_move_candidates(A, allowed_mask=None):
    """
    Enumerate birth/death/swap candidates as boolean (n, n) masks plus counts,
    instead of materializing Python lists of (i, j) tuples.

    The candidate SETS and their row-major ordering are identical to the old
    list-based version: the k-th candidate of a type is the k-th True of its
    mask in C-order (see `_kth_true`), which matches `np.where` + `list(zip())`.

    Returns
    -------
    masks : dict[str, np.ndarray]   {"birth","death","swap"} -> (n, n) bool
    counts : dict[str, int]         number of candidates of each type
    move_types : list[str]          nonempty types in ["birth","death","swap"] order
    """
    A = np.asarray(A)
    p = A.shape[0]
    if A.ndim != 2 or p != A.shape[1]:
        raise ValueError("A must be square.")

    Ai = np.ascontiguousarray(A, dtype=np.int8)

    if allowed_mask is not None:
        allowed_mask = np.asarray(allowed_mask, dtype=bool)
        if allowed_mask.shape != A.shape:
            raise ValueError("allowed_mask must have the same shape as A.")

    # ---- deaths: existing off-diagonal edges ----
    nz = (Ai != 0)
    np.fill_diagonal(nz, False)
    death_mask = nz

    # ---- births: allowed, absent, and non-cycle-creating ----
    # Adding i->j creates a cycle iff j already reaches i, i.e. R[j, i].
    R = _reachability(Ai)
    no_edge = ~nz
    np.fill_diagonal(no_edge, False)
    if allowed_mask is not None:
        birth_mask = allowed_mask & no_edge & (~R.T)
    else:
        birth_mask = no_edge & (~R.T)

    # ---- swaps: reverse an existing edge without creating a cycle ----
    # Vectorized cycle test via R2 (paths of length >= 2); see _reach2. This
    # reproduces the old per-edge _has_cycle enumeration exactly -- same set,
    # same row-major order -- with no per-edge array copies.
    R2 = _reach2(Ai, R)
    swap_mask = nz & (~R2)              # existing edge, no alternate i->j path
    swap_mask = swap_mask & (~nz.T)     # reverse (j,i) is not already an edge
    if allowed_mask is not None:
        swap_mask = swap_mask & allowed_mask.T   # reverse (j,i) must be allowed

    masks = {"birth": birth_mask, "death": death_mask, "swap": swap_mask}
    counts = {mt: int(m.sum()) for mt, m in masks.items()}
    move_types = [mt for mt in ("birth", "death", "swap") if counts[mt] > 0]

    return masks, counts, move_types


def _prob_from_candidates(move_type, edge, masks, counts, move_types):
    """q(A -> A') for (move_type, edge) from candidate masks + counts."""
    if move_type not in move_types:
        return 0.0
    i, j = edge
    if not masks[move_type][i, j]:
        return 0.0
    return (1.0 / len(move_types)) * (1.0 / counts[move_type])


def propose_graph_move_with_probs(A, rng, allowed_mask=None):
    """
    Accelerated drop-in for graph.propose_graph_move_with_probs.

    Identical proposal distribution and return signature:
        (A_new, move_type, move_detail, log_q_fwd, log_q_bwd)
    """
    A = np.asarray(A)
    A_new = A.copy()
    n = A.shape[0]

    masks, counts, move_types = _graph_move_candidates(A, allowed_mask=allowed_mask)

    if not move_types:
        return A_new, "none", {}, 0.0, 0.0

    move_type = rng.choice(move_types)

    if move_type == "birth":
        parent, child = _kth_true(masks["birth"], int(rng.integers(counts["birth"])), n)
        A_new[parent, child] = 1
        move_detail = {"parent": parent, "child": child}
        edge_forward = (parent, child)
        edge_backward = (parent, child)      # reverse is death
        back_type = "death"

    elif move_type == "death":
        parent, child = _kth_true(masks["death"], int(rng.integers(counts["death"])), n)
        A_new[parent, child] = 0
        move_detail = {"parent": parent, "child": child}
        edge_forward = (parent, child)
        edge_backward = (parent, child)      # reverse is birth
        back_type = "birth"

    else:  # swap
        parent, child = _kth_true(masks["swap"], int(rng.integers(counts["swap"])), n)
        A_new[parent, child] = 0
        A_new[child, parent] = 1
        move_detail = {
            "old_parent": parent,
            "old_child": child,
            "new_parent": child,
            "new_child": parent,
        }
        edge_forward = (parent, child)
        edge_backward = (child, parent)      # reverse is swap
        back_type = "swap"

    # Forward probability: reuse the enumeration we already have (no re-scan).
    q_fwd = _prob_from_candidates(move_type, edge_forward, masks, counts, move_types)

    # Backward probability: one enumeration of the proposed graph (masks/counts
    # only -- no candidate lists are built).
    masks2, counts2, mt2 = _graph_move_candidates(A_new, allowed_mask=allowed_mask)
    q_bwd = _prob_from_candidates(back_type, edge_backward, masks2, counts2, mt2)

    log_q_fwd = -np.inf if q_fwd == 0.0 else float(np.log(q_fwd))
    log_q_bwd = -np.inf if q_bwd == 0.0 else float(np.log(q_bwd))

    return A_new, move_type, move_detail, log_q_fwd, log_q_bwd
