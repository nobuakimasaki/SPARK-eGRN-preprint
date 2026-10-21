"""
eval.py
=======
Posterior summaries of the real-data MCMC run. Reads the sampler output (config.CHAIN_PATH)
and writes to config.EVAL_DIR:

  edge_recovery.csv : one row per directed edge present in >= 1 retained draw, with node
                      names: parent, child, post_prob, w_post_median, w_post_lower95,
                      w_post_upper95. post_prob is the edge's posterior inclusion
                      probability (PIP); w_post_* summarize its coefficient CONDITIONAL on
                      the edge being present (W is stored masked to the sampled graph, so
                      the marginal would be biased toward 0), and are NaN for an edge
                      present in fewer than W_MIN_PRESENT_DRAWS draws.
  posterior_dir.npy : (n, n) matrix of PIPs
  W_median.npy      : (n, n) marginal posterior median of W (0 when the edge is absent)
  rho_samples.npy   : (S, K+1) retained draws of the class edge probabilities
                      [rho_0, rho_1, ..., rho_K]
  nodes.csv         : node index -> name, in model order

The analysis window is config.BURN_IN to config.WIN_END, thinned by config.THIN (the
paper's 225,000 draws). Credible intervals are central 95% intervals.

Usage
-----
  python eval.py                                # the paper's window
  python eval.py --start 0 --end 1000000 --thin 1
"""

import io
import os
import gc
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit

import config

# Minimum number of draws in which an edge must be present for its
# conditional-on-present w_ij posterior to be summarized; fewer -> NaN.
W_MIN_PRESENT_DRAWS = 10


# ---------------------------------------------------------------------------
# Memory-bounded streaming loader
# ---------------------------------------------------------------------------
# The MCMC result is one monolithic pickle whose six per-draw sample lists
# (graphs/b/W/log_psi/alpha/rho_logits, one entry per stored draw) dominate its
# size -- reconstructed in full it does not fit in RAM, so a plain pickle.load gets
# OOM-killed. `stream_load` instead diverts those six lists to per-item callbacks as they
# are unpickled (leaving the stored lists empty) and nulls large objects out of the memo,
# so the file streams through in a few GB. Everything else (col_names, n_nodes,
# diagnostics, ...) loads normally.

_STREAM_MEMO_DROP_BYTES = 256   # null memoized bytes longer than this (raw array data)

# Nulling the value is not enough on its own: the memo still gets one dict entry
# per memoized object, and every state and shape tuple built for a draw is kept,
# neither of which is an ndarray or a long bytes. That grows with draws STREAMED,
# not draws kept, so thinning does not bound it -- a 5,250,000-draw checkpoint was
# OOM-killed at 41 GB. _StreamMemo below counts every slot but stores only what a
# later GET can still ask for.
#
# Memo slots below this index are kept whatever they hold. Pickling an ndarray
# reuses objects that are not shareable by type -- numpy's reducer hands back the
# same (0,) shape tuple and b'b' dtype argument on every call, and later draws GET
# them rather than rewriting them. Those constants are memoized within the first
# draw or two, so a generous prefix covers them without having to predict which
# containers are shared.
_STREAM_MEMO_PREFIX_KEEP = 100_000

# Above this, a memoized buffer is one of the trace arrays rather than a per-draw
# one, and everything from there on is kept. The traces and `diagnostics` follow
# the six sample lists in the pickle, and logpost_trace -- 8 bytes per iteration --
# is the first buffer this large, so the latch flips exactly where the per-draw
# data ends. Without it the small arrays inside `diagnostics` would be
# indistinguishable from a sample draw and would be dropped too.
_STREAM_MEMO_LATCH_BYTES = 256 * 1024


def _stream_memo_shareable(v):
    """Can a later GET still ask for this object once we are past the prefix?
    Objects the pickler memoizes once and reuses: dtypes, classes and functions
    (numpy.ndarray and its reconstructor), interned strings and scalars. The
    containers and arrays rebuilt for every draw are not."""
    return (isinstance(v, (np.dtype, str, int, float, bool, type(None)))
            or isinstance(v, type) or callable(v))


class _StreamMemo:
    """Memo that counts every entry but stores only what stays reachable.

    The unpickler assigns memo keys by position (`memo[len(memo)] = obj`), so the
    count has to stay exact or every later GET reads the wrong slot. Storing is
    what we skip, not counting. Dropping something still needed is an
    UnpicklingError on a missing memo index, not a silently wrong result."""

    __slots__ = ("_d", "_n", "_owner")

    def __init__(self, owner):
        self._d = {}
        self._n = 0
        self._owner = owner

    def __len__(self):
        return self._n

    def __setitem__(self, key, value):
        if key >= self._n:
            self._n = key + 1
        if (key < _STREAM_MEMO_PREFIX_KEEP or self._owner._memo_latch
                or _stream_memo_shareable(value)):
            self._d[key] = value

    def __getitem__(self, key):
        return self._d[key]


class _StreamUnpickler(pickle._Unpickler):
    """Pure-Python Unpickler that streams selected top-level dict-value lists.

    `handlers` maps a dict key (e.g. "graphs_samples") to a callable invoked once
    per list element, in stored order; those elements are then discarded rather
    than appended, so the corresponding list in the returned object stays empty.
    The pure-Python unpickler is required so the append/memoize opcodes can be
    overridden -- the C `Unpickler` cannot be subclassed this way."""

    dispatch = dict(pickle._Unpickler.dispatch)   # copy so overrides stay local

    def __init__(self, file, handlers, on_key_change=None):
        super().__init__(file)
        self._handlers = handlers
        self._on_key_change = on_key_change
        self._cur_key = None
        self._memo_latch = False

    # -- memo policy (see _StreamMemo) --------------------------------------
    def load_proto(self):
        # load() installs a plain dict as the memo just before entering the
        # dispatch loop, and PROTO is the first opcode in it, so this is where
        # ours goes in.
        pickle._Unpickler.load_proto(self)
        self.memo = _StreamMemo(self)
    dispatch[pickle.PROTO[0]] = load_proto

    def _note_buffer(self, n):
        """Flip the memo latch once a buffer is too big to be a per-draw one."""
        if n >= _STREAM_MEMO_LATCH_BYTES:
            self._memo_latch = True

    def load_short_binbytes(self):
        n = self.read(1)[0]
        self._note_buffer(n)
        self.append(self.read(n))
    dispatch[pickle.SHORT_BINBYTES[0]] = load_short_binbytes

    def load_binbytes(self):
        n = int.from_bytes(self.read(4), "little")
        self._note_buffer(n)
        self.append(self.read(n))
    dispatch[pickle.BINBYTES[0]] = load_binbytes

    def load_binbytes8(self):
        n = int.from_bytes(self.read(8), "little")
        self._note_buffer(n)
        self.append(self.read(n))
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8

    def _note_key(self, key):
        if key != self._cur_key:
            self._cur_key = key
            if self._on_key_change is not None:
                self._on_key_change(key)

    # While a dict value list is being filled the stack is [..., key, list_obj],
    # so the key immediately below the list identifies which dict entry it is.
    def load_append(self):
        stack = self.stack
        value = stack.pop()
        key = stack[-2] if len(stack) >= 2 and type(stack[-2]) is str else None
        if key in self._handlers:
            self._note_key(key)
            self._handlers[key](value)
        else:
            stack[-1].append(value)
    dispatch[pickle.APPEND[0]] = load_append

    def load_appends(self):
        items = self.pop_mark()
        stack = self.stack
        key = stack[-2] if len(stack) >= 2 and type(stack[-2]) is str else None
        if key in self._handlers:
            self._note_key(key)
            handler = self._handlers[key]
            for it in items:
                handler(it)
        else:
            stack[-1].extend(items)
    dispatch[pickle.APPENDS[0]] = load_appends

    def load_memoize(self):
        memo = self.memo
        obj = self.stack[-1]
        # The per-draw sample arrays (and their raw data bytes) are memoized but
        # never referenced again, so store a placeholder to keep memo indices
        # aligned while letting the object be freed. _StreamMemo decides the rest
        # -- and drops the slot entirely rather than holding a None.
        if not self._memo_latch and (type(obj) is np.ndarray
                                     or (type(obj) is bytes
                                         and len(obj) > _STREAM_MEMO_DROP_BYTES)):
            memo[len(memo)] = None
        else:
            memo[len(memo)] = obj
    dispatch[pickle.MEMOIZE[0]] = load_memoize


def stream_load(path, handlers, on_key_change=None):
    """Stream-load an MCMC result pickle (optionally zstd-compressed), diverting the
    large per-draw sample lists named in `handlers` to callbacks so peak memory
    stays a few GB. Returns the reconstructed object with those lists left empty."""
    if path.endswith(".zst"):
        import zstandard as zstd
        with open(path, "rb") as f:
            stream = io.BufferedReader(zstd.ZstdDecompressor().stream_reader(f))
            return _StreamUnpickler(stream, handlers, on_key_change).load()
    with open(path, "rb") as f:
        return _StreamUnpickler(f, handlers, on_key_change).load()


def median_ci(x, cred=0.95):
    """Posterior median and central (equal-tailed) credible interval of a 1-D
    sample, returned as (median, lower, upper). NaNs for an empty sample."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo_q = 100.0 * (1.0 - cred) / 2.0
    return (float(np.percentile(x, 50)),
            float(np.percentile(x, lo_q)),
            float(np.percentile(x, 100.0 - lo_q)))


# ---------------------------------------------------------------------------
# Accumulators fed by the streaming loader
# ---------------------------------------------------------------------------

class _SparseWindowAccumulator:
    """Chunked, low-memory posterior summaries of the sparse graph/weight draws.

    Graphs and weights stream in separate passes (all graphs, then all weights), so the
    window's graph keys are buffered until the matching weights arrive; draws are then
    reduced in chunks of `chunk`: the edge counts accumulate additively via bincount, and
    each edge's present weights are grouped within each chunk and appended, then sorted
    once per edge at finalize().

    finalize(S) returns (post, W_median, edges), where `edges` lists
    (parent_idx, child_idx, post_prob, n_present, sorted_present_weights) for every
    entry present in >= 1 window draw. W_median matches np.median over the dense
    per-draw matrices: entry (i,j)'s sample is its `n_present` weights padded with
    `S - n_present` structural zeros.

    `thin` keeps every thin-th draw of the window and skips the rest. The buffered
    graph keys are what bound memory here -- roughly 1.3e3 incidences per draw --
    so a window long enough to need a 3,000,000-draw burn-in only fits once
    thinned. Successive draws are in any case heavily autocorrelated, so thinning
    costs far less effective sample size than it does memory."""

    def __init__(self, n, start, end, chunk=100_000, thin=1):
        self.n = n
        self.start = start
        self.end = end
        self.chunk = chunk
        self.thin = thin
        self.nn = n * n
        self._gi = 0                          # running graph-draw index (all draws)
        self._wi = 0                          # running weight-draw index (all draws)
        self._gkeys = []                      # kept graph keys; [0] == draw `start`
        self.cnt  = np.zeros(self.nn, dtype=np.float64)
        self.edge_w = defaultdict(list)       # flat key -> list of per-chunk weights
        self._kbuf = []
        self._vbuf = []
        self._buf_draws = 0
        self.n_window_draws = 0

    def _kept(self, i):
        """Is draw `i` in the window and on the thinning grid? The two passes run
        this over the same draw indices, so they keep the same draws in the same
        order and pair up by position in _gkeys."""
        return self.start <= i < self.end and (i - self.start) % self.thin == 0

    # -- graphs pass: buffer flattened edge keys for kept draws --
    def add_graph(self, edges):
        i = self._gi
        self._gi += 1
        if self._kept(i):
            if edges.shape[0]:
                key = edges[:, 0].astype(np.int64) * self.n + edges[:, 1]
            else:
                key = np.empty(0, dtype=np.int64)
            self._gkeys.append(key)
            if len(self._gkeys) % 200_000 == 0:
                print(f"      [sparse] buffered {len(self._gkeys):,} graph draws ...",
                      flush=True)

    # -- weights pass: pair with buffered keys, reduce chunk-by-chunk --
    def add_weights(self, w):
        i = self._wi
        self._wi += 1
        if self._kept(i):
            idx = (i - self.start) // self.thin
            self._kbuf.append(self._gkeys[idx])
            self._gkeys[idx] = None                          # free as consumed
            self._vbuf.append(np.asarray(w, dtype=np.float32))
            self._buf_draws += 1
            self.n_window_draws += 1
            if self._buf_draws >= self.chunk:
                self._flush()

    def _flush(self):
        if self._buf_draws == 0:
            return
        key = np.concatenate(self._kbuf) if self._kbuf else np.empty(0, np.int64)
        val = np.concatenate(self._vbuf) if self._vbuf else np.empty(0, np.float32)
        self._kbuf.clear()
        self._vbuf.clear()
        self._buf_draws = 0
        self.cnt += np.bincount(key, minlength=self.nn)
        # Group this chunk's incidences by edge (sort by key only; weights are
        # sorted per edge at finalize) and append each edge's weights.
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        val_s = val[order]
        del key, val, order
        uniq, first, counts = np.unique(key_s, return_index=True, return_counts=True)
        ew = self.edge_w
        for g in range(uniq.size):
            c = int(counts[g])
            ew[int(uniq[g])].append(val_s[first[g]:first[g] + c].copy())
        print(f"      [sparse] reduced chunk -> {self.n_window_draws:,} window draws, "
              f"{len(ew):,} distinct edges so far", flush=True)

    def finalize(self, S):
        self._flush()
        n = self.n
        post = (self.cnt / S).reshape(n, n)
        W_median = np.zeros((n, n), dtype=np.float64)
        half = S // 2
        even = (S % 2 == 0)
        edges = []
        for flat_key, arrs in self.edge_w.items():
            v = np.concatenate(arrs).astype(np.float64)
            v.sort()                                          # ascending present weights
            c = v.size
            i, j = divmod(flat_key, n)
            # Median over the dense per-entry sample = v with (S - c) zeros mixed in.
            z = S - c
            n_neg = int(np.searchsorted(v, 0.0, side="left"))  # v[:n_neg] < 0 <= v[n_neg:]

            def order_stat(t, v=v, n_neg=n_neg, z=z):
                if t < n_neg:
                    return v[t]
                if t < n_neg + z:
                    return 0.0
                return v[t - z]

            W_median[i, j] = (0.5 * (order_stat(half - 1) + order_stat(half))
                              if even else order_stat(half))
            edges.append((i, j, c / S, c, v))
        return post, W_median, edges


class _RhoWindowCollector:
    """Collects the window's draws of alpha (rho_0 on the logit scale) and rho_logits
    (rho_1..rho_K) as they stream past, on the same grid as _SparseWindowAccumulator so
    every summary describes the same set of draws."""

    def __init__(self, start, end, thin=1):
        self.start = start
        self.end = end
        self.thin = thin
        self._ai = self._ri = 0
        self.alpha = []
        self.rho_logits = []

    def _kept(self, i):
        return self.start <= i < self.end and (i - self.start) % self.thin == 0

    def add_alpha(self, x):
        i = self._ai; self._ai += 1
        if self._kept(i):
            self.alpha.append(float(x))

    def add_rho_logits(self, x):
        i = self._ri; self._ri += 1
        if self._kept(i):
            self.rho_logits.append(np.asarray(x, dtype=np.float64))


def _discard(_x):
    """Handler for a sample list nothing here uses. It still has to be streamed: left to
    the unpickler, the list would be built in full."""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(path, out_dir, start, end, thin=1):
    size_mb = os.path.getsize(path) / 1e6
    print(f"streaming {os.path.basename(path)} ({size_mb:,.1f} MB on disk); "
          f"window [{start:,}:{end:,})"
          f"{f' thinned by {thin}' if thin > 1 else ''} ...", flush=True)

    # n flattens edge (i, j) -> i*n + j while graphs stream, but res["n_nodes"] only
    # appears at the END of the pickle. Take n from the prior matrix (same node set as
    # the run) and verify against res["n_nodes"] afterwards.
    n = pd.read_csv(config.PRIOR_PATH, index_col=0).shape[0]

    acc = _SparseWindowAccumulator(n, start, end, chunk=100_000, thin=thin)
    rho = _RhoWindowCollector(start, end, thin=thin)
    handlers = {
        "graphs_samples":     acc.add_graph,
        "W_samples":          acc.add_weights,
        "b_samples":          _discard,
        "log_psi_samples":    _discard,
        "alpha_samples":      rho.add_alpha,
        "rho_logits_samples": rho.add_rho_logits,
    }
    t0 = time.time()
    res = stream_load(
        path, handlers,
        on_key_change=lambda k: print(f"  streaming {k} ...", flush=True))
    print(f"streamed in {time.time() - t0:,.1f}s", flush=True)

    col_names = list(res["col_names"])
    if int(res["n_nodes"]) != n:
        raise ValueError(
            f"node-count mismatch: prior matrix has n={n} but run stored "
            f"n_nodes={int(res['n_nodes'])}")
    n_draws = acc._gi                          # total draws seen in the graphs pass
    S = acc.n_window_draws
    if S == 0:
        raise ValueError(f"window start ({start:,}) >= stored draws ({n_draws:,})")
    win_end = min(end, n_draws)
    if win_end < end:
        print(f"[warn] only {n_draws:,} draws stored; clamped window end "
              f"{end:,} -> {win_end:,}")
    print(f"{n_draws:,} draws stored; summarized window [{start:,}:{win_end:,}) "
          f"= {S:,} draws"
          f"{f' (every {thin}th of {win_end - start:,})' if thin > 1 else ''}")

    # --- PIPs and the marginal median of W ---
    post, W_median, edges = acc.finalize(S)
    del acc
    np.save(f"{out_dir}/posterior_dir.npy", post)
    np.save(f"{out_dir}/W_median.npy", W_median)

    # --- edge_recovery.csv: directed edges present in >= 1 window draw ---
    # w_post_* summarize the weight CONDITIONAL on the edge being present, so they
    # use only the present draws; NaN when fewer than W_MIN_PRESENT_DRAWS of them.
    rows = []
    for i, j, pp, c, v in edges:
        if i == j:
            continue
        if c >= W_MIN_PRESENT_DRAWS:
            wm, wlo, whi = median_ci(v)
        else:
            wm = wlo = whi = float("nan")
        rows.append((col_names[i], col_names[j], pp, wm, wlo, whi))
    edge_df = (pd.DataFrame(
        rows, columns=["parent", "child", "post_prob",
                       "w_post_median", "w_post_lower95", "w_post_upper95"])
        .sort_values("post_prob", ascending=False).reset_index(drop=True))
    edge_df.to_csv(f"{out_dir}/edge_recovery.csv", index=False)

    # --- rho draws: rho_0 = expit(alpha), rho_k = expit(rho_logits[:, k-1]) ---
    alpha      = np.asarray(rho.alpha, dtype=np.float64)                 # (S,)
    rho_logits = np.stack(rho.rho_logits).astype(np.float64)             # (S, K)
    np.save(f"{out_dir}/rho_samples.npy",
            np.hstack([expit(alpha)[:, None], expit(rho_logits)]))       # (S, K+1)

    # --- Node order for the figure scripts ---
    pd.DataFrame({"index": range(n), "node": col_names}).to_csv(
        f"{out_dir}/nodes.csv", index=False)

    print(f"{len(edge_df)} edges (post_prob > 0); saved to {out_dir}")
    del res, post, edges
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Posterior summaries of the real-data MCMC run.")
    parser.add_argument("--chain", default=config.CHAIN_PATH,
                        help="MCMC output (.pkl or .pkl.zst).")
    parser.add_argument("--start", type=int, default=config.BURN_IN,
                        help=f"First stored draw of the window (default {config.BURN_IN:,}).")
    parser.add_argument("--end", type=int, default=config.WIN_END,
                        help=f"One past the last stored draw (default {config.WIN_END:,}).")
    parser.add_argument("--thin", type=int, default=config.THIN,
                        help=f"Keep every THIN-th draw of the window (default {config.THIN}).")
    parser.add_argument("--out", default=config.EVAL_DIR,
                        help=f"Output directory (default {config.EVAL_DIR}).")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error(f"--start ({args.start}) must be < --end ({args.end})")
    if args.thin < 1:
        parser.error(f"--thin ({args.thin}) must be >= 1")

    os.makedirs(args.out, exist_ok=True)
    evaluate(args.chain, args.out, args.start, args.end, args.thin)


if __name__ == "__main__":
    main()
