"""CHIMERA CA v2: verified local-rule (cellular automaton) candidates for ARC-AGI-2.

Extends the v10 principle used in ``chimera_fill.ca_candidates`` (grid = texture, each cell
updated from its neighbourhood, rule induced from the demonstrations as a lookup table).
New feature families, each tried from coarse to fine (Occam):

* ``rel``    - colours encoded relative to the centre (same / background / other), output
               encoded as a role (keep centre, background, or an absolute colour);
* ``ray``    - for every direction, the first non-background colour seen along the ray;
* ``rowcol`` - whether each colour appears in the cell's row / column;
* ``m5``     - 5x5 colour counts;
* ``iter``   - the 3x3 rule iterated until a fixed point (the shader loop of CHIMERA v10).

A candidate is emitted only when it reproduces every training pair and a leave-one-out check
over the training pairs has no failure (same gate as v1).  Test outputs are never read.
"""
from __future__ import annotations

import numpy as np

PAD = 10
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _a(g):
    return np.array(g, dtype=int)


def _bg(xs):
    vals, cnt = np.unique(np.concatenate([x.ravel() for x in xs]), return_counts=True)
    return int(vals[cnt.argmax()])


def _win(a, r):
    p = np.pad(a, r, constant_values=PAD)
    h, w = a.shape
    k = 2 * r + 1
    return np.stack([p[dy:dy + h, dx:dx + w] for dy in range(k) for dx in range(k)], -1)


def _rays(a, bg, n_dirs=8):
    h, w = a.shape
    out = np.full((h, w, n_dirs), PAD, dtype=int)
    for d, (dy, dx) in enumerate(DIRS[:n_dirs]):
        for y in range(h):
            for x in range(w):
                yy, xx = y + dy, x + dx
                while 0 <= yy < h and 0 <= xx < w:
                    if a[yy, xx] != bg:
                        out[y, x, d] = a[yy, xx]
                        break
                    yy += dy
                    xx += dx
    return out


def _feat(a, mode, bg):
    """(H*W, F) integer feature matrix for every cell."""
    h, w = a.shape
    c = a.reshape(-1, 1)
    if mode == "rel3":
        win = _win(a, 1).reshape(h * w, 9)
        ctr = win[:, 4:5]
        rel = np.where(win == PAD, 3, np.where(win == ctr, 0, np.where(win == bg, 1, 2)))
        return np.concatenate([(ctr == bg).astype(int), rel], 1)
    if mode == "ray4":
        return np.concatenate([c, _rays(a, bg, 4).reshape(h * w, 4)], 1)
    if mode == "ray8":
        return np.concatenate([c, _rays(a, bg, 8).reshape(h * w, 8)], 1)
    if mode == "rowcol":
        rows = np.stack([(a == k).any(1) for k in range(10)], -1)  # (h,10)
        cols = np.stack([(a == k).any(0) for k in range(10)], -1)  # (w,10)
        rf = np.repeat(rows, w, 0)
        cf = np.tile(cols, (h, 1))
        return np.concatenate([c, rf.astype(int), cf.astype(int)], 1)
    if mode == "m5count":
        win = _win(a, 2).reshape(h * w, 25)
        return np.concatenate([c] + [(win == k).sum(1, keepdims=True) for k in range(11)], 1)
    if mode == "moore":
        return _win(a, 1).reshape(h * w, 9)
    raise ValueError(mode)


def _enc_out(x, y, bg, rel):
    xv, yv = x.reshape(-1), y.reshape(-1)
    if not rel:
        return yv.tolist()
    return [("k",) if a == b else ("b",) if b == bg else ("c", int(b)) for a, b in zip(xv, yv)]


def _dec_out(x, codes, bg):
    xv = x.reshape(-1)
    return np.array([a if c[0] == "k" else bg if c[0] == "b" else c[1] for a, c in zip(xv, codes)],
                    dtype=int).reshape(x.shape)


def _fit(pairs, mode, bg, rel):
    table = {}
    for x, y in pairs:
        keys = [tuple(r) for r in _feat(x, mode, bg).tolist()]
        for k, v in zip(keys, _enc_out(x, y, bg, rel)):
            if table.setdefault(k, v) != v:
                return None
    return table


def _apply(table, x, mode, bg, rel):
    codes = []
    for k in (tuple(r) for r in _feat(x, mode, bg).tolist()):
        v = table.get(k)
        if v is None:
            return None
        codes.append(v)
    return _dec_out(x, codes, bg) if rel else np.array(codes, dtype=int).reshape(x.shape)


def _iterate(table, x, mode, bg, rel, max_steps=30):
    cur = x
    for _ in range(max_steps):
        nxt = _apply(table, cur, mode, bg, rel)
        if nxt is None:
            return None
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt
    return None


def _fit_iter(pairs, mode, bg, rel):
    """Rule whose fixed point maps x to y: one-step table from (x, y) plus y being stable."""
    return _fit(list(pairs) + [(y, y) for _, y in pairs], mode, bg, rel)


# (name, feature mode, relative output, iterated)
FAMILIES = [
    ("rel3", "rel3", True, False),
    ("ray4", "ray4", True, False),
    ("ray8", "ray8", True, False),
    ("rowcol", "rowcol", True, False),
    ("m5count", "m5count", True, False),
    ("iter_rel3", "rel3", True, True),
    ("iter_moore", "moore", False, True),
]


def _predict(fam, pairs, x, bg):
    _, mode, rel, it = fam
    if it:
        t = _fit_iter(pairs, mode, bg, rel)
        return None if t is None else _iterate(t, x, mode, bg, rel)
    t = _fit(pairs, mode, bg, rel)
    return None if t is None else _apply(t, x, mode, bg, rel)


def ca2_candidates(task: dict, families=None) -> list[tuple[list, str]]:
    train = [(_a(p["input"]), _a(p["output"])) for p in task.get("train", [])]
    if len(train) < 2 or any(x.shape != y.shape for x, y in train):
        return []
    if all(np.array_equal(x, y) for x, y in train):
        return []
    tests = [_a(t["input"]) for t in task.get("test", [])]
    bg = _bg([x for x, _ in train] + tests)
    found, seen = [], set()
    for fam in (families or FAMILIES):
        # must reproduce every training pair
        fits = [_predict(fam, train, x, bg) for x, _ in train]
        if any(f is None or not np.array_equal(f, y) for f, (_, y) in zip(fits, train)):
            continue
        preds = [_predict(fam, train, x, bg) for x in tests]
        if any(p is None for p in preds):
            continue
        covered, ok = 0, True
        for i in range(len(train)):
            rest = train[:i] + train[i + 1:]
            p = _predict(fam, rest, train[i][0], bg)
            if p is None:
                continue
            covered += 1
            if not np.array_equal(p, train[i][1]):
                ok = False
                break
        if not ok or covered == 0:
            continue
        if any(np.array_equal(p, x) for p, x in zip(preds, tests)):
            continue
        key = str([p.tolist() for p in preds])
        if key not in seen:
            seen.add(key)
            found.append(([p.tolist() for p in preds], f"ca2:{fam[0]}"))
    return found
