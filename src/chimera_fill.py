"""ARC2_CHIMERA high-precision candidate generator for ARC-AGI-2.

Two sources, both from the ARC2_CHIMERA repository (Agnuxo1/ARC2_CHIMERA, Apache-2.0):

* ``ca``  - NumPy port of the CHIMERA v10 principle: the grid is a texture whose cells are
  updated from their 3x3 neighbourhood (SPATIAL_OPERATOR_SHADER).  Here the local update rule
  is not hand-written but induced from the demonstrations as a lookup table
  neighbourhood -> output colour.  OpenGL is not guaranteed in the Kaggle notebook, so the
  same local-rule principle runs on the CPU.
* ``dsl`` - exact programs of the CHIMERA v8 symbolic search (``chimera_dsl.py``, copied
  unchanged from ``arc2_chimera/arc_solver.py``).

A candidate is emitted only when it is *verified*: it reproduces every training pair and a
leave-one-out check over the training pairs has no failure.  The generator never reads test
outputs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import chimera_dsl as dsl

PAD = 10  # outside-the-grid marker

# Neighbourhood encodings, from coarse (more general) to fine.
CA_MODES = ("vn_count", "vn", "moore_count", "moore")


def _windows(a: np.ndarray) -> np.ndarray:
    """(H, W, 9) array with the 3x3 neighbourhood of every cell, row-major, centre at 4."""
    p = np.pad(a, 1, constant_values=PAD)
    h, w = a.shape
    return np.stack([p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], -1)


def _keys(a: np.ndarray, mode: str) -> list[tuple]:
    win = _windows(a)
    h, w = a.shape
    if mode == "moore":
        flat = win.reshape(h * w, 9)
    elif mode == "vn":
        flat = win[..., [1, 3, 4, 5, 7]].reshape(h * w, 5)
    elif mode in ("moore_count", "vn_count"):
        idx = [0, 1, 2, 3, 5, 6, 7, 8] if mode == "moore_count" else [1, 3, 5, 7]
        nb = win[..., idx].reshape(h * w, len(idx))
        counts = np.stack([(nb == c).sum(-1) for c in range(11)], -1)
        flat = np.concatenate([win[..., 4].reshape(h * w, 1), counts], -1)
    else:
        raise ValueError(mode)
    return [tuple(r) for r in flat.tolist()]


def _fit_table(pairs: Sequence[tuple[np.ndarray, np.ndarray]], mode: str) -> dict | None:
    table: dict = {}
    for x, y in pairs:
        for k, v in zip(_keys(x, mode), y.reshape(-1).tolist()):
            if table.setdefault(k, v) != v:
                return None
    return table


def _apply_table(table: dict, x: np.ndarray, mode: str) -> np.ndarray | None:
    out = []
    for k in _keys(x, mode):
        v = table.get(k)
        if v is None:
            return None  # unseen neighbourhood: no prediction
        out.append(v)
    return np.array(out, dtype=int).reshape(x.shape)


def ca_candidates(task: dict) -> list[tuple[list[list[list[int]]], str]]:
    """Verified local-rule (cellular automaton) predictions: list of (outputs per test, tag)."""
    train = [(dsl._a(p["input"]), dsl._a(p["output"])) for p in task.get("train", [])]
    if len(train) < 2 or any(x.shape != y.shape for x, y in train):
        return []
    if all(np.array_equal(x, y) for x, y in train):
        return []
    tests = [dsl._a(t["input"]) for t in task.get("test", [])]
    found = []
    for mode in CA_MODES:
        table = _fit_table(train, mode)
        if table is None:
            continue
        preds = [_apply_table(table, x, mode) for x in tests]
        if any(p is None for p in preds):
            continue
        # leave-one-out: a held-out pair that is fully covered must be reproduced exactly
        covered = 0
        ok = True
        for i in range(len(train)):
            rest = train[:i] + train[i + 1:]
            t = _fit_table(rest, mode)
            if t is None:
                ok = False
                break
            p = _apply_table(t, train[i][0], mode)
            if p is None:
                continue
            covered += 1
            if not np.array_equal(p, train[i][1]):
                ok = False
                break
        if not ok or covered == 0:
            continue
        if any(np.array_equal(p, x) for p, x in zip(preds, tests)):
            continue  # a no-op on the test input is not a prediction
        found.append(([p.tolist() for p in preds], f"ca:{mode}"))
        break  # coarsest verified rule wins (Occam)
    return found


def dsl_candidates(task: dict, solver: dsl.ChimeraArcSolver | None = None) -> list[tuple[list, str]]:
    """Exact CHIMERA v8 programs whose leave-one-out score is perfect."""
    train = task.get("train", [])
    if len(train) < 2:
        return []
    solver = solver or dsl.ChimeraArcSolver()
    shape_rule = dsl._shape_rule(train)
    exact = []
    for cand in solver.candidates + solver._translation_candidates(train):
        score, mapping = solver._score(cand, train)
        if mapping is None or score != 1.0:
            continue
        if solver._holdout_score(cand, train) < 1.0:
            continue
        exact.append((cand.complexity, cand.name, cand, mapping))
    found, seen = [], set()
    for _, name, cand, mapping in sorted(exact, key=lambda e: (e[0], e[1])):
        outs = []
        for t in task.get("test", []):
            x = dsl._a(t["input"])
            outs.append(dsl._apply_map(cand.op(x, shape_rule(x.shape)), mapping).astype(int).tolist())
        if all(o == t["input"] for o, t in zip(outs, task["test"])):
            continue  # identity on the test input
        key = str(outs)
        if key not in seen:
            seen.add(key)
            found.append((outs, f"dsl:{name}"))
    return found


def chimera_candidates(task: dict, solver: dsl.ChimeraArcSolver | None = None) -> list[tuple[list, str]]:
    """All verified CHIMERA candidates, CA first, then DSL, without duplicates."""
    out, seen = [], set()
    for outs, tag in ca_candidates(task) + dsl_candidates(task, solver):
        if str(outs) not in seen:
            seen.add(str(outs))
            out.append((outs, tag))
    return out


def _valid(g) -> bool:
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1)


def fill_submission(submission: dict, tasks: dict, policy: str = "free_slot", log: list | None = None) -> dict:
    """Insert verified CHIMERA candidates into ``submission`` (Kaggle format).

    policy ``free_slot``: only replace attempt_2 when it is missing, invalid or equal to attempt_1
    (the control can never lose a correct answer).  ``override``: also replace a distinct attempt_2.
    """
    solver = dsl.ChimeraArcSolver()
    for tid, task in tasks.items():
        try:
            cands = chimera_candidates(task, solver)
        except Exception:  # never let the add-on break the submission
            continue
        if not cands:
            continue
        outs, tag = cands[0]
        for i, entry in enumerate(submission.get(tid, [])):
            if i >= len(outs):
                break
            a1, a2 = entry.get("attempt_1"), entry.get("attempt_2")
            cand = outs[i]
            if not _valid(cand) or cand == a1:
                continue
            free = (not _valid(a2)) or a2 == a1
            if free or policy == "override":
                entry["attempt_2"] = cand
                if log is not None:
                    log.append({"task": tid, "test": i, "source": tag, "slot_was_free": free})
    return submission
