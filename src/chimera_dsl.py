"""Deterministic CPU program-synthesis backend for ARC-AGI-2.

The OpenGL experiments remain useful as CHIMERA research infrastructure, but
ARC submissions need a small, inspectable symbolic path.  This module searches
over transformations that preserve colors, learns a color permutation from
the demonstrations, and accepts a program only when it reproduces every
training output exactly.  A scored fallback is kept for the second/third
attempts, but it is never presented as a verified solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Sequence

import numpy as np

Grid = np.ndarray
Op = Callable[[Grid, tuple[int, int]], Grid]


def _a(grid: Sequence[Sequence[int]]) -> Grid:
    return np.asarray(grid, dtype=np.int16)


def _copy_shape(a: Grid, shape: tuple[int, int]) -> Grid:
    out = np.zeros(shape, dtype=np.int16)
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def _crop_or_pad(a: Grid, shape: tuple[int, int], anchor: str = "tl") -> Grid:
    h, w = shape
    out = np.zeros(shape, dtype=np.int16)
    sy = max(0, (a.shape[0] - h) // 2) if anchor == "center" else max(0, a.shape[0] - h) if anchor == "br" else 0
    sx = max(0, (a.shape[1] - w) // 2) if anchor == "center" else max(0, a.shape[1] - w) if anchor == "br" else 0
    dy = max(0, (h - a.shape[0]) // 2) if anchor == "center" else max(0, h - a.shape[0]) if anchor == "br" else 0
    dx = max(0, (w - a.shape[1]) // 2) if anchor == "center" else max(0, w - a.shape[1]) if anchor == "br" else 0
    hh = min(a.shape[0] - sy, h - dy)
    ww = min(a.shape[1] - sx, w - dx)
    if hh > 0 and ww > 0:
        out[dy : dy + hh, dx : dx + ww] = a[sy : sy + hh, sx : sx + ww]
    return out


def _center_object(a: Grid, shape: tuple[int, int]) -> Grid:
    box = _bbox(a, 0)
    if box is None:
        return np.zeros(shape, dtype=np.int16)
    obj = a[box[0] : box[2], box[1] : box[3]]
    out = np.zeros(shape, dtype=np.int16)
    y = max(0, (shape[0] - obj.shape[0]) // 2)
    x = max(0, (shape[1] - obj.shape[1]) // 2)
    hh = min(obj.shape[0], shape[0] - y)
    ww = min(obj.shape[1], shape[1] - x)
    out[y : y + hh, x : x + ww] = obj[:hh, :ww]
    return out


def _resize_nearest(a: Grid, shape: tuple[int, int]) -> Grid:
    if a.size == 0 or min(shape) <= 0:
        return np.zeros(shape, dtype=np.int16)
    ys = np.minimum((np.arange(shape[0]) * a.shape[0]) // shape[0], a.shape[0] - 1)
    xs = np.minimum((np.arange(shape[1]) * a.shape[1]) // shape[1], a.shape[1] - 1)
    return a[np.ix_(ys, xs)]


def _tile_to(a: Grid, shape: tuple[int, int]) -> Grid:
    if a.size == 0:
        return np.zeros(shape, dtype=np.int16)
    return np.tile(a, (int(np.ceil(shape[0] / a.shape[0])), int(np.ceil(shape[1] / a.shape[1]))))[: shape[0], : shape[1]]


def _periodic_core(a: Grid, shape: tuple[int, int]) -> Grid:
    for h in range(1, a.shape[0] + 1):
        if a.shape[0] % h:
            continue
        for w in range(1, a.shape[1] + 1):
            if a.shape[1] % w:
                continue
            tile = a[:h, :w]
            if np.array_equal(np.tile(tile, (a.shape[0] // h, a.shape[1] // w)), a):
                return tile.copy()
    return a.copy()


def _trim_uniform_border(a: Grid, shape: tuple[int, int]) -> Grid:
    out = a.copy()
    changed = True
    while changed and min(out.shape) > 1:
        changed = False
        if len(set(out[0].tolist())) == 1 and np.all(out[0] == out[0, 0]):
            out = out[1:]
            changed = True
        if min(out.shape) > 1 and len(set(out[-1].tolist())) == 1 and np.all(out[-1] == out[-1, 0]):
            out = out[:-1]
            changed = True
        if min(out.shape) > 1 and len(set(out[:, 0].tolist())) == 1 and np.all(out[:, 0] == out[0, 0]):
            out = out[:, 1:]
            changed = True
        if min(out.shape) > 1 and len(set(out[:, -1].tolist())) == 1 and np.all(out[:, -1] == out[0, -1]):
            out = out[:, :-1]
            changed = True
    return out.copy()


def _inner_border(a: Grid, shape: tuple[int, int]) -> Grid:
    if min(a.shape) <= 2:
        return np.zeros((1, 1), dtype=np.int16)
    return a[1:-1, 1:-1].copy()


def _bbox(a: Grid, bg: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(a != bg)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _components(a: Grid, diagonals: bool = False) -> list[tuple[int, int, int, int, int, Grid]]:
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonals:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    result = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or a[y, x] == 0:
                continue
            color = int(a[y, x])
            stack = [(y, x)]
            seen[y, x] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            y0, x0, y1, x1 = min(ys), min(xs), max(ys) + 1, max(xs) + 1
            crop = np.zeros((y1 - y0, x1 - x0), dtype=np.int16)
            for py, px in pts:
                crop[py - y0, px - x0] = color
            result.append((len(pts), y0, x0, y1, x1, crop))
    return result


def _components_any(a: Grid, diagonals: bool = False) -> list[tuple[int, int, int, int, int, Grid]]:
    """Connected components treating all non-zero colors as one object."""
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonals:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    result = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or a[y, x] == 0:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] != 0:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            y0, x0, y1, x1 = min(ys), min(xs), max(ys) + 1, max(xs) + 1
            result.append((len(pts), y0, x0, y1, x1, a[y0:y1, x0:x1].copy()))
    return result


def _paint_component(out: Grid, crop: Grid, y0: int, x0: int) -> None:
    """Paint a component at a possibly clipped destination."""
    sy0, sy1 = max(0, -y0), min(crop.shape[0], out.shape[0] - y0)
    sx0, sx1 = max(0, -x0), min(crop.shape[1], out.shape[1] - x0)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    source = crop[sy0:sy1, sx0:sx1]
    target = out[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1]
    mask = source != 0
    target[mask] = source[mask]


def _relative_objects(a: Grid, shape: tuple[int, int], placement: str, diagonals: bool = False) -> Grid:
    """Re-render objects while preserving their relative placement.

    The operator removes absolute canvas position from the object arrangement,
    then aligns the arrangement to the requested target anchor.  It is kept
    deliberately geometric: no task-specific object labels or learned constants
    are introduced.
    """
    components = _components_any(a, diagonals)
    out = np.zeros(shape, dtype=np.int16)
    if not components:
        return out

    min_y = min(component[1] for component in components)
    min_x = min(component[2] for component in components)
    max_y = max(component[3] for component in components)
    max_x = max(component[4] for component in components)
    layout_h, layout_w = max_y - min_y, max_x - min_x

    if placement == "center":
        shift_y = (shape[0] - layout_h) // 2 - min_y
        shift_x = (shape[1] - layout_w) // 2 - min_x
    elif placement == "bottom_right":
        shift_y = shape[0] - max_y
        shift_x = shape[1] - max_x
    else:
        shift_y = -min_y
        shift_x = -min_x

    for _, y0, x0, _, _, crop in sorted(components, key=lambda c: (c[1], c[2], -c[0])):
        _paint_component(out, crop, y0 + shift_y, x0 + shift_x)
    return out


def _relative_to_anchor(a: Grid, shape: tuple[int, int], anchor: str, diagonals: bool = False) -> Grid:
    """Align a whole object arrangement by its largest object anchor."""
    components = _components_any(a, diagonals)
    out = np.zeros(shape, dtype=np.int16)
    if not components:
        return out
    anchor_component = max(components, key=lambda c: (c[0], -c[1], -c[2]))
    ay0, ax0, ay1, ax1 = anchor_component[1:5]
    if anchor == "center":
        anchor_y = (ay0 + ay1 - 1) / 2.0
        anchor_x = (ax0 + ax1 - 1) / 2.0
        shift_y = int(round((shape[0] - 1) / 2.0 - anchor_y))
        shift_x = int(round((shape[1] - 1) / 2.0 - anchor_x))
    elif anchor == "bottom_right":
        shift_y = shape[0] - ay1
        shift_x = shape[1] - ax1
    else:
        shift_y = -ay0
        shift_x = -ax0
    for _, y0, x0, _, _, crop in sorted(components, key=lambda c: (c[1], c[2], -c[0])):
        _paint_component(out, crop, y0 + shift_y, x0 + shift_x)
    return out


def _nonzero_crop(a: Grid, shape: tuple[int, int]) -> Grid:
    box = _bbox(a, 0)
    return a[box[0] : box[2], box[1] : box[3]].copy() if box else np.zeros((1, 1), dtype=np.int16)


def _largest(a: Grid, shape: tuple[int, int], diagonals: bool = False) -> Grid:
    cs = _components(a, diagonals)
    if not cs:
        return np.zeros((1, 1), dtype=np.int16)
    return max(cs, key=lambda c: (c[0], -c[1], -c[2]))[-1].copy()


def _smallest(a: Grid, shape: tuple[int, int], diagonals: bool = False) -> Grid:
    cs = _components(a, diagonals)
    if not cs:
        return np.zeros((1, 1), dtype=np.int16)
    return min(cs, key=lambda c: (c[0], c[1], c[2]))[-1].copy()


def _largest_any(a: Grid, shape: tuple[int, int], diagonals: bool = False) -> Grid:
    cs = _components_any(a, diagonals)
    return max(cs, key=lambda c: (c[0], -c[1], -c[2]))[-1].copy() if cs else np.zeros((1, 1), dtype=np.int16)


def _smallest_any(a: Grid, shape: tuple[int, int], diagonals: bool = False) -> Grid:
    cs = _components_any(a, diagonals)
    return min(cs, key=lambda c: (c[0], c[1], c[2]))[-1].copy() if cs else np.zeros((1, 1), dtype=np.int16)


def _compress(a: Grid, shape: tuple[int, int]) -> Grid:
    rows = np.any(a != 0, axis=1)
    cols = np.any(a != 0, axis=0)
    return a[rows][:, cols].copy() if rows.any() and cols.any() else np.zeros((1, 1), dtype=np.int16)


def _gravity(a: Grid, shape: tuple[int, int], direction: str) -> Grid:
    out = np.zeros_like(a)
    if direction in ("down", "up"):
        for x in range(a.shape[1]):
            vals = a[:, x][a[:, x] != 0]
            if len(vals) == 0:
                continue
            if direction == "down":
                out[-len(vals) :, x] = vals
            else:
                out[: len(vals), x] = vals
    else:
        for y in range(a.shape[0]):
            vals = a[y][a[y] != 0]
            if len(vals) == 0:
                continue
            if direction == "right":
                out[y, -len(vals) :] = vals
            else:
                out[y, : len(vals)] = vals
    return out


def _translate(a: Grid, shape: tuple[int, int], dy: int, dx: int) -> Grid:
    out = np.zeros(shape, dtype=np.int16)
    sy0, sy1 = max(0, -dy), min(a.shape[0], shape[0] - dy)
    sx0, sx1 = max(0, -dx), min(a.shape[1], shape[1] - dx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = a[sy0:sy1, sx0:sx1]
    return out


def _centroid_nonzero(a: Grid) -> tuple[float, float] | None:
    ys, xs = np.where(a != 0)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _mirror(a: Grid, shape: tuple[int, int], axis: str) -> Grid:
    out = a.copy()
    if axis == "h":
        for y in range(a.shape[0]):
            for x in range(a.shape[1] // 2):
                if out[y, x] == 0:
                    out[y, x] = out[y, -x - 1]
                if out[y, -x - 1] == 0:
                    out[y, -x - 1] = out[y, x]
    else:
        for y in range(a.shape[0] // 2):
            for x in range(a.shape[1]):
                if out[y, x] == 0:
                    out[y, x] = out[-y - 1, x]
                if out[-y - 1, x] == 0:
                    out[-y - 1, x] = out[y, x]
    return out


def _frame(a: Grid, shape: tuple[int, int]) -> Grid:
    out = np.zeros_like(a)
    if a.size:
        out[0, :] = a[0, :]
        out[-1, :] = a[-1, :]
        out[:, 0] = a[:, 0]
        out[:, -1] = a[:, -1]
    return out


def _fill_holes(a: Grid, shape: tuple[int, int]) -> Grid:
    out = a.copy()
    for y in range(1, a.shape[0] - 1):
        for x in range(1, a.shape[1] - 1):
            if a[y, x] == 0:
                n = a[y - 1 : y + 2, x - 1 : x + 2]
                colors = n[n != 0]
                if len(colors) and np.all(colors == colors[0]):
                    out[y, x] = colors[0]
    return out


def _keep_color(a: Grid, shape: tuple[int, int], color: int) -> Grid:
    return np.where(a == color, a, 0).astype(np.int16)


def _extract_color(a: Grid, shape: tuple[int, int], color: int) -> Grid:
    ys, xs = np.where(a == color)
    return a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy() if len(ys) else np.zeros((1, 1), dtype=np.int16)


def _remove_color(a: Grid, shape: tuple[int, int], color: int) -> Grid:
    return np.where(a == color, 0, a).astype(np.int16)


def _connect_lines(a: Grid, shape: tuple[int, int], diagonal: bool = False) -> Grid:
    out = a.copy()
    for color in sorted(int(x) for x in np.unique(a) if x != 0):
        for y in range(a.shape[0]):
            xs = np.where(a[y] == color)[0]
            if len(xs) >= 2:
                out[y, xs.min() : xs.max() + 1] = color
        for x in range(a.shape[1]):
            ys = np.where(a[:, x] == color)[0]
            if len(ys) >= 2:
                out[ys.min() : ys.max() + 1, x] = color
        if diagonal:
            ys, xs = np.where(a == color)
            for key in (ys - xs, ys + xs):
                for value in np.unique(key):
                    idx = np.where(key == value)[0]
                    if len(idx) >= 2:
                        lo, hi = int(idx[np.argmin(ys[idx])]), int(idx[np.argmax(ys[idx])])
                        y0, y1 = int(ys[lo]), int(ys[hi])
                        x0, x1 = int(xs[lo]), int(xs[hi])
                        n = abs(y1 - y0) + 1
                        out[np.linspace(y0, y1, n, dtype=int), np.linspace(x0, x1, n, dtype=int)] = color
    return out


def _symmetry_union(a: Grid, shape: tuple[int, int], axis: str) -> Grid:
    reflected = np.fliplr(a) if axis == "h" else np.flipud(a)
    out = a.copy()
    empty = out == 0
    out[empty] = reflected[empty]
    return out


def _bbox_shape(a: Grid, shape: tuple[int, int], fill: bool = False) -> Grid:
    box = _bbox(a, 0)
    out = np.zeros_like(a)
    if box is None:
        return out
    y0, x0, y1, x1 = box
    if fill:
        colors = a[a != 0]
        out[y0:y1, x0:x1] = int(colors[0]) if len(colors) else 0
    else:
        out[y0, x0:x1] = a[y0, x0:x1]
        out[y1 - 1, x0:x1] = a[y1 - 1, x0:x1]
        out[y0:y1, x0] = a[y0:y1, x0]
        out[y0:y1, x1 - 1] = a[y0:y1, x1 - 1]
    return out


@dataclass(frozen=True)
class Candidate:
    name: str
    op: Op
    complexity: int


def _base_candidates() -> list[Candidate]:
    def fixed(name: str, fn: Callable[[Grid], Grid], complexity: int = 1) -> Candidate:
        return Candidate(name, lambda a, shape: fn(a), complexity)

    c = [
        fixed("identity", lambda a: a.copy()),
        fixed("rot90", lambda a: np.rot90(a, 1).copy()),
        fixed("rot180", lambda a: np.rot90(a, 2).copy()),
        fixed("rot270", lambda a: np.rot90(a, 3).copy()),
        fixed("flip_h", np.fliplr),
        fixed("flip_v", np.flipud),
        fixed("transpose", lambda a: a.T.copy()),
        fixed("anti_diag", lambda a: np.fliplr(a.T).copy()),
        Candidate("crop_nonzero", _nonzero_crop, 2),
        Candidate("compress", _compress, 2),
        Candidate("largest4", lambda a, s: _largest(a, s, False), 3),
        Candidate("largest8", lambda a, s: _largest(a, s, True), 3),
        Candidate("smallest4", lambda a, s: _smallest(a, s, False), 3),
        Candidate("smallest8", lambda a, s: _smallest(a, s, True), 3),
        Candidate("largest_any4", lambda a, s: _largest_any(a, s, False), 3),
        Candidate("largest_any8", lambda a, s: _largest_any(a, s, True), 3),
        Candidate("smallest_any4", lambda a, s: _smallest_any(a, s, False), 3),
        Candidate("smallest_any8", lambda a, s: _smallest_any(a, s, True), 3),
        Candidate("gravity_down", lambda a, s: _gravity(a, s, "down"), 2),
        Candidate("gravity_up", lambda a, s: _gravity(a, s, "up"), 2),
        Candidate("gravity_left", lambda a, s: _gravity(a, s, "left"), 2),
        Candidate("gravity_right", lambda a, s: _gravity(a, s, "right"), 2),
        Candidate("mirror_h", lambda a, s: _mirror(a, s, "h"), 2),
        Candidate("mirror_v", lambda a, s: _mirror(a, s, "v"), 2),
        Candidate("frame", _frame, 2),
        Candidate("fill_holes", _fill_holes, 2),
        Candidate("connect", lambda a, s: _connect_lines(a, s, False), 2),
        Candidate("connect_diagonal", lambda a, s: _connect_lines(a, s, True), 3),
        Candidate("symmetry_h", lambda a, s: _symmetry_union(a, s, "h"), 2),
        Candidate("symmetry_v", lambda a, s: _symmetry_union(a, s, "v"), 2),
        Candidate("bbox_outline", lambda a, s: _bbox_shape(a, s, False), 2),
        Candidate("bbox_fill", lambda a, s: _bbox_shape(a, s, True), 2),
        Candidate("resize", _resize_nearest, 2),
        Candidate("tile", _tile_to, 2),
        Candidate("periodic_core", _periodic_core, 2),
        Candidate("trim_uniform_border", _trim_uniform_border, 2),
        Candidate("inner_border", _inner_border, 2),
        Candidate("center_object", _center_object, 2),
        Candidate("relative_objects_tl", lambda a, s: _relative_objects(a, s, "top_left"), 3),
        Candidate("relative_objects_center", lambda a, s: _relative_objects(a, s, "center"), 3),
        Candidate("relative_objects_br", lambda a, s: _relative_objects(a, s, "bottom_right"), 3),
        Candidate("relative_anchor_tl", lambda a, s: _relative_to_anchor(a, s, "top_left"), 3),
        Candidate("relative_anchor_center", lambda a, s: _relative_to_anchor(a, s, "center"), 3),
        Candidate("relative_anchor_br", lambda a, s: _relative_to_anchor(a, s, "bottom_right"), 3),
        Candidate("crop_tl", lambda a, s: _crop_or_pad(a, s, "tl"), 2),
        Candidate("crop_center", lambda a, s: _crop_or_pad(a, s, "center"), 2),
        Candidate("crop_br", lambda a, s: _crop_or_pad(a, s, "br"), 2),
    ]
    # A few low-cost shape constructors cover common ARC counting tasks.
    c.extend([
        Candidate("count_nonzero", lambda a, s: np.array([[int(np.count_nonzero(a))]], dtype=np.int16), 3),
        Candidate("count_colors", lambda a, s: np.array([[int(len(set(a.reshape(-1)) - {0}))]], dtype=np.int16), 3),
        Candidate("count_objects", lambda a, s: np.array([[int(len(_components(a)))]], dtype=np.int16), 3),
    ])
    for color in range(1, 10):
        c.append(Candidate(f"keep_color_{color}", lambda a, s, color=color: _keep_color(a, s, color), 2))
        c.append(Candidate(f"extract_color_{color}", lambda a, s, color=color: _extract_color(a, s, color), 2))
        c.append(Candidate(f"remove_color_{color}", lambda a, s, color=color: _remove_color(a, s, color), 2))
    return c


def _shape_rule(train: Sequence[dict]) -> Callable[[tuple[int, int]], tuple[int, int]]:
    pairs = [(_a(x["input"]).shape, _a(x["output"]).shape) for x in train]
    if not pairs:
        return lambda s: s
    outs = [o for _, o in pairs]
    if all(o == outs[0] for o in outs):
        return lambda s: outs[0]
    deltas = [(o[0] - i[0], o[1] - i[1]) for i, o in pairs]
    if len(set(deltas)) == 1:
        dh, dw = deltas[0]
        return lambda s: (max(1, s[0] + dh), max(1, s[1] + dw))
    ratios = []
    for (ih, iw), (oh, ow) in pairs:
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            break
        ratios.append((oh // ih, ow // iw))
    if ratios and len(set(ratios)) == 1:
        rh, rw = ratios[0]
        return lambda s: (s[0] * rh, s[1] * rw)

    # A single affine rule per dimension covers shape changes such as
    # ``2 * height + 1`` or ``width + 1`` that are not constant deltas or
    # integer scale ratios.  Fractions keep the inference exact and require
    # every training pair to agree before the rule is used.
    if len(pairs) >= 2:
        coefficients: list[tuple[Fraction, Fraction]] = []
        for dimension in (0, 1):
            coefficient: tuple[Fraction, Fraction] | None = None
            for first_index, (first_input, first_output) in enumerate(pairs):
                for second_input, second_output in pairs[first_index + 1 :]:
                    if first_input[dimension] == second_input[dimension]:
                        continue
                    slope = Fraction(
                        first_output[dimension] - second_output[dimension],
                        first_input[dimension] - second_input[dimension],
                    )
                    intercept = Fraction(first_output[dimension]) - slope * first_input[dimension]
                    coefficient = (slope, intercept)
                    break
                if coefficient is not None:
                    break
            if coefficient is None:
                break
            slope, intercept = coefficient
            valid = True
            for input_shape, output_shape in pairs:
                value = slope * input_shape[dimension] + intercept
                if value.denominator != 1 or value < 1 or value != output_shape[dimension]:
                    valid = False
                    break
            if not valid:
                break
            coefficients.append(coefficient)
        else:
            return lambda s: (
                int(coefficients[0][0] * s[0] + coefficients[0][1]),
                int(coefficients[1][0] * s[1] + coefficients[1][1]),
            )
    return lambda s: s


def _infer_color_map(base: Sequence[Grid], target: Sequence[Grid]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for a, b in zip(base, target):
        if a.shape != b.shape:
            return None
        for x, y in zip(a.reshape(-1), b.reshape(-1)):
            xi, yi = int(x), int(y)
            if xi in mapping and mapping[xi] != yi:
                return None
            mapping[xi] = yi
    return mapping


def _apply_map(a: Grid, mapping: dict[int, int]) -> Grid:
    out = a.copy()
    for x, y in mapping.items():
        out[a == x] = y
    return out


def _exact_program(cand: Candidate, train: Sequence[dict]) -> tuple[dict[int, int], list[Grid]] | None:
    expected = [_a(x["output"]) for x in train]
    base = [cand.op(_a(x["input"]), y.shape) for x, y in zip(train, expected)]
    mapping = _infer_color_map(base, expected)
    if mapping is None:
        return None
    predicted = [_apply_map(x, mapping) for x in base]
    if all(np.array_equal(x, y) for x, y in zip(predicted, expected)):
        return mapping, base
    return None


def _compose(a: Candidate, b: Candidate) -> Candidate:
    return Candidate(
        f"{a.name}>{b.name}",
        lambda x, s: b.op(a.op(x, s), s),
        a.complexity + b.complexity + 1,
    )


class ChimeraArcSolver:
    """Exact-first solver with deterministic ranked fallback candidates."""

    def __init__(self, max_compositions: int = 90):
        self.base = _base_candidates()
        geom = [x for x in self.base if x.name in {"identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_diag"}]
        structural = [x for x in self.base if x.name in {"crop_nonzero", "compress", "largest4", "largest8", "smallest4", "smallest8", "largest_any4", "largest_any8", "smallest_any4", "smallest_any8", "gravity_down", "gravity_up", "gravity_left", "gravity_right", "mirror_h", "mirror_v", "fill_holes", "connect", "symmetry_h", "symmetry_v", "periodic_core", "trim_uniform_border", "inner_border", "center_object", "resize", "tile"}]
        self.candidates = list(self.base)
        self.candidates.extend(_compose(g, s) for g in geom for s in structural)
        self.candidates.extend(_compose(s, g) for s in structural[:8] for g in geom[:6])
        self.candidates = sorted(self.candidates, key=lambda x: (x.complexity, x.name))[: max_compositions + len(self.base)]

    @staticmethod
    def _score(cand: Candidate, train: Sequence[dict]) -> tuple[float, dict[int, int] | None]:
        expected = [_a(x["output"]) for x in train]
        base = [cand.op(_a(x["input"]), y.shape) for x, y in zip(train, expected)]
        mapping = _infer_color_map(base, expected)
        if mapping is None:
            return 0.0, None
        good = 0
        total = 0
        for p, y in zip(base, expected):
            q = _apply_map(p, mapping)
            if q.shape == y.shape:
                good += int(np.sum(q == y))
                total += int(y.size)
        return (good / total if total else 0.0), mapping

    @staticmethod
    def _holdout_score(cand: Candidate, train: Sequence[dict]) -> float:
        """Estimate structural generalization among exact-fit candidates.

        Each demonstration is held out in turn.  The color map is inferred only
        from the remaining demonstrations, so this tie-breaker rewards a
        candidate whose transformation is stable across examples instead of
        changing the operator vocabulary or consulting test labels.
        """
        if len(train) < 2:
            return 0.0
        good = 0
        for index, holdout in enumerate(train):
            fit = train[:index] + train[index + 1 :]
            fit_expected = [_a(example["output"]) for example in fit]
            fit_base = [cand.op(_a(example["input"]), expected.shape) for example, expected in zip(fit, fit_expected)]
            mapping = _infer_color_map(fit_base, fit_expected)
            if mapping is None:
                continue
            expected = _a(holdout["output"])
            predicted = _apply_map(cand.op(_a(holdout["input"]), expected.shape), mapping)
            good += int(np.array_equal(predicted, expected))
        return good / len(train)

    @staticmethod
    def _translation_candidates(train: Sequence[dict]) -> list[Candidate]:
        deltas = []
        for example in train:
            ci = _centroid_nonzero(_a(example["input"]))
            co = _centroid_nonzero(_a(example["output"]))
            if ci is not None and co is not None:
                dy = int(round(co[0] - ci[0]))
                dx = int(round(co[1] - ci[1]))
                deltas.append((dy, dx))
        if not deltas or len(set(deltas)) != 1 or deltas[0] == (0, 0):
            return []
        dy, dx = deltas[0]
        return [Candidate(f"translate_{dy}_{dx}", lambda a, s, dy=dy, dx=dx: _translate(a, s, dy, dx), 2)]

    def solve_task(self, task: dict, attempts: int = 3) -> list[list[list[list[int]]]]:
        train = task.get("train", [])
        if not train:
            return [[test["input"] for _ in range(attempts)] for test in task.get("test", [])]
        shape_rule = _shape_rule(train)
        candidates = self.candidates + self._translation_candidates(train)
        exact: list[tuple[Candidate, dict[int, int], float]] = []
        scored: list[tuple[float, int, str, Candidate, dict[int, int]]] = []
        for cand in candidates:
            score, mapping = self._score(cand, train)
            if mapping is not None:
                scored.append((score, -cand.complexity, cand.name, cand, mapping))
                if score == 1.0:
                    exact.append((cand, mapping, self._holdout_score(cand, train)))
        ranked: list[tuple[Candidate, dict[int, int]]] = []
        ranked.extend(
            (candidate, mapping)
            for candidate, mapping, _ in sorted(
                exact,
                key=lambda x: (-x[2], x[0].complexity, x[0].name),
            )
        )
        seen = {(c.name, tuple(sorted(m.items()))) for c, m in ranked}
        for _, _, _, cand, mapping in sorted(scored, key=lambda x: (x[0], x[1], x[2]), reverse=True):
            key = (cand.name, tuple(sorted(mapping.items())))
            if key not in seen:
                ranked.append((cand, mapping))
                seen.add(key)
            if len(ranked) >= max(attempts, 3):
                break
        while len(ranked) < attempts:
            identity = next(c for c in self.base if c.name == "identity")
            ranked.append((identity, {}))
        outputs = []
        for test in task.get("test", []):
            x = _a(test["input"])
            shape = shape_rule(x.shape)
            row = []
            for cand, mapping in ranked[:attempts]:
                out = _apply_map(cand.op(x, shape), mapping)
                row.append(out.astype(int).tolist())
            outputs.append(row)
        return outputs


def solve_task(task: dict, attempts: int = 3) -> list[list[list[list[int]]]]:
    return ChimeraArcSolver().solve_task(task, attempts=attempts)
