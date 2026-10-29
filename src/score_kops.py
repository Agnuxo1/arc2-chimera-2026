"""Scorer local ARC-AGI-2: exact-match por salida de test, 2 intentos, media sobre todas las salidas."""
import json, sys

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def _is_grid(g):
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

def validate_format(sub, challenges):
    """Devuelve lista de errores (vacia = valido). challenges: dict task_id -> {'test': [...]}"""
    errs = []
    if not isinstance(sub, dict):
        return ["la entrega no es un dict"]
    for tid in challenges:
        if tid not in sub:
            errs.append(f"falta task_id {tid}"); continue
        n = len(challenges[tid]["test"])
        ent = sub[tid]
        if not isinstance(ent, list) or len(ent) != n:
            errs.append(f"{tid}: se esperaban {n} entradas"); continue
        for i, e in enumerate(ent):
            if not isinstance(e, dict) or "attempt_1" not in e or "attempt_2" not in e:
                errs.append(f"{tid}[{i}]: faltan attempt_1/attempt_2"); continue
            for a in ("attempt_1", "attempt_2"):
                if not _is_grid(e[a]):
                    errs.append(f"{tid}[{i}].{a}: rejilla invalida")
    extra = set(sub) - set(challenges)
    if extra:
        errs.append(f"task_id sobrantes: {sorted(extra)[:3]}")
    return errs

def score_detail(sub, solutions):
    """Devuelve lista (task_id, idx, acierto 0/1). solutions: task_id -> [grid,...]."""
    rows = []
    for tid, sols in solutions.items():
        for i, gt in enumerate(sols):
            e = sub[tid][i]
            hit = int(e["attempt_1"] == gt or e["attempt_2"] == gt)
            rows.append((tid, i, hit))
    return rows

def score(sub, solutions, challenges=None):
    if challenges is not None:
        errs = validate_format(sub, challenges)
        if errs:
            raise ValueError("formato invalido: " + "; ".join(errs[:5]))
    rows = score_detail(sub, solutions)
    return sum(r[2] for r in rows) / len(rows)

if __name__ == "__main__":
    sub, sol, ch = (load(p) for p in sys.argv[1:4])
    print(score(sub, sol, ch))
