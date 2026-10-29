"""Coverage and precision of the verified ARC2_CHIMERA candidates on a labelled ARC split.

usage: python eval_chimera.py <challenges.json> <solutions.json> <out.json> [--workers N]
Counts per test output: emitted (a verified candidate exists) and correct (exact match).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chimera_fill as cf  # noqa: E402


def _one(item):
    tid, task = item
    t0 = time.time()
    try:
        cands = cf.chimera_candidates(task)
        err = None
    except Exception as e:  # recorded, not hidden
        cands, err = [], repr(e)
    return tid, cands, time.time() - t0, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("challenges")
    ap.add_argument("solutions")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ch = json.load(open(a.challenges))
    so = json.load(open(a.solutions))
    with Pool(a.workers) as pool:
        res = pool.map(_one, sorted(ch.items()), chunksize=4)
    rows, stats = [], {}
    for tid, cands, dt, err in res:
        sols = so[tid]
        for i, gt in enumerate(sols):
            first = cands[0] if cands else None
            emitted = first is not None and i < len(first[0])
            correct = emitted and first[0][i] == gt
            any_correct = any(i < len(c[0]) and c[0][i] == gt for c in cands)
            src = first[1].split(":")[0] if first else "none"
            rows.append({"task": tid, "test": i, "source": first[1] if first else None, "emitted": emitted,
                         "correct": correct, "any_correct": any_correct, "n_cands": len(cands),
                         "sec": round(dt, 3), "error": err})
            s = stats.setdefault(src, {"emitted": 0, "correct": 0})
            s["emitted"] += int(emitted)
            s["correct"] += int(correct)
    n = len(rows)
    emitted = sum(r["emitted"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    tasks_correct = len({r["task"] for r in rows if r["correct"]})
    summary = {"outputs": n, "tasks": len(ch), "emitted": emitted, "correct": correct,
               "precision": round(correct / emitted, 4) if emitted else None,
               "tasks_with_correct_first_candidate": tasks_correct,
               "any_candidate_correct": sum(r["any_correct"] for r in rows),
               "by_source": stats, "errors": sum(1 for r in rows if r["error"]),
               "max_task_sec": max(r["sec"] for r in rows), "total_task_sec": round(sum(x[2] for x in res), 1)}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
