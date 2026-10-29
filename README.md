# ARC2 CHIMERA — verified symbolic candidates for ARC-AGI-2

Contest implementation derived from [ARC2_CHIMERA](https://github.com/Agnuxo1/ARC2_CHIMERA), combining a symbolic DSL, induced cellular-automaton rules and a separately credited Qwen control. [Official competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).

## Measured evidence

| Experiment | Outcome | Interpretation |
|---|---|---|
| E001, symbolic candidates on public evaluation | 0/172 outputs emitted | No demonstrated coverage on this split |
| E002, original candidates on training | 50 emitted, 44 correct; precision 0.88 | Positive candidate precision, limited coverage; training evidence |
| E003, expanded CA families | 59 emitted, 53 correct on training; CA v2 11/11, 9 additional outputs | Positive training contribution; still 0/172 evaluation coverage |
| E004, Kaggle control + CHIMERA commit | 3/172, same as control in its restricted commit mode | Successful integration, no measured gain on public commit tasks |

These counts refer to outputs, not interchangeable task accuracy. The inherited control notebook restricts its ordinary commit evaluation to four tasks. Its 3/172 figure is not a full-budget hidden score. Official submission S01 is recorded as 57759538 in the project log; the 29October authenticated API readback still shows PENDING with no score. [Official readback](docs/official-readback.json). The 24-task fixed GPU control is still running on servidor-ia (job 32495d89b1c5419c). No additional compute is launched by this repository.

## Reproduce

Python 3.11+ and NumPy. Obtain the allowed labelled ARC files from the competition, outside the repository:

```bash
python -m pip install -r requirements.txt
python src/eval_chimera.py /path/challenges.json /path/solutions.json /path/result.json --workers 4
```

The evaluator writes coverage, precision, per-output records and exceptions. It evaluates the original candidate generator; CA v2 is a separately exposed module, not silently substituted into that command. See [experiment log](experiments/LOG.md) and [research protocol](docs/RESEARCH_PROTOCOL.md). Data, weights and the third-party LLM notebook are not redistributed here. The separately reproduced Qwen control is not our model contribution.

## Layout and attribution

`src/chimera_dsl.py`: own prior symbolic solver; `chimera_fill.py`: verified candidate gate; `chimera_ca2.py`: expanded rules; `eval_chimera.py`: matched labelled evaluation. Upstream ARC2_CHIMERA licence is retained. The Qwen/control component needs its own model and notebook licences and competition packaging before deployment. Repository publication does not establish acceptance, rank or prize.
