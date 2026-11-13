# evalstop

Monitor a running evaluation with intervals that stay valid however often you look, so a run can stop as soon as
the answer is settled.

**Status: early scaffold.** One command works so far, `evalstop monitor`, with two confidence sequences for a
single model's mean score. Comparing two models, group-sequential designs, the naive repeated-test baseline,
`simulate`, `replay`, charts and a study on real public results are not written yet. See [Status](#status).

## Why this is hard

Evaluating an agent is slow and expensive, so people watch the running score and stop when it looks settled. An
ordinary confidence interval or significance test is valid at one sample size fixed in advance. Recomputed after
every item and acted on the first time it looks convincing, it is wrong more often than its nominal level says,
because each extra look is another chance for noise to cross the line. Stopping early without that inflation needs
intervals that hold at all sample sizes at once, and those have to be wider than a fixed-sample interval, so the
question is how little width one can pay for the right to look.

## Install

```
pip install git+https://github.com/vipul21435/evalstop
```

Python 3.10 or later. The only runtime dependency is numpy.

## Quickstart

`examples/results.jsonl` holds simulated results (not real data) for two models on 300 items, one JSON object per
line in evaluation order, written by `examples/make_example.py`:

```
{"item": "item-001", "model": "model-a", "score": 1}
```

Is the pass rate of `model-a` above one half, and when is its interval no wider than 0.2?

```
evalstop monitor examples/results.jsonl --model model-a --threshold 0.5 --width 0.2
```

```
# evalstop monitor

- input: `examples/results.jsonl`, model `model-a`
- scores: 300 in [0, 1], mean 0.7200
- method: betting (hedged capital), alpha 0.05, grid 1000
- 95% confidence sequence after 300 scores: [0.6770, 0.8540]

## Stopping rules

| rule | stopped after | decision | interval at that look |
|---|---|---|---|
| threshold 0.5 | 15 of 300 scores | mean above 0.5 | [0.5110, 1.0000] |
| width <= 0.2 | 186 of 300 scores | narrow enough | [0.6770, 0.8770] |

## Trace

| scores | mean | lower | upper | width |
|---|---|---|---|---|
| 30 | 0.8000 | 0.6050 | 0.9760 | 0.3710 |
| 60 | 0.8000 | 0.6670 | 0.9480 | 0.2810 |
| 90 | 0.7889 | 0.6670 | 0.9290 | 0.2620 |
| 120 | 0.7500 | 0.6770 | 0.9060 | 0.2290 |
| 150 | 0.7200 | 0.6770 | 0.8870 | 0.2100 |
| 180 | 0.7278 | 0.6770 | 0.8790 | 0.2020 |
| 210 | 0.7190 | 0.6770 | 0.8700 | 0.1930 |
| 240 | 0.7167 | 0.6770 | 0.8630 | 0.1860 |
| 270 | 0.7185 | 0.6770 | 0.8580 | 0.1810 |
| 300 | 0.7200 | 0.6770 | 0.8540 | 0.1770 |

The interval holds at every look at once with probability at least 95%, provided the scores arrive in an order unrelated to their values.
```

The threshold question was settled after 15 of the 300 scores. The limits only ever move inwards, because each
reported interval is the intersection of all intervals so far.

Input is JSON Lines or CSV with the fields `item`, `model` and `score` (only `score` is required); `-` reads
standard input until it ends. Scores are numbers between `--lo` and `--hi` (default 0 and 1), JSON booleans, or the
words `true`, `false`, `pass`, `fail`. `--format json` prints the same report as JSON, `--method mixture` switches
to the other sequence, and `evalstop monitor --help` lists the rest.

From Python, one score at a time:

```python
import json

from evalstop import BettingCS

monitor = BettingCS(alpha=0.05)
with open("examples/results.jsonl") as handle:
    for line in handle:
        row = json.loads(line)
        if row["model"] != "model-a":
            continue
        lower, upper = monitor.update(row["score"])
        if lower >= 0.5:  # the pass rate is above one half
            break
print(monitor.n, monitor.interval)
```

```
15 (0.511, 1.0)
```

`betting_cs(scores)` and `mixture_cs(scores)` return the whole sequence for a list of scores.

## How it works

A confidence sequence is a sequence of intervals `[L_t, U_t]`, one after each score, such that

```
P(L_t <= mu <= U_t for every t) >= 1 - alpha
```

where `mu` is the mean score. Because the statement covers every `t` at once, the interval can be read after every
item and the run stopped at any look without the error rate growing. Both sequences here use the same argument:
for a candidate mean `m`, build a nonnegative process that starts at 1 and is a martingale (or supermartingale)
when `m` is the true mean. By Ville's inequality such a process ever reaches `1 / alpha` with probability at most
`alpha`, so the interval is the set of candidates whose process is still below `1 / alpha`.

**Betting** (`--method betting`, the default). For each candidate `m` on a grid, two wealth processes bet on every
score `x` in [0, 1], one that the mean is above `m` and one that it is below:

```
K+_t(m) = prod_{i<=t} (1 + lam+_i(m) * (x_i - m))        lam+_i(m) = min(lam_i, c / m)
K-_t(m) = prod_{i<=t} (1 - lam-_i(m) * (x_i - m))        lam-_i(m) = min(lam_i, c / (1 - m))

lam_t = sqrt(2 * log(2 / alpha) / (var_{t-1} * t * log(t + 1)))
```

with `c = 1/2` and `var_{t-1}` a running variance estimate from the first `t - 1` scores. A candidate stays in the
interval while `max(K+_t(m), K-_t(m)) / 2 < 1 / alpha`. This is the hedged capital confidence sequence with the
predictable plug-in bet from Ian Waudby-Smith and Aaditya Ramdas, "Estimating means of bounded random variables by
betting", Journal of the Royal Statistical Society Series B 86(1):1-27, 2024
([arXiv:2010.09686](https://arxiv.org/abs/2010.09686), Theorem 3). The bet is larger when the scores vary little,
so the interval adapts to the variance. Each limit is reported at the nearest rejected grid point outside the set,
so the grid can widen the interval but not narrow it.

**Normal mixture** (`--method mixture`). A score in [0, 1] minus its mean is sub-Gaussian with variance proxy 1/4
(Hoeffding's lemma). With `S_t` the sum of the centred scores and `V_t = t / 4`,

```
|S_t| < sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha^2)))   for every t, with probability >= 1 - alpha
```

which is the two-sided normal-mixture boundary, eq. (14) of Steven R. Howard, Aaditya Ramdas, Jon McAuliffe and
Jasjeet Sekhon, "Time-uniform, nonparametric, nonasymptotic confidence sequences", Annals of Statistics
49(2):1055-1080, 2021 ([arXiv:1810.08240](https://arxiv.org/abs/1810.08240)). `rho` is set by their Proposition 3
so that the interval is tightest after `--tune-at` scores (default 200). It is a closed form and cheap to compute,
but it uses the worst-case variance and does not narrow when the pass rate is near 0 or 1.

On the example above the two behave differently. The command

```
evalstop monitor examples/results.jsonl --model model-a --method mixture --threshold 0.5 --width 0.2
```

settles the threshold after 32 scores (betting: 15) and ends with the interval [0.6543, 0.8053], width 0.151
(betting: 0.177). That is one simulated run, not a comparison of the methods.

## Status

Works now:

- `evalstop monitor`: betting and normal-mixture confidence sequences for one model's mean score, a threshold rule
  and a width rule, Markdown and JSON output, JSON Lines and CSV input.
- The same from Python: `BettingCS`, `MixtureCS`, `betting_cs`, `mixture_cs`.

Not implemented yet:

- two models on the same items (a confidence sequence on the paired difference, a sequential test on discordant
  pairs, a futility rule);
- group-sequential designs (O'Brien-Fleming and Pocock boundaries by alpha spending);
- the naive baseline of a fixed-sample test repeated at every look;
- `simulate` (false-positive rate, power and items used by each rule) and `replay` (rules run over real results in
  random orders);
- charts, and any result on real data. No number in this file comes from a real evaluation.

## Limitations

- The guarantee needs every score to have the same expected value given the earlier ones: independent items from
  one pool, in an order unrelated to their difficulty. A file sorted by difficulty breaks it. `monitor` warns when
  the running intersection becomes empty, which is one symptom, but it cannot detect the problem in general.
- A fixed benchmark run in random order is sampling without replacement. The sequences here do not model that, so
  they stay wide even when almost every item has been seen, and the guarantee for the mean over exactly that
  benchmark is not proved here.
- Scores must be bounded and the bounds known.
- Betting limits are multiples of `(hi - lo) / grid`, and each score costs time proportional to `--grid`.

## Development

```
uv sync --extra dev
uv run ruff check
uv run pytest -q
```

The tests check the mixture boundary against its closed form, numerical integration and scipy's Lambert W, the
betting interval against a direct recomputation of the capital, and the coverage of both sequences at every look
by seeded simulation (miss rate at most alpha plus three binomial standard errors).

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars, paired comparisons and sample-size planning for
  evaluations at a fixed sample size. evalstop is the sequential case.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
