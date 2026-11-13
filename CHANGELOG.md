# Changelog

## 0.1.0 (2026-11-13)

First version.

- Confidence sequences for a mean score: hedged-capital betting (with a without-replacement form for a fixed
  benchmark) and a two-sided normal mixture, both with a running intersection, a threshold rule and a width rule.
- Paired comparison of two models: a confidence sequence on the mean difference, a mixture likelihood-ratio test
  on discordant pairs, a futility rule, and the exact McNemar test at every look and on all pairs for contrast.
- Wald's sequential probability ratio test for a pass rate, and group-sequential boundaries by alpha spending
  (O'Brien-Fleming type and Pocock type).
- Commands `monitor`, `compare`, `design`, `simulate` and `replay`, with Markdown and JSON output and an optional
  chart.
- Scripts that fetch public SWE-bench Verified per-instance results and replay the rules over pairs of submissions.
