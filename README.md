# Inferential Statistics Case Studies

A running collection of applied case studies working through core inferential statistics concepts end to end — framing the question, choosing and justifying a test, checking assumptions, running the analysis, and writing up the result the way you'd present it to a manager or professor.

## Why this repo exists

Built while preparing for a data quality analyst role, based on a suggestion to strengthen inferential statistics fundamentals through applied case studies rather than isolated formula practice. Each case study picks a real-world-flavored scenario, works through test selection reasoning explicitly, and documents limitations honestly rather than treating a p-value as the end of the analysis.

## Case studies

| # | Case Study | Concepts Covered | Folder |
|---|---|---|---|
| 1 | PTSD Nightmare Frequency Drug Trial | Paired design, Wilcoxon signed-rank test, paired t-test, Hodges-Lehmann estimator, power analysis, diagnostic plots | [`ptsd-nightmare-case-study/`](./ptsd-nightmare-case-study) |

More case studies will be added here over time, working through additional concepts such as z-tests, F-tests / ANOVA, chi-square tests, and simple and multiple linear regression.

## Repository structure

Each case study lives in its own folder and follows the same layout:

```
case-study-name/
├── README.md      # Summary specific to that case study
├── report/         # Full write-up (PDF): abstract, methods, results, discussion, limitations
├── code/            # Analysis scripts (Python and/or R)
└── figures/          # Diagnostic plots and visuals
```

## Tools used

- **Python** — SciPy, NumPy, statsmodels, Matplotlib
- **R** — base stats, pwr

Each case study's analysis is typically run independently in both languages and cross-checked for agreement.

## About

Maintained by Anantha Krishna as part of ongoing preparation and practice in applied inferential statistics.
