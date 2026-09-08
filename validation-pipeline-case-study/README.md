# Data Validation Pipeline Comparison — McNemar's Test & Bayesian Analysis

A case study evaluating whether a proposed data validation pipeline (with added cross-field and duplicate-detection checks) catches more invalid records than a legacy pipeline, using a paired binary classifier comparison.

Unlike a purely hypothetical exercise, this case study generates 800 synthetic records with real fields and implements both pipelines as actual rule-checking functions — the confusion table and every downstream statistic is a **measured output** of running that code, not a hand-picked input.

## A note on process: this dataset was audited before use

An earlier version of the duplicate-record generator had a bug: the injected email variation (case change + added whitespace) caused an *unrelated* email-format check to reject those records before the duplicate-detection logic ever ran. All 10 duplicate-type records were being "correctly" rejected, but for the wrong reason — meaning Pipeline B's duplicate-detection code was never actually exercised or tested.

This was caught by auditing each defect mechanism's actual correct/incorrect rate against design intent *before* trusting any downstream statistics — not by inspecting results after the fact. The generator was fixed (case-only variation, no added whitespace) and every mechanism was re-audited to confirm it now triggers as intended. **The numbers in this README and the report reflect the corrected, audited data.**

## Summary

- **Design:** Paired comparison — 800 generated order records, each scored independently by two rule-based pipelines against ground truth
- **Primary test:** McNemar's test (exact) — p = 1.16 × 10⁻¹⁸
- **Effect:** Accuracy: Pipeline A = 85.12%, Pipeline B = 96.62%; 95% CI for the difference: (8.94, 14.06) percentage points
- **Bayesian cross-check:** Two models (Beta-Binomial on discordant pairs; Dirichlet-Multinomial on the full 2×2 table) both estimate P(Pipeline B is more accurate) > 0.999
- **Key finding beyond significance:** Tracing the 14 cases where the legacy pipeline outperformed the new one revealed two specific, fixable regressions — a stale zip/state reference table (7 cases) and a regex change that became too permissive (7 cases) — turning the statistical result into a concrete pre-deployment fix list

Full reasoning, real worked examples (pulled directly from the generated data), the audit process, and limitations are in the report.

## Repository structure

```
validation-pipeline-case-study/
├── README.md
├── report/
│   └── Validation_Pipeline_Case_Study_Report.pdf   # Full write-up, including the audit process
├── code/
│   ├── validation_pipeline_case_study.py           # Generates data, runs pipelines, runs stats (Python)
│   └── validation_pipeline_case_study.R            # Same, independently, in R
├── data/
│   ├── audit_records_800.csv                       # Raw generated records + ground truth (post-fix)
│   └── audit_records_800_scored.csv                # Same, plus each pipeline's verdict
└── figures/
    └── validation_pipeline_results.png              # Confusion table + posterior distribution plot
```

## What the code actually does

Nothing in this case study starts from a pre-decided result. The scripts:
1. Generate 800 synthetic order records (order ID, email, order date, ship date, zip code, state, price), each assigned a ground-truth validity label and, for invalid records, one of three defect mechanisms (basic field-level, cross-field, or duplicate)
2. Implement Pipeline A and Pipeline B as real functions — real regex, real date comparison, a real (deliberately incomplete) zip→state lookup table, a real price-plausibility check, real duplicate detection
3. Run both pipelines against every record and derive the confusion table from the actual results
4. Run McNemar's test and two Bayesian models on whatever table results

## Reproducing the analysis

### Python
Requires Python 3.9+, `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`.

```bash
pip install numpy pandas scipy matplotlib statsmodels
python code/validation_pipeline_case_study.py
```

### R
Requires R 4.0+. No extra packages needed — Dirichlet sampling is implemented from base R's `rgamma()` via the standard normalized-Gamma trick, since base R has no `rdirichlet()`.

```bash
Rscript code/validation_pipeline_case_study.R
```

**Note on cross-language reproducibility:** Python and R use different random number generator algorithms. Even with a fixed seed in each language, they will generate *different* specific samples of 800 records, and therefore slightly different confusion table counts. This is expected, not a bug — agreement in the overall conclusion across two independently-generated samples is stronger evidence than either sample alone. The CSV files in `data/` reflect the Python run specifically.

## Why McNemar's test instead of a two-proportion z-test

Both pipelines score the *same* 800 records, so their outputs are dependent, not independent — a two-proportion z-test (used in this series' A/B testing checkout case study) would be the wrong tool here. McNemar's test uses only the records where the two pipelines disagree, since records both pipelines classify correctly (or both misclassify) carry no information about which pipeline performs better.

## Notes on the data

All 800 records, their fields, and their ground-truth labels are synthetically generated by the code in `code/`, following a documented generative process (see the report's Methods section). This is not real production data. Defect injection rates were chosen by the author rather than estimated from a real audit.

## License

Educational / portfolio use.
