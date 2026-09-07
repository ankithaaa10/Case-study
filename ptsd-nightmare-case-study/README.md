# PTSD Nightmare Frequency Drug Trial — Inferential Statistics Case Study

A worked case study applying paired-sample inferential statistics to evaluate whether a hypothetical new drug reduces nightmare frequency in patients with PTSD. Built as part of a case-study series covering hypothesis testing, parametric vs. non-parametric test selection, effect sizes, and power analysis.

## Summary

- **Design:** Single-arm, pre/post paired study, n = 12 patients
- **Outcome:** Nightmares per week, measured at baseline and after 8 weeks of treatment
- **Primary test:** One-tailed Wilcoxon signed-rank test — V = 78, p = 0.0012
- **Effect estimate:** Hodges-Lehmann median reduction = 3.5 nightmares/week (95% CI: ≥ 2.0)
- **Secondary test:** One-tailed paired t-test — t(11) = 5.63, p < 0.001, Cohen's d = 1.63
- **Result:** All 12 patients improved; reduction unlikely to be due to chance, though the single-arm design cannot establish causation (see Limitations in the report)

Full reasoning, assumption checks, and discussion are in the report — this README just orients the repo.

## Repository structure

```
ptsd-nightmare-case-study/
├── README.md
├── report/
│   └── PTSD_Nightmare_Case_Study_Report.pdf   # Full write-up: abstract, methods, results, discussion, limitations
├── code/
│   ├── ptsd_nightmare_case_study_v2.py        # Python analysis (SciPy, statsmodels)
│   └── ptsd_nightmare_case_study_v2.R         # R analysis (base stats, pwr)
└── figures/
    └── diagnostic_plots.png                    # Q-Q plot of differences + before/after boxplot
```

## Reproducing the analysis

### Python
Requires Python 3.9+, `scipy`, `numpy`, `matplotlib`, `statsmodels`.

```bash
pip install scipy numpy matplotlib statsmodels
python code/ptsd_nightmare_case_study_v2.py
```

### R
Requires R 4.0+ and the `pwr` package (installed automatically by the script if missing).

```bash
Rscript code/ptsd_nightmare_case_study_v2.R
```

Both scripts run the same analysis independently and were cross-checked for agreement — see the Discussion section of the report for a note on where R and Python differ in implementation (e.g. R returns the Hodges-Lehmann CI natively via `wilcox.test(..., conf.int=TRUE)`, while the Python script derives it via bootstrap).

## Notes on the data

The dataset (`before`/`after` nightmare counts for 12 patients) is synthetic, generated for teaching purposes and modeled on the structure of real PTSD nightmare-frequency trials (e.g. prazosin studies). It is not real patient data.

## Method notes worth flagging to a reader

- Test selection (Wilcoxon as primary) was made on design grounds — small n, count-type outcome, tied values, and a visible outlier — rather than purely from the Shapiro-Wilk result on this sample, since post-hoc test selection based on a preliminary normality test is itself a known source of distorted Type I error rates.
- Power analysis is reported post-hoc and explicitly flagged as such — high achieved power at the observed effect size does not imply n=12 is generally adequate for this type of study.

## License

Educational / portfolio use.
