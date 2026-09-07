"""
Case Study (v2 - master's level): Does a new drug reduce nightmare frequency
in PTSD patients?

Design: Paired (same 12 patients measured before and after 8 weeks of treatment)
Outcome: Nightmares per week (count data)

v2 upgrades over the first draft:
  1. Hodges-Lehmann estimator + bootstrap 95% CI for the median difference
  2. One-tailed tests throughout, consistent with the directional hypothesis
  3. Diagnostic plots (Q-Q plot of differences, before/after boxplot)
  4. Post-hoc power + required-n-for-80%-power calculation
  5. Output organized as Methods / Results / Discussion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestPower

rng = np.random.default_rng(7)

# ============================================================
# METHODS
# ============================================================
print("=" * 60)
print("METHODS")
print("=" * 60)
print("""
Design: Single-arm, pre/post paired design. 12 patients with chronic
PTSD-related nightmares had weekly nightmare frequency recorded at
baseline and again after 8 weeks on the study drug.

Hypotheses (one-tailed, pre-specified direction):
  H0: median(before - after) = 0   (drug has no effect)
  H1: median(before - after) > 0   (drug reduces nightmare frequency)

Primary test: Wilcoxon signed-rank test (one-tailed), chosen over the
paired t-test because the differences are count-derived, the sample is
small (n=12), and several tied difference values are present -
conditions under which a rank-based test is more defensible than a
mean-based test that assumes normality.
Secondary test: paired t-test (one-tailed), reported for comparison.
""")

patients = [f"Patient_{i+1}" for i in range(12)]
before = np.array([7, 5, 9, 4, 8, 12, 6, 5, 10, 7, 6, 15])
after  = np.array([2, 3, 4, 3, 5, 6, 4, 4, 3, 5, 4, 9])
diff = before - after
n = len(diff)

print("Patient    | Before | After | Diff")
for p, b, a, d in zip(patients, before, after, diff):
    print(f"{p:11} {b:6} {a:6} {d:6}")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# --- Descriptive stats ---
print("\n--- Descriptive statistics ---")
print(f"Mean before:   {before.mean():.2f}  (sd={before.std(ddof=1):.2f})")
print(f"Median before: {np.median(before):.2f}")
print(f"Mean after:    {after.mean():.2f}  (sd={after.std(ddof=1):.2f})")
print(f"Median after:  {np.median(after):.2f}")
print(f"Mean difference:   {diff.mean():.2f}  (sd={diff.std(ddof=1):.2f})")
print(f"Median difference: {np.median(diff):.2f}")

# --- Diagnostic plots ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

stats.probplot(diff, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot of Paired Differences\n(before - after)")

axes[1].boxplot([before, after], tick_labels=["Before", "After"])
axes[1].set_ylabel("Nightmares / week")
axes[1].set_title("Nightmare Frequency: Before vs After")

plt.tight_layout()
plt.savefig("/home/claude/diagnostic_plots.png", dpi=150)
print("\n[Diagnostic plots saved: Q-Q plot of differences + before/after boxplot]")

# --- Assumption check ---
print("\n--- Assumption check: normality of differences (Shapiro-Wilk) ---")
shapiro_stat, shapiro_p = stats.shapiro(diff)
print(f"W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
print("Note: with n=12, Shapiro-Wilk has low power to detect non-normality -")
print("a non-significant result here is weak evidence FOR normality, not")
print("strong confirmation. The Q-Q plot and the tied values among |diff|")
print("(ties invalidate the exact Wilcoxon distribution) are the stronger")
print("reasons to prefer the rank-based test here.")

# --- Primary test: one-tailed Wilcoxon signed-rank ---
print("\n--- Wilcoxon signed-rank test (one-tailed, primary) ---")
w_stat, p_val_w = stats.wilcoxon(before, after, alternative='greater',
                                   zero_method='wilcox', method='approx')
print(f"W = {w_stat:.4f}, p (one-tailed) = {p_val_w:.6f}")

# Effect size: matched-pairs rank-biserial correlation
ranks = stats.rankdata(np.abs(diff))
pos_sum = ranks[diff > 0].sum()
neg_sum = ranks[diff < 0].sum()
rank_biserial = (pos_sum - neg_sum) / (pos_sum + neg_sum)
print(f"Matched-pairs rank-biserial correlation (effect size) = {rank_biserial:.3f}")

# --- Hodges-Lehmann estimator + bootstrap CI ---
print("\n--- Hodges-Lehmann estimator (median of Walsh averages) ---")
walsh_averages = np.array([(diff[i] + diff[j]) / 2
                            for i in range(n) for j in range(i, n)])
hl_estimate = np.median(walsh_averages)
print(f"Hodges-Lehmann point estimate of the median difference: {hl_estimate:.3f}")

# Bootstrap 95% CI (percentile method, resampling patients with replacement)
n_boot = 10000
boot_estimates = np.empty(n_boot)
idx_range = np.arange(n)
for b in range(n_boot):
    sample_idx = rng.choice(idx_range, size=n, replace=True)
    d_boot = diff[sample_idx]
    walsh_boot = np.array([(d_boot[i] + d_boot[j]) / 2
                            for i in range(n) for j in range(i, n)])
    boot_estimates[b] = np.median(walsh_boot)

ci_low, ci_high = np.percentile(boot_estimates, [2.5, 97.5])
print(f"Bootstrap 95% CI for the median difference: ({ci_low:.3f}, {ci_high:.3f})")
print("(Note: R's wilcox.test(..., conf.int=TRUE) returns an analogous")
print("rank-based CI natively; scipy has no built-in equivalent, so a")
print("percentile bootstrap is used here as a transparent alternative.)")

# --- Secondary test: one-tailed paired t-test ---
print("\n--- Paired t-test (one-tailed, secondary/comparison) ---")
t_stat, p_val_t = stats.ttest_rel(before, after, alternative='greater')
print(f"t = {t_stat:.4f}, p (one-tailed) = {p_val_t:.6f}")

cohens_d = diff.mean() / diff.std(ddof=1)
print(f"Cohen's d (effect size) = {cohens_d:.3f}")

# --- Power analysis ---
print("\n--- Power analysis ---")
power_analysis = TTestPower()
achieved_power = power_analysis.power(effect_size=cohens_d, nobs=n,
                                       alpha=0.05, alternative='larger')
print(f"Post-hoc achieved power (given observed d={cohens_d:.2f}, n={n}): "
      f"{achieved_power:.4f}")

required_n = power_analysis.solve_power(effect_size=cohens_d, alpha=0.05,
                                          power=0.80, alternative='larger')
print(f"Sample size needed for 80% power at this effect size: "
      f"{np.ceil(required_n):.0f} patients")

print(f"\nPatients who improved: {(diff > 0).sum()} / {n}")

# ============================================================
# DISCUSSION
# ============================================================
print("\n" + "=" * 60)
print("DISCUSSION")
print("=" * 60)
print(f"""
All {n} patients showed a reduction in nightmare frequency after 8 weeks
of treatment (median: {np.median(before):.0f}/week -> {np.median(after):.0f}/week). The
one-tailed Wilcoxon signed-rank test found this reduction unlikely to be
due to chance (p = {p_val_w:.4f}). The Hodges-Lehmann point estimate of the
median reduction is {hl_estimate:.2f} nightmares/week, with a bootstrap 95% CI
of ({ci_low:.2f}, {ci_high:.2f}) - i.e. we are confident the true typical
reduction is somewhere in that range, not just "probably positive."

The paired t-test agrees in direction and significance (p = {p_val_t:.6f}),
but achieved power was already high at this observed effect size
({achieved_power:.4f}), and the required-n calculation ({np.ceil(required_n):.0f}) is
a reminder that this specific effect size happened to be large enough to
detect even at n=12 - it does not mean n=12 is generally an adequate
sample size for a trial of this kind, and a pre-registered a priori
power calculation would be expected before running a real study.

Limitations: no placebo/control arm, small n, self-reported outcome
(recall bias), and tied values in |diff| that technically preclude an
exact (rather than asymptotic) Wilcoxon p-value.
""")