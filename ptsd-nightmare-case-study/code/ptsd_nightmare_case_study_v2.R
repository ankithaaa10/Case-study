# =============================================================
# Case Study (v2 - master's level): Does a new drug reduce nightmare
# frequency in PTSD patients?
#
# Design: Paired (same 12 patients, before vs after 8 weeks of treatment)
# Outcome: Nightmares per week (count data)
#
# v2 upgrades over the first draft:
#   1. Hodges-Lehmann estimator + CI (native to wilcox.test in R)
#   2. One-tailed tests throughout, consistent with the directional hypothesis
#   3. Diagnostic plots (Q-Q plot of differences, before/after boxplot)
#   4. Post-hoc power + required-n-for-80%-power calculation (pwr package)
#   5. Output organized as Methods / Results / Discussion
#
# Requires: install.packages("pwr")  # for power analysis
# =============================================================

if (!requireNamespace("pwr", quietly = TRUE)) install.packages("pwr")
library(pwr)

cat(strrep("=", 60), "\n")
cat("METHODS\n")
cat(strrep("=", 60), "\n")
cat("
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
\n")

patients <- paste0("Patient_", 1:12)
before <- c(7, 5, 9, 4, 8, 12, 6, 5, 10, 7, 6, 15)
after  <- c(2, 3, 4, 3, 5, 6, 4, 4, 3, 5, 4, 9)
diff <- before - after
n <- length(diff)

print(data.frame(Patient = patients, Before = before, After = after, Diff = diff))

cat("\n", strrep("=", 60), "\n")
cat("RESULTS\n")
cat(strrep("=", 60), "\n")

# --- Descriptive stats ---
cat("\n--- Descriptive statistics ---\n")
cat(sprintf("Mean before:   %.2f  (sd=%.2f)\n", mean(before), sd(before)))
cat(sprintf("Median before: %.2f\n", median(before)))
cat(sprintf("Mean after:    %.2f  (sd=%.2f)\n", mean(after), sd(after)))
cat(sprintf("Median after:  %.2f\n", median(after)))
cat(sprintf("Mean difference:   %.2f  (sd=%.2f)\n", mean(diff), sd(diff)))
cat(sprintf("Median difference: %.2f\n", median(diff)))

# --- Diagnostic plots ---
png("diagnostic_plots.png", width = 1100, height = 450, res = 130)
par(mfrow = c(1, 2))
qqnorm(diff, main = "Q-Q Plot of Paired Differences\n(before - after)")
qqline(diff, col = "red")
boxplot(before, after, names = c("Before", "After"),
        ylab = "Nightmares / week",
        main = "Nightmare Frequency: Before vs After")
dev.off()
cat("\n[Diagnostic plots saved to diagnostic_plots.png]\n")

# --- Assumption check ---
cat("\n--- Assumption check: normality of differences (Shapiro-Wilk) ---\n")
shapiro_result <- shapiro.test(diff)
print(shapiro_result)
cat("Note: with n=12, Shapiro-Wilk has low power to detect non-normality -\n")
cat("a non-significant result here is weak evidence FOR normality, not\n")
cat("strong confirmation. The Q-Q plot and the tied values among |diff|\n")
cat("(ties invalidate the exact Wilcoxon distribution) are the stronger\n")
cat("reasons to prefer the rank-based test here.\n")

# --- Primary test: one-tailed Wilcoxon signed-rank, WITH Hodges-Lehmann CI ---
cat("\n--- Wilcoxon signed-rank test (one-tailed, primary) ---\n")
# exact=FALSE because of ties in |diff|; conf.int=TRUE returns the
# Hodges-Lehmann estimate + CI natively - no manual bootstrap needed in R
wilcox_result <- wilcox.test(before, after, paired = TRUE, alternative = "greater",
                              exact = FALSE, conf.int = TRUE, conf.level = 0.95)
print(wilcox_result)
cat(sprintf("Hodges-Lehmann point estimate: %.3f\n", wilcox_result$estimate))
cat(sprintf("95%% CI (one-sided, matches alternative='greater'): (%.3f, %.3f)\n",
            wilcox_result$conf.int[1], wilcox_result$conf.int[2]))

# Effect size: matched-pairs rank-biserial correlation
ranks <- rank(abs(diff))
pos_sum <- sum(ranks[diff > 0])
neg_sum <- sum(ranks[diff < 0])
rank_biserial <- (pos_sum - neg_sum) / (pos_sum + neg_sum)
cat(sprintf("Matched-pairs rank-biserial correlation (effect size) = %.3f\n", rank_biserial))

# --- Secondary test: one-tailed paired t-test ---
cat("\n--- Paired t-test (one-tailed, secondary/comparison) ---\n")
t_result <- t.test(before, after, paired = TRUE, alternative = "greater")
print(t_result)

cohens_d <- mean(diff) / sd(diff)
cat(sprintf("Cohen's d (effect size) = %.3f\n", cohens_d))

# --- Power analysis (pwr package) ---
cat("\n--- Power analysis ---\n")
power_result <- pwr.t.test(n = n, d = cohens_d, sig.level = 0.05,
                            type = "paired", alternative = "greater")
cat(sprintf("Post-hoc achieved power (given observed d=%.2f, n=%d): %.4f\n",
            cohens_d, n, power_result$power))

n_result <- pwr.t.test(d = cohens_d, sig.level = 0.05, power = 0.80,
                        type = "paired", alternative = "greater")
cat(sprintf("Sample size needed for 80%% power at this effect size: %.0f patients\n",
            ceiling(n_result$n)))

cat(sprintf("\nPatients who improved: %d / %d\n", sum(diff > 0), n))

cat("\n", strrep("=", 60), "\n")
cat("DISCUSSION\n")
cat(strrep("=", 60), "\n")
cat(sprintf("
All %d patients showed a reduction in nightmare frequency after 8 weeks
of treatment (median: %.0f/week -> %.0f/week). The one-tailed Wilcoxon
signed-rank test found this reduction unlikely to be due to chance
(p = %.4f). The Hodges-Lehmann point estimate of the median reduction
is %.2f nightmares/week, with a 95%% CI of (%.2f, %.2f) - i.e. we are
confident the true typical reduction is somewhere in that range, not
just \"probably positive.\"

The paired t-test agrees in direction and significance (p = %.6f), but
achieved power was already high at this observed effect size (%.4f),
and the required-n calculation (%.0f) is a reminder that this specific
effect size happened to be large enough to detect even at n=12 - it
does not mean n=12 is generally an adequate sample size for a trial of
this kind, and a pre-registered a priori power calculation would be
expected before running a real study.

Limitations: no placebo/control arm, small n, self-reported outcome
(recall bias), and tied values in |diff| that technically preclude an
exact (rather than asymptotic) Wilcoxon p-value.
",
n, median(before), median(after), wilcox_result$p.value,
wilcox_result$estimate, wilcox_result$conf.int[1], wilcox_result$conf.int[2],
t_result$p.value, power_result$power, ceiling(n_result$n)))
