#!/usr/bin/env Rscript
# Reply-level logistic mixed model of linguistic accommodation.
#
# Outcome: does B's reply contain the family (b_uses)? Accommodation is the effect of
# A having just used it (a_uses) on that probability, in log-odds. Moderators enter as
# a_uses:<factor> interactions: "does accommodation differ by genre / language / ...".
# Main effects of the moderators only describe base rates, not accommodation.
#
# Random effects:
#   (1 + a_uses || work)      work = the play regardless of language (12); EN/FR share it
#   (1 | play)                the language version within the work (24)
#   (1 | b_speaker:family)    each character's habitual rate for each family
# Work slopes and intercepts are uncorrelated (||): with correlations the fit was singular
# (estimated at -1). A dyad intercept was also dropped: its variance was estimated at 0
# once b_speaker:family is in the model, and so was an a_uses slope for play (accommodation
# does not vary between the EN and FR versions of a work beyond the fixed language effect).
#
# Input:  outputs/alignment/reply_level.csv.gz  (python scripts/stats/build_alignment_table.py)
# Output: outputs/alignment/glmm_*.csv
#
# Usage (from repo root):
#   Rscript scripts/stats/alignment_glmm.R          # fast: Laplace-free fits (nAGQ = 0)
#   Rscript scripts/stats/alignment_glmm.R --full   # nAGQ = 1 (slower, use for final numbers)
#   Rscript scripts/stats/alignment_glmm.R --test   # 10% of pairs, results in outputs/alignment/test/

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
NAGQ <- if ("--full" %in% args) 1 else 0
OUT <- "outputs/alignment"
CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# ── Data ───────────────────────────────────────────────────────────────────
d <- read.csv(file.path(OUT, "reply_level.csv.gz"), stringsAsFactors = FALSE)
d <- d[d$condition == "adjacent", ]    # controls are compared in alignment_conditions.R
n_all <- nrow(d)
d <- d[!grepl("U", d$gender_dyad), ]   # speakers with unknown gender (< 0.3% of rows)
if ("--test" %in% args) {              # quick check of the code on 10% of pairs
  set.seed(1)
  keys <- unique(paste(d$play, d$pair_id))
  d <- d[paste(d$play, d$pair_id) %in% sample(keys, length(keys) %/% 10), ]
  OUT <- file.path(OUT, "test")
  dir.create(OUT, showWarnings = FALSE)
}
cat(sprintf("Rows: %d (dropped %d with unknown gender)\n", nrow(d), n_all - nrow(d)))

d$genre           <- relevel(factor(d$genre), "Absurdist")
d$language        <- relevel(factor(d$language), "English")
d$family          <- relevel(factor(d$family), "adverbs")
d$gender_dyad     <- relevel(factor(d$gender_dyad), "F-F")
d$importance_dyad <- relevel(factor(d$importance_dyad), "intermediate->intermediate")
d$log_a_len <- as.numeric(scale(log(d$a_len)))
d$log_b_len <- as.numeric(scale(log(d$b_len)))

# ── Model ──────────────────────────────────────────────────────────────────
MODERATORS <- c("genre", "language", "family", "importance_dyad", "gender_dyad")

f_full <- b_uses ~ a_uses * (genre + language + family + importance_dyad + gender_dyad) +
  family * log_b_len + log_a_len +
  (1 + a_uses || work) + (1 | play) + (1 | b_speaker:family)

fit <- function(f) glmer(f, data = d, family = binomial, nAGQ = NAGQ, control = CTRL)

t0 <- Sys.time()
cat(sprintf("Fitting full model (nAGQ = %d)...\n", NAGQ))
m_full <- fit(f_full)
cat(sprintf("  done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
print(summary(m_full), correlation = FALSE)

# ── Fixed effects ──────────────────────────────────────────────────────────
co <- summary(m_full)$coefficients
fe <- data.frame(term = rownames(co), estimate = co[, 1], se = co[, 2], z = co[, 3], p = co[, 4],
                 odds_ratio = exp(co[, 1]), row.names = NULL)
write.csv(fe, file.path(OUT, "glmm_fixed_effects.csv"), row.names = FALSE)

# ── Random effects ─────────────────────────────────────────────────────────
vc <- as.data.frame(VarCorr(m_full))
write.csv(vc, file.path(OUT, "glmm_random_effects.csv"), row.names = FALSE)

# ── Likelihood-ratio tests: does accommodation differ by each factor? ──────
lrt_row <- function(name, reduced, bigger) {
  a <- anova(reduced, bigger)
  data.frame(test = name, df = a$Df[2], chi2 = a$Chisq[2], p = a$`Pr(>Chisq)`[2])
}
lrt <- list()
for (m in MODERATORS) {
  cat(sprintf("LRT: a_uses:%s\n", m))
  reduced <- fit(update(f_full, as.formula(paste(". ~ . - a_uses:", m))))
  lrt[[m]] <- lrt_row(paste0("a_uses:", m), reduced, m_full)
}
cat("LRT: a_uses:genre:family\n")
m_gf <- fit(update(f_full, . ~ . + a_uses:genre:family))
lrt[["genre_x_family"]] <- lrt_row("a_uses:genre:family", m_full, m_gf)
lrt <- do.call(rbind, lrt)
write.csv(lrt, file.path(OUT, "glmm_lrt.csv"), row.names = FALSE)
print(lrt, row.names = FALSE)

# ── Accommodation (a_uses slope, log-odds) for each level of each moderator ─
# Other moderators held at their reference level.
# Contrast vector for the a_uses slope at the given factor levels (e.g. "genreRealist",
# "familyarticles"): every a_uses term whose other parts are all among those levels.
contrast <- function(model, levels_) {
  b <- fixef(model)
  parts <- strsplit(names(b), ":", fixed = TRUE)
  keep <- vapply(parts, function(p) "a_uses" %in% p && all(p %in% c("a_uses", levels_)), logical(1))
  setNames(as.numeric(keep), names(b))
}
estimate <- function(model, L) {
  est <- sum(L * fixef(model))
  se <- sqrt(as.numeric(t(L) %*% as.matrix(vcov(model)) %*% L))
  c(estimate = est, se = se, z = est / se, p = 2 * pnorm(-abs(est / se)))
}
acc <- list()
for (m in MODERATORS) {
  for (lev in levels(d[[m]])) {
    L <- contrast(m_full, paste0(m, lev))
    acc[[length(acc) + 1]] <- data.frame(factor = m, level = lev, t(estimate(m_full, L)))
  }
}
acc <- do.call(rbind, acc)
acc$odds_ratio <- exp(acc$estimate)
write.csv(acc, file.path(OUT, "glmm_accommodation_by_level.csv"), row.names = FALSE)
print(acc, row.names = FALSE, digits = 3)

# ── Genre x family (from the model with the 3-way interaction) ─────────────
gf <- list()
for (fam in levels(d$family)) {
  f_lev <- paste0("family", fam)
  L_abs <- contrast(m_gf, f_lev)
  L_rea <- contrast(m_gf, c(f_lev, "genreRealist"))
  gf[[fam]] <- data.frame(
    family = fam,
    absurdist = estimate(m_gf, L_abs)[["estimate"]],
    realist = estimate(m_gf, L_rea)[["estimate"]],
    t(setNames(estimate(m_gf, L_rea - L_abs), c("diff_realist_minus_absurdist", "diff_se", "diff_z", "diff_p"))))
}
gf <- do.call(rbind, gf)
write.csv(gf, file.path(OUT, "glmm_genre_by_family.csv"), row.names = FALSE)
print(gf, row.names = FALSE, digits = 3)

writeLines(c(sprintf("nAGQ,%d", NAGQ), sprintf("n_obs,%d", nrow(d)),
             sprintf("n_pairs,%d", nrow(unique(d[c("play", "pair_id")]))),
             sprintf("n_dyads,%d", length(unique(d$dyad))),
             sprintf("n_plays,%d", length(unique(d$play))),
             sprintf("n_works,%d", length(unique(d$work))),
             sprintf("singular,%s", isSingular(m_full)),
             sprintf("loglik,%.2f", as.numeric(logLik(m_full)))),
           file.path(OUT, "glmm_model_info.csv"))
cat(sprintf("\nAll outputs in %s/ (total %.1f min)\n", OUT,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
