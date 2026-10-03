#!/usr/bin/env Rscript
# Adjacent vs non-adjacent vs randomized pairs in the reply-level logistic model.
#
# Same model as alignment_glmm.R, but fitted on all three pair conditions and without
# the metadata moderators. Accommodation (the a_uses effect) may differ by condition and
# family. If characters really respond to the previous turn, the a_uses effect should be
# clearly larger for adjacent pairs than for non-adjacent (3-5 turns later) and
# randomized pairs (B's replies shuffled within the dyad). This is the model-based version
# of the adjacent / non-adjacent / randomized comparison in Danescu-Niculescu-Mizil & Lee (2011).
#
# Note: randomized pairs reuse the same B replies as adjacent pairs (in shuffled order),
# so the two conditions are not independent samples; the b_speaker:family intercept
# absorbs part of this.
#
# Input:  outputs/alignment/reply_level.csv.gz  (python scripts/stats/build_alignment_table.py)
# Output: outputs/alignment/conditions_*.csv
#
# Usage (from repo root):
#   Rscript scripts/stats/alignment_conditions.R          # nAGQ = 0
#   Rscript scripts/stats/alignment_conditions.R --full   # nAGQ = 1 (slower, final numbers)
#   Rscript scripts/stats/alignment_conditions.R --test   # 10% of pairs, outputs in outputs/alignment/test/

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
NAGQ <- if ("--full" %in% args) 1 else 0
OUT <- "outputs/alignment"
CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# ── Data ───────────────────────────────────────────────────────────────────
d <- read.csv(file.path(OUT, "reply_level.csv.gz"), stringsAsFactors = FALSE)
if ("--test" %in% args) {
  set.seed(1)
  keys <- unique(paste(d$condition, d$play, d$pair_id))
  d <- d[paste(d$condition, d$play, d$pair_id) %in% sample(keys, length(keys) %/% 10), ]
  OUT <- file.path(OUT, "test")
  dir.create(OUT, showWarnings = FALSE)
}
cat(sprintf("Rows: %d\n", nrow(d)))
print(table(d$condition) / 9)

d$condition <- relevel(factor(d$condition), "adjacent")
d$family    <- relevel(factor(d$family), "adverbs")
d$log_a_len <- as.numeric(scale(log(d$a_len)))
d$log_b_len <- as.numeric(scale(log(d$b_len)))

# ── Models ─────────────────────────────────────────────────────────────────
# As in alignment_glmm.R, minus the work intercept: on the 3-condition data its variance
# is estimated at 0 (the play intercept already covers it) and makes the fit singular.
RE <- "(0 + a_uses | work) + (1 | play) + (1 | b_speaker:family)"
f_full    <- as.formula(paste("b_uses ~ a_uses * family * condition + family * log_b_len + log_a_len +", RE))
f_reduced <- as.formula(paste("b_uses ~ a_uses * family + condition * family + family * log_b_len + log_a_len +", RE))

fit <- function(f) glmer(f, data = d, family = binomial, nAGQ = NAGQ, control = CTRL)

t0 <- Sys.time()
cat(sprintf("Fitting full model (nAGQ = %d)...\n", NAGQ))
m_full <- fit(f_full)
print(summary(m_full)$varcor)
cat("Fitting reduced model (accommodation equal across conditions)...\n")
m_red <- fit(f_reduced)

a <- anova(m_red, m_full)
lrt <- data.frame(test = "a_uses:condition (+ a_uses:condition:family)",
                  df = a$Df[2], chi2 = a$Chisq[2], p = a$`Pr(>Chisq)`[2])
write.csv(lrt, file.path(OUT, "conditions_lrt.csv"), row.names = FALSE)
print(lrt, row.names = FALSE)

# ── Accommodation per condition x family, and family averages ──────────────
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

conds <- levels(d$condition)
fams <- levels(d$family)
L <- list()   # L[[condition]][[family]]
for (cnd in conds) {
  L[[cnd]] <- list()
  for (fam in fams) L[[cnd]][[fam]] <- contrast(m_full, c(paste0("family", fam), paste0("condition", cnd)))
  L[[cnd]][["all families"]] <- Reduce(`+`, L[[cnd]][fams]) / length(fams)
}

slopes <- do.call(rbind, lapply(conds, function(cnd) do.call(rbind, lapply(names(L[[cnd]]), function(fam)
  data.frame(condition = cnd, family = fam, t(estimate(m_full, L[[cnd]][[fam]])))))))
slopes$odds_ratio <- exp(slopes$estimate)
write.csv(slopes, file.path(OUT, "conditions_slopes.csv"), row.names = FALSE)

diffs <- do.call(rbind, lapply(setdiff(conds, "adjacent"), function(cnd) do.call(rbind, lapply(names(L[[cnd]]), function(fam)
  data.frame(comparison = paste0("adjacent - ", cnd), family = fam,
             t(estimate(m_full, L[["adjacent"]][[fam]] - L[[cnd]][[fam]])))))))
write.csv(diffs, file.path(OUT, "conditions_differences.csv"), row.names = FALSE)
print(diffs, row.names = FALSE, digits = 3)

writeLines(c(sprintf("nAGQ,%d", NAGQ), sprintf("n_obs,%d", nrow(d)),
             sprintf("singular,%s", isSingular(m_full)),
             sprintf("loglik,%.2f", as.numeric(logLik(m_full)))),
           file.path(OUT, "conditions_model_info.csv"))
cat(sprintf("\nAll outputs in %s/ (total %.1f min)\n", OUT,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
