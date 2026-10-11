#!/usr/bin/env Rscript
# Which feature families drive the English-French difference in accommodation?
#
# Same data and model as alignment_glmm.R, plus the three-way interaction a_uses:language:family
# (the analogue of the genre x family test). Reports the LRT for the interaction and, per family,
# the accommodation slope (log-odds of B using f after A used f) in English and in French.
#
# Input:  outputs/alignment/reply_level.csv.gz
# Output: outputs/alignment/language_family_lrt.csv, language_by_family.csv
#
# Usage (from repo root):
#   Rscript scripts/stats/language_family_glmm.R          # nAGQ = 0
#   Rscript scripts/stats/language_family_glmm.R --full   # nAGQ = 1

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
NAGQ <- if ("--full" %in% args) 1 else 0
OUT <- "outputs/alignment"
SUF <- if (NAGQ == 0) "_nagq0" else ""
CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

d <- read.csv(file.path(OUT, "reply_level.csv.gz"), stringsAsFactors = FALSE)
d <- d[d$condition == "adjacent", ]
d <- d[!grepl("U", d$gender_dyad), ]
d$genre           <- relevel(factor(d$genre), "Absurdist")
d$language        <- relevel(factor(d$language), "English")
d$family          <- relevel(factor(d$family), "adverbs")
d$gender_dyad     <- relevel(factor(d$gender_dyad), "F-F")
d$importance_dyad <- relevel(factor(d$importance_dyad), "intermediate->intermediate")
d$log_a_len <- as.numeric(scale(log(d$a_len)))
d$log_b_len <- as.numeric(scale(log(d$b_len)))

f_full <- b_uses ~ a_uses * (genre + language + family + importance_dyad + gender_dyad) +
  family * log_b_len + log_a_len +
  (1 + a_uses || work) + (1 | play) + (1 | b_speaker:family)
fit <- function(f) glmer(f, data = d, family = binomial, nAGQ = NAGQ, control = CTRL)

t0 <- Sys.time()
m_full <- fit(f_full)
m_lf <- fit(update(f_full, . ~ . + a_uses:language:family))
a <- anova(m_full, m_lf)
lrt <- data.frame(test = "a_uses:language:family", df = a$Df[2], chi2 = a$Chisq[2], p = a$`Pr(>Chisq)`[2])
write.csv(lrt, file.path(OUT, paste0("language_family_lrt", SUF, ".csv")), row.names = FALSE)
print(lrt, row.names = FALSE)

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
lf <- list()
for (fam in levels(d$family)) {
  f_lev <- paste0("family", fam)
  L_en <- contrast(m_lf, f_lev)
  L_fr <- contrast(m_lf, c(f_lev, "languageFrench"))
  lf[[fam]] <- data.frame(
    family = fam,
    english = estimate(m_lf, L_en)[["estimate"]],
    french = estimate(m_lf, L_fr)[["estimate"]],
    t(setNames(estimate(m_lf, L_fr - L_en), c("diff_french_minus_english", "diff_se", "diff_z", "diff_p"))))
}
lf <- do.call(rbind, lf)
write.csv(lf, file.path(OUT, paste0("language_by_family", SUF, ".csv")), row.names = FALSE)
print(lf, row.names = FALSE, digits = 3)
cat(sprintf("\nDone in %.1f min (nAGQ = %d, singular = %s)\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins")), NAGQ, isSingular(m_lf)))
