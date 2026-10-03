#!/usr/bin/env Rscript
# Robustness check: is the genre effect on accommodation explained by play-level turn length?
#
# Absurdist plays have shorter turns on average (Beckett, Ionesco), and at the play level
# mean turn length predicts convergence better than genre (scripts/stats/turn_length.py).
# The main model (alignment_glmm.R) already controls for the length of each turn; here we
# add the play's mean (log) turn length as a moderator of accommodation (a_uses:play_len),
# which captures the style of a play (rapid exchanges vs long speeches), and ask whether
# a_uses:genre survives it.
#
# Same data, fixed and random effects as alignment_glmm.R, plus play_len and a_uses:play_len.
# LRTs (each against the model with both genre and play_len):
#   a_uses:genre     given play_len
#   a_uses:play_len  given genre
#
# Input:  outputs/alignment/reply_level.csv.gz
# Output: outputs/alignment/turnlen_lrt.csv, turnlen_fixed_effects.csv, turnlen_model_info.csv
#
# Usage (from repo root):
#   Rscript scripts/stats/turn_length_glmm.R          # nAGQ = 0
#   Rscript scripts/stats/turn_length_glmm.R --full   # nAGQ = 1 (final numbers)

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
NAGQ <- if ("--full" %in% args) 1 else 0
OUT <- "outputs/alignment"
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

# Play mean log turn length (one value per reply, not per row), standardized over plays
turns <- unique(d[c("play", "pair_id", "b_len")])
pl <- aggregate(cbind(play_len = log(b_len)) ~ play, data = turns, FUN = mean)
pl$play_len <- as.numeric(scale(pl$play_len))
d <- merge(d, pl, by = "play")
cat(sprintf("Rows: %d, plays: %d\n", nrow(d), nrow(pl)))

f_both <- b_uses ~ a_uses * (genre + language + family + importance_dyad + gender_dyad + play_len) +
  family * log_b_len + log_a_len +
  (1 + a_uses || work) + (1 | play) + (1 | b_speaker:family)

fit <- function(f) glmer(f, data = d, family = binomial, nAGQ = NAGQ, control = CTRL)
lrt_row <- function(name, reduced, bigger) {
  a <- anova(reduced, bigger)
  data.frame(test = name, df = a$Df[2], chi2 = a$Chisq[2], p = a$`Pr(>Chisq)`[2])
}

t0 <- Sys.time()
cat(sprintf("Fitting model with genre and play_len (nAGQ = %d)...\n", NAGQ))
m_both <- fit(f_both)
print(summary(m_both), correlation = FALSE)

cat("LRT: a_uses:genre given play_len\n")
m_no_genre <- fit(update(f_both, . ~ . - a_uses:genre))
cat("LRT: a_uses:play_len given genre\n")
m_no_len <- fit(update(f_both, . ~ . - a_uses:play_len))
lrt <- rbind(lrt_row("a_uses:genre | play_len", m_no_genre, m_both),
             lrt_row("a_uses:play_len | genre", m_no_len, m_both))
write.csv(lrt, file.path(OUT, "turnlen_lrt.csv"), row.names = FALSE)
print(lrt, row.names = FALSE)

co <- summary(m_both)$coefficients
fe <- data.frame(term = rownames(co), estimate = co[, 1], se = co[, 2], z = co[, 3], p = co[, 4],
                 odds_ratio = exp(co[, 1]), row.names = NULL)
write.csv(fe, file.path(OUT, "turnlen_fixed_effects.csv"), row.names = FALSE)

writeLines(c(sprintf("nAGQ,%d", NAGQ), sprintf("n_obs,%d", nrow(d)),
             sprintf("singular,%s", isSingular(m_both)),
             sprintf("loglik,%.2f", as.numeric(logLik(m_both)))),
           file.path(OUT, "turnlen_model_info.csv"))
cat(sprintf("\nDone in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
