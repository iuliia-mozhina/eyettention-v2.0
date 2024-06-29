
library(dplyr)
library(tidyr)
library(ggplot2)
library(parallel)
library(hypr)
library(lme4)
library(lmerTest)
library(scales)
library(cowplot)

source("newstats.R")

new_factor_labels <- function(x) factor(x, levels = c("N","mL","sL","iW","mW"), labels=c("Normal","Mirrored letters","Scrambled","Reverse letters","Mirrored words"))

save_fig <- function(..., width, aspect) ggsave(..., width = width, height = width/aspect)
save_fig_narrow <- function(...) save_fig(..., width = 4)
save_fig_wide <- function(...) save_fig(..., width = 8)

corpus <- read.corpus("../DATA18/corpus_oculo_01.dat")

`%>>%` <- function(lhs, rhs) {
  ret <- eval(substitute(lhs %>% rhs, as.list(match.call())))
  message(paste(as.character(as.expression(match.call()$rhs)),"done."))
  ret
}

word_borders <- corpus %>% group_by(sno) %>% transmute(fw = iw, x0 = cumsum(lag(1+nl, default=0)))


priors <- list(
  'msac'=     trpd(  1.2  ,  3.2 ),
  'delta0'=   trpd(  1.5  , 15.0 ),
#  'asym'=     trpd(  0.01 ,  1.0 ),
  'log10omega'=trpd(-3.0  ,  0.0 ),
  'beta'=     trpd(  0.0  ,  1.0 ),
  'eta'=      trpd(  0.0  ,  1.0 ),
  'misfac'=   trpd(  0.0  ,  2.0 ),
  'tau_n2l'=  trpd(  0.05 ,  1.5 ),
  'refix'=    trpd(  0.0  ,  2.0 ),
  'sre_fssk1'=trpd(  0.1  ,  9.0 ),
  'sre_fs2'=  trpd(  0.0  ,  1.0 ),
  'sre_sk2'=  trpd(  0.0  ,  1.0 ),
  'sre_rf1'=  trpd(  0.1  ,  9.0 ),
  'sre_rf2'=  trpd(  0.0  ,  1.0 ),
  'sre_rg1'=  trpd( -9.0  , -0.1 ),
  'sre_rg2'=  trpd( -1.0  ,  0.0 )
)

match_pars <- function(p, v) {
  do.call(rbind, lapply(seq_along(p), function(i) {
    x <- p[i]
    y <- NULL
    if(x == "omn_1") {
      x <- c("omn_fs1", "omn_sk1", "omn_rg1", "omn_brf1", "omn_frf1")
    } else if(x == "omn_2") {
      x <- c("omn_fs2", "omn_sk2", "omn_rg2", "omn_brf2", "omn_frf2")
    } else if(x == "sre_fssk1") {
      x <- c("sre_fs1", "sre_sk1")
    } else if(x == "sre_fssk2") {
      x <- c("sre_fs2", "sre_sk2")
    } else if(x == "tau_n2l") {
      x <- c("tau_n", "tau_l")
      y <- c(v[i], 2*v[i])
    } else if(x == "sre_rf1") {
      x <- c("sre_frf1", "sre_brf1")
      y <- c(v[i], -v[i])
    } else if(x == "sre_rf2") {
      x <- c("sre_frf2", "sre_brf2")
      y <- c(v[i], -v[i])
    } else if(x == "log10omega") {
      x <- c("decay")
      y <- 10^v[i]
    }
    if(is.null(y)) {
      y <- rep(v[i], length(x))
    }
    data.frame(x, y)
  }))
}

pcat <- function(p) {
  cats <- c("(0,.001)","[.001, .01)","[.01, .05)")
  ret <- factor(rep(NA, length(p)), levels= cats)
  ret[p<.001] <- cats[1]
  ret[p>=.001&p<.01] <- cats[2]
  ret[p>=.01&p<.05] <- cats[3]
  ret
}


ll <- function(x) list(lapply(x[,1], function(z) {
  #"fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m", "fp_skipped_p", "fp_fwrefix_p", "fp_regin_p"
  if(z == "fp_skipped_p") return("Skipping")
  else if(z == "fp_fwrefix_p") return("Refixation")
  else if(z == "fp_regin_p") return("Regression")
  else if(z == "fp_first_fwfix_dur_m") return("First-fix.n dur.")
  else if(z == "fp_fwrefix_dur_m") return("Refix. dur.")
  else if(z == "fp_gaze_dur_m") return("Gaze dur.")
  else if(z == "fp_single_fwfix_dur_m") return("Single-fix. dur.")
  else if(z == "msac") return(expression(italic(t[sac])))
  else if(z == "delta0") return(expression(italic(delta)))
  else if(z == "alpha") return(expression(italic(alpha)))
  else if(z == "decay") return(expression(italic(omega)))
  else if(z == "log10omega") return(expression(log[10]~italic(omega)))
  else if(z == "beta") return(expression(italic(beta)))
  else if(z == "gamma") return(expression(italic(gamma)))
  else if(z == "eta") return(expression(italic(eta)))
  else if(z == "misprob") return(expression(italic(p[mis])))
  else if(z == "misfac") return(expression(italic(M)))
  else if(z == "tau_n2l") return(expression(italic(tau[n/l])))
  else if(z == "refix") return(expression(italic(R)))
  else if(z == "omn_1") return(expression(italic(omn[1])))
  else if(z == "omn_2") return(expression(italic(omn[2])))
  else if(z == "sre_fssk1") return(expression(italic(sre[1])))
  else if(z == "sre_fs2") return(expression(italic(sre[2]^(FS))))
  else if(z == "sre_sk2") return(expression(italic(sre[2]^(SK))))
  else if(z == "sre_rf1") return(expression(italic(sre[1]^(RF))))
  else if(z == "sre_rf2") return(expression(italic(sre[2]^(RF))))
  else if(z == "sre_rg1") return(expression(italic(sre[1]^(RG))))
  else if(z == "sre_rg2") return(expression(italic(sre[2]^(RG))))
  else return(as.character(z))
}))

n_burnin <- 5000

theme_apa <- function() {
  theme_bw() + 
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black"),
      text = element_text(family = "serif"),
      axis.text = element_text(color = "black", size = rel(.8)),
      axis.ticks.x.top = element_line(),
      axis.ticks.x.bottom = element_line(),
      axis.ticks.y.left = element_line(),
      axis.ticks.y.right = element_line(),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_blank(),
      strip.background = element_blank(),
      complete = FALSE
    )
}

color_apa <- function(pal = "Dark2", aes = c("colour", "fill"),...) {
  scale_color_brewer(palette = pal, aesthetics = aes, ...)
}


datasets <- read.delim("datasets.txt") %>% subset(session %in% c(1,2))


if(file.exists("chains_v33.rda")) {
  load("chains_v33.rda")
} else {
  all_chains <- do.call(rbind, lapply(seq_len(nrow(datasets)), function(i) do.call(rbind, lapply(1:5, function(j) {
    fname <- sprintf("../fitting/oc33_oculo_01_%s_%d.dat", datasets$file[i], j)
    message(sprintf("Reading %s...", fname))
    df <- read.table(fname, header = TRUE)
    cbind(df, vp = datasets$vp[i], cond = datasets$condition[i], set = datasets$set[i], n = 1:nrow(df))
  }))))
  save(all_chains, file="chains_v33.rda", compress="xz")
}




mcmc_chains <- all_chains %>% subset(n > 5000)
mcmc_chains_narrow <- mcmc_chains %>% gather(param, value, -loglik, -vp, -cond, -n, -set)
mcmc_thinnable <- mcmc_chains_narrow %>% 
  group_by(vp, cond, set, param) %>%
  mutate(ran = sample(n()))
mcmc_thinned <- mcmc_thinnable %>% subset(ran <= 100)



empirical_fixseqs <- do.call(rbind, lapply(1:nrow(datasets), function(i) {
  df <- read.table(sprintf("../DATA18/fixseqin_%s_test.dat", datasets$file[i]), col.names = c("sno", "fw", "fl", "tfix", "tsac", "first_last"))
  df[,c("subject","set","cond")] <- datasets[i,c("vp","set","condition")]
  df
}))

empirical_fixseqs_annotated <- empirical_fixseqs %>>% 
  mutate(subject = set) %>>% 
  fixseq_annotate(corpus)

empirical_fixseqs_annotated_trial_stats <- empirical_fixseqs_annotated %>>% fixseq_stats_by_trial(corpus)

empirical_fixstats <- empirical_fixseqs_annotated_trial_stats %>>% 
  mutate(nl = factor(ifelse(nl <= 3, "3-", ifelse(nl >= 10, "10+", nl)), levels=c("3-","4","5","6","7","8","9","10+"))) %>>%
  group_by(subject, nl) %>>% 
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])

empirical_fixstats_agg <- empirical_fixseqs_annotated_trial_stats%>>%
  group_by(subject) %>>%
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])


param_template <- read.table("../DATA18/swpar_default.par")
dat1 <- do.call(rbind, lapply(1:nrow(datasets), function(i) {
  sentences <- unique(subset(empirical_fixseqs, set == datasets$set[i])$sno)
  #sentences <- 1:150
  params <- mcmc_chains[sample(which(mcmc_chains$set == datasets$set[i]), 10), names(priors)]
  df <- do.call(rbind, lapply(seq_len(nrow(params)), function(j) {
    matched_params <- match_pars(colnames(params), t(params[j,]))
    new_params <- param_template
    new_params[match(matched_params$x, as.character(new_params[,1])), 2] <- matched_params$y
    #write.table(new_params, sprintf("SIM35/swpar_oculo_01_%s.par", datasets$file[i]), col.names = FALSE, row.names = FALSE, quote = FALSE)
    system2("../SWIFT/swiftstat7p_v33", c("-qgxc", "oculo_01", "-s", as.character(datasets$file[i]), "-i", "../DATA18", "-o", "SIM35", "-I", paste(sentences, collapse=","), "-P", paste(sprintf("%s=%g", new_params[,1], new_params[,2]), collapse=",")))
    read.table(sprintf("SIM35/fixseqin_%s.dat", datasets$file[i]), col.names = c("sno", "fw", "fl", "tfix", "tsac", "first_last"))
  }))
  df[,c("subject","set","cond")] <- datasets[i,c("vp","set","condition")]
  df
}))

trial_length <- which(dat1$first_last == 2)-which(dat1$first_last == 1)+1
dat1$trial_id <- rep(seq_along(trial_length), trial_length)

dat2 <- do.call(rbind, lapply(1:nrow(datasets), function(i) {
  sentences <- unique(subset(empirical_fixseqs, set == datasets$set[i])$sno)
  #sentences <- 1:150
  params <- (mcmc_chains %>% subset(set == datasets$set[i]))[,names(priors)] %>% summarize_all(hpd.midpoint, p=.3)
  df <- do.call(rbind, lapply(seq_len(nrow(params)), function(j) {
    matched_params <- match_pars(colnames(params), t(params[j,]))
    new_params <- param_template
    new_params[match(matched_params$x, as.character(new_params[,1])), 2] <- matched_params$y
    #write.table(new_params, sprintf("SIM35/swpar_oculo_01_%s.par", datasets$file[i]), col.names = FALSE, row.names = FALSE, quote = FALSE)
    system2("../SWIFT/swiftstat7p_v33", c("-qgxc", "oculo_01", "-s", as.character(datasets$file[i]), "-i", "../DATA18", "-o", "SIM35", "-I", paste(sentences, collapse=","), "-P", paste(sprintf("%s=%g", new_params[,1], new_params[,2]), collapse=",")))
    read.table(sprintf("SIM35/fixseqin_%s.dat", datasets$file[i]), col.names = c("sno", "fw", "fl", "tfix", "tsac", "first_last"))
  }))
  df[,c("subject","set","cond")] <- datasets[i,c("vp","set","condition")]
  df
}))


trial_length <- which(dat2$first_last == 2)-which(dat2$first_last == 1)+1
dat2$trial_id <- rep(seq_along(trial_length), trial_length)

#dat3 <- do.call(rbind, lapply(1:nrow(datasets), function(i) {
#  sentences <- unique(subset(empirical_fixseqs, set == datasets$set[i])$sno)
#  #sentences <- 1:150
#  params <- (mcmc_chains %>% subset(set == datasets$set[i]))[,names(priors)] %>% summarize_all(hpd.midpoint, p=.3)
#  df <- do.call(rbind, lapply(seq_len(nrow(params)), function(j) {
#    matched_params <- match_pars(colnames(params), t(params[j,]))
#    new_params <- param_template
#    new_params[match(matched_params$x, as.character(new_params[,1])), 2] <- matched_params$y
#    #write.table(new_params, sprintf("SIM35/swpar_oculo_01_%s.par", datasets$file[i]), col.names = FALSE, row.names = FALSE, quote = FALSE)
#    system2("../SWIFT/swiftstat7p_v33_execsaccgauss", c("-gxc", "oculo_01", "-s", as.character(datasets$file[i]), "-i", "../DATA18", "-o", "SIM35", "-I", paste(sentences, collapse=","), "-P", paste(sprintf("%s=%g", new_params[,1], new_params[,2]), collapse=",")))
#    read.table(sprintf("SIM35/fixseqin_%s.dat", datasets$file[i]), col.names = c("sno", "fw", "fl", "tfix", "tsac", "first_last"))
#  }))
#  df[,c("subject","set","cond")] <- datasets[i,c("vp","set","condition")]
#  df
#}))
#
#trial_length <- which(dat3$first_last == 2)-which(dat3$first_last == 1)+1
#dat3$trial_id <- rep(seq_along(trial_length), trial_length)

slen <- corpus$nw[c(1, which(corpus$sno != lag(corpus$sno)))]

simulated_fixseqs <- dat1 %>% 
  group_by(trial_id) %>% 
  subset(tfix >= 40) %>%
  mutate(after_last_fixated = {
    last_fixated <- which(fw >= slen[sno])
    if(length(last_fixated) > 0) {
      i <- last_fixated[1]
      while(i < length(fw) && fw[i+1] == fw[i]) {
        i <- i + 1
      }
      # i is the last fixation of that sequence to include
      ret <- c(rep(FALSE, i), rep(TRUE, n()-i))
    } else {
      FALSE
    }
  }) %>%
  subset(!after_last_fixated) %>%
  mutate(ifix = seq_len(n())) %>%
  mutate(first_last = c(1L, rep(0L, n()-2), 2L))

simulated_fixseqs_annotated <- simulated_fixseqs %>>% 
  mutate(subject = set) %>>% 
  fixseq_annotate(corpus)

simulated_fixseqs_annotated_trial_stats <- simulated_fixseqs_annotated %>>%
  fixseq_stats_by_trial(corpus)

simulated_fixstats <- simulated_fixseqs_annotated_trial_stats %>>% 
  mutate(nl = factor(ifelse(nl <= 3, "3-", ifelse(nl >= 10, "10+", nl)), levels=c("3-","4","5","6","7","8","9","10+"))) %>%
  group_by(subject, nl) %>>% 
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])

simulated_fixstats_agg <- simulated_fixseqs_annotated_trial_stats %>>% 
  group_by(subject) %>>% 
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])


simulated_fixseqs2 <- dat2 %>% 
  group_by(trial_id) %>% 
  subset(tfix >= 40) %>%
  mutate(after_last_fixated = {
    last_fixated <- which(fw >= slen[sno])
    if(length(last_fixated) > 0) {
      i <- last_fixated[1]
      while(i < length(fw) && fw[i+1] == fw[i]) {
        i <- i + 1
      }
      # i is the last fixation of that sequence to include
      ret <- c(rep(FALSE, i), rep(TRUE, n()-i))
    } else {
      FALSE
    }
  }) %>%
  subset(!after_last_fixated) %>%
  mutate(ifix = seq_len(n())) %>%
  mutate(first_last = c(1L, rep(0L, n()-2), 2L))

simulated_fixseqs_annotated2 <- simulated_fixseqs2 %>>% 
  mutate(subject = set) %>>% 
  fixseq_annotate(corpus)

simulated_fixseqs_annotated_trial_stats2 <- simulated_fixseqs_annotated2 %>>%
  fixseq_stats_by_trial(corpus)

simulated_fixstats2 <- simulated_fixseqs_annotated_trial_stats2 %>>% 
  mutate(nl = factor(ifelse(nl <= 3, "3-", ifelse(nl >= 10, "10+", nl)), levels=c("3-","4","5","6","7","8","9","10+"))) %>%
  group_by(subject, nl) %>>% 
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])

simulated_fixstats_agg2 <- simulated_fixseqs_annotated_trial_stats2 %>>% 
  group_by(subject) %>>% 
  fixstats_stats_summary() %>>%
  rename(set = subject) %>>%
  mutate(vp = datasets$vp[set], cond = datasets$condition[set])

#simulated_fixseqs3 <- dat3 %>% 
#  group_by(trial_id) %>% 
#  subset(tfix >= 40) %>%
#  mutate(after_last_fixated = {
#    last_fixated <- which(fw >= slen[sno])
#    if(length(last_fixated) > 0) {
#      i <- last_fixated[1]
#      while(i < length(fw) && fw[i+1] == fw[i]) {
#        i <- i + 1
#      }
#      # i is the last fixation of that sequence to include
#      ret <- c(rep(FALSE, i), rep(TRUE, n()-i))
#    } else {
#      FALSE
#    }
#  }) %>%
#  subset(!after_last_fixated) %>%
#  mutate(ifix = seq_len(n())) %>%
#  mutate(first_last = c(1L, rep(0L, n()-2), 2L))

#simulated_fixseqs_annotated3 <- simulated_fixseqs3 %>>% 
#  mutate(subject = set) %>>% 
#  fixseq_annotate(corpus)
#
#simulated_fixseqs_annotated_trial_stats3 <- simulated_fixseqs_annotated3 %>>%
#  fixseq_stats_by_trial(corpus)

#simulated_fixstats3 <- simulated_fixseqs_annotated_trial_stats3 %>>% 
#  mutate(nl = factor(ifelse(nl <= 3, "3-", ifelse(nl >= 10, "10+", nl)), levels=c("3-","4","5","6","7","8","9","10+"))) %>%
#  group_by(subject, nl) %>>% 
#  fixstats_stats_summary() %>>%
#  rename(set = subject) %>>%
#  mutate(vp = datasets$vp[set], cond = datasets$condition[set])

#simulated_fixstats_agg3 <- simulated_fixseqs_annotated_trial_stats3 %>>% 
#  group_by(subject) %>>% 
#  fixstats_stats_summary() %>>%
#  rename(set = subject) %>>%
#  mutate(vp = datasets$vp[set], cond = datasets$condition[set])


all_fixstats <- rbind(
  empirical_fixstats %>% mutate(type = "empirical"),
  simulated_fixstats %>% mutate(type = "simulated")
) %>% 
  gather(stat, val, -type, -cond, -vp, -set, -nl)

all_fixstats2 <- rbind(
  empirical_fixstats %>% mutate(type = "empirical"),
  simulated_fixstats %>% mutate(type = "simulated_ps"),
  simulated_fixstats2 %>% mutate(type = "simulated_pe")
) %>% 
  gather(stat, val, -type, -cond, -vp, -set, -nl)

#all_fixstats3 <- rbind(
#  empirical_fixstats %>% mutate(type = "empirical"),
#  simulated_fixstats2 %>% mutate(type = "simulated_gamma"),
#  simulated_fixstats3 %>% mutate(type = "simulated_gauss")
#) %>% 
#  gather(stat, val, -type, -cond, -vp, -set, -nl)

all_fixstats_agg <- all_fixstats %>%
  group_by(cond, type, nl, stat) %>%
  select(-set, -vp) %>%
  summarize(
    m = mean(val, na.rm = TRUE),
    sd = sd(val, na.rm = TRUE),
    se = sd / sqrt(n()),
    m_lb = quantile(val, .025, na.rm = T),
    m_ub = quantile(val, .975, na.rm = T)
  )

all_fixstats_agg2 <- all_fixstats2 %>%
  group_by(cond, type, nl, stat) %>%
  select(-set, -vp) %>%
  summarize(
    m = mean(val, na.rm = TRUE), 
    sd = sd(val, na.rm = TRUE), 
    m_lb = quantile(val, .025, na.rm = T),
    m_ub = quantile(val, .975, na.rm = T)
  )

#all_fixstats_agg3 <- all_fixstats3 %>%
#  group_by(cond, type, nl, stat) %>%
#  select(-set, -vp) %>%
#  summarize(
#    m = mean(val, na.rm = TRUE), 
#    sd = sd(val, na.rm = TRUE), 
#    m_lb = quantile(val, .025, na.rm = T),
#    m_ub = quantile(val, .975, na.rm = T)
#  )

all_agg_fixstats <- rbind(
  empirical_fixstats_agg %>% mutate(type = "empirical"),
  simulated_fixstats_agg %>% mutate(type = "simulated")
) %>% 
  gather(stat, val, -type, -cond, -vp, -set) %>%
  spread(type, val)


all_agg_fixstats2 <- rbind(
  empirical_fixstats_agg %>% mutate(type = "empirical"),
  simulated_fixstats_agg %>% mutate(type = "simulated_ps"),
  simulated_fixstats_agg2 %>% mutate(type = "simulated_pe")
) %>% 
  gather(stat, val, -type, -cond, -vp, -set) %>%
  spread(type, val)


#all_agg_fixstats3 <- rbind(
#  empirical_fixstats_agg %>% mutate(type = "empirical"),
#  simulated_fixstats_agg2 %>% mutate(type = "simulated_gamma"),
#  simulated_fixstats_agg3 %>% mutate(type = "simulated_gauss")
#) %>% 
#  gather(stat, val, -type, -cond, -vp, -set) %>%
#  spread(type, val)


all_fixseqs <- bind_rows(
  empirical_fixseqs_annotated %>% mutate(type = "empirical"),
  simulated_fixseqs_annotated %>% mutate(type = "simulated")
) %>% left_join(word_borders, by=c("sno","fw")) %>% mutate(x = x0+fl)

trial_length <- which(all_fixseqs$first_last == 2)-which(all_fixseqs$first_last == 1)+1
all_fixseqs$trial_id <- rep(seq_along(trial_length), trial_length)

all_fixseqs2 <- bind_rows(
  empirical_fixseqs_annotated %>% mutate(type = "empirical"),
  simulated_fixseqs_annotated %>% mutate(type = "simulated_ps"),
  simulated_fixseqs_annotated2 %>% mutate(type = "simulated_pe")
) %>% left_join(word_borders, by=c("sno","fw")) %>% mutate(x = x0+fl)

trial_length <- which(all_fixseqs2$first_last == 2)-which(all_fixseqs2$first_last == 1)+1
all_fixseqs2$trial_id <- rep(seq_along(trial_length), trial_length)


#all_fixseqs3 <- bind_rows(
#  empirical_fixseqs_annotated %>% mutate(type = "empirical"),
#  simulated_fixseqs_annotated2 %>% mutate(type = "simulated_gamma"),
#  simulated_fixseqs_annotated3 %>% mutate(type = "simulated_gauss")
#) %>% left_join(word_borders, by=c("sno","fw")) %>% mutate(x = x0+fl)
#
#trial_length <- which(all_fixseqs3$first_last == 2)-which(all_fixseqs3$first_last == 1)+1
#all_fixseqs3$trial_id <- rep(seq_along(trial_length), trial_length)


saccamps <- all_fixseqs %>% 
  group_by(cond, type, trial_id) %>% 
  transmute(amp = x - lag(x), ftype = ifelse(is_refix, "refixation", ifelse(is_reg, "regression", ifelse(is_skip, "skipping", "forward"))))

saccamps2 <- all_fixseqs2 %>% 
  group_by(cond, type, trial_id) %>% 
  transmute(amp = x - lag(x), ftype = ifelse(is_refix, "refixation", ifelse(is_reg, "regression", ifelse(is_skip, "skipping", "forward"))))

#saccamps3 <- all_fixseqs3 %>% 
#  group_by(cond, type, trial_id) %>% 
#  transmute(amp = x - lag(x), ftype = ifelse(is_refix, "refixation", ifelse(is_reg, "regression", ifelse(is_skip, "skipping", "forward"))))

centered_landing_positions <- all_fixseqs %>%
  ungroup() %>%
  mutate(nl = corpus$nl[match(paste(sno, fw), paste(corpus$sno, corpus$iw))], nw = corpus$nw[match(paste(sno, fw), paste(corpus$sno, corpus$iw))]) %>%
  group_by(cond, type, subject, trial_id) %>%
  transmute(
    sno, fw, fl, nw,
    centered_landing_position = floor(fl) + 0.5 - nl/2 - 1,
    centered_launch_site_distance = floor(lag(x)) - (x0 + 1 + nl/2)
  ) %>%
  group_by(cond, type, centered_launch_site_distance)

centered_landing_positions_all <- centered_landing_positions %>%
  subset(fw != 1 & fw != nw) %>%
  summarize_at(vars(centered_landing_position), list(m = mean, sd = sd))
centered_landing_positions_fs <- centered_landing_positions %>%
  subset(fw != 1 & fw != nw & fw == lag(fw) + 1) %>%
  summarize_at(vars(centered_landing_position), list(m = mean, sd = sd))
centered_landing_positions_sk <- centered_landing_positions %>%
  subset(fw != 1 & fw != nw & fw > lag(fw) + 1) %>%
  summarize_at(vars(centered_landing_position), list(m = mean, sd = sd))
centered_landing_positions_rf <- centered_landing_positions %>%
  subset(fw != 1 & fw != nw & fw == lag(fw)) %>%
  summarize_at(vars(centered_landing_position), list(m = mean, sd = sd))
centered_landing_positions_rg <- centered_landing_positions %>%
  subset(fw != 1 & fw != nw & fw < lag(fw)) %>%
  summarize_at(vars(centered_landing_position), list(m = mean, sd = sd))

prior.y <- do.call(rbind, lapply(
  seq_along(priors),
  function(i) {
    x <- seq(priors[[i]]$a, priors[[i]]$b, length.out = 1000)
    y <- do.call(priors[[i]][[1]], c(list(x), priors[[i]][-1]))
    data.frame(x, y, param = names(priors)[i])
  }
))


forward_fixations <- all_fixseqs %>% 
  subset(is_fwfix | is_fwrefix & lag(is_fwfix)) %>% 
  mutate(ffix_type = ifelse(is_refix, "second", ifelse(is_singlefix, "single", "first"))) %>%
  left_join(corpus[,c("sno","iw","nl")], by = c("sno", "fw"="iw"))

landing_positions <- forward_fixations %>% 
  subset(nl >= 4 & nl <= 7 & !is.na(ffix_type)) %>%
  mutate(rfl = floor(fl), cfl = rfl - nl/2 - 0.5) %>% 
  group_by(type, cond, ffix_type, nl) %>% 
  mutate(.n = n()) %>%
  group_by(rfl, add = TRUE) %>%
  summarize(d = n() / first(.n), cfl = first(cfl))

p5 <- ggplot(mcmc_chains_narrow %>% subset(n %% 10 == 0) %>% mutate(cond = new_factor_labels(cond))) +
  theme_apa() +  color_apa(aes = "color") + theme(legend.direction = "horizontal", legend.position = "bottom", strip.placement = "outside", strip.switch.pad.wrap = unit(-3,"pt")) +
  labs(x = NULL, y = NULL, color = "Condition") +
  facet_wrap(~param, scales = "free", labeller = ll, strip.position = "bottom", ncol = 5) +
  geom_density(aes(x=value, color=cond)) +
  geom_line(aes(x=x, y=y), color="white", linetype="solid", size = 1, data = prior.y) +
  geom_line(aes(x=x, y=y), color="gray", linetype="dashed", size = .5, data = prior.y)

p5a <- ggplot(mcmc_chains_narrow %>% subset(n %% 10 == 0 & cond == "N")) +
  theme_apa() + theme(legend.position = "none", strip.placement = "outside", strip.switch.pad.wrap = unit(-3,"pt")) +
  labs(x = NULL, y = NULL, color = "Subject") +
  facet_wrap(~param, scales = "free", labeller = ll, strip.position = "bottom", ncol = 5) +
  geom_density(aes(x=value, color=as.factor(vp)), size = .5, alpha = .5) +
  geom_density(aes(x=value), color = "black") +
  geom_line(aes(x=x, y=y), color="white", linetype="solid", size = 1, data = prior.y) +
  geom_line(aes(x=x, y=y), color="gray", linetype="dashed", size = .5, data = prior.y)

mcmc_cor <- as.data.frame(cor(mcmc_chains[,names(priors)])) %>% 
  mutate(x=names(priors)) %>% 
  gather(y, r, -x) %>% 
  subset(match(x, names(priors)) <= match(y, names(priors))) %>%
  mutate(x = factor(x, levels=names(priors)), y = factor(y, levels=rev(names(priors))))

p5b <- ggplot(mcmc_cor) +
  theme_apa() +
  theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5), legend.position = c(1,1), legend.direction = "horizontal", legend.justification = c(1,1)) +
  scale_fill_gradient2(limits = c(-1,1), low = "red", high = "blue", mid = "white", midpoint = 0) +
  labs(x = NULL, y = NULL, fill=NULL) +
  coord_fixed() +
  geom_tile(aes(x=x,y=y,fill=r),width=1,height=1)


p6 <- ggplot(all_fixstats_agg %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p")) %>% mutate(stat = factor(stat, levels = c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p"))), aes(x=nl, y=m, group=type, color=type, shape=type, linetype=type)) +
  theme_apa() + color_apa() + theme(strip.placement = "outside", legend.position = "bottom", legend.direction = "horizontal") +
  labs(y = NULL, x = "Word length", color = NULL, shape=NULL, linetype=NULL, fill=NULL) +
  facet_grid(stat~cond, scales = "free_y", switch = "y", labeller = ll) +
  geom_line() +
  geom_point() + 
  geom_ribbon(aes(x=nl, ymin=pmax(0,m-sd), ymax=pmin(1,m+sd), fill=type, color=NA), alpha=.2)

p6a <- ggplot(all_fixstats_agg2 %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p")) %>% mutate(stat = factor(stat, levels = c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p"))), aes(x=nl, y=m, group=type, color=type, shape=type, linetype=type)) +
  theme_apa() + color_apa() + theme(strip.placement = "outside", legend.position = "bottom", legend.direction = "horizontal") +
  labs(y = NULL, x = "Word length", color = NULL, shape=NULL, linetype=NULL) +
  facet_grid(stat~cond, scales = "free_y", switch = "y", labeller = ll) +
  geom_line() +
  geom_point()


cstats <- all_agg_fixstats %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p", "fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m"))

ts <- unique(cstats[,c("cond","stat")])

ts <- cbind(ts, bind_rows(lapply(1:nrow(ts), function(i) {
  dat <- merge(cstats, ts[i,])
  test <- cor.test(x=dat$empirical, dat$simulated)
  data.frame(rho = test$estimate, p = test$p.value)
}))) %>% mutate(p.s = 1-(1-p)^n(), sig = p < .05)

ts %>% mutate(txt = sprintf("%.2f (.%03d)", rho, ceiling(p*1000))) %>% select(cond, stat, txt) %>% spread(cond, txt)


p7 <- ggplot(all_agg_fixstats %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p")) %>% mutate(stat = factor(stat, levels=c("fp_skipped_p", "fp_fwrefix_p", "fp_regin_p"), labels=c("Skipping", "Refixation", "Regression"))) , aes(x=empirical, y=simulated, color=stat, linetype=stat, shape=stat)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Empirical", y = "Simulated", color = NULL, linetype=NULL, shape=NULL, fill=NULL) +
  facet_wrap(~cond, scales = "free", ncol = 5) +
  geom_point(size = 1) +
  stat_ellipse() +
  geom_abline(slope = 1)

p7a1 <- ggplot(all_agg_fixstats2 %>% ungroup() %>% subset(cond == "mL") %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p")) %>% mutate(stat = factor(stat, levels=c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p"), labels=c("Skipping", "Refixation", "Regression"))) , aes(x=empirical, y=simulated_pe, color=stat, linetype=stat, shape=stat)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Empirical", y = "Simulated", color = NULL, linetype=NULL, shape=NULL) +
  geom_point(size = 1) +
  stat_ellipse() +
  geom_abline(slope = 1)

p7a2 <- ggplot(all_agg_fixstats2 %>% ungroup() %>% subset(cond == "mL") %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p")) %>% mutate(stat = factor(stat, levels=c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p"), labels=c("Skipping", "Refixation", "Regression"))) , aes(x=empirical, y=simulated_ps, color=stat, linetype=stat, shape=stat)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Empirical", y = "Simulated", color = NULL, linetype=NULL, shape=NULL) +
  geom_point(size = 1) +
  stat_ellipse() +
  geom_abline(slope = 1)

p7b1 <- ggplot(all_fixstats_agg2 %>% ungroup() %>% subset(cond == "mL") %>% mutate(type = factor(type, levels=c("empirical","simulated_ps","simulated_pe"), labels=c("Empirical", "Posterior sampling", "Point estimate"))) %>% subset(stat %in% c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p")) %>% mutate(stat = factor(stat, levels = c("fp_skipped_p", "fp_fwrefix_p", "fp_regout_p"))), aes(x=nl, y=m, group=type, color=type, shape=type, linetype=type)) +
  theme_apa() + color_apa() + theme(strip.placement = "outside", legend.position = "bottom", legend.direction = "horizontal") +
  labs(y = NULL, x = "Word length", color = NULL, shape=NULL, linetype=NULL) +
  facet_grid(stat~., scales = "free_y", switch = "y", labeller = ll) +
  geom_line() +
  geom_point()

p7x <- plot_grid(p7b1, plot_grid(p7a1, p7a2, ncol=1, labels=c("B","C")), ncol = 2, labels = c("A",""))

p8 <- ggplot(saccamps %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(amp >= -20 & amp <= 20), aes(x=amp, y=..density.., color=type, linetype=type, shape=type)) + 
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Saccade amplitude", y = "Relative frequency", color=NULL, linetype=NULL, shape=NULL) +
  facet_grid(~cond) + 
  stat_bin(geom = "line", position = "identity", binwidth = 1)

p8a <- ggplot(saccamps %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(amp >= -20 & amp <= 20), aes(x=amp, y=..density.., color=type, linetype=type, shape=type)) + 
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Saccade amplitude", y = "Relative frequency", color=NULL, linetype=NULL, shape=NULL) +
  facet_wrap(~cond, ncol = 3) + 
  stat_bin(geom = "line", position = "identity", binwidth = 1) + 
  stat_bin(geom = "point", position = "identity", binwidth = 1)

p8b <- ggplot(saccamps %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(amp >= -20 & amp <= 20), aes(x=amp, y=..density.., color=type, linetype=type, shape=type)) + 
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Saccade amplitude", y = "Relative frequency", color=NULL, shape=NULL, linetype=NULL) +
  facet_grid(ftype~cond) + 
  stat_bin(geom = "line", position = "identity", binwidth = 1, color = "gray", linetype="solid", data=saccamps3 %>% na.omit %>% subset(type == "simulated_gauss")  %>% ungroup() %>% mutate(cond = new_factor_labels(cond))) +
  stat_bin(geom = "line", position = "identity", binwidth = 1) + 
  geom_vline(xintercept = 0, color="red")

p9 <- ggplot(landing_positions %>% ungroup() %>% mutate(cond = new_factor_labels(cond)), aes(x = rfl, y = d, color = type, shape = type, linetype=type)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Within-word landing position", y = "Relative frequency", color = NULL, shape = NULL, linetype = NULL) +
  facet_grid(cond ~ nl + ffix_type, scales = "free", labeller = function(df) {
    if(ncol(df) == 1) {
      data.frame(as.character(df[,1]), stringsAsFactors = FALSE)
    } else {
      df[,1] <- sprintf("%d letters", df[,1])
      df
    }
  }) +
  stat_smooth(method="lm", formula = y ~ I(x^2) + I(x), se = FALSE) +
  geom_point()

p10 <- ggplot(all_fixstats_agg %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m")) %>% mutate(stat = factor(stat, c("fp_gaze_dur_m", "fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_single_fwfix_dur_m"))), aes(x=nl, y=m, group=type, color=type, linetype=type, shape=type)) +
  theme_apa() + color_apa() + theme(strip.placement = "outside", legend.position = "bottom", legend.direction = "horizontal") +
  labs(y = NULL, x = "Word length", color = NULL, linetype=NULL, shape=NULL, fill=NULL) +
  facet_grid(stat~cond, scales = "free_y", switch = "y", labeller = ll) +
  geom_line() +
  geom_point() + 
  geom_ribbon(aes(x=nl, ymin=pmax(0,m-sd), ymax=m+sd, fill=type, linetype=NA), alpha=.2)

p11 <- ggplot(all_agg_fixstats %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m")) %>% mutate(stat = factor(stat, levels=c("fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m"), labels=c("First fixation", "Refixation", "Gaze", "Single fixation"))), aes(x=empirical, y=simulated, color=stat, linetype=stat, shape=stat)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  labs(x = "Empirical", y = "Simulated", color = NULL, linetype=NULL, shape=NULL) +
  facet_wrap(~cond, scales = "free", ncol = 5) +
  geom_point(size = 1) +
  stat_ellipse() +
  geom_abline(slope = 1)

p11a <- ggplot(all_agg_fixstats %>% ungroup() %>% mutate(cond = new_factor_labels(cond)) %>% subset(stat %in% c("fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m"))  %>% mutate(stat = factor(stat, levels=c("fp_first_fwfix_dur_m", "fp_fwrefix_dur_m", "fp_gaze_dur_m", "fp_single_fwfix_dur_m"), labels=c("First fixation", "Refixation", "Gaze", "Single fixation"))), aes(x=empirical, y=simulated, color=stat, linetype=stat, shape=stat)) +
  theme_apa() + color_apa() + theme(legend.position = "bottom", legend.direction = "horizontal") +
  theme(axis.text = element_text(size=rel(.5))) +
  labs(x = "Empirical", y = "Simulated", color = NULL, linetype = NULL, shape = NULL) +
  facet_wrap(~cond, scales = "free", ncol = 5) +
  geom_point(size = .2) +
  stat_ellipse() +
  geom_abline(slope = 1)

save_fig_wide("figs-33/fig5-posterior-densities.pdf", p5, aspect = 1)
save_fig_wide("figs-33/fig5a-posterior-densities.pdf", p5a, aspect = 1)
save_fig_narrow("figs-33/fig5b-posterior-densities.pdf", p5b, aspect = 1)
save_fig_wide("figs-33/fig6-spatial-sumstats-by-wlen.pdf", p6, aspect = 2)
save_fig_wide("figs-33/fig7-spatial-sumstats-by-vp.pdf", p7, aspect = 3)
save_fig_wide("figs-33/fig7x-compare-stats-ps-pe.pdf", p7x, aspect = 1.5)
save_fig_wide("figs-33/fig8-saccade-amplitudes.pdf", p8, aspect = 3)
save_fig_narrow("figs-33/fig8-n-saccade-amplitudes.pdf", p8a, aspect = 1)
save_fig_wide("figs-33/fig8b-saccade-amplitudes.pdf", p8b, aspect = 1)
save_fig_wide("figs-33/fig9-landing-positions.pdf", p9, aspect = 1.5)
save_fig_wide("figs-33/fig10-temporal-sumstats-by-wlen.pdf", p10, aspect = 1.7)
save_fig_wide("figs-33/fig11-temporal-sumstats-by-vp.pdf", p11, aspect = 3)





hyps <- hypr(
  baseline = ~N,
  letter_flip = mL~N,
  word_flip = iW~N,
  both_flip = mW~N+(mL-N)+(iW-N),
  scrambled = sL~mL,
  levels = levels(mcmc_chains$cond)
)

contrasts(mcmc_thinned$cond) <- contr.hypothesis(hyps, remove_intercept = TRUE)

#mcmc_thinned %>% ungroup() %>% select(vp, n, cond, param, value) %>% spread(param, value)

effs <- do.call(rbind, lapply(names(priors), function(parname) {
  mmat <- model.matrix(~1+cond, mcmc_thinned %>% subset(param == parname))
  cbind(as.data.frame(coef(summary(lmer(value~0+mmat + (0+mmat|vp), data = mcmc_thinned %>% subset(param == parname), REML = FALSE, control = lmerControl(calc.derivs = FALSE, optimizer = "bobyqa", optCtrl = list(maxfun = 1e6)))))), param = parname, effect = names(hyps))
})) %>% 
  group_by(param) %>% 
  mutate(
    EstimateMean = Estimate + ifelse(effect == "baseline", 0, Estimate[effect == "baseline"]), 
    p = 1-(1-`Pr(>|t|)`)^length(priors),
    effect = factor(effect, levels = c("both_flip", "letter_flip", "changed_order", "word_flip", "scrambled", "manipulation", "baseline"), labels = c("Both flipped", "Letters flipped", "Changed order", "Word inverted", "Scrambled", "Manipulated", "Normal"))
  )

model_plot_alpha <- .05

p12 <- ggplot(effs) + 
  theme_apa() + 
  scale_y_continuous(breaks = function(l) {
    l[1] <- 0
    l[2] <- ceiling(l[2]/0.1)*0.1
    seq(l[1],l[2],by=ceiling((l[2]-l[1])/5/0.1)*0.1)
  }) +
  coord_flip() +
  geom_col(aes(x=effect, y=EstimateMean, fill=pcat(ifelse(effect=="Normal", NA, p))), width = 0.75, color="black") + 
  scale_fill_brewer(palette = "Dark2", limits = c("(0,.001)","[.001, .01)","[.01, .05)"), labels = c(".001", ".01", ".05")) +
  #  scale_fill_manual(values = c("p < .01" = "gray30", "p < .05" = "gray60", "p > .05" = "white"), na.value = "white", limits=c("p < .01", "p < .05")) +
  geom_errorbar(aes(x=effect, ymin=EstimateMean-qnorm(1-model_plot_alpha/2)*`Std. Error`, ymax=EstimateMean+qnorm(1-model_plot_alpha/2)*`Std. Error`), width=.25, size=.5, color="black") + 
  facet_wrap(~param, scales = "free_x", ncol = 6, labeller = ll, strip.position = "bottom") + 
  labs(x = NULL, y = NULL, fill = expression(p[S] < "")) + 
  theme(legend.position = "bottom", legend.direction = "horizontal", strip.background = element_blank(), strip.placement = "outside",  strip.switch.pad.wrap = unit(-3,"pt"))

p12 <- ggplot(effs %>% subset(effect != "Normal")) + 
  theme_apa() + 
  coord_flip() +
  geom_hline(yintercept = 0) +
  scale_y_continuous(breaks = scales::pretty_breaks(3)) +
  geom_col(aes(x=effect, y=Estimate), fill ="white", width = 0.75, color="black") + 
  geom_errorbar(aes(x=effect, ymin=Estimate-qnorm(1-model_plot_alpha/2)*`Std. Error`, ymax=Estimate+qnorm(1-model_plot_alpha/2)*`Std. Error`), width=.25, size=.5, color="black") + 
  facet_wrap(~param, scales = "free_x", ncol = 6, labeller = ll, strip.position = "bottom") + 
  labs(x = NULL, y = NULL, fill = expression(p[S] < "")) + 
  theme(strip.background = element_blank(), strip.placement = "outside",  strip.switch.pad.wrap = unit(-3,"pt"))

p12a <- ggplot(effs %>% subset(effect != "Normal" & param != "sre_sk2")) + 
  theme_apa() + 
  coord_flip() +
  scale_y_continuous(breaks = pretty_breaks(3)) +
  geom_hline(yintercept = 0) +
  geom_col(aes(x=effect, y=Estimate), fill ="white", width = 0.75, color="black") + 
  geom_errorbar(aes(x=effect, ymin=Estimate-qnorm(1-model_plot_alpha/2)*`Std. Error`, ymax=Estimate+qnorm(1-model_plot_alpha/2)*`Std. Error`), width=.25, size=.5, color="black") + 
  facet_wrap(~param, scales = "free_x", ncol = 4, labeller = ll, strip.position = "bottom") + 
  labs(x = NULL, y = NULL, fill = expression(p[S] < "")) + 
  theme(strip.background = element_blank(), strip.placement = "outside",  strip.switch.pad.wrap = unit(-3,"pt"))


effs_tbl <- effs %>% 
  group_by(effect, param) %>% 
  transmute(txt = ifelse(p < .05, sprintf("%.2f, p_S<%.3f", Estimate, ceiling(pmax(1,p*1000))/1000), sprintf("%.2f", Estimate))) %>%
  spread(param, txt)

write.csv2(effs_tbl, "figs-33/lmer-tbl-model-parameters.csv", row.names = FALSE)

save_fig_wide("figs-33/fig12-model-parameters.pdf", p12, aspect = 2.5)

save_fig("figs-33/pfig12-model-parameters.pdf", p12, width = 5.47, aspect = 1.5)
save_fig("figs-33/pfig12a-model-parameters.pdf", p12a, width = 5.47, aspect = 1)


write.csv(t(empirical_fixstats_agg %>% group_by(cond) %>% select(cond, fp_regout_p, fp_refix_p, fp_skipped_p, fp_gaze_dur_m, fp_first_fix_dur_m, fp_refix_dur_m, fp_single_fix_dur_m) %>% summarize_all(list(m=mean, se=function(x) sd(x)/sqrt(length(x))))))
