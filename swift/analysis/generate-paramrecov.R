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

set.seed(19384)

params <- as.data.frame(lapply(priors, function(prior) {
  do.call(truncnorm::rtruncnorm, c(list(n=75), prior[-1]))
}))

param_template <- read.table("../DATA18/swpar_default.par")
for(j in 1:nrow(params)) {
  matched_params <- match_pars(colnames(params), t(params[j,]))
  new_params <- param_template
  new_params[match(matched_params$x, as.character(new_params[,1])), 2] <- matched_params$y
  write.table(new_params, sprintf("SIM33_recov/swpar_oculo_01_set%d.par", j), col.names = FALSE, row.names = FALSE, quote = FALSE)
  system2("../SWIFT/swiftstat7p_v33", c("-qgxc", "oculo_01", "-s", sprintf("set%d",j), "-i", "../DATA18", "-o", "SIM33_recov", "-I", "1-150", "-P", paste(sprintf("%s=%g", new_params[,1], new_params[,2]), collapse=",")))
}

write.csv(params, "SIM33_recov/swpar_set.csv", row.names = FALSE)


params <- mcmc_chains %>% group_by(vp, session = ifelse(cond == "N", 1, 2)) %>% select(-set, -n, -loglik, -cond) %>% summarise_all(hpd.midpoint, p=.3)
param_template <- read.table("../DATA18/swpar_default.par")
for(j in 1:nrow(params)) {
  matched_params <- match_pars(names(priors), t(params[j,names(priors)]))
  new_params <- param_template
  new_params[match(matched_params$x, as.character(new_params[,1])), 2] <- matched_params$y
  write.table(new_params, sprintf("SIM33_recov/swpar_oculo_01_vp%d_%d.par", params$vp[j], params$session[j]), col.names = FALSE, row.names = FALSE, quote = FALSE)
  system2("../SWIFT/swiftstat7p_v33", c("-qgxc", "oculo_01", "-s", sprintf("vp%d_%d", params$vp[j], params$session[j]), "-i", "../DATA18", "-o", "SIM33_recov", "-I", "1-150", "-P", paste(sprintf("%s=%g", new_params[,1], new_params[,2]), collapse=",")))
}

write.csv(params, "SIM33_recov/swpar_vp.csv", row.names = FALSE)
