
true_params <- read.csv("SIM33_recov/swpar_set.csv") %>% mutate(set = 1:n(), file = sprintf("set%d", set))

files_exist <- which(file.exists(sprintf("../fitting/oc33_recov_oculo_01_%s_1.dat", true_params$file)))

chains <- bind_rows(lapply(files_exist, function(i) {
  bind_rows(lapply(1:5, function(j) {
    fn <- sprintf("../fitting/oc33_recov_oculo_01_%s_%d.dat", true_params$file[i], j)
    message(sprintf("Read %s...", fn))
    read.table(fn, header = TRUE) %>% mutate(set = i, chain = j, n = seq_len(n()))
  }))
}))



mcmc_chains_summary <- expand.grid(set = files_exist, param = names(priors)) %>% mutate(y_min = NA_real_, y_max = NA_real_, true_value = NA_real_)

chains_narrow <- chains %>% subset(n > 5000) %>% select(-chain, -loglik) %>% gather(param, value, -set, -n) %>% group_by(set, param) %>% mutate(true_value = unlist(true_params[first(set), as.character(first(param))]))

for(i in files_exist) {
  mcmc_mat <- as.mcmc(chains %>% subset(n > 5000 & set == i) %>% select(-n, -chain, -loglik, -set))
  hpdis <- HPDinterval(mcmc_mat, .6)
  
  is <- match(paste(i, rownames(hpdis)), paste(mcmc_chains_summary$set, mcmc_chains_summary$param))
  
  mcmc_chains_summary$y_min[is] <- hpdis[,"lower"]
  mcmc_chains_summary$y_max[is] <- hpdis[,"upper"]
  
  
  mcmc_chains_summary$true_value[is] <- unlist(true_params[i, rownames(hpdis)])
}

mcmc_chains_summary <- mcmc_chains_summary %>% mutate(true_value_in_hpdi = true_value >= y_min & true_value <= y_max)

mcmc_chains_summary %>% group_by(param) %>% summarize(p = mean(true_value_in_hpdi))

p15 <- ggplot(mcmc_chains_summary) +
  theme_apa() + theme(strip.placement = "outside", strip.switch.pad.wrap = unit(-3,"pt")) +
  facet_wrap(~param, scales = "free", ncol=5, labeller = ll, strip.position = "bottom") +
  geom_abline(slope = 1) +
  labs(x = "True value", y = "Recovered value") +
  geom_linerange(aes(x=true_value, ymin = y_min, ymax = y_max), color="gray") +
  geom_point(aes(x=true_value, y = (y_max+y_min)/2), size=.5)

save_fig_wide("figs-33/p15-parameter-recovery.pdf", p15, aspect=1.5)  
  


