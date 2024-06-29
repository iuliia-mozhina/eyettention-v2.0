


swift.dir <- "../SWIFT"
data.dir <- "../DATA18"
corpus.name <- "oculo_01"
parvals <- list(
  beta = rep(seq(0,1,length.out=100), 1),
  delta0 = rep(seq(5,13,length.out=100), 1),
  eta = rep(seq(0,1,length.out=100), 1),
  misfac = rep(seq(0,2,length.out=100), 1),
  refix = rep(seq(0,2,length.out=100), 1),
  msac = rep(seq(1,4,length.out=100), 1)
)
sets <- c("test")


library(ggplot2)
library(dplyr)
library(tidyr)
dyn.load(swiftdyn_location <- file.path(swift.dir, "SIM", "r", "swiftstat7_r.so"))

load.model <- function (corpus, seqid, dir = "../DATA", seed = runif(1, 0, 2**31)) .Call("swiftr_loadmodel", dir, file.path(dir, sprintf("swpar_%s_%s.par", corpus, seqid)), file.path(dir, sprintf("corpus_%s.dat", corpus)), seed)
load.data <- function (seqid, dir = "../DATA") .Call("swiftr_loaddata", file.path(dir, sprintf("fixseqin_%s.dat", seqid)))
update.parameter <- function(param_name, param_value, model=1L) {
  stopifnot(length(param_name)==length(param_value))
  for(i in seq_along(param_name)) {
    if(is.null(.Call("swiftr_update", model, param_name[i], param_value[i]))) {
      stop(sprintf("Setting %s to %f failed.", param_name[i], param_value[i]))
    }
  }
}

get.parameter <- function(param_name, model=1L) {
  ret <- .Call("swiftr_getparam", model, param_name);
  if(is.null(ret)) NA else ret
}
loglik <- function(model=1L, data=1L, threads=0L) .Call("swiftr_eval", model, data, threads)
swift.version <- function(model=1L, data=1L, threads=0L) {
  ret <- .Call("swiftr_version")
  names(ret) <- c("package", "API", "SWIFT")
  return(ret)
}
free.model <- function(model=1L) invisible(.Call("swiftr_freemodel", model))
free.data <- function(data=1L) invisible(.Call("swiftr_freedata", data))





logliks <- do.call(rbind, lapply(sets, function(s) {
  data <- load.data(s, dir = data.dir)
  model <- load.model(corpus.name, s, dir = data.dir)
  true.vals <- vapply(names(parvals), get.parameter, double(1), USE.NAMES = T, model = model)
  logliks <- do.call(rbind, lapply(names(parvals), function(parname) {
    message(sprintf("Compute likelihood profile for dataset %s and parameter %s...", s, parname))
    model <- load.model(corpus.name, s, dir = data.dir)
    pb <- txtProgressBar(max = length(parvals[[parname]]))
    logliks <- as.data.frame(t(vapply(seq_along(parvals[[parname]]), function(i) {
      val <- parvals[[parname]][i]
      update.parameter(param_name = parname, param_value = val, model = model)
      updated_val <- get.parameter(param_name = parname, model = model)
      if(updated_val != val) warning(sprintf("Tried to set %s in model %d to %f but retrieved %f!", parname, model, val, updated_val))
      ll <- loglik(model, data, 0L)
      setTxtProgressBar(pb, i)
      ll
    }, double(3))))
    close(pb)
    colnames(logliks) <- c("loglik_comb","loglik_temp","loglik_spat")
    logliks$parname <- parname
    logliks$trueval <- true.vals[parname]
    logliks$val <- parvals[[parname]]
    logliks
  }))
  logliks$set <- s
  #free.model(model)
  #free.data(data)
  logliks
}))

logliks_narrow <- logliks %>% gather(loglik_type, loglik, loglik_comb, loglik_temp, loglik_spat)



true.vals <- expand.grid(set = sets, parname = names(parvals))
true.vals$val <- logliks$trueval[match(paste(true.vals$set, true.vals$parname), paste(logliks$set, logliks$parname))]


mean_logliks <- logliks_narrow %>% group_by(loglik_type, parname, set) %>% summarize(m = mean(loglik))

p3 <- ggplot(logliks_narrow %>% mutate(loglik = loglik - mean_logliks$m[match(paste(loglik_type, parname, set), paste(mean_logliks$loglik_type, mean_logliks$parname, mean_logliks$set))], loglik_type = factor(loglik_type, levels = c("loglik_comb","loglik_spat","loglik_temp"), c("combined","spatial","temporal")))) + 
  theme_apa() + theme(legend.position = "bottom") +
  facet_wrap(~ parname, scales = "free", strip.position = "bottom", labeller = ll) + 
  labs(x=NULL, y="Centered log-likelihood", linetype=NULL, color=NULL) +
  geom_point(aes(y = loglik, x = val, color=loglik_type), alpha = .1) + 
  geom_smooth(aes(y = loglik, x = val, color=loglik_type, linetype=loglik_type), method="gam", size = .5, se = F) +
  geom_vline(aes(xintercept = val), color="red", data=true.vals) +
  scale_color_manual(values = c(combined = "black", spatial = "red", temporal = "blue")) +
  theme(strip.background = element_blank(), strip.placement = "outside",  strip.switch.pad.wrap = unit(-3,"pt"))

save_fig_wide("figs-32/fig3-likelihood-profiles.pdf", p3, aspect = 1.5)
save_fig_narrow("figs-32/fig3n-likelihood-profiles.pdf", p3, aspect = 1.5)


dyn.unload(swiftdyn_location)
