dyn.load("swiftstat7_r.so")

load.model <- function (corpus, seqid, dir = "../DATA", seed = runif(1, 0, 2**31)) .Call("swiftr_loadmodel", dir, file.path(dir, "swpar_default.par"), file.path(dir, sprintf("corpus_%s.dat", corpus)), seed)

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
gensp <- function(model=1L, seq_name=1L, dir_name=0L, threads=0L) .Call("swiftr_generate", model, data, dir, threads)

swift.version <- function(model=1L, data=1L, threads=0L) {
	ret <- .Call("swiftr_version")
	names(ret) <- c("package", "API", "SWIFT")
	return(ret)
}

swift.version()

model <- load.model("oculo_01", "vp1_1_train", "../../DATA18", seed=1234)

data <- load.data("vp1_1_train", "../../DATA18")

update.parameter(c("msac", "delta0"), c(1.2,8.77), model = model)
print('hi');

get.parameter("delta0", model)
print('hi');

loglik(model, data) # by default all available cores, add threads=N to set number of threads
gensp(model, data)
print('hi');
