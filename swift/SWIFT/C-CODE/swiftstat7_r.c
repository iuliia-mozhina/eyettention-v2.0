

#include <R.h>
#include <Rdefines.h>
#include <stdio.h>

void R_init_swiftr(DllInfo* info) {
  R_registerRoutines(info, NULL, NULL, NULL, NULL);
  R_useDynamicSymbols(info, 1);
}

#define NO_FAILDEFS // this suppresses the error() and warn() declarations in the api.c file and instead uses R's functions
#define error(code, ...) Rf_error( /* ignore the error code */ __VA_ARGS__ )
#define warn(...) Rf_warning( __VA_ARGS__ )

#include "swiftstat7_api.c"


#define SWIFTR_MAJOR 0
#define SWIFTR_MINOR 1


#ifdef SWIFT_MPI

int swiftr_initialized = 0;

SEXP swiftr_initmpi() {
  swift_init_mpi();
  return R_NilValue;
}

SEXP swiftr_finalizempi() {
  swift_finalize_mpi();
  return R_NilValue;
}

SEXP swiftr_loadmpi(SEXP arg1, SEXP arg2, SEXP arg3, SEXP arg4, SEXP arg5, SEXP arg6) {
  if(!swiftr_initialized) {
    Rf_error("Please initialize and finalize SWIFT MPI!");
    UNPROTECT(0);
    return R_NilValue;
  }

  SEXP result;
  const char* inputPath = CHAR(STRING_ELT(AS_CHARACTER(arg1), 0));
  const char* seqId = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  const char* corpusId = CHAR(STRING_ELT(AS_CHARACTER(arg3), 0));
  const char* fixseqId = CHAR(STRING_ELT(AS_CHARACTER(arg4), 0));
  const long seed = INTEGER(AS_INTEGER(arg5))[0];
  const int how_many_evals = INTEGER(AS_INTEGER(arg6))[0];
  swift_load_mpi(inputPath, seqId, corpusId, fixseqId, seed, how_many_evals);
  return R_NilValue;
}

SEXP swiftr_updatempi(SEXP arg2, SEXP arg3) {
  if(!swiftr_initialized) {
    Rf_error("Please initialize and finalize SWIFT MPI!");
    UNPROTECT(0);
    return R_NilValue;
  }
  const char* param_name = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  int param_id = swift_find_param(param_name);
  if(param_id == -1) {
    Rf_error("There is no parameter “%s”.", param_name);
    return R_NilValue;
  }
  if(swift_parameters_meta[param_id].type == PARTYPE_INTEGER) {
    int param_value = INTEGER(AS_INTEGER(arg3))[0];
    swift_update_parameter_mpi(param_id, param_value);
  } else if(swift_parameters_meta[param_id].type == PARTYPE_DOUBLE) {
    double param_value = REAL(AS_NUMERIC(arg3))[0];
    swift_update_parameter_mpi(param_id, param_value);
  } else {
    Rf_error("Can’t update “%s”. Parameters of this type haven’t been implemented in this library yet.", param_name);
  }
  return R_NilValue;
}

SEXP swiftr_evalmpi() {
  if(!swiftr_initialized) {
    Rf_error("Please initialize and finalize SWIFT MPI!");
    UNPROTECT(0);
    return R_NilValue;
  }
  SEXP result;
  PROTECT(result = NEW_NUMERIC(N_LOGLIKS));
  int i;
  for(i=0;i<N_LOGLIKS;i++) {
    REAL(result)[i] = 0.0;
  }
  swift_eval_mpi(REAL(result));
  UNPROTECT(1);
  return result;
}

SEXP swiftr_evalmpilb() {
  if(!swiftr_initialized) {
    Rf_error("Please initialize and finalize SWIFT MPI!");
    UNPROTECT(0);
    return R_NilValue;
  }
  SEXP result;
  PROTECT(result = NEW_NUMERIC(N_LOGLIKS));
  int i;
  for(i=0;i<N_LOGLIKS;i++) {
    REAL(result)[i] = 0.0;
  }
  swift_eval_mpi_lb(REAL(result));
  UNPROTECT(1);
  return result;
}

#endif

void swiftr_gcmodel(SEXP model) {
  free_swift_model((swift_model*) R_ExternalPtrAddr(model));
}

SEXP swiftr_loadmodel(SEXP arg1, SEXP arg2, SEXP arg3, SEXP arg4) {
  SEXP result;
  const char* inputPath = CHAR(STRING_ELT(AS_CHARACTER(arg1), 0));
  const char* seqId = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  const char* corpusId = CHAR(STRING_ELT(AS_CHARACTER(arg3), 0));
  const long seed = INTEGER(AS_INTEGER(arg4))[0];
  swift_model * m;
  if(!swift_load_model(inputPath, seqId, corpusId, seed, &m, 1)) {
    Rf_error("Loading model from path <%s>, seq <%s>, corpus <%s> failed.", inputPath, seqId, corpusId);
    UNPROTECT(0);
    return R_NilValue;
  }else{
    // model_counter++;
    // PROTECT(result = NEW_INTEGER(1));
    // INTEGER(result)[0] = model_counter;
    // UNPROTECT(1);
    // return result;
    result = PROTECT(R_MakeExternalPtr(m, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, swiftr_gcmodel, (Rboolean) 1);
    UNPROTECT(1);
    return result;
  }
}

void swiftr_gcdata(SEXP data) {
  free_swift_dataset((swift_dataset*) R_ExternalPtrAddr(data));
}

SEXP swiftr_loaddata(SEXP arg1) {
  const char *inputPath = CHAR(STRING_ELT(AS_CHARACTER(arg1), 0));
  swift_dataset * d;
  if(!swift_load_data(inputPath, &d, 1)) {
    Rf_error("Loading fixseqs from <%s> failed.", inputPath);
    return R_NilValue;
  }else{
    SEXP result;
    result = PROTECT(R_MakeExternalPtr(d, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, swiftr_gcdata, (Rboolean) 1);
    UNPROTECT(1);
    return result;
  }
}

SEXP swiftr_params() {
  int i,n;
  for(n=0;swift_parameters_meta[n].name!=NULL;n++);
  SEXP ret = PROTECT(allocVector(STRSXP, n));
  for(i=0;i<n;i++) {
    SET_STRING_ELT(ret, i, PROTECT(mkChar(swift_parameters_meta[i].name)));
  }
  UNPROTECT(n+1);
  return ret;
}

SEXP swiftr_update(SEXP arg1, SEXP arg2, SEXP arg3) {
  swift_model * m = (swift_model*) EXTPTR_PTR(arg1);
  char* param_name = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  int param_id = swift_find_param(param_name);
  if(param_id == -1) {
    Rf_error("There is no parameter “%s”.", param_name);
    return R_NilValue;
  }
  if(swift_parameters_meta[param_id].type == PARTYPE_INTEGER) {
    int param_value = INTEGER(AS_INTEGER(arg3))[0];
    if(param_value == NA_INTEGER) {
      hasvalbyid(m->params, param_id, swift_parameter_int) = 0;
    } else {
      setvalbyid(m->params, param_id, swift_parameter_int, param_value);
    }
  } else if(swift_parameters_meta[param_id].type == PARTYPE_DOUBLE) {
    double param_value = REAL(AS_NUMERIC(arg3))[0];
    if(param_value == NA_REAL) {
      hasvalbyid(m->params, param_id, swift_parameter_dbl) = 0;
    } else {
      setvalbyid(m->params, param_id, swift_parameter_dbl, param_value);
    }
  } else {
    Rf_error("Can’t update “%s”. Parameters of this type haven’t been implemented in this library yet.", param_name);
    return R_NilValue;
  }
  SEXP result;
  PROTECT(result = NEW_LOGICAL(1));
  LOGICAL(result)[0] = 1;
  UNPROTECT(1);
  return result;
}


int trial2sexp(swift_trial * trial, SEXP * ret) {
  int i, j;
  SEXP sno = PROTECT(allocVector(INTSXP, trial->nfix));
  SEXP fw = PROTECT(allocVector(INTSXP, trial->nfix));
  SEXP fl = PROTECT(allocVector(REALSXP, trial->nfix));
  SEXP tfix = PROTECT(allocVector(INTSXP, trial->nfix));
  SEXP tsac = PROTECT(allocVector(INTSXP, trial->nfix));
  SEXP first_last = PROTECT(allocVector(INTSXP, trial->nfix));
  SEXP idum[N_FIXSEQ_IDUM], ddum[N_FIXSEQ_DDUM], cdum[N_FIXSEQ_CDUM];
  for(j=0;j<N_FIXSEQ_IDUM;j++) {
    idum[j] = PROTECT(allocVector(INTSXP, trial->nfix));
  }
  for(j=0;j<N_FIXSEQ_DDUM;j++) {
    ddum[j] = PROTECT(allocVector(REALSXP, trial->nfix));
  }
  for(j=0;j<N_FIXSEQ_CDUM;j++) {
    cdum[j] = PROTECT(allocVector(STRSXP, trial->nfix));
  }
  for(i=1;i<=trial->nfix;i++) {
    INTEGER(sno)[i-1] = trial->sentence;
    INTEGER(fw)[i-1] = trial->fixations[i].fw;
    REAL(fl)[i-1] = trial->fixations[i].fl;
    INTEGER(tfix)[i-1] = trial->fixations[i].tfix;
    INTEGER(tsac)[i-1] = trial->fixations[i].tsac;
    if(i > 1 && i < trial->nfix) INTEGER(first_last)[i-1] = 0;
    for(j=0;j<N_FIXSEQ_IDUM;j++) {
      INTEGER(idum[j])[i-1] = trial->fixations[i].idum[j+1];
    }
    for(j=0;j<N_FIXSEQ_DDUM;j++) {
      REAL(idum[j])[i-1] = trial->fixations[i].ddum[j+1];
    }
    for(j=0;j<N_FIXSEQ_CDUM;j++) {
      SET_STRING_ELT(cdum[j], i-1, mkChar(trial->fixations[i].cdum[j+1]));
    }
  }
  INTEGER(first_last)[0] = 1;
  INTEGER(first_last)[trial->nfix-1] = 2;
  *ret = PROTECT(allocVector(VECSXP, 6+N_FIXSEQ_IDUM+N_FIXSEQ_DDUM+N_FIXSEQ_CDUM));
  SET_VECTOR_ELT(*ret, 0, sno);
  SET_VECTOR_ELT(*ret, 1, fw);
  SET_VECTOR_ELT(*ret, 2, fl);
  SET_VECTOR_ELT(*ret, 3, tfix);
  SET_VECTOR_ELT(*ret, 4, tsac);
  SET_VECTOR_ELT(*ret, 5, first_last);
  for(j=0;j<N_FIXSEQ_IDUM;j++) {
    SET_VECTOR_ELT(*ret, 6+j, idum[j]);
  }
  for(j=0;j<N_FIXSEQ_DDUM;j++) {
    SET_VECTOR_ELT(*ret, 6+N_FIXSEQ_IDUM+j, ddum[j]);
  }
  for(j=0;j<N_FIXSEQ_CDUM;j++) {
    SET_VECTOR_ELT(*ret, 6+N_FIXSEQ_IDUM+N_FIXSEQ_DDUM+j, cdum[j]);
  }
  return 7+N_FIXSEQ_IDUM+N_FIXSEQ_DDUM+N_FIXSEQ_CDUM;
}

int dataset2sexp(swift_dataset * seqs, SEXP * l) {
  *l = PROTECT(allocVector(VECSXP, seqs->n));
  int i, n;
  n = 1;
  for(i=1;i<=seqs->n;i++) {
    SEXP trial;
    n += trial2sexp(&seqs->trials[i], &trial);
    SET_VECTOR_ELT(*l, i-1, trial);
  }
  return n;
}

SEXP swiftr_data2df(SEXP arg1) {
  swift_dataset * d = (swift_dataset*) EXTPTR_PTR(arg1);
  SEXP ret;
  int n;
  n = dataset2sexp(d, &ret);
  UNPROTECT(n);
  return ret;
}

SEXP swiftr_getparam(SEXP arg1, SEXP arg2) {
  swift_model * m = (swift_model*) EXTPTR_PTR(arg1);
  const char* param_name = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  int param_id = swift_find_param(param_name);
  if(param_id == -1) {
    Rf_error("There is no parameter “%s”.", param_name);
    return R_NilValue;
  }
  SEXP result;
  if(swift_parameters_meta[param_id].type == PARTYPE_INTEGER) {
    PROTECT(result = NEW_INTEGER(1));
    INTEGER(result)[0] = hasvalbyid(m->params, param_id, swift_parameter_int) ? valbyid(m->params, param_id, swift_parameter_int) : NA_INTEGER;
    UNPROTECT(1);
  } else if(swift_parameters_meta[param_id].type == PARTYPE_DOUBLE) {
    PROTECT(result = NEW_NUMERIC(1));
    REAL(result)[0] = hasvalbyid(m->params, param_id, swift_parameter_dbl) ? valbyid(m->params, param_id, swift_parameter_dbl) : NA_REAL;
    UNPROTECT(1);
  } else {
    Rf_error("Can’t get “%s”. Parameters of this type haven’t been implemented in this library yet.", param_name);
    result = R_NilValue;
  }
  return result;
}

SEXP swiftr_version() {
  SEXP result;
  PROTECT(result = NEW_LIST(4));
  SEXP v_swift, v_api, v_swiftr, v_str;
  char variant[1024];
  swift_variant_string(variant);
  PROTECT(v_swiftr = NEW_INTEGER(2));
  INTEGER(v_swiftr)[0] = SWIFTR_MAJOR;
  INTEGER(v_swiftr)[1] = SWIFTR_MINOR;
  PROTECT(v_swift = NEW_INTEGER(2));
  INTEGER(v_swift)[0] = SWIFT_VERSION_MAJOR;
  INTEGER(v_swift)[1] = SWIFT_VERSION_MINOR;
  PROTECT(v_api = NEW_INTEGER(2));
  INTEGER(v_api)[0] = SWIFT_API_VERSION_MAJOR;
  INTEGER(v_api)[1] = SWIFT_API_VERSION_MINOR;
  PROTECT(v_str = allocVector(STRSXP, 1));
  SET_STRING_ELT(v_str, 0, mkChar(variant));
  SET_ELEMENT(result, 0, v_swiftr);
  SET_ELEMENT(result, 1, v_api);
  SET_ELEMENT(result, 2, v_swift);
  SET_ELEMENT(result, 3, v_str);
  UNPROTECT(5);
  return result;
}

SEXP swiftr_eval(SEXP arg1, SEXP arg2, SEXP arg3) {
  swift_model * m = (swift_model*) EXTPTR_PTR(arg1);
  swift_dataset * d = (swift_dataset*) EXTPTR_PTR(arg2);

  const int threads = INTEGER(AS_INTEGER(arg3))[0];

#if !defined(_OPENMP)
  if(threads > 1) {
    Rf_warning("You’re trying to use multithreading but this module isn’t supporting it. Please recompile with OpenMP support or set the `threads` parameter to 0 or 1.");
  }
#endif

  SEXP result;
  PROTECT(result = NEW_NUMERIC(N_LOGLIKS));
  int i;
  for(i=0;i<N_LOGLIKS;i++) {
    REAL(result)[i] = 0.0;
  }
  swift_eval_all(m, d, REAL(result), threads, 0);
  UNPROTECT(1);
  return result;
}



SEXP swiftr_generatefile(SEXP arg1, SEXP arg2, SEXP arg3, SEXP arg4) {
  swift_model * m = (swift_model*) EXTPTR_PTR(arg1);
  const char* seq_name = CHAR(STRING_ELT(AS_CHARACTER(arg2), 0));
  const char* dir_name = CHAR(STRING_ELT(AS_CHARACTER(arg3), 0));

  const int threads = INTEGER(AS_INTEGER(arg4))[0];

#if !defined(_OPENMP)
  if(threads > 1) {
    Rf_warning("You’re trying to use multithreading but this module isn’t supporting it. Please recompile with OpenMP support or set the `threads` parameter to 0 or 1.");
  }
#endif

  swift_generate(m, dir_name, seq_name, NULL, threads, 1, 0);

  return R_NilValue;
}



SEXP swiftr_generate(SEXP arg1, SEXP arg4) {
  swift_model * m = (swift_model*) EXTPTR_PTR(arg1);
  
  int ntrials = nsentences(m->corpus)*val(m->params, runs);

  swift_dataset * data = malloc(sizeof(swift_dataset));
  data->n = ntrials;
  data->trials = vector(swift_trial, ntrials);

  const int threads = INTEGER(AS_INTEGER(arg4))[0];

#if !defined(_OPENMP)
  if(threads > 1) {
    Rf_warning("You’re trying to use multithreading but this module isn’t supporting it. Please recompile with OpenMP support or set the `threads` parameter to 0 or 1.");
  }
#endif
  
  gaengine(m->params, m->corpus, data, &m->seed, NULL, 0, threads, NULL, 0, NULL);
  
  SEXP ret = PROTECT(R_MakeExternalPtr(data, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, swiftr_gcdata, (Rboolean) 1);
  UNPROTECT(1);

  return ret;
}

