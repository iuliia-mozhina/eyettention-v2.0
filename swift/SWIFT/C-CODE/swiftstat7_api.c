
#ifndef __SWIFT_API_LOADED__
#define __SWIFT_API_LOADED__


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>


#define stop(code, ...) {error(code, __VA_ARGS__); exit(code);}

#ifndef NO_FAILDEFS

void error(int code, char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    #pragma omp critical
    {
        if(code != 1)
            fprintf(stderr, "\e[0m\e[1m\e[31mError %d:\e[0m ", code);
        else
            fprintf(stderr, "\e[0m\e[1m\e[31mError:\e[0m ");
        vfprintf(stderr, fmt, argptr);
        fputc('\n', stderr);
    }
    fflush(stderr);
    va_end(argptr);
}

void warn(char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    #pragma omp critical
    {
        fprintf(stderr, "\e[0m\e[1m\e[31mWarning:\e[0m ");
        vfprintf(stderr, fmt, argptr);
        fputc('\n', stderr);
    }
    fflush(stderr);
    va_end(argptr);
}

#endif



#define SWIFT_API_VERSION_MAJOR 2
#define SWIFT_API_VERSION_MINOR 1


#include "array.c"
#if defined(RANGEN_POSIX)
#include "rangen_posix.c"
#elif defined(RANGEN_MT)
#include "rangen_mt.c"
#else
#warning You have not specified a random number generator. Using RANGEN_MT by default.
#include "rangen_mt.c"
#endif


// Some compilers enable OpenMP by default, but some require a flag such as -fopenmp to load the libraries
// To enable threading, make sure the compiler supports OpenMP (i.e., -fopenmp) and is called without -D DISABLE_THREADS
// To disable threading, even if OpenMP is loaded, add -D DISABLE_THREADS to the compiler call
// gcc ... -fopenmp ... : threading is enabled if the compiler supports OpenMP (will not compile if flag is not recognized)
// gcc ... -fopenmp -D DISABLE_THREADS ... : threading is disabled
// gcc ... -D DISABLE_THREADS ... : threading is disabled
// Note: Multithreading is only relevant for performance issues. Results are guaranteed to be identical, no matter how many cores are used.
#if !defined(DISABLE_THREADS)
// We have not explicitly disabled threading
#if defined(_OPENMP)
// OpenMP is supported -> include header file!
#include <omp.h>
#else
// OpenMP is not supported -> define DISABLE_THREADS to let the rest of the code know we're not multithreading!
#define DISABLE_THREADS
#warning You have not explicitly disabled multithreading but no OpenMP support was detected. Therefore, this will now compile without multithreading support. If you want to use multithreading, please make sure to set the approporiate compiler flag (such as -fopenmp). If you want to avoid multithreading, please explicitly set DISABLE_THREADS (e.g., -D DISABLE_THREADS) to confirm you want to disable multithreading and suppress this warning.
#endif
#endif


typedef struct {
    char *style;
    char *plain;
    char *markdown;
} swift_citation;

static swift_citation swift_citations[] = {
    {
        "AMA",
        "Engbert R, Nuthmann A, Richter E, Kliegl R. SWIFT: A Dynamical Model of Saccade Generation During Reading. \e[3mPsychological Review\e[23m [serial online]. October 2005;112(4):777-813.",
        "Engbert R, Nuthmann A, Richter E, Kliegl R. SWIFT: A Dynamical Model of Saccade Generation During Reading. *Psychological Review* [serial online]. October 2005;112(4):777-813."
    },
    {
        "APA",
        "Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: A Dynamical Model of Saccade Generation During Reading. \e[3mPsychological Review, 112\e[23m, 777-813. doi:10.1037/0033-295X.112.4.777",
        "Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: A Dynamical Model of Saccade Generation During Reading. *Psychological Review, 112*, 777-813. doi:10.1037/0033-295X.112.4.777"
    },
    {
        "Chicago",
        "Engbert, Ralf, Antje Nuthmann, Eike M. Richter, and Reinhold Kliegl. 2005. “SWIFT: A Dynamical Model of Saccade Generation During Reading.” \e[3mPsychological Review\e[23m 112, no. 4: 777-813.",
        "Engbert, Ralf, Antje Nuthmann, Eike M. Richter, and Reinhold Kliegl. 2005. “SWIFT: A Dynamical Model of Saccade Generation During Reading.” *Psychological Review* 112, no. 4: 777-813."
    },
    {
        "Harvard",
        "Engbert, R, Nuthmann, A, Richter, E, & Kliegl, R 2005, ‘SWIFT: A Dynamical Model of Saccade Generation During Reading’, \e[3mPsychological Review\e[23m, 112, 4, pp. 777-813",
        "Engbert, R, Nuthmann, A, Richter, E, & Kliegl, R 2005, ‘SWIFT: A Dynamical Model of Saccade Generation During Reading’, *Psychological Review*, 112, 4, pp. 777-813"
    },
    {
        "MLA",
        "Engbert, Ralf, et al. “SWIFT: A Dynamical Model of Saccade Generation during Reading.” \e[3mPsychological Review\e[23m, vol. 112, no. 4, Oct. 2005, pp. 777-813. \e[3mAPA PsycNET\e[23m, doi:10.1037/0033-295X.112.4.777.",
        "Engbert, Ralf, et al. “SWIFT: A Dynamical Model of Saccade Generation during Reading.” *Psychological Review*, vol. 112, no. 4, Oct. 2005, pp. 777-813. *APA PsycNET*, doi:10.1037/0033-295X.112.4.777."
    },
    {
        "BibTex",
        "@article{Engbert2005,\n  Author = {Engbert, Ralf and Nuthmann, Antje and Richter, Eike M. and Kliegl, Reinhold},\n  ISSN = {0033-295X},\n  Journal = {Psychological Review},\n  Volume = {112},\n  Number = {4},\n  Pages = {777-813},\n  Title = {SWIFT: A Dynamical Model of Saccade Generation During Reading.},\n  Year = {2005},\n  DOI = {10.1037/0033-295X.112.4.777}\n}",
        0
    },
    {NULL}
};

swift_citation *swift_find_citation_style(char *style_name) {
    size_t i;
    for(i=0;swift_citations[i].style!=NULL;i++)
        if(!strcmp(swift_citations[i].style, style_name))
            return &swift_citations[i];
    return NULL;
}

typedef struct {
    unsigned int major, minor;
} version;

// TYPE DEFINITIONS

typedef enum {
    PARTYPE_INTEGER=1, PARTYPE_DOUBLE=2
} swift_parameter_type;

typedef struct {
    unsigned char hasval;
    int val;
} swift_parameter_int;

typedef struct {
    unsigned char hasval;
    double val;
} swift_parameter_dbl;

typedef struct {
    char *name;
    swift_parameter_type type;
    int id;
} swift_parameter_meta;

typedef struct {
    int nl;
    double freq;
    double pred;
    int *idum;
    double *ddum;
    char **cdum;
} swift_word;

typedef struct {
    int nw;
    swift_word *words;
} swift_sentence;

typedef struct {
    int ns;
    swift_sentence *sentences;
    char *name;
} swift_corpus;

typedef struct {
    int fw;
    double fl;
    int tfix;
    int tsac;
    int *idum;
    double *ddum;
    char **cdum;
} swift_fixation;

typedef struct {
    int nfix;
    swift_fixation *fixations;
    int sentence;
} swift_trial;

typedef struct {
    int n;
    swift_trial* trials;
    char *name;
} swift_dataset;


// Macros for retrieving and updating parameters
#define valbyid(pars, parid, type) (((type*) swift_param_addr(pars, parid))->val)
#define hasvalbyid(pars, parid, type) (((type*) swift_param_addr(pars, parid))->hasval)
#define setvalbyid(pars, parid, type, val) {valbyid(pars, parid, type) = val; hasvalbyid(pars, parid, type) = 1;}
#define val(pars, par) (pars->par.val)
#define hasval(pars, par) (pars->par.hasval)
#define setval(pars, par, value) {val(pars, par) = value; hasval(pars, par) = 1;}



// define parameters from generated parameter definition (tmp_pardef.c) file
// Note: This file is generated by the generate_pardef.py script using the swpar.cfg parameter configuration file
// You may also create your own tmp_pardef.c file but it is strongly recommended that you use the aforementioned
// Python script (or just use the convenient build.sh) in order to create this file automatically.
// If of interest, tmp_pardef.c remains as a temporary file in ./SIM/tmp after building (i.e., executing build.sh).
#include "tmp_pardef.c"

// PARAMETER STORAGE

swift_parameters* alloc_parameters(swift_parameters* default_params) {
    swift_parameters* ret = malloc(sizeof(swift_parameters));
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        // iterate through all parameters and initialize params to 0 or copy default
        swift_parameter_id id = swift_parameters_meta[i].id;
        switch(swift_parameters_meta[i].type) {
            case PARTYPE_INTEGER:
                if(default_params != NULL && hasvalbyid(default_params, id, swift_parameter_int)) {
                    setvalbyid(ret, id, swift_parameter_int, valbyid(default_params, id, swift_parameter_int));
                } else {
                    valbyid(ret, id, swift_parameter_int) = 0;
                    hasvalbyid(ret, id, swift_parameter_int) = 0;
                }
                break;
            case PARTYPE_DOUBLE:
                if(default_params != NULL && hasvalbyid(default_params, id, swift_parameter_dbl)) {
                    setvalbyid(ret, id, swift_parameter_dbl, valbyid(default_params, id, swift_parameter_dbl));
                } else {
                    valbyid(ret, id, swift_parameter_dbl) = 0.0;
                    hasvalbyid(ret, id, swift_parameter_dbl) = 0;
                }
                break;
            default:
                error(1, "Unknown parameter type!");
                return NULL;
        }
    }
    return ret;
}

int swift_find_param(char* name) {
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        if(!strcmp(swift_parameters_meta[i].name, name)) {
            return i;
        }
    }
    return -1;
}

void free_parameters(swift_parameters* obj) {
    free(obj);
}

int scan_parameter(char *f, swift_parameters *pars, swift_parameter_id id) {
    swift_parameter_meta par = swift_parameters_meta[id];
    char dummy;
    if(par.type == PARTYPE_INTEGER) {
        int tmp;
        if(sscanf(f, "%d%c", &tmp, &dummy) != 1) {
            error(1, "Cannot read integer value for “%s” from parfile!", par.name);
            return 0;
        }
        setvalbyid(pars, id, swift_parameter_int, tmp);
    } else if(par.type == PARTYPE_DOUBLE) {
        double tmp;
        if(sscanf(f, "%lf%c", &tmp, &dummy) != 1) {
            error(1, "Cannot read double value for “%s” from parfile!", par.name);
            return 0;
        }
        setvalbyid(pars, id, swift_parameter_dbl, tmp);
    } else {
        warn("Parameter “%s” is known but there is no rule for reading. It is therefore skipped.", par.name);
    }
    return 1;
}

int fscan_parameter(FILE *f, swift_parameters *pars, swift_parameter_id id) {
    char tmp[200];
    if(fscanf(f, "%s", tmp) == 1) {
        return scan_parameter(tmp, pars, id);
    }else {
        swift_parameter_meta par = swift_parameters_meta[id];
        warn("Could not fetch value for “%s” from parfile.", par.name);
        return 0;
    }
}

int fscan_parameters(FILE *f, swift_parameters *pars) {
    char parname[20];
    while(fscanf(f, "%s", parname) == 1) {
        int parid = swift_find_param(parname);
        if(parid < 0) {
            error(1, "Parameter “%s” loaded from parfile is unknown!", parname);
            return 0;
        }
        if(!fscan_parameter(f, pars, parid)) {
            error(1, "Reading parameter “%s” failed.", parname);
            return 0;
        }
    }
    return 1;
}

void write_parameters(FILE *f, swift_parameters *pars) {
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        // iterate through all parameters
        // params without a value or unknown type are skipped and not written into the file!
        swift_parameter_meta par = swift_parameters_meta[i];
        if(par.type == PARTYPE_INTEGER && hasvalbyid(pars, par.id, swift_parameter_int)) {
            fprintf(f, "%s\t%d\n", par.name, valbyid(pars, par.id, swift_parameter_int));
        }else if(par.type == PARTYPE_DOUBLE && hasvalbyid(pars, par.id, swift_parameter_dbl)) {
            fprintf(f, "%s\t%lf\n", par.name, valbyid(pars, par.id, swift_parameter_dbl));
        }
    }
}

void print_all_parameters(FILE *f, swift_parameters *pars) {
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        // iterate through all parameters
        // all parameters are printed, including those without a value or unknown type (printed as NA)
        swift_parameter_meta par = swift_parameters_meta[i];
        fprintf(f, "%-10s = ", par.name);
        if(par.type == PARTYPE_INTEGER && hasvalbyid(pars, par.id, swift_parameter_int)) {
            fprintf(f, "%d", valbyid(pars, par.id, swift_parameter_int));
        }else if(par.type == PARTYPE_DOUBLE && hasvalbyid(pars, par.id, swift_parameter_dbl)) {
            fprintf(f, "%g", valbyid(pars, par.id, swift_parameter_dbl));
        }else {
            fprintf(f, "NA");
        }
        fputc('\n', f);
    }
}

void print_parameters(FILE *f, swift_parameters *pars) {
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        // iterate through all parameters
        // all parameters are printed, including those without a value or unknown type (printed as NA)
        swift_parameter_meta par = swift_parameters_meta[i];
        if(par.type == PARTYPE_INTEGER && hasvalbyid(pars, par.id, swift_parameter_int)) {
            fprintf(f, "%-10s = %d\n", par.name, valbyid(pars, par.id, swift_parameter_int));
        }else if(par.type == PARTYPE_DOUBLE && hasvalbyid(pars, par.id, swift_parameter_dbl)) {
            fprintf(f, "%-10s = %g\n", par.name, valbyid(pars, par.id, swift_parameter_dbl));
        }
    }
}

unsigned int is_param_set(swift_parameters *pars, swift_parameter_meta par) {
    if(par.type == PARTYPE_INTEGER) {
        return hasvalbyid(pars, par.id, swift_parameter_int);
    }else if(par.type == PARTYPE_DOUBLE) {
        return hasvalbyid(pars, par.id, swift_parameter_dbl);
    }
    return 0;
}

unsigned int is_param_set_by_id(swift_parameters *pars, swift_parameter_id id) {
    return is_param_set(pars, swift_parameters_meta[id]);
}

int require_parameter(swift_parameters *pars, swift_parameter_id id) {
    swift_parameter_meta par = swift_parameters_meta[id];
    if(!is_param_set(pars, par)){
        error(2, "Parameter “%s” is required but not set!", par.name);
        return 0;
    }
    return 1;
}

int require_all_parameters(swift_parameters *pars) {
    int i;
    for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
        if(!require_parameter(pars, i)) return 0;
    }
    return 1;
}

int require_parameters(swift_parameters *pars, int n, swift_parameter_id par_ids[]) {
    int i;
    for(i=0;i<n;i++) {
        if(!require_parameter(pars, par_ids[i])) return 0;
    }
    return 1;
}




// DATA TYPES

// Macros for counting words in given sentence, sentences in given corpus, laoded sequences, and fixations in given sequence
// e.g., the number of words in sentence #10: nwords(corpus, 10)
#define nwords(corpus, sentence) ((int) (corpus)->sentences[sentence].nw)
#define nsentences(corpus) ((int) (corpus)->ns)
#define ntrials(seqs) ((int) (seqs)->n)
#define nfixations(seqs, seq) ((int) (seqs)->trials[seq].nfix)

// Macros for retrieving single properties (fields) of words, sentences, and fixations
// e.g., to get the frequency of word #2 in sentence #10: word_prop(corpus, 10, 2, freq)
// Note that these can also be used for setting those properties, such as: word_prop(corpus, 10, 2, freq) = 1000.0;
#define word_prop(corpus, sentence, word, prop) ((corpus)->sentences[sentence].words[word].prop)
#define sentence_prop(corpus, sentence, prop) ((corpus)->sentences[sentence].prop)
#define fixation_prop(seqs, seq, fix, prop) ((seqs)->trials[seq].fixations[fix].prop)
#define trial_prop(seqs, seq, prop) ((seqs)->trials[seq].prop)

// Macros for filling vectors with a given property of all words in a given sentence, sentences in a given corpus, and fixations in a given sequence
// Note that the vector should have been properly initialized, i.e. starting index at 1 and length matches number of words/sentences/fixations!
// e.g., to retrieve the frequency of all words in sentence #10:
//    double freqs = dvector(1, nwords(corpus, 10));
//    word_vec(corpus, 10, freq, freqs);
//  -> frequency of word #2 in sentence #10 is then freqs[2]
//  -> don't forget to free the vector! -> free_dvector(freqs, 1, nwords(corpus, 10));
#define word_vec(corpus, sentence, prop, target) {int __i; for(__i=1;__i<=nwords(corpus, sentence);__i++) target[__i]=word_prop(corpus, sentence, __i, prop);}
#define sentence_vec(corpus, prop, target) {int __i; for(__i=1;__i<=nsentences(corpus);__i++) target[__i]=sentence_prop(corpus, __i, prop);}
#define fixation_vec(seqs, seq, prop, target) {int __i; for(__i=1;__i<=nfixations(seqs, seq);__i++) {target[__i]=fixation_prop(seqs, seq, __i, prop);}}
#define trial_vec(seqs, prop, target) {int __i; for(__i=1;__i<=ntrials(seqs);__i++) target[__i]=trial_prop(seqs, __i, prop);}

#define new_sentence(n) {n, vector(swift_word, n)}
#define new_corpus(n) {n, vector(swift_sentence, n)}
#define new_trial(n) {n, vector(swift_fixation, n)}
#define new_dataset(n) {n, vector(swift_trial, n)}



struct swift_files {
    FILE *fsim, *fseq, *f1, *f2, *f3;
};


// GENERIC ALL-PURPOSE FUNCTION

static void write_trial(FILE *f, swift_trial seq);
static void clear_trial(swift_trial obj);

#include "gaengine9a.c"


const version swift_version = {SWIFT_VERSION_MAJOR, SWIFT_VERSION_MINOR};
const version swift_api_version = {SWIFT_API_VERSION_MAJOR, SWIFT_API_VERSION_MINOR};


int swift_version_string(char * str) {
    return sprintf(str, "SWIFT %d.%d", swift_version.major, swift_version.minor);
}

int swift_variant_string(char * str) {
    // base invparab d-dep mt-rand saccgauss
    return sprintf(str, "%s lexrate:%s selectar:%s rangen:%s execsacc:%s", SWIFT_VARIANT, SWIFT_VARIANT_LEXRATE, SWIFT_VARIANT_SELECTAR, SWIFT_VARIANT_RANGEN, SWIFT_VARIANT_EXECSACC);
}

int swift_api_version_string(char * str) {
    return sprintf(str, "%d.%d", swift_api_version.major, swift_api_version.minor);
}

int swift_complete_version_string(char * str, char * custom) {
    int ptr = 0, rval = 0;
    rval = swift_version_string(&str[ptr]);
    if(rval < 0) return rval;
    rval = sprintf(&str[ptr += rval], " (");
    if(rval < 0) return rval;
    rval = swift_variant_string(&str[ptr += rval]);
    if(rval < 0) return rval;
    rval = sprintf(&str[ptr += rval], ", API ");
    if(rval < 0) return rval;
    rval = swift_api_version_string(&str[ptr += rval]);
    if(rval < 0) return rval;
    if(custom != NULL) {
        rval = sprintf(&str[ptr += rval], ", %s", custom);
        if(rval < 0) return rval;
    }
    #ifndef DISABLE_THREADS
    rval = sprintf(&str[ptr += rval], ", OpenMP threads)");
    #else
    rval = sprintf(&str[ptr += rval], ", no threading)");
    #endif
    if(rval < 0) return rval;
    else return ptr + rval;
}

int load_word(FILE *f, swift_word *w) {
    swift_word new_w;
    if(fscanf(f, "%d", &new_w.nl) != 1) {
        error(1, "Loading word failed: word length invalid.");
        return 0;
    }
    if(fscanf(f, "%lf", &new_w.freq) != 1) {
        error(1, "Loading word failed: word frequency invalid.");
        return 0;
    }
    if(fscanf(f, "%lf", &new_w.pred) != 1) {
        error(1, "Loading word failed: word predictability invalid.");
        return 0;
    }
    int col;
    new_w.idum = vector(int, N_CORPUS_IDUM);
    for(col=1;col<=N_CORPUS_IDUM;col++) {
        if(fscanf(f, "%d", &new_w.idum[col]) != 1) {
            error(1, "Loading word failed: idum[%d] could not be read!", col);
            return 0;
        }
    }
    new_w.ddum = vector(double, N_CORPUS_DDUM);
    for(col=1;col<=N_CORPUS_DDUM;col++) {
        if(fscanf(f, "%lf", &new_w.ddum[col]) != 1) {
            error(1, "Loading word failed: ddum[%d] could not be read!", col);
            return 0;
        }
    }
    new_w.cdum = vector(char*, N_CORPUS_CDUM);
    for(col=1;col<=N_CORPUS_CDUM;col++) {
        new_w.cdum[col] = malloc(sizeof(char) * 64);
        if(fscanf(f, "%s", new_w.cdum[col]) != 1) {
            error(1, "Loading word failed: cdum[%d] could not be read!", col);
            return 0;
        }
    }
    *w = new_w;
    return 1;
}

void write_word(FILE *f, swift_word w) {
    fprintf(f, "\t%d\t%lf\t%lf", w.nl, w.freq, w.pred);
    int col;
    for(col=1;col<=N_CORPUS_IDUM;col++) {
        fprintf(f, "\t%d", w.idum[col]);
    }
    for(col=1;col<=N_CORPUS_DDUM;col++) {
        fprintf(f, "\t%lf", w.ddum[col]);
    }
    for(col=1;col<=N_CORPUS_CDUM;col++) {
        fprintf(f, "\t%s", w.cdum[col]);
    }
    fputc('\n', f);
}

void clear_word(swift_word obj) {
    // free allocated memory
    free_vector(int, obj.idum);
    free_vector(double, obj.ddum);
    int i;
    for(i=1;i<=N_CORPUS_CDUM;i++) {
        free(obj.cdum[i]);
    }
    free_vector(char*, obj.cdum);
}

int load_sentence(FILE *f, swift_sentence *s) {
    int n, j, k;
    if(fscanf(f, "%d", &n)!=1) return 0; // number of words in sentence
    swift_sentence new_s = new_sentence(n);
    for(j=1;j<=n;j++) {
        if(j>1) fscanf(f, "%*d");
        if(!load_word(f, &new_s.words[j])) {
            error(1, "Loading word %d failed!", j);
            return 0;
        }
    }
    *s = new_s;
    return 1;
}

void write_sentence(FILE *f, swift_sentence s) {
    int i;
    for(i=1;i<=s.nw;i++) {
        fprintf(f, "%d", s.nw);
        write_word(f, s.words[i]);
    }
}

void clear_sentence(swift_sentence obj) {
    int i;
    for(i=1;i<=obj.nw;i++)
        clear_word(obj.words[i]);
    free_vector(swift_word, obj.words);
}

int load_corpus(FILE *f, char* name, swift_corpus *corpus) {
    // we need to read the file twice -> once to count sentences and then to actually read the info
    long fpos = ftell(f);
    int nos = 0;
    int n, i, j;
    int cols = 3 + N_CORPUS_IDUM + N_CORPUS_DDUM + N_CORPUS_CDUM; // number of columns in corpus*.dat (without leading number-of-words column)
    while(fscanf(f, "%d", &n)==1) {
        // In first line, skip all expected columns
        for(i=1;i<=cols;i++)
            fscanf(f, "%*s");
        // For following lines, skip first column plus all expected other columns
        for(i=1;i<n;i++) {
            fscanf(f, "%*d");
            for(j=1;j<=cols;j++) {
                // skip word info stuff
                fscanf(f, "%*s");
            }
        }
        nos++;
    }
    // go back to where we started reading in order to read in the info
    fseek(f, fpos, SEEK_SET);
    swift_corpus new_c = new_corpus(nos);
    // copy name into new string
    if(name != NULL) {
        new_c.name = malloc(strlen(name)+1);
        strcpy(new_c.name, name);
    } else {
        new_c.name = NULL;
    }
    for(i=1;i<=nos;i++) {
        if(!load_sentence(f, &new_c.sentences[i])) {
            error(1, "Loading sentence %d failed.", i);
            return 0;
        }
    }
    *corpus = new_c;
    return 1;
}

void write_corpus(FILE *f, swift_corpus corpus) {
    int i;
    for(i=1;i<=corpus.ns;i++) {
        write_sentence(f, corpus.sentences[i]);
    }
}

void clear_corpus(swift_corpus obj) {
    int i;
    for(i=1;i<=obj.ns;i++) {
        clear_sentence(obj.sentences[i]);
    }
    if(obj.name != NULL)
        free(obj.name);
    free_vector(swift_sentence, obj.sentences);
}

swift_corpus *alloc_corpus(FILE *f, char *name) {
    swift_corpus* ret = malloc(sizeof(swift_corpus));
    if(!load_corpus(f, name, ret)) {
        error(1, "Loading corpus “%s” failed!", name);
        return NULL;
    }
    return ret;
}

void free_corpus(swift_corpus *obj) {
    clear_corpus(*obj);
    free(obj);
}

int load_trial(FILE *f, swift_trial *ret) {
    int nfix = 0, i, j, tmp;
    long fpos = ftell(f);
    //rewind(f);
    const int ncol = N_FIXSEQ_IDUM + N_FIXSEQ_DDUM + N_FIXSEQ_CDUM; // number of additional fixseq columns
    do {
        for(i=1;i<=5;i++) // ignore 5 columns
            fscanf(f, "%*s");
        fscanf(f, "%d", &tmp); // 6th column == first/last
        for(i=1;i<=ncol;i++) // ignore 4 columns
            fscanf(f, "%*s");
        if(nfix == 0 && tmp != 1) {
            error(2, "First fixation in sequence must start with [6]=1!");
            return 0;
        } else if(nfix > 0 && tmp == 1) {
            error(2, "Only first fixation in sequence may have [6]=1!");
            return 0;
        }
        nfix++;
    } while(tmp!=2);
    fseek(f, fpos, SEEK_SET);
    //rewind(f);
    swift_trial new_s = new_trial(nfix);
    for(j=1;j<=nfix;j++) {
        swift_fixation* fix = &new_s.fixations[j];
        if(j==1) {
            // Only if this is the first fixation of the sequence, read the first column and save as the fixation sequence's sentence nr.
            if(fscanf(f, "%d", &new_s.sentence) != 1) {
                error(1, "Error reading sentence number!");
                return 0;
            }else if(new_s.sentence < 1) {
                error(1, "Sentence number cannot be smaller than 1 (%d given)!", new_s.sentence);
                return 0;
            }
        }else{
            // Otherwise, just ignore first column
            fscanf(f, "%d", &tmp);
            if(tmp != new_s.sentence) {
                error(1, "All fixations within one sequence must belong to the same sentence. Fixation %d is associated with sentence %d but sentence %d was expected.", j, tmp, new_s.sentence);
                return 0;
            }
        }
        if(fscanf(f, "%d", &fix->fw) != 1) {
            error(1, "Error reading fixated word for fixation %d!", j);
            return 0;
        }else if(fix->fw < 1) {
            error(1, "Fixated word cannot be smaller than 1 (%d given for fixation %d)!", fix->fw, j);
            return 0;
        }
        if(fscanf(f, "%lf", &fix->fl) != 1) {
            error(1, "Error reading fixated letter for fixation %d!", j);
            return 0;
        }else if(fix->fl < 0.0) {
            error(1, "Fixated letter cannot be smaller than 0.0 (%lf given for fixation %d)!", fix->fl, j);
            return 0;
        }
        if(fscanf(f, "%d", &fix->tfix) != 1) {
            error(1, "Error reading fixation duration for fixation %d!", j);
            return 0;
        }else if(fix->tfix < 0) {
            error(1, "Fixation duration cannot be smaller than 0 (%d given for fixation %d)!", fix->tfix, j);
            return 0;
        }
        if(fscanf(f, "%d", &fix->tsac) != 1) {
            error(1, "Error reading saccade duration for fixation %d!", j);
            return 0;
        }else if(fix->fw < 1) {
            error(1, "Saccade duration cannot be smaller than 0 (%d given for fixation %d)!", fix->tsac, j);
            return 0;
        }
        fscanf(f, "%*d"); // skip first/last value (already evaluated in load_sequences(...))
        fix->idum = vector(int, N_FIXSEQ_IDUM);
        for(i=1;i<=N_FIXSEQ_IDUM;i++) {
            if(fscanf(f, "%d", &fix->idum[i]) != 1) {
                error(1, "Error reading idum[%d] for fixation %d!", i, j);
                return 0;
            }
        }
        fix->ddum = vector(double, N_FIXSEQ_DDUM);
        for(i=1;i<=N_FIXSEQ_DDUM;i++) {
            if(fscanf(f, "%lf", &fix->ddum[i]) != 1) {
                error(1, "Error reading ddum[%d] for fixation %d!", i, j);
                return 0;
            }
        }
        fix->cdum = vector(char*, N_FIXSEQ_CDUM);
        for(i=1;i<=N_FIXSEQ_CDUM;i++) {
            fix->cdum[i] = malloc(sizeof(char)*64);
            if(fscanf(f, "%s", fix->cdum[i]) != 1) {
                error(1, "Error reading cdum[%d] for fixation %d!", i, j);
                return 0;
            }
        }
    }
    *ret = new_s;
    return 1;
}

int validate_trial(swift_trial* seq, swift_corpus* corpus) {
    int i;
    if(seq->sentence < 1 || seq->sentence > nsentences(corpus)) {
        error(1, "A fixation sequence refers to sentence no. %d, which does not exist in this corpus (contains %d sentences)!", seq->sentence, nsentences(corpus));
        return 0;
    } else {
        for(i=1;i<=seq->nfix;i++) {
            if(seq->fixations[i].fw < 1 || seq->fixations[i].fw > nwords(corpus, seq->sentence)) {
                error(1, "Fixation #%d in sentence no. %d lies on word %d, which does not exist in this sentence (contains %d words)!", i, seq->sentence, seq->fixations[i].fw, nwords(corpus, seq->sentence));
                return 0;
            }
            if(seq->fixations[i].fl < 0.0 || seq->fixations[i].fl >= word_prop(corpus, seq->sentence, seq->fixations[i].fw, nl) + 1) {
                error(1, "Fixation #%d in sentence no. %d lies on letter %.1lf but it must lie within [0.0, %d.0) (word length)!", i, seq->sentence, seq->fixations[i].fl, word_prop(corpus, seq->sentence, seq->fixations[i].fw, nl) + 1);
                return 0;
            }
            if(seq->fixations[i].tfix < 0) {
                error(1, "Fixation duration #%d in sentence no. %d must not be negative!", i, seq->sentence);
                return 0;
            }
            if(i < seq->nfix && seq->fixations[i].tsac < 0) {
                error(1, "Saccade duration #%d in sentence no. %d must not be negative!", i, seq->sentence);
                return 0;
            }
        }
    }
    return 1;
}

void write_trial(FILE *f, swift_trial seq) {
    int i, j;
    for(i=1;i<=seq.nfix;i++) {
        swift_fixation fix = seq.fixations[i];
        fprintf(f, "%d\t%d\t%lf\t%d\t%d\t%d", seq.sentence, fix.fw, fix.fl, fix.tfix, fix.tsac, i==seq.nfix?2:i==1);
        for(j=1;j<=N_FIXSEQ_IDUM;j++) {
            if(fix.idum != NULL)
                fprintf(f, "\t%d", fix.idum[j]);
            else
                fprintf(f, "\t0");
        }
        for(j=1;j<=N_FIXSEQ_DDUM;j++) {
            if(fix.ddum != NULL)
                fprintf(f, "\t%lf", fix.ddum[j]);
            else
                fprintf(f, "\t0.0");
        }
        for(j=1;j<=N_FIXSEQ_CDUM;j++) {
            if(fix.cdum != NULL)
                fprintf(f, "\t%s", fix.cdum[j]);
            else
                fprintf(f, "\t-");
        }
        fputc('\n', f);
    }
}

void clear_fixation(swift_fixation obj) {
    free_vector(int, obj.idum);
    free_vector(double, obj.ddum);
    int i;
    for(i=1;i<=N_FIXSEQ_CDUM;i++) {
        free(obj.cdum[i]);
    }
    free_vector(char*, obj.cdum);
}

void clear_trial(swift_trial obj) {
    free_vector(swift_fixation, obj.fixations);
}

int load_dataset(FILE *f, char *name, swift_dataset* ret) {
    int ntrials = 0, i, k, tmp;
    long fpos = ftell(f);
    const int ncol = N_FIXSEQ_IDUM + N_FIXSEQ_DDUM + N_FIXSEQ_CDUM; // number of additional fixseq columns
    while(!feof(f)) {
        for(i=1;i<=5;i++) // ignore 5 columns
            fscanf(f, "%*s");
        fscanf(f, "%d", &tmp); // 6th column == first/last
        for(i=1;i<=ncol;i++) // ignore 4 columns
            fscanf(f, "%*s");
        if(tmp == 1) ntrials++;
    }
    swift_dataset new_s = new_dataset(ntrials);
    if(name != NULL) {
        new_s.name = malloc(strlen(name)+1);
        strcpy(new_s.name, name);
    } else {
        new_s.name = NULL;
    }
    // go back in file where we started
    fseek(f, fpos, SEEK_SET);
    for(k=1;k<=ntrials;k++) {
        if(!load_trial(f, &new_s.trials[k])) {
            error(1, "Loading trial %d failed!", k);
            return 0;
        }
    }
    *ret = new_s;
    return 1;
}

int validate_dataset(swift_dataset* seqs, swift_corpus* corpus) {
    int i;
    for(i = 1; i <= ntrials(seqs); i++) {
        if(!validate_trial(&seqs->trials[i], corpus)) {
            error(1, "There was a validation error in sequence #%d!", i);
            return 0;
        }
    }
    return 1;
}

void write_dataset(FILE *f, swift_dataset seqs) {
    int i;
    for(i=1;i<=seqs.n;i++) {
        write_trial(f, seqs.trials[i]);
    }
}

void clear_dataset(swift_dataset obj) {
    int i;
    for(i=1;i<=obj.n;i++) {
        clear_trial(obj.trials[i]);
    }
    free_vector(swift_trial, obj.trials);
    if(obj.name != NULL) {
        free(obj.name);
    }
}

swift_dataset* alloc_dataset(FILE *f, char *name) {
    swift_dataset* ret = malloc(sizeof(swift_dataset));
    if(!load_dataset(f, name, ret)) {
        error(3, "Allocating dataset failed!");
        return NULL;
    }
    return ret;
}

void free_dataset(swift_dataset* obj) {
    clear_dataset(*obj);
    free(obj);
}

#define free_swift_dataset free_dataset


// SWIFT ALGORITHM


typedef struct {
    swift_parameters* params;
    swift_corpus* corpus;
    RANSEED_TYPE seed;
} swift_model;

void free_swift_model(swift_model *obj) {
    free_parameters(obj->params);
    free_corpus(obj->corpus);
    free(obj);
}

/*
    This loads model parameters and configuration from environmentPath, parmPath, corpusFile
*/


int swift_load_model(char *environmentPath, char *parmFile, char *corpusFile, uint64_t seed, swift_model **dat, int verbose)
{

    FILE *fin;


    swift_model *m = (swift_model*) malloc(sizeof(swift_model));

    initSeed(seed, &m->seed);

    m->params = alloc_parameters(NULL);

    setdefaults(m->params);

    /* -----------------------
     SIMULATION PARAMETERS
     ----------------------- */

    char environmentFile[strlen(environmentPath)+20];
    sprintf(environmentFile, "%s/swiftstat.inp", environmentPath);

    fin = fopen( environmentFile, "r" );
    if ( fin == NULL ) {
        warn("“%s” not found.", environmentFile);
    }else{
        fscan_parameters(fin, m->params);
        fclose(fin);
    }


    /* ------------------
     MODEL PARAMETERS
     ------------------- */

    int default_or_specific_file_loaded = 0;

    char defaultFilePath[PATH_MAX];

    sprintf(defaultFilePath, "%s/swpar_default.par", environmentPath);

    fin = fopen(defaultFilePath,"r");
    if ( fin != NULL ) {
        fscan_parameters(fin, m->params);
        fclose(fin);
        default_or_specific_file_loaded = 1;
    }

    fin = fopen(parmFile,"r");
    if ( fin != NULL ) {
        fscan_parameters(fin, m->params);
        fclose(fin);
        default_or_specific_file_loaded = 1;
    }

    if(!default_or_specific_file_loaded) {
        warn("Did not find %s or %s. Only parameters in %s will be considered. This is not recommended!", defaultFilePath, parmFile, environmentFile);
    }

    /* -------------------------
     TEXT CORPUS
     ------------------------- */


    fin = fopen(corpusFile,"r");
    if (fin == NULL) {
        error(1, "Wrong corpus specified! Expected file: %s", corpusFile);
        return 0;
    }
    m->corpus = alloc_corpus(fin, corpusFile);
    if(m->corpus == NULL) {
        error(1, "Loading corpus from “%s” failed.", corpusFile);
        return 0;
    }
    fclose(fin);


    *dat = m;

    return 1;
}

int swift_load_model_std(char *inputPath, char *environmentPath, char *corpusName, char *seqName, uint64_t seed, swift_model **dat, int verbose) {
    char *envPath = environmentPath == NULL ? inputPath : environmentPath;
    //char envFile[strlen(envPath)+30];
    char parmFile[strlen(inputPath)+strlen(corpusName)+strlen(seqName)+30];
    char corpusFile[strlen(inputPath)+strlen(corpusName)+30];
    //sprintf(envFile, "%s/swiftstat.inp", envPath);
    sprintf(parmFile, "%s/swpar_%s_%s.par", inputPath, corpusName, seqName);
    sprintf(corpusFile, "%s/corpus_%s.dat", inputPath, corpusName);
    return swift_load_model(envPath, parmFile, corpusFile, seed, dat, verbose);
}

/*
    This loads fixation sequences from file specified by fixseqName into fixseqs, flet, Nfix and Ntrials
*/
int swift_load_data(char *fixseqFile, swift_dataset **dat, int verbose)
{
    FILE *fin;


    fin = fopen(fixseqFile, "rb");
    if (fin == NULL) {
        error(1, "Wrong fixation sequence specified! Expected file: %s", fixseqFile);
        return 0;
    }

    swift_dataset * d = alloc_dataset(fin, fixseqFile);

    fclose(fin);

    if(d == NULL) {
        return 0;
    }

    *dat = d;

    return 1;

}

int swift_validate(swift_dataset *dat, swift_model* m) {
    return validate_dataset(dat, m->corpus);
}

int swift_load_data_std(char *inputPath, char *seqName, swift_dataset **dat, int verbose) {
    char seqFile[strlen(inputPath)+strlen(seqName)+30];
    sprintf(seqFile, "%s/fixseqin_%s.dat", inputPath, seqName);
    return swift_load_data(seqFile, dat, verbose);
}


// CONVENIENCE CALLERS

void swift_single_eval(swift_model *m, swift_dataset* d, int * trials, double *logliks, unsigned int threads, int verbose) {
    gaengine(m->params, m->corpus, d, &m->seed, logliks, 1, threads, trials, verbose, NULL);
}

void swift_single_eval_all(swift_model *m, swift_dataset* d, double *logliks, unsigned int threads, int verbose) {
    swift_single_eval(m, d, NULL, logliks, threads, verbose);
}

void swift_eval(swift_model *m, swift_dataset* d, int * trials, double *logliks, unsigned int threads, int verbose) {
    gaengine(m->params, m->corpus, d, &m->seed, logliks, 0, threads, trials, verbose, NULL);
}

void swift_eval_all(swift_model *m, swift_dataset* d, double *logliks, unsigned int threads, int verbose) {
    swift_eval(m, d, NULL, logliks, threads, verbose);
}

void swift_eval_single(swift_model *m, swift_dataset* d, int trial, double *logliks, unsigned int threads, int verbose) {
    int trials[] = {trial, 0};
    swift_eval(m, d, trials, logliks, threads, verbose);
}

void swift_generate(swift_model* m, char* output_dir, char* seqname, int * items, unsigned int threads, int make_fixseqin, int verbose) {
    struct swift_files files;
    char file_fsim[PATH_MAX], file_fseq[PATH_MAX], file_proc1[PATH_MAX], file_proc2[PATH_MAX];
    sprintf(file_fsim, "%s/seq_%s.dat", output_dir, seqname);
    sprintf(file_fseq, "%s/fixseqin_%s.dat", output_dir, seqname);
    files.fsim = fopen(file_fsim, "w");
    if(make_fixseqin) {
        files.fseq = fopen(file_fseq, "w");
    } else {
        files.fseq = NULL;
    }
    if(files.fsim == NULL) {
        stop(1, "Could not open “%s” for writing!", file_fsim);
    }
    // sprintf(file_proc1, "%s/seq_procord1_%s.dat", output_dir, seqname);
    // sprintf(file_proc2, "%s/seq_procord2_%s.dat", output_dir, seqname);
    // files.f1 = fopen(file_proc1, "w");
    // files.f2 = fopen(file_proc2, "w");

    if(val(m->params, output_ahist)) {
        char file_ahist[PATH_MAX];
        sprintf(file_ahist, "%s/seq_ahist_%s.dat", output_dir, seqname);
        files.f3 = fopen(file_ahist, "w");
    } else {
        files.f3 = NULL;
    }


    gaengine(m->params, m->corpus, NULL, &m->seed, NULL, 0, threads, items, verbose, &files);

    if(files.fsim != NULL) {
        fflush(files.fsim);
        fclose(files.fsim);
    }

    if(files.fseq != NULL) {
        fflush(files.fseq);
        fclose(files.fseq);
    }

    if(files.f3 != NULL) {
        fflush(files.f3);
        fclose(files.f3);
    }
}

void swift_generate_all(swift_model *m, char* output_dir, char* seqname, unsigned int threads, int make_fixseqin, int verbose) {
    swift_generate(m, output_dir, seqname, NULL, threads, make_fixseqin, verbose);
}

void swift_generate_single(swift_model *m, char* output_dir, char* seqname, unsigned int sentence, int verbose) {
    int items[] = {sentence, 0};
    swift_generate(m, output_dir, seqname, items, 1, 0, verbose);
}

#endif
