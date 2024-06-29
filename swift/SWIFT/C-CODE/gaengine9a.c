/*------------------------------------------
 Function GAENGINE_FITTING.C
 ===================
 Simulation of the SWIFT model along experimental data
 --------------------------------------------*/

// Set dummy variable dimensions
// Dummy variables are additional columns in corpus and fixseq files
// They can be integers (idum), doubles (ddum) or strings (cdum)
// The number of dummy variables is defined in the following constants
// The order of variables in corpus/fixseq files is always:
// corpus: sentence length | word length | word freq | word predictability | idum[1..n] | ddum[1..n] | cdum[1..n]
// fixseq: sentence | fixated word | fixated letter | fixation duration | saccade duration | idum[1..n] | ddum[1..n] | cdum[1..n]
#define N_FIXSEQ_IDUM 0
#define N_FIXSEQ_DDUM 0
#define N_FIXSEQ_CDUM 0
#define N_CORPUS_IDUM 0
#define N_CORPUS_DDUM 0
#define N_CORPUS_CDUM 0
#define N_LOGLIKS 3

// What is the version of this algorithm?
#define SWIFT_VERSION_MAJOR 14
#define SWIFT_VERSION_MINOR 1

#define SWIFT_VARIANT "base"

#ifdef P_STOP
#define N_LOGLIKS 4
#define SWIFT_VARIANT "base pstop"
#endif

// Numerical recipes includes
//#include "./NumericalRecipes/nrutil.c"
#include "logsumexp.c"


#include "gausslike.c"
#include "epallike.c"


#if defined(LEXRATE_GAUSS)
#include "lexrate7_gauss.c" // gaussian span
#elif defined(LEXRATE_INVPARAB)
#include "lexrate8_invparab.c" // gaussian span
#else
#warning You have not specified a lexrate module. Using LEXRATE_INVPARAB by default.
#include "lexrate8_invparab.c" // inverse parabolic span
#endif

#if defined(EXECSACC_GAUSS)
#include "execsacc_gauss.c"
#elif defined(EXECSACC_INVGAUSS)
#include "execsacc_invgauss.c"
#elif defined(EXECSACC_INVGAMMA)
#include "execsacc_invgamma.c"
#elif defined(EXECSACC_GAMMA)
#include "execsacc_gamma.c"
#else
#warning You have not specified a saccade execution module. Using EXECSACC_INVGAMMA by default.
#include "execsacc_gamma.c"
#endif

#include "selectar3a.c" // distance-dependent target selection
//#include "selectar3b.c" // distance-independent target selection
//#include "selectar3d.c" // distance-independent target selection with improved fallback

#include "logliktimer4.c"


// Macros for dynamic resizing of buffer vectors (inhvec, mat, kvec, tvec, a_hist, t_hist)
// Added by Max, Feb 27, 2019
// Initialize vectors as usual with dvector(1, ...), ivector(1, ...) and matrices with dmatrix(1, ..., 1, ...), imatrix(1, ..., 1, ...)
// Ensure length of a vector with ensure_ivector_size(...), ensure_dvector(size) and equivalents for matrices
// Note: indices should start at 1, not 0, i.e. initialize as shown above
// Note: Value of current_size is changed if vector/matrix is resized!
// Size is incremented in desired step size, i.e. if requested size is smaller than current size, extend length by integer increments of step_size until current size is greater than requested size
// If length should be extended to exactly the requested size but not more, use step_size=1
// #define ensure_vector_size(vec, current_size, requested_size, step_size, element_type) {if(requested_size > current_size) stop(5, "Requested vector size (%d) exceeds allocated vector size (%d)! Please consider allocating more memory!", requested_size, current_size);}
// #define ensure_vector_size(vec, current_size, requested_size, step_size, element_type) {if(requested_size > current_size) {current_size += step_size * (1 + (requested_size - current_size) / step_size); /*printf("Resizing vector to %d elements.\n", current_size);*/ vec = (element_type*) realloc(vec, (size_t) (sizeof(element_type) * (current_size + 1))); if(vec == NULL) stop(1, "Could not increase vector size, probably due to memory overflow! If this happened in generative mode, please consider abort conditions.");}}
// #define ensure_ivector_size(vec, current_size, requested_size, step_size) ensure_vector_size(vec, current_size, requested_size, step_size, int)
// #define ensure_dvector_size(vec, current_size, requested_size, step_size) ensure_vector_size(vec, current_size, requested_size, step_size, double)
// #define ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, element_type, element_type_init) {int old_size = current_rows, i; ensure_vector_size(mat, current_rows, requested_rows, step_size, element_type*); for(i=old_size+1;i<=current_rows;i++) mat[i] = element_type_init(1, n_cols);}
// #define ensure_imatrix_rows(mat, current_rows, requested_rows, step_size, n_cols) ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, int, ivector)
// #define ensure_dmatrix_rows(mat, current_rows, requested_rows, step_size, n_cols) ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, double, dvector)

#define ensure_vector_size(vec, current_size, requested_size, step_size, element_type) if(requested_size >= current_size) {int new_size = current_size + step_size * (1 + (requested_size - current_size) / step_size); vec = resize_array(element_type, vec, 1, current_size, new_size); current_size = new_size;}
#define ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, element_type) if(requested_rows >= current_rows) {int new_rows = current_rows + step_size * (1 + (requested_rows - current_rows) / step_size); mat = resize_array(element_type, mat, 2, current_rows, n_cols, new_rows, n_cols); current_rows = new_rows;}
#define ensure_ivector_size(vec, current_size, requested_size, step_size) ensure_vector_size(vec, current_size, requested_size, step_size, int)
#define ensure_dvector_size(vec, current_size, requested_size, step_size) ensure_vector_size(vec, current_size, requested_size, step_size, double)
#define ensure_imatrix_rows(mat, current_rows, requested_rows, step_size, n_cols) ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, int)
#define ensure_dmatrix_rows(mat, current_rows, requested_rows, step_size, n_cols) ensure_matrix_rows(mat, current_rows, requested_rows, step_size, n_cols, double)



// This is a shorthand version to require that specific parameters have been set
#define require(params, par) require_parameter(params, swift_parameter_ids.par)

#define sq(A) ((A)*(A))



void gaengine(swift_parameters* params, swift_corpus* corpus, swift_dataset* dataset, RANSEED_TYPE* super_seed, double *logliks, int single_logliks, unsigned int threads, int * items, int verbose, struct swift_files * files) {

    // Make sure we are not running the algorithm with NULL values

    #ifdef CHECK_MEMORY
    init_memory_stuff();
    #endif

    if(params == NULL)
        stop(3, "The parameter storage pointer (“params”) cannot be NULL!");
    if(corpus == NULL)
        stop(3, "The sentence corpus pointer (“corpus”) cannot be NULL!");
    if(super_seed == NULL)
        stop(3, "The random seed pointer (“super_seed”) cannot be NULL!");

    // Assume we are in fitting mode if sequences and logliks are given
    const int is_fitting = logliks != NULL;

    if(!is_fitting && files == NULL && dataset == NULL)
        stop(3, "In generative mode, the files pointer (“files”) and the dataset pointer (“dataset”) cannot both be NULL!");

    // Use val(params, param) macros to retrieve param from param storage
    // For definition of the val(...) macro, see API
    // val(params, x) is identical to params->x.val
    // Note: val(...) ALWAYS returns a value, even if the param is undefined!
    //     -> ALWAYS (!) check that parameters have been set using the require(x) macro!
    //     -> The macro will do nothing if x has a value but exit with an error if x has no value

    // first, check technical parameters
    require(params, runs);
    require(params, nsims);
    int nread = val(params, runs);
    int nsims = val(params, nsims);

    // base SWIFT model parameters -> make sure all of those are defined, then retrieve their values
    require(params, delta0);
    require(params, delta1);
    require(params, asym);
    require(params, eta);
    require(params, alpha);
    require(params, beta);
    require(params, gamma);
    require(params, minact);
    require(params, theta);
    require(params, msac0);
    require(params, msac);
    require(params, h);
    require(params, h1);
    require(params, ppf);
    require(params, iota);
    require(params, refix);
    require(params, misfac);
    require(params, kappa0);
    require(params, kappa1);
    require(params, proc);
    require(params, decay);
    require(params, tau_l);
    require(params, tau_n);
    require(params, tau_ex);
    require(params, aord);
    require(params, cord);
    require(params, lord);
    require(params, nord);
    require(params, xord);
    require(params, ocshift);
    require(params, omn_fs1);
    require(params, omn_fs2);
    require(params, omn_sk1);
    require(params, omn_sk2);
    require(params, omn_frf1);
    require(params, omn_frf2);
    require(params, omn_brf1);
    require(params, omn_brf2);
    require(params, omn_rg1);
    require(params, omn_rg2);
    require(params, sre_fs1);
    require(params, sre_fs2);
    require(params, sre_sk1);
    require(params, sre_sk2);
    require(params, sre_frf1);
    require(params, sre_frf2);
    require(params, sre_brf1);
    require(params, sre_brf2);
    require(params, sre_rg1);
    require(params, sre_rg2);
    #ifdef P_STOP
    //require(params, sprob);
    #endif

    double delta0 = val(params, delta0);   // nondynamical processing span in letter spaces
    double delta1 = val(params, delta1);   // dynamical processing span in letter spaces
    double asym = val(params, asym);       // asymmetry of processing span to the right
    double eta = val(params, eta);         // word length exponent
    double alpha = val(params, alpha);     //
    double beta = val(params, beta);       // frequency modulation
    double gamma = val(params, gamma);     // target selection exponent
    double minact = val(params, minact);   // minimum activation threshold of words for target selection
    double theta = val(params, theta);     // influence of predicatibility processing speed
    double msac0 = val(params, msac0);     // relative duration of first fixation in the sentence
    double msac = val(params, msac);       // relative duration of global saccade program
    double h = val(params, h);             // foveal inhibition-factor
    double h1 = val(params, h1);           // parafoveal inhibition-factor
    double ppf = val(params, ppf);         // inhibition from words to the left of fixation
    double iota = val(params, iota);       // transfer across saccades (activation loss during saccade)
    double refix = val(params, refix);     // relative duration of the labile saccade stage for refixations
    double misfac = val(params, misfac);   // relative duration of the labile saccade stage for misplaced fixations
    double kappa0 = val(params, kappa0);   // nonlabile latency dependence on target distance (Kalesnykas & Hallet)
    double kappa1 = val(params, kappa1);   //
    double proc = val(params, proc);       // relative processing speed for postlexical processing (as[]=2)
    double decay = val(params, decay);     // global decay during postlexical processing
    double tau_l = val(params, tau_l);     // mean duration of the labile saccade program
    double tau_n = val(params, tau_n);     // mean duration of the nonlabile saccade program
    double tau_ex = val(params, tau_ex);   // mean saccade duration
    double aord = val(params, aord);       // order of random walks for word activation (maximum)
    double cord = val(params, cord);       // order of random walks for global saccade program & labile stage
    double lord = val(params, lord);       // order of random walks for labile stage
    double nord = val(params, nord);       // order of random walks for nonlabile stage
    double xord = val(params, xord);       // order of random walks for saccade execution
    double ocshift = val(params, ocshift); // oculomotor shift parameter
    double *s1 = dvector(1, 5);            // oculomotor noise intercept parameters
    s1[1] = val(params, omn_fs1);          // ... forward saccades
    s1[2] = val(params, omn_sk1);          // ... skippings
    s1[3] = val(params, omn_frf1);         // ... forward refixations
    s1[4] = val(params, omn_brf1);         // ... backward refixations
    s1[5] = val(params, omn_rg1);          // ... regressions
    double *s2 = dvector(1, 5);            // oculomotor noise slope parameters
    s2[1] = val(params, omn_fs2);          // ... forward saccades
    s2[2] = val(params, omn_sk2);          // ... skippings
    s2[3] = val(params, omn_frf2);         // ... forward refixations
    s2[4] = val(params, omn_brf2);         // ... backward refixations
    s2[5] = val(params, omn_rg2);          // ... regressions
    double *r1 = dvector(1, 5);            // saccadic range error intercept parameters
    r1[1] = val(params, sre_fs1);          // ... forward saccades
    r1[2] = val(params, sre_sk1);          // ... skippings
    r1[3] = val(params, sre_frf1);         // ... forward refixations
    r1[4] = val(params, sre_brf1);         // ... backward refixations
    r1[5] = val(params, sre_rg1);          // ... regressions
    double *r2 = dvector(1, 5);            // saccadic range error slope parameters
    r2[1] = val(params, sre_fs2);          // ... forward saccades
    r2[2] = val(params, sre_sk2);          // ... skippings
    r2[3] = val(params, sre_frf2);         // ... forward refixations
    r2[4] = val(params, sre_brf2);         // ... backward refixations
    r2[5] = val(params, sre_rg2);          // ... regressions
    //double sprob = val(params, sprob);


    // rescale parameters
    msac = msac*100.0;
    tau_l = tau_l*100.0;
    tau_n = tau_n*100.0;
    tau_ex = tau_ex*100.0;

    int ix, jx;

    // determine highest word frequency in corpus
    double maxfreq = 0.0;

    for(ix=1;ix<=nsentences(corpus);ix++) {
        // nsentences(corpus) is the number of sentences stored in corpus
        // nwords(corpus, i) is the number of words in sentence no. i
        // word_prop(corpus, i, j, prop) extract property/field "prop" from word j of sentence i (see struct swift_word in swiftstat7_api for details)
        // word_prop(corpus, ix, jx, freq) == corpus->sentences[ix].words[jx].freq
        for(jx=1;jx<=nwords(corpus, ix);jx++) {
            if(word_prop(corpus, ix, jx, freq) > maxfreq) {
                maxfreq = word_prop(corpus, ix, jx, freq);
            }
        }
    }

    int sn;

    // depending on running mode (fitting/generative), determine first and last sequence/sentence between which to iterate
    // this determines the number of compute jobs, which are then distributed across cores using OpenMP (see for-loop below)

    // sn = how many items are we generating/evaluating?
    if(items == NULL) {
        if(is_fitting) {
            sn = ntrials(dataset);
        } else {
            sn = nsentences(corpus);
        }
    } else {
        sn = 0;
        while(items[sn] > 0) {
            sn++;
        }
    }

    uint64_t * seeds = vector(uint64_t, sn);

    if(seeds == NULL) error(1, "Could not allocate memory for %d random number generators!", sn);

    // for each job, generate an individual random seed
    // the result (both generative and fitting) is replicable, no matter how many cores are used, as long as the super_seed is the same
    //RANSEED_TYPE seeds[sn];

    for(ix=1;ix<=sn;ix++) {
        seeds[ix] = ranlong(super_seed);
    }

    // Default number of threads is 1 (all jobs are executed by one worker in sequential order)
    int n_threads = 1;

    // Only change n_threads if we have not disabled threading
    #ifndef DISABLE_THREADS
    if(threads > 0) {
        // We are in fitting mode or we are in single-sentence generative mode
        // A thread number has been requested --> set n_threads to that number
        n_threads = threads;
    } else {
        // This should not occur but just in case, abort the procedure
        n_threads = omp_get_max_threads();
    }
    #endif

    double    sum_Pmisloc = 0.0;
    int       Nmisloc = 0;
    int       Nfix_total = 0;

    if(is_fitting && !single_logliks) {
        // initialize loglik return array if used for sum of logliks
        for(ix=0;ix<N_LOGLIKS;ix++) {
            logliks[ix] = 0.0;
        }
    }

    // If generative mode and dataset pointer != NULL, use it to store sequences
    if(!is_fitting && dataset != NULL && dataset->n != sn*nread) {
      stop(1, "Dataset pointer in generative mode points to storage with %d slots but needs %d (%d items, %d replications)!", dataset->n, sn*nread, sn, nread);
    }

    // This pragma makes the for-loop parallel, using OpenMP parallelization
    // If line is commented out, OpenMP is not supported, or threading has been disabled, it will just run serially
    // schedule(guided) uses OpenMP's “guided” scheduling technique to distribute jobs across cores
    //   -> “guided” was tested to be the most efficient for fitting purposes
    //   -> alternatives: schedule(static), schedule(dynamic), schedule(auto), ect., see OpenMP manual
    // all variables from higher scope (defined before for loop) are shared across workers
    // all variables defined within for loop are private for each worker
    // private(ix) makes sure that each worker has a private ix
    // use n_threads to determine number of threads (see above)
    // as each job has its own random seed, results are guaranteed to be identical for the same super_seed, no matter in which sequence jobs are executed and finished.
    // Slight variations might occur in generative output as far as the sequence of sentence numbers is concerned. This can be avoided by using a single thread.
    #ifndef DISABLE_THREADS
    #pragma omp parallel for schedule(guided) private(ix) num_threads(n_threads)
    #endif
    for(ix=1;ix<=sn;ix++)
    {

        //printf("Worker %d takes job %d.\n", omp_get_thread_num(), ix);

        int fitting = is_fitting;
        //RANSEED_TYPE rangen;
        RANSEED_TYPE seed[1];
        //printf("%d %lx\n", ix, seed);
        initSeed(seeds[ix], seed);




        double    **inhvec;
        double    *lex, *view, *border, *kvec, *kv, *tvec;
        double    *labv1, *labv2, *nlabv1, *nlabv2;
        double    *r_count, *W, *p_trans, *t_hist, dt_nlab;
        double    *procrate, *Ptar, *Psac, *Ptmp;
        double    dynfac, inhibrate;
        double    inhib, kpos, saccerr, disptime, labrate;
        double    presaclen, Pcomb, prevact;
        double    kapparate, dt;
        double    x, Wsum, rnd, psum, dist, sacamp;
        double    ifovea, iparafovea, ptheo, lastprob;
        double    mlp, prev_mlp, upcoming_l_pos;
        double    *loglik_temp, *loglik_spat, *loglik_comb;
        int       *acompletion, **mat;
        double    *aa,  **a_hist;
        int       **c_hist;
        int       *n1, *nf1, *n2, *nf2;
        int       *as, *adir;
        int       *n_count, *N_count;
        int       refix_sw_cur, refix_sw_prev;
        int       lab, nlab, sacc, num, misloc, mis, Nrefix = 0;
        int       k_fix, n_fix, next_tar, intended;
        int       endsw_sentence, endsw_trial, canc;
        int       presac, testreg, firstpass, dispfix;
        int       i, j, k, l, last, r, s, s0, w, n_hist, trans_point;
        int       state;
        int       n_trans;
        double    t = 0.0, t_l, t_n, t_i, t_fix;

        // experimental manipulation
        #ifdef P_STOP
        double *loglik_stop;
        double p_stop = 0.0;
        int last_fixation;
        double sum_lexproc, sum_postlexproc, NW_processed;
        #endif

        // initialize P(mislocated fixation) = 0.0 --> this is updated after each saccade, i.e. only after the first fixation
        double Pmisloc = 0.0;

        unsigned int Nfix;
        double *flet;
        int *fwrd, *tfix, *tsac; // Max, 20.11.18: Previously, these used to be combined in the "fixseq" imatrix, now single ivectors of same length (hence more meaningful variable names and more straightforward use of different data types, such as flet)
        if(is_fitting) {
            int seqid;
            if(items == NULL) {
                seqid = ix;
            } else {
                seqid = items[ix-1];
            }
            s = trial_prop(dataset, seqid, sentence);
            Nfix = nfixations(dataset, seqid);
            flet = dvector(1, Nfix);
            fixation_vec(dataset, seqid, fl, flet);   // copy .fl property from all fixations in sequence ix to vector flet
            fwrd = ivector(1, Nfix);
            fixation_vec(dataset, seqid, fw, fwrd);   // copy .fw property from all fixations in sequence ix to vector fwrd
            tfix = ivector(1, Nfix);
            fixation_vec(dataset, seqid, tfix, tfix); // copy .tfix property from all fixations in sequence ix to vector tfix
            tsac = ivector(1, Nfix);
            fixation_vec(dataset, seqid, tsac, tsac); // copy .tsac property from all fixations in sequence ix to vector tsac
        } else {
            if(items == NULL) {
                s = ix;
            } else {
                s = items[ix-1];
            }
        }


        //printf("%d %d %lu @#%lx\n", ix, s, ranlong(seed), (size_t) seed);

        int NW = nwords(corpus, s);
        // independent variables (predictors)
        double * freq = dvector(1, NW);
        word_vec(corpus, s, freq, freq);    // copy .freq property from all words in sentence s to vector freq
        int * len = ivector(1, NW);
        word_vec(corpus, s, nl, len);       // copy .nl property from all words in sentence s to vector len
        double * pred = dvector(1, NW);
        word_vec(corpus, s, pred, pred);

        /* declaration of vectors */
        as = ivector(1,NW);              /* pre-proc./lex. completion? (0/1) */
        adir = ivector(1,NW);            /* rw direction */
        lex = dvector(1,NW);             /* lexical difficulty */
        view = dvector(1,NW);            /* optimal viewing positions */
        border = dvector(1,NW);          /* word borders */
        labv1 = dvector(1,cord);
        labv2 = dvector(1,cord);
        nlabv1 = dvector(1,cord);
        nlabv2 = dvector(1,cord);
        procrate = dvector(1,NW);        /* processing rate (time-dependent)*/
        Ptar = dvector(1,NW);            /* probabilities of target words being selected */
        Psac = dvector(1,NW);            /* probabilities of target words being selected */
        Ptmp = dvector(1, NW);
        // int ** proc_order = matrix(int, 2, NW);     //  output vector for order in which words have been processed
        // int * orders = ivector(1,2);

        int max_nfix = NW * 100;
        int max_inhvec_len = NW * 1000;
        int max_hist_len = ceil(aord*NW) * 3;
        int max_mat_len = max_nfix, max_tvec_len = max_nfix, max_kvec_len = max_nfix;
        kvec = dvector(1,max_kvec_len);        /* fixation position */
        tvec = dvector(1,max_tvec_len);        /* time */
        mat = imatrix(1,max_mat_len,1,3);     /* fixations */
        inhvec = dmatrix(1,max_inhvec_len,1,4);  /* inhibition vector */
        int max_a_hist_len = max_hist_len, max_t_hist_len = max_hist_len, max_c_hist_len = max_hist_len;
        a_hist = dmatrix(1,max_a_hist_len,1,NW);  /* activation field history (with t_hist & n_hist) */
        t_hist = dvector(1,max_t_hist_len);
        c_hist = imatrix(1,max_c_hist_len,1,NW+4);

        endsw_sentence = 0;

        int cur_fix = 1;

        r = 1;

        if(is_fitting) {
            loglik_comb = dvector(1, nread);
            loglik_temp = dvector(1, nread);                   /* support variable for fitting */
            loglik_spat = dvector(1, nread);                   /* support variable for fitting */
            #ifdef P_STOP
            loglik_stop = dvector(1, nread);
            #endif
        }

        // loop over trials
        while (endsw_sentence == 0) {  //for ( r=1; r<=nread; r++ ) {




            /** DEFAULT BUFFER VECTOR LENGTHS **/
            /** Buffer vectors hold temporary or output values and the required length is unknown beforehand. **/
            /** Therefore, we need to allocate an arbitrary high size and resize them later if necessary (see ensure_vector_size definitions above) **/
            /** Changed by Max, Feb 27, 2019 **/

            #ifdef P_STOP
            last_fixation = 0;
            p_stop = 0.0;
            #endif

            if(is_fitting) {
                loglik_spat[r] = 0.0; // initialize log-likelihood to 0.0 (P=1.0) for the first fixation
                loglik_temp[r] = 0.0;
                #ifdef P_STOP
                loglik_stop[r] = 0.0;
                #endif
            }

            // orders[1] = orders[2] = 0;

            // variable controlling random walks
            // NW+5 random walks for NW words plus timer, 2 perceptual delays, 2 saccade stages
            N_count = ivector(1,4+NW);
            N_count[1] = (int) cord;
            N_count[2] = (int) lord;
            N_count[3] = (int) nord;
            N_count[4] = (int) xord;

            r_count = dvector(1,4+NW);
            r_count[1] = (1.0*N_count[1])/(1.0*msac);
            r_count[2] = (1.0*N_count[2])/(1.0*tau_l);
            r_count[3] = (1.0*N_count[3])/(1.0*tau_n);
            r_count[4] = (1.0*N_count[4])/(1.0*tau_ex);
            /* N_count[x]: threshold of each random walk; maximum number of steps/states to complete */
            /* n_count[x]: count for the current number of steps completed */
            /* r_count[x]: rate with which each random walk proceeds a step; determines transition probabilities W or p_trans below */
            n_count = ivector(1,4+NW);
            for ( l=1; l<=4+NW; l++ )  n_count[l] = 0;
            W = dvector(1,4+NW);
            p_trans = dvector(1,4+NW);

            /* initialization of dynamical variables */
            aa = dvector(1,NW);
            acompletion = ivector(1,NW);


            for ( i=1; i<=NW; i++ )  {
                double logf;
                aa[i] = 0.0;      // word-based activations
                as[i] = 1;      // processing state
                adir[i] = 1;    // random-walk direction
                acompletion[i] = 1;   // =0 if word is completely processed
                logf = log(freq[i])/log(maxfreq);
                if ( logf<0.0 )  logf = 0.0;
                lex[i] = 1.0 - beta*logf;   // word frequency effect
                N_count[4+i] = (int) ceil(aord*lex[i]);  // proportion of maximum activation aord

                /* compute optimal viewing positions and word borders */
                if ( i==1 )  {
                    view[i] = len[i] * asym / (asym + 1.0) + 1.0; // changed by Ralf on Dec. 22, 2016
                    border[i] = len[i] + 1.0;
                }
                else  {
                    view[i] = border[i-1] + len[i] * asym / (asym + 1.0) + 1.0;  // changed by Ralf on Dec. 22, 2016
                    border[i] = border[i-1] + len[i] + 1.0;
                }
            }

            if ( fitting <= 0 ) {
                k = 1;                       /* fixated word */
                kpos = view[1];              /* first fixation position given by OVP */
            } else {
                k = fwrd[cur_fix];     /* fixated word */
                if ( k==1 ) {                /* first fixation position given by data */
                    kpos = flet[cur_fix];
                } else {
                    kpos = border[k] - (len[k] + 1) + flet[cur_fix];
                }
            }

            last        = 0;                 /* last fixated word */
            t           = 0.0;               /* time */
            lab         = 0;                 /* labile saccade program? (0/1) */
            t_l         = 0.0;               /* start of labile program */
            nlab        = 0;                 /* non-labile saccade program? (0/1) */
            t_n         = 0.0;               /* start of non-labile saccade program */
            sacc        = 0;                 /* saccade in execution? (0/1) */
            t_i         = 0.0;               /* onset of saccade */
            next_tar    = 1;                 /* target of next saccade */
            intended    = 1;                 /* intended target word */
            dist        = 0.0;               /* distance of next saccade target */

            lastprob    = 0.0;
            mlp         = 0.0;

            /*-------------------------
             simulation algorithm
             -------------------------*/
            t_fix = 0.0;
            k_fix = 1;
            n_fix = 0;
            n_count[1] = (int) (msac0*N_count[1]);
            canc = 0;

            if ( fitting <= 0 ) {
                endsw_trial = 0;
            } else {
                endsw_trial = cur_fix >= Nfix;
                n_trans = 0;                     /* transition counter for inhibition vector */
                upcoming_l_pos = 0.0;           /* support variable for fitting */
            }


            // initialize history if no activations have been saved yet
            n_hist = 0;                     /* count changes of activation */
            for (i = 1; i<=NW; i++) a_hist[1][i] = 0.0;
            for (i = 1; i<=NW+4; i++) c_hist[1][i] = 0;
            t_hist[1] = 0.0;

            // sim: read sentence either until completion, running out of time or to a maximum of 3*NW fixations
            // fit: read sentence until end of trial (don't evaluate last fixation)
            while ( endsw_trial == 0 ) {
                /* compute inhibition */
            //printf("B %d %d %d %ld %lx\n", last_fixation, ix, s, ranlong(seed), seed);


                // fovea
                ifovea = aa[k];

                for (i=k+1, iparafovea = 0.0; i<=NW; i++)  {
                    iparafovea += aa[i];
                }

                inhib = h*ifovea + h1*iparafovea;
                if ( inhib<0.0 )  inhib = 0.0;

                /* iterate lexical processing with time step dt */
                lexrate(aa,as,view,len,aord,alpha,delta0,delta1,asym,pred,proc,ppf,theta,decay,eta,NW,kpos,k,0,procrate,fitting);

                for (i=1; i <= NW; i++) procrate[i] *= alpha;

                /* Gillespie algorithm */
                /* total transition probability */
                inhibrate = 1.0/(1.0+inhib);
                W[1] = r_count[1]*inhibrate;                          /* rate of random walk for timer */
                W[2] = r_count[2]*lab*labrate;                          /* rate of random walk for labile sacprog stage */
                kapparate = 1.0/(1.0 + kappa0*exp(-kappa1*sq(dist)));
                W[3] = r_count[3]*nlab*kapparate;                        /* ... nonlabile stage */
                W[4] = r_count[4]*sacc;                                 /* ... saccade execution */
                for ( i=1; i<=NW; i++ )  {
                    W[4+i] = procrate[i]*acompletion[i];
                }
                /* acompletion = 1: not fully processed; acompletion = 0: fully processed */
                /* exponential pausing time */
                for ( i=1, Wsum = 0.0; i<=4+NW; i++ )  {
                    Wsum += W[i];
                }

                dt = -1.0/Wsum*log(1.0-ran1(seed));



                /* fitting: update time & compare with experimental fixation duration */
                // don't increment time above fixation duration
                if ( fitting==1 && (t+dt-t_fix)>= (double) tfix[cur_fix] ) {
                    fitting = 2;

                    // instead of simulated duration of nonlabile phase use its mean duration "tau_n"
                    dt_nlab = tau_n;

                    // set end-time of fixation according to data
                    t = t_fix + (double) tfix[cur_fix];

                    if(cur_fix < Nfix) {

                        // select upcoming target from experimental data
                        next_tar = fwrd[cur_fix+1];

                        // deal with cases of very short initial fixations (sentence wise)
                        if ( n_hist == 0 || (t-t_hist[1]) < dt_nlab ) {
                            trans_point = 1;
                        } else {
                            // find index of activations at transition from labile to nonlabile stage
                            trans_point = n_hist+1;
                            while (t_hist[--trans_point] > (t - dt_nlab));
                        }

                        // compute probabilities for all target words at time of transition
                        selectar(a_hist[trans_point],as,k,NW,gamma,seed,fitting,minact,Ptar);

                        // upcoming letter position
                        if ( fwrd[cur_fix+1]==1 ) {
                            upcoming_l_pos = flet[cur_fix+1];
                        } else {
                            upcoming_l_pos = border[fwrd[cur_fix+1]] - ( len[fwrd[cur_fix+1]] + 1 ) + flet[cur_fix+1];
                        }

                        sacamp = fabs(upcoming_l_pos - kpos);

                        // printf("Sacc %d %d:%lf to %d:%lf\n", cur_fix, k, kpos, next_tar, upcoming_l_pos);

                        // Ptar[i] is now the probability (!) of selecting word i as a target

                        // compute probabilities for letter position, given all target words
                        for ( i=1; i<=NW; i++) {
                            double psac = execsacc(&kpos,&k,&i,view,border,len,NW,s1,s2,r1,r2,seed,fitting,upcoming_l_pos,ocshift,0);
                            // printf("Word %d (%.1lf .. %.1lf .. %.1lf): Ptar = %lf, Psac = e^%lf => %lf\n", i, border[i]-len[i], view[i], border[i], Ptar[i], psac, log(Ptar[i]) + psac);
                            Psac[i] = psac;
                        }

                        // Psac[i] is now the log-probability (!) of landing on kpos given that i was selected as a target


                        // sum probabilities, calculate likelihood
                        if (cur_fix < Nfix) {  // exclude first and last fixation (==1 or 2)
                            // If the sum of all likelihoods is sufficiently high, we can easily sum exp(Ptar[.])

                            for ( i=1; i<=NW; i++) Ptmp[i] = log(Ptar[i]) + Psac[i];

                            Pcomb = logsumexp(Ptmp, NW); // Pcomb now is the spatial loglik for the upcoming fixation location

                            loglik_spat[r] += Pcomb;
                            //if(s==63) printf("LLspat[%d,%d] = %lf, loglik_spat[1..%d] = %lf \n", s, cur_fix, Pcomb, cur_fix, loglik_spat);

                        }

                        // Is the current fixation a refixation?
                        refix_sw_cur = 0;
                        if ( ( cur_fix>1 ) && (fwrd[cur_fix]==fwrd[cur_fix-1] ) ) refix_sw_cur = 1;

                        refix_sw_prev = 0;
                        if ( ( cur_fix>2 ) && (fwrd[cur_fix-1]==0) && ( fwrd[cur_fix-1]==fwrd[cur_fix-2] ) ) refix_sw_prev = 1;

                        prev_mlp = mlp;

                        // printf("%d %d %f\n", s, cur_fix, Pmisloc);

                        // Note: Calculate mlp for current fixation before updating
                        // This mlp value is the value for the current fixaion

                        mlp = Pmisloc;
                        if(mlp < 0.0) mlp = 0.0;
                        if(mlp > 1.0) mlp = 1.0;

                        // While mlp now is the P(misloc) for the current fixation, set the mislocation prob
                        // for the next fixation by determining the landing position likelihood based on the
                        // current selection/landing probabilities

                        if( cur_fix < Nfix ) {
                            // set Pmisloc for next fixation
                            Pmisloc = 1.0-exp(Psac[next_tar]-Pcomb)*Ptar[next_tar]; // P(mislocated fixation) = P(not v|x) = 1 - P(v|x) = 1 - P(x|v) * P(v) / P(x)
                            sum_Pmisloc += Pmisloc;
                            // printf("P(misloc|%d,%d)=1-P(x|v)/P(x)*P(v)=1-exp(%.2f-%.2f)*%.2f=%.2f\n", cur_fix, next_tar, Ptar[next_tar], Pcomb, Ptar[next_tar], Pmisloc);
                        }

                        // if ( k==1 )  mlp = misprob*sq(sq((kpos-0.5*len[k])/(0.5*len[k])));
                        // else mlp = misprob*sq(sq((kpos-border[k-1]-0.5*len[k])/(0.5*len[k])));
                        // if ( mlp>misprob )  mlp=misprob;
                        // sum_Pmisloc += mlp/misprob;







                        // compute probability for fixation duration
                        if ( cur_fix > 1 && cur_fix < Nfix ) { // exclude first and last fixation
                            x = (t - t_fix);
                            ptheo = logliktimer(x,cord,lord,nord,xord,msac,tau_l,tau_n,tau_ex,misfac,refix_sw_cur,refix_sw_prev,0,refix,sacamp,kappa0,kappa1,t,t_fix,inhvec,n_trans,nsims,2,seed,mlp,prev_mlp);
                            loglik_temp[r] += ptheo;
                        }
                    }


                    // initiate saccade
                    n_count[1] = 0;
                    n_count[2] = 0;
                    n_count[3] = N_count[3];
                    n_count[4] = 0;

                    // prevent anything else from happening
                    for ( i=1; i<=4+NW; i++) W[i] = 0.0;

                } else {
                    if ( fitting > 0 ) {
                        n_trans++;
                        // check that inhvec is long enough, otherwise double its size
                        ensure_dmatrix_rows(inhvec, max_inhvec_len, n_trans, max_inhvec_len, 4); // Added by Max, Feb 27, 2019
                        inhvec[n_trans][1] = t;
                        inhvec[n_trans][2] = inhibrate;
                        inhvec[n_trans][3] = labrate;
                        inhvec[n_trans][4] = kapparate;
                    }
                    t += dt; // update timer according to
                }

                /* determine transition by relative transition probability */
                psum = 0.0;
                for ( i=1; i<=4+NW; i++ )  {
                    p_trans[i] = W[i];
                    psum += p_trans[i];
                }
                for ( i=1; i<=4+NW; i++ )  p_trans[i] /= psum;

                /* linear selection algorithm */
                rnd = ran1(seed);
                psum = p_trans[1];
                state = 1;
                while ( psum < rnd )  {
                    state++;
                    psum += p_trans[state];
                }

                /* perform transition     */
                /* oculomotor transitions */
                if ( state<=4 )  n_count[state]++;
                else  {
                    /* transitions in lexical activations */
                    j = state-4;
                    n_count[state] += adir[j];   /* lexical activations are changing in incremental steps of +1 or -1 */

                    /* from lexical to postlexical processing */
                    if ( as[j]==1 && n_count[state]>=N_count[state] )  {
                        as[j] = 2;
                        adir[j] = -1;
                        n_count[state] = N_count[state];
                        // proc_order[1][++orders[1]] = j;
                        
                    }

                    /* processing completed: lexical activation back to zero */
                    if ( as[j]==2 && n_count[state]<=0 )  {
                        as[j] = 3;
                        acompletion[j] = 0;
                        n_count[state] = 0;
                        // proc_order[2][++orders[2]] = j;
                    }

                    if ( as[j]==1 )  adir[j] = 1;

                    /* update activation */
                    aa[j] = ((double) n_count[state])/aord;

                    /* save activation field history */
                    if ( 1 /*fitting > 0*/ ) {
                        n_hist++;
                        // check that history vectors is are long enough, otherwise double their size
                        ensure_dmatrix_rows(a_hist, max_a_hist_len, n_hist, max_a_hist_len, NW); // Added by Max, Feb 27, 2019
                        ensure_dvector_size(t_hist, max_t_hist_len, n_hist, max_t_hist_len); // Added by Max, Feb 27, 2019
                        ensure_imatrix_rows(c_hist, max_c_hist_len, n_hist, max_c_hist_len, NW); // Added by Max, Mar 6, 2020
                        t_hist[n_hist] = t;
                        for ( i=1; i<=NW; i++ )  a_hist[n_hist][i] = aa[i];
                        for ( i=1; i<=NW+4; i++ )  c_hist[n_hist][i] = n_count[i];
                    }
                }

                /* check ongoing nlabile saccade program)  */
                if ( n_count[2]>=N_count[2] && (nlab==1 || sacc == 1))  {
                    n_count[2] = N_count[2]; // labile stage offset cannot start two concurring nonlabile phases
                    labrate = 0.0;
                }

                /* start new labile saccade program */
                if ( n_count[1]>=N_count[1] )  {
                    if ( lab==1 )  canc++;
                    n_count[1] = 0;
                    n_count[2] = 0;
                    t_l = t;
                    lab = 1;
                    labrate = 1.0;
                }

                if (n_count[3] >= N_count[3]) {
                    // initiation of saccade
                    #ifdef P_STOP
                    sum_lexproc = 2.0;
                    sum_postlexproc = 1.0;
                    NW_processed = 0.0;
                    if((is_fitting ? cur_fix : n_fix) < 2) {
                        // stopping probability cannot be >0 before 2nd fixation
                        // -> we cannot stop before 2 fixations, i.e. all sequences
                        // with fewer fixations have LL=-inf!
                        p_stop = 0.0;
                    } else {
                        // abort if all words have been processed
                        for ( i=1; i<=NW; i++ ) {
                            if(as[i] >= 3) {
                                sum_lexproc += N_count[i+4];
                                sum_postlexproc += N_count[i+4];
                                NW_processed ++;
                            } else if(as[i] >= 2) {
                                sum_lexproc += N_count[i+4];
                                sum_postlexproc += fmin(N_count[i+4], fmax(0, N_count[i+4]-n_count[i+4]));
                                NW_processed += 1.0 - (double) n_count[i+4]/N_count[i+4];
                            } else if(as[i] >= 1) {
                                sum_lexproc += N_count[i+4];
                                //sum_lexproc += fmin(N_count[i+4], fmax(0, n_count[i+4]));
                            }
                        }
                        // p_stop = pow((0.5+NW_processed) / (1.0+NW), 10.0);
                        // p_stop = sprob * pow(1.0-sprob, NW-NW_processed); // p_stop(all words processed) = sprob, p_stop(no words processed) -> 0.0 for large #(words)
                        p_stop = pow(sum_postlexproc/sum_lexproc, log(0.5) / log(1.0-1.0/NW));
                        if(p_stop <= DBL_EPSILON) {
                            p_stop = DBL_EPSILON;
                        }
                        //printf("S%dR%dF%d: pstop = %lf * (1-%lf)^(%d-%lf) = %lf\n", s, r, cur_fix, sprob, sprob, NW, NW_processed, p_stop);
                        //printf("S%dR%dF%d: nw %.2lf/%d, act %.1lf/%.1lf=%.1lf\n", s, r, cur_fix, NW_processed, NW, sum_postlexproc, sum_lexproc, p_stop);
                    }
                    #endif
                }

                /* initiate non-labile saccade program */
                if ( nlab==0 && sacc == 0 && n_count[2]>=N_count[2] ) {
                    canc = 0;
                    n_count[2] = 0;
                    t_n = t;
                    nlab = 1;
                    n_count[3] = 0;
                    lab = 0;

                    /* select next target word */
                    if ( fitting <= 0 ) {
                        next_tar = selectar(aa,as,k,NW,gamma,seed,fitting,minact,0);
                        /* non-labile saccade program as a function of saccade length */
                        dist = fabs( view[next_tar] - kpos );
                    }

                }


                /* initiate saccade */
                if ( n_count[3]>=N_count[3] && fitting != 1 ) {
                    nlab = 0;
                    t_i = t;    /* onset of saccade */
                    if (sacc == 1) stop(1, "saccade selfintercept; should not happen??");
                    sacc = 1;
                    n_count[4] = 0;
                    n_count[3] = 0;

                    /* save fixation data in matrix mat() */
                    n_fix++;

                    // Check that output buffer vectors are long enough, otherwise double their size
                    ensure_imatrix_rows(mat, max_mat_len, n_fix, max_mat_len, 3); // Added by Max, Feb 27, 2019
                    ensure_dvector_size(kvec, max_kvec_len, n_fix, max_kvec_len); // Added by Max, Feb 27, 2019
                    ensure_dvector_size(tvec, max_tvec_len, n_fix, max_tvec_len); // Added by Max, Feb 27, 2019

                    mat[n_fix][1] = k_fix;
                    mat[n_fix][2] = (int) (t-t_fix);
                    // if(s < 10) printf("%lf-%lf=%lf\n", t, t_fix, t-t_fix);
                    mat[n_fix][3] = intended;
                    kvec[n_fix] = kpos;
                    tvec[n_fix] = t_fix;
                    num++;
                    if ( intended!=k_fix )  misloc++;
                }

                /* end of saccade execution/start of new fixation */
                if ( n_count[4]>=N_count[4] ) {

                    #ifdef P_STOP
                    if(is_fitting) {
                        if(cur_fix < Nfix) {
                            //printf("Continued after %d<%d: += %.1lf\n", cur_fix, Nfix, log1p(-p_stop));
                            loglik_stop[r] += log1p(-p_stop); // == log(1.0-p_stop)
                        } else {
                            //printf("Stopped after %d=%d: += %.1lf\n", cur_fix, Nfix, log(p_stop));
                            loglik_stop[r] += log(p_stop);
                        }
                        // if(cur_fix >= Nfix-1) {
                        //     printf("%d/%d: %lf %lf (%.1lf/%d)\n", cur_fix, Nfix, p_stop, loglik_stop[cur_fix][r], NW_processed, NW);
                        // }
                        //printf("S%dR%dF%d: lstop += %lf, lstop = %lf, NW_processed = %.2lf/%d, actp = %.0lf/%.0lf\n", s, r, cur_fix, cur_fix < Nfix ? log1p(-p_stop) : log(p_stop), loglik_stop[r], NW_processed, NW, sum_postlexproc, sum_lexproc);
                    } else {
                        last_fixation = ran1(seed) <= p_stop;
                    }
                    #endif


                    intended = next_tar;
                    last = k;          /* last fixation */
                    if ( fitting <= 0 ) {
                        saccerr = execsacc(&kpos,&k,&next_tar,view,border,len,NW,s1,s2,r1,r2,seed,fitting,0,ocshift,0);
                    } else {
                        fitting = 1;
                        cur_fix++;
                    //if(s==80) printf("LLspat[%d,%d,%d,%d]=%lf\n", ix, s, r, cur_fix, loglik_spat[r]);
                    }

                    if(is_fitting && cur_fix < Nfix) {
                        kpos = upcoming_l_pos;
                        saccerr = kpos - view[next_tar];
                        k = fwrd[cur_fix];
                    }



                    t_fix = t;         /* start of fixation */
                    k_fix = k;         /* fixation position */
                    sacc = 0;          /* stop saccade */
                    n_count[4] = 0;

                    /* reset of lexical activation after saccade */
                    for ( i=1; i<=NW; i++ )  {
                        if ( as[i]==1 )  n_count[i+4] = (int) (iota*n_count[i+4]);
                    }


                    if ( fitting==0 ) {

                        /* refixation? */
                        if ( last==k && lab==0 )  {
                            lab = 1;
                            labrate = refix;
                            n_count[2] = 0;
                            n_count[1] = 0;
                            Nrefix++;
                        }

                        /* mislocated fixation: program error-correcting saccade */

                        if ( intended != k )  {
                            Nmisloc++;
                            sum_Pmisloc++;
                            lab = 1;
                            labrate = misfac;
                            n_count[2] = 0;
                            n_count[1] = 0;
                        }

                        // mlp = misprob*sq(sq((kpos-view[k])/(0.5*len[k])));
                        // if ( mlp>misprob )  mlp=misprob;
                        // if ( ran1(seed)<=mlp )  {
                        //     lab = 1;
                        //     labrate = misfac;
                        //     n_count[2] = 0;
                        //     n_count[1] = 0;
                        // }


                        /* waiting for end of nl phase + saccade */
                        if ( n_count[2]>=N_count[2] ) {
                            n_count[2] = 0;
                            lab = 0;
                            nlab = 1;
                        }
                    }
                }


                /* check for end of simulation of trial */
                if ( fitting <= 0 ){

                    #ifdef P_STOP

                        endsw_trial = last_fixation;

                    #else

                        // abort reading if all words have been processed
                        for ( i=1, endsw_trial=1; i<=NW; i++ ) if ( as[i]!=3 ) {
                            endsw_trial = 0;
                            break;
                        }

                    #endif

                } else {
                    #ifdef P_STOP
                    endsw_trial = cur_fix > Nfix;
                    #else
                    endsw_trial = cur_fix >= Nfix;
                    #endif
                }

            } /* end of while loop Gillespie algorithm */

            if ( fitting > 0 ) {
                // advance fixation counter to the first fixation of the upcoming trial/sentence
                cur_fix++;
            }

            #pragma omp atomic update
            Nfix_total += n_fix;

            /* write output (sequence) */
            if ( fitting == 0 ) {
                /* number of fixation in first-pass and second-pass (cnfix) */
                nf1 = ivector(1,NW); /* total number of fixation in first-pass */
                nf2 = ivector(1,NW); /* total number of fixation in second-pass */
                testreg = 0;
                for ( i=1; i<=NW; i++ )  nf1[i] = nf2[i] = 0;
                firstpass = 0;
                for ( i=1; i<=n_fix; i++ ) {
                    k = mat[i][1];
                    if ( i>1 ) {
                        if ( k<mat[i-1][1] )  {
                            if ( mat[i-1][1]>firstpass )  firstpass = mat[i-1][1];
                            testreg = 1;
                        }
                    }
                    if ( k>firstpass ) nf1[k]++;
                    else nf2[k]++;
                }
                /* Sarah, 22.05.12: after extensive testing - count seems correct! */

                /* write output file */
                kv = dvector(1,n_fix);
                n1 = ivector(1,NW);
                n2 = ivector(1,NW);
                for ( i=1; i<=NW; i++ )  n1[i] = n2[i] = 0;
                firstpass = 0;

                // if executed in parallel, the following for loop can only be executed by one worker at a time (thereby preventing scrambled output)
                #pragma omp critical(WriteOutput1)
                if(files != NULL && files->fsim != NULL) for ( i=1; i<=n_fix; i++ ) {
                    k = mat[i][1];   /* fixated word */
                    /* fixation in first or second-pass (xnfix) */
                    if ( i>1 ) {
                        if ( k<mat[i-1][1] )  {
                            /* Sarah, 22.05.12: firstpass = mat[i-1][1];*/
                            if ( mat[i-1][1]>firstpass )  firstpass = mat[i-1][1];
                            /* so that firstpass can never be set back: after long regressions
                             a following period of forward reading can now never be firstpass again*/

                            testreg = 1;
                        }
                    }
                    /*Sarah, 22.05.12: for ( j=1; j<=firstpass; j++ )  n1[j] = nf1[j] = 0;*/
                    for ( j=1; j<=firstpass; j++ )  n1[j] = 0;
                    if ( k>firstpass ) n1[k]++;
                    else n2[k]++;
                    /* Sarah, 22.05.12: after extensive testing - index seems correct! */
                    /* type of preceding saccade */
                    presac = -999;
                    if ( i>1 )  presac = k - mat[i-1][1];
                    /* length of preceding saccade */
                    presaclen = -999.99;
                    if ( i>1 )  presaclen = kvec[i] - kvec[i-1];
                    /* landing position (continuous) */
                    //if ( k==1)  kv[i] = kvec[i] + 1.0;
                    if ( k==1)  kv[i] = kvec[i]; // modified by Ralf on Dec. 23, 2016
                    if ( k>1 )  kv[i] = kvec[i] - border[k-1];
                    /* first or last fixation (1/2)? */
                    w = 0;
                    if ( i==1 )  w = 1;
                    if ( i==n_fix )  w = 2;

                    if ( i==dispfix )  {
                        fprintf(files->fsim,"%d\t%d\t%d\t%.2f\t%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%.2f\t%.2f\n",
                                r,s,k,kv[i],mat[i][2],presac,presaclen,
                                len[k],freq[k],pred[k],w,n1[k],nf1[k],n2[k],nf2[k],testreg,mat[i][3],0,prevact,disptime,tvec[i]);
                    }  else  {
                        fprintf(files->fsim,"%d\t%d\t%d\t%.2f\t%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%.2f\t%.2f\n",
                                r,s,k,kv[i],mat[i][2],presac,presaclen,
                                len[k],freq[k],pred[k],w,n1[k],nf1[k],n2[k],nf2[k],testreg,mat[i][3],0,0.0,0.0,tvec[i]);
                    }
                }

                // If a fixseqin file is requested, populate a swift_sequence object and then call the write_sequence function to write it to the output file
                if((files != NULL && files->fseq != NULL) || dataset != NULL) {
                    swift_trial outseq = new_trial(n_fix);
                    outseq.sentence = s;
                    for ( i=1; i<=n_fix; i++ ) {
                        outseq.fixations[i].fw = mat[i][1];
                        outseq.fixations[i].fl = mat[i][1] > 1 ? kvec[i] - border[mat[i][1]-1] : kvec[i];
                        outseq.fixations[i].tfix = mat[i][2];
                        outseq.fixations[i].tsac = i < n_fix ? tvec[i+1]-tvec[i] - outseq.fixations[i].tfix : 0;
                        outseq.fixations[i].idum = NULL;
                        outseq.fixations[i].ddum = NULL;
                        outseq.fixations[i].cdum = NULL;
                    }
                    // if executed in parallel, the following method can only be executed by one worker at a time (thereby preventing scrambled output)
                    #pragma omp critical(WriteOutput2)
                    if(files != NULL && files->fseq != NULL) write_trial(files->fseq, outseq);
                    if(dataset == NULL) {
                      clear_trial(outseq);
                    } else {
                      dataset->trials[(ix-1)*nread+r] = outseq;
                    }
                }


                // #pragma omp critical(WriteOutput3)
                // if(files != NULL && files->f1 != NULL){
                //     fprintf(files->f1, "%d", s);
                //     for(i=1;i<=NW;i++) {
                //         fprintf(files->f1, "\t%d", proc_order[1][i]);
                //     }
                //     fprintf(files->f1, "\n");
                // }

                // #pragma omp critical(WriteOutput4)
                // if(files != NULL && files->f2 != NULL){
                //     fprintf(files->f2, "%d", s);
                //     for(i=1;i<=NW;i++) {
                //         fprintf(files->f2, "\t%d", proc_order[2][i]);
                //     }
                //     fprintf(files->f2, "\n");
                // }

                #pragma omp critical(WriteOutput5)
                if(files != NULL && files->f3 != NULL){
                    fprintf(files->f3, "%d\t%d\t%d\t%d", s, r, n_hist, NW+4);
                    for(j=1;j<=NW+4;j++) {
                        fprintf(files->f3, "\t%d", N_count[j]);
                    }
                    fprintf(files->f3, "\n");
                    int first_last;
                    for(i=1;i<=n_hist;i++) {
                        if(i==n_hist) first_last = 2;
                        else if(i==1) first_last = 1;
                        else if(i==2) first_last = 0;
                        fprintf(files->f3, "%d\t%d\t%d\t%.1lf", s, r, first_last, t_hist[i]);
                        for(j=1;j<=NW+4;j++) {
                            fprintf(files->f3, "\t%d", c_hist[i][j]);
                        }
                        fprintf(files->f3, "\n");
                    }
                }

                free_dvector(kv,1,n_fix);
                free_ivector(n2,1,NW);
                free_ivector(n1,1,NW);
                free_ivector(nf2,1,NW);
                free_ivector(nf1,1,NW);
            }


            free_ivector(acompletion,1,NW);
            free_dvector(aa,1,NW);
            free_dvector(p_trans,1,4+NW);
            free_dvector(W,1,4+NW);
            free_ivector(n_count,1,4+NW);
            free_dvector(r_count,1,4+NW);
            free_ivector(N_count,1,4+NW);

            // endswitch for realizations of one sentence -> next sentence please
            if ( fitting <= 0 ) {
                r++;
                endsw_sentence = r > nread;
            } else {
                if(cur_fix > Nfix) {
                    r++;
                    cur_fix = 1;
                }
                endsw_sentence = r > nread;
            }


        }

        free_dvector(t_hist,1,max_t_hist_len);
        free_dmatrix(a_hist,1,max_a_hist_len,1,NW);
        free_imatrix(c_hist,1,max_c_hist_len,1,NW);
        free_imatrix(mat,1,max_nfix,1,3);
        free_dmatrix(inhvec,1,max_inhvec_len,1,4);
        free_dvector(tvec,1,max_nfix);
        free_dvector(kvec,1,max_nfix);

        if(is_fitting) {
            // average likelihoods and add then add them to return array
            // The “#pragma omp atomic update” lines ensure that (if executed in parallel) only one worker at a time updates the logliks[.] values

            // int ix;
            // for(ix=1;ix<=Nfix-2;ix++) logliks[1] += loglik_temp[ix]/(Nfix-2);
            // for(ix=1;ix<=Nfix-1;ix++) logliks[2] += loglik_spat[ix]/(Nfix-1);

            int i;

            // average computed likelihoods of preceding trial by number of fixations

            for(i=1;i<=nread;i++) {
                //loglik_temp[i] /= Nfix;
                //loglik_spat[i] /= Nfix;
                //printf("STOP %d %lf\n", i, loglik_stop[i]);
                loglik_comb[i] = loglik_temp[i] + loglik_spat[i];
                #ifdef P_STOP
                //loglik_stop[i] /= Nfix;
                loglik_comb[i] += loglik_stop[i];
                #endif
                //printf("S%dR%d logliks: %lf, %lf, %lf\n", s, r-1, loglik_temp[i], loglik_spat[i], loglik_stop[i]);
            }

            if(single_logliks) {
                int o = (ix-1) * N_LOGLIKS;
                logliks[o]   = logsumexp(loglik_comb, nread) - log((double) nread);
                logliks[o+1] = logsumexp(loglik_temp, nread) - log((double) nread);
                logliks[o+2] = logsumexp(loglik_spat, nread) - log((double) nread);
                #ifdef P_STOP
                logliks[o+3] = logsumexp(loglik_stop, nread) - log((double) nread);
                #endif
            } else {
                #pragma omp atomic update
                logliks[0] += logsumexp(loglik_comb, nread) - log((double) nread);
                #pragma omp atomic update
                logliks[1] += logsumexp(loglik_temp, nread) - log((double) nread);
                #pragma omp atomic update
                logliks[2] += logsumexp(loglik_spat, nread) - log((double) nread);
                #ifdef P_STOP
                #pragma omp atomic update
                logliks[3] += logsumexp(loglik_stop, nread) - log((double) nread);
                #endif
            }

            // for(i=1; i<=Nfix; i++) {
            //     if(nread > 1) {
            //         #pragma omp atomic update
            //         logliks[1] += (logsumexp(loglik_temp[i], nread) - log((double) nread)) / (Nfix);
            //         #pragma omp atomic update
            //         logliks[2] += (logsumexp(loglik_spat[i], nread) - log((double) nread)) / (Nfix);
            //     } else {
            //         #pragma omp atomic update
            //         logliks[1] += loglik_temp[i][1] / (Nfix);
            //         #pragma omp atomic update
            //         logliks[2] += loglik_spat[i][1] / (Nfix);
            //     }
            //     #ifdef P_STOP
            //     #pragma omp atomic update
            //     logliks[3] += (logsumexp(loglik_stop[i], nread) - log((double) nread));
            //     #endif
            // }

            free_dvector(loglik_spat, 1, nread);
            free_dvector(loglik_temp, 1, nread);
            free_dvector(loglik_comb, 1, nread);
            #ifdef P_STOP
            free_dvector(loglik_stop, 1, nread);
            #endif


        }

        // free_ivector(orders, 1, 2);
        // free_imatrix(proc_order, 1, 2, 1, NW);
        free_dvector(Ptmp, 1, NW);
        free_dvector(Ptar, 1, NW);
        free_dvector(Psac, 1, NW);
        free_dvector(procrate,1,NW);
        free_dvector(labv1,1,cord);
        free_dvector(labv2,1,cord);
        free_dvector(nlabv1,1,cord);
        free_dvector(nlabv2,1,cord);
        free_dvector(border,1,NW);
        free_dvector(view,1,NW);
        free_dvector(lex,1,NW);
        free_ivector(adir,1,NW);
        free_ivector(as,1,NW);

        if(is_fitting) {
            free_dvector(flet, 1, Nfix);
            free_ivector(fwrd, 1, Nfix);
            free_ivector(tfix, 1, Nfix);
            free_ivector(tsac, 1, Nfix);
        }


        free_dvector(freq, 1, NW);
        free_ivector(len, 1, NW);
        free_dvector(pred, 1, NW);

    }

    free_dvector(s1, 1, 5);
    free_dvector(s2, 1, 5);
    free_dvector(r1, 1, 5);
    free_dvector(r2, 1, 5);
    free_vector(uint64_t, seeds);

    /* if(verbose) {
        if(is_fitting) {
            printf("Overall mislocation probability: %.3lf\n", sum_Pmisloc/Nfix_total);
        }else {
            printf("Overall mislocation probability: %.3lf\n", sum_Pmisloc/Nfix_total);
        }
    } */


    // if(is_fitting) {
    //     #ifdef P_STOP
    //     logliks[0] = logliks[1] + logliks[2] + logliks[3];
    //     #else
    //     logliks[0] = logliks[1] + logliks[2];
    //     #endif
    // }

    #ifdef CHECK_MEMORY
    check_unfreed_memory();
    #endif



}
