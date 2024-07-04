/*--------------------------------------------
    Likelihood-Evaluation of SWIFT III

    parallel execution of gaengine with subsetted fixation sequences; using MPI
 =========================================
 (Version 1.0, 26-AUG-02)    
 (Version 2.0, 23-JUN-03)
 (Version 3.0, 10-DEC-03)
 (Version 4.0, 25-JUN-04)
 (Version 4.1, 19-MAY-07)
 (Version 5.0, 07-OCT-11)
 ----------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <limits.h>

#include "swiftstat7_api.c"

#define verbose_mpi 0

int MSG_CLOSE = 99;
int MSG_INIT = 2;
int MSG_PARAM = 3;
int MSG_EVAL = 4;
int MSG_EVAL_SINGLE = 5;
int MSG_EVAL_SINGLE_MANY = 6;
#define TAG_LIK 333
int MSG_VERSION = 1;
int VERSION = 1001;

MPI_Comm mpi_comm;

MPI_Request mpi_request;
int mpi_rank, mpi_size, mpi_model_loaded = 0;
swift_model *sw_model = NULL;
swift_dataset *sw_data = NULL;
int * sw_trials = NULL;
int loglik_evals;
int * stack_load, * stack_count, ** stacks;

int swift_init_mpi() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_comm = MPI_COMM_WORLD;
    mpi_model_loaded = 0;
    return 1;
}

int swift_is_mpi_slave() {
    return mpi_rank > 0;
}

void swift_mpi_free() {
    free_swift_model(sw_model);
    free_swift_dataset(sw_data);
    if(!swift_is_mpi_slave()) {
        free(stack_count);
        free(stack_load);
        int i;
        for(i=0;i<mpi_size;i++) {
            free(stacks[i]);
        }
        free(stacks);
    }
}

int swift_finalize_mpi() {
    if(mpi_model_loaded) {
        swift_mpi_free();
    }
    if(!swift_is_mpi_slave()) {
        MPI_Bcast(&MSG_CLOSE, 1, MPI_INT, 0, mpi_comm);
    }
    MPI_Finalize();
    return 1;
}

int swift_mpi_slave() {
    int msg = 0;
    int idum, ntrial, i;
    double ddum;
    long super_seed;

    int my_stack_count;


    do {

        MPI_Bcast(&msg, 1, MPI_INT, 0, mpi_comm);
        //MPI_Recv(&msg, 1, MPI_INT, 0, TAG_CMD, mpi_comm, MPI_STATUS_IGNORE);
        if(verbose_mpi) printf("\tMPI (%d/%d): BCAST %d RECVD\n", mpi_rank+1, mpi_size, msg);
        if(msg == MSG_CLOSE) {
            // 0 means we want to close this instance
            printf("Disconnecting from MPI (%d/%d)...\n", mpi_rank+1, mpi_size);
            //MPI_Comm_disconnect(&mpi_comm);
            //printf("\tDisconnected from MPI (%d/%d).\n", mpi_rank+1, mpi_size);
            //MPI_Comm_disconnect(&mpi_comm);
            break;
        }

        if(msg == MSG_INIT) {

            if(mpi_model_loaded) {
                swift_mpi_free();
            }

            char environmentPath[PATH_MAX], parmFile[PATH_MAX], corpusFile[PATH_MAX], fixseqFile[PATH_MAX];

            memset(environmentPath, 0, PATH_MAX);
            memset(parmFile, 0, PATH_MAX);
            memset(corpusFile, 0, PATH_MAX);
            memset(fixseqFile, 0, PATH_MAX);

            MPI_Bcast(environmentPath, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
            MPI_Bcast(parmFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
            MPI_Bcast(corpusFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
            MPI_Bcast(fixseqFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
            MPI_Scatter(NULL, 1, MPI_LONG, &super_seed, 1, MPI_LONG, 0, mpi_comm);
            MPI_Bcast(&loglik_evals, 1, MPI_INT, 0, mpi_comm);
            MPI_Recv(&my_stack_count, 1, MPI_INT, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
            sw_trials = calloc(sizeof(int), my_stack_count+1);
            MPI_Recv(sw_trials, my_stack_count, MPI_INT, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
            sw_trials[my_stack_count] = 0;

            // int swift_load_model(char *environmentPath, char *parmFile, char *corpusFile, uint64_t seed, swift_model **dat, int verbose)

            if(!swift_load_model(environmentPath, parmFile, corpusFile, super_seed, &sw_model, 0)) {
                exit(1);
            }

            if(!swift_load_data(fixseqFile, &sw_data, 0)) {
                exit(1);
            }

            setval(sw_model->params, runs, 1);

            printf("MPI process %d/%d initiated with envpath %s, param file %s, corpus %s, fixseq file %s, seed %lu\n\n", mpi_rank+1, mpi_size, environmentPath, parmFile, corpusFile, fixseqFile, ranlong(&sw_model->seed));

            mpi_model_loaded = 1;
        } else if(msg == MSG_PARAM) {
            int parid;
            MPI_Bcast(&parid, 1, MPI_INT, 0, mpi_comm);
            if(swift_parameters_meta[parid].type == PARTYPE_INTEGER) {
                MPI_Bcast(&idum, 1, MPI_INT, 0, mpi_comm);
                if(verbose_mpi) printf("MPI (%d/%d) sets par %s (#%d) to %d.\n", mpi_rank+1, mpi_size, swift_parameters_meta[parid].name, parid, idum);
                setvalbyid(sw_model->params, parid, swift_parameter_int, idum);
            } else if(swift_parameters_meta[parid].type == PARTYPE_DOUBLE) {
                MPI_Bcast(&ddum, 1, MPI_DOUBLE, 0, mpi_comm);
                if(verbose_mpi) printf("MPI (%d/%d) sets par %s (#%d) to %lf.\n", mpi_rank+1, mpi_size, swift_parameters_meta[parid].name, parid, ddum);
                setvalbyid(sw_model->params, parid, swift_parameter_dbl, ddum);
            } else {
                printf("\tMPI (%d/%d) FATAL ERROR: Request received to set parameter %d but parameter type is unknown.\n", mpi_rank+1, mpi_size, parid);
                exit(1);
            }
        } else if(msg == MSG_EVAL) {
            /* SWIFT fitting to fixation sequences */
            double ret[N_LOGLIKS];

            if(verbose_mpi) printf("%d/%d next random number %lu, start at %d\n", mpi_rank+1, mpi_size, ranlong(&sw_model->seed), sw_trials[0]);

            swift_eval(sw_model, sw_data, sw_trials, ret, 0, 0);

            if(verbose_mpi) printf("Result %d/%d: %lf\n", mpi_rank+1, mpi_size, ret[0]);

            MPI_Reduce(ret, NULL, N_LOGLIKS, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
            
        } else if(msg == MSG_EVAL_SINGLE) {
            /* SWIFT fitting to fixation sequences */
            double ret[N_LOGLIKS*my_stack_count];
            MPI_Request mpi_request;

            if(verbose_mpi) printf("%d/%d next random number %lu, start at %d\n", mpi_rank+1, mpi_size, ranlong(&sw_model->seed), sw_trials[0]);

            swift_single_eval(sw_model, sw_data, sw_trials, ret, 0, 0);

            MPI_Isend(ret, N_LOGLIKS*my_stack_count, MPI_DOUBLE, 0, 0, mpi_comm, &mpi_request);

            if(verbose_mpi) printf("%d/%d complete\n", mpi_rank+1, mpi_size);

            MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
            
        } else if(msg == MSG_EVAL_SINGLE_MANY) {
            /* SWIFT fitting to fixation sequences */
            double ret[N_LOGLIKS];
            int trial;

            MPI_Recv(&trial, 1, MPI_INT, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
            while(trial > 0) {
                if(verbose_mpi) printf("%d/%d next random number %lu, start at %d\n", mpi_rank+1, mpi_size, ranlong(&sw_model->seed), trial);
                swift_eval_single(sw_model, sw_data, trial, ret, 0, 0);
                MPI_Sendrecv(ret, N_LOGLIKS, MPI_DOUBLE, 0, 0, &trial, 1, MPI_INT, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
            }
            
        }

    } while(1);

    
    
    return 1;
}

void swift_update_parameter_mpi(int parid, ...) {
    MPI_Bcast(&MSG_PARAM, 1, MPI_INT, 0, mpi_comm);
    MPI_Bcast(&parid, 1, MPI_INT, 0, mpi_comm);
    va_list args;
    va_start(args, parid);
    if(swift_parameters_meta[parid].type == PARTYPE_INTEGER) {
        int idum = va_arg(args, int);
        MPI_Bcast(&idum, 1, MPI_INT, 0, mpi_comm);
        setvalbyid(sw_model->params, parid, swift_parameter_int, idum);
    } else if(swift_parameters_meta[parid].type == PARTYPE_DOUBLE) {
        double ddum = va_arg(args, double);
        MPI_Bcast(&ddum, 1, MPI_DOUBLE, 0, mpi_comm);
        setvalbyid(sw_model->params, parid, swift_parameter_dbl, ddum);
    } else {
        printf("\tMPI (%d/%d) FATAL ERROR: Request received to set parameter %d but parameter type is unknown.\n", mpi_rank+1, mpi_size, parid);
        exit(1);
    }
    va_end(args);
}

void swift_load_mpi(char *environmentPath, char *parmFile, char *corpusFile, char *fixseqFile, long super_seed, int how_many_evals) {
    RANSEED_TYPE ran;
    initSeed(super_seed, &ran);
    long ranseeds[mpi_size], my_seed;
    int i,j;
    for(i=0;i<mpi_size;i++) {
        ranseeds[i] = ranlong(&ran);
    }
    MPI_Bcast(&MSG_INIT, 1, MPI_INT, 0, mpi_comm);
    MPI_Bcast(environmentPath, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
    MPI_Bcast(parmFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
    MPI_Bcast(corpusFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
    MPI_Bcast(fixseqFile, PATH_MAX-1, MPI_CHAR, 0, mpi_comm);
    MPI_Scatter(&my_seed, 1, MPI_LONG, ranseeds, 1, MPI_LONG, 0, mpi_comm);
    MPI_Bcast(&how_many_evals, 1, MPI_INT, 0, mpi_comm);
    swift_load_model(environmentPath, parmFile, corpusFile, my_seed, &sw_model, 0);
    swift_load_data(fixseqFile, &sw_data, 0);

    if(how_many_evals > 0) {
        loglik_evals = how_many_evals;
    } else {
        loglik_evals = val(sw_model->params, runs);
    }
    setval(sw_model->params, runs, 1);

    int n_trials = ntrials(sw_data) * loglik_evals;
    int trials_to_distribute[n_trials], trial_ids[n_trials];
    stack_count = calloc(sizeof(int), mpi_size);
    stack_load = calloc(sizeof(int), mpi_size);
    stacks = calloc(sizeof(int*), mpi_size);
    for(i=0;i<n_trials;i++) {
        trials_to_distribute[i] = 1;
        trial_ids[i] = i/loglik_evals+1;
    }
    for(i=0;i<mpi_size;i++) {
        stack_load[i] = 0;
        stack_count[i] = 0;
        stacks[i] = calloc(sizeof(int), mpi_size < n_trials ? n_trials - mpi_size + 1 : 1);
    }
    int longest_trial, smallest_stack;
    do {
        longest_trial = -1;
        for(i=0;i<n_trials;i++) {
            if(trials_to_distribute[i] && (longest_trial == -1 || nfixations(sw_data, trial_ids[longest_trial]) < nfixations(sw_data, trial_ids[i]))) {
                longest_trial = i;
            }
        }
        if(longest_trial != -1) {
            smallest_stack = 0;
            for(i=1;i<mpi_size;i++) {
                if(stack_load[i] < stack_load[smallest_stack]) {
                    smallest_stack = i;
                }
            }
            stacks[smallest_stack][stack_count[smallest_stack]++] = trial_ids[longest_trial];
            stack_load[smallest_stack] += nfixations(sw_data, trial_ids[longest_trial]);
            trials_to_distribute[longest_trial] = 0;
        }
    } while(longest_trial != -1);

    for(i=0;i<mpi_size;i++) {
        stacks[i] = realloc(stacks[i], sizeof(int) * (stack_count[i]+1));
        stacks[i][stack_count[i]] = 0;
        if(i > 0) {
            MPI_Send(&stack_count[i], 1, MPI_INT, i, 0, mpi_comm);
            MPI_Send(stacks[i], stack_count[i], MPI_INT, i, 0, mpi_comm);
        } else {
            sw_trials = stacks[0];
        }
        printf("MPI process %d/%d will evaluate %d trials (%d fixations total): ", i+1, mpi_size, stack_count[i], stack_load[i]);
        for(j=0;j<stack_count[i];j++) {
            if(j>0) printf(",");
            printf("%d", stacks[i][j]);
        }
        printf("\n");
    }
    mpi_model_loaded = 1;

    printf("MPI process %d/%d initiated with envpath %s, param file %s, corpus %s, fixseq file %s, seed %lu\n", mpi_rank+1, mpi_size, environmentPath, parmFile, corpusFile, fixseqFile, my_seed);

}

// void swift_eval_mpi(double *logliks) {
//     //printf("Start evaluation...\n");
//     MPI_Bcast(&MSG_EVAL, 1, MPI_INT, 0, mpi_comm);
//     double my_loglik[N_LOGLIKS];
//     swift_eval(sw_model, sw_data, sw_trials, my_loglik, 0, 0);
//     if(verbose_mpi) printf("Result %d/%d: %lf\n", mpi_rank+1, mpi_size, my_loglik[0]);
//     MPI_Reduce(my_loglik, logliks, N_LOGLIKS, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
// }

void swift_eval_mpi(double *logliks) {
    //printf("Start evaluation...\n");
    int i, j, k;
    MPI_Bcast(&MSG_EVAL_SINGLE, 1, MPI_INT, 0, mpi_comm);
    double single_logliks[N_LOGLIKS][ntrials(sw_data)][loglik_evals];
    int single_loglik_counts[N_LOGLIKS][ntrials(sw_data)];
    for(i=0;i<N_LOGLIKS;i++) {
        for(j=0;j<ntrials(sw_data);j++) {
            single_loglik_counts[i][j] = 0;
        }
    }
    for(i=0;i<mpi_size;i++) {
        double worker_logliks[N_LOGLIKS * stack_count[i]];
        if(i == 0) {
            swift_single_eval(sw_model, sw_data, sw_trials, worker_logliks, 0, 0);
        } else {
            MPI_Recv(worker_logliks, N_LOGLIKS * stack_count[i], MPI_DOUBLE, i, 0, mpi_comm, MPI_STATUS_IGNORE);
        }
        for(j=0;j<N_LOGLIKS;j++) {
            for(k=0;k<stack_count[i];k++) {
                int trial_id = stacks[i][k]-1;
                single_logliks[j][trial_id][single_loglik_counts[j][trial_id]++] = worker_logliks[j+N_LOGLIKS*k];
            }
        }
    }
    for(i=0;i<N_LOGLIKS;i++) {
        logliks[i] = 0.0;
        for(j=0;j<ntrials(sw_data);j++) {
            if(single_loglik_counts[i][j] != loglik_evals) {
                warn("For trial #%d, only %d logliks were recorded but there should be %d!", j+1, single_loglik_counts[i][j], loglik_evals);
            }
            logliks[i] += logsumexp(single_logliks[i][j], single_loglik_counts[i][j]) - log((double) single_loglik_counts[i][j]);
        }
    }
}

void swift_eval_mpi_lb(double * logliks) {
    MPI_Bcast(&MSG_EVAL_SINGLE_MANY, 1, MPI_INT, 0, mpi_comm);
    int i, j, k, done;
    int n_available_workers = mpi_size - 1;
    int jobs_left = ntrials(sw_data);
    int * available_workers = vector(int, n_available_workers);
    int * workers_available = vector(int, n_available_workers);
    MPI_Request * statuses = vector(MPI_Request, n_available_workers);
    double buf[n_available_workers][N_LOGLIKS];
    for(i=1;i<=n_available_workers;i++) {
        available_workers[i] = i;
        workers_available[i] = 1;
    }
    for(i=0;i<N_LOGLIKS;i++) {
        logliks[i] = 0.0;
    }
    i = 1;
    int complete;
    while(done < ntrials(sw_data)) {
        while(n_available_workers > 0 && jobs_left > 0) {
            int worker = available_workers[n_available_workers];
            n_available_workers--;
            workers_available[worker] = 0;
            if(verbose_mpi) {
                printf("Worker %d is busy with job %d.\n", worker, i);
                if(n_available_workers > 0) {
                    printf("%d workers available: ", n_available_workers);
                    for(j = 1; j <= n_available_workers; j++) {
                        if(j > 1) printf(", ");
                        printf("%d", available_workers[j]);
                    }
                    printf("\n");
                } else {
                    printf("No more workers available.\n");
                }
                fflush(stdout);
            }
            MPI_Irecv(buf[worker-1], N_LOGLIKS, MPI_DOUBLE, worker, 0, mpi_comm, &statuses[worker]);
            MPI_Send(&i, 1, MPI_INT, worker, 0, mpi_comm);
            i++;
            jobs_left--;
        }
        if(n_available_workers < mpi_size) {
            for(j = 1; j <= mpi_size-1; j++) {
                if(!workers_available[j]) {
                    MPI_Test(&statuses[j], &complete, MPI_STATUS_IGNORE);
                    if(complete) {
                        if(verbose_mpi) {
                            printf("Worker %d is idle.\n", j);      
                            fflush(stdout);
                        }
                        for(k=0;k<N_LOGLIKS;k++) {
                            logliks[k] += buf[j-1][k];
                        }
                        workers_available[j] = 1;
                        available_workers[++n_available_workers] = j;
                        done++;
                    }
                }
            }
        }
    }
    j=0;
    for(i=1;i<mpi_size;i++) {
        MPI_Send(&j, 1, MPI_INT, i, 0, mpi_comm);
    }
    free_vector(int, available_workers);
    free_vector(int, workers_available);
    free_vector(MPI_Request, statuses);
}


#ifdef SWIFT_MPI_MAIN
int main(int argn, char *argv[])
{
    //printf("\tThis is MPI rank %d of %d with %d parallel OpenMP threads.\n", mpi_rank+1, mpi_size, omp_threads);
    
    swift_init_mpi();

    if(swift_is_mpi_slave()) {
        swift_mpi_slave();
    }else{

        fprintf(stderr, "Note: You are running %s as a master process. This is only advisable for infrastructure checks. Please use the R or Python module as the master process and this process as a slave.\n", argv[0]);

        int i;


        long seed;
        int how_many_evals;


        if(argn != 7 || sscanf(argv[5], "%lu", &seed) != 1 || sscanf(argv[6], "%d", &how_many_evals) != 1) {
            fprintf(stderr, "Error: This command takes exactly 6 arguments in this order: environment path (containing swiftstat.inp), parameter file, corpus file, fixseq file, random seed, num evals\n");
            exit(1);
        }

        swift_load_mpi(argv[1], argv[2], argv[3], argv[4], seed, how_many_evals);

        printf("Initiated %d MPI processes.\n", mpi_size);

        double logliks[N_LOGLIKS];

        swift_eval_mpi_lb(logliks);

        for(i=0;i<N_LOGLIKS;i++) {
            if(i>0) printf(", ");
            printf("%lf", logliks[i]);
        }
        printf("\n");

        swift_update_parameter_mpi(par_eta, 0.1);

        swift_eval_mpi_lb(logliks);

        for(i=0;i<N_LOGLIKS;i++) {
            if(i>0) printf(", ");
            printf("%lf", logliks[i]);
        }
        printf("\n");

        swift_update_parameter_mpi(par_eta, 0.6);

        swift_eval_mpi_lb(logliks);

        for(i=0;i<N_LOGLIKS;i++) {
            if(i>0) printf(", ");
            printf("%lf", logliks[i]);
        }
        printf("\n");

    }

    swift_finalize_mpi();


    if(!swift_is_mpi_slave()) {
        printf("All %d MPI processes are finalized. Good bye.\n", mpi_size);
    }
    
}
#endif
