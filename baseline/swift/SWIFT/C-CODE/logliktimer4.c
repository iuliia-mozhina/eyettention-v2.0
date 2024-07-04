/*--------------------------------------------
 Computation of log likelihood for fixation durations
 ----------------------------------------------*/
#define sq(A) ((A)*(A))
#define pi 3.1415926536
#define realmin 1e-10
#define EPSILON 0.0000001

#include "gammalike.c"

double logliktimer(double x,int cord,int lord,int nord,int xord,double msac,double tau_l,double tau_n,double tau_x,double misfac,
                    int refix_sw_cur,int refix_sw_prev, int misfac_sw, double refix,double sacamp,double kappa0,double kappa1,
                    double t,double t_fix,double **inhvec,int ntrans,int nruns,int n_fix,RANSEED_TYPE *seed, double mlp, double prev_mlp)
{
    double      *fd;
    double      loglik;
    double      r_timer, r_tau_l, r_tau_n, r_tau_x, labrate;
    double      t1, tn1, tl1, tx1, t2, tn2, tl2, dt2;
    double      fix2_onset, fix2_offs;
    int         i, k;
    int         *refixvec, *misfixvec;

    // inhibition
    int         ix, ix_curfixstart;
    double      ti, ti_last, dti, dtisum, inhibfac, step;

    int attempts;
    const int max_attempts = 100;

    for(attempts = 0; attempts < max_attempts; attempts++) {

        if(attempts % 5 == 0 && attempts > 0) {
            warn("logliktimer has not returned a value after %d attempts! Program will die after %d unsuccessful attempts!", attempts, max_attempts);
        }

       /* base rates */ 
        r_timer = cord/(1.0*msac);
        r_tau_l = lord/(1.0*tau_l);
        r_tau_n = nord/(1.0*tau_n);
        r_tau_x = xord/(1.0*tau_x);
      
        // refixation markers
        refixvec = ivector(1,n_fix);
        for (i=1; i<=n_fix; i++) refixvec[i] = 0; // initialize
        refixvec[n_fix] = refix_sw_cur;
        if (n_fix >= 2) refixvec[n_fix-1] = refix_sw_prev;

        misfixvec = ivector(1,n_fix);


        // compute distribution
        fd = dvector(1,nruns);
        //fout = fopen("out.dat","w");
        for ( i=1; i<=nruns; i++ )  {

            labrate = 1.0;
            misfixvec[1] = ran1(seed) <= prev_mlp;
            misfixvec[2] = ran1(seed) <= mlp;

            // timer 1

            // for ( k=1, tn1=0.0; k<=nord; k++) tn1 += log(1.0-ran1(seed));
            // tn1 /= -r_tau_n;
            tn1 = rgammaint(nord, r_tau_n, seed);
            
            // for ( k=1, tx1=0.0; k<=xord; k++) tx1 += log(1.0-ran1(seed));
            // tx1 /= -r_tau_x;
            tx1 = rgammaint(xord, r_tau_x, seed);

            if (refixvec[1] == 1) {
                labrate = refix;
            }
            if (misfixvec[1] == 1) {
                labrate = misfac;
            }

            t1 = 0.0; 
            tl1 = 1.0;
            while ( t1<=tl1 ) {

                // for ( k=1, tl1=0.0; k<=lord; k++) tl1 += log(1.0-ran1(seed));
                // tl1 /= -r_tau_l*labrate;
                tl1 = rgammaint(lord, r_tau_l*labrate, seed);

                labrate = 1.0;

                ti = t_fix-tx1-tn1-tl1;
                ti_last = ti-1.0;
                t1 = 0.0; // timers are not relative

                /* adjust ix and ix_curfixstart indices */
                for (ix=ntrans; ix>1 && inhvec[ix][1]>=ti; ix--) {
                    if (inhvec[ix][1]>= t_fix) ix_curfixstart = ix;
                }
                for ( k=1; k<=cord; k++)  {
                    dtisum = 0.0;
                    inhibfac = 0.0;
                    if (ti - inhvec[ix][1] < EPSILON) {
                        // no inhibition available for the full interval
                        inhibfac = 1.0;
                        dtisum = 1.0;
                    } else {
                        while (((ti_last+dtisum) - ti) < -EPSILON) {
                            
                            if (((ti_last+dtisum) - inhvec[ix][1]) < -EPSILON) {
                                // if t < t(first recorded inhibition)
                                dti = inhvec[ix][1]-(ti_last+dtisum);
                                inhibfac += 1.0*dti;
                            } else { 
                                // if t > t(last recorded inhibition)
                                if ((ix+1) <= ntrans) {
                                    // if t <= last inhibition value
                                    if (EPSILON >= (inhvec[ix+1][1] - ti)) {
                                        dti = inhvec[ix+1][1]-(ti_last+dtisum);
                                    } else {
                                        dti = ti-(ti_last+dtisum);
                                    }
                                    ix++; 
                                    inhibfac += inhvec[ix][2]*dti;
                                } else {
                                    dti = ti-(ti_last+dtisum);
                                    inhibfac += inhvec[ix][2]*dti;
                                }
                            }
                            dtisum += dti;
                        }
                        if (ix > 1 ) ix--;
                    }
                    inhibfac /= dtisum;
                    do { // for computational reasons step must not be 0 or very very small
                        step = log(1.0-ran1(seed))/-r_timer/inhibfac;
                    } while (step<EPSILON);
                    ti_last = ti;
                    ti += step;
                    t1 += step;
                }
            }

            fix2_onset = tl1 + tn1 + tx1;
        
        // timer 2
            dt2 = 0.0;
            t2 = 0.0; 
            tl2 = 1.0;
            if (refixvec[2] == 1) {
                dt2 = fix2_onset - t1;
                labrate = refix;
            }
            if (misfixvec[2] == 1) {
                dt2 = fix2_onset - t1;
                labrate = misfac;
            }

            while ( t2<=tl2 || ((t1+dt2+tl2)<(fix2_onset)) ) {
                dt2 = dt2 + t2;

                // for ( k=1, tl2=0.0; k<=lord; k++) tl2 += log(1.0-ran1(seed));
                // tl2 /= -r_tau_l*labrate;
                tl2 = rgammaint(lord, r_tau_l*labrate, seed);

                labrate = 1.0;
                
                t2 = 0.0;
                for ( k=1; k<=cord; k++)  {
                    dtisum = 0.0;
                    inhibfac = 0.0;
                    if (ti - inhvec[ix][1] < EPSILON) {
                        // no inhibition available for the full interval
                        inhibfac = 1.0;
                        dtisum = 1.0;
                    } else {
                        while (((ti_last+dtisum) - ti) < -EPSILON) {
                            
                            if (((ti_last+dtisum) - inhvec[ix][1]) < -EPSILON) {
                                // if t < t(first recorded inhibition)
                                dti = inhvec[ix][1]-(ti_last+dtisum);
                                inhibfac += 1.0*dti;
                            } else { 
                                // if t > t(last recorded inhibition)
                                if ((ix+1) <= ntrans) {
                                    // if t <= last inhibition value
                                    if (EPSILON >= (inhvec[ix+1][1] - ti)) {
                                        dti = inhvec[ix+1][1]-(ti_last+dtisum);
                                    } else {
                                        dti = ti-(ti_last+dtisum);
                                    }
                                    ix++; 
                                    inhibfac += inhvec[ix][2]*dti;
                                } else {
                                    dti = ti-(ti_last+dtisum);
                                    inhibfac += inhvec[ix][2]*dti;
                                }
                            }
                            dtisum += dti;
                        }
                        if (ix > 1 ) ix--;
                    }
                    inhibfac /= dtisum;
                    do { // for computational reasons step must not be 0 or very very small
                        step = log(1.0-ran1(seed))/-r_timer/inhibfac;
                    } while (step<EPSILON);
                    ti_last = ti;
                    ti += step;
                    t2 += step; 
                }            
                if ( (t1+dt2+tl2)<fix2_onset ) tl2 = fix2_onset - (t1 + dt2);
            }

            // for ( k=1, tn2=0.0; k<=nord; k++) tn2 += log(1.0-ran1(seed));
            // tn2 /= -r_tau_n;
            tn2 = rgammaint(nord, r_tau_n, seed);
        
            fix2_offs = t1 + dt2 + tl2 + tn2;

            fd[i] = fix2_offs - fix2_onset; 

            if ( fd[i]<=0.0 ) stop(1, "logliktimer3: simulated fixation duration below zero");
            //fprintf(fout,"%f\n",fd[i]);
        }
        free_ivector(refixvec,1,n_fix);
        free_ivector(misfixvec,1,n_fix);
        // loglik = log(gausslike(fd,nruns,x));
        //loglik = log(epallike(fd,nruns,x));
        loglik = logepallike(fd,nruns,x);
        //loglik = log(hist(fd,50,nruns,x));
        //fclose(fout);

        // char fn[60];
        // sprintf(fn, "out%06d.dat", (int) (ran1(seed)*1000000));

        // fout = fopen(fn, "w");
        // for(i=1;i<=nruns;i++) {
        //     fprintf(fout, "%lf\n", fd[i]);
        // }
        // fclose(fout);

        free_dvector(fd,1,nruns);

        if (!isnan(loglik)) {
            return loglik; 
        }

        // fout = fopen("out2.dat","w");
        // for (int asd=1; asd<=800; asd++) {
        //     loglik = gausslike(fd,nruns,asd);
        //     fprintf(fout,"1\t%d\t%f\n",asd,loglik);
        //     loglik = epallike(fd,nruns,asd);
        //     fprintf(fout,"2\t%d\t%f\n",asd,loglik);
        //     loglik = hist(fd,50,nruns,asd);
        //     fprintf(fout,"3\t%d\t%f\n",asd,loglik);
        // }
        // fclose(fout);
        //nrerror("stoopp");

    }
    
    stop(1, "logliktimer returned nan after %d attempts. Exiting!", attempts+1);
 
}

#undef sq
#undef pi
#undef realmin
#undef EPSILON