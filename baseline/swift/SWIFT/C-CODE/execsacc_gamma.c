/*------------------------------------------------------
 Execute saccade
 (Version 1.0, 05-AUG-02)
 (Version 2.0, 12-NOV-03)
 (Version 2.1, 06-APR-16) Stefan Seelig: if fitting <= 0 return sacerr & update kpos, next_tar, else return P(landing_position|intended_word)
 (Version 3.0, 23-APR-19) Maximilian Rabe: assume gamma-distributed landing position with E(x) ~ view[.] + sre and Var(x) = omn^2
 -----------------------------------------------------*/

/*
 *kpos             == letter position in sentence
 *next_tar         == word number of intended saccade target
 view[(*next_tar)] == OVP of intended saccade target
 knew              == actual letter position after saccade with distance-dependent error
 */


#define sq(x) ((x) * (x))

#define SWIFT_VARIANT_EXECSACC "rtgamma2"

#include "gammalike.c"

#define sd(type) (s1[type] + s2[type]*d)
#define sre(type) (r1[type] - r2[type]*d)

double execsacc(double *kpos,int *k,int *next_tar,double *view,
                double *border,int *len,int NW,double *s1,double *s2,
                double *r1,double *r2,RANSEED_TYPE *seed, int fitting, double upcoming_letter_pos,double ocshift,int verbose)
{
    double   dist, d, knew, saccerr;
    int      l, w, type;

    if(ocshift != 0.0) stop(1, "ocshift <> 0.0 is not supported for rtgamma2 saccade execution!");
    
    dist = view[(*next_tar)] - (*kpos);
    
    /* 1. Determine saccade type */
    /*    forward saccade */
    type = 1;
    /*    skipping saccade   */
    if ( (*next_tar)>(*k)+1 )  type = 2;
    /*    refixation saccade */
    if ( (*k)==(*next_tar) )  type = 3;
    /*    regressive saccade */
    if ( (*k)>(*next_tar) )  type = 5;
    
    
    /* 2. Oculomotor parameters */
    d = fabs(dist);

    /* 3. Parameters for Gamma distribution */
    // error is gamma distributed so that E(x) = sre and Var(x) = omn^2
    // ux = upper bound for truncated Gamma distribution, so that x <= ux
    
    
    double ux_r, Var_r, E_r, a_r, b_r, ux_l, Var_l, E_l, a_l, b_l, x0;

    // Note that the gamma distribution is semi-infinite
    // This is to determine in which direction the open end should face, which is to the left for backward saccades and to the right for forward saccades
    int fwsacc = type == 1 || type == 2;
    int bwsacc = type == 5;
    int refix = type == 3;

    // Determine the desired properties of the landing position distribution
    // This is independent of the distribution type as long as it has a defined E(x) and Var(x)
    // In some situations, the SRE could pull E(x) below 0.0 (the reference/starting point of the distribution)
    // However, E(x) should not be below 0.5 (center of first letter/white space) because the gamma distribution is not defined for E(x) < 0.0
    if(fwsacc || refix) {
        // For forward saccades (frf & fs & sk)
        Var_r = sq(sd(refix ? 3 : type));
        E_r = view[*next_tar] + sre(refix ? 3 : type) - *kpos; // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        //E_r = *kpos + sre(refix ? 3 : type) - view[*next_tar]; // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        if(E_r < 0.5) E_r = 0.5;
        ux_r = border[NW] - *kpos;                      // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_r = sq(E_r) / Var_r;
        b_r = E_r / Var_r;
    }
    if(bwsacc || refix) {
        // For backward saccades (brf & rg)
        Var_l = sq(sd(refix ? 4 : type));
        E_l = *kpos - (view[*next_tar] + sre(refix ? 4 : type)); // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        if(E_l < 0.5) E_l = 0.5;
        ux_l = *kpos;                                            // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_l = sq(E_l) / Var_l;
        b_l = E_l / Var_l;
    }

    x0 = *kpos;                                   // The reference point is the current fixation location (==kpos)

    double relative_position;

    if(*k == 1) {
        relative_position = *kpos / (1.0 + len[1]);
    } else {
        relative_position = (*kpos - border[*k - 1]) / (1.0 + len[*k]);
    }

    if ( fitting <= 0 ) {
        /* 4. Compute landing position */

        if(refix) {
            // For non-truncated Gamma distribution (pG = gamma density, PG = gamma cumulative density):
            // p(x) = pG(-x | brf) * relative_position + pG(x | frf) * (1 - relative_position)
            // for p(x < 0) := pG(-x | brf) * relative_position
            // for p(x > 0) := pG(x | frf) * (1 - relative_position)
            // for p(x = 0) := 0
            // P(x) = (1 - PG(-x | brf)) * relative_position + PG(x | frf) * (1 - relative_position)
            // for P(x < 0) := (1 - PG(-x | brf)) * relative_position
            // for P(x > 0) := relative_position + PG(x | frf) * (1 - relative_position)
            // for P(x = 0) := relative_position
            // For truncated Gamma distribution (scaling = total probability between ux_brf and ux_frf, ux_brf < 0, ux_frf > 0):
            // scaling = P(ux_frf) - P(ux_brf) = relative_position + PG(ux_frf | frf) * (1 - relative_position) - (1 - PG(-ux_brf | brf)) * relative_position = PG(ux_frf | frf) * (1 - relative_position) + PG(-ux_brf | brf) * relative_position
            // p_trunc(x) = p(x) / scaling
            // for p_trunc(ux_brf < x < 0) := pG(-x | brf) * relative_position / scaling
            // for p_trunc(0 < x < ux_frf) := pG(x | frf) * (1 - relative_position) / scaling
            // for p_trunc(x < ux_brf || x > ux_frf || x = 0) := 0
            // P_trunc(x) = (P(x) - P(ux_brf)) / scaling
            // for P_trunc(x < ux_brf) := 0
            // for P_trunc(x > ux_frf) := 1
            // for P_trunc(x = 0) := (P(0) - P(ux_brf)) / scaling = (relative_position - P(ux_brf)) / scaling = relative_position * PG(-ux_brf | brf) / scaling
            // generate brf if R < P_trunc(0) for R ~ Unif(0,1), otherwise perform frf
            //double scaling = gammacdf(ux_r, a_r, b_r) * (1 - relative_position) + gammacdf(ux_l, a_l, b_l) * relative_position;
            // log(a+b)=log(a)+log(1+b/a) for a > b
            
            double log_scaling_left = gammalogcdf(ux_r, a_r, b_r) + log(1 - relative_position);
            double log_scaling_right = gammalogcdf(ux_l, a_l, b_l) + log(relative_position);
            double log_scaling;
            if(log_scaling_left > log_scaling_right)
                log_scaling = log_scaling_left + log1p(exp(log_scaling_right-log_scaling_left));
            else
                log_scaling = log_scaling_right + log1p(exp(log_scaling_left-log_scaling_right));
            //if(ran1(seed) < relative_position * gammacdf(ux_l, a_l, b_l) / scaling) {
            if(log(ran1(seed)) < log(relative_position) + gammalogcdf(ux_l, a_l, b_l) - log_scaling) {
                // perform a backward refixation
                knew = x0 - rtruncrgamma(a_l, b_l, ux_l, seed);
            } else {
                // perform a forward refixation
                knew = x0 + rtruncrgamma(a_r, b_r, ux_r, seed);
            }
        } else if(fwsacc) {
            knew = x0 + rtruncrgamma(a_r, b_r, ux_r, seed);
        } else if(bwsacc) {
            // this is a backward saccade, so the direction is reversed
            knew = x0 - rtruncrgamma(a_l, b_l, ux_l, seed);
        } else {
            stop(1, "Unknown saccade type!");
        }

        saccerr = knew - view[*next_tar];

        // Which is the target word (first word for which border[k-1] < knew)
        for(l = 1; l < NW && knew > border[l]; l++);

        (*kpos) = knew;
        (*k) = l;
        
    } else {

        #if defined(DISCRETE_LIK)
            #define SWIFT_VARIANT_EXECSACC "rtgamma-discrete"

            double x1, dx;


            x0 = floor(*kpos) + 0.5;
            x1 = floor(upcoming_letter_pos) + 0.5;

            if(refix) {
                double cdf_r, cdf_l;
                if(x1 > x0 + 0.5) {
                    dx = x1 - x0;
                    saccerr = log(gammacdf(dx+0.5, a_r, b_r) - gammacdf(dx-0.5, a_r, b_r)) - gammalogcdf(ux_r, a_r, b_r) + log1p(-relative_position);
                } else if(x1 < x0 - 0.5) {
                    dx = x0 - x1;
                    saccerr = log(gammacdf(dx+0.5, a_l, b_l) - gammacdf(dx-0.5, a_l, b_l)) - gammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                } else {
                    cdf_r = gammalogcdf(0.5, a_r, b_r) - gammalogcdf(ux_r, a_r, b_r) + log1p(-relative_position);
                    cdf_l = gammalogcdf(0.5, a_l, b_l) - gammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                    saccerr = logaddexp(cdf_r, cdf_l);
                }
            } else if(fwsacc) {
                dx = x1 - x0;
                saccerr = log(gammacdf(dx+0.5, a_r, b_r) - gammacdf(dx-0.5, a_r, b_r)) - gammalogcdf(ux_r, a_r, b_r);
            } else if(bwsacc) {
                dx = x0 - x1;
                saccerr = log(gammacdf(dx+0.5, a_l, b_l) - gammacdf(dx-0.5, a_l, b_l)) - gammalogcdf(ux_l, a_l, b_l);
            }

            if(verbose) printf("%d:%.2lf->%d:%.2lf (r=%.2lf) : Pspat = %.0lf\n", *k, x0, *next_tar, upcoming_letter_pos, relative_position, saccerr);



        #else
            #define SWIFT_VARIANT_EXECSACC "rtgamma-exact"


            if(refix) {
                // for formulation of scaling and p_trunc, see above
                // x is guaranteed to be ux_brf > x > ux_frf, see formulation of ux_l (brf) and ux_r (frf) above
                // p_trunc(x) = p(x) / scaling
                // for p_trunc(ux_brf < x < 0) := pG(-x | brf) * relative_position / scaling = log(pG(-x | brf)) + log(relative_position / scaling)
                // for p_trunc(0 < x < ux_frf) := pG(x | frf) * (1 - relative_position) / scaling = log(pG(x | frf)) + log((1 - relative_position) / scaling)
                double dx = upcoming_letter_pos - x0;
                //double scaling = gammacdf(ux_r, a_r, b_r) * (1 - relative_position) + gammacdf(ux_l, a_l, b_l) * relative_position;
                // log(a+b)=log(a)+log(1+b/a) for a > b
                double log_scaling_left = gammalogcdf(ux_r, a_r, b_r) + log(1 - relative_position);
                double log_scaling_right = gammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                double log_scaling;
                if(log_scaling_left > log_scaling_right)
                    log_scaling = log_scaling_left + log1p(exp(log_scaling_right-log_scaling_left));
                else
                    log_scaling = log_scaling_right + log1p(exp(log_scaling_left-log_scaling_right));
                if(verbose) printf("scaling=F(%lf,%lf,%lf)*(1-%lf)+F(%lf,%lf,%lf)*%lf=exp(%lf)\n", ux_r, a_r, b_r, relative_position, ux_l, a_l, b_l, relative_position, log_scaling);
                if(dx > 0.0) {
                    saccerr = gammaloglike(dx, a_r, b_r) + log(1 - relative_position) - log_scaling;
                    if(verbose) printf("log(f(%lf,%lf,%lf))+log(1-%lf)-log(%lf)=%lf\n", dx, a_r, b_r, relative_position, log_scaling, saccerr);
                } else {
                    saccerr = gammaloglike(-dx, a_l, b_l) + log(relative_position) - log_scaling;
                    if(verbose) printf("log(f(-(%lf),%lf,%lf))+log(%lf)-log(%lf)=%lf\n", dx, a_l, b_l, relative_position, log_scaling, saccerr);
                }
                if(verbose) printf("log(DoubleGamma(%.2lf | a_r=%.2lf, b_r=%.2lf, a_l=%.2lf, b_l=%.2lf, ux_l=%.2lf, ux_r%.2lf, r=%.2lf)) = %lf\n", dx, a_r, b_r, a_l, b_l, ux_l, ux_r, relative_position, saccerr);
            } else {
                double dx, a, b, ux;
                if(fwsacc) {
                    dx = upcoming_letter_pos - x0;
                    a = a_r;
                    b = b_r;
                    ux = ux_r;
                } else if(bwsacc) {
                    // this is a backward saccade, so the direction is reversed
                    dx = x0 - upcoming_letter_pos;
                    a = a_l;
                    b = b_l;
                    ux = ux_l;
                }  else {
                    stop(1, "Unknown saccade type!");
                }
                saccerr = rtruncgammaloglike(dx, a, b, ux); // saccerr in this case means P(landing_position|intended_word)
                if(verbose) printf("log(Gamma(%.2lf | a=%.2lf, b=%.2lf, ux=%.2lf)) = %lf\n", dx, a, b, ux, saccerr);
            }

        #endif
        

    }

    return saccerr;
}

#undef sq
