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

#define SWIFT_VARIANT_EXECSACC "rtinvgamma"

#include "invgammalike.c"
#include "logsumexp.c"

#define sd(type) (s1[type] + s2[type]*d)
#define sre(type) (r1[type] - r2[type]*d)

double execsacc(double *kpos,int *k,int *next_tar,double *view,
                double *border,int *len,int NW,double *s1,double *s2,
                double *r1,double *r2,RANSEED_TYPE *seed, int fitting, double upcoming_letter_pos,double ocshift,int verbose)
{
    double   d, knew, saccerr;
    int      l, type;

    if(ocshift != 0.0) stop(1, "ocshift <> 0.0 is not supported for rtgamma2 saccade execution!");
    
    
    /* 1. Determine saccade type */
    /*    forward saccade */
    type = 1;
    /*    skipping saccade   */
    if ( (*next_tar)>(*k)+1 )  type = 2;
    /*    refixation saccade */
    if ( (*k)==(*next_tar) )  type = 3;
    /*    regressive saccade */
    if ( (*k)>(*next_tar) )  type = 5;

    /* 3. Parameters for Gamma distribution */
    // error is gamma distributed so that E(x) = sre and Var(x) = omn^2
    // ux = upper bound for truncated Gamma distribution, so that x <= ux
    
    
    double ux_r, Var_r, E_r, a_r, b_r, ux_l, Var_l, E_l, a_l, b_l, x0;

    // Note that the gamma distribution is semi-infinite
    // This is to determine in which direction the open end should face, which is to the left for backward saccades and to the right for forward saccades
    int fwsacc = type == 1 || type == 2;
    int bwsacc = type == 5;
    int refix = type == 3;

    const double min_mean = 1.0;

    // Determine the desired properties of the landing position distribution
    // This is independent of the distribution type as long as it has a defined E(x) and Var(x)
    // In some situations, the SRE could pull E(x) below 0.0 (the reference/starting point of the distribution)
    // However, E(x) should not be below 0.5 (center of first letter/white space) because the gamma distribution is not defined for E(x) < 0.0
    if(fwsacc) {
        d = view[(*next_tar)] - (*kpos);
        // For forward saccades (frf & fs & sk)
        Var_r = sq(sd(type));
        E_r = d + sre(type); // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        //E_r = *kpos + sre(refix ? 3 : type) - view[*next_tar]; // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        if(E_r < min_mean) E_r = min_mean;
        ux_r = border[NW] - *kpos;                      // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_r = E_r * E_r / Var_r + 2.0;
        b_r = E_r * E_r * E_r / Var_r + E_r;
    }
    if(bwsacc) {
        d = (*kpos) - view[(*next_tar)];
        // For backward saccades (brf & rg)
        Var_l = sq(sd(type));
        E_l = d - sre(type); // In reference to x0, the expected value E(x) is the center of the word PLUS sre
        if(E_l < min_mean) E_l = min_mean;
        ux_l = *kpos;                                            // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_l = E_l * E_l / Var_l + 2.0;
        b_l = E_l * E_l * E_l / Var_l + E_l;
    }
    if(refix) {
        // For forward saccades (frf)
        d = (border[*next_tar] - *kpos) / 2.0;          // midpoint between gaze and right border of the word
        Var_r = sq(sd(3));
        E_r = d + sre(3);
        if(E_r < min_mean) E_r = min_mean;
        ux_r = border[NW] - *kpos;                      // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_r = E_r * E_r / Var_r + 2.0;
        b_r = E_r * E_r * E_r / Var_r + E_r;
        // For backward saccades (brf)
        if(*next_tar == 1) {
            d = (*kpos - 1.0) / 2.0;                    // midpoint between gaze and left edge of the word (without leading whitespace)
        } else {
            d = (*kpos - border[*next_tar - 1] - 1.0) / 2.0;    // midpoint between gaze and left border of the word (without leading whitespace)
        }
        if(d < 0.0) d = 0.0;
        Var_l = sq(sd(4));
        E_l = d - sre(4);
        if(E_l < min_mean) E_l = min_mean;
        ux_l = *kpos;                                            // The support (valid X) of the distribution is the length of the sentence in letter spaces minus the beginning of the word
        a_l = E_l * E_l / Var_l + 2.0;
        b_l = E_l * E_l * E_l / Var_l + E_l;
    }


    x0 = *kpos;                                   // The reference point is the current fixation location (==kpos)

    double relative_position;

    // relative_position is really only useful for refixations but we'll
    // calculate it anyway, just in case we need it for something else later

    if(*k == 1) {
        relative_position = (*kpos - 1.0) / len[*k];
    } else {
        relative_position = (*kpos - border[*k-1] - 1.0) / len[*k];
    }

    if(*k == 1 && relative_position < 0.5/len[1]) {
        // relative_position should not be 0.0 or very small for intended refixations
        // on the first word. otherwise, backward refixations starting from the leading
        // whitespace before that word would have zero probability because no other
        // saccade type has that direction!
        // so we'll assume 1/(2*wordlen) as the minimum relative position if next_tar=1
        relative_position = 0.5/len[1];
    } else if(relative_position < 0.0) {
        // for all other words, the minimum relpos is 0.0 (also 0.0 for leading whitespace!)
        // therefore, intended backward refixations cannot originate from leading whitespace!
        relative_position = 0.0;
    } else if(relative_position > 1.0) {
        // if we're right of the last letter (should not happen), no intended forward refixations!
        relative_position = 1.0;
    }

    if ( fitting <= 0 ) {
        /* 4. Compute landing position */


        double dx, a, b, ux;
        int fw;



        if((refix && ran1(seed) < relative_position) || bwsacc) {
            // we go left if this is a refixation (refix==1) and U < relative_position OR if a regression (bwsacc==1)
            a = a_l;
            b = b_l;
            ux = ux_l;
            fw = 0;
            //knew = x0 - rtruncrinvgamma(a_l, b_l, ux_l, seed);
        } else if(refix || fwsacc) {
            // we go right if this is a refixation (refix==1) and U >= relative_position OR if a forward/skipping saccade (fwsacc==1)
            a = a_r;
            b = b_r;
            ux = ux_r;
            fw = 1;
            //knew = x0 + rtruncrinvgamma(a_r, b_r, ux_r, seed);
        } else {
            stop(1, "Unknown saccade type!");
        }

        dx = rtruncrinvgamma(a, b, ux, seed);

        if(fw) {
            knew = x0 + dx;
        } else {
            knew = x0 - dx;
        }




        saccerr = knew - view[*next_tar];

        // Which is the target word (first word for which border[k-1] < knew)
        for(l = 1; l < NW && knew > border[l]; l++);

        (*kpos) = knew;
        (*k) = l;
        
    } else {

        #if defined(DISCRETE_LIK)

            double x1, dx;

            #define SWIFT_VARIANT_EXECSACC "rtinvgamma-discrete"


            x0 = floor(*kpos) + 0.5;
            x1 = floor(upcoming_letter_pos) + 0.5;

            if(refix) {
                double cdf_r, cdf_l;
                if(x1 > x0 + 0.5) {
                    dx = x1 - x0;
                    saccerr = log(invgammacdf(dx+0.5, a_r, b_r) - invgammacdf(dx-0.5, a_r, b_r)) - invgammalogcdf(ux_r, a_r, b_r) + log1p(-relative_position);
                } else if(x1 < x0 - 0.5) {
                    dx = x0 - x1;
                    saccerr = log(invgammacdf(dx+0.5, a_l, b_l) - invgammacdf(dx-0.5, a_l, b_l)) - invgammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                } else {
                    cdf_r = invgammalogcdf(0.5, a_r, b_r) - invgammalogcdf(ux_r, a_r, b_r) + log1p(-relative_position);
                    cdf_l = invgammalogcdf(0.5, a_l, b_l) - invgammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                    saccerr = logaddexp(cdf_r, cdf_l);
                }
            } else if(fwsacc) {
                dx = x1 - x0;
                saccerr = log(invgammacdf(dx+0.5, a_r, b_r) - invgammacdf(dx-0.5, a_r, b_r)) - invgammalogcdf(ux_r, a_r, b_r);
            } else if(bwsacc) {
                dx = x0 - x1;
                saccerr = log(invgammacdf(dx+0.5, a_l, b_l) - invgammacdf(dx-0.5, a_l, b_l)) - invgammalogcdf(ux_l, a_l, b_l);
            }

            if(verbose) printf("%d:%.2lf->%d:%.2lf (r=%.2lf) : Pspat = %.0lf\n", *k, x0, *next_tar, upcoming_letter_pos, relative_position, saccerr);


        #else

            #define SWIFT_VARIANT_EXECSACC "rtinvgamma-exact"


            double log_scaling;
            double dx, a, b, ux;
            if(refix) {
                // for formulation of scaling and p_trunc, see above
                // x is guaranteed to be ux_brf > x > ux_frf, see formulation of ux_l (brf) and ux_r (frf) above
                // p_trunc(x) = p(x) / scaling
                // for p_trunc(ux_brf < x < 0) := pG(-x | brf) * relative_position / scaling = log(pG(-x | brf)) + log(relative_position / scaling)
                // for p_trunc(0 < x < ux_frf) := pG(x | frf) * (1 - relative_position) / scaling = log(pG(x | frf)) + log((1 - relative_position) / scaling)
                //double scaling = gammacdf(ux_r, a_r, b_r) * (1 - relative_position) + gammacdf(ux_l, a_l, b_l) * relative_position;
                // log(a+b)=log(a)+log(1+b/a) for a > b
                double log_scaling_right = invgammalogcdf(ux_r, a_r, b_r) + log1p(-relative_position);
                double log_scaling_left = invgammalogcdf(ux_l, a_l, b_l) + log(relative_position);
                //if(verbose) printf("scaling=F(%lf,%lf,%lf)*(1-%lf)+F(%lf,%lf,%lf)*%lf=exp(%lf)\n", ux_r, a_r, b_r, relative_position, ux_l, a_l, b_l, relative_position, log_scaling);
                if(upcoming_letter_pos > x0) {
                    dx = upcoming_letter_pos - x0;
                    a = a_r;
                    b = b_r;
                    ux = ux_r;
                    log_scaling = logaddexp(log_scaling_left, log_scaling_right) - log1p(-relative_position);
                    //saccerr = invgammaloglike(dx, a_r, b_r) + log1p(-relative_position) - log_scaling;
                    //if(verbose) printf("log(f(%lf,%lf,%lf))+log(1-%lf)-log(%lf)=%lf\n", dx, a_r, b_r, relative_position, log_scaling, saccerr);
                } else {
                    dx = x0 - upcoming_letter_pos;
                    a = a_l;
                    b = b_l;
                    ux = ux_l;
                    log_scaling = logaddexp(log_scaling_left, log_scaling_right) - log(relative_position);
                    //saccerr = invgammaloglike(-dx, a_l, b_l) + log(relative_position) - log_scaling;
                    //if(verbose) printf("log(f(-(%lf),%lf,%lf))+log(%lf)-log(%lf)=%lf\n", dx, a_l, b_l, relative_position, log_scaling, saccerr);
                }
                //printf("log(DoubleGamma(%.2lf | a_r=%.2lf, b_r=%.2lf, a_l=%.2lf, b_l=%.2lf, ux_l=%.2lf, ux_r%.2lf, r=%.2lf)) = %lf\n", dx, a_r, b_r, a_l, b_l, ux_l, ux_r, relative_position, saccerr);
            } else {
                if(fwsacc) {
                    dx = upcoming_letter_pos - x0;
                    a = a_r;
                    b = b_r;
                    ux = ux_r;
                    log_scaling = invgammalogcdf(ux_r, a_r, b_r);
                } else if(bwsacc) {
                    // this is a backward saccade, so the direction is reversed
                    dx = x0 - upcoming_letter_pos;
                    a = a_l;
                    b = b_l;
                    ux = ux_l;
                    log_scaling = invgammalogcdf(ux_l, a_l, b_l);
                }  else {
                    stop(1, "Unknown saccade type!");
                }
                //saccerr = rtruncinvgammaloglike(dx, a, b, ux); // saccerr in this case means P(landing_position|intended_word)
                //if(verbose) printf("log(Gamma(%.2lf | a=%.2lf, b=%.2lf, ux=%.2lf)) = %lf\n", dx, a, b, ux, saccerr);
            }



            if(dx <= 0.0) {
                saccerr = -INFINITY;
            } else {
                saccerr = invgammaloglike(dx, a, b) - log_scaling;
                if(verbose) printf("%d:%.2lf->%d:%.2lf (r=%.2lf,d=%.3lf,a=%.1lf,b=%.1lf) : Pspat = %.0lf (/%.0lf)\n", *k, x0, *next_tar, upcoming_letter_pos, relative_position, dx, a, b, saccerr, log_scaling);
            }
        


        #endif

    }

    return saccerr;
}

#undef sq
