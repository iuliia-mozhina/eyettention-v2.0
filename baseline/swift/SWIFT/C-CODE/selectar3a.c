/*---------------------------------------------------
 Target selection according to activation field
 (Version 1.0, 06-Nov-00)
 (Version 2.0, 15-JUL-02)
 (Version 3.0, 19-FEB-10)
 (Version 3.1, 06-Apr-16) 
 (Version 3.2, 11_Nov-16) 
 (Version 3.3, 14_Jul_17)
 (Version 3.4, 26_Mar_18)
 -----------------------------------------------------*/
#define sq(A) ((A)*(A))
#define SWIFT_VARIANT_SELECTAR "dist-dep"

int selectar(double *aa,int *as,int k,int NW,double gamma,RANSEED_TYPE *seed, int fitting, double minact, double *Ptar)
{
    double    *p;
    double    psum, r, test;
    int       j, tar = 1, ik, d;
    double minact_l, minact_r;

    /* compute activations and target selection probabilities */

    p = dvector(1,NW);

    // get index of leftmost unfinished word
    for (ik = 1; ik < NW; ik++) {
        if (as[ik] < 3) break;
    }

    //ik = k;

    minact_r = minact;
    //minact_r = minact/2.0;
    minact_l = minact;
    
    psum = 0.0;
    for (j=1; j<=NW; j++) {
        // compute distances (word scale) from this word
        if (j!=ik) {
            d = abs(ik-j);
        } else {
            d = 1;
        }

        // p(select) is dependent on activation, gamma, and threshold (minact) depending on ik
        p[j] = pow(aa[j] + exp((j<k?minact_l:minact_r) * (double) d), gamma);
        psum += p[j];
    }

    for (j=1; j<=NW; j++) p[j] /= psum;
    
    if ( fitting <= 0 ) {
        /* linear selection algorithm */
        for(tar = 1, test = p[1], r = ran1(seed); test < r; ) {
            tar++;
            test += p[tar];
        }
    } else {
        for ( j=1; j<=NW; j++ ) Ptar[j] = p[j];
    }

    free_dvector(p,1,NW);
    return tar;
}

#undef sq