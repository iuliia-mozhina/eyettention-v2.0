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

int selectar(double *aa,int *as,int k,int NW,double gamma,RANSEED_TYPE *seed, int fitting, double minact, double *Ptar)
{
    double    *p;
    double    psum, r, test;
    int       j, i, tar, ik, d;

    double a;


    /* compute activations and target selection probabilities */
    p = dvector(1,NW);
    psum = 0.0;
    for (j=1; j<=NW; j++) {
        // p(select) is dependent on activation, gamma, and threshold (minact)
        a = aa[j];

        #if defined(SUB1)

            if(j > k) {
                for(i = k; i < j; i++) {
                    a -= aa[i];
                }
            }
            #if defined(MREG)
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-subtract-k-regs"
            else if(j < k) {
                for(i = k; i > j; i--) {
                    a -= aa[i];
                }
            }
            #else
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-subtract-k"
            #endif

        #elif defined(SUB2)

            if(j > k) {
                for(i = k+1; i < j; i++) {
                    a -= aa[i];
                }
            }
            #if defined(MREG)
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-subtract-nok-regs"
            else if(j < k) {
                for(i = k-1; i > j; i--) {
                    a -= aa[i];
                }
            }
            #else
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-subtract-nok"
            #endif


        #elif defined(DIV1)


            if(j > k) {
                for(i = k; i < j; i++) {
                    a *= 1.0-aa[i];
                }
            }
            #if defined(MREG)
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-divide-k-regs"
            else if(j < k) {
                for(i = k; i > j; i--) {
                    a *= 1.0-aa[i];
                }
            }
            #else
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-divide-k"
            #endif


        #elif defined(DIV2)

            if(j > k) {
                for(i = k+1; i < j; i++) {
                    a *= 1.0-aa[i];
                }
            }
            #if defined(MREG)
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-divide-k-regs"
            else if(j < k) {
                for(i = k-1; i > j; i--) {
                    a *= 1.0-aa[i];
                }
            }
            #else
            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-divide-k"
            #endif


        #elif defined(NORMAL)

            #define SWIFT_VARIANT_SELECTAR "dist-keepclose-off"

        #else

            #error "Must use NORMAL, SUB1, SUB2, DIV1 or DIV2"

        #endif

        if(a < 0.0) {
            a = 0.0;
        }

        //printf("%d to %d: %lf - %lf\n", k, j, aa[j], a);

        p[j] = pow(a + exp(minact), gamma);

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
