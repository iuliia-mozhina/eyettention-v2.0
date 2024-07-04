/*------------------------------------------------------
 Execute saccade
 (Version 1.0, 05-AUG-02)
 (Version 2.0, 12-NOV-03)
 (Version 2.1, 06-APR-16) Stefan Seelig: if fitting <= 0 return sacerr & update kpos, next_tar, else return P(landing_position|intended_word)
 -----------------------------------------------------*/

/*
 *kpos             == letter position in sentence
 *next_tar         == word number of intended saccade target
 view[(*next_tar)] == OVP of intended saccade target
 knew              == actual letter position after saccade with distance-dependent error
 */

#define SWIFT_VARIANT_EXECSACC "gauss"

double execsacc(double *kpos,int *k,int *next_tar,double *view,
                double *border,int *len,int NW,double *s1,double *s2,
                double *r1,double *r2,RANSEED_TYPE *seed, int fitting, double upcoming_letter_pos,double ocshift,int verbose)
{
    double   dist, d, dx, knew, sd, sre, saccerr;
    int      l, w, type;
    
    dist = view[(*next_tar)] - (*kpos);
    
    /* 1. Determine saccade type */
    /*    forward saccade */
    type = 1;
    /*    skipping saccade   */
    if ( (*next_tar)>(*k)+1 )  type = 2;
    /*    refixation saccade */
    if ( (*k)==(*next_tar) )  {
        if ( dist>=0.0 )
            type = 3;  /* forward refixation */
        else
            type = 4; /* backward refixation */
    }
    /*    regressive saccade */
    if ( (*k)>(*next_tar) )  type = 5;
    
    
    /* 2. Oculomotor parameters */
    d = fabs(dist);
    //d = dist;
    sd = s1[type] + s2[type]*d;   /* oculomotor noise    */
    //sre = r2[type]*(r1[type] - d);  /* saccade range error */ // McConkie's Version
    sre = r1[type] - r2[type]*d;  /* saccade range error */ // changed by Ralf on Dec. 27, 2016
    
    if ( fitting <= 0 ) {
        /* 3. Compute landing position */
        knew = view[(*next_tar)] + ocshift + sre + sd*gasdev(seed);
        saccerr = knew - view[(*next_tar)];
        for ( w=1, l=1; w<=NW-1; w++ )  if ( knew >= border[w] ) l++;
        if ( knew >= border[NW] )  knew = border[NW]-0.5;
        if ( knew<=1.0 )  knew = 1.5;
        (*kpos) = knew;
        (*k) = l;
        
    } else {
        dx = upcoming_letter_pos - (view[(*next_tar)] + sre + ocshift);
        saccerr = 1.0/(sqrt(2.0*M_PI)*sd)*exp(-((dx*dx)/(2.0*sd*sd))); // saccerr in this case means P(landing_position|intended_word)
    }
    return saccerr;
}

