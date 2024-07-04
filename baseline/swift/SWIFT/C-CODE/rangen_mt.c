
#ifndef __RANGEN_MT__

#ifdef __RANGEN__
#error You can only include one random number generator!
#endif

#define __RANGEN__
#define __RANGEN_MT__
#define SWIFT_VARIANT_RANGEN "mt"

#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <limits.h>

#define SFMT_MEXP 19937
#include "./SFMT-1.5.1/SFMT.c"
#undef SFMT_MEXP

typedef struct {

	sfmt_t sfmt;
	int gasdev_iset;
	double gasdev_gset;

} RANSEED_TYPE;

#define RANSEED_MAXSEED (((unsigned long) ULONG_MAX)-1)

static inline double randouble(RANSEED_TYPE * seed) {
	return sfmt_genrand_real2(&seed->sfmt);
}

#define ran1(seed) randouble(seed)

static inline uint64_t ranlong(RANSEED_TYPE * seed) {
	return sfmt_genrand_uint64(&seed->sfmt);
}

void initSeed(uint64_t seed, RANSEED_TYPE * gen) {
	sfmt_init_by_array(&gen->sfmt, (uint32_t*) &seed, 2);
	gen->gasdev_iset = 0;
}

#endif
