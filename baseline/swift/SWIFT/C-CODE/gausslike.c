#ifndef __GAUSSLIKE__

#define __GAUSSLIKE__

#include <math.h>
#include <limits.h>

double gasdev(RANSEED_TYPE * r) {
	// Marsaglia polar method (https://en.wikipedia.org/wiki/Marsaglia_polar_method)
	if  (r->gasdev_iset == 0) {
		double p,q,u,v;
		do {
			u=2.0*ran1(r)-1.0;
			v=2.0*ran1(r)-1.0;
			q=u*u+v*v;
		} while (q >= 1.0 || q == 0.0);
		p=sqrt(-2.0*log(q)/q);
		r->gasdev_gset=u*p;
		r->gasdev_iset=1;
		return v*p;
	} else {
		r->gasdev_iset=0;
		return r->gasdev_gset;
	}
}

double gaussloglike(double x, double mu, double sigma) {
	return -0.5*log(2.0*M_PI*sigma*sigma)-(x-mu)*(x-mu)/(2.0*sigma*sigma);
}

double gausslike(double x, double mu, double sigma) {
	return 1.0/sqrt(2.0*M_PI*sigma*sigma)*exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));
}

double stdgausscdf(double x) {
	return 0.5 * erfc(-x*M_SQRT1_2);
}

double stdgausslogcdf(double x) {
	return -M_LN2 + log(erfc(-x*M_SQRT1_2));
}

double gausscdf(double x, double mu, double sigma) {
	return stdgausscdf((mu-x)/sigma);
}

double gausslogcdf(double x, double mu, double sigma) {
	return stdgausslogcdf((mu-x)/sigma);
}

double rgauss(double mu, double sigma, RANSEED_TYPE * r) {
	return gasdev(r) * sigma + mu;
}

#endif