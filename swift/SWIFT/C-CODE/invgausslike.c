/**
 *
 *	Density function and CDF for gamma distribution
 *
 **/


#ifndef __INVGAUSSLIKE__

#define __INVGAUSSLIKE__

#include <math.h>
#include <limits.h>
#include "gausslike.c"
#include "logsumexp.c"

double invgaussloglike(double x, double mu, double lambda) {
	if(x <= 0.0) return -INFINITY;
	return 0.5*log(lambda/(2.0*M_PI*x*x*x))+(-lambda*(x-mu)*(x-mu)/(2.0*mu*mu*x));
}

double invgausslike(double x, double mu, double lambda) {
	if(x <= 0.0) return 0.0;
	return sqrt(lambda/(2.0*M_PI*x*x*x))*exp(-lambda*(x-mu)*(x-mu)/(2.0*mu*mu*x));
}

double invgausscdf(double x, double mu, double lambda) {
	if(x <= 0.0) return 0.0;
	else return stdgausscdf(sqrt(lambda/x)*(x/mu-1.0)) + exp(2.0*lambda/mu) * stdgausscdf(-sqrt(lambda/x)*(x/mu+1.0));
}

double invgausslogcdf(double x, double mu, double lambda) {
	if(x <= 0.0) return -INFINITY;
	double z[2] = {
		stdgausslogcdf(sqrt(lambda/x)*(x/mu-1.0)),
		2.0*lambda/mu + stdgausslogcdf(-sqrt(lambda/x)*(x/mu+1.0))
	};
	return logsumexp(z-1, 2);
}

double rtruncinvgausslike(double x, double mu, double lambda, double ux) {
	if(x <= 0.0 || x > ux) return 0.0;
	else return invgausslike(x, mu, lambda) / invgausscdf(ux, mu, lambda);
}

double rtruncinvgaussloglike(double x, double mu, double lambda, double ux) {
	if(x <= 0.0 || x > ux) return -INFINITY;
	else return invgaussloglike(x, mu, lambda) - invgausslogcdf(ux, mu, lambda);
}

double rtruncinvgausscdf(double x, double mu, double lambda, double ux) {
	if(x <= 0.0) return 0.0;
	else if(x > ux) return 1.0;
	else return invgausscdf(x, mu, lambda) / invgausscdf(ux, mu, lambda);
}

double rtruncinvgausslogcdf(double x, double mu, double lambda, double ux) {
	if(x <= 0.0) return -INFINITY;
	else if(x > ux) return 0.0;
	else return invgausslogcdf(x, mu, lambda) - invgausslogcdf(ux, mu, lambda);
}

double rinvgauss(double mu, double lambda, RANSEED_TYPE * r) {
	// https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
	double y, x, z;
	y = gasdev(r);
	y *= y;
	x = mu + mu*mu*y/(2.0*lambda) - mu/(2.0*lambda)*sqrt(4.0*mu*lambda*y+mu*mu*y*y);
	z = ran1(r);
	if(z <= mu / (mu+x)) {
		return x;
	}else {
		return mu*mu/x;
	}
}

double rtruncrinvgauss(double mu, double lambda, double ux, RANSEED_TYPE * r) {
	double z;
	do {
		z = rinvgauss(mu, lambda, r);
	} while(z > ux);
	return z;
}


#endif
