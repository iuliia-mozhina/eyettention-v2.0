/**
 *
 *	Density function and CDF for inverse gamma distribution
 *
 **/


#ifndef __INVGAMMALIKE__

#define __INVGAMMALIKE__

#include "gammalike.c"
#include <math.h>
#include <limits.h>

double invgammaloglike(double x, double a, double b) {
	if(x <= 0.0) return -INFINITY;
	else return a * log(b) - (1.0+a) * log(x) - b/x - lgamma(a);
}

double invgammalike(double x, double a, double b) {
	if(x <= 0.0) return 0.0;
	else return pow(b, a) * pow(x, -1.0-a) * exp(-b/x) / tgamma(a);
}

double invgammacdf(double x, double a, double b) {
	if(x < 0.0) return 0.0;
	else return upper_inc_gamma_reg(a, b/x);
}

double invgammalogcdf(double x, double a, double b) {
	if(x < 0.0) return -INFINITY;
	else return upper_inc_log_gamma_reg(a, b/x);
}

double rtruncinvgammalike(double x, double a, double b, double ux) {
	if(x < 0.0 || x > ux) return 0.0;
	else return invgammalike(x, a, b) / invgammacdf(ux, a, b);
}

double rtruncinvgammaloglike(double x, double a, double b, double ux) {
	if(x < 0.0 || x > ux) return -INFINITY;
	else return invgammaloglike(x, a, b) - invgammalogcdf(ux, a, b);
}

double rtruncinvgammacdf(double x, double a, double b, double ux) {
	if(x < 0.0) return 0.0;
	else if(x > ux) return 1.0;
	else return invgammacdf(x, a, b) / invgammacdf(ux, a, b);
}

double rtruncinvgammalogcdf(double x, double a, double b, double ux) {
	if(x < 0.0) return -INFINITY;
	else if(x > ux) return 0.0;
	else return invgammalogcdf(x, a, b) - invgammalogcdf(ux, a, b);
}


double rinvgamma(double a, double b, RANSEED_TYPE * r) {
	if(a < 1.0) stop(1, "rinvgamma() may not be used for a<1.0 (%lf given)", a);
	return 1.0/rgamma(a, b, r);
}

double rtruncrinvgamma(double a, double b, double ux, RANSEED_TYPE * r) {
	return 1.0/(ltruncrgamma(a, b, 1.0/ux, r));
}

double ltruncrinvgamma(double a, double b, double lx, RANSEED_TYPE * r) {
	return 1.0/(rtruncrgamma(a, b, 1.0/lx, r));
}

double truncrinvgamma(double a, double b, double lx, double ux, RANSEED_TYPE * r) {
	return 1.0/truncrgamma(a, b, 1.0/ux, 1.0/lx, r);
}


#endif
