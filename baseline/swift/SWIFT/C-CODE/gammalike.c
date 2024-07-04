/**
 *
 *	Density function and CDF for gamma distribution
 *
 **/


#ifndef __GAMMALIKE__

#define __GAMMALIKE__

#include <math.h>
#include <limits.h>
#include "gausslike.c"

#ifndef sq
#define sq(x) ((x)*(x))
#endif

double gammaloglike(double x, double a, double b) {
	if(x < 0.0) return -INFINITY;
	else return a * log(b) - lgamma(a) + (a - 1.0) * log(x) - b * x;
}

double gammalike(double x, double a, double b) {
	if(x < 0.0) return 0.0;
	else if(a > 150) return exp(gammaloglike(x, a, b));
	else return pow(b, a) / tgamma(a) * pow(x, a - 1.0) * exp(-b * x);
}

// double lower_inc_gamma_reg(double s, double z) {
// 	double sum, x, y;
// 	double lim, liminv;
// 	lim = pow(z, s) * exp(-z) / tgamma(s + 1.0);
// 	// For very large s or z, untransformed terms in lim can approach infinity, in those cases use log-transform (see lower_inc_log_gamma_reg) and retransform exponentially
// 	if(isnan(lim) || isinf(lim)) lim = exp(s * log(z) - z - lgamma(s + 1.0));
// 	liminv = 1.0/lim;
// 	int k;
// 	for (k = 1, sum = x = 1.0; ; k++) {
// 		sum += (x *= z / (s + k));
// 		if(sum >= liminv) return 1.0 - DBL_EPSILON;
// 		if (x / sum < 1e-10) break;
// 		if (k > 1000000 && k%100000 == 0) fprintf(stderr, "lower_inc_gamma_reg(%lf, %lf) not done after %d iterations (lim = %lf, sum = %lf)\n", s, z, k, lim, sum);
// 	}
// 	y = lim * sum;
// 	if(y >= 1.0) {
// 		return 1.0 - DBL_EPSILON;
// 	} else {
// 		return y;
// 	}
// }

double lower_inc_gamma_reg(double a, double x);
double upper_inc_gamma_reg(double a, double x);
double lower_inc_log_gamma_reg(double a, double x);
double upper_inc_log_gamma_reg(double a, double x);

double lower_inc_gamma_reg(double a, double x) {
   /* Check parameters */
   if(x==0.0) return(0.0);
   if((x<0.0) || (a<=0.0)) stop(1, "lower_inc_gamma_reg(%lf, %lf) is an invalid call", a, x);
  
   if((x>1.0) && (x>a)) return(1.0 - upper_inc_gamma_reg(a, x));
  
   /* Left tail of incomplete Gamma function:
     x^a * e^-x * Sum(k=0,Inf)(x^k / Gamma(a+k+1)) */
  
   double ans, ax, c, r;
  
   /* Compute  x**a * exp(-x) / Gamma(a) */
   ax=a*log(x) - x - lgamma(a);
   if(ax<-DBL_MAX_10_EXP) return(0.0); // underflow
   ax=exp(ax);
  
   /* power series */
   r=a; c=1.0; ans=1.0;
   do {
     r+=1.0;
     c*=x/r;
     ans+=c;
   } while(c/ans > DBL_EPSILON);
   return(ans*ax/a);
}

double lower_inc_log_gamma_reg(double a, double x) {
   /* Check parameters */
   if(x==0.0) return(-INFINITY);
   if((x<0.0) || (a<=0.0)) stop(1, "lower_inc_gamma_reg(%lf, %lf) is an invalid call", a, x);
  
   if((x>1.0) && (x>a)) return(log1p(-upper_inc_gamma_reg(a, x)));
  
   /* Left tail of incomplete Gamma function:
     x^a * e^-x * Sum(k=0,Inf)(x^k / Gamma(a+k+1)) */
  
   double ans, ax, c, r;
  
   /* Compute  x**a * exp(-x) / Gamma(a) */
   ax=a*log(x) - x - lgamma(a);
   if(ax<-DBL_MAX_10_EXP) return(0.0); // underflow
   ax=exp(ax);
  
   /* power series */
   r=a; c=1.0; ans=1.0;
   do {
     r+=1.0;
     c*=x/r;
     ans+=c;
   } while(c/ans > DBL_EPSILON);
   return(log(ans)+log(ax)-log(a));
}

double upper_inc_gamma_reg(double a, double x) {
   /* Check parameters */
   if((x<0.0) || (a<=0.0)) stop(1, "upper_inc_gamma_reg(%lf, %lf) is an invalid call", a, x);
  
   if((x<1.0) || (x<a)) return(1.0-lower_inc_gamma_reg(a, x));
  
   double ans, ax, c, yc, r, t, y, z;
   double pk, pkm1, pkm2, qk, qkm1, qkm2;
   double big=4.503599627370496E+015;
   double biginv=2.22044604925031308085E-016;
  
   ax = a*log(x) - x - lgamma(a);
   if(ax < -DBL_MAX_10_EXP) return(0.0); // underflow
   ax=exp(ax);
  
   /* continued fraction */
   y=1.0-a; z=x+y+1.0; c=0.0;
   pkm2=1.0; qkm2=x; pkm1=x+1.0; qkm1=z*x;
   ans=pkm1/qkm1;
   do {
     c+=1.0; y+=1.0; z+=2.0;
     yc=y*c; pk=pkm1*z - pkm2*yc; qk=qkm1*z - qkm2*yc;
     if(qk!=0.0) {r=pk/qk; t=fabs((ans-r)/r); ans=r;}
     else t=1.0;
     pkm2=pkm1; pkm1=pk; qkm2=qkm1; qkm1=qk;
     if(fabs(pk)>big) {pkm2*=biginv; pkm1*=biginv; qkm2*=biginv; qkm1*=biginv;}
   } while(t>DBL_EPSILON);
   return(ans*ax);
}

double upper_inc_log_gamma_reg(double a, double x) {
   /* Check parameters */
   if((x<0.0) || (a<=0.0)) stop(1, "upper_inc_gamma_reg(%lf, %lf) is an invalid call", a, x);
  
   if((x<1.0) || (x<a)) return(log1p(-lower_inc_gamma_reg(a, x)));
  
   double ans, ax, c, yc, r, t, y, z;
   double pk, pkm1, pkm2, qk, qkm1, qkm2;
   double big=4.503599627370496E+015;
   double biginv=2.22044604925031308085E-016;
  
   ax = a*log(x) - x - lgamma(a);
   if(ax < -DBL_MAX_10_EXP) return(0.0); // underflow
   ax=exp(ax);
  
   /* continued fraction */
   y=1.0-a; z=x+y+1.0; c=0.0;
   pkm2=1.0; qkm2=x; pkm1=x+1.0; qkm1=z*x;
   ans=pkm1/qkm1;
   do {
     c+=1.0; y+=1.0; z+=2.0;
     yc=y*c; pk=pkm1*z - pkm2*yc; qk=qkm1*z - qkm2*yc;
     if(qk!=0.0) {r=pk/qk; t=fabs((ans-r)/r); ans=r;}
     else t=1.0;
     pkm2=pkm1; pkm1=pk; qkm2=qkm1; qkm1=qk;
     if(fabs(pk)>big) {pkm2*=biginv; pkm1*=biginv; qkm2*=biginv; qkm1*=biginv;}
   } while(t>DBL_EPSILON);
   return(log(ans)+log(ax));
}

double gammacdf(double x, double a, double b) {
	if(x < 0.0) return 0.0;
	else return lower_inc_gamma_reg(a, x * b);
}

double gammalogcdf(double x, double a, double b) {
	if(x < 0.0) return -INFINITY;
	else return lower_inc_log_gamma_reg(a, x * b);
}

double rtruncgammalike(double x, double a, double b, double ux) {
	if(x < 0.0 || x > ux) return 0.0;
	else return gammalike(x, a, b) / gammacdf(ux, a, b);
}

double rtruncgammaloglike(double x, double a, double b, double ux) {
	if(x < 0.0 || x > ux) return -INFINITY;
	else return gammaloglike(x, a, b) - gammalogcdf(ux, a, b);
}

double rtruncgammacdf(double x, double a, double b, double ux) {
	if(x < 0.0) return 0.0;
	else if(x > ux) return 1.0;
	else return gammacdf(x, a, b) / gammacdf(ux, a, b);
}

double rtruncgammalogcdf(double x, double a, double b, double ux) {
	if(x < 0.0) return -INFINITY;
	else if(x > ux) return 0.0;
	else return gammalogcdf(x, a, b) - gammalogcdf(ux, a, b);
}


double rgamma(double a, double b, RANSEED_TYPE * r) {
	/** This is a modification of gsl_ran_gamma from the GSL library **/
	double x, v, u;
	if (a < 1.0) {
		// gsl_ran_gamma (r, 1.0 + a, b) * pow (u, 1.0 / a);
		u = ran1(r);
		return rgamma(1.0 + a, b, r) * pow(u, 1.0 / a);
	}
	double d = a - 1.0 / 3.0;
	double c = (1.0 / 3.0) / sqrt (d);
	while(1) {
		do {
			x = gasdev(r);
			v = 1.0 + c * x;
		} while (v <= 0);
		v = v * v * v;
		u = ran1(r);
		if (u < 1.0 - 0.0331 * x * x * x * x) 
			break;
		if (log (u) < 0.5 * x * x + d * (1.0 - v + log(v)))
			break;
	}
	return d * v / b;
}

double rgammaint(int a, double b, RANSEED_TYPE * r) {
	int k;
	double ret = 0.0;
	for ( k=1; k<=a; k++) ret += log(1.0-ran1(r));
	return ret / -b;
}

double truncrgamma(double a, double b, double lx, double ux, RANSEED_TYPE * r) {
	// Chung, Y. Korean J. Comput. & Appl. Math. (1998) 5: 601. https://doi.org/10.1007/BF03008885
	double x, x0, gx;
	x0 = (a - 1.0) / b;
	do {
		x = lx + ran1(r) * (ux - lx);
		if(x0 >= lx && x0 <= ux && a >= 1.0) {
			gx = pow(x, a - 1.0) * exp(-b * x + a - 1.0) / pow((a - 1.0) / b, a - 1.0);
		} else if(ux < x0 && a >= 1.0) {
			gx = pow(x, a - 1.0) * exp(-b * x + b * ux) / pow(ux, a - 1.0);
		} else {
			assert(x0 < lx && a >= 1.0 || a < 1.0);
			gx = pow(x, a - 1.0) * exp(-b * x + b * lx) / pow(lx, a - 1.0);
		}
	} while(ran1(r) > gx);
	return x;
}

static inline double rexp(double lambda, RANSEED_TYPE * r) {
	return -log(1.0-ran1(r))/lambda;
}

double ltruncrgamma(double a, double b, double lx, RANSEED_TYPE * r) {
	// Based on Chung, Y. Korean J. Comput. & Appl. Math. (1998) 5: 601. https://doi.org/10.1007/BF03008885
	if(a <= 1.0) {
		double x, g, gx, u;
		g = (1.0/lx<b) ? 1.0/lx : b;
		do {
			x = rexp(g, r);
			gx = pow(x, a-1.0) * exp(-(b-g)*x) / (pow(x, a-1.0) * exp(-(b-g)*lx));
			u = ran1(r);
		} while(u > gx);
		return x;
	} else if((a-1.0)/b < lx) {
		double x, gx, u, g, g1;
		g1 = 1.0/(2.0*lx) * (b*lx - a + sqrt(sq(b*lx-a)+4.0*b*lx));
		g = (g1<=b) ? g1 : b;
		do {
			x = rexp(g, r) + lx;
			u = ran1(r);
			gx = pow(x, a-1.0) * exp(-(b-g)*x+(a-1.0)) / (pow((a-1.0)/(b-g), a-1.0));
		} while(u > gx);
		return x;
	} else {
		double x;
		do {
			x = rgamma(a, b, r);
		} while(x < lx);
		return x;
	}
}

double rtruncrgamma(double a, double b, double ux, RANSEED_TYPE * r) {
	// Based on Chung, Y. Korean J. Comput. & Appl. Math. (1998) 5: 601. https://doi.org/10.1007/BF03008885
	double x, x0, gx;
	x0 = (a - 1.0) / b;
	do {
		x = ran1(r) * ux;
		if(x0 <= ux && a >= 1.0) {
			gx = pow(x, a - 1.0) * exp(-b * x + a - 1.0) / pow((a - 1.0) / b, a - 1.0);
		} else if(ux < x0 && a >= 1.0) {
			gx = pow(x, a - 1.0) * exp(-b * x + b * ux) / pow(ux, a - 1.0);
		} else {
			gx = pow(x, a - 1.0) * exp(-b * x);
		}
	} while(ran1(r) > gx);
	return x;
}


#endif
