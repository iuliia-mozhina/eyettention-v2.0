

#ifndef LOGSUMEXPDEF

#define LOGSUMEXPDEF

#include <math.h>
#include <stdio.h>

// This function returns the log of the sum of the exponentials of a vector, i.e. log(sum(exp(.)))
// This is useful when calculating the sum of probabilities for which only the log-probabilities are known
double logsumexp(double * logs, size_t N) {
	if(N <= 0) {
		// there is no summand, therefore the sum is exp(-inf)!
		return -INFINITY;
	} else if(N == 1) {
		// there is only one summand, hence the sum is that number
		return logs[1];
	} else {
		// find maximum value
		size_t i, imax = 1;
		for(i = 2; i <= N; i++) {
			if(logs[imax] < logs[i]) {
				imax = i;
			}
		}
		if(isinf(logs[imax])) {
			// if the greatest value if exp(-inf)=0.0, all other values (if any) will also be exp(-inf)=0.0, hence the sum is exp(-inf)=0.0
			// if the greatest value is exp(inf)=inf, the sum will be exp(inf), no matter what the other summands are because there can't be any negative exp(.)
			return logs[imax];
		} else {
			// if no special case applies, apply the log summation rule
			double sum;
			for(i = 1, sum = 0.0; i <= N; i++) {
				if(i == imax || logs[i] == -INFINITY) continue;
				sum += exp(logs[i] - logs[imax]);
			}
			return logs[imax] + log1p(sum);
		}
	}
}

double logaddexp(double log1, double log2) {
	if(log1 == log2)
		return log1 + M_LN2;
	else if(log1 > log2)
        return log1 + log1p(exp(log2-log1));
    else
        return log2 + log1p(exp(log1-log2));
}

#endif

