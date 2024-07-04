/*
 function EPALLIKE
 ==================
 (Version 1.0, 11-JUN-2017)
 (Adjusted 1.1, 29-SEP-2017)
 by Stefan Seelig
 */


#define sq(A) ((A)*(A))

// epanechnikov kernel
// double Jepanetschnikow(double z) {
// 	if (fabs(z)<=1.0) {
// 		return (3.0/4.0)*(1.0-sq(z));
// 	} else return 0.0;
	// instead: fabs(z) <= 1.0 ? (3.0/4.0)*(1.0-sq(z)) : 0.0;
// }

double epallike(double *d,int N,double val) {
	double sum,sqs,sd,bandwidth,P,dist,mindist;
	int i;
	//posmin = ((double) REALMIN)/((double) N);
	/* test */
	// low = high = d[1];
	// for ( i=1; i<=N; i++ ) { 
	// 	if (d[i] < low) low = d[i];
	// 	if (d[i] > high) high = d[i];
	// }
	//  bandwidth = (high-low) / 15.0; 

	sum = d[1];
	sqs = sq(d[1]);
	mindist = fabs(val - d[1]);
	for ( i=2; i<=N; i++ )  {
	        sum += d[i];
	        sqs += sq(d[i]);
	        dist = fabs(val - d[i]);
	        mindist = mindist < dist ? mindist : dist;
	}
	sd = sqrt((sqs-sq(sum)/(1.0*N))/(1.0*N - 1.0));

	/* Scott's rule */ 
	bandwidth = 3.49*sd/pow(N,1.0/3.0);

	/* quick & dirty adjustment to cope with regions of 0-probability  */
	bandwidth =  mindist > bandwidth ? (mindist*1.1) : bandwidth;

	// calculate probability for val
	P = 0.0;
	for ( i=1; i<=N; i++ ) {
		// only include durations that won't evaluate to 0
		if (d[i]>(val-bandwidth) && d[i]<(val+bandwidth)) { 
			//P += Jepanetschnikow((val-d[i])/bandwidth);
			P += (3.0/4.0)*(1.0-sq((val-d[i])/bandwidth));
		} // else P += posmin;
	}
	P /= (double) N * bandwidth;
	//P = log(P);
	// done

	return P;
}

double logepallike(double *d,int N,double val) {
	double sum,sqs,sd,bandwidth,dist,mindist,P_sum;
	int i;
	//posmin = ((double) REALMIN)/((double) N);
	/* test */
	// low = high = d[1];
	// for ( i=1; i<=N; i++ ) { 
	// 	if (d[i] < low) low = d[i];
	// 	if (d[i] > high) high = d[i];
	// }
	//  bandwidth = (high-low) / 15.0; 

	sum = d[1];
	sqs = sq(d[1]);
	mindist = fabs(val - d[1]);
	for ( i=2; i<=N; i++ )  {
	        sum += d[i];
	        sqs += sq(d[i]);
	        dist = fabs(val - d[i]);
	        mindist = mindist < dist ? mindist : dist;
	}
	sd = sqrt((sqs-sq(sum)/(1.0*N))/(1.0*N - 1.0));

	/* Scott's rule */ 
	bandwidth = 3.49*sd/pow(N,1.0/3.0);

	/* quick & dirty adjustment to cope with regions of 0-probability  */
	if(mindist > bandwidth) {
		bandwidth = mindist*1.1;	
	}

	// calculate probability for val

	double * P = dvector(1, N);
	int P_nonzero = 0;
	const double log_3o4 = log(3.0/4.0);

	for ( i=1; i<=N; i++ ) {
		// only include durations that won't evaluate to 0
		if (d[i]>(val-bandwidth) && d[i]<(val+bandwidth)) { 
			//P += Jepanetschnikow((val-d[i])/bandwidth);
			P[++P_nonzero] = log_3o4 + log1p(-sq((val-d[i])/bandwidth));
		}
	}

	P_sum = logsumexp(P, P_nonzero);

	free_dvector(P, 1, N);

	return P_sum - log(N) - log(bandwidth);
}

#undef sq