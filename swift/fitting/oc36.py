from pydreamx.core import run_dream as pd_run
from pydreamx.parameters import SampledParam as pd_param

import numpy as np
from scipy.stats import norm as normal, truncnorm as truncated_normal

from datetime import datetime

import sys, os, imp
swift = imp.load_dynamic("swiftstat7", "../SWIFT/MCMC/swiftstat7.so")

import math

import threading, time





def swift_loglik(parvals):
	global eno
	# check that all parameters are within prior bounds and set model parameters
	within_bounds = True
	loglik_lock.acquire()
	model = models.pop(0)
	for i in xrange(len(parvals)):
		pname = params.keys()[i]
		if np.isinf(params[pname].prior(parvals[i])):
			print "Trying to set %s to %lf which is out of bounds!" % (pname, parvals[i])
			within_bounds = False
		# set model parameter values but only if all parameters so far were within bounds
		if within_bounds:
			if pname == "omn_1":
				for px in ["omn_fs1", "omn_sk1", "omn_rg1", "omn_frf1", "omn_brf1"]:
					swift.updatemodel(param=px, value=parvals[i], model=model)
			elif pname == "omn_2":
				for px in ["omn_fs2", "omn_sk2", "omn_rg2", "omn_frf2", "omn_brf2"]:
					swift.updatemodel(param=px, value=parvals[i], model=model)
			elif pname == "sre_rf1":
				swift.updatemodel(param="sre_frf1", value=parvals[i], model=model)
				swift.updatemodel(param="sre_brf1", value=-parvals[i], model=model)
			elif pname == "sre_rf2":
				swift.updatemodel(param="sre_frf2", value=parvals[i], model=model)
				swift.updatemodel(param="sre_brf2", value=-parvals[i], model=model)
			elif pname == "sre_fssk1":
				swift.updatemodel(param="sre_fs1", value=parvals[i], model=model)
				swift.updatemodel(param="sre_sk1", value=parvals[i], model=model)
			elif pname == "tau_n2l":
				swift.updatemodel(param="tau_n", value=parvals[i], model=model)
				swift.updatemodel(param="tau_l", value=2.0*parvals[i], model=model)
			else:
				swift.updatemodel(param=pname, value=parvals[i], model=model)
	# if all parameters are within bounds, evaluate likelihood, otherwise return -inf
	if within_bounds:
		loglik = swift.eval(model = model, data = data, threads = ncores, validate = False)[0]
	else:
		loglik = -np.inf
	models.append(model)
	loglik_lock.release()
	return loglik


# short function for a truncated normal prior distribution bounded between lb and ub with a sd of (ub-lb)/2 and a mean at the centre of the range
def trpd(lb, ub):
	my_mean = lb+(ub-lb)/2.
	my_std = (ub-lb)/2.
	a, b = (lb - my_mean) / my_std, (ub - my_mean) / my_std
	return pd_param(truncated_normal, a = a, b = b, scale = my_std, loc = my_mean)


params = {
	'msac':     trpd(  1.2  ,  3.2 ),
	'delta0':   trpd(  2.5  , 20.0 ),
#	'asym':     trpd(  0.1  ,  1.5 ),
	'decay':    trpd(  0.01 ,  0.5 ),
	'gamma':    trpd(  0.1  ,  1.0 ),
	'beta':     trpd(  0.0  ,  1.0 ),
	'eta':      trpd(  0.0  ,  1.0 ),
#	'proc':     trpd(  0.0  ,  1.0 ),
#	'misprob':  trpd(  0.0  ,  1.0 ),
	'misfac':   trpd(  0.0  ,  2.0 ),
	'alpha':    trpd(  0.5  ,  5.0 ),
#	'tau_l':    trpd(  0.2  ,  2.5 ),
#	'tau_l2':   trpd(  1.0  ,  3.0 ),
#	'tau_n':    trpd(  0.05 ,  1.0 ),
	'tau_n2l':  trpd(  0.05 ,  1.5 ),
	'refix':    trpd(  0.0  ,  1.0 ),
	'omn_1':    trpd(  0.1  ,  3.0 ),
	'omn_2':    trpd(  0.0  ,  0.3 ),
	'sre_rg1':  trpd( -7.0  , -0.1 ),
	'sre_rg2':  trpd( -1.0  ,  0.0 ),
#	'sre_fs1':  trpd(  0.1  ,  7.0 ),
	'sre_fssk1':trpd(  0.1  ,  7.0 ),
	'sre_fs2':  trpd(  0.0  ,  1.0 ),
	'sre_rf1':  trpd(  0.1  ,  7.0 ),
	'sre_rf2':  trpd(  0.0  ,  1.0 ),
#	'sre_frf1': trpd(  0.1  ,  7.0 ),
#	'sre_frf2': trpd(  0.0  ,  1.0 ),
#	'sre_brf1': trpd( -7.0  , -0.1 ),
#	'sre_brf2': trpd( -1.0  ,  0.0 ),
#	'sre_sk1':  trpd(  0.1  ,  7.0 ),
	'sre_sk2':  trpd(  0.0  ,  1.0 )
#	'sprob': trpd(0.01, 0.99)
}

rand = np.random
niter = 20000
nchains = 5
ncores = 16
path = "../DATA18"
corpus = "oculo_01"
seqname = ["vp1_1", "vp2_1", "vp3_1", "vp4_1", "vp5_1", "vp6_1", "vp7_1", "vp8_1", "vp9_1", "vp11_1", "vp12_1", "vp13_1", "vp14_1", "vp15_1", "vp16_1", "vp17_1", "vp18_1", "vp19_1", "vp20_1", "vp21_1", "vp22_1", "vp23_1", "vp24_1", "vp25_1", "vp26_1", "vp27_1", "vp28_1", "vp29_1", "vp30_1", "vp31_1", "vp32_1", "vp33_1", "vp34_1", "vp35_1", "vp36_1", "vp37_1", "vp1_2", "vp2_2", "vp3_2", "vp4_2", "vp5_2", "vp6_2", "vp7_2", "vp8_2", "vp9_2", "vp11_2", "vp12_2", "vp13_2", "vp14_2", "vp15_2", "vp16_2", "vp17_2", "vp18_2", "vp19_2", "vp20_2", "vp21_2", "vp22_2", "vp23_2", "vp24_2", "vp25_2", "vp26_2", "vp27_2", "vp28_2", "vp29_2", "vp30_2", "vp31_2", "vp32_2", "vp33_2", "vp34_2", "vp35_2", "vp36_2", "vp37_2", "vp1_3", "vp2_3", "vp3_3", "vp4_3", "vp5_3", "vp6_3", "vp7_3", "vp8_3", "vp9_3", "vp11_3", "vp12_3", "vp13_3", "vp14_3", "vp15_3", "vp16_3", "vp17_3", "vp18_3", "vp19_3", "vp20_3", "vp21_3", "vp22_3", "vp23_3", "vp24_3", "vp25_3", "vp26_3", "vp27_3", "vp28_3", "vp29_3", "vp30_3", "vp31_3", "vp32_3", "vp33_3", "vp34_3", "vp35_3", "vp36_3", "vp37_3", "vp1_4", "vp2_4", "vp3_4", "vp4_4", "vp5_4", "vp6_4", "vp7_4", "vp8_4", "vp9_4", "vp11_4", "vp12_4", "vp13_4", "vp14_4", "vp15_4", "vp16_4", "vp17_4", "vp18_4", "vp19_4", "vp20_4", "vp21_4", "vp22_4", "vp23_4", "vp24_4", "vp25_4", "vp26_4", "vp27_4", "vp28_4", "vp29_4", "vp30_4", "vp31_4", "vp32_4", "vp33_4", "vp34_4", "vp35_4", "vp36_4", "vp37_4", "vp1", "vp2", "vp3", "vp4", "vp5", "vp6", "vp7", "vp8", "vp9", "vp11", "vp12", "vp13", "vp14", "vp15", "vp16", "vp17", "vp18", "vp19", "vp20", "vp21", "vp22", "vp23", "vp24", "vp25", "vp26", "vp27", "vp28", "vp29", "vp30", "vp31", "vp32", "vp33", "vp34", "vp35", "vp36", "vp37"][int(sys.argv[1])-1 if len(sys.argv) > 1 else 0]
custom_string = None

script_name = os.path.splitext(__file__)[0]
if custom_string is not None:
	outfile = "%s_%s_%s_%s" % (script_name, custom_string, corpus, seqname)
else:
	outfile = "%s_%s_%s" % (script_name, corpus, seqname)
models = [swift.loadmodel(outputPath=path, parmPath="%s/swpar_default.par" % (path), corpusFile="%s/corpus_%s.dat" % (path, corpus), seed=rand.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max), verbose=True) for i in xrange(nchains)]
loglik_lock = threading.Semaphore(nchains)
data = swift.loaddata("%s/fixseqin_%s_train.dat" % (path, seqname))

if swift.validate(data = data, model = models[0]):
	print "Valid data."
else:
	print "Invalid data."
	exit(1)

sampled_params, log_ps = pd_run(params.values(), swift_loglik, nchains = nchains, niterations = niter, restart = False, verbose = False, model_name = outfile, stochastic_loglike = True, multiprocessing = False)

for i in xrange(nchains):
	with open("%s_%d.dat" % (outfile, i+1), "w") as f:
		f.write("\t".join(params.keys()+["loglik"])+"\n")
		for j in xrange(niter):
			f.write("\t".join(str(x) for x in sampled_params[i][j,:]))
			f.write("\t"+str(log_ps[i][j,0])+"\n")
		f.flush()
		f.close()
